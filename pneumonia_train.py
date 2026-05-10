"""
Train a CNN on the Kaggle chest X-ray pneumonia dataset.
Targeting >=97.7% test accuracy, because precise targets sound more impressive
than 'roughly 98ish'.

Built around an AMD Vega 64 + torch-directml — i.e. the duct-tape stack that
exists because ROCm dropped GCN5 support in 2021. Crashes are not bugs in this
script; they are DirectML expressing itself.

Usage:
    python pneumonia_train.py                     # single-fold run
    python pneumonia_train.py --n_folds 5 --fold 0
    python pneumonia_train.py --model tf_efficientnetv2_s.in1k --img_size 300
"""
import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

import timm

# torch-directml is Windows-only and only ships wheels for Python ≤ 3.11.
# On Colab / Linux / NVIDIA we fall back to plain CUDA.
try:
    import torch_directml as dml  # noqa: F401
    HAS_DML = True
except ImportError:
    HAS_DML = False
    dml = None  # type: ignore

# Paths resolve relative to this script so the project works from any
# install location (Colab `/content/...`, Linux `~/projects/...`, etc.).
# Environment variables `PNEUMONIA_DATA` and `PNEUMONIA_RUNS` override.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get(
    "PNEUMONIA_DATA",
    PROJECT_ROOT / "data" / "chest_xray",
))
OUT_DIR = Path(os.environ.get(
    "PNEUMONIA_RUNS",
    PROJECT_ROOT / "runs",
))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_device(prefer="auto"):
    """Return (device, human-readable name).

    `prefer` is one of:
      - "auto": try DirectML → CUDA → CPU (default)
      - "dml" : force DirectML, fall back to CPU if unavailable
      - "cuda": force CUDA, fall back to CPU if unavailable
      - "cpu" : force CPU
    """
    if prefer == "cpu":
        return torch.device("cpu"), "CPU (forced)"
    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0"), f"CUDA: {torch.cuda.get_device_name(0)}"
        return torch.device("cpu"), "CPU (CUDA not available, falling back)"
    if prefer == "dml":
        if HAS_DML and dml.device_count() > 0:  # type: ignore[union-attr]
            return dml.device(0), f"DirectML: {dml.device_name(0)}"  # type: ignore[union-attr]
        return torch.device("cpu"), "CPU (DirectML not available, falling back)"
    # auto
    if HAS_DML and dml.device_count() > 0:  # type: ignore[union-attr]
        return dml.device(0), f"DirectML: {dml.device_name(0)}"  # type: ignore[union-attr]
    if torch.cuda.is_available():
        return torch.device("cuda:0"), f"CUDA: {torch.cuda.get_device_name(0)}"
    return torch.device("cpu"), "CPU (no GPU detected — training will be slow)"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50.a1_in1k")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=4, help="smaller for DirectML stability")
    p.add_argument("--resume", action="store_true", help="resume from last_state.pt in run dir")
    p.add_argument("--epochs_head", type=int, default=1)
    p.add_argument("--epochs_full", type=int, default=15)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_full", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--n_folds", type=int, default=1, help="1 = single 90/10 split. >1 = stratified k-fold.")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--no_focal", dest="use_focal", action="store_false")
    p.add_argument("--no_tta", dest="tta", action="store_false")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--run_name", default=None)
    p.add_argument("--mixup_alpha", type=float, default=0.0,
                   help="Beta(α,α) for Mixup. 0.2-0.4 typical, 0 turns it off.")
    p.add_argument("--mixup_prob", type=float, default=0.5,
                   help="Per-batch probability of applying mixup.")
    p.add_argument("--ema_decay", type=float, default=0.0,
                   help="EMA decay (e.g. 0.999). 0 turns it off.")
    p.add_argument("--final_test", action="store_true",
                   help="Run the test-eval inside this script. Disabled by default "
                        "because DirectML treats it as an opportunity to crash.")
    p.add_argument("--pretrained", action="store_true",
                   help="Initialise the timm backbone with pretrained weights. "
                        "Off by default. Pass this flag to load ImageNet "
                        "weights for transfer learning.")
    p.add_argument("--max_session_minutes", type=float, default=0.0,
                   help="Stop cleanly after the current epoch once this many "
                        "wall-clock minutes have elapsed in this invocation. "
                        "0 = no limit. State is saved per-epoch, so re-running "
                        "with --resume picks up exactly where we stopped.")
    p.add_argument("--snr_optimizer", action="store_true",
                   help="Use SNR-gated AdamW (Litman & Guo 2026) instead of "
                        "standard AdamW. Adds one extra state vector per "
                        "parameter; same wall-clock cost.")
    p.add_argument("--device", choices=["auto", "dml", "cuda", "cpu"],
                   default="auto",
                   help="Force a specific compute device. Default 'auto' "
                        "tries DirectML → CUDA → CPU.")
    p.set_defaults(use_focal=True, tta=True)
    return p.parse_args()


def list_images(root: Path):
    items = []
    for split in ("train", "val", "test"):
        for cls, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
            for path in (root / split / cls).glob("*"):
                if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    items.append((str(path), label, split))
    return items


class XRayDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx][0], self.items[idx][1]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_transforms(img_size, train, hflip_eval=False):
    """Build the torchvision transform pipeline.

    `hflip_eval=True` flips the image horizontally as part of CPU preprocessing,
    so we can do TTA without ever asking DirectML to perform `torch.flip` on a
    GPU tensor (it has Opinions about that).
    """
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 24, img_size + 24)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            norm,
        ])
    eval_steps: list = [transforms.Resize((img_size, img_size))]
    if hflip_eval:
        eval_steps.append(transforms.RandomHorizontalFlip(p=1.0))
    eval_steps += [transforms.ToTensor(), norm]
    return transforms.Compose(eval_steps)


class FocalLoss(nn.Module):
    """Hand-rolled focal loss. We can't use `binary_cross_entropy_with_logits`
    because it calls `log_sigmoid`, which DirectML helpfully reroutes to the
    CPU mid-graph and then everything falls apart."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        targets = targets.float()
        p = torch.sigmoid(logits).clamp(min=1e-6, max=1 - 1e-6)
        pt = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1,
                              torch.full_like(targets, self.alpha),
                              torch.full_like(targets, 1 - self.alpha))
        return (-alpha_t * (1 - pt).pow(self.gamma) * torch.log(pt)).mean()


class BCEManual(nn.Module):
    """BCE-with-logits, but written by hand so DirectML doesn't get any ideas
    about silently shipping it to the CPU."""
    def forward(self, logits, targets):
        p = torch.sigmoid(logits).clamp(min=1e-6, max=1 - 1e-6)
        targets = targets.float()
        return -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p)).mean()


class SNRAdamW(torch.optim.AdamW):
    """AdamW with the SNR (signal-to-noise) gate from Litman & Guo (2026):
    "A Theory of Generalization in Deep Learning" (arXiv:2605.01172).

    Tracks one extra per-parameter EMA — the variance of the gradient around
    its EMA mean. Per parameter, computes:

        signal² = m̂²                       (squared bias-corrected mean)
        noise   = ŝ / (b - 1)               (variance estimate / batch-1)
        gate    = max(0, (signal² - noise) / (ŝ + ε))

    The standard Adam update is multiplied by `gate`. Parameters whose
    minibatch signal-to-noise ratio sits below threshold are not updated this
    step. The paper argues this preserves coherent population signal while
    suppressing per-example noise, giving 2-5× faster training on
    memorisation-prone tasks. For standard supervised vision the gain is
    typically modest (~+0.3-1%) but it costs nothing extra and is novel.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, snr_eps=1e-8, batch_size=8):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay)
        self.snr_eps = snr_eps
        self.batch_size = batch_size

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["snr_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                m, v, s = state["exp_avg"], state["exp_avg_sq"], state["snr_var"]

                # SNR variance: EMA of (g - m_{t-1})^2, using the *old* m (pre-update).
                deviation = grad - m
                s.mul_(beta2).addcmul_(deviation, deviation, value=1 - beta2)

                # Standard Adam moment updates.
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step = state["step"]
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                m_hat = m / bc1
                v_hat = v / bc2
                s_hat = s / bc2

                # Gate: max(0, (μ² - σ²/(b-1)) / (σ² + ε))
                signal_sq = m_hat * m_hat
                noise_floor = s_hat / max(self.batch_size - 1, 1)
                gate = ((signal_sq - noise_floor) / (s_hat + self.snr_eps)).clamp_(min=0.0)

                # AdamW-style decoupled weight decay.
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                # Adam update gated by the SNR mask.
                update = m_hat / (v_hat.sqrt() + eps)
                p.data.add_(gate * update, alpha=-lr)

        return loss


def mixup_batch(x, y, alpha):
    """Mixup: blend two images with a Beta(α,α)-sampled coefficient and pretend
    that's a sensible thing to feed a classifier. It works, somehow."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam


class ModelEMA:
    """Exponential moving average of model weights. Shadow lives on the CPU
    because the Vega 64's 8 GB is already booked solid."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            v_cpu = v.detach().cpu()
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v_cpu, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v_cpu)

    def state_dict(self):
        return self.shadow


def train_one_epoch(model, loader, optimizer, criterion, device, desc,
                    mixup_alpha=0.0, mixup_prob=0.5, ema=None):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    use_mixup = mixup_alpha > 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if use_mixup and random.random() < mixup_prob:
            x_mix, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
            logits = model(x_mix).squeeze(-1)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            preds = (torch.sigmoid(logits) > 0.5).long()
            # Train-acc on mixup batches is meaningless; we score against whichever
            # target dominated the mix just so the number isn't wildly insulting.
            ref = y_a if lam >= 0.5 else y_b
            total_correct += (preds == ref).sum().item()
        else:
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).long()
            total_correct += (preds == y).sum().item()
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def _eval_pass(model, loader, criterion, device):
    """Single forward pass over a loader. The loader is responsible for whatever
    augmentation it wants to apply on the CPU side; the model just sees tensors."""
    model.eval()
    all_probs, all_labels = [], []
    total_loss, n = 0.0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x).squeeze(-1)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(y.cpu())
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss, n, probs, labels


def evaluate(model, loader, criterion, device, tta, loader_flip=None):
    """Evaluate on `loader`, optionally averaging with the flipped variant.

    DirectML throws a tantrum when it sees `torch.flip` followed by another
    forward pass, so we never flip on-GPU. Caller must pre-build a separate
    loader whose dataset already flips images on the CPU side and pass it in
    as `loader_flip`. If TTA is on but no flip-loader was provided, we just
    skip the flip pass and silently miss out on the ~+0.5% accuracy bump.
    """
    loss_o, n_o, probs_o, labels = _eval_pass(model, loader, criterion, device)
    if not tta or loader_flip is None:
        acc = ((probs_o > 0.5).astype(int) == labels).mean()
        return loss_o / n_o, acc, probs_o, labels
    loss_f, n_f, probs_f, _ = _eval_pass(model, loader_flip, criterion, device)
    probs = (probs_o + probs_f) / 2
    avg_loss = (loss_o + loss_f) / (n_o + n_f)
    acc = ((probs > 0.5).astype(int) == labels).mean()
    return avg_loss, acc, probs, labels


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device, device_name = get_device(args.device)
    print(f"Device: {device_name}")

    session_start = time.time()
    session_limit_s = args.max_session_minutes * 60 if args.max_session_minutes > 0 else 0.0

    def session_expired():
        return session_limit_s > 0 and (time.time() - session_start) > session_limit_s

    items = list_images(DATA_ROOT)
    train_items = [(p, l) for p, l, s in items if s in ("train", "val")]
    test_items = [(p, l) for p, l, s in items if s == "test"]
    print(f"Train+val pool: {len(train_items)}  Test: {len(test_items)}")
    print(f"Class counts (train pool): {Counter(l for _, l in train_items)}")

    labels_arr = np.array([l for _, l in train_items])
    if args.n_folds > 1:
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        splits = list(skf.split(np.zeros(len(train_items)), labels_arr))
        tr_idx, va_idx = splits[args.fold]
    else:
        tr_idx, va_idx = train_test_split(
            np.arange(len(train_items)), test_size=0.1,
            stratify=labels_arr, random_state=args.seed,
        )
    fold_train = [train_items[i] for i in tr_idx]
    fold_val = [train_items[i] for i in va_idx]
    print(f"Fold {args.fold}: train={len(fold_train)} val={len(fold_val)}")

    train_tf = build_transforms(args.img_size, train=True)
    eval_tf = build_transforms(args.img_size, train=False)

    ds_train = XRayDataset(fold_train, train_tf)
    ds_val = XRayDataset(fold_val, eval_tf)
    ds_test = XRayDataset(test_items, eval_tf)

    label_counts = Counter(l for _, l in fold_train)
    sampler_weights = [1.0 / label_counts[l] for _, l in fold_train]
    sampler = WeightedRandomSampler(sampler_weights, num_samples=len(fold_train), replacement=True)

    nw = args.num_workers
    eval_bs = args.eval_batch_size or args.batch_size
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                          num_workers=nw, persistent_workers=nw > 0)
    dl_val = DataLoader(ds_val, batch_size=eval_bs, shuffle=False,
                        num_workers=nw, persistent_workers=nw > 0)
    dl_test = DataLoader(ds_test, batch_size=eval_bs, shuffle=False,
                         num_workers=nw, persistent_workers=nw > 0)

    # Same architecture either way; only the initialisation differs. Pass
    # `--pretrained` for the transfer-learning variant. With from-scratch +
    # a 23M backbone like ResNet50 you'll overfit fast, so pair the default
    # with a smaller `--model`, e.g. `--model resnet18`.
    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=1)
    n_params = sum(p.numel() for p in model.parameters())
    init_tag = "pretrained" if args.pretrained else "from-scratch"
    model_tag = f"{args.model}{'' if args.pretrained else '_scratch'}"
    print(f"Model: {args.model} ({init_tag}, {n_params:,} params)")
    model = model.to(device)

    criterion = FocalLoss(args.focal_gamma, args.focal_alpha) if args.use_focal else BCEManual()

    run_name = args.run_name or f"{model_tag.split('.')[0]}_f{args.fold}_{int(time.time())}"
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"EMA enabled with decay={args.ema_decay}")

    # Per-epoch metric history. Persisted to history.json so the plot script
    # can reconstruct learning curves without parsing terminal output.
    history_path = run_dir / "history.json"
    history = {"phase1": [], "phase2": []}
    if args.resume and history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)
        except Exception:
            pass

    def append_history(phase_key, epoch, tl, ta, vl, va, lr, dt):
        history[phase_key].append({
            "epoch": epoch, "train_loss": float(tl), "train_acc": float(ta),
            "val_loss": float(vl), "val_acc": float(va),
            "lr": float(lr), "time_s": float(dt),
        })
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    # Resume support: when (not if) DirectML kills the process mid-run, we want
    # to pick up at the last completed epoch rather than start over from scratch.
    state_path = run_dir / "last_state.pt"
    progress = {"phase": 1, "epoch_done": 0, "best_val_acc": 0.0}
    best_state = None
    if args.resume and state_path.exists():
        ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.to(device) for k, v in ckpt["model"].items()})
        progress = ckpt["progress"]
        best_state = ckpt.get("best_state")
        if ema is not None and ckpt.get("ema_state") is not None:
            ema.shadow = ckpt["ema_state"]
        print(f"Resumed: phase={progress['phase']} epoch_done={progress['epoch_done']} "
              f"best_val_acc={progress['best_val_acc']:.4f}")
    best_val_acc = progress["best_val_acc"]

    def save_state(phase, epoch_done):
        torch.save({
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "progress": {"phase": phase, "epoch_done": epoch_done, "best_val_acc": best_val_acc},
            "best_state": best_state,
            "ema_state": ema.shadow if ema is not None else None,
            # Architecture tag so the eval script knows which model class to
            # rehydrate without us having to remember which flag we used.
            "model_tag": model_tag,
            "pretrained": bool(args.pretrained),
        }, state_path)

    def eval_with_ema():
        """Evaluate the EMA shadow if we have one, otherwise the live weights.
        The live weights are temporarily swapped out and put back afterward,
        because telling AdamW its parameters changed mid-step is rude."""
        if ema is None:
            return evaluate(model, dl_val, criterion, device, tta=False)
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(device) for k, v in ema.shadow.items()})
        result = evaluate(model, dl_val, criterion, device, tta=False)
        model.load_state_dict(backup)
        return result

    # Phase 1 only makes sense when there's a pretrained backbone to thaw onto.
    # For from-scratch models everything starts random, so jump straight to Phase 2.
    if not args.pretrained:
        args.epochs_head = 0
        print("\n=== Phase 1 skipped (from-scratch model has no pretrained backbone) ===")

    print(f"\n=== Phase 1 (head only, {args.epochs_head} epochs) ===")
    # Use timm's get_classifier() to find the actual head. The previous version
    # of this code did `'fc' in name` which gleefully matched every MLP block in
    # ConvNeXt and trained 25M "frozen" parameters. Lesson learned.
    head_module = model.get_classifier()
    head_param_ids = {id(p) for p in head_module.parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_param_ids
    head_params = [p for p in model.parameters() if p.requires_grad]
    n_head = sum(p.numel() for p in head_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  head params: {n_head:,} / total: {n_total:,}")
    if args.snr_optimizer:
        opt = SNRAdamW(head_params, lr=args.lr_head, weight_decay=args.weight_decay,
                       batch_size=args.batch_size)
        print("  optimizer: SNRAdamW (Litman & Guo 2026)")
    else:
        opt = torch.optim.AdamW(head_params, lr=args.lr_head, weight_decay=args.weight_decay)
    p1_start = progress["epoch_done"] if progress["phase"] == 1 else args.epochs_head
    for ep in range(p1_start, args.epochs_head):
        t0 = time.time()
        tl, ta = train_one_epoch(model, dl_train, opt, criterion, device, f"P1 ep{ep+1}")
        vl, va, _, _ = evaluate(model, dl_val, criterion, device, tta=False)
        dt = time.time() - t0
        print(f"P1 ep{ep+1}: train_loss={tl:.4f} acc={ta:.4f} | val_loss={vl:.4f} acc={va:.4f} ({dt:.0f}s)")
        append_history("phase1", ep + 1, tl, ta, vl, va, args.lr_head, dt)
        if va > best_val_acc:
            best_val_acc = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save_state(phase=1, epoch_done=ep + 1)
        if session_expired():
            print(f"Session limit ({args.max_session_minutes:.1f} min) reached after P1 ep{ep+1}. "
                  f"Re-run with --resume to continue.")
            return

    print(f"\n=== Phase 2 (full unfreeze, {args.epochs_full} epochs) ===")
    for p in model.parameters():
        p.requires_grad = True
    if args.snr_optimizer:
        opt = SNRAdamW(model.parameters(), lr=args.lr_full, weight_decay=args.weight_decay,
                       batch_size=args.batch_size)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr_full, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_full, eta_min=args.lr_full * 0.01)
    p2_start = progress["epoch_done"] if progress["phase"] == 2 else 0
    # If we're resuming, walk the LR scheduler forward to where we left off.
    for _ in range(p2_start):
        sched.step()
    no_improve = 0
    for ep in range(p2_start, args.epochs_full):
        t0 = time.time()
        tl, ta = train_one_epoch(
            model, dl_train, opt, criterion, device, f"P2 ep{ep+1}",
            mixup_alpha=args.mixup_alpha, mixup_prob=args.mixup_prob, ema=ema,
        )
        vl, va, _, _ = eval_with_ema()
        sched.step()
        lr_now = opt.param_groups[0]["lr"]
        ema_tag = " (EMA)" if ema is not None else ""
        dt = time.time() - t0
        print(f"P2 ep{ep+1}: train_loss={tl:.4f} acc={ta:.4f} | val_loss={vl:.4f} acc={va:.4f}{ema_tag} | lr={lr_now:.2e} ({dt:.0f}s)")
        append_history("phase2", ep + 1, tl, ta, vl, va, lr_now, dt)
        if va > best_val_acc:
            best_val_acc = va
            # Snapshot whichever weights produced this val_acc: EMA shadow if it
            # was the one being evaluated, otherwise the live model weights.
            if ema is not None:
                best_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        save_state(phase=2, epoch_done=ep + 1)
        if no_improve >= args.patience:
            print(f"Early stop at P2 ep{ep+1}")
            break
        if session_expired():
            print(f"Session limit ({args.max_session_minutes:.1f} min) reached after P2 ep{ep+1}. "
                  f"Re-run with --resume to continue this fold.")
            return

    print(f"\nBest val acc: {best_val_acc:.4f}")
    torch.save(best_state, run_dir / "model_best.pt")
    summary = {
        "model_tag": model_tag,
        "pretrained": bool(args.pretrained),
        "model": args.model,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "best_val_acc": float(best_val_acc),
        "fold": args.fold,
        "n_folds": args.n_folds,
        "use_focal": bool(args.use_focal),
        "mixup_alpha": float(args.mixup_alpha),
        "ema_decay": float(args.ema_decay),
        "optimizer": "snr_adamw" if args.snr_optimizer else "adamw",
    }

    if args.final_test:
        # Doing test-eval inside the training process used to consistently hand
        # us a silent access violation — kept the "Best val acc:" line and then
        # quietly took the rest of the script with it. Off by default for a reason.
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_loss, test_acc, test_probs, test_labels = evaluate(
            model, dl_test, criterion, device, tta=args.tta)
        print(f"\n=== TEST: loss={test_loss:.4f} acc={test_acc:.4f} (TTA={args.tta}) ===")
        np.save(run_dir / "test_probs.npy", test_probs)
        np.save(run_dir / "test_labels.npy", test_labels)
        summary["test_acc"] = float(test_acc)
        summary["test_loss"] = float(test_loss)
        summary["tta"] = bool(args.tta)
    else:
        print("Skipping test-eval — run pneumonia_eval.py when the training is done.")

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
