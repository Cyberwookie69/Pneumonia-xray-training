"""Parametric from-scratch CNN trainer.

Every architectural decision is a CLI flag, so a single codebase generates
every row of every ablation table.

Usage examples:
    python pneumonia_cnn_custom.py --n_blocks 4 --run_name d4_relu_pool
    python pneumonia_cnn_custom.py --activation gelu --run_name d4_gelu_pool
    python pneumonia_cnn_custom.py --use_bn --use_dropout 0.3 --augment \
        --run_name d4_bn_drop_aug
    python pneumonia_cnn_custom.py --n_folds 5 --fold 0 --run_name champion_f0
"""
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# Reuse the data layout helpers from the main project. Keeps the
# train/val/test split and DATA_ROOT logic in one place.
from pneumonia_train import DATA_ROOT, OUT_DIR, get_device, list_images


# ---------------------------------------------------------------------------
# Parametric CNN
# ---------------------------------------------------------------------------


def make_activation(name: str) -> nn.Module:
    # inplace=False is required for the Grad-CAM cell's backward hook to work.
    # Inplace ReLU modifies a view of the conv output, which clashes with
    # `register_full_backward_hook` on that conv. Mathematically identical to
    # the inplace variant, so existing checkpoints remain valid.
    return {
        "relu": nn.ReLU(inplace=False),
        "leaky": nn.LeakyReLU(0.1, inplace=False),
        "gelu": nn.GELU(),
    }[name]


class ConvPoolBlock(nn.Module):
    """One convolution-pooling building block.

    Conv 3x3 (with chosen padding + stride) -> [BatchNorm] -> activation -> [Pool].

    `stride_mode` controls how spatial reduction happens:
      - "pool": stride-1 conv followed by 2x2 max-pool (textbook VGG-style)
      - "strided": stride-2 conv replaces the pool entirely (modern style)
    """

    def __init__(self, in_ch: int, out_ch: int, activation: str,
                 padding: str, stride_mode: str, use_bn: bool):
        super().__init__()
        # PyTorch Conv2d accepts "same" / "valid" strings since 1.9; map ourselves.
        pad = 1 if padding == "same" else 0
        stride = 2 if stride_mode == "strided" else 1
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = make_activation(activation)
        self.pool = nn.MaxPool2d(2) if stride_mode == "pool" else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class CustomCNN(nn.Module):
    """N-block CNN: head produces a single logit for binary pneumonia classification."""

    def __init__(self, n_blocks: int = 4, base_channels: int = 32,
                 activation: str = "relu", padding: str = "same",
                 stride_mode: str = "pool", use_bn: bool = False,
                 use_dropout: float = 0.0, init: str = "kaiming"):
        super().__init__()
        blocks = []
        in_ch = 1  # grayscale X-ray
        for i in range(n_blocks):
            out_ch = base_channels * (2 ** i)
            blocks.append(ConvPoolBlock(in_ch, out_ch, activation, padding,
                                        stride_mode, use_bn))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(use_dropout) if use_dropout > 0 else nn.Identity()
        self.head = nn.Linear(in_ch, 1)

        # Initialise convs. PyTorch defaults to kaiming_uniform_ already, but
        # we make the choice explicit and offer Glorot as a controlled
        # comparison (He et al. 2015 argue Glorot fails with stacked ReLUs,
        # producing vanishing pre-activations at depth ≥ 4).
        self._apply_init(init, activation)

    def _apply_init(self, init: str, activation: str):
        nonlinearity = "relu" if activation in ("relu", "leaky") else "linear"
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                             nonlinearity=nonlinearity)
                elif init == "glorot":
                    nn.init.xavier_uniform_(m.weight)
                else:
                    raise ValueError(f"Unknown init: {init}")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Output layer always gets Glorot — the lecture-notes rule
                # that regularisation/init for ReLU doesn't apply to the
                # final sigmoid head.
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class XRayDataset(Dataset):
    """Grayscale 1-channel X-ray dataset. We deliberately stay 1-channel so the
    custom CNN's first-layer input matches the data instead of being padded to
    3 channels for ImageNet compatibility — that would be a transfer-learning
    accommodation we do not need here."""

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx][0], self.items[idx][1]
        img = Image.open(path).convert("L")  # 1-channel
        return self.transform(img), label


def build_transforms(img_size: int, train: bool, augment: bool,
                     policy: str = "standard"):
    """Resize → [augment] → ToTensor → Normalise.

    `policy`:
      - "standard": flip + affine + jitter (the existing pipeline)
      - "trivial":  TrivialAugmentWide (Müller & Hutter 2021, one-knob aug)
    """
    base = [transforms.Resize((img_size, img_size))]
    if train and augment:
        if policy == "trivial":
            # TrivialAugmentWide assumes 3-channel input; we round-trip via
            # a fake-RGB grayscale duplication so colour ops act as no-ops.
            base += [
                transforms.Grayscale(num_output_channels=3),
                transforms.TrivialAugmentWide(),
                transforms.Grayscale(num_output_channels=1),
            ]
        else:
            base += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=8, translate=(0.04, 0.04),
                                        scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
    base += [
        transforms.ToTensor(),  # already 1-channel
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ]
    return transforms.Compose(base)


class Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al. 2023, NeurIPS).

    Sign-based update with EMA-tracked momentum. Same wall-clock per step
    as AdamW, often matches or slightly beats it on CV tasks. Tends to
    benefit from a 3–10× smaller learning rate.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(b1).add(g, alpha=1 - b1).sign_()
                p.data.add_(update, alpha=-lr)
                exp_avg.mul_(b2).add_(g, alpha=1 - b2)
        return loss


def cutmix_batch(x, y, alpha=1.0):
    """Apply CutMix (Yun et al. 2019, ICCV) within a batch.

    Returns the mixed input, original labels, shuffled labels, and the
    effective mixing weight λ (= area kept of the original images).
    """
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(W, cx + cut_w // 2)
    bby2 = min(H, cy + cut_h // 2)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    # Recompute lambda to match the actual cut area.
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x, y, y[index], lam


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, opt, device, criterion,
                    label_smoothing=0.0, cutmix_alpha=0.0):
    """One pass over the training loader.

    `label_smoothing` ∈ [0, 1) softens the binary targets symmetrically:
    y=1 → 1 − α/2, y=0 → α/2 (Müller et al. 2019, NeurIPS).

    `cutmix_alpha` > 0 enables CutMix with Beta(α, α) sampling per batch,
    applied with 50% probability. Loss becomes a λ-weighted sum of the
    losses on the original and shuffled labels (Yun et al. 2019, ICCV).

    Both default to 0 = original behaviour preserved.
    """
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()
        opt.zero_grad()
        use_cutmix = cutmix_alpha > 0 and np.random.random() < 0.5
        if use_cutmix:
            x, y_a, y_b, lam = cutmix_batch(x, y, alpha=cutmix_alpha)
            logits = model(x)
            if label_smoothing > 0:
                y_a_t = y_a * (1.0 - label_smoothing) + 0.5 * label_smoothing
                y_b_t = y_b * (1.0 - label_smoothing) + 0.5 * label_smoothing
            else:
                y_a_t, y_b_t = y_a, y_b
            loss = lam * criterion(logits, y_a_t) + (1 - lam) * criterion(logits, y_b_t)
            # Accuracy against the dominant target (the one with higher λ).
            y_for_acc = y_a if lam >= 0.5 else y_b
            total_correct += ((logits > 0).long() == y_for_acc.long()).sum().item()
        else:
            logits = model(x)
            if label_smoothing > 0:
                y_target = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
            else:
                y_target = y
            loss = criterion(logits, y_target)
            total_correct += ((logits > 0).long() == y.long()).sum().item()
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n, total_correct / n


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        y_t = y.to(device).float()
        logits = model(x)
        loss = criterion(logits, y_t)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs > 0.5).astype(int)
    acc = float((preds == labels).mean())
    return total_loss / n, acc, probs, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_blocks", type=int, default=4)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--activation", choices=["relu", "leaky", "gelu"], default="relu")
    p.add_argument("--padding", choices=["same", "valid"], default="same")
    p.add_argument("--stride_mode", choices=["pool", "strided"], default="pool")
    p.add_argument("--init", choices=["kaiming", "glorot"], default="kaiming",
                   help="Conv weight init. 'kaiming' (He, default) is the "
                        "standard for stacked ReLUs; 'glorot' (Xavier) is "
                        "kept as a controlled comparison for the depth ablation.")
    p.add_argument("--use_bn", action="store_true",
                   help="add BatchNorm after each conv")
    p.add_argument("--use_dropout", type=float, default=0.0,
                   help="dropout p before final linear (0 = off)")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="L2 penalty (0 = off)")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Symmetric binary label smoothing α ∈ [0, 1). Targets "
                        "are softened: y=1→1−α/2, y=0→α/2. Default 0 = no "
                        "smoothing (Müller et al. 2019, NeurIPS).")
    p.add_argument("--augment", action="store_true",
                   help="enable training-time augmentation (flip+affine+jitter)")
    p.add_argument("--augment_policy", choices=["standard", "trivial"],
                   default="standard",
                   help="Augmentation pipeline. 'standard' = flip+affine+jitter. "
                        "'trivial' = TrivialAugmentWide (Müller & Hutter 2021).")
    p.add_argument("--cutmix_alpha", type=float, default=0.0,
                   help="CutMix β-distribution α. 0 = off. Yun et al. 2019.")
    p.add_argument("--optimizer", choices=["adamw", "lion"], default="adamw",
                   help="Choice of optimizer. 'lion' = Chen et al. 2023 (NeurIPS).")
    p.add_argument("--use_swa", action="store_true",
                   help="Enable Stochastic Weight Averaging over the last "
                        "(1 - swa_start_frac) fraction of epochs. Replaces "
                        "best_state with the SWA average and reruns BN stats. "
                        "Izmailov et al. 2018.")
    p.add_argument("--swa_start_frac", type=float, default=0.75,
                   help="Fraction of total epochs after which SWA averaging "
                        "kicks in (default 0.75 = last 25%% of training).")
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="stop if val loss doesn't improve for N epochs (0 = off)")
    # Training
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=78)
    # Splitting
    p.add_argument("--n_folds", type=int, default=1,
                   help="1 = single 88/12 split; >1 = stratified k-fold")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.12,
                   help="only used when n_folds=1")
    # Output
    p.add_argument("--run_name", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device, device_name = get_device()
    print(f"Device: {device_name}")
    print(f"Run: {args.run_name}")
    print(f"Architecture: n_blocks={args.n_blocks}, act={args.activation}, "
          f"pad={args.padding}, stride_mode={args.stride_mode}, "
          f"bn={args.use_bn}, dropout={args.use_dropout}, "
          f"wd={args.weight_decay}, aug={args.augment}")

    # Build dataset items, train/val split
    items = list_images(DATA_ROOT)
    train_pool = [(p, l) for p, l, s in items if s in ("train", "val")]
    test_items = [(p, l) for p, l, s in items if s == "test"]
    labels = np.array([l for _, l in train_pool])
    paths = np.array([p for p, _ in train_pool])

    if args.n_folds == 1:
        tr_idx, va_idx = train_test_split(
            np.arange(len(train_pool)), test_size=args.val_frac,
            stratify=labels, random_state=args.seed,
        )
    else:
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                              random_state=args.seed)
        splits = list(skf.split(np.zeros(len(train_pool)), labels))
        tr_idx, va_idx = splits[args.fold]

    fold_train = [(paths[i], int(labels[i])) for i in tr_idx]
    fold_val = [(paths[i], int(labels[i])) for i in va_idx]
    print(f"Train: {len(fold_train)}  Val: {len(fold_val)}  Test: {len(test_items)}")

    # Class-weighted sampler (mild overfitting/imbalance mitigation; always on)
    label_counts = Counter(l for _, l in fold_train)
    sampler_weights = [1.0 / label_counts[l] for _, l in fold_train]
    sampler = WeightedRandomSampler(sampler_weights, num_samples=len(fold_train),
                                    replacement=True)

    train_tf = build_transforms(args.img_size, train=True, augment=args.augment,
                                policy=args.augment_policy)
    eval_tf = build_transforms(args.img_size, train=False, augment=False)

    dl_train = DataLoader(XRayDataset(fold_train, train_tf),
                          batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.num_workers,
                          persistent_workers=args.num_workers > 0)
    dl_val = DataLoader(XRayDataset(fold_val, eval_tf),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        persistent_workers=args.num_workers > 0)
    dl_test = DataLoader(XRayDataset(test_items, eval_tf),
                         batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers,
                         persistent_workers=args.num_workers > 0)

    model = CustomCNN(
        n_blocks=args.n_blocks, base_channels=args.base_channels,
        activation=args.activation, padding=args.padding,
        stride_mode=args.stride_mode, use_bn=args.use_bn,
        use_dropout=args.use_dropout, init=args.init,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    if args.optimizer == "lion":
        # Lion typically wants a smaller LR than AdamW; user controls via --lr.
        opt = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # SWA setup — runs alongside normal training and replaces best_state at end.
    swa_model = None
    swa_start_epoch = 0
    if args.use_swa:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
        swa_start_epoch = max(0, int(args.epochs * args.swa_start_frac))
        print(f"SWA: averaging from epoch {swa_start_epoch + 1} onwards.")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    t0 = time.time()
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, opt, device, criterion,
                                          label_smoothing=args.label_smoothing,
                                          cutmix_alpha=args.cutmix_alpha)
        va_loss, va_acc, _, _ = evaluate(model, dl_val, device, criterion)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"  ep{epoch + 1:>2}  train {tr_loss:.4f}/{tr_acc:.4f}  "
              f"val {va_loss:.4f}/{va_acc:.4f}")

        # SWA: accumulate weight average from swa_start_epoch onwards.
        if swa_model is not None and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)

        # Track best by val_loss (avoids picking a noisy spike in val_acc).
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            # When SWA is on we want the full schedule; skip early stopping.
            if (not args.use_swa and args.early_stop_patience
                    and no_improve >= args.early_stop_patience):
                print(f"  early-stop at epoch {epoch + 1} (no val improvement "
                      f"in {args.early_stop_patience} epochs)")
                break

    # SWA: replace best_state with the averaged weights and re-fit BN stats.
    if swa_model is not None:
        from torch.optim.swa_utils import update_bn
        print("SWA: updating BatchNorm stats on the averaged model...")
        update_bn(dl_train, swa_model, device=device)
        # Extract the underlying module's state dict (strip 'module.' prefix).
        swa_state = swa_model.module.state_dict()
        best_state = {k: v.detach().cpu().clone() for k, v in swa_state.items()}
        # Re-evaluate to log the SWA val_loss/val_acc.
        model.load_state_dict(best_state)
        swa_val_loss, swa_val_acc, _, _ = evaluate(model, dl_val, device, criterion)
        print(f"SWA result: val_loss={swa_val_loss:.4f}  val_acc={swa_val_acc:.4f}")
        best_val_loss = swa_val_loss
        best_val_acc = swa_val_acc

    if best_state is not None:
        model.load_state_dict(best_state)

    # Val predictions from the chosen model — needed for post-hoc calibration
    # (temperature scaling) in _helpers/medical_kpis.py.
    _, _, val_probs, val_labels_arr = evaluate(model, dl_val, device, criterion)

    # Final test eval, touched once.
    test_loss, test_acc, test_probs, test_labels = evaluate(
        model, dl_test, device, criterion)
    print(f"\nFinal test: loss={test_loss:.4f} acc={test_acc:.4f}")

    out_dir = OUT_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "test_probs.npy", test_probs)
    np.save(out_dir / "test_labels.npy", test_labels)
    np.save(out_dir / "val_probs.npy", val_probs)
    np.save(out_dir / "val_labels.npy", val_labels_arr)
    if best_state is not None:
        torch.save(best_state, out_dir / "best_state.pt")

    summary = {
        "run_name": args.run_name,
        "architecture": {
            "n_blocks": args.n_blocks, "base_channels": args.base_channels,
            "activation": args.activation, "padding": args.padding,
            "stride_mode": args.stride_mode, "use_bn": args.use_bn,
            "use_dropout": args.use_dropout, "init": args.init,
            "n_params": n_params,
        },
        "regularization": {
            "weight_decay": args.weight_decay, "augment": args.augment,
            "augment_policy": args.augment_policy,
            "label_smoothing": args.label_smoothing,
            "cutmix_alpha": args.cutmix_alpha,
            "early_stop_patience": args.early_stop_patience,
        },
        "training": {
            "img_size": args.img_size, "batch_size": args.batch_size,
            "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
            "optimizer": args.optimizer,
            "use_swa": args.use_swa,
            "swa_start_frac": args.swa_start_frac if args.use_swa else None,
            "n_folds": args.n_folds, "fold": args.fold,
            "elapsed_min": (time.time() - t0) / 60,
        },
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
