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


def build_transforms(img_size: int, train: bool, augment: bool):
    base = [transforms.Resize((img_size, img_size))]
    if train and augment:
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


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, opt, device, criterion):
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        total_correct += ((logits > 0).long() == y.long()).sum().item()
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
    p.add_argument("--augment", action="store_true",
                   help="enable training-time augmentation (flip+affine+jitter)")
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="stop if val loss doesn't improve for N epochs (0 = off)")
    # Training
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
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

    train_tf = build_transforms(args.img_size, train=True, augment=args.augment)
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

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    t0 = time.time()
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, opt, device, criterion)
        va_loss, va_acc, _, _ = evaluate(model, dl_val, device, criterion)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"  ep{epoch + 1:>2}  train {tr_loss:.4f}/{tr_acc:.4f}  "
              f"val {va_loss:.4f}/{va_acc:.4f}")

        # Track best by val_loss (avoids picking a noisy spike in val_acc).
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if args.early_stop_patience and no_improve >= args.early_stop_patience:
                print(f"  early-stop at epoch {epoch + 1} (no val improvement "
                      f"in {args.early_stop_patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test eval, touched once.
    test_loss, test_acc, test_probs, test_labels = evaluate(
        model, dl_test, device, criterion)
    print(f"\nFinal test: loss={test_loss:.4f} acc={test_acc:.4f}")

    out_dir = OUT_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "test_probs.npy", test_probs)
    np.save(out_dir / "test_labels.npy", test_labels)
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
            "early_stop_patience": args.early_stop_patience,
        },
        "training": {
            "img_size": args.img_size, "batch_size": args.batch_size,
            "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
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
