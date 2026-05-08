"""Report-quality figures from training runs.

Subcommands:
    curves   — learning curves (loss + acc, per fold) from history.json
    features — t-SNE projection of penultimate features on the test set

Examples:
    python pneumonia_plots.py curves --runs ens_f0,ens_f1,ens_f2,ens_f3,ens_f4
    python pneumonia_plots.py curves --runs scratch_f0,scratch_f1,...
    python pneumonia_plots.py features --run ens_f3 --use_best
    python pneumonia_plots.py features --run ens_f3 --use_best --max_samples 624
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import timm
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from pneumonia_train import (
    DATA_ROOT, OUT_DIR, XRayDataset, build_transforms, get_device, list_images,
)


# Academic-friendly defaults — readable in print, no decorations.
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
})


def cmd_curves(args):
    """Plot training curves: train/val loss + acc, one line per fold."""
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    fold_data = []
    for run in runs:
        hist_path = OUT_DIR / run / "history.json"
        if not hist_path.exists():
            print(f"WARNING: {hist_path} missing — skipping {run}", file=sys.stderr)
            continue
        with open(hist_path) as f:
            hist = json.load(f)
        # Concatenate phase1 + phase2 epochs into one continuous timeline
        merged = []
        for entry in hist.get("phase1", []):
            merged.append({"global_ep": len(merged) + 1, "phase": 1, **entry})
        for entry in hist.get("phase2", []):
            merged.append({"global_ep": len(merged) + 1, "phase": 2, **entry})
        if not merged:
            print(f"WARNING: {run} has empty history — skipping", file=sys.stderr)
            continue
        fold_data.append((run, merged))

    if not fold_data:
        sys.exit("No usable history files found.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("tab10")

    # Loss panel
    for i, (run, merged) in enumerate(fold_data):
        eps = [e["global_ep"] for e in merged]
        tl = [e["train_loss"] for e in merged]
        vl = [e["val_loss"] for e in merged]
        c = cmap(i % 10)
        axes[0].plot(eps, tl, color=c, linewidth=1.4, label=f"{run} train")
        axes[0].plot(eps, vl, color=c, linewidth=1.4, linestyle="--",
                     label=f"{run} val")
    # Mark phase transition with vertical line at the first phase2 epoch
    p1_lens = [sum(1 for e in m if e["phase"] == 1) for _, m in fold_data]
    if p1_lens and any(p1_lens):
        avg_p1 = max(set(p1_lens), key=p1_lens.count)  # most common phase1 length
        axes[0].axvline(avg_p1 + 0.5, color="grey", linewidth=0.7, linestyle=":")
        axes[0].text(avg_p1 + 0.5, axes[0].get_ylim()[1] * 0.95, "  P2 starts",
                     color="grey", fontsize=8, va="top")
    axes[0].set_xlabel("Epoch (continuous, P1 + P2)")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & validation loss")
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")

    # Accuracy panel
    for i, (run, merged) in enumerate(fold_data):
        eps = [e["global_ep"] for e in merged]
        ta = [e["train_acc"] for e in merged]
        va = [e["val_acc"] for e in merged]
        c = cmap(i % 10)
        axes[1].plot(eps, ta, color=c, linewidth=1.4, label=f"{run} train")
        axes[1].plot(eps, va, color=c, linewidth=1.4, linestyle="--",
                     label=f"{run} val")
    axes[1].set_xlabel("Epoch (continuous, P1 + P2)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & validation accuracy")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")

    fig.suptitle(f"Learning curves — {len(fold_data)} folds", fontsize=12)
    # Derive a tag from the common prefix of the run names so each
    # `curves` invocation writes a unique pair of PNGs (no more overwrites
    # when plotting r50_288, cnx224, snr_r50 in sequence).
    common_tag = runs[0].rsplit("_f", 1)[0] if runs else "curves"
    if args.out:
        out = Path(args.out)
    else:
        out = OUT_DIR / "plots" / f"learning_curves_{common_tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"Saved: {out}")

    # Also a compact "summary" plot: best val_acc per fold
    fig2, ax = plt.subplots(figsize=(6, 3.5))
    bests = [max(e["val_acc"] for e in m) for _, m in fold_data]
    labels = [r for r, _ in fold_data]
    ax.bar(range(len(labels)), bests, color=[cmap(i) for i in range(len(labels))])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Best val accuracy")
    ax.set_ylim(0.7, 1.0)
    ax.axhline(np.mean(bests), color="black", linewidth=0.8, linestyle="--",
               label=f"mean = {np.mean(bests):.4f}")
    for i, b in enumerate(bests):
        ax.text(i, b + 0.003, f"{b:.3f}", ha="center", fontsize=8)
    ax.legend()
    ax.set_title(f"Per-fold best validation accuracy ({common_tag})")
    out2 = out.parent / f"fold_best_val_{common_tag}.png"
    fig2.tight_layout()
    fig2.savefig(out2)
    print(f"Saved: {out2}")


def cmd_features(args):
    """Extract penultimate features on the test set and visualise via t-SNE."""
    device, device_name = get_device()
    print(f"Device: {device_name}")

    # Load checkpoint (re-using the eval script's logic, but inline here so this
    # script stands alone)
    state_path = OUT_DIR / args.run / "last_state.pt"
    if not state_path.exists():
        sys.exit(f"No checkpoint at {state_path}")
    ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
    arch = (ckpt.get("model_tag") or "resnet50.a1_in1k").removesuffix("_scratch")
    model = timm.create_model(arch, pretrained=False, num_classes=1)
    state = ckpt.get("best_state") if args.use_best and ckpt.get("best_state") is not None else ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    model = model.to(device)
    print(f"Loaded {args.run} (arch={arch}, "
          f"{'best' if args.use_best and ckpt.get('best_state') else 'last'})")

    # Test set
    items = list_images(DATA_ROOT)
    test = [(p, l) for p, l, s in items if s == "test"]
    if args.max_samples and len(test) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(test), args.max_samples, replace=False)
        test = [test[i] for i in idx]

    eval_tf = build_transforms(args.img_size, train=False)
    ds = XRayDataset(test, eval_tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Extracting features from {len(test)} test images...")

    feats_all, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            # `forward_head(..., pre_logits=True)` returns the penultimate feature
            # vector (post-pool, pre-classifier) for almost any timm model.
            features = model.forward_features(x)
            pre_logits = model.forward_head(features, pre_logits=True)
            logits = model.forward_head(features).squeeze(-1)
            feats_all.append(pre_logits.cpu().numpy())
            labels_all.append(y.numpy())
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
    feats = np.concatenate(feats_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    probs = np.concatenate(probs_all, axis=0)
    print(f"Feature shape: {feats.shape}")

    # t-SNE
    print("Running t-SNE (perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=args.seed, init="pca")
    proj = tsne.fit_transform(feats)
    print("Done.")

    preds = (probs > 0.5).astype(int)
    correct = preds == labels

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: coloured by true label
    for cls, name, color in [(0, "NORMAL", "#3ba9ee"), (1, "PNEUMONIA", "#f48975")]:
        mask = labels == cls
        axes[0].scatter(proj[mask, 0], proj[mask, 1], s=14, alpha=0.7,
                        color=color, label=f"{name} (n={int(mask.sum())})",
                        edgecolors="none")
    axes[0].set_title("Penultimate features by true label")
    axes[0].set_xlabel("t-SNE dim 1")
    axes[0].set_ylabel("t-SNE dim 2")
    axes[0].legend()

    # Right: coloured by correct/wrong, marker by class
    for cls, marker in [(0, "o"), (1, "^")]:
        for ok, color, name in [(True, "#2e7d32", "correct"), (False, "#c62828", "wrong")]:
            mask = (labels == cls) & (correct == ok)
            if mask.sum() == 0:
                continue
            axes[1].scatter(proj[mask, 0], proj[mask, 1], s=14, alpha=0.7,
                            marker=marker, color=color, edgecolors="none",
                            label=f"{'NORMAL' if cls == 0 else 'PNEUMONIA'} {name} ({int(mask.sum())})")
    axes[1].set_title("Penultimate features — correct vs misclassified")
    axes[1].set_xlabel("t-SNE dim 1")
    axes[1].set_ylabel("t-SNE dim 2")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"t-SNE projection of {arch} penultimate features on test set ({args.run})",
                 fontsize=11)
    fig.tight_layout()

    out_dir = OUT_DIR / args.run / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "best" if args.use_best else "last"
    out = out_dir / f"tsne_{suffix}.png"
    fig.savefig(out)

    # Also dump the raw data so the figure can be re-rendered without
    # recomputing features (which is the expensive part).
    np.savez(
        out_dir / f"tsne_{suffix}_data.npz",
        features=feats, projection=proj, labels=labels, probs=probs,
        run=args.run, arch=arch,
    )
    print(f"Saved: {out}")
    print(f"Raw data: {out_dir / f'tsne_{suffix}_data.npz'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_curves = sub.add_parser("curves", help="Learning curves from history.json")
    p_curves.add_argument("--runs", required=True,
                          help="comma-separated run names (e.g. ens_f0,ens_f1,...)")
    p_curves.add_argument("--out", default=None,
                          help="output PNG path; defaults to runs/plots/learning_curves.png")
    p_curves.set_defaults(func=cmd_curves)

    p_feat = sub.add_parser("features", help="t-SNE of penultimate features")
    p_feat.add_argument("--run", required=True, help="single run name")
    p_feat.add_argument("--use_best", action="store_true")
    p_feat.add_argument("--img_size", type=int, default=224)
    p_feat.add_argument("--batch_size", type=int, default=8)
    p_feat.add_argument("--max_samples", type=int, default=624,
                        help="cap on test images to project (full test = 624)")
    p_feat.add_argument("--seed", type=int, default=42)
    p_feat.set_defaults(func=cmd_features)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
