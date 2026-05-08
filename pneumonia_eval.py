"""Evaluate one or more checkpoints on the official test set.

Single model:
    python pneumonia_eval.py --run_name resnet50_f0_1778105035 --use_best
    python pneumonia_eval.py --run_name <name> --no_tta --num_workers 0

Ensemble (averages probabilities across runs):
    python pneumonia_eval.py --ensemble ens_f0,ens_f1,ens_f2,ens_f3,ens_f4 --use_best

TTA is on by default. Flipping happens on the CPU side because DirectML treats
`torch.flip` followed by a forward pass as a personal insult.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from torch.utils.data import DataLoader

import timm
from pneumonia_train import (
    DATA_ROOT, OUT_DIR, FocalLoss, XRayDataset, build_transforms,
    evaluate, get_device, list_images,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default=None, help="single run name")
    p.add_argument("--ensemble", default=None, help="comma-separated list of run names")
    p.add_argument("--model", default="resnet50.a1_in1k")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--no_tta", dest="tta", action="store_false")
    p.add_argument("--use_best", action="store_true",
                   help="load best_state from checkpoint instead of last")
    p.set_defaults(tta=True)
    return p.parse_args()


def load_checkpoint(run_name, model_arch, use_best, device):
    """Reanimate a model from disk. Whether it's any good is your problem.

    The checkpoint records its own architecture tag (`model_tag`); we trust
    that over the CLI `--model` flag so old runs trained with different
    backbones still rehydrate correctly.
    """
    run_dir = OUT_DIR / run_name
    state_path = run_dir / "last_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"No checkpoint at {state_path}")
    ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
    progress = ckpt["progress"]
    best_val_acc = progress.get("best_val_acc", 0.0)
    print(f"  [{run_name}] phase {progress['phase']} ep{progress['epoch_done']}, "
          f"best_val_acc={best_val_acc:.4f}")

    # Architecture detection: prefer the saved tag (strip our `_scratch`
    # suffix to recover the underlying timm name), fall back to the CLI flag
    # for legacy runs that didn't save it.
    saved_tag = ckpt.get("model_tag") or model_arch
    arch = saved_tag.removesuffix("_scratch")
    model = timm.create_model(arch, pretrained=False, num_classes=1)
    print(f"  [{run_name}] arch: {arch}{' (from-scratch)' if saved_tag.endswith('_scratch') else ''}")

    if use_best and ckpt.get("best_state") is not None:
        model.load_state_dict(ckpt["best_state"])
        print(f"  [{run_name}] loaded BEST state")
    else:
        model.load_state_dict(ckpt["model"])
        print(f"  [{run_name}] loaded LAST state")
    return model.to(device), best_val_acc


def report_confusion(probs, labels, threshold=0.5, prefix=""):
    """The four-cell story every binary classifier wants to hide."""
    preds = (probs > threshold).astype(int)
    acc = (preds == labels).mean()
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    print(f"{prefix}acc={acc:.4f}  TP={tp} TN={tn} FP={fp} FN={fn}")
    if tp + fn:
        print(f"{prefix}recall(PNE)={tp / (tp + fn):.4f}  specificity(NORM)={tn / (tn + fp):.4f}  "
              f"precision(PNE)={tp / (tp + fp):.4f}")
    return {"acc": float(acc), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": threshold}


def kappa_interpretation(k):
    """Landis & Koch 1977. Subjective scale that everyone pretends is rigorous."""
    if k < 0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def report_advanced(probs, labels, threshold=0.5):
    """Cohen's kappa, ROC-AUC, calibration-by-confidence, and the
    'confidently wrong' bucket — i.e. the cases that tell you whether your
    model is overconfident or whether the labels were just bad."""
    preds = (probs > threshold).astype(int)

    print("\n--- Advanced metrics ---")
    kappa = cohen_kappa_score(labels, preds)
    print(f"Cohen's kappa: {kappa:.4f}  ({kappa_interpretation(kappa)})")
    print(f"  (Radiologists agreeing with each other on chest X-rays usually score 0.6-0.85.)")

    try:
        auc = roc_auc_score(labels, probs)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception as e:
        auc = None
        print(f"ROC-AUC: n/a ({e})")

    # Bin by max(p, 1-p): how far the model's prediction is from "I have no idea".
    confidence = np.where(probs > 0.5, probs, 1 - probs)
    print("\nAccuracy per confidence bin (how sure the model claims to be):")
    print(f"  {'bin':<14}  {'count':>6}  {'acc':>7}  {'avg_conf':>8}")
    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 0.95),
            (0.95, 0.99), (0.99, 1.001)]
    bin_stats = []
    for lo, hi in bins:
        mask = (confidence >= lo) & (confidence < hi)
        n = int(mask.sum())
        if n == 0:
            print(f"  [{lo:.2f},{hi:.2f})    {n:>6}  {'-':>7}  {'-':>8}")
            continue
        acc = float((preds[mask] == labels[mask]).mean())
        avg_c = float(confidence[mask].mean())
        print(f"  [{lo:.2f},{hi:.2f})    {n:>6}  {acc:>7.4f}  {avg_c:>8.4f}")
        bin_stats.append({"lo": lo, "hi": hi, "n": n, "acc": acc, "avg_conf": avg_c})

    # Borderline: probability in [0.40, 0.60]. The model is basically shrugging.
    borderline_mask = (probs >= 0.40) & (probs <= 0.60)
    n_border = int(borderline_mask.sum())
    if n_border:
        border_correct = int((preds[borderline_mask] == labels[borderline_mask]).sum())
        print(f"\nBorderline cases (prob in [0.40, 0.60]): {n_border} images, "
              f"{border_correct}/{n_border} correct ({border_correct / n_border:.2%})")
        print("  -> these are the same scans where two radiologists would disagree.")

    # >90% confident and wrong. Either label noise or genuinely weird scans.
    confidently_wrong = (confidence > 0.90) & (preds != labels)
    n_cw = int(confidently_wrong.sum())
    if n_cw:
        print(f"\nConfidently wrong (>90% sure but mistaken): {n_cw} images")
        print("  -> probably mislabeled in the dataset, or atypical presentations.")

    return {
        "kappa": float(kappa),
        "kappa_interpretation": kappa_interpretation(kappa),
        "roc_auc": float(auc) if auc is not None else None,
        "borderline_count": n_border,
        "confidently_wrong_count": n_cw,
        "calibration_bins": bin_stats,
    }


def main():
    args = parse_args()
    if not args.run_name and not args.ensemble:
        raise SystemExit("Provide --run_name or --ensemble (or both, if you're feeling fancy).")

    device, device_name = get_device()
    print(f"Device: {device_name}")

    items = list_images(DATA_ROOT)
    test_items = [(p, l) for p, l, s in items if s == "test"]

    # Two test datasets: identical except the second pre-flips horizontally on
    # the CPU. The model never gets asked to flip anything itself.
    eval_tf = build_transforms(args.img_size, train=False)
    eval_tf_flip = build_transforms(args.img_size, train=False, hflip_eval=True)
    ds_test = XRayDataset(test_items, eval_tf)
    ds_test_flip = XRayDataset(test_items, eval_tf_flip)
    nw = args.num_workers
    dl_test = DataLoader(ds_test, batch_size=args.eval_batch_size, shuffle=False,
                         num_workers=nw, persistent_workers=nw > 0)
    dl_test_flip = DataLoader(ds_test_flip, batch_size=args.eval_batch_size, shuffle=False,
                              num_workers=nw, persistent_workers=nw > 0)
    print(f"Test images: {len(test_items)} | eval_batch_size={args.eval_batch_size} | "
          f"TTA={args.tta} | num_workers={nw}")

    criterion = FocalLoss()
    run_names = args.ensemble.split(",") if args.ensemble else [args.run_name]
    run_names = [r.strip() for r in run_names if r.strip()]

    all_probs = []
    per_run_summaries = []
    labels = None
    for run_name in run_names:
        print(f"\n--- Loading {run_name} ---")
        model, best_val_acc = load_checkpoint(run_name, args.model, args.use_best, device)
        print(f"  Running test eval (TTA={args.tta})...")
        loss, acc, probs, labels = evaluate(
            model, dl_test, criterion, device,
            tta=args.tta, loader_flip=dl_test_flip if args.tta else None,
        )
        print(f"  test acc={acc:.4f} loss={loss:.4f}")
        report_confusion(probs, labels, prefix=f"  [{run_name}] ")
        all_probs.append(probs)
        per_run_summaries.append({
            "run": run_name, "best_val_acc": float(best_val_acc),
            "test_acc": float(acc), "test_loss": float(loss),
        })
        del model
        import gc; gc.collect()

    assert labels is not None, "no checkpoints evaluated — nothing to ensemble"
    if len(all_probs) == 1:
        ensemble_probs = all_probs[0]
    else:
        ensemble_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
        print(f"\n=== ENSEMBLE of {len(all_probs)} models (mean of probabilities) ===")

    print("\n=== FINAL TEST RESULT ===")
    final = report_confusion(ensemble_probs, labels, prefix="  ")

    # Threshold sweep — 0.5 is rarely the right answer if your classes are imbalanced.
    print("\nThreshold sweep:")
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.30, 0.70, 21):
        acc = ((ensemble_probs > t).astype(int) == labels).mean()
        marker = "  <--" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
        print(f"  t={t:.2f}: acc={acc:.4f}{marker}")
    print(f"\nBest threshold: {best_t:.2f} -> acc={best_acc:.4f} "
          f"(default 0.5 -> acc={final['acc']:.4f})")

    advanced = report_advanced(ensemble_probs, labels, threshold=best_t)

    out_dir = OUT_DIR / "ensemble" if len(run_names) > 1 else OUT_DIR / run_names[0]
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "test_probs.npy", ensemble_probs)
    np.save(out_dir / "test_labels.npy", labels)
    summary = {
        "runs": run_names,
        "model": args.model,
        "tta": bool(args.tta),
        "use_best": bool(args.use_best),
        "default_threshold": final,
        "best_threshold": {"t": best_t, "acc": float(best_acc)},
        "per_run": per_run_summaries,
        "advanced": advanced,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
