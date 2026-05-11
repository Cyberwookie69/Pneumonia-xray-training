"""Compute the four medical KPIs from a saved ensemble run.

Sensitivity, Specificity, AUROC, ECE — at the default 0.5 threshold,
the val-tuned best-accuracy threshold, and a sensitivity-targeted
threshold (>= 0.97 sensitivity, max specificity).

Reads runs/<name>/test_probs.npy and test_labels.npy. Run after pneumonia_eval.py.

Usage:
    python _helpers/medical_kpis.py                       # defaults to runs/ensemble
    python _helpers/medical_kpis.py --run runs/ensemble
    python _helpers/medical_kpis.py --target_sensitivity 0.98
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def metrics_at(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / len(labels)
    return {"threshold": float(threshold), "acc": acc, "sensitivity": sens,
            "specificity": spec, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def fit_temperature(val_probs, val_labels, bounds=(0.5, 20.0)):
    """Fit a single scalar T on validation by minimising binary NLL.

    Standard Platt-style post-hoc calibration. Returns the optimal T.
    T == 1 means no rescaling. T > 1 softens overconfident predictions;
    T < 1 sharpens under-confident ones.
    """
    from scipy.optimize import minimize_scalar
    eps = 1e-7
    val_probs = np.clip(np.asarray(val_probs, dtype=np.float64), eps, 1 - eps)
    val_labels = np.asarray(val_labels, dtype=np.float64)
    val_logits = np.log(val_probs / (1 - val_probs))

    def neg_ll(T):
        p = 1.0 / (1.0 + np.exp(-val_logits / T))
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(val_labels * np.log(p) + (1 - val_labels) * np.log(1 - p))

    res = minimize_scalar(neg_ll, bounds=bounds, method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


def apply_temperature(probs, T):
    """Apply temperature T to existing sigmoid probabilities."""
    eps = 1e-7
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    return 1.0 / (1.0 + np.exp(-logits / T))


def expected_calibration_error(probs, labels, n_bins=10):
    """Standard ECE: weighted gap between predicted confidence and actual accuracy.

    confidence = max(p, 1-p) (distance from "I have no idea")
    """
    confidence = np.where(probs > 0.5, probs, 1 - probs)
    preds = (probs > 0.5).astype(int)
    correct = (preds == labels).astype(float)

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    bin_data = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        n = int(mask.sum())
        if n == 0:
            bin_data.append({"lo": float(lo), "hi": float(hi), "n": 0,
                             "acc": None, "conf": None, "gap": None})
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidence[mask].mean())
        gap = abs(bin_acc - bin_conf)
        ece += (n / len(labels)) * gap
        bin_data.append({"lo": float(lo), "hi": float(hi), "n": n,
                         "acc": bin_acc, "conf": bin_conf, "gap": gap})
    return float(ece), bin_data


def find_sensitivity_threshold(probs, labels, target_sensitivity=0.97):
    """Lowest threshold that still hits the sensitivity target, with max specificity."""
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 199), probs]))
    best = None
    for t in sorted(candidates):
        m = metrics_at(probs, labels, t)
        if m["sensitivity"] >= target_sensitivity:
            if best is None or m["specificity"] > best["specificity"]:
                best = m
    return best


def best_accuracy_threshold(probs, labels):
    best, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        acc = ((probs >= t).astype(int) == labels).mean()
        if acc > best:
            best, best_t = acc, float(t)
    return best_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/ensemble",
                   help="run dir containing test_probs.npy and test_labels.npy")
    p.add_argument("--target_sensitivity", type=float, default=0.97)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--no_temperature", action="store_true",
                   help="Skip temperature scaling even if val_probs.npy exists.")
    args = p.parse_args()

    run_dir = Path(args.run)
    probs = np.load(run_dir / "test_probs.npy")
    labels = np.load(run_dir / "test_labels.npy").astype(int)

    # Optional: post-hoc temperature scaling using held-out val predictions.
    T_star = None
    probs_uncal = probs.copy()
    val_p_path = run_dir / "val_probs.npy"
    val_l_path = run_dir / "val_labels.npy"
    if not args.no_temperature and val_p_path.exists() and val_l_path.exists():
        val_probs = np.load(val_p_path)
        val_labels = np.load(val_l_path).astype(int)
        T_star = fit_temperature(val_probs, val_labels)
        probs = apply_temperature(probs, T_star)
        print(f"Temperature scaling: fitted T* = {T_star:.4f} on "
              f"{len(val_labels)} val samples. Applied to test.")
    elif not args.no_temperature:
        print(f"(No val_probs.npy in {run_dir} — skipping temperature scaling.)")

    print(f"Run: {run_dir}")
    print(f"Test images: {len(labels)} (PNE prevalence: {labels.mean():.3f})")
    print()

    # AUROC (threshold-independent — invariant under temperature scaling)
    auroc = float(roc_auc_score(labels, probs))

    # ECE on calibrated probs (after temperature scaling if applied)
    ece, bin_data = expected_calibration_error(probs, labels, n_bins=args.n_bins)
    # Also compute the uncalibrated ECE for comparison.
    ece_uncal, _ = expected_calibration_error(probs_uncal, labels,
                                              n_bins=args.n_bins)

    # Three operating points
    m_default = metrics_at(probs, labels, 0.5)
    best_t = best_accuracy_threshold(probs, labels)
    m_best_acc = metrics_at(probs, labels, best_t)
    m_sens = find_sensitivity_threshold(probs, labels, args.target_sensitivity)

    # ── Print KPI table ─────────────────────────────────────────────────────
    print("=" * 78)
    print("MEDICAL KPI SUMMARY")
    print("=" * 78)
    print(f"{'Metric':<28}{'@ t=0.5':>14}{'@ best-acc':>14}{'@ sens>=' + f'{args.target_sensitivity:.2f}':>22}")
    print("-" * 78)
    rows = [
        ("Threshold t", m_default["threshold"], m_best_acc["threshold"], m_sens["threshold"]),
        ("Accuracy", m_default["acc"], m_best_acc["acc"], m_sens["acc"]),
        ("Sensitivity (PNE recall)", m_default["sensitivity"], m_best_acc["sensitivity"], m_sens["sensitivity"]),
        ("Specificity (NORM recall)", m_default["specificity"], m_best_acc["specificity"], m_sens["specificity"]),
    ]
    for name, a, b, c in rows:
        print(f"{name:<28}{a:>14.4f}{b:>14.4f}{c:>22.4f}")

    print(f"{'Confusion (TP/TN/FP/FN)':<28}"
          f"{m_default['tp']}/{m_default['tn']}/{m_default['fp']}/{m_default['fn']:>4}".rjust(42)
          + f"  {m_best_acc['tp']}/{m_best_acc['tn']}/{m_best_acc['fp']}/{m_best_acc['fn']}"
          + f"   {m_sens['tp']}/{m_sens['tn']}/{m_sens['fp']}/{m_sens['fn']}")

    print()
    print(f"AUROC                       {auroc:.4f}      (threshold-independent)")
    if T_star is not None:
        print(f"ECE  ({args.n_bins} bins) uncalibrated  {ece_uncal:.4f}")
        print(f"ECE  ({args.n_bins} bins) calibrated    {ece:.4f}      "
              f"({'well-calibrated' if ece < 0.05 else 'mis-calibrated' if ece > 0.10 else 'borderline'})  "
              f"[T*={T_star:.3f}]")
    else:
        print(f"ECE  ({args.n_bins} bins)              {ece:.4f}      "
              f"({'well-calibrated' if ece < 0.05 else 'mis-calibrated' if ece > 0.10 else 'borderline'})")

    # Binomial standard error: any accuracy delta < 2*sigma is within noise.
    import math
    p = m_default["acc"]
    sigma = math.sqrt(p * (1 - p) / len(labels))
    print(f"Test accuracy noise floor:  sigma ~= {sigma:.4f} "
          f"({sigma*100:.2f} pp); 95% CI half-width ~= {1.96*sigma*100:.2f} pp")

    # ── Calibration table ───────────────────────────────────────────────────
    print()
    print("Calibration breakdown (uniform bins of confidence = max(p, 1-p)):")
    print(f"  {'bin':<14}{'n':>6}{'avg_conf':>12}{'acc':>10}{'gap':>10}")
    for b in bin_data:
        if b["n"] == 0:
            print(f"  [{b['lo']:.2f},{b['hi']:.2f}){'-':>9}{'-':>12}{'-':>10}{'-':>10}")
        else:
            print(f"  [{b['lo']:.2f},{b['hi']:.2f}){b['n']:>9}{b['conf']:>12.4f}"
                  f"{b['acc']:>10.4f}{b['gap']:>10.4f}")

    # ── Save augmented summary ──────────────────────────────────────────────
    out = {
        "auroc": auroc,
        "ece": ece,
        "ece_uncalibrated": ece_uncal,
        "temperature_T_star": T_star,
        "ece_n_bins": args.n_bins,
        "calibration_bins_uniform": bin_data,
        "operating_points": {
            "default_0.5": m_default,
            "best_accuracy": m_best_acc,
            f"sensitivity_target_{args.target_sensitivity:.2f}": m_sens,
        },
        "test_prevalence_pne": float(labels.mean()),
        "n_test": int(len(labels)),
        "noise_floor_sigma": sigma,
        "noise_floor_95ci_pp": 1.96 * sigma * 100,
    }
    with open(run_dir / "medical_kpis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {run_dir / 'medical_kpis.json'}")


if __name__ == "__main__":
    main()
