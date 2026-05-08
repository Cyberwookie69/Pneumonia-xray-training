"""Run k folds of pneumonia_train.py back-to-back, each in its own subprocess.

A fresh Python process per fold means a fresh DirectML context per fold, which
means whatever low-level state leaked during fold N can't haunt fold N+1.
This is the kind of architectural decision you only make after watching three
folds in a row die at the same step.

Usage:
    python pneumonia_run_folds.py
    python pneumonia_run_folds.py --start_fold 2          # resume from fold 2
    python pneumonia_run_folds.py --tag run5              # custom run-name prefix
    python pneumonia_run_folds.py --extra "--epochs_full 20 --mixup_alpha 0.4"
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# Use whatever Python is currently running this script — works on Colab,
# Linux, Windows venv, etc., without hardcoding any path.
VENV_PYTHON = Path(sys.executable)
TRAIN_SCRIPT = Path(__file__).resolve().parent / "pneumonia_train.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--start_fold", type=int, default=0)
    p.add_argument("--tag", default=None,
                   help="run-name prefix. Defaults to 'ens' (pretrained) or "
                        "'scratch' (when --from_scratch).")
    # Defaults reflect what actually worked: the ens_f0..f4 run with
    # mixup=0.2 + ema=0.999 finished slightly *below* the no-mixup/no-ema
    # baseline (ensemble 92.47% vs single-fold 93.11%). Mixup at 0.2 was too
    # aggressive for a pretrained backbone; EMA at 0.999 had a half-life of
    # ~700 steps, longer than our typical fold's effective length.
    p.add_argument("--mixup_alpha", type=float, default=0.0,
                   help="Beta(α,α) Mixup. 0 = off (default; safer for pretrained).")
    p.add_argument("--ema_decay", type=float, default=0.0,
                   help="EMA decay. 0 = off (default). Try 0.99 if you turn it on.")
    p.add_argument("--epochs_full", type=int, default=15)
    p.add_argument("--pretrained", action="store_true",
                   help="Run the pretrained transfer-learning variant. "
                        "Default is from-scratch (no pretraining).")
    p.add_argument("--max_session_minutes", type=float, default=0.0,
                   help="Hard wall-clock budget for the whole session (across "
                        "all folds). After each fold subprocess returns, if "
                        "elapsed > limit, we exit instead of starting the next "
                        "fold. The limit is also forwarded into each fold so "
                        "training stops mid-fold at a clean epoch boundary.")
    p.add_argument("--extra", default="", help="extra flags forwarded to pneumonia_train.py")
    return p.parse_args()


def main():
    args = parse_args()
    extra_flags = args.extra.split() if args.extra else []
    if args.pretrained:
        extra_flags = ["--pretrained"] + extra_flags
    tag = args.tag or ("ens" if args.pretrained else "scratch")
    started = time.time()
    session_limit_s = args.max_session_minutes * 60 if args.max_session_minutes > 0 else 0.0

    def remaining_minutes():
        if session_limit_s == 0:
            return None
        return max(0.0, (session_limit_s - (time.time() - started)) / 60)

    for fold in range(args.start_fold, args.n_folds):
        run_name = f"{tag}_f{fold}"
        # Skip folds that are already fully trained — summary.json exists.
        # Saves ~15 s per chunk-restart in long chunked sessions.
        # Resolve via env override or default to <project>/runs (same logic as
        # pneumonia_train.py, but we don't import it to avoid heavy timm/torch
        # imports just for a path lookup).
        import os as _os
        runs_root = Path(_os.environ.get("PNEUMONIA_RUNS",
                                          TRAIN_SCRIPT.parent / "runs"))
        summary = runs_root / run_name / "summary.json"
        if summary.exists():
            print(f"\nFold {fold} ({run_name}) already complete (summary.json exists) — skipping.",
                  flush=True)
            continue
        # If a budget is set, give the fold whatever remains. The fold itself
        # checks --max_session_minutes after each epoch and exits cleanly when
        # exceeded; here we also check between folds to avoid starting a new
        # one we can't finish.
        rem = remaining_minutes()
        if rem is not None and rem <= 0.5:  # less than 30s left — don't bother
            print(f"\nSession time budget exhausted ({args.max_session_minutes:.1f} min). "
                  f"Stopping before fold {fold}. Re-run later to continue.", flush=True)
            sys.exit(0)
        cmd = [
            str(VENV_PYTHON), "-u", str(TRAIN_SCRIPT),
            "--n_folds", str(args.n_folds),
            "--fold", str(fold),
            "--epochs_full", str(args.epochs_full),
            "--mixup_alpha", str(args.mixup_alpha),
            "--ema_decay", str(args.ema_decay),
            "--run_name", run_name,
            # --resume is harmless when there's no checkpoint, and a lifesaver
            # when there is. Always pass it.
            "--resume",
        ] + extra_flags
        if rem is not None:
            cmd += ["--max_session_minutes", f"{rem:.2f}"]
        print(f"\n{'=' * 70}\nFOLD {fold}/{args.n_folds - 1}  -->  {run_name}\n{'=' * 70}", flush=True)
        print("Command:", " ".join(cmd), flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd)
        elapsed = time.time() - t0
        print(f"\nFold {fold} finished with exit code {rc} in {elapsed / 60:.1f} min", flush=True)
        if rc != 0:
            # Don't barrel into the next fold while DirectML is still picking up
            # the pieces. Bail and let the human deal with it.
            print(f"WARNING: fold {fold} returned non-zero exit code. "
                  f"Re-run with --start_fold {fold} to continue (the train script "
                  f"already passes --resume).")
            sys.exit(rc)

    total = (time.time() - started) / 60
    print(f"\nAll {args.n_folds} folds done in {total:.1f} min.")
    print(f"Run dirs: c:\\temp\\pneumonia\\runs\\{tag}_f0 ... {tag}_f{args.n_folds - 1}")
    print(f"Next: python pneumonia_eval.py --ensemble {tag}_f0,{tag}_f1,..., --use_best")


if __name__ == "__main__":
    main()
