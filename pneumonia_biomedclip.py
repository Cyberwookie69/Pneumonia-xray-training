"""Linear-probe baseline using BiomedCLIP (vision-language pretrained on biomedical literature).

BiomedCLIP is Microsoft's vision-language CLIP-style model pretrained on
~15M biomedical image-text pairs from PubMed Open Access. We use only the
image encoder, frozen, plus a 5-fold logistic regression head — the
standard linear-probe protocol for evaluating self-supervised features.

Usage:
    python pneumonia_biomedclip.py --run_name biomedclip_ensemble
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from pneumonia_train import DATA_ROOT, OUT_DIR, get_device, list_images


MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="biomedclip_ensemble")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=72)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--feature_cache", default=None)
    p.add_argument("--C", type=float, default=1.0)
    return p.parse_args()


class ImageDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.preprocess(img)


def extract_features(model, loader, device):
    feats = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            f = model.encode_image(batch)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"Loading {MODEL_ID}...")
    from open_clip import create_model_from_pretrained
    model, preprocess = create_model_from_pretrained(MODEL_ID)

    device, device_name = get_device()
    print(f"Device: {device_name}")
    model = model.to(device).eval()

    items = list_images(DATA_ROOT)
    train_pool = [(p, l) for p, l, s in items if s in ("train", "val")]
    test_items = [(p, l) for p, l, s in items if s == "test"]
    train_paths = [p for p, _ in train_pool]
    train_labels = np.array([l for _, l in train_pool])
    test_paths = [p for p, _ in test_items]
    test_labels = np.array([l for _, l in test_items])
    print(f"Train+val: {len(train_paths)}  Test: {len(test_paths)}")

    cache_dir = Path(args.feature_cache or (OUT_DIR / "biomedclip_features"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_feat_path = cache_dir / "train_pool_features.npy"
    test_feat_path = cache_dir / "test_features.npy"

    if train_feat_path.exists():
        print(f"Cached: {train_feat_path}")
        train_feats = np.load(train_feat_path)
    else:
        print(f"Extracting train+val features ({len(train_paths)} images)...")
        dl = DataLoader(ImageDataset(train_paths, preprocess), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
        t0 = time.time()
        train_feats = extract_features(model, dl, device)
        print(f"  done in {(time.time() - t0) / 60:.1f} min, shape={train_feats.shape}")
        np.save(train_feat_path, train_feats)

    if test_feat_path.exists():
        print(f"Cached: {test_feat_path}")
        test_feats = np.load(test_feat_path)
    else:
        print(f"Extracting test features ({len(test_paths)} images)...")
        dl = DataLoader(ImageDataset(test_paths, preprocess), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
        t0 = time.time()
        test_feats = extract_features(model, dl, device)
        print(f"  done in {(time.time() - t0) / 60:.1f} min, shape={test_feats.shape}")
        np.save(test_feat_path, test_feats)

    print(f"\n{args.n_folds}-fold linear classifier (C={args.C})...")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    test_probs_per_fold = []
    fold_results = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train_pool)),
                                                      train_labels)):
        clf = LogisticRegression(max_iter=2000, C=args.C, n_jobs=-1)
        clf.fit(train_feats[tr_idx], train_labels[tr_idx])
        val_acc = float(clf.score(train_feats[va_idx], train_labels[va_idx]))
        test_p = clf.predict_proba(test_feats)[:, 1]
        test_acc = float(((test_p > 0.5).astype(int) == test_labels).mean())
        print(f"  fold {fold}: val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")
        test_probs_per_fold.append(test_p)
        fold_results.append({"fold": fold, "val_acc": val_acc, "test_acc": test_acc})

    ensemble_probs = np.mean(np.stack(test_probs_per_fold, axis=0), axis=0)
    ensemble_acc = float(((ensemble_probs > 0.5).astype(int) == test_labels).mean())
    print(f"\nEnsemble test acc: {ensemble_acc:.4f}")

    out_dir = OUT_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "test_probs.npy", ensemble_probs)
    np.save(out_dir / "test_labels.npy", test_labels)
    summary = {
        "approach": "biomedclip_linear_probe",
        "model_id": MODEL_ID,
        "n_train_pool": len(train_pool),
        "n_test": len(test_items),
        "n_folds": args.n_folds,
        "feature_dim": int(train_feats.shape[1]),
        "logistic_C": args.C,
        "ensemble_test_acc": ensemble_acc,
        "per_fold": fold_results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
