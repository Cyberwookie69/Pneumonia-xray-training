"""Generate Grad-CAM heatmaps for a trained checkpoint.

Shows where the model is looking when it makes a prediction. Useful for
clinician trust and for catching cases where the model relies on dataset
artefacts (text annotations, machine identifiers) instead of pathology.

Usage:
    python pneumonia_gradcam.py --run_name ens_f0
    python pneumonia_gradcam.py --run_name ens_f0 --n_samples 8 --use_best
    python pneumonia_gradcam.py --run_name ens_f0 --image c:/path/to/xray.jpeg
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from pneumonia_train import (
    DATA_ROOT, OUT_DIR, build_transforms, get_device, list_images,
)

OUT_BASE = OUT_DIR  # alias for back-compat with earlier code in this file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True, help="folder under runs/")
    p.add_argument("--use_best", action="store_true",
                   help="use best_state from checkpoint instead of last")
    p.add_argument("--image", default=None,
                   help="single image path to visualise; if omitted, samples N from test set")
    p.add_argument("--n_samples", type=int, default=6,
                   help="number of test images to visualise (when --image is omitted)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--out_subdir", default="gradcam",
                   help="subdir inside the run dir to save heatmaps to")
    return p.parse_args()


def find_target_layer(model):
    """Return the last conv-block of a timm ResNet/ConvNeXt-style backbone.

    Different timm architectures expose their final feature stage under
    different attribute names; we just check the common ones.
    """
    for attr in ("layer4", "stages", "blocks", "features"):
        if hasattr(model, attr):
            mod = getattr(model, attr)
            # Take the last sub-module (last residual block / last stage)
            if hasattr(mod, "__iter__") or hasattr(mod, "__getitem__"):
                try:
                    return mod[-1]
                except Exception:
                    pass
            return mod
    raise RuntimeError("Could not auto-detect a target layer for Grad-CAM")


def load_model(run_name, use_best, device):
    state_path = OUT_BASE / run_name / "last_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
    arch = (ckpt.get("model_tag") or "resnet50.a1_in1k").removesuffix("_scratch")
    model = timm.create_model(arch, pretrained=False, num_classes=1)
    state = ckpt.get("best_state") if use_best and ckpt.get("best_state") is not None else ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    return model.to(device), arch


def preprocess(img_path, img_size):
    """Return (display_rgb_0_1, tensor_for_model)."""
    pil = Image.open(img_path).convert("RGB")
    pil_resized = pil.resize((img_size, img_size), Image.BILINEAR)
    rgb_disp = np.asarray(pil_resized).astype(np.float32) / 255.0
    tf = build_transforms(img_size, train=False)
    tensor = tf(pil).unsqueeze(0)
    return rgb_disp, tensor


def predict_label(model, tensor, device):
    """Return (prob, label_str) for the single sample in tensor."""
    with torch.no_grad():
        logit = model(tensor.to(device)).squeeze().cpu().item()
    p = float(1 / (1 + np.exp(-logit)))
    return p, ("PNEUMONIA" if p > 0.5 else "NORMAL")


def make_heatmap(cam, tensor, device, target_class):
    """Run Grad-CAM and return a [H,W] uint8 heatmap in [0,1]."""
    # ClassifierOutputTarget for binary single-logit setup is awkward because
    # most pytorch-grad-cam targets assume multi-class logits. We bypass it by
    # passing a custom function that just returns the logit directly.
    class SingleLogitTarget:
        def __init__(self, want_positive):
            self.want_positive = want_positive

        def __call__(self, model_output):
            # model_output is a tensor of shape (n_classes,) — but for our
            # binary single-logit network it's just (1,). We negate when we
            # want the "NORMAL" class so the gradient direction flips.
            return model_output[0] if self.want_positive else -model_output[0]

    targets = [SingleLogitTarget(want_positive=(target_class == 1))]
    grayscale = cam(input_tensor=tensor.to(device), targets=targets)
    return grayscale[0]  # [H,W]


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    device, device_name = get_device()
    print(f"Device: {device_name}")

    model, arch = load_model(args.run_name, args.use_best, device)
    target_layer = find_target_layer(model)
    print(f"Run: {args.run_name} | arch: {arch} | target layer: {type(target_layer).__name__}")

    cam = GradCAM(model=model, target_layers=[target_layer])

    if args.image:
        items = [(args.image, -1)]
    else:
        all_items = list_images(DATA_ROOT)
        test_items = [(p, l) for p, l, s in all_items if s == "test"]
        random.shuffle(test_items)
        items = test_items[:args.n_samples]

    out_dir = OUT_BASE / args.run_name / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving heatmaps to: {out_dir}")

    summary = []
    raw_heatmaps = []  # collect for npz dump
    for i, (path, true_label) in enumerate(items):
        rgb, tensor = preprocess(path, args.img_size)
        prob, pred_label = predict_label(model, tensor, device)
        # CAM for the predicted class — what the model itself relied on
        target_cls = 1 if prob > 0.5 else 0
        heatmap = make_heatmap(cam, tensor, device, target_cls)
        overlay = show_cam_on_image(rgb, heatmap, use_rgb=True)

        # Save side-by-side: original | heatmap | overlay
        h_only = (heatmap * 255).astype(np.uint8)
        h_color = cv2.applyColorMap(h_only, cv2.COLORMAP_JET)
        h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
        original_u8 = (rgb * 255).astype(np.uint8)
        sep = np.full((args.img_size, 4, 3), 30, dtype=np.uint8)
        triptych = np.concatenate([original_u8, sep, h_color, sep, overlay], axis=1)

        true_str = "?" if true_label == -1 else ("PNEUMONIA" if true_label == 1 else "NORMAL")
        fname = f"{i:02d}_pred-{pred_label}_p{prob:.2f}_true-{true_str}_{Path(path).stem}.png"
        cv2.imwrite(str(out_dir / fname), cv2.cvtColor(triptych, cv2.COLOR_RGB2BGR))
        summary.append({
            "file": fname, "source": str(path),
            "true": true_str, "pred": pred_label, "prob": prob,
        })
        raw_heatmaps.append(heatmap)
        correct = "OK" if (true_label == -1 or true_str == pred_label) else "WRONG"
        print(f"  [{i:02d}] {Path(path).name[:45]:<45} pred={pred_label} (p={prob:.3f}) "
              f"true={true_str:<10}{correct}")

    # Dump raw heatmap arrays + metadata so figures can be re-rendered later
    # (different colormap, bigger panels, alternative layouts) without
    # re-running the model.
    np.savez(
        out_dir / "raw_data.npz",
        heatmaps=np.stack(raw_heatmaps, axis=0) if raw_heatmaps else np.empty((0,)),
        sources=np.array([s["source"] for s in summary]),
        true_labels=np.array([s["true"] for s in summary]),
        pred_labels=np.array([s["pred"] for s in summary]),
        probs=np.array([s["prob"] for s in summary]),
    )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Open the .png files in {out_dir}")


if __name__ == "__main__":
    main()
