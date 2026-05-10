"""Generate an infographic showing Mixup, CutMix and Manifold Mixup on real chest X-rays."""
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA_ROOT = Path(r"c:\temp\pneumonia\data\chest_xray\train")
OUT_PATH = Path(r"c:\temp\pneumonia\mixup_cutmix_demo.png")
IMG_SIZE = 256

random.seed(42)
np.random.seed(42)

normal_imgs = list((DATA_ROOT / "NORMAL").glob("*.jpeg"))
pneumonia_imgs = list((DATA_ROOT / "PNEUMONIA").glob("*.jpeg"))

# Pick one NORMAL and one PNEUMONIA scan with reasonable visible difference
img_a_path = random.choice(normal_imgs)
img_b_path = random.choice(pneumonia_imgs)

def load(p):
    img = Image.open(p).convert("L")  # grayscale, X-rays are 1-channel
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0

A = load(img_a_path)
B = load(img_b_path)

# Mixup: pixel-wise blend
LAM_MIX = 0.6
mix = LAM_MIX * A + (1 - LAM_MIX) * B
label_mix = f"{LAM_MIX:.1f}*NORMAL + {1-LAM_MIX:.1f}*PNEUMONIA"

# CutMix: paste rectangle from B into A
LAM_CUT = 0.65  # area kept of A
cut_w = int(IMG_SIZE * np.sqrt(1 - LAM_CUT))
cut_h = int(IMG_SIZE * np.sqrt(1 - LAM_CUT))
cx = np.random.randint(IMG_SIZE - cut_w)
cy = np.random.randint(IMG_SIZE - cut_h)
cutmix = A.copy()
cutmix[cy:cy+cut_h, cx:cx+cut_w] = B[cy:cy+cut_h, cx:cx+cut_w]
area_b = (cut_w * cut_h) / (IMG_SIZE * IMG_SIZE)
label_cut = f"{1-area_b:.2f}*NORMAL + {area_b:.2f}*PNEUMONIA"

# Manifold Mixup: blend in feature space, NOT pixel space.
# True implementation: forward A and B through k layers, mix activations, then
# continue forward. Here we proxy "feature-space" with a 14x14 downsample
# (the spatial size of the last conv block in a 224-input ResNet50), blend
# at that resolution, then upsample for display. Conveys the key idea that
# mixing happens at a coarse, abstract grid — fine pixel-level features are
# already gone by the time we mix.
FEAT_RES = 14
LAM_MAN = 0.6


def to_feat(img):
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil = pil.resize((FEAT_RES, FEAT_RES), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def upsample(feat):
    pil = Image.fromarray((feat * 255).astype(np.uint8))
    pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


A_feat = to_feat(A)
B_feat = to_feat(B)
manifold_feat = LAM_MAN * A_feat + (1 - LAM_MAN) * B_feat
manifold_vis = upsample(manifold_feat)
label_man = f"{LAM_MAN:.1f}*NORMAL + {1-LAM_MAN:.1f}*PNEUMONIA"

# Plot
fig, axes = plt.subplots(3, 3, figsize=(13, 13))
fig.suptitle("Mixup vs. CutMix vs. Manifold Mixup on Chest X-rays",
             fontsize=18, fontweight="bold", y=0.99)

# ── Row 1: Mixup ───────────────────────────────────────────────────────────
axes[0, 0].imshow(A, cmap="gray", vmin=0, vmax=1)
axes[0, 0].set_title("Image A — NORMAL\nlabel = 0", fontsize=12)
axes[0, 0].axis("off")

axes[0, 1].imshow(B, cmap="gray", vmin=0, vmax=1)
axes[0, 1].set_title("Image B — PNEUMONIA\nlabel = 1", fontsize=12)
axes[0, 1].axis("off")

axes[0, 2].imshow(mix, cmap="gray", vmin=0, vmax=1)
axes[0, 2].set_title(f"MIXUP (λ={LAM_MIX})\nλ·A + (1-λ)·B\nlabel = {1-LAM_MIX:.1f}",
                     fontsize=12, color="darkred")
axes[0, 2].axis("off")

# ── Row 2: CutMix ──────────────────────────────────────────────────────────
axes[1, 0].imshow(A, cmap="gray", vmin=0, vmax=1)
rect = plt.Rectangle((cx, cy), cut_w, cut_h, linewidth=2,
                     edgecolor="red", facecolor="none", linestyle="--")
axes[1, 0].add_patch(rect)
axes[1, 0].set_title("Image A — NORMAL\n(red box = paste target)", fontsize=12)
axes[1, 0].axis("off")

axes[1, 1].imshow(B, cmap="gray", vmin=0, vmax=1)
rect2 = plt.Rectangle((cx, cy), cut_w, cut_h, linewidth=2,
                      edgecolor="red", facecolor="none", linestyle="--")
axes[1, 1].add_patch(rect2)
axes[1, 1].set_title("Image B — PNEUMONIA\n(red box = source)", fontsize=12)
axes[1, 1].axis("off")

axes[1, 2].imshow(cutmix, cmap="gray", vmin=0, vmax=1)
rect3 = plt.Rectangle((cx, cy), cut_w, cut_h, linewidth=2,
                      edgecolor="red", facecolor="none")
axes[1, 2].add_patch(rect3)
axes[1, 2].set_title(f"CUTMIX (area_B={area_b:.2f})\nlabel = {area_b:.2f}",
                     fontsize=12, color="darkred")
axes[1, 2].axis("off")

# ── Row 3: Manifold Mixup ──────────────────────────────────────────────────
axes[2, 0].imshow(upsample(A_feat), cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
axes[2, 0].set_title(f"f_A — features of A\nat hidden layer ({FEAT_RES}×{FEAT_RES} grid)",
                     fontsize=12)
axes[2, 0].axis("off")

axes[2, 1].imshow(upsample(B_feat), cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
axes[2, 1].set_title(f"f_B — features of B\nat hidden layer ({FEAT_RES}×{FEAT_RES} grid)",
                     fontsize=12)
axes[2, 1].axis("off")

axes[2, 2].imshow(manifold_vis, cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
axes[2, 2].set_title(f"MANIFOLD MIXUP (λ={LAM_MAN})\nλ·f_A + (1-λ)·f_B\n"
                     f"then forward through rest of net\nlabel = {1-LAM_MAN:.1f}",
                     fontsize=11, color="darkred")
axes[2, 2].axis("off")

# Side annotations
fig.text(0.02, 0.82, "MIXUP",
         fontsize=14, fontweight="bold", rotation=90, va="center", color="darkred")
fig.text(0.02, 0.50, "CUTMIX",
         fontsize=14, fontweight="bold", rotation=90, va="center", color="darkred")
fig.text(0.02, 0.18, "MANIFOLD\nMIXUP",
         fontsize=12, fontweight="bold", rotation=90, va="center", color="darkred")

# Footer with key insight
fig.text(0.5, 0.015,
         "All three produce 'unrealistic' samples — yet all reliably improve generalization.\n"
         "Manifold Mixup mixes hidden activations, not pixels (shown here as a 14×14 proxy);\n"
         "the label becomes a soft target between 0 and 1.",
         ha="center", fontsize=11, style="italic", color="gray")

plt.tight_layout(rect=[0.03, 0.05, 1, 0.96])
plt.savefig(OUT_PATH, dpi=110, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PATH}")
print(f"  Image A: {img_a_path.name}")
print(f"  Image B: {img_b_path.name}")
