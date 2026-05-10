"""Generate methodology_flow.png — a single-figure 5-stage pipeline overview.

Stages, top to bottom:
  1. Source data (train/val/test counts)
  2. Data preparation (patient-isolation + val redistribution)
  3. Methodology comparison (4 tracks, heterogeneous ablation depth)
  4. External-dataset generalization (future work, dashed border)
  5. Synthesis (KPIs + conclusion + future work)
"""
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_PATH = Path(__file__).resolve().parent / "methodology_flow.png"

fig, ax = plt.subplots(figsize=(13, 16))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.suptitle(
    "Methodology pipeline — pneumonia X-ray classification",
    fontsize=15, fontweight="bold", y=0.985,
)


def stage_box(y_top, h, color, edge, title, dashed=False):
    """Outer container box for a stage, with title at top-left."""
    style = "round,pad=0.6,rounding_size=0.8"
    ls = "--" if dashed else "-"
    p = FancyBboxPatch((3, y_top - h), 94, h, boxstyle=style,
                       linewidth=1.6, edgecolor=edge, facecolor=color,
                       linestyle=ls)
    ax.add_patch(p)
    ax.text(5, y_top - 2.5, title, fontsize=12, fontweight="bold", color=edge,
            va="top")


def inner_box(x, y, w, h, text, color="white", edge="#444", fontsize=9,
              dashed=False, italic=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.5",
                       linewidth=1.0, edgecolor=edge, facecolor=color,
                       linestyle="--" if dashed else "-")
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, style="italic" if italic else "normal",
            wrap=True)


def stage_arrow(y_from, y_to, dashed=False):
    a = FancyArrowPatch((50, y_from), (50, y_to), arrowstyle="-|>",
                        mutation_scale=18, linewidth=1.4, color="#666",
                        linestyle="--" if dashed else "-")
    ax.add_patch(a)


# Colour palette
DATA_FILL, DATA_EDGE = "#FFF4E5", "#F29900"
PREP_FILL, PREP_EDGE = "#F1F8E9", "#33691E"
METH_FILL, METH_EDGE = "#E8F0FE", "#1A73E8"
GEN_FILL,  GEN_EDGE  = "#F3E8FD", "#7B1FA2"
SYN_FILL,  SYN_EDGE  = "#FCE8E6", "#D93025"


# ── STAGE 1: Source data ───────────────────────────────────────────────────
stage_box(95, 13, DATA_FILL, DATA_EDGE,
          "Stage 1 — Source data: Kaggle Chest X-Ray Pneumonia (Kermany 2018)")
inner_box(8, 83, 26, 6,
          "Train\n5,216 images",
          color="white", edge=DATA_EDGE)
inner_box(37, 83, 26, 6,
          "Val\n16 images (too small for tuning)",
          color="white", edge=DATA_EDGE)
inner_box(66, 83, 26, 6,
          "Test\n624 images (held out)",
          color="white", edge=DATA_EDGE)
stage_arrow(82, 79)

# ── STAGE 2: Data preparation ──────────────────────────────────────────────
stage_box(79, 13, PREP_FILL, PREP_EDGE,
          "Stage 2 — Data preparation")
inner_box(8, 67, 41, 7,
          "Patient-isolation verification\n(filename namespace check)",
          color="white", edge=PREP_EDGE)
inner_box(51, 67, 41, 7,
          "Val redistribution: merge train+val → 5,232 pool\n→ 5-fold StratifiedKFold",
          color="white", edge=PREP_EDGE)
stage_arrow(66, 63)

# ── STAGE 3: Methodology comparison (heterogeneous ablation depth) ─────────
stage_box(63, 18, METH_FILL, METH_EDGE,
          "Stage 3 — Methodology comparison on this dataset (heterogeneous ablation depth)")
# Four parallel approach boxes
inner_box(5, 47.5, 22, 11,
          "Custom CNN\n(from scratch)\n\n DEEP ablation:\nA1 depth\nA2 stride/pad/act\nA3 regularisation",
          color="white", edge=METH_EDGE, fontsize=8)
inner_box(28.5, 47.5, 22, 11,
          "ResNet50\n+ ImageNet\n\n shallow tuning:\nLR + batch only\n(architecture\nfixed)",
          color="white", edge=METH_EDGE, fontsize=8)
inner_box(52, 47.5, 22, 11,
          "BiomedCLIP\nlinear probe\n\n no architecture\nablation:\nfrozen biomedical\nfeatures + LogReg",
          color="white", edge=METH_EDGE, fontsize=8)
inner_box(75.5, 47.5, 22, 11,
          "RAD-DINO\nlinear probe\n\n no architecture\nablation:\nfrozen chest-X-ray\nfeatures + LogReg",
          color="white", edge=METH_EDGE, fontsize=8)
stage_arrow(45, 41)

# ── STAGE 4: External-dataset generalization (future work, dashed) ─────────
stage_box(41, 17, GEN_FILL, GEN_EDGE,
          "Stage 4 — Generalization to external datasets (future work)",
          dashed=True)
inner_box(5, 30, 28, 6,
          "NIH ChestX-ray14",
          color="white", edge=GEN_EDGE, dashed=True)
inner_box(36, 30, 28, 6,
          "RSNA Pneumonia",
          color="white", edge=GEN_EDGE, dashed=True)
inner_box(67, 30, 28, 6,
          "CheXpert",
          color="white", edge=GEN_EDGE, dashed=True)
inner_box(8, 25, 84, 4,
          "Caveat: BiomedCLIP / RAD-DINO have leakage risk on these datasets — they were "
          "pretrained on (subsets of) the same data.",
          color="#FFFBE5", edge=GEN_EDGE, fontsize=8, italic=True)
stage_arrow(24, 20, dashed=True)

# ── STAGE 5: Synthesis ─────────────────────────────────────────────────────
stage_box(20, 17, SYN_FILL, SYN_EDGE,
          "Stage 5 — Synthesis")
inner_box(5, 9, 28, 7,
          "Headline KPI table\n(Sens / Spec / AUROC / ECE)\nper approach × dataset",
          color="white", edge=SYN_EDGE, fontsize=8)
inner_box(36, 9, 28, 7,
          "Conclusion\n(which methodology generalizes\nbest, with caveats)",
          color="white", edge=SYN_EDGE, fontsize=8)
inner_box(67, 9, 28, 7,
          "Future work\n(bullet list)",
          color="white", edge=SYN_EDGE, fontsize=8)

# ── Footer note ────────────────────────────────────────────────────────────
ax.text(50, 1.5,
        "Discipline: test set held out from every tuning decision until the final eval. "
        "Stage 4 is dashed (= prospective).",
        ha="center", va="center", fontsize=9, style="italic", color="#444")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_PATH, dpi=120, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PATH}")
