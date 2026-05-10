"""Generate methodology_flow.png — a single-figure overview of the full
end-to-end pipeline.

Three vertical lanes:
  1. Data flow (Kaggle → pool → 5-fold + held-out test)
  2. Experimental flow (A1 depth → A2 stride/pad/act → A3 overfitting → champion)
  3. Evaluation flow (champion → 5-fold ensemble → 4 KPIs at 3 thresholds)
"""
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_PATH = Path(__file__).resolve().parent / "methodology_flow.png"

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.suptitle(
    "Methodology pipeline — pneumonia X-ray classification",
    fontsize=15, fontweight="bold", y=0.97,
)


def box(x, y, w, h, text, color="#E8F0FE", edge="#1A73E8", fontsize=10,
        bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=0.6",
                       linewidth=1.2, edgecolor=edge, facecolor=color)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal",
            wrap=True)


def arrow(x1, y1, x2, y2, color="#1A73E8", style="-|>", lw=1.4):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=14, linewidth=lw, color=color)
    ax.add_patch(a)


# ---- Lane 1: Data flow (left, vertical) -----------------------------------
LANE1_X = 5
DATA_COLOR = "#FFF4E5"
DATA_EDGE = "#F29900"

box(LANE1_X, 80, 24, 8,
    "Kaggle Chest X-Ray\n5,856 images, 2 classes",
    color=DATA_COLOR, edge=DATA_EDGE, bold=True)

box(LANE1_X, 67, 24, 8,
    "Patient-isolation check\n(verify_patient_isolation.py)",
    color="#F1F8E9", edge="#33691E", fontsize=9)

box(LANE1_X, 52, 24, 10,
    "Train+val pool: 5,232\n(merged for stable CV)\n\nTest: 624 (held out)",
    color=DATA_COLOR, edge=DATA_EDGE)

box(LANE1_X, 37, 24, 10,
    "5-fold StratifiedKFold\non the merged pool",
    color=DATA_COLOR, edge=DATA_EDGE)

# Arrows down the data lane
arrow(LANE1_X + 12, 80, LANE1_X + 12, 75, color=DATA_EDGE)
arrow(LANE1_X + 12, 67, LANE1_X + 12, 62, color=DATA_EDGE)
arrow(LANE1_X + 12, 52, LANE1_X + 12, 47, color=DATA_EDGE)

# ---- Lane 2: Experimental flow (centre, vertical) -------------------------
LANE2_X = 38
EXP_COLOR = "#E8F0FE"
EXP_EDGE = "#1A73E8"

box(LANE2_X, 80, 24, 8,
    "Custom CNN\nparametric arch",
    color=EXP_COLOR, edge=EXP_EDGE, bold=True)

box(LANE2_X, 67, 24, 8,
    "A1 — Depth\n2/3/4/5 blocks + Glorot ctrl",
    color=EXP_COLOR, edge=EXP_EDGE)

box(LANE2_X, 54, 24, 8,
    "A2 — Stride/pad/act\n6 representative variants",
    color=EXP_COLOR, edge=EXP_EDGE)

box(LANE2_X, 41, 24, 8,
    "A3 — Overfitting\nnone/BN/dropout/L2/aug/combo",
    color=EXP_COLOR, edge=EXP_EDGE)

box(LANE2_X, 26, 24, 10,
    "Champion architecture\n(combine A1+A2+A3 winners)\n5-fold CV ensemble",
    color="#FCE8E6", edge="#D93025", bold=True)

# Arrows down the experimental lane
arrow(LANE2_X + 12, 80, LANE2_X + 12, 75, color=EXP_EDGE)
arrow(LANE2_X + 12, 67, LANE2_X + 12, 62, color=EXP_EDGE)
arrow(LANE2_X + 12, 54, LANE2_X + 12, 49, color=EXP_EDGE)
arrow(LANE2_X + 12, 41, LANE2_X + 12, 36, color=EXP_EDGE)

# Cross-arrows: data → each experimental rung uses CV pool
for y in (71, 58, 45, 31):
    arrow(LANE1_X + 24, y, LANE2_X, y, color="#888", style="->", lw=0.8)

# ---- Lane 3: Evaluation flow (right, vertical) ---------------------------
LANE3_X = 71
EVAL_COLOR = "#F3E8FD"
EVAL_EDGE = "#7B1FA2"

box(LANE3_X, 80, 24, 8,
    "Test set\n(touched once, at end)",
    color=EVAL_COLOR, edge=EVAL_EDGE, bold=True)

box(LANE3_X, 67, 24, 8,
    "5-fold ensemble\n(mean of probabilities)",
    color=EVAL_COLOR, edge=EVAL_EDGE)

box(LANE3_X, 54, 24, 8,
    "Threshold tuning on val\n(default / best-acc / sens-target)",
    color=EVAL_COLOR, edge=EVAL_EDGE)

box(LANE3_X, 38, 24, 12,
    "Medical KPIs\n• Sensitivity (FN cost)\n• Specificity (FP cost)\n• AUROC\n• ECE (calibration)",
    color=EVAL_COLOR, edge=EVAL_EDGE)

box(LANE3_X, 22, 24, 8,
    "Grad-CAM + curves\n(qualitative + diagnostic)",
    color=EVAL_COLOR, edge=EVAL_EDGE)

# Arrows down the evaluation lane
arrow(LANE3_X + 12, 80, LANE3_X + 12, 75, color=EVAL_EDGE)
arrow(LANE3_X + 12, 67, LANE3_X + 12, 62, color=EVAL_EDGE)
arrow(LANE3_X + 12, 54, LANE3_X + 12, 50, color=EVAL_EDGE)
arrow(LANE3_X + 12, 38, LANE3_X + 12, 30, color=EVAL_EDGE)

# Cross-arrow: champion → evaluation
arrow(LANE2_X + 24, 31, LANE3_X, 71, color="#D93025", lw=2.0)

# Cross-arrow: data → test (top of eval lane)
arrow(LANE1_X + 24, 56, LANE3_X, 84, color="#888", style="->", lw=0.8)

# ---- Lane labels --------------------------------------------------------
ax.text(LANE1_X + 12, 95, "DATA", fontsize=11, fontweight="bold",
        ha="center", color=DATA_EDGE)
ax.text(LANE2_X + 12, 95, "EXPERIMENT", fontsize=11, fontweight="bold",
        ha="center", color=EXP_EDGE)
ax.text(LANE3_X + 12, 95, "EVALUATION", fontsize=11, fontweight="bold",
        ha="center", color=EVAL_EDGE)

# Footer note: discipline highlighted
ax.text(50, 8,
        "Discipline: test set is held out from every ablation choice. "
        "Threshold and calibration tuned on val only.\n"
        "Patient-level isolation between train+val and test verified by "
        "filename namespace analysis (Kermany et al. 2018).",
        ha="center", va="center", fontsize=9, style="italic", color="#444",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAFAFA", edgecolor="#CCC"))

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(OUT_PATH, dpi=120, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PATH}")
