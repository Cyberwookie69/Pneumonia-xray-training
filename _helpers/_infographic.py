"""Generate a project-summary infographic in the same dark-panel style."""
from pathlib import Path as PPath

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path

OUT = PPath(r"c:\temp\pneumonia\project_summary.png")

# Palette
BG_OUTER = "#0a1626"
BG_PANEL = "#10243a"
BG_BOX = "#142e4a"
BORDER = "#1f3d62"
ACCENT = "#3ba9ee"
ACCENT_DIM = "#5b8bb8"
PNE = "#f48975"  # pneumonia color
NORM = "#3ba9ee"  # normal color
TEXT = "#ffffff"
TEXT_DIM = "#a4b4c8"
RED_X = "#e85a5a"

fig = plt.figure(figsize=(11, 14), facecolor=BG_OUTER)


def panel(left, bottom, width, height, label=None):
    """Draw a rounded dark panel; return its axes for content placement."""
    ax = fig.add_axes((left, bottom, width, height))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(BG_PANEL)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    box = FancyBboxPatch((0.005, 0.005), 0.99, 0.99,
                         boxstyle="round,pad=0.01,rounding_size=0.02",
                         linewidth=0, facecolor=BG_PANEL, transform=ax.transAxes)
    ax.add_patch(box)
    if label:
        ax.text(0.025, 0.93, label, color=TEXT_DIM, fontsize=11, ha="left", va="top",
                transform=ax.transAxes)
    return ax


def inner_box(ax, x, y, w, h, color=BG_BOX, edge=ACCENT):
    """Cyan-bordered inner box for the method strip."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.005,rounding_size=0.025",
                         linewidth=1.5, edgecolor=edge, facecolor=color,
                         transform=ax.transAxes)
    ax.add_patch(box)


def draw_lungs_in_axes(ax, color=PNE):
    """Draw a stylised pair of lungs filling the given axes (kept square).

    Coordinate system: x ∈ [-1, 1], y ∈ [-1.2, 1.2]. Built from cubic Bezier
    paths — one per lobe — plus a trachea and Y-shaped bronchi. Tuned by eye.
    """
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Trachea (rounded vertical pill)
    ax.add_patch(FancyBboxPatch(
        (-0.08, 0.55), 0.16, 0.55,
        boxstyle="round,pad=0,rounding_size=0.07",
        linewidth=0, facecolor=color))

    # Bronchi: short fat wedges from trachea base to inner-top of each lobe
    for sign in (-1, 1):
        verts = [
            (0.0, 0.60),
            (sign * 0.20, 0.40),
            (sign * 0.40, 0.20),
            (sign * 0.50, 0.10),
            (sign * 0.40, 0.30),
            (sign * 0.18, 0.50),
            (0.0, 0.60),
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, linewidth=0))

    # Lung lobes (mirrored)
    for sign in (-1, 1):
        verts = [
            # Start: top inner (just outside the bronchus stem)
            (sign * 0.22, 0.30),
            # Outer top — balloon out and up
            (sign * 0.65, 0.60),
            (sign * 0.95, 0.25),
            (sign * 0.95, -0.20),
            # Outer side down to bottom
            (sign * 0.95, -0.65),
            (sign * 0.75, -1.05),
            (sign * 0.45, -1.10),
            # Bottom curve back to inner
            (sign * 0.20, -1.15),
            (sign * 0.05, -1.00),
            (sign * 0.10, -0.65),
            # Inner edge back up with a gentle cardiac-notch nudge
            (sign * 0.18, -0.40),
            (sign * 0.05, -0.10),
            (sign * 0.22, 0.30),
        ]
        codes = [Path.MOVETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, linewidth=0))


# ============================================================
# HEADER
# ============================================================
ax_h = panel(0.02, 0.86, 0.96, 0.12)
ax_h.text(0.5, 0.65,
          "Pneumonia Detection from Chest\nX-Rays  —  Project Summary",
          color=TEXT, fontsize=22, fontweight="bold", ha="center", va="center",
          transform=ax_h.transAxes)
ax_h.text(0.5, 0.18,
          "Pretrained backbone, k-fold ensemble, focal loss + Mixup + EMA",
          color=ACCENT, fontsize=12, style="italic", ha="center", va="center",
          transform=ax_h.transAxes)


# ============================================================
# DATASET STRIP
# ============================================================
ax_d = panel(0.02, 0.66, 0.96, 0.18, label="Dataset Strip")

# Hand-drawn lungs in a square inset axes (so they don't get stretched
# by the parent panel's wide aspect ratio).
ax_lungs = fig.add_axes((0.05, 0.685, 0.13, 0.135))
ax_lungs.set_facecolor(BG_PANEL)
draw_lungs_in_axes(ax_lungs, color=PNE)
ax_d.text(0.13, 0.10, "5,856 paediatric\nchest radiographs",
          color=TEXT, fontsize=10, ha="center", va="center",
          transform=ax_d.transAxes)

# Two horizontal stacked bars showing class balance
def stacked_bar(ax, x_center, y, width, height, pne_pct, norm_pct, label_below):
    pne_w = width * pne_pct / 100
    norm_w = width * norm_pct / 100
    x_left = x_center - width / 2
    # PNE side
    ax.add_patch(patches.FancyBboxPatch((x_left, y), pne_w, height,
                  boxstyle="round,pad=0,rounding_size=0.012",
                  linewidth=0, facecolor=PNE, transform=ax.transAxes))
    # NORMAL side
    ax.add_patch(patches.FancyBboxPatch((x_left + pne_w, y), norm_w, height,
                  boxstyle="round,pad=0,rounding_size=0.012",
                  linewidth=0, facecolor=NORM, transform=ax.transAxes))
    # Labels inside
    ax.text(x_left + pne_w / 2, y + height / 2 + 0.012,
            f"{pne_pct:.1f}%", color=TEXT, fontsize=10, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(x_left + pne_w / 2, y + height / 2 - 0.025,
            "PNEUMONIA", color=TEXT, fontsize=7, ha="center", va="center",
            transform=ax.transAxes)
    ax.text(x_left + pne_w + norm_w / 2, y + height / 2 + 0.012,
            f"{norm_pct:.1f}%", color=TEXT, fontsize=10, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(x_left + pne_w + norm_w / 2, y + height / 2 - 0.025,
            "NORMAL", color=TEXT, fontsize=7, ha="center", va="center",
            transform=ax.transAxes)
    ax.text(x_center, y - 0.13, label_below,
            color=TEXT, fontsize=10, ha="center", va="center",
            transform=ax.transAxes)


stacked_bar(ax_d, x_center=0.40, y=0.55, width=0.20, height=0.18,
            pne_pct=74.2, norm_pct=25.8, label_below="Train + Val (5,232 imgs)")
stacked_bar(ax_d, x_center=0.78, y=0.55, width=0.20, height=0.18,
            pne_pct=62.5, norm_pct=37.5, label_below="Test (624 imgs)")

# Arrow + prior shift annotation
ax_d.annotate("", xy=(0.66, 0.64), xytext=(0.52, 0.64),
              arrowprops=dict(arrowstyle="->", color=TEXT_DIM, lw=2),
              xycoords=ax_d.transAxes, textcoords=ax_d.transAxes)
ax_d.text(0.59, 0.42, "11.7 pp\nprior shift", color=TEXT_DIM, fontsize=8,
          ha="center", va="center", style="italic", transform=ax_d.transAxes)


# ============================================================
# METHOD STRIP
# ============================================================
ax_m = panel(0.02, 0.38, 0.96, 0.27, label="Method Strip")

method_boxes = [
    {
        "title": "Backbone",
        "lines": ["ResNet50", "ImageNet pretrained",
                  "2-phase train:", "head freeze → full unfreeze"],
        "highlight": "transfer learning",
    },
    {
        "title": "Loss + Sampling",
        "lines": ["Focal loss (γ=2, α=0.25)", "WeightedRandomSampler",
                  "(both fight class", "imbalance independently)"],
        "highlight": "imbalance-aware",
    },
    {
        "title": "Regularisation",
        "lines": ["Mixup (α=0.2, p=0.5)", "EMA decay 0.999",
                  "AdamW + cosine LR", "+ random crop/flip/rot/color"],
        "highlight": "stack of regularisers",
    },
]

box_left = [0.04, 0.36, 0.68]
for i, m in enumerate(method_boxes):
    bx, by, bw, bh = box_left[i], 0.10, 0.28, 0.72
    inner_box(ax_m, bx, by, bw, bh)
    ax_m.text(bx + bw / 2, by + bh - 0.10, m["title"],
              color=TEXT, fontsize=12, fontweight="bold",
              ha="center", va="center", transform=ax_m.transAxes)
    for li, line in enumerate(m["lines"]):
        ax_m.text(bx + bw / 2, by + bh - 0.22 - li * 0.10, line,
                  color=TEXT, fontsize=9.5, ha="center", va="center",
                  transform=ax_m.transAxes)
    ax_m.text(bx + bw / 2, by + 0.06, m["highlight"],
              color=ACCENT, fontsize=10, fontweight="bold",
              ha="center", va="center", transform=ax_m.transAxes)


# ============================================================
# RESULTS STRIP
# ============================================================
ax_r = panel(0.02, 0.10, 0.96, 0.27,
             label="Results Strip  —  single-fold ResNet50 baseline (15 ep, focal loss only)")

# Three big numbers
big_metrics = [
    ("93.11%", "test accuracy", "(default threshold 0.5)"),
    ("0.981", "test ROC-AUC", "(threshold-free)"),
    ("96.92%", "PNEU sensitivity", "(12 missed in 390)"),
]
for i, (val, lbl, sub) in enumerate(big_metrics):
    cx = 0.18 + i * 0.32
    ax_r.text(cx, 0.65, val, color=ACCENT, fontsize=34, fontweight="bold",
              ha="center", va="center", transform=ax_r.transAxes)
    ax_r.text(cx, 0.42, lbl, color=TEXT, fontsize=12,
              ha="center", va="center", transform=ax_r.transAxes)
    ax_r.text(cx, 0.31, sub, color=TEXT_DIM, fontsize=9,
              ha="center", va="center", transform=ax_r.transAxes)

# Secondary line
ax_r.text(0.5, 0.18,
          "86.75% NORMAL specificity   •   Cohen's κ = 0.85 (almost perfect)   •   31 FP, 12 FN",
          color=TEXT, fontsize=10, ha="center", va="center", transform=ax_r.transAxes)


# ============================================================
# FOOTER
# ============================================================
ax_f = panel(0.02, 0.02, 0.96, 0.07)
ax_f.text(0.025, 0.65, "next steps",
          color=TEXT_DIM, fontsize=10, style="italic", ha="left", va="center",
          transform=ax_f.transAxes)
ax_f.text(0.025, 0.30,
          "5-fold ensemble + Mixup + EMA + TTA + threshold tuning  →  expected 96-98% test acc",
          color=TEXT, fontsize=11, ha="left", va="center", transform=ax_f.transAxes)
ax_f.text(0.97, 0.50, "◆", color=ACCENT, fontsize=14, ha="right", va="center",
          transform=ax_f.transAxes)


plt.savefig(OUT, dpi=120, facecolor=BG_OUTER, bbox_inches=None, pad_inches=0)
print(f"Saved: {OUT}")
