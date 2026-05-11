"""Generate visualization mockups comparing 4 approaches × 4 KPIs,
with semantic colour zones: red = below clinical minimum (unusable),
green = clinical sweet zone, grey = above ceiling (suspect)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

APPROACHES = ['Custom CNN', 'ResNet50+IN', 'BiomedCLIP', 'RAD-DINO', 'cnn-kamal']
METRICS = ['Sensitivity', 'Specificity', 'AUROC', '1 − ECE']
THR_MIN = {'Sensitivity': 0.95, 'Specificity': 0.90, 'AUROC': 0.95, '1 − ECE': 0.90}
THR_MAX = {'Sensitivity': 0.97, 'Specificity': 0.95, 'AUROC': 0.98, '1 − ECE': 0.95}

VALUES = {
    'Custom CNN':  [0.92, 0.85, 0.94, 0.85],
    'ResNet50+IN': [0.97, 0.88, 0.98, 0.84],
    'BiomedCLIP':  [0.94, 0.91, 0.96, 0.88],
    'RAD-DINO':    [0.96, 0.93, 0.97, 0.90],
    # Peer reference (anonymous): high sens via threshold tuning at the cost
    # of specificity, AUROC, and calibration. Included as an instructive
    # example of an "unusable" classifier despite impressive sensitivity.
    'cnn-kamal':   [0.997, 0.252, 0.924, 0.75],
}

# Semantic zone colours
COLOR_BAD   = '#D32F2F'   # red = clinically unusable (below min)
COLOR_GOOD  = '#2E7D32'   # green = clinical sweet zone (min to ceiling)
COLOR_SUS   = '#757575'   # grey = suspect (above ceiling)

BG_BAD  = '#FFEBEE'
BG_GOOD = '#C8E6C9'
BG_SUS  = '#E0E0E0'


def zone_of(metric, value):
    if value < THR_MIN[metric]:
        return 'bad'
    elif value > THR_MAX[metric]:
        return 'sus'
    else:
        return 'good'


def color_for(zone):
    return {'bad': COLOR_BAD, 'good': COLOR_GOOD, 'sus': COLOR_SUS}[zone]


# ─── Option 3 (revised): semantic bullet charts ─────────────────────────────
def option3():
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(11, 7), sharex=True)
    y_offsets = np.linspace(-0.6, 0.6, len(APPROACHES))
    for ax_idx, (ax, m) in enumerate(zip(axes, METRICS)):
        # Background zones
        ax.axvspan(0.75, THR_MIN[m], facecolor=BG_BAD, alpha=0.55,
                   label='Clinically unusable' if ax_idx == 0 else None)
        ax.axvspan(THR_MIN[m], THR_MAX[m], facecolor=BG_GOOD, alpha=0.55,
                   label='Clinical sweet zone' if ax_idx == 0 else None)
        ax.axvspan(THR_MAX[m], 1.0, facecolor=BG_SUS, alpha=0.5,
                   label='Suspect (above ceiling)' if ax_idx == 0 else None)
        # Threshold lines
        ax.axvline(THR_MIN[m], color=COLOR_BAD, linestyle='--', linewidth=1.2)
        ax.axvline(THR_MAX[m], color=COLOR_SUS, linestyle='--', linewidth=1.2)
        # Dots — coloured by zone, labelled with approach name
        for i, app in enumerate(APPROACHES):
            v = VALUES[app][ax_idx]
            z = zone_of(m, v)
            c = color_for(z)
            y = y_offsets[i]
            ax.scatter(v, y, s=220, color=c, zorder=3,
                       edgecolor='black', linewidth=0.8)
            ax.text(v + 0.003, y, f' {app}: {v:.3f}',
                    va='center', fontsize=8.5, fontweight='bold', color=c)
        # Threshold labels at bottom
        ax.text(THR_MIN[m], -1.0, f'min {THR_MIN[m]}', ha='center',
                fontsize=7, color=COLOR_BAD)
        ax.text(THR_MAX[m], -1.0, f'ceil {THR_MAX[m]}', ha='center',
                fontsize=7, color=COLOR_SUS)
        ax.set_ylabel(m, fontsize=10, rotation=0, ha='right', va='center')
        ax.set_yticks([]); ax.set_ylim(-1.2, 1.0)
        ax.set_xlim(0.20, 1.0)
    axes[-1].set_xlabel('Score')
    axes[0].set_title('Option 3 — Stacked bullet charts with semantic zone colouring\n'
                      '(dot colour = zone status, label = approach)',
                      fontsize=12, pad=10)
    # Single legend at the bottom of the whole figure — no overlap with title.
    fig.legend(loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    plt.savefig(OUT_DIR / 'viz_opt3_bullets.png', dpi=110, facecolor='white',
                bbox_inches='tight')
    plt.close()


# ─── Option 5 (revised): semantic sens × spec scatter ───────────────────────
def option5():
    # Skip cnn-kamal here — its spec=0.25 falls off the 0.78–1.0 axis we use
    # for the sweet-spot scatter. It's already shown in the bullet chart.
    scatter_approaches = [a for a in APPROACHES if a != 'cnn-kamal']
    fig, ax = plt.subplots(figsize=(8.5, 8))
    # Background quadrant zones
    # Below either minimum = red
    ax.axvspan(0.78, THR_MIN['Specificity'], facecolor=BG_BAD, alpha=0.45)
    ax.axhspan(0.85, THR_MIN['Sensitivity'], facecolor=BG_BAD, alpha=0.45,
               xmin=(THR_MIN['Specificity'] - 0.78) / (1.0 - 0.78),
               xmax=1.0)
    # Sweet zone (both in their [min, ceiling] range)
    ax.fill_between([THR_MIN['Specificity'], THR_MAX['Specificity']],
                    THR_MIN['Sensitivity'], THR_MAX['Sensitivity'],
                    facecolor=BG_GOOD, alpha=0.65)
    # Above ceiling on either = grey
    ax.axvspan(THR_MAX['Specificity'], 1.0, facecolor=BG_SUS, alpha=0.45,
               ymin=(THR_MIN['Sensitivity'] - 0.85) / (1.0 - 0.85),
               ymax=1.0)
    ax.axhspan(THR_MAX['Sensitivity'], 1.0, facecolor=BG_SUS, alpha=0.45,
               xmin=(THR_MIN['Specificity'] - 0.78) / (1.0 - 0.78),
               xmax=(THR_MAX['Specificity'] - 0.78) / (1.0 - 0.78))
    # Threshold lines
    ax.axvline(THR_MIN['Specificity'], color=COLOR_BAD, linestyle='--', linewidth=1.2)
    ax.axhline(THR_MIN['Sensitivity'], color=COLOR_BAD, linestyle='--', linewidth=1.2)
    ax.axvline(THR_MAX['Specificity'], color=COLOR_SUS, linestyle='--', linewidth=1.2)
    ax.axhline(THR_MAX['Sensitivity'], color=COLOR_SUS, linestyle='--', linewidth=1.2)
    # Approach dots — colour = worst-zone-of-the-two
    for app in scatter_approaches:
        sens = VALUES[app][0]; spec = VALUES[app][1]
        zone_s = zone_of('Sensitivity', sens)
        zone_p = zone_of('Specificity', spec)
        if 'bad' in (zone_s, zone_p):
            zone = 'bad'
        elif 'sus' in (zone_s, zone_p):
            zone = 'sus'
        else:
            zone = 'good'
        c = color_for(zone)
        ax.scatter(spec, sens, s=380, color=c, edgecolor='black',
                   linewidth=1.5, zorder=3)
        ax.annotate(f'{app}\n({sens:.2f}, {spec:.2f})',
                    (spec, sens), textcoords='offset points',
                    xytext=(12, 10), fontsize=9, fontweight='bold', color=c)
    # Legend
    ax.plot([], [], 'o', color=COLOR_BAD, markersize=12,
            markeredgecolor='black', label='Clinically unusable')
    ax.plot([], [], 'o', color=COLOR_GOOD, markersize=12,
            markeredgecolor='black', label='Clinical sweet zone')
    ax.plot([], [], 'o', color=COLOR_SUS, markersize=12,
            markeredgecolor='black', label='Suspect (above ceiling)')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlabel('Specificity'); ax.set_ylabel('Sensitivity')
    ax.set_xlim(0.78, 1.0); ax.set_ylim(0.85, 1.0)
    ax.set_title('Option 5 — Sens × Spec scatter with semantic zone colouring\n'
                 '(dot colour = worst of the two metric zones)')
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'viz_opt5_scatter.png', dpi=110, facecolor='white')
    plt.close()


option3()
option5()
print("Done")
