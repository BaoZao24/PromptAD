#!/usr/bin/env python3
"""Draw a publication-quality vector schematic of the current fusion module.

The figure is deliberately rendered with Matplotlib rather than generated as a
bitmap so that labels, formulas, arrows, and typography remain editable and
scalable in a paper workflow.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/01_figures/paper_arch_tta"

INK = "#1F2937"
MUTED = "#64748B"
PANEL = "#F8FAFC"
PANEL_EDGE = "#CBD5E1"
BLUE = "#2563A8"
BLUE_FILL = "#EEF5FF"
ORANGE = "#C86624"
ORANGE_FILL = "#FFF3EA"
GREEN = "#3C7D3E"
GREEN_FILL = "#F0F8EE"
PURPLE = "#6941A5"
PURPLE_FILL = "#F5F0FF"
GRAY_FILL = "#F8FAFC"


def rounded_box(ax, x, y, w, h, text, edge, fill, *, fontsize=11, lw=1.5,
                text_color=INK, weight="normal", radius=0.12, zorder=4):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.15,
        zorder=zorder + 1,
    )
    return patch


def panel(ax, x, y, w, h, title, letter):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.0,
            edgecolor=PANEL_EDGE,
            facecolor=PANEL,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (x, y + h - 0.62), w, 0.62,
            linewidth=0,
            facecolor="#EEF2F6",
            zorder=1,
        )
    )
    ax.text(
        x + 0.22, y + h - 0.31,
        f"{letter}. {title}",
        ha="left", va="center",
        fontsize=13, fontweight="bold", color=INK, zorder=2,
    )


def arrow(ax, points, color, *, lw=1.7, ls="-", mutation=13, zorder=2):
    vertices = [points[0], *points[1:]]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 1)
    path = MplPath(vertices, codes)
    patch = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        linestyle=ls,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, color=MUTED, fontsize=8.5, *, italic=False, ha="left"):
    ax.text(
        x, y, text,
        ha=ha, va="center",
        fontsize=fontsize,
        color=color,
        fontstyle="italic" if italic else "normal",
        zorder=5,
    )


def draw():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis("off")

    panel(ax, 0.25, 0.65, 4.15, 8.45, "Support construction", "A")
    panel(ax, 4.6, 0.65, 7.1, 8.45, "Query inference", "B")
    panel(ax, 11.95, 0.65, 7.75, 8.45, "Support-only confidence fusion", "C")

    ax.text(
        0.35, 9.52,
        "Training-free few-shot spectrogram anomaly detection",
        fontsize=17, fontweight="bold", color=INK, ha="left", va="center",
    )

    # A. Normal support and the two memory banks.
    rounded_box(ax, 0.72, 7.40, 1.55, 0.62, "Normal support", GREEN, GREEN_FILL,
                fontsize=11.5, weight="bold")
    rounded_box(ax, 2.48, 7.02, 1.58, 1.15, "Spectral TTA\n(support only)", GREEN,
                GREEN_FILL, fontsize=10.5, weight="bold", lw=1.35)
    tta_items = [("orig", 2.61), ("t−", 2.96), ("t+", 3.31), ("Δf", 3.66)]
    for text, x in tta_items:
        rounded_box(ax, x, 7.13, 0.28, 0.24, text, GREEN, "white", fontsize=7.1,
                    lw=0.8, radius=0.04)
    rounded_box(ax, 0.72, 5.32, 1.58, 0.72, "ViT normal\nmemory", BLUE, BLUE_FILL,
                fontsize=10.5, weight="bold")
    rounded_box(ax, 2.48, 5.32, 1.58, 0.72, "CNN normal\nmemory", ORANGE, ORANGE_FILL,
                fontsize=10.5, weight="bold")
    arrow(ax, [(2.27, 7.71), (2.48, 7.71)], GREEN)
    arrow(ax, [(3.27, 7.02), (3.27, 6.72), (1.51, 6.72), (1.51, 6.04)], GREEN)
    arrow(ax, [(1.49, 7.40), (1.49, 6.42), (3.27, 6.42), (3.27, 6.04)], ORANGE)
    label(ax, 2.48, 4.72, "TTA expands ViT memory only", GREEN, 8.2, italic=True, ha="center")

    # B. Query and branch feature extraction.
    rounded_box(ax, 4.98, 6.78, 1.25, 0.82, "Query\nspectrogram\n(original)", INK, "white",
                fontsize=10.2, weight="bold", lw=1.25)
    ax.text(6.78, 8.24, "Global structure branch", ha="left", va="center",
            fontsize=9.0, color=BLUE, fontweight="bold")
    ax.text(6.78, 5.87, "Local detail branch", ha="left", va="center",
            fontsize=9.0, color=ORANGE, fontweight="bold")

    rounded_box(ax, 6.78, 6.84, 1.35, 0.68, "Frozen ViT", BLUE, BLUE_FILL,
                fontsize=10.4, weight="bold")
    rounded_box(ax, 8.36, 6.84, 1.42, 0.68, "Layer 1 +\nLayer 2", BLUE, BLUE_FILL,
                fontsize=10.1, weight="bold")
    rounded_box(ax, 10.02, 6.84, 1.20, 0.68, "kNN(M_V, q_V)\nscore V", BLUE, BLUE_FILL,
                fontsize=10.0, weight="bold")

    rounded_box(ax, 6.78, 4.38, 1.35, 0.68, "Frozen\nResNet18", ORANGE, ORANGE_FILL,
                fontsize=9.9, weight="bold")
    rounded_box(ax, 8.36, 4.38, 1.42, 0.68, "Layer 3\nlocal features", ORANGE, ORANGE_FILL,
                fontsize=9.7, weight="bold")
    rounded_box(ax, 10.02, 4.38, 1.20, 0.68, "1-NN(M_C, q_C)\ntop-10%", ORANGE, ORANGE_FILL,
                fontsize=9.6, weight="bold")
    arrow(ax, [(6.23, 7.19), (6.52, 7.19), (6.52, 7.18), (6.78, 7.18)], BLUE)
    arrow(ax, [(6.23, 7.00), (6.50, 7.00), (6.50, 4.72), (6.78, 4.72)], ORANGE)
    arrow(ax, [(8.13, 7.18), (8.36, 7.18)], BLUE)
    arrow(ax, [(9.78, 7.18), (10.02, 7.18)], BLUE)
    arrow(ax, [(8.13, 4.72), (8.36, 4.72)], ORANGE)
    arrow(ax, [(9.78, 4.72), (10.02, 4.72)], ORANGE)

    label(ax, 2.48, 5.10, r"$M_V$, $M_C$: normal memory banks", MUTED, 7.5,
          italic=True, ha="center")

    # C. Score fusion and final output.
    rounded_box(ax, 12.35, 6.84, 1.05, 0.68, "V", BLUE, BLUE_FILL,
                fontsize=13, weight="bold")
    rounded_box(ax, 12.35, 4.38, 1.05, 0.68, "C", ORANGE, ORANGE_FILL,
                fontsize=13, weight="bold")
    rounded_box(ax, 12.35, 1.68, 2.22, 0.78, "Normal-support statistics\nmedian / IQR / rank",
                GREEN, GREEN_FILL, fontsize=9.4, weight="bold")
    ax.add_patch(
        FancyBboxPatch(
            (15.02, 4.58), 2.58, 1.78,
            boxstyle="round,pad=0.025,rounding_size=0.12",
            linewidth=1.7,
            edgecolor=GREEN,
            facecolor=GREEN_FILL,
            zorder=4,
        )
    )
    ax.text(16.31, 6.02, "Confidence fusion", ha="center", va="center",
            fontsize=12.2, color=GREEN, fontweight="bold", zorder=6)
    ax.text(16.31, 5.57, "S = V + α IQR(Rᵥ)",
            ha="center", va="center", fontsize=10.1, color=INK, zorder=6)
    ax.text(16.31, 5.27, "[pC − pV]+ 1{rC ≥ 0.8}",
            ha="center", va="center", fontsize=10.1, color=INK, zorder=6)
    ax.text(16.31, 4.87, "CNN contributes only reliable positive evidence",
            ha="center", va="center", fontsize=8.6, color=GREEN, fontstyle="italic", zorder=6)
    rounded_box(ax, 18.02, 5.00, 1.15, 0.96, "Final\nanomaly\nscore S", PURPLE, PURPLE_FILL,
                fontsize=10.8, weight="bold", lw=1.7, text_color=PURPLE)
    arrow(ax, [(11.22, 7.18), (11.78, 7.18), (11.78, 7.18), (12.35, 7.18)], BLUE)
    arrow(ax, [(11.22, 4.72), (11.78, 4.72), (11.78, 4.72), (12.35, 4.72)], ORANGE)
    arrow(ax, [(13.40, 7.18), (14.18, 7.18), (14.18, 6.10), (15.02, 6.10)], BLUE)
    arrow(ax, [(13.40, 4.72), (15.02, 4.72)], ORANGE)
    arrow(ax, [(14.57, 2.07), (14.80, 2.07), (14.80, 4.58)], GREEN)
    arrow(ax, [(17.60, 5.47), (18.02, 5.47)], PURPLE)

    ax.text(10.0, 0.30, "No target-domain training  •  no anomaly labels  •  query remains unaugmented",
            ha="center", va="center", fontsize=10.0, color=MUTED, fontstyle="italic")

    fig.savefig(OUT / "fusion_architecture_top_journal.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(OUT / "fusion_architecture_top_journal.svg", bbox_inches="tight",
                facecolor="white")
    fig.savefig(OUT / "fusion_architecture_top_journal.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
