#!/usr/bin/env python3
"""
make_figure3_1.py
Generate Figure 3.1: Sample construction and data integration pipeline.

Flowchart from the thesis showing how CRSP, NYSE TAQ,
Refinitiv StreetEvents, and FinBERT/WRDS data sources are integrated
into analysis_dataset_MASTER.parquet.

Output: figures/fig3_1_pipeline.pdf
        figures/fig3_1_pipeline.png

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang, Princeton Senior Thesis
# Advisor: Daniel Rigobon
"""

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

warnings.filterwarnings("ignore")

base    = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
fig_dir = base / "figures"
fig_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

#  Color palette 
DARK_BLUE  = "#2166ac"   # CRSP, 310 firm-quarters, final dataset
MED_BLUE   = "#4393c3"   # Stratified sample
GREEN      = "#4dac26"   # Refinitiv / OpenSMILE
ORANGE     = "#f4a582"   # FinBERT
TEAL       = "#35978f"   # NYSE TAQ
GRAY       = "#666666"   # Arrows, labels

def draw_box(ax, x, y, w, h, text, color, fontsize=9, text_color="white",
             style="round,pad=0.1"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=style,
                          facecolor=color, edgecolor="white",
                          linewidth=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold", zorder=4,
            wrap=True, multialignment="center")

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=GRAY,
                                lw=1.5, mutation_scale=15),
                zorder=2)

#  Row 1: CRSP Universe 
draw_box(ax, 5, 9.0, 3.2, 0.7,
         "CRSP Universe\n1,792 firms (2022–2023)", DARK_BLUE, fontsize=9)

#  Arrow down 
arrow(ax, 5, 8.65, 5, 8.05)

#  Row 2: Stratified Sample 
draw_box(ax, 5, 7.7, 3.8, 0.65,
         "Stratified Sample\n40 firms × 5 volatility quintiles", MED_BLUE, fontsize=9)

#  Arrow down 
arrow(ax, 5, 7.37, 5, 6.77)

#  Row 3: 310 Firm-Quarters 
draw_box(ax, 5, 6.45, 4.0, 0.65,
         "310 Firm-Quarters\n(8 quarters × 40 firms, 3 missing audio)", DARK_BLUE, fontsize=9)

#  Three branches down from 310 firm-quarters 
# Left branch → TAQ
arrow(ax, 3.0, 6.12, 2.0, 5.32)
# Center branch → Refinitiv
arrow(ax, 5.0, 6.12, 5.0, 5.32)
# Right branch → FinBERT
arrow(ax, 7.0, 6.12, 8.1, 5.32)

#  Row 4: Three data source boxes 
# Left: NYSE TAQ
draw_box(ax, 2.0, 4.9, 2.8, 0.75,
         "NYSE TAQ\nMillisecond trades\n(Lee-Ready signed)", TEAL, fontsize=8)

# Center: Refinitiv / OpenSMILE
draw_box(ax, 5.0, 4.9, 2.8, 0.75,
         "Refinitiv StreetEvents\nAudio → OpenSMILE\n88 eGeMAPS features", GREEN, fontsize=8)

# Right: FinBERT
draw_box(ax, 8.1, 4.9, 2.6, 0.75,
         "FinBERT NLP\nAnalyst Q&A tone\n(3-class sentiment)", ORANGE,
         text_color="#333333", fontsize=8)

#  OI Shift label on left side 
ax.text(0.3, 4.9, "OI Shift\n(ΔBuying\npressure)",
        ha="center", va="center", fontsize=7.5, color=GRAY,
        style="italic", multialignment="center")
ax.annotate("", xy=(1.2, 4.9), xytext=(0.7, 4.9),
            arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2, mutation_scale=10),
            zorder=2)

#  Arrows from three boxes down to controls label 
arrow(ax, 2.0, 4.52, 3.5, 3.82)
arrow(ax, 5.0, 4.52, 5.0, 3.82)
arrow(ax, 8.1, 4.52, 6.5, 3.82)

#  Controls label 
ax.text(5.0, 3.65, "+ WRDS Compustat / CRSP / I/B/E/S controls",
        ha="center", va="center", fontsize=8.5, color=GRAY,
        style="italic")

#  Arrow down to final dataset 
arrow(ax, 5, 3.45, 5, 2.85)

#  Final dataset box 
draw_box(ax, 5, 2.45, 5.5, 0.75,
         "analysis_dataset_MASTER.parquet\n310 obs × 156 variables  |  281 complete cases",
         DARK_BLUE, fontsize=9)

#  Figure caption 
ax.text(5, 1.7,
        "Figure 3.1: Sample construction and data integration pipeline.",
        ha="center", va="center", fontsize=10, color="#333333",
        style="italic")

plt.tight_layout(pad=0.5)
plt.savefig(fig_dir / "fig3_1_pipeline.pdf", bbox_inches="tight", dpi=150)
plt.savefig(fig_dir / "fig3_1_pipeline.png", bbox_inches="tight", dpi=150)
plt.close()

print(f"Saved: {fig_dir}/fig3_1_pipeline.pdf")
print(f"Saved: {fig_dir}/fig3_1_pipeline.png")
