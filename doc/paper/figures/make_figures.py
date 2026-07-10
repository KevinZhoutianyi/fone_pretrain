"""
Generate paper figures from experiments/*/results.json files.

Run from the repo root:
    uv run python doc/paper/figures/make_figures.py

Outputs PDF figures into doc/paper/figures/ for the LaTeX project.

This is a starter template. Add one function per figure; each reads the
relevant results.json file(s), plots, and saves into doc/paper/figures/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[3]   # doc/paper/figures/X.py → repo
OUT_DIR = Path(__file__).resolve().parent


# Research-template style — clean, publication-ready.
plt.rcParams.update({
    # Font
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":             11,
    "axes.titlesize":        11,
    "axes.labelsize":        11,
    "xtick.labelsize":       10,
    "ytick.labelsize":       10,
    "legend.fontsize":       10,
    # Grid / axes
    "axes.grid":             True,
    "grid.color":            "#E0E0E0",
    "grid.linewidth":        0.6,
    "axes.axisbelow":        True,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.linewidth":        0.8,
    # Ticks
    "xtick.major.size":      4,
    "ytick.major.size":      4,
    # Output
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "figure.facecolor":      "white",
    "axes.facecolor":        "white",
})

# Colorblind-friendly palette (Okabe–Ito). Use these named constants — never raw hex.
C_BLUE    = "#0072B2"
C_ORANGE  = "#E69F00"
C_GREEN   = "#009E73"
C_RED     = "#D55E00"
C_LBLUE   = "#56B4E9"
C_YELLOW  = "#F0E442"
C_GRAY    = "#999999"
C_BLACK   = "#000000"


def load(path: str) -> dict:
    return json.loads((REPO / path).read_text())


# =============================================================================
# Figure 1 — [headline discriminator] (placeholder)
# =============================================================================
def fig1_main():
    # d = load("experiments/01_main/results.json")
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.text(0.5, 0.5, "[fig1_main placeholder]\nReplace with real data",
            ha="center", va="center", transform=ax.transAxes)
    out = OUT_DIR / "fig1_main.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"  → {out.name}")


if __name__ == "__main__":
    print("Generating figures:")
    fig1_main()
    print(f"\nDone. Figures in {OUT_DIR}/")
