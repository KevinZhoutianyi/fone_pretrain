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
# Figure 1 — Does FoNE help number-magnitude comparison, and where?
# Answer: the gain appears at long digit lengths, and learning the frequencies wins.
# =============================================================================
def fig1_compare():
    d = load("experiments/chunk_fone/results.json")
    digits = d["digits"]
    style = {  # variant -> (label, color)
        "baseline":     ("Baseline (learned emb.)", C_GRAY),
        "fone":         ("FoNE, fixed 3 periods",   C_LBLUE),
        "fone_12d":     ("FoNE, fixed 6 periods",   C_ORANGE),
        "fone_learned": ("FoNE, learned freq.",     C_BLUE),
    }
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for v, (label, color) in style.items():
        if v not in d["variants"]:
            continue
        s = d["variants"][v]
        m = [100 * x for x in s["mean"]]
        sd = [100 * x for x in s["std"]]
        lo = [max(0, mi - si) for mi, si in zip(m, sd)]
        hi = [mi + si for mi, si in zip(m, sd)]
        ax.fill_between(digits, lo, hi, alpha=0.15, color=color, linewidth=0)
        ax.plot(digits, m, marker="o", color=color, label=label,
                linewidth=2 if v == "fone_learned" else 1.4)
    ax.set_xlabel("number length (digits)")
    ax.set_ylabel("comparison accuracy (%)")
    ax.set_xticks(digits)
    ax.set_ylim(0, 70)
    n_seeds = d["variants"]["baseline"]["n_seeds"]
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"Number-magnitude comparison (125M, mean of {n_seeds} seeds, band = std)",
                 fontsize=10)
    out = OUT_DIR / "fig1_compare.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"  → {out.name}")


if __name__ == "__main__":
    print("Generating figures:")
    fig1_compare()
    print(f"\nDone. Figures in {OUT_DIR}/")
