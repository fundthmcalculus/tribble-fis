"""Comparison figure: FES against the small-embedding baselines, plus the
fuzziness ablation.

All points are measured through the same harness (fuzzyembed/evaluate.py), so they
are mutually comparable. Palette is the validated 3-slot categorical set
(#2a78d6 / #eb6834 / #1baf7a), all-pairs gates passed; the aqua contrast WARN is
discharged by direct-labelling every point.

    python scripts/plot_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def mean(xs):
    return sum(xs) / len(xs)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fes_comparison.png"

# --- palette (validated) ---------------------------------------------------
FES_FUZZY = "#2a78d6"  # slot 1 blue
BASELINE = "#eb6834"  # slot 2 orange
FES_CTRL = "#1baf7a"  # slot 3 aqua
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# --- data (self-measured, MTEB-14 + NanoBEIR) ------------------------------
# name, params(M), MTEB-14, NanoBEIR
BASELINES = [
    ("potion-2M", 1.89, 49.62, 0.3666),
    ("potion-8M", 7.56, 52.74, 0.4421),
    ("potion-32M", 32.30, 54.08, 0.4637),
    ("static-retrieval-mrl", 31.25, 51.02, 0.5032),
    ("MiniLM-L6", 22.71, 60.52, 0.5623),
]
FES_BEST = ("FES R=32", 2.52, 51.38, 0.4220)
FES_CONTROL = ("FES R=1 (control)", 2.52, 51.44, 0.4033)

# fuzziness axis: H_fire, MTEB-14, NanoBEIR, label
FUZZ = [
    (0.010, 51.38, 0.4220, "A4 no-UR"),
    (0.141, 50.86, 0.4181, "A3 product"),
    (0.208, 50.96, 0.4166, "A2 HTSK"),
    (0.503, 51.25, 0.4200, "A8 anchored"),
]
# Three seeds each for the two configs the argument rests on (E010). Bands, not
# lines -- the separation between them is the project's one surviving claim.
CTRL_SEEDS = [0.40332, 0.40959, 0.39625]   # A1b, R=1 param-matched control
FUZZ_SEEDS = [0.42200, 0.42738, 0.42595]   # A4, R=32 rule base


def _style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11.5, fontweight="600", loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


def _scatter_panel(ax, metric_idx, ylabel, title, ylim):
    for name, p, m14, nb in BASELINES:
        y = (m14, nb)[metric_idx]
        ax.plot(p, y, "o", ms=9, color=BASELINE, mec=SURFACE, mew=2, zorder=3)
        ax.annotate(name, (p, y), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=8, color=INK2)

    for (name, p, m14, nb), col in ((FES_CONTROL, FES_CTRL), (FES_BEST, FES_FUZZY)):
        y = (m14, nb)[metric_idx]
        ax.plot(p, y, "D", ms=10, color=col, mec=SURFACE, mew=2, zorder=4)

    # Both FES points sit at 2.52M, so label them apart by hand.
    yb = (FES_BEST[2], FES_BEST[3])[metric_idx]
    yc = (FES_CONTROL[2], FES_CONTROL[3])[metric_idx]
    arrow = dict(arrowstyle="-", lw=0.9, color=INK3, shrinkA=0, shrinkB=5)
    # Offset left, not right: a rightward callout ran into the potion-8M label.
    ax.annotate("FES R=32", (2.52, yb), textcoords="offset points", xytext=(-30, 30),
                ha="center", fontsize=8.5, color=INK, fontweight="bold", arrowprops=arrow)
    ax.annotate("FES R=1 ctrl", (2.52, yc), textcoords="offset points", xytext=(30, -32),
                ha="center", fontsize=8.5, color=INK2, arrowprops=arrow)

    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 40])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # A log axis draws minor ticks with their own labels by default; at this span
    # they rendered as overlapping "3x10^0" strings on top of the major labels.
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(1.4, 55)
    ax.set_ylim(*ylim)
    _style(ax, "parameters (millions, log scale)", ylabel, title)


def main() -> int:
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0))
    fig.patch.set_facecolor(SURFACE)

    _scatter_panel(axes[0], 0, "MTEB-14 average",
                   "A · Aggregate quality vs size", (42, 63))
    _scatter_panel(axes[1], 1, "NanoBEIR nDCG@10",
                   "B · Retrieval quality vs size", (0.34, 0.60))

    # Panel C: does the *fuzziness* matter?
    ax = axes[2]
    # Two seed bands (3 seeds each). Labels sit inside their own band so they
    # cannot collide with the marks or each other.
    ax.axhspan(min(FUZZ_SEEDS), max(FUZZ_SEEDS), color=FES_FUZZY, alpha=0.20, zorder=1)
    ax.axhspan(min(CTRL_SEEDS), max(CTRL_SEEDS), color=FES_CTRL, alpha=0.22, zorder=1)
    ax.axhline(mean(CTRL_SEEDS), color=FES_CTRL, lw=1.6, zorder=2)
    ax.text(0.612, mean(FUZZ_SEEDS), "R=32 rule base\n3 seeds", ha="right", va="center",
            fontsize=7.5, color=INK, fontweight="bold", zorder=6)
    ax.text(0.612, mean(CTRL_SEEDS), "R=1 control\n3 seeds, same params", ha="right",
            va="center", fontsize=7.5, color=INK2, zorder=6)
    # The gap between the bands is the result, so mark it explicitly.
    ax.annotate("", xy=(0.05, min(FUZZ_SEEDS)), xytext=(0.05, max(CTRL_SEEDS)),
                arrowprops=dict(arrowstyle="<->", lw=1.2, color=INK))
    ax.text(0.072, max(CTRL_SEEDS) + 0.0012,
            "bands never overlap\n+5.5%,  exact perm. p = 0.05",
            ha="left", va="bottom", fontsize=8, color=INK, zorder=6)

    xs = [f[0] for f in FUZZ]
    ys = [f[2] for f in FUZZ]
    ax.plot(xs, ys, "-", lw=2, color=FES_FUZZY, alpha=0.45, zorder=3)
    for x, _m, y, lab in FUZZ:
        ax.plot(x, y, "o", ms=9, color=FES_FUZZY, mec=SURFACE, mew=2, zorder=4)
        # A8 is the rightmost point; label it below so it clears the band caption.
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(0, -17 if lab.startswith("A8") else 11),
                    ha="center", fontsize=8, color=INK2)

    ax.set_xlim(-0.05, 0.62)
    ax.set_ylim(0.3910, 0.4385)
    _style(ax, "per-token firing entropy  (0 = hard router, 1 = uniform)",
           "NanoBEIR nDCG@10", "C · Rule base vs control (3 seeds each)")
    ax.annotate("flat across a 50x range of softness",
                (0.32, 0.4345), ha="center", fontsize=8.5, color=INK, style="italic")

    legend = [
        Line2D([], [], marker="D", ls="", ms=9, color=FES_FUZZY, mec=SURFACE, mew=1.5,
               label="FES, fuzzy rule base (R=32)"),
        Line2D([], [], marker="D", ls="", ms=9, color=FES_CTRL, mec=SURFACE, mew=1.5,
               label="FES, R=1 control (no rule base)"),
        Line2D([], [], marker="o", ls="", ms=9, color=BASELINE, mec=SURFACE, mew=1.5,
               label="small-model baselines (measured here, same harness)"),
    ]
    leg = fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
                     fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle(
        "A fuzzy inference system as an embedding model: 2.5M parameters vs the small-model field",
        color=INK, fontsize=13.5, fontweight="700", x=0.006, ha="left", y=0.985,
    )
    fig.text(0.006, 0.925,
             "All points measured through one harness (14-task MTEB subset + NanoBEIR). "
             "At matched parameters the rule base gives NO aggregate gain (A: -0.12, p=0.75) "
             "but a replicated retrieval gain (C: +5.5%, p=0.05).",
             color=INK2, fontsize=9.5, ha="left")

    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    fig.savefig(OUT, dpi=170, facecolor=SURFACE)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
