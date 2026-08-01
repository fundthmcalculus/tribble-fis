"""Performance vs tunable parameters, with the Pareto frontier.

Answers "what does each parameter buy?" rather than "who wins outright". All points
are self-measured through fuzzyembed/evaluate.py; seed replicates are averaged.

    python scripts/plot_params.py
"""

from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fes_params.png"

FES_C = "#2a78d6"
BASE_C = "#eb6834"
CTRL_C = "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE = "#e4e3df", "#fcfcfb"

BASELINES = {"potion-base-2M", "potion-base-8M", "potion-base-32M",
             "static-retrieval-mrl-en-v1", "all-MiniLM-L6-v2"}
# R=1 configs: no rule base, so they are the "plain static" arm of the comparison.
CONTROLS = {"FES-A1b-ctrl-matched", "FES-A0-static-256"}  # A1 (2.0M) dropped: A1b supersedes it
# Shown points: the headline configs, not every ablation rung.
SHOW = BASELINES | CONTROLS | {"FES-A4-no-ur", "FES-S4-potion8M-matched"}
LABEL = {
    "potion-base-2M": "potion-2M", "potion-base-8M": "potion-8M",
    "potion-base-32M": "potion-32M", "static-retrieval-mrl-en-v1": "static-retr-mrl",
    "all-MiniLM-L6-v2": "MiniLM-L6", "FES-A1-lowrank-ctrl": "FES R=1 (2.0M)",
    "FES-A1b-ctrl-matched": "FES R=1 (2.5M)", "FES-A0-static-256": "FES R=1 (7.9M)",
    "FES-A4-no-ur": "FES R=32", "FES-S4-potion8M-matched": "FES R=32 (7.4M)",
}


def load():
    recs = []
    for f in ("results/records.jsonl", "results/records_seeds.jsonl"):
        p = ROOT / f
        if p.exists():
            recs += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if "summary" in r]
    g = collections.defaultdict(list)
    for r in recs:
        g[r["model"].split("-s7")[0].split("-s1234")[0]].append(r)
    out = {}
    for k, v in g.items():
        out[k] = {
            "params": v[0]["params"],
            "mteb": st.mean(x["summary"]["mteb14_avg"] for x in v),
            "nano": st.mean(x.get("nanobeir_ndcg@10", float("nan")) for x in v),
            "n": len(v),
        }
    return out


def pareto(pts):
    """Points not dominated on (low params, high score)."""
    best, front = -1e9, []
    for p, s, k in sorted(pts):
        if s > best:
            front.append((p, s, k))
            best = s
    return front


def panel(ax, data, metric, ylabel, title):
    pts = [(d["params"] / 1e6, d[metric], k) for k, d in data.items() if k in SHOW]
    front = pareto(pts)
    ax.plot([p for p, _, _ in front], [s for _, s, _ in front],
            "-", lw=1.4, color=INK3, alpha=0.7, zorder=2, label="_")

    for p, s, k in pts:
        if k in BASELINES:
            c, mk, ms = BASE_C, "o", 9
        elif k in CONTROLS:
            c, mk, ms = CTRL_C, "s", 8
        else:
            c, mk, ms = FES_C, "D", 10
        ax.plot(p, s, mk, ms=ms, color=c, mec=SURFACE, mew=2, zorder=4)

    # Hand-placed offsets: the 2-2.5M cluster is too tight for any heuristic.
    off = {
        # Keep everything inside the axes: at x=1.9-2.5 on a log axis starting at
        # 1.5 there is almost no room to the left, so these lean right.
        "potion-base-2M": (8, -17, "left"),
        "FES-A1b-ctrl-matched": (26, 12, "center"),
        "FES-A4-no-ur": (16, -19, "left"),
        "FES-A0-static-256": (-10, 13, "right"),
        "FES-S4-potion8M-matched": (6, -20, "center"),
        "potion-base-8M": (14, 9, "left"),
        "potion-base-32M": (0, 13, "center"),
        "static-retrieval-mrl-en-v1": (0, -19, "center"),
        "all-MiniLM-L6-v2": (0, 13, "center"),
    }
    for p, s, k in pts:
        dx, dy, ha = off.get(k, (0, 12, "center"))
        ax.annotate(LABEL.get(k, k), (p, s), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=8,
                    color=INK if k.startswith("FES-A4") or k.startswith("FES-S4") else INK2,
                    fontweight="bold" if k.startswith(("FES-A4", "FES-S4")) else "normal")

    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 40])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(1.5, 55)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel("tunable parameters (millions, log scale)", color=INK2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    for sd in ("left", "bottom"):
        ax.spines[sd].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


def main() -> int:
    data = load()
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.1))
    fig.patch.set_facecolor(SURFACE)

    panel(axes[0], data, "mteb", "MTEB-14 average", "A · Aggregate vs parameters")
    panel(axes[1], data, "nano", "NanoBEIR nDCG@10", "B · Retrieval vs parameters")

    # Panel C: where FES actually spends its parameters.
    ax = axes[2]
    cfgs = [("FES R=1 ctrl\n2.5M", [(2472282, "vocabulary table"), (30522, "pooling"),
                                    (21155, "rule base")]),
            ("FES R=32\n2.5M", [(1953408, "vocabulary table"), (30522, "pooling"),
                                (536705, "rule base")]),
            ("FES R=32\n7.4M", [(5799180, "vocabulary table"), (30522, "pooling"),
                                (1577213, "rule base")])]
    colours = {"vocabulary table": BASE_C, "pooling": CTRL_C, "rule base": FES_C}
    ys = range(len(cfgs))
    for i, (name, parts) in enumerate(cfgs):
        left = 0
        total = sum(v for v, _ in parts)
        for val, lab in parts:
            ax.barh(i, val / 1e6, left=left / 1e6, height=0.55,
                    color=colours[lab], edgecolor=SURFACE, lw=2, zorder=3)
            if val / total > 0.12:
                ax.text((left + val / 2) / 1e6, i, f"{val / total * 100:.0f}%",
                        ha="center", va="center", fontsize=8.5, color="white",
                        fontweight="bold", zorder=5)
            left += val
    ax.set_yticks(list(ys))
    ax.set_yticklabels([c[0] for c in cfgs], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_facecolor(SURFACE)
    ax.set_title("C · Where the parameters go", color=INK, fontsize=11.5,
                 fontweight="bold", loc="left", pad=10)
    ax.set_xlabel("parameters (millions)", color=INK2, fontsize=9.5)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sd in ("top", "right", "left"):
        ax.spines[sd].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.set_xlim(0, 8.9)
    # Rows 0-1 are short bars, so the upper-right corner is the only clear space.
    ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=9, color=colours[k], label=k)
                       for k in ("vocabulary table", "rule base", "pooling")],
              loc="upper right", bbox_to_anchor=(1.0, 0.90), frameon=False, fontsize=8.5)
    ax.text(0.02, 0.97, "the vocabulary table dominates at every size",
            transform=ax.transAxes, fontsize=8.5, color=INK2, style="italic", va="top")

    legend = [
        Line2D([], [], marker="D", ls="", ms=9, color=FES_C, mec=SURFACE, mew=1.5,
               label="FES with fuzzy rule base (R=32)"),
        Line2D([], [], marker="s", ls="", ms=8, color=CTRL_C, mec=SURFACE, mew=1.5,
               label="FES R=1 controls (no rule base)"),
        Line2D([], [], marker="o", ls="", ms=9, color=BASE_C, mec=SURFACE, mew=1.5,
               label="published small models (measured here)"),
        Line2D([], [], ls="-", lw=1.4, color=INK3, label="Pareto frontier"),
    ]
    leg = fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
                     fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.suptitle("Performance vs tunable parameters", color=INK, fontsize=13.5,
                 fontweight="bold", x=0.006, ha="left", y=0.985)
    fig.text(0.006, 0.925,
             "FES matches potion-base-8M at 98% of its parameters and leads it on retrieval; "
             "the rule base is 21% of the budget, the vocabulary table ~78%.",
             color=INK2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    fig.savefig(OUT, dpi=170, facecolor=SURFACE)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
