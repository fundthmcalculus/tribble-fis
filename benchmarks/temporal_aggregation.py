"""Learned fuzzy aggregation over time vs hand-picked window statistics (#127).

Issue #127 describes three ways the N-CMAPSS RUL work collapsed a flight cycle
to a row: whole-cycle summary statistics, the same statistics per crisp third
of the cycle, and fixed-width rolling averages. All three pick *where* to look
and *what to compute* by hand. `tribblefis.temporal_aggregation` learns both.
This script measures whether that buys anything, and -- the part a single
dataset could not show -- whether the gain *tracks the mechanism*.

The generator plants the signal where the hand-engineered arm is weakest:

* sensor ``s0`` carries a bump whose height is part of the target, centred at
  ``phase`` in normalised time. Put ``phase`` at 0.5 and the bump sits inside
  the middle crisp third, where "mean of the middle third" is close to the
  right statistic. Move it to 2/3 and it straddles the tercile boundary, so
  every crisp third dilutes it with unrelated samples.
* sensor ``s1`` carries a 2-3 sample spike at a random time whose height is
  the other part of the target. That is a max-like statistic; the hand arm
  includes ``max`` and should handle it, so it is a control on the learned
  quantifier rather than a place to win.
* ``s2``, ``s3`` are nuisance: per-cycle offset, slow drift, noise.
* every sensor gets a per-cycle random offset, so a plain mean is
  contaminated and contrast between regions matters.

Cycle lengths vary (60-140 samples), as flight cycles do.

Arms, each scored under two heads -- ``TribbleRegressor(n_gaussians=3)`` and a
standardised ``RidgeCV``. The ridge head is there because a FIS on 60 features
and ~180 rows can fail for reasons unrelated to the representation; the ridge
head reads the representation alone. (It is also the learned arm's surrogate,
i.e. home turf -- which is why the FIS head is reported too.)

=================  ==========================================================
``whole-cycle``    mean/std/min/max/last over the cycle            (#127 arm 1)
``crisp-thirds``   mean/std/min/max/last per crisp third           (#127 arm 2)
``soft-thirds``    the aggregator at its start point, nothing learned
``learned``        the aggregator ``TribbleTimeSeriesRegressor`` fits
=================  ==========================================================

``soft-thirds`` is the ablation that matters: it isolates *learning* from
merely using a fuzzy partition.

Usage::

    uv run python -m benchmarks.temporal_aggregation              # main table
    uv run python -m benchmarks.temporal_aggregation --dial       # move the phase
    uv run python -m benchmarks.temporal_aggregation --stat-ablation  # which stats matter
    uv run python -m benchmarks.temporal_aggregation --json out.json

Nothing here runs in CI; numbers belong in a PR body or the design doc.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.workloads import _quiet
from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.temporal_aggregation import FuzzyTemporalAggregator

SENSORS = ["s0", "s1", "s2", "s3"]
_STATS = ("mean", "std", "min", "max", "last")


def make_cycles(n_cycles: int, seed: int, phase: float = 0.68, phase_width: float = 0.06):
    """Variable-length multivariate cycles with a planted temporal structure."""
    rng = np.random.default_rng(seed)
    cycles, y = [], np.empty(n_cycles)
    for i in range(n_cycles):
        length = int(rng.integers(60, 141))
        tau = (np.arange(length) + 0.5) / length
        bump_height = rng.uniform(0.0, 1.0)
        spike_height = rng.uniform(0.0, 1.0)
        data = np.empty((length, len(SENSORS)))
        for j in range(len(SENSORS)):
            offset = rng.normal(0.0, 0.5)
            drift = rng.normal(0.0, 0.3) * tau
            data[:, j] = offset + drift + rng.normal(0.0, 0.25, length)
        data[:, 0] += 1.5 * bump_height * np.exp(-0.5 * ((tau - phase) / phase_width) ** 2)
        start = int(rng.integers(int(0.1 * length), int(0.9 * length)))
        data[start:start + int(rng.integers(2, 4)), 1] += 2.0 * spike_height
        cycles.append(pd.DataFrame(data, columns=SENSORS))
        y[i] = bump_height + 0.5 * spike_height + rng.normal(0.0, 0.03)
    return cycles, y


def _stats(block: np.ndarray) -> list[float]:
    return [block.mean(), block.std(), block.min(), block.max(), block[-1]]


def hand_features(cycles, n_parts: int) -> pd.DataFrame:
    """#127's hand-engineered arms: summary statistics per crisp 1/n_parts."""
    rows = []
    for c in cycles:
        arr = c.to_numpy()
        bounds = np.linspace(0, len(arr), n_parts + 1).round().astype(int)
        row = {}
        for p in range(n_parts):
            seg = arr[bounds[p]:max(bounds[p + 1], bounds[p] + 1)]
            for j, name in enumerate(c.columns):
                for stat, val in zip(_STATS, _stats(seg[:, j])):
                    row[f"{name}_p{p}_{stat}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


HEADS = ("tribble", "ridge")


def _fit_head(head, Xtr, ytr, Xte, seed):
    if head == "tribble":
        model = TribbleRegressor(n_gaussians=3, random_state=seed)
    else:
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    model.fit(Xtr, ytr)
    return np.asarray(model.predict(Xte), dtype=float).ravel()


def _features(arm: str, ctr, ytr, cte, seed):
    """Fit the arm's sequence -> row map on the training cycles only."""
    info = {}
    if arm == "whole-cycle":
        return hand_features(ctr, 1), hand_features(cte, 1), info
    if arm == "crisp-thirds":
        return hand_features(ctr, 3), hand_features(cte, 3), info
    if arm == "soft-thirds":
        agg = FuzzyTemporalAggregator(n_segments=3).fit(ctr)  # y=None: nothing learned
    elif arm == "learned":
        agg = FuzzyTemporalAggregator(n_segments=3, random_state=seed).fit(ctr, ytr)
        info["centers"] = agg.centers_.round(3).tolist()
        info["s0_orness"] = agg.orness_[:, 0].round(2).tolist()
        info["s1_orness"] = agg.orness_[:, 1].round(2).tolist()
        info["surrogate_init"] = agg.objective_init_
        info["surrogate_learned"] = agg.objective_
    else:
        raise ValueError(arm)
    return agg.transform(ctr), agg.transform(cte), info


def _one_seed(arm: str, cycles, y, seed: int):
    """Test R^2 of each head on one 75/25 split, plus the learned parameters.

    The `learned` arm's aggregator is exactly what `TribbleTimeSeriesRegressor`
    fits; it is called directly here so both heads see the same features.
    """
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=seed)
    ctr = [cycles[i] for i in idx_tr]
    cte = [cycles[i] for i in idx_te]
    ytr, yte = y[idx_tr], y[idx_te]
    t0 = time.perf_counter()
    with warnings.catch_warnings(), _quiet():
        warnings.simplefilter("ignore")
        Ztr, Zte, info = _features(arm, ctr, ytr, cte, seed)
        info["seconds"] = time.perf_counter() - t0
        info["n_features"] = Ztr.shape[1]
        scores = {h: float(r2_score(yte, _fit_head(h, Ztr, ytr, Zte, seed))) for h in HEADS}
    return scores, info


ARMS = ("whole-cycle", "crisp-thirds", "soft-thirds", "learned")


def run(n_cycles: int, seeds: int, phase: float, arms=ARMS):
    rows = []
    details = []
    for arm in arms:
        scores = {h: [] for h in HEADS}
        secs, n_feat = [], 0
        for seed in range(seeds):
            cycles, y = make_cycles(n_cycles, seed, phase=phase)
            s, info = _one_seed(arm, cycles, y, seed)
            for h in HEADS:
                scores[h].append(s[h])
            secs.append(info["seconds"])
            n_feat = info["n_features"]
            if arm == "learned":
                details.append({"seed": seed, **info})
        row = {"arm": arm, "n_features": n_feat, "feature_seconds": float(np.mean(secs))}
        for h in HEADS:
            row[f"{h}_r2_by_seed"] = scores[h]
            row[f"{h}_r2_mean"] = float(np.mean(scores[h]))
        rows.append(row)
    return rows, details


def _sem(a) -> float:
    a = np.asarray(a, dtype=float)
    return float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def render(rows, baseline="crisp-thirds"):
    base = next(r for r in rows if r["arm"] == baseline)
    lines = []
    for h in HEADS:
        lines += [f"head = {h}",
                  f"{'arm':<14}  {'feats':>5}  {'test R^2':>16}  {'vs ' + baseline:>16}  {'wins':>5}",
                  "-" * 64]
        for r in rows:
            d = np.array(r[f"{h}_r2_by_seed"]) - np.array(base[f"{h}_r2_by_seed"])
            lines.append(
                f"{r['arm']:<14}  {r['n_features']:>5}  {r[f'{h}_r2_mean']:7.3f} +/- "
                f"{_sem(r[f'{h}_r2_by_seed']):.3f}  {d.mean():+7.3f} +/- {_sem(d):.3f}"
                f"  {int(np.sum(d > 0)):>2}/{len(d):<2}"
            )
        lines.append("")
    lines.append("feature-map fit seconds: " + ", ".join(
        f"{r['arm']} {r['feature_seconds']:.1f}" for r in rows))
    return "\n".join(lines)


def dial(n_cycles: int, seeds: int):
    """Move the planted phase from inside the middle third onto its boundary.

    If learning helps *because* it moves the segments to where the signal is,
    the gap over crisp thirds has to be small at phase=0.5 and open as the
    phase approaches 2/3. A gap that is flat across the sweep would mean
    something else is doing the work.
    """
    out = []
    lines = [f"{'phase':>6}  {'head':>8}  {'crisp-thirds':>8}  {'learned':>8}  {'delta':>16}",
             "-" * 56]
    for phase in (0.5, 0.58, 0.62, 0.667, 0.72):
        rows, _ = run(n_cycles, seeds, phase, arms=("crisp-thirds", "learned"))
        cr, le = rows
        for h in HEADS:
            d = np.array(le[f"{h}_r2_by_seed"]) - np.array(cr[f"{h}_r2_by_seed"])
            out.append({"phase": phase, "head": h, "crisp": cr[f"{h}_r2_mean"],
                        "learned": le[f"{h}_r2_mean"], "delta": float(d.mean()),
                        "delta_sem": _sem(d)})
            lines.append(f"{phase:6.3f}  {h:>8}  {cr[f'{h}_r2_mean']:12.3f}  "
                         f"{le[f'{h}_r2_mean']:8.3f}  {d.mean():+7.3f} +/- {_sem(d):.3f}")
    return out, "\n".join(lines)


def stat_ablation(n_cycles: int, seeds: int):
    """Which crisp-thirds statistics carry the score, at two phases.

    Added after the dial came back flat against the full five-statistic arm.
    The hypothesis -- a crisp boundary through the signal hurts -- predicts the
    *mean*-per-third arm (#127's literal "mean of the first third") collapses
    when the phase moves onto the boundary, while redundant statistics
    (max/std/min/last) can cover for it. Ridge head only; nothing learned, so
    this runs in seconds.
    """
    subsets = {
        "mean": ("mean",),
        "max": ("max",),
        "mean+max": ("mean", "max"),
        "all 5": _STATS,
    }
    out = []
    lines = [f"{'statistics':<10}  {'feats':>5}" + "".join(f"  {'phase ' + str(p):>12}" for p in (0.5, 0.68)),
             "-" * 44]
    for name, stats in subsets.items():
        row = {"stats": name, "n_features": 3 * len(stats) * len(SENSORS)}
        for phase in (0.5, 0.68):
            scores = []
            for seed in range(seeds):
                cycles, y = make_cycles(n_cycles, seed, phase=phase)
                tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=seed)
                feats = hand_features(cycles, 3)
                feats = feats[[c for c in feats.columns if c.rsplit("_", 1)[-1] in stats]]
                pred = _fit_head("ridge", feats.iloc[tr], y[tr], feats.iloc[te], seed)
                scores.append(float(r2_score(y[te], pred)))
            row[f"phase_{phase}"] = float(np.mean(scores))
            row[f"phase_{phase}_sem"] = _sem(scores)
        out.append(row)
        lines.append(f"{name:<10}  {row['n_features']:>5}" + "".join(
            f"  {row[f'phase_{p}']:6.3f}+/-{row[f'phase_{p}_sem']:.3f}" for p in (0.5, 0.68)))
    return out, "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cycles", type=int, default=240)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--phase", type=float, default=0.68)
    parser.add_argument("--dial", action="store_true", help="sweep the planted phase")
    parser.add_argument("--stat-ablation", action="store_true",
                        help="which crisp-thirds statistics carry the score (fast)")
    parser.add_argument("--json", metavar="PATH", help="also write the raw numbers")
    args = parser.parse_args(argv)

    if args.dial:
        payload, text = dial(args.cycles, args.seeds)
        print(text)
    elif args.stat_ablation:
        payload, text = stat_ablation(args.cycles, args.seeds)
        print(text)
    else:
        rows, details = run(args.cycles, args.seeds, args.phase)
        payload = {"rows": rows, "learned_details": details, "phase": args.phase}
        print(f"{args.cycles} cycles (60-140 samples, {len(SENSORS)} sensors), {args.seeds} seeds, "
              f"75/25 split, phase={args.phase}; heads: TribbleRegressor(n_gaussians=3), "
              f"standardised RidgeCV")
        print()
        print(render(rows))
        print()
        for d in details:
            print(f"seed {d['seed']}: centers={d['centers']} s0 orness={d['s0_orness']} "
                  f"s1 orness={d['s1_orness']} surrogate {d['surrogate_init']:.3f}"
                  f" -> {d['surrogate_learned']:.3f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
