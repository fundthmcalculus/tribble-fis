"""Does making a marginal uniform actually help a FIS? Measured, here.

Issue #220 proposes the three uniformity-preserving scalers in
:mod:`tribblefis.scaling` on the strength of an accuracy table from
`grad-school/reproduce/experiments/uniformity_scaling_sweep.py`, run against UCI
Concrete, Body Fat and Bike Sharing. Those numbers are not reproducible from
this repository: the datasets are fetched over the network and the sweep
harness lives elsewhere. Restating them in this package's test suite would pin a
claim rather than a behaviour.

This script is the part that *is* reproducible. It generates regression problems
whose feature marginals are non-uniform in specific, named ways -- right skew, a
zero atom, bimodality, a heavy tail, a discrete ladder -- and measures two
things across seeds:

1. **Uniformity achieved**, as a Kolmogorov-Smirnov distance against
   Uniform(0, 1). This is what the scalers claim to do.
2. **Downstream test R^2** of a flat MoG-TSK regressor fitted through each
   scaler. This is what #220 claims it buys.

The synthetic generator is the point, not a weakness. On a real dataset an
improvement is a fact about that dataset; here the pathology is *dialled in*, so
"the improvement tracks the pathology" is checkable -- run `--dial` and watch
the gap open as skew increases and close as it goes to zero. A mechanism that
only shows up when the mechanism's precondition holds is evidence; a number from
one dataset is an anecdote.

Usage::

    uv run python -m benchmarks.uniformity_scaling                # the main table
    uv run python -m benchmarks.uniformity_scaling --seeds 30     # tighter error bars
    uv run python -m benchmarks.uniformity_scaling --dial         # skew sweep
    uv run python -m benchmarks.uniformity_scaling --json out.json

Nothing here runs in CI. It is a measurement tool, and its output belongs in a
pull request body where a reader can weigh it, not in an assertion that fails
when a library changes a tie-break.
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from benchmarks.workloads import _quiet
from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.scaling import (
    EmpiricalCDFScaler,
    MinMaxScaler,
    PiecewiseLinearCDFScaler,
    QuantileUniformScaler,
)

# Every arm is bounded to [0, 1] except `raw`, so the comparison isolates the
# *shape* change rather than the bounding. `raw` is kept because it is the
# floor the whole exercise has to clear, and #220's sibling finding is that
# z-score -- unbounded -- lands below it.
ARMS = {
    "raw (none)": None,
    "MinMaxScaler": lambda: MinMaxScaler(log_dynamic_range=None),
    "MinMaxScaler(log)": lambda: MinMaxScaler(log_dynamic_range=1.0),
    "EmpiricalCDF": EmpiricalCDFScaler,
    "PL-CDF k=5": lambda: PiecewiseLinearCDFScaler(n_pieces=5),
    "PL-CDF k=10": lambda: PiecewiseLinearCDFScaler(n_pieces=10),
    "PL-CDF k=25": lambda: PiecewiseLinearCDFScaler(n_pieces=25),
    "QuantileUniform": QuantileUniformScaler,
}


def make_pathological_regression(n_samples: int, seed: int, skew: float = 2.0):
    """A regression problem whose features are non-uniform in five named ways.

    The response is a smooth function of the features' *ranks* rather than of
    their raw magnitudes. That choice is load-bearing and worth defending: a
    target built as a linear function of the raw values would be trivially
    recovered by the 1st-order TSK consequents whatever the antecedents did,
    and the benchmark would measure the consequent solver rather than membership
    placement. Ranks put the signal exactly where membership functions have to
    resolve it.
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    columns = {
        # Right skew, the case log1p already handles -- included so the table
        # shows what the uniformity transforms add *beyond* it.
        "skewed": rng.lognormal(0.0, skew, n),
        # A heavy atom at the minimum. Membership functions placed on the mean
        # and variance of this column sit inside the atom and resolve nothing
        # above it.
        "zero_atom": np.where(rng.random(n) < 0.4, 0.0, rng.uniform(1.0, 10.0, n)),
        # Two clusters with a gap. Data statistics put a membership function in
        # the middle, where there is no data at all.
        "bimodal": np.concatenate(
            [rng.normal(-4.0, 0.7, n // 2), rng.normal(4.0, 0.7, n - n // 2)]
        ),
        # Heavy tail: a few points hundreds of times the median drag the
        # variance, so every membership function is far too wide.
        "heavy_tail": rng.standard_t(df=2.0, size=n),
        # A discrete ladder. No monotone map can make this uniform; it is here
        # to confirm the scalers do not make it *worse*.
        "ladder": rng.integers(0, 6, n).astype(float),
        # A well-behaved control. If a scaler helps here too, the mechanism
        # claimed in #220 is not the mechanism doing the work.
        "uniform_control": rng.uniform(0.0, 1.0, n),
    }
    X = pd.DataFrame(columns)

    ranks = X.rank(pct=True)
    y = (
        2.0 * ranks["skewed"]
        + 1.5 * ranks["zero_atom"]
        - 1.0 * ranks["bimodal"]
        + 0.8 * ranks["heavy_tail"]
        + 0.5 * ranks["ladder"]
        + rng.normal(0.0, 0.05, n)
    )
    return X, y.to_numpy()


def _ks_against_uniform(u: np.ndarray) -> float:
    x = np.sort(np.asarray(u, dtype=float))
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan")
    lo, hi = x.min(), x.max()
    x = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    return float(
        max(np.max(np.arange(1, n + 1) / n - x), np.max(x - np.arange(0, n) / n))
    )


def _one_seed(arm_factory, X, y, seed: int):
    """Fit one arm on one split. Returns (test R^2, mean KS of the features).

    The scaler is fitted on the training fold only -- #220's table was measured
    that way, and it is the only usage that is not leakage.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    if arm_factory is None:
        Xtr, Xte = X_train, X_test
    else:
        scaler = arm_factory().set_output(transform="pandas")
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)

    ks = float(np.mean([_ks_against_uniform(Xtr[c].to_numpy()) for c in Xtr.columns]))

    # `_quiet` because `fit` unconditionally prints a feature-ranking table
    # with no verbose switch, and this loop calls it ARMS x seeds times.
    with warnings.catch_warnings(), _quiet():
        warnings.simplefilter("ignore")
        model = TribbleRegressor(n_gaussians=3, random_state=seed)
        model.fit(Xtr, y_train)
        predictions = model.predict(Xte)
    return float(r2_score(y_test, predictions)), ks


def run(n_samples: int, seeds: int, skew: float):
    X_ref, _ = make_pathological_regression(n_samples, 0, skew)
    rows = []
    for name, factory in ARMS.items():
        scores, ks_values = [], []
        for seed in range(seeds):
            X, y = make_pathological_regression(n_samples, seed, skew)
            score, ks = _one_seed(factory, X, y, seed)
            scores.append(score)
            ks_values.append(ks)
        rows.append(
            {
                "arm": name,
                "r2_mean": float(np.mean(scores)),
                "r2_std": float(np.std(scores)),
                "ks_mean": float(np.mean(ks_values)),
                "n_seeds": seeds,
            }
        )
    return rows, list(X_ref.columns)


def render(rows, baseline="MinMaxScaler"):
    base = next((r["r2_mean"] for r in rows if r["arm"] == baseline), None)
    width = max(len(r["arm"]) for r in rows)
    lines = [
        f"{'arm'.ljust(width)}  {'test R^2':>17}  {'KS(uniform)':>11}  {'vs ' + baseline:>14}",
        "-" * (width + 50),
    ]
    for r in rows:
        delta = "" if base is None else f"{r['r2_mean'] - base:+.3f}"
        lines.append(
            f"{r['arm'].ljust(width)}  {r['r2_mean']:8.3f} +/- {r['r2_std']:.3f}"
            f"  {r['ks_mean']:11.4f}  {delta:>14}"
        )
    return "\n".join(lines)


def dial(n_samples: int, seeds: int):
    """Sweep the skew parameter from "no pathology" to "severe".

    This is the argument the accuracy table alone cannot make. If the uniformity
    transforms are helping *because* they fix non-uniform marginals, the gap has
    to vanish when the marginals are already fine. If it does not, something
    else is doing the work.
    """
    lines = [f"{'skew':>6}  {'MinMaxScaler':>16}  {'EmpiricalCDF':>16}  {'delta':>8}"]
    lines.append("-" * 54)
    out = []
    for skew in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0):
        rows, _ = run(n_samples, seeds, skew)
        mm = next(r for r in rows if r["arm"] == "MinMaxScaler")
        ec = next(r for r in rows if r["arm"] == "EmpiricalCDF")
        delta = ec["r2_mean"] - mm["r2_mean"]
        out.append({"skew": skew, "minmax": mm["r2_mean"], "ecdf": ec["r2_mean"], "delta": delta})
        lines.append(
            f"{skew:6.2f}  {mm['r2_mean']:8.3f}+/-{mm['r2_std']:.3f}"
            f"  {ec['r2_mean']:8.3f}+/-{ec['r2_std']:.3f}  {delta:+8.3f}"
        )
    return out, "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--skew", type=float, default=2.0)
    parser.add_argument(
        "--dial",
        action="store_true",
        help="sweep the skew parameter instead of printing the main table.",
    )
    parser.add_argument("--json", metavar="PATH", help="also write the raw numbers.")
    args = parser.parse_args(argv)

    if args.dial:
        payload, text = dial(args.samples, args.seeds)
        print(text)
    else:
        rows, features = run(args.samples, args.seeds, args.skew)
        payload = {"rows": rows, "features": features, "skew": args.skew}
        print(
            f"N={args.samples}, {len(features)} features, {args.seeds} seeds, "
            f"80/20 split, TribbleRegressor(n_gaussians=3), scaler fit on train fold only"
        )
        print()
        print(render(rows))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
