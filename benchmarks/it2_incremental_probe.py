"""Where IT2/GT2 acceleration should land: per-candidate refinement cost.

Backs docs/it2-gt2-acceleration-design.md (issue #146). Issue #146 asked for a
GPU forward pass for IT2/GT2; #143/#213 found refinement, not the forward pass,
is where type-2 time goes. This probe measures the cost of *one refinement
candidate* -- moving a single IT2 slot's ``(mu, sigma_lower, sigma_upper)`` and
re-evaluating firing bounds -- three ways:

1. ``rebuild``: what ``it2_refine`` does today -- rebuild the upper/lower
   ``GaussianMixtureModel`` from the IT2 model and run
   ``gauss_math.tsk_firing_strengths`` on each.
2. ``full_compiled``: the same two full passes, but on persistent
   ``kernel.CompiledFIS`` objects with parameters written in place (no rebuild).
3. ``incremental``: two ``kernel.IncrementalFIS.evaluate_slot`` calls (upper
   and lower), the path Type-1 coordinate refinement already uses.

GT2 multiplies every row by ``n_alpha_planes`` (one upper/lower pair per plane,
all sharing one layout), so the ratios carry over unchanged.

    python -m benchmarks.it2_incremental_probe
    python -m benchmarks.it2_incremental_probe --n 20000 --features 20 --labels 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from tribblefis import kernel
from tribblefis.gauss_data import (
    DefaultNormCornorm,
    GaussianMembership,
    IT2FeatureModel,
    IT2GaussianMembership,
    IT2GaussianMixtureModel,
    IT2LabelModel,
    resolve_norm_pair,
)
from tribblefis.gauss_math import tsk_firing_strengths
from tribblefis.it2_kernel import _extract_lower_model, _extract_upper_model


def _build(n_features: int, n_labels: int, n_mf: int, rng) -> IT2GaussianMixtureModel:
    fms = {}
    for f in range(n_features):
        lms = {}
        for label in range(n_labels):
            mfs = []
            for _ in range(n_mf):
                mu, sigma = rng.normal(), rng.uniform(0.3, 1.0)
                mfs.append(IT2GaussianMembership(
                    upper_mf=GaussianMembership(mu=mu, sigma=sigma * 1.5),
                    lower_mf=GaussianMembership(mu=mu, sigma=sigma * 0.75),
                ))
            lms[label] = IT2LabelModel(memberships=mfs)
        fms[f"x{f}"] = IT2FeatureModel(lms)
    return IT2GaussianMixtureModel(fms)


def _time(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--features", type=int, default=10)
    p.add_argument("--labels", type=int, default=6)
    p.add_argument("--mf", type=int, default=3)
    p.add_argument("--candidates", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args(argv)

    rng = np.random.default_rng(0)
    model = _build(args.features, args.labels, args.mf, rng)
    X = pd.DataFrame(rng.normal(size=(args.n, args.features)),
                     columns=[f"x{f}" for f in range(args.features)])
    norms = resolve_norm_pair(DefaultNormCornorm)
    arrays = {c: X[c].to_numpy() for c in X.columns}

    up_c = kernel.compile_model(_extract_upper_model(model), list(X.columns))
    lo_c = kernel.compile_model(_extract_lower_model(model), list(X.columns))
    fm = up_c.feature_matrix(arrays)
    inc_up = kernel.IncrementalFIS(up_c, fm, norms)
    inc_lo = kernel.IncrementalFIS(lo_c, fm, norms)
    slots = rng.integers(0, up_c.n_slots, size=args.candidates)

    def rebuild():
        for _ in slots:
            tsk_firing_strengths(X, _extract_upper_model(model), norms=norms, feature_arrays=arrays)
            tsk_firing_strengths(X, _extract_lower_model(model), norms=norms, feature_arrays=arrays)

    def full_compiled():
        for _ in slots:
            kernel.firing_strengths(up_c, fm, norms)
            kernel.firing_strengths(lo_c, fm, norms)

    def incremental():
        for s in slots:
            inc_up.evaluate_slot(int(s), 0.1, 0.9)
            inc_lo.evaluate_slot(int(s), 0.1, 0.4)

    t_rb = _time(rebuild, args.repeats) / args.candidates
    t_fc = _time(full_compiled, args.repeats) / args.candidates
    t_in = _time(incremental, args.repeats) / args.candidates
    print(f"shape: n={args.n} F={args.features} L={args.labels} K={args.mf} "
          f"cython={kernel.HAVE_CYTHON_KERNEL}")
    print("| path | ms / candidate | vs rebuild |")
    print("|---|--:|--:|")
    for name, t in (("rebuild (today)", t_rb), ("full_compiled", t_fc), ("incremental", t_in)):
        print(f"| {name} | {t * 1e3:.2f} | {t_rb / t:.1f}x |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
