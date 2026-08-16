"""How expensive would general type-2 (GT2) fuzzy inference be, given the
standard alpha-plane decomposition, on top of this repo's existing interval
type-2 (IT2) machinery?

Alpha-plane GT2 (Mendel, Liu 2008) decomposes each membership's secondary
grade into K nested IT2-shaped intervals -- alpha in (0, 1], narrowest to
widest -- and runs *ordinary, unmodified* IT2 inference and Karnik-Mendel type
reduction on each level independently, then combines the K results with an
alpha-weighted average. That means the forward-pass and type-reduction cost of
a K-plane GT2 model is mechanically "K x today's IT2 cost, plus a cheap
combination step" -- and today's IT2 kernel (`it2_firing_strengths`,
`karnik_mendel_tsk`) already exists and needs no changes to be called that way.
This script measures that multiplier for real, at this repo's typical model
sizes, rather than assuming it.

Run with ``python -m benchmarks.gt2_alpha_plane_probe``. Backs the cost-
estimate section of ``docs/gt2-evaluation.md`` (issue #122's research spike).
This is a measurement script, not a proposed implementation: the K-plane loop
below calls each function directly rather than through a real
``GT2GaussianMembership`` alpha-plane extractor, because no such type exists
yet -- what is being measured is the shape of the cost, which does not depend
on that type's existence.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from tribblefis.gauss_data import resolve_norm_pair
from tribblefis.it2_classifier import IT2TribbleClassifier
from tribblefis.it2_kernel import it2_firing_strengths, karnik_mendel_tsk

from .workloads import make_dataset, make_model


def _time_min(fn, *args, repeats: int, **kwargs) -> float:
    fn(*args, **kwargs)  # warm up (numba JIT, caches)
    return min(_timed(fn, args, kwargs) for _ in range(repeats))


def _timed(fn, args, kwargs) -> float:
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


def _forward_pass_multiplier(n_samples: int, n_features: int, n_labels: int,
                              n_mf: int, k_values: tuple[int, ...], repeats: int) -> None:
    """Classification-shaped cost: `it2_firing_strengths` called once per
    alpha-plane, on the same shape as the `forward-large` benchmark workload."""
    X, _ = make_dataset(n_samples, n_features, n_labels, seed=0)
    model = make_model(n_features, n_labels, n_mf, seed=0)
    it2_model = IT2TribbleClassifier()._convert_to_it2(model)
    norms = resolve_norm_pair("min/max")

    def one_plane():
        return it2_firing_strengths(X, it2_model, norms, km_iterations=None)

    def k_planes(K):
        return [one_plane() for _ in range(K)]

    base = _time_min(one_plane, repeats=repeats)
    print(f"\n=== Forward pass (it2_firing_strengths), {n_samples}x{n_features} "
          f"x {n_labels} labels x {n_mf} MF ===")
    print(f"K=1 (today's IT2): {base * 1000:8.2f} ms")
    for K in k_values:
        t = _time_min(k_planes, K, repeats=max(repeats - 2, 1))
        print(f"K={K:<3d}            : {t * 1000:8.2f} ms  "
              f"({t / base:5.2f}x base, {t / base / K:.3f}x per-plane)")


def _km_search_multiplier(n_samples: int, n_rules: int,
                           k_values: tuple[int, ...], repeats: int, seed: int) -> None:
    """Regression-shaped cost: `karnik_mendel_tsk` -- the genuinely new
    switch-point search -- called once per alpha-plane. `n_rules` defaults to
    this library's typical "handful of output buckets" (IT2_GUIDE.md)."""
    rng = np.random.default_rng(seed)
    rule_values = rng.normal(0.0, 1.0, size=(n_samples, n_rules))
    firing_lower = rng.uniform(0.0, 0.5, size=(n_samples, n_rules))
    firing_upper = firing_lower + rng.uniform(0.0, 0.5, size=(n_samples, n_rules))

    def one_plane():
        return karnik_mendel_tsk(rule_values, firing_lower, firing_upper, max_iterations=50)

    def k_planes(K):
        return [one_plane() for _ in range(K)]

    base = _time_min(one_plane, repeats=repeats)
    print(f"\n=== Karnik-Mendel search (karnik_mendel_tsk), {n_samples} samples "
          f"x {n_rules} rules ===")
    print(f"K=1 (today's IT2 KM): {base * 1000:8.2f} ms")
    for K in k_values:
        t = _time_min(k_planes, K, repeats=max(repeats - 2, 1))
        print(f"K={K:<3d}                : {t * 1000:8.2f} ms  "
              f"({t / base:5.2f}x base, {t / base / K:.3f}x per-plane)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-k", "--alpha-planes", type=int, nargs="+", default=[3, 5, 10, 20],
                   help="alpha-plane counts K to measure the multiplier at")
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args(argv)

    _forward_pass_multiplier(50_000, 20, 8, 4, tuple(args.alpha_planes), args.repeats)
    _km_search_multiplier(50_000, 8, tuple(args.alpha_planes), args.repeats, seed=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
