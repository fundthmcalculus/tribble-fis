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
    python -m benchmarks.it2_incremental_probe --refine   # end-to-end, after the #146 port

``--refine`` times whole ``refine_it2*``/``refine_gt2*`` runs on wine
(classifier) and friedman1 (regressor), as shipped versus with the
incremental path disabled (the pre-#146 rebuild path).
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


def _refine_end_to_end(repeats: int) -> None:
    from contextlib import contextmanager
    from unittest import mock

    from sklearn.datasets import load_wine, make_friedman1

    from tribblefis import gt2_refine, it2_refine
    from tribblefis.gt2_classifier import GT2TribbleClassifier
    from tribblefis.gt2_regressor import GT2TribbleRegressor
    from tribblefis.it2_classifier import IT2TribbleClassifier
    from tribblefis.it2_regressor import IT2TribbleRegressor

    @contextmanager
    def rebuild_only(module):
        with mock.patch.object(module, "_try_incremental", lambda build: None):
            yield

    Xw, yw = load_wine(return_X_y=True)
    Xw = pd.DataFrame(Xw, columns=[f"x{i}" for i in range(Xw.shape[1])])
    Xf, yf = make_friedman1(n_samples=500, n_features=10, noise=1.0, random_state=42)
    Xf = pd.DataFrame(Xf, columns=[f"x{i}" for i in range(Xf.shape[1])])

    cases = []
    for family, module, cls, kw in (
        ("IT2", it2_refine, IT2TribbleClassifier, {}),
        ("GT2", gt2_refine, GT2TribbleClassifier, {"n_alpha_planes": 5}),
    ):
        clf = cls(top_n=8, n_gaussians=3, random_state=42, **kw).fit(Xw, yw)
        y_idx = np.searchsorted(clf.classes_, yw)
        if family == "IT2":
            fn = lambda clf=clf, y_idx=y_idx: it2_refine.refine_it2_antecedents(
                Xw, y_idx, clf.model_, clf.norms_, n_sweeps=2, verbose=False)
        else:
            fn = lambda clf=clf, y_idx=y_idx: gt2_refine.refine_gt2_antecedents(
                Xw, y_idx, clf.model_, clf.norms_, n_alpha_planes=5, n_sweeps=2, verbose=False)
        cases.append((f"{family} classifier (wine)", module, fn))
    for family, module, cls, kw in (
        ("IT2", it2_refine, IT2TribbleRegressor, {}),
        ("GT2", gt2_refine, GT2TribbleRegressor, {"n_alpha_planes": 5}),
    ):
        reg = cls(top_n=5, n_gaussians=3, random_state=42, **kw).fit(Xf, yf)
        b = reg._base_regressor
        common = dict(order=b.tsk_order, l2_reg=b.l2_reg, basis=b.consequent_basis,
                      cross_pairs=b.cross_pairs_, n_sweeps=2, verbose=False)
        if family == "IT2":
            fn = lambda reg=reg, b=b, common=common: it2_refine.refine_it2_regressor_antecedents(
                Xf, yf, reg.model_, reg.norms_, b.top_features_, **common)
        else:
            fn = lambda reg=reg, b=b, common=common: gt2_refine.refine_gt2_regressor_antecedents(
                Xf, yf, reg.model_, reg.norms_, b.top_features_, n_alpha_planes=5, **common)
        cases.append((f"{family} regressor (friedman1)", module, fn))

    print(f"end-to-end refinement, 2 sweeps, cython={kernel.HAVE_CYTHON_KERNEL}")
    print("| case | rebuild (pre-#146) | incremental | speedup |")
    print("|---|--:|--:|--:|")
    for name, module, fn in cases:
        t_new = _time(fn, repeats)
        with rebuild_only(module):
            t_old = _time(fn, repeats)
        print(f"| {name} | {t_old:.2f} s | {t_new:.2f} s | {t_old / t_new:.1f}x |")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--features", type=int, default=10)
    p.add_argument("--labels", type=int, default=6)
    p.add_argument("--mf", type=int, default=3)
    p.add_argument("--candidates", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--refine", action="store_true",
                   help="time whole refinement runs, incremental vs rebuild")
    args = p.parse_args(argv)
    if args.refine:
        _refine_end_to_end(max(1, min(args.repeats, 2)))
        return 0

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
