"""T1 vs IT2 vs GT2: construction time vs. accuracy/R^2, Pareto-style.

Fits TribbleClassifier/IT2TribbleClassifier/GT2TribbleClassifier on
wine/breast_cancer/digits and TribbleRegressor/IT2TribbleRegressor/
GT2TribbleRegressor on friedman1/diabetes/make_regression, sweeping
n_gaussians (all three families) and n_alpha_planes (GT2 only) to trace a
cost/performance curve per family per dataset instead of one point each.

Also covers what the first pass (#141) left open:
- **seed robustness**: T1/IT2/GT2 repeated over 5 seeds on the flagship
  wine/friedman1 datasets (refine excluded -- its cost is already
  characterized as a single, expensive point; see below).
- **calibration**: regression gets empirical coverage + mean relative width
  of `predict_intervals()`; classification has no coverage analogue (its
  `predict_intervals()` returns per-class firing bounds, not a label
  interval), so it gets mean predicted-class interval width split by
  correct vs. incorrect -- a calibrated model should be wider on its errors.

`refine_it2`/`refine_gt2` only run on wine/friedman1 (the datasets #141
already measured refinement's ~80-1500x cost on) -- not repeated per new
dataset or seed, to keep total runtime bounded.

Backs docs/t1-it2-gt2-tradeoff.md. Run with:

    python -m benchmarks.t1_it2_gt2_tradeoff
    python -m benchmarks.t1_it2_gt2_tradeoff -o benchmarks/results/mine.json
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import (
    load_breast_cancer, load_diabetes, load_digits, load_wine,
    make_friedman1, make_regression,
)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.gt2_classifier import GT2TribbleClassifier
from tribblefis.gt2_regressor import GT2TribbleRegressor
from tribblefis.it2_classifier import IT2TribbleClassifier
from tribblefis.it2_regressor import IT2TribbleRegressor

DEFAULT_SEED = 42
SEED_ROBUSTNESS_SEEDS = (42, 7, 123, 2024, 99)
N_GAUSSIANS = (1, 2, 3, 4, 5)
N_ALPHA_PLANES = (1, 2, 3, 5, 8, 10)
GT2_FIXED_N_GAUSSIANS = 3
SEED_ROBUSTNESS_N_GAUSSIANS = 2

# name -> (loader() -> (X, y), top_n)
CLASSIFICATION_DATASETS = {
    "wine": (lambda: load_wine(return_X_y=True), 8),
    "breast_cancer": (lambda: load_breast_cancer(return_X_y=True), 10),
    "digits": (lambda: load_digits(return_X_y=True), 12),
}
REGRESSION_DATASETS = {
    "friedman1": (lambda: make_friedman1(n_samples=500, n_features=10, noise=1.0,
                                          random_state=DEFAULT_SEED), 5),
    "diabetes": (lambda: load_diabetes(return_X_y=True), 6),
    "make_regression": (lambda: make_regression(n_samples=500, n_features=10, n_informative=5,
                                                  noise=10.0, random_state=DEFAULT_SEED), 5),
}
# refine_it2/refine_gt2 only run on these -- see module docstring.
REFINE_DATASETS = {"classification": "wine", "regression": "friedman1"}


def _split(X, y, seed):
    stratify = y if np.issubdtype(np.asarray(y).dtype, np.integer) else None
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=stratify)


def _fit_time_and_score(model, X_train, y_train, X_test, y_test, score_fn):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    y_pred = model.predict(X_test)
    return fit_time, score_fn(y_test, y_pred), y_pred


def _regression_calibration(model, X_test, y_test):
    """Empirical coverage and mean relative width of predict_intervals()."""
    if not hasattr(model, "predict_intervals"):
        return None
    y_lower, y_upper = model.predict_intervals(X_test)
    y_test = np.asarray(y_test)
    coverage = float(np.mean((y_test >= y_lower) & (y_test <= y_upper)))
    y_range = float(y_test.max() - y_test.min()) or 1.0
    mean_width = float(np.mean(y_upper - y_lower)) / y_range
    return dict(coverage=coverage, mean_relative_width=mean_width)


def _classification_calibration(model, X_test, y_test, y_pred):
    """Mean predicted-class firing-interval width, split by correct/incorrect.

    A model whose interval width tracks its own error rate is doing
    something informative with its uncertainty; one where both groups look
    the same is just carrying a constant-width footprint around.
    """
    if not hasattr(model, "predict_intervals"):
        return None
    upper, lower = model.predict_intervals(X_test)
    classes = list(model.classes_)
    pred_idx = np.array([classes.index(p) for p in y_pred])
    rows = np.arange(len(y_pred))
    width = upper[rows, pred_idx] - lower[rows, pred_idx]
    correct = np.asarray(y_pred) == np.asarray(y_test)
    return dict(
        mean_width_correct=float(np.mean(width[correct])) if correct.any() else None,
        mean_width_incorrect=float(np.mean(width[~correct])) if (~correct).any() else None,
    )


def _record(records, **kwargs):
    records.append(kwargs)


def run_classification_sweep(records, dataset_name):
    """Full n_gaussians (+ GT2 n_alpha_planes) sweep on one classification dataset."""
    loader, top_n = CLASSIFICATION_DATASETS[dataset_name]
    X, y = loader()
    X_train, X_test, y_train, y_test = _split(X, y, DEFAULT_SEED)

    def add(family, config, ng, extra, model):
        t, acc, y_pred = _fit_time_and_score(model, X_train, y_train, X_test, y_test, accuracy_score)
        calib = _classification_calibration(model, X_test, y_test, y_pred)
        _record(records, task="classification", dataset=dataset_name, family=family, config=config,
                 n_gaussians=ng, extra=extra, fit_time=t, performance=acc, calibration=calib)

    for ng in N_GAUSSIANS:
        add("T1", f"n_gaussians={ng}", ng, None,
            TribbleClassifier(n_gaussians=ng, top_n=top_n, random_state=DEFAULT_SEED))
        add("IT2", f"n_gaussians={ng}", ng, None,
            IT2TribbleClassifier(n_gaussians=ng, top_n=top_n, uncertainty_width=0.5,
                                  random_state=DEFAULT_SEED))
        add("GT2", f"n_gaussians={ng},K=5", ng, 5,
            GT2TribbleClassifier(n_gaussians=ng, top_n=top_n, uncertainty_width=0.5,
                                  n_alpha_planes=5, random_state=DEFAULT_SEED))

    for k in N_ALPHA_PLANES:
        if k == 5:
            continue  # already covered above
        add("GT2", f"n_gaussians={GT2_FIXED_N_GAUSSIANS},K={k}", GT2_FIXED_N_GAUSSIANS, k,
            GT2TribbleClassifier(n_gaussians=GT2_FIXED_N_GAUSSIANS, top_n=top_n,
                                  uncertainty_width=0.5, n_alpha_planes=k, random_state=DEFAULT_SEED))

    if REFINE_DATASETS["classification"] == dataset_name:
        add("IT2+refine", "n_gaussians=2", 2, None,
            IT2TribbleClassifier(n_gaussians=2, top_n=top_n, uncertainty_width=0.5,
                                  refine_it2=True, refine_it2_n_sweeps=2, random_state=DEFAULT_SEED))


def run_regression_sweep(records, dataset_name):
    """Full n_gaussians (+ GT2 n_alpha_planes) sweep on one regression dataset."""
    loader, top_n = REGRESSION_DATASETS[dataset_name]
    X, y = loader()
    X_train, X_test, y_train, y_test = _split(X, y, DEFAULT_SEED)

    def add(family, config, ng, extra, model):
        t, r2, y_pred = _fit_time_and_score(model, X_train, y_train, X_test, y_test, r2_score)
        calib = _regression_calibration(model, X_test, y_test)
        _record(records, task="regression", dataset=dataset_name, family=family, config=config,
                 n_gaussians=ng, extra=extra, fit_time=t, performance=r2, calibration=calib)

    for ng in N_GAUSSIANS:
        add("T1", f"n_gaussians={ng}", ng, None,
            TribbleRegressor(n_gaussians=ng, top_n=top_n, tsk_order="1st", random_state=DEFAULT_SEED))
        add("IT2", f"n_gaussians={ng}", ng, None,
            IT2TribbleRegressor(n_gaussians=ng, top_n=top_n, uncertainty_width=0.5,
                                 km_iterations=10, random_state=DEFAULT_SEED))
        add("GT2", f"n_gaussians={ng},K=5", ng, 5,
            GT2TribbleRegressor(n_gaussians=ng, top_n=top_n, uncertainty_width=0.5,
                                 n_alpha_planes=5, km_iterations=10, random_state=DEFAULT_SEED))

    for k in N_ALPHA_PLANES:
        if k == 5:
            continue
        add("GT2", f"n_gaussians={GT2_FIXED_N_GAUSSIANS},K={k}", GT2_FIXED_N_GAUSSIANS, k,
            GT2TribbleRegressor(n_gaussians=GT2_FIXED_N_GAUSSIANS, top_n=top_n, uncertainty_width=0.5,
                                 n_alpha_planes=k, km_iterations=10, random_state=DEFAULT_SEED))

    if REFINE_DATASETS["regression"] == dataset_name:
        add("IT2+refine", "n_gaussians=2", 2, None,
            IT2TribbleRegressor(n_gaussians=2, top_n=top_n, uncertainty_width=0.5, km_iterations=10,
                                 refine_it2=True, refine_it2_n_sweeps=2, random_state=DEFAULT_SEED))


def run_seed_robustness(records):
    """T1/IT2/GT2 (no refine) over several seeds, on the flagship datasets only."""
    clf_loader, clf_top_n = CLASSIFICATION_DATASETS["wine"]
    reg_loader, reg_top_n = REGRESSION_DATASETS["friedman1"]

    for seed in SEED_ROBUSTNESS_SEEDS:
        X, y = clf_loader()
        X_train, X_test, y_train, y_test = _split(X, y, seed)
        for family, model in (
            ("T1", TribbleClassifier(n_gaussians=SEED_ROBUSTNESS_N_GAUSSIANS, top_n=clf_top_n,
                                      random_state=seed)),
            ("IT2", IT2TribbleClassifier(n_gaussians=SEED_ROBUSTNESS_N_GAUSSIANS, top_n=clf_top_n,
                                          uncertainty_width=0.5, random_state=seed)),
            ("GT2", GT2TribbleClassifier(n_gaussians=SEED_ROBUSTNESS_N_GAUSSIANS, top_n=clf_top_n,
                                          uncertainty_width=0.5, n_alpha_planes=5, random_state=seed)),
        ):
            t, acc, _ = _fit_time_and_score(model, X_train, y_train, X_test, y_test, accuracy_score)
            _record(records, task="seed_robustness_classification", dataset="wine", family=family,
                     config=f"n_gaussians={SEED_ROBUSTNESS_N_GAUSSIANS}", seed=seed,
                     fit_time=t, performance=acc)

        X, y = reg_loader()
        X_train, X_test, y_train, y_test = _split(X, y, seed)
        for family, model in (
            ("T1", TribbleRegressor(n_gaussians=SEED_ROBUSTNESS_N_GAUSSIANS, top_n=reg_top_n,
                                     tsk_order="1st", random_state=seed)),
            ("IT2", IT2TribbleRegressor(n_gaussians=SEED_ROBUSTNESS_N_GAUSSIANS, top_n=reg_top_n,
                                         uncertainty_width=0.5, km_iterations=10, random_state=seed)),
            ("GT2", GT2TribbleRegressor(n_gaussians=SEED_ROBUSTNESS_N_GAUSSIANS, top_n=reg_top_n,
                                         uncertainty_width=0.5, n_alpha_planes=5, km_iterations=10,
                                         random_state=seed)),
        ):
            t, r2, _ = _fit_time_and_score(model, X_train, y_train, X_test, y_test, r2_score)
            _record(records, task="seed_robustness_regression", dataset="friedman1", family=family,
                     config=f"n_gaussians={SEED_ROBUSTNESS_N_GAUSSIANS}", seed=seed,
                     fit_time=t, performance=r2)


def _flag_pareto(records):
    """Mark non-dominated points (min fit_time, max performance) within each (task, dataset)."""
    groups = {(r["task"], r.get("dataset")) for r in records}
    for key in groups:
        subset = [r for r in records if (r["task"], r.get("dataset")) == key]
        for r in subset:
            r["pareto"] = not any(
                s is not r and s["fit_time"] <= r["fit_time"] and s["performance"] >= r["performance"]
                and (s["fit_time"] < r["fit_time"] or s["performance"] > r["performance"])
                for s in subset
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="write results as JSON to this path")
    args = p.parse_args(argv)

    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in CLASSIFICATION_DATASETS:
            run_classification_sweep(records, name)
            if args.output:
                args.output.write_text(json.dumps(records, indent=2))
        for name in REGRESSION_DATASETS:
            run_regression_sweep(records, name)
            if args.output:
                args.output.write_text(json.dumps(records, indent=2))
        run_seed_robustness(records)
    _flag_pareto(records)

    for r in sorted(records, key=lambda r: (r["task"], r.get("dataset", ""), r["fit_time"])):
        star = " *pareto*" if r.get("pareto") else ""
        seed = f" seed={r['seed']}" if "seed" in r else ""
        print(f"[{r['task']}/{r.get('dataset')}] {r['family']:12s} {r['config']:22s}{seed} "
              f"time={r['fit_time']:.4f}s perf={r['performance']:.4f}{star}")

    if args.output:
        args.output.write_text(json.dumps(records, indent=2))
        print(f"\nWrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
