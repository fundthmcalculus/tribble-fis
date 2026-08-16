"""T1 vs IT2 vs GT2: construction time vs. accuracy/R^2, Pareto-style.

Fits TribbleClassifier/IT2TribbleClassifier/GT2TribbleClassifier on wine and
TribbleRegressor/IT2TribbleRegressor/GT2TribbleRegressor on make_friedman1,
sweeping n_gaussians (all three families) and n_alpha_planes (GT2 only) to
trace a cost/performance curve per family instead of one point each.

Backs docs/t1-it2-gt2-tradeoff.md -- answers the "is GT2 worth its cost on
real data?" question docs/gt2-evaluation.md left open. Run with:

    python -m benchmarks.t1_it2_gt2_tradeoff
    python -m benchmarks.t1_it2_gt2_tradeoff -o benchmarks/results/mine.json
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

from sklearn.datasets import load_wine, make_friedman1
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.gt2_classifier import GT2TribbleClassifier
from tribblefis.gt2_regressor import GT2TribbleRegressor
from tribblefis.it2_classifier import IT2TribbleClassifier
from tribblefis.it2_regressor import IT2TribbleRegressor

RANDOM_STATE = 42
N_GAUSSIANS = (1, 2, 3, 4, 5)
N_ALPHA_PLANES = (1, 2, 3, 5, 8, 10)
GT2_FIXED_N_GAUSSIANS = 3


def _fit_time_and_score(model, X_train, y_train, X_test, y_test, score_fn):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    return fit_time, score_fn(y_test, model.predict(X_test))


def _record(records, task, family, config, n_gaussians, extra, fit_time, performance):
    records.append(dict(task=task, family=family, config=config, n_gaussians=n_gaussians,
                         extra=extra, fit_time=fit_time, performance=performance))


def run_classification(records):
    """Wine: 178 rows, 13 features, 3 classes."""
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

    for ng in N_GAUSSIANS:
        clf = TribbleClassifier(n_gaussians=ng, top_n=8, random_state=RANDOM_STATE)
        t, acc = _fit_time_and_score(clf, X_train, y_train, X_test, y_test, accuracy_score)
        _record(records, "classification", "T1", f"n_gaussians={ng}", ng, None, t, acc)

    for ng in N_GAUSSIANS:
        clf = IT2TribbleClassifier(n_gaussians=ng, top_n=8, uncertainty_width=0.5,
                                    random_state=RANDOM_STATE)
        t, acc = _fit_time_and_score(clf, X_train, y_train, X_test, y_test, accuracy_score)
        _record(records, "classification", "IT2", f"n_gaussians={ng}", ng, None, t, acc)

    # One refined point -- refinement is the expensive knob, so one is enough
    # to show the trade (see docs/t1-it2-gt2-tradeoff.md).
    clf = IT2TribbleClassifier(n_gaussians=2, top_n=8, uncertainty_width=0.5,
                                refine_it2=True, refine_it2_n_sweeps=2, random_state=RANDOM_STATE)
    t, acc = _fit_time_and_score(clf, X_train, y_train, X_test, y_test, accuracy_score)
    _record(records, "classification", "IT2+refine", "n_gaussians=2", 2, None, t, acc)

    for ng in N_GAUSSIANS:
        clf = GT2TribbleClassifier(n_gaussians=ng, top_n=8, uncertainty_width=0.5,
                                    n_alpha_planes=5, random_state=RANDOM_STATE)
        t, acc = _fit_time_and_score(clf, X_train, y_train, X_test, y_test, accuracy_score)
        _record(records, "classification", "GT2", f"n_gaussians={ng},K=5", ng, 5, t, acc)

    for k in N_ALPHA_PLANES:
        if k == 5:
            continue  # already covered above
        clf = GT2TribbleClassifier(n_gaussians=GT2_FIXED_N_GAUSSIANS, top_n=8,
                                    uncertainty_width=0.5, n_alpha_planes=k,
                                    random_state=RANDOM_STATE)
        t, acc = _fit_time_and_score(clf, X_train, y_train, X_test, y_test, accuracy_score)
        _record(records, "classification", "GT2",
                 f"n_gaussians={GT2_FIXED_N_GAUSSIANS},K={k}", GT2_FIXED_N_GAUSSIANS, k, t, acc)


def run_regression(records):
    """make_friedman1: 500 rows, 10 features, 5 informative, nonlinear."""
    X, y = make_friedman1(n_samples=500, n_features=10, noise=1.0, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                          random_state=RANDOM_STATE)

    for ng in N_GAUSSIANS:
        reg = TribbleRegressor(n_gaussians=ng, top_n=5, tsk_order="1st", random_state=RANDOM_STATE)
        t, r2 = _fit_time_and_score(reg, X_train, y_train, X_test, y_test, r2_score)
        _record(records, "regression", "T1", f"n_gaussians={ng}", ng, None, t, r2)

    for ng in N_GAUSSIANS:
        reg = IT2TribbleRegressor(n_gaussians=ng, top_n=5, uncertainty_width=0.5,
                                   km_iterations=10, random_state=RANDOM_STATE)
        t, r2 = _fit_time_and_score(reg, X_train, y_train, X_test, y_test, r2_score)
        _record(records, "regression", "IT2", f"n_gaussians={ng}", ng, None, t, r2)

    reg = IT2TribbleRegressor(n_gaussians=2, top_n=5, uncertainty_width=0.5, km_iterations=10,
                               refine_it2=True, refine_it2_n_sweeps=2, random_state=RANDOM_STATE)
    t, r2 = _fit_time_and_score(reg, X_train, y_train, X_test, y_test, r2_score)
    _record(records, "regression", "IT2+refine", "n_gaussians=2", 2, None, t, r2)

    for ng in N_GAUSSIANS:
        reg = GT2TribbleRegressor(n_gaussians=ng, top_n=5, uncertainty_width=0.5,
                                   n_alpha_planes=5, km_iterations=10, random_state=RANDOM_STATE)
        t, r2 = _fit_time_and_score(reg, X_train, y_train, X_test, y_test, r2_score)
        _record(records, "regression", "GT2", f"n_gaussians={ng},K=5", ng, 5, t, r2)

    for k in N_ALPHA_PLANES:
        if k == 5:
            continue
        reg = GT2TribbleRegressor(n_gaussians=GT2_FIXED_N_GAUSSIANS, top_n=5,
                                   uncertainty_width=0.5, n_alpha_planes=k, km_iterations=10,
                                   random_state=RANDOM_STATE)
        t, r2 = _fit_time_and_score(reg, X_train, y_train, X_test, y_test, r2_score)
        _record(records, "regression", "GT2",
                 f"n_gaussians={GT2_FIXED_N_GAUSSIANS},K={k}", GT2_FIXED_N_GAUSSIANS, k, t, r2)


def _flag_pareto(records):
    """Mark non-dominated points (min fit_time, max performance) within each task."""
    for task in {r["task"] for r in records}:
        subset = [r for r in records if r["task"] == task]
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
        run_classification(records)
        run_regression(records)
    _flag_pareto(records)

    for r in sorted(records, key=lambda r: (r["task"], r["fit_time"])):
        star = " *pareto*" if r["pareto"] else ""
        print(f"[{r['task']}] {r['family']:12s} {r['config']:22s} "
              f"time={r['fit_time']:.4f}s perf={r['performance']:.4f}{star}")

    if args.output:
        args.output.write_text(json.dumps(records, indent=2))
        print(f"\nWrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
