"""Basic, one-host measurements behind the numbers in `docs/grow-to-target.md` (#242, #247, #248).

NOT runs of record: one machine, a few seeds, synthetic data, whatever BLAS/compiler this host has. They exist so
the claims in the docs that were once guesses -- or hedges -- have a reproducible measurement under them. Ratios
transfer across machines; seconds do not.

  m1  regressor growth at tsk_order "1st" vs "full-2nd" (cost; the cross terms grow quadratically)
  m2  does the compiled kernel change growth? growth vs refit scores and the cost ratio, with the kernel flag shown
      (run it twice -- once with the extension built, once without -- to compare)
  m3  growth vs refit sweep for each membership builder (the slow EM builders are where growth saves the most)
  m4  how optimistic is selecting a size on one held-out split? validation score at the chosen size vs an untouched test set

    python -m benchmarks.grow_basic_measurements [--only m1,m3] [--m4-seeds 20]

About 3 minutes in total. Every comparison asserts nothing: the point is to print what happened.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split

from tribblefis import kernel
from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.gaussian_regressor import TribbleRegressor

warnings.filterwarnings("ignore")


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):  # the ranking prints to stdout
        return fn(*a, **k)


def _f1(a, b):
    return f1_score(a, b, average="macro", zero_division=0)


def _grow(est, Xtr, ytr, val, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        _quiet(est.fit_growing, Xtr, ytr, validation=val, refit=False, **kw)
        return time.perf_counter() - t


def m1_full2nd() -> None:
    print(
        "\nm1  regressor growth, 1st vs full-2nd (cross terms are quadratic in the features)"
    )
    X, y = make_regression(
        n_samples=2500, n_features=14, n_informative=8, noise=10.0, random_state=0
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(14)])
    y = pd.Series(y)
    Xtr, Xva, ytr, yva = (
        a.reset_index(drop=True)
        for a in train_test_split(X, y, test_size=0.25, random_state=0)
    )
    print("  10 sizes, growing vs a refit sweep (scores compared too):")
    for order in ("1st", "full-2nd"):
        kw = dict(
            correlation_threshold=0.0,
            tsk_order=order,
            n_output_buckets=3,
            random_state=42,
        )
        t = time.perf_counter()
        sweep = []
        for k in range(1, 11):
            m = TribbleRegressor(top_n=k, **kw)
            _quiet(m.fit, Xtr, ytr)
            sweep.append(r2_score(yva, m.predict(Xva)))
        sweep_s = time.perf_counter() - t
        g = TribbleRegressor(**kw)
        grow_s = _grow(g, Xtr, ytr, (Xva, yva), target=1.0, max_n=10)
        h = g.growth_history_
        print(
            f"   {order:9s} growing {grow_s:5.2f}s vs sweep {sweep_s:5.2f}s ({sweep_s / grow_s:.1f}x) | "
            f"per-step s at k=1/5/10: {h.step_seconds.iloc[0]:.3f} {h.step_seconds.iloc[4]:.3f} {h.step_seconds.iloc[9]:.3f} | "
            f"max score diff vs refit {np.max(np.abs(h.score.values - np.array(sweep))):.1e}"
        )
    print("  growth only, 24 features: per-step seconds at k=1/6/12/18/24")
    X, y = make_regression(
        n_samples=10000, n_features=24, n_informative=10, noise=10.0, random_state=0
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(24)])
    y = pd.Series(y)
    for n_rows in (2500, 10000):
        Xtr, Xva, ytr, yva = (
            a.reset_index(drop=True)
            for a in train_test_split(
                X.iloc[:n_rows], y.iloc[:n_rows], test_size=0.25, random_state=0
            )
        )
        for order in ("1st", "full-2nd"):
            g = TribbleRegressor(
                correlation_threshold=0.0,
                tsk_order=order,
                n_output_buckets=3,
                random_state=42,
            )
            total = _grow(g, Xtr, ytr, (Xva, yva), target=1.0)
            s = g.growth_history_.step_seconds
            steps = " ".join(f"{s.iloc[i - 1]:.3f}" for i in (1, 6, 12, 18, 24))
            print(f"   n_train={len(Xtr):5d} {order:9s} total {total:5.2f}s | {steps}")


def m2_compiled() -> None:
    print(f"\nm2  kernel check: HAVE_CYTHON_KERNEL={kernel.HAVE_CYTHON_KERNEL}")
    diffs, ratios = [], []
    n_val = 0
    for seed in range(5):
        X, y = make_classification(
            n_samples=3000,
            n_features=14,
            n_informative=7,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            class_sep=1.1,
            random_state=seed,
        )
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(14)])
        y = pd.Series(y)
        Xtr, Xva, ytr, yva = (
            a.reset_index(drop=True)
            for a in train_test_split(
                X, y, test_size=0.25, random_state=seed, stratify=y
            )
        )
        n_val = len(yva)
        kw = dict(correlation_threshold=0.0, random_state=42)
        t = time.perf_counter()
        oracle = []
        for k in range(1, 15):
            m = TribbleClassifier(top_n=k, **kw)
            _quiet(m.fit, Xtr, ytr)
            oracle.append(_f1(yva, m.predict(Xva)))
        sweep_s = time.perf_counter() - t
        g = TribbleClassifier(**kw)
        grow_s = _grow(g, Xtr, ytr, (Xva, yva), target=1.0)
        diffs.append(
            float(np.max(np.abs(g.growth_history_.score.values - np.array(oracle))))
        )
        ratios.append(sweep_s / grow_s)
    print(
        f"   5 seeds x 14 sizes ({n_val} held-out rows): max |growth - refit| per seed {[f'{d:.1e}' for d in diffs]} | "
        f"cost ratio per seed {[round(r, 1) for r in ratios]} (mean {np.mean(ratios):.1f}x)"
    )


def m3_builders() -> None:
    print(
        "\nm3  growth vs refit sweep per membership builder (n_train=300, 4 features, sizes 1..4)"
    )
    X, y = make_classification(
        n_samples=400,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=1,
    )
    X = pd.DataFrame(X, columns=list("abcd"))
    y = pd.Series(y)
    Xtr, Xva, ytr, yva = (
        a.reset_index(drop=True)
        for a in train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
    )
    for mf, tm in (
        ("gaussian", "fast"),
        ("trap", "fast"),
        ("triangular", "fast"),
        ("trap", "em"),
    ):
        kw = dict(
            member_function=mf,
            trapz_method=tm,
            correlation_threshold=0.0,
            random_state=42,
        )
        t = time.perf_counter()
        sweep = []
        for k in range(1, 5):
            m = TribbleClassifier(top_n=k, **kw)
            _quiet(m.fit, Xtr, ytr)
            sweep.append(_f1(yva, m.predict(Xva)))
        sweep_s = time.perf_counter() - t
        g = TribbleClassifier(**kw)
        grow_s = _grow(g, Xtr, ytr, (Xva, yva), target=1.0)
        print(
            f"   {mf:10s} {tm:4s} growing {grow_s:6.2f}s vs sweep {sweep_s:6.2f}s ({sweep_s / grow_s:.1f}x, saves {sweep_s - grow_s:5.1f}s) | "
            f"max score diff {np.max(np.abs(g.growth_history_.score.values - np.array(sweep))):.1e}"
        )


def m4_optimism(seeds: int = 20) -> None:
    print(
        f"\nm4  selection optimism: validation score at the chosen size minus an untouched test set ({seeds} seeds)"
    )
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=8,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=0,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    y = pd.Series(y)
    for name, target in (
        ("reachable target 0.75", 0.75),
        ("unreachable (best size)", 1.0),
    ):
        d_same, d_refit, ns, reached = [], [], [], 0
        for seed in range(seeds):
            Xtr, Xte, ytr, yte = (
                a.reset_index(drop=True)
                for a in train_test_split(
                    X, y, test_size=0.3, random_state=seed, stratify=y
                )
            )
            for refit in (False, True):
                g = TribbleClassifier(correlation_threshold=0.0, random_state=seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _quiet(
                        g.fit_growing,
                        Xtr,
                        ytr,
                        target=target,
                        validation=0.2,
                        refit=refit,
                    )
                val = g.growth_history_["score"].iloc[g.n_features_selected_ - 1]
                (d_refit if refit else d_same).append(val - _f1(yte, g.predict(Xte)))
            ns.append(g.n_features_selected_)
            reached += g.target_reached_
        se = np.std(d_same) / np.sqrt(seeds)
        print(
            f"   {name:24s} val - test (same model) {np.mean(d_same):+.4f} +/- {np.std(d_same):.4f} (SE {se:.4f}) | "
            f"val - test (refit on all) {np.mean(d_refit):+.4f} | mean n {np.mean(ns):.1f} | target reached {reached}/{seeds}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only", default="m1,m2,m3,m4", help="comma-separated subset of m1,m2,m3,m4"
    )
    ap.add_argument("--m4-seeds", type=int, default=20)
    args = ap.parse_args()
    chosen = {s.strip() for s in args.only.split(",")}
    print(
        "basic measurements -- one host, not runs of record; ratios transfer, seconds do not"
    )
    if "m1" in chosen:
        m1_full2nd()
    if "m2" in chosen:
        m2_compiled()
    if "m3" in chosen:
        m3_builders()
    if "m4" in chosen:
        m4_optimism(args.m4_seeds)
