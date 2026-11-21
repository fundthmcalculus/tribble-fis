"""Reproduces the engine and calibration tables in `docs/interaction-detection.md`.

Three measurements, all on synthetic XOR (``y = [x0 * x1 > 0]``: both marginals identical across
classes, so no per-feature score can see the pair):

  1. SPEED / DETECTION  the original ``engine="product"`` pair lift vs the vectorised
     ``engine="gaussian"``, at 11 / 30 / 60 features (n = 20,000): rank of the true pair and seconds.
  2. FALSE ALARMS       `find_cross_pairs` on data with NO interaction (pure noise; an additive
     signal), counting datasets where it still returns a pair. The permutation null predicts about
     1 / (n_permutations + 1).
  3. POWER              the same detector on XOR, counting datasets where the true pair is first.

Timings are host-specific (ratios transfer, seconds do not); the detection and false-alarm counts
are not. Run directly: ``python -m benchmarks.interaction_engine_comparison [--repeats 100]``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time

import numpy as np
import pandas as pd

import tribblefis.gauss_math as gm
from tribblefis.feature_selection import find_cross_pairs


def _data(kind: str, seed: int, n: int, m: int) -> tuple[pd.DataFrame, pd.Series]:
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.normal(size=(n, m)), columns=[f"x{i}" for i in range(m)])
    if kind == "noise":
        y = r.integers(0, 2, n)
    elif kind == "additive":
        y = (X.x2 + 0.3 * r.normal(size=n) > 0).astype(int)
    elif kind == "xor":
        y = (X.x0 * X.x1 > 0).astype(int)
    else:
        raise ValueError(kind)
    return X, pd.Series(y, index=X.index)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):  # the ranking prints to stdout
        return fn(*a, **k)


def speed_and_detection() -> None:
    print("1. XOR pair rank and seconds, n = 20,000")
    print(f"{'features':>8} {'pairs':>6} | {'product (wasserstein)':>24} | {'gaussian engine':>20}")
    for m in (11, 30, 60):
        X, y = _data("xor", 0, 20000, m)
        fd = _quiet(gm.calculate_gaussian_correlation, X, y, method="wasserstein")
        cells = []
        for engine in ("product", "gaussian"):
            t = time.perf_counter()
            pairs = _quiet(gm.calculate_interaction_scores, X, y, fd, engine=engine, max_pairs=10**6)
            dt = time.perf_counter() - t
            rank = [frozenset((a, b)) for a, b, _ in pairs].index(frozenset(("x0", "x1"))) + 1
            cells.append(f"rank {rank:>2}, {dt:6.2f} s")
        print(f"{m:>8} {m * (m - 1) // 2:>6} | {cells[0]:>24} | {cells[1]:>20}")


def calibration(repeats: int) -> None:
    print(f"\n2. False alarms (no interaction), {repeats} datasets per row")
    for kind in ("noise", "additive"):
        for n, m in ((500, 11), (2000, 11), (2000, 50)):
            hits = sum(bool(_quiet(find_cross_pairs, *_data(kind, s, n, m), random_state=s)) for s in range(repeats))
            print(f"   {kind:9s} n={n:5d} features={m:3d}: {hits:3d}/{repeats}")
    print(f"\n3. Power on XOR, {repeats} datasets per row (true pair returned first)")
    for n, m in ((300, 11), (1000, 11), (3000, 50)):
        ok = 0
        for s in range(repeats):
            p = _quiet(find_cross_pairs, *_data("xor", s, n, m), random_state=s)
            ok += bool(p) and set(p[0]) == {"x0", "x1"}
        print(f"   n={n:5d} features={m:3d}: {ok:3d}/{repeats}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=100)
    args = ap.parse_args()
    speed_and_detection()
    calibration(args.repeats)
