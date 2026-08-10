"""Reproduces the measured comparison table in `docs/interaction-detection.md`.

Not a checksum-verified `benchmarks.bench` workload -- this measures
*accuracy* (expected to move slightly across numpy/sklearn versions and
platforms), not a wall-clock number that must stay bit-stable. Run directly:
``python -m benchmarks.interaction_detection_comparison``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.regression import _rsquared

_SEEDS = (0, 1, 2)


def make_problem(name: str, seed: int, n: int = 1500) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Two problems where a strict feature budget drops half of a real
    interaction unless something rescues it -- and the top_n budget that
    would do the dropping.

    - ``pure_interaction``: y depends on x0*x1 alone; three unrelated noise
      columns. top_n=1 keeps only one of {x0, x1} without detection.
    - ``interaction_plus_dominant``: a strong additive feature (x2) plus a
      secondary x0*x1 interaction. top_n=2 keeps x2 and the stronger of
      {x0, x1} without detection -- the interaction is invisible either way,
      because *both* halves are needed for it to exist.
    """
    rng = np.random.default_rng(seed)
    if name == "pure_interaction":
        x0 = rng.uniform(-1, 1, n)
        x1 = rng.uniform(-1, 1, n)
        noise = pd.DataFrame({f"noise{i}": rng.uniform(-1, 1, n) for i in range(3)})
        y = x0 * x1 + rng.normal(0, 0.02, n)
        X = pd.concat([pd.DataFrame({"x0": x0, "x1": x1}), noise], axis=1)
        return X, y, 1
    if name == "interaction_plus_dominant":
        x0 = rng.uniform(-1, 1, n)
        x1 = rng.uniform(-1, 1, n)
        x2 = rng.uniform(-1, 1, n)
        noise = pd.DataFrame({f"noise{i}": rng.uniform(-1, 1, n) for i in range(3)})
        y = 2.0 * x2 + 0.8 * (x0 * x1) + rng.normal(0, 0.02, n)
        X = pd.concat([pd.DataFrame({"x0": x0, "x1": x1, "x2": x2}), noise], axis=1)
        return X, y, 2
    raise ValueError(name)


def run_problem(name: str) -> dict:
    r2_without, r2_with = [], []
    for seed in _SEEDS:
        X, y, top_n = make_problem(name, seed)

        without = TribbleRegressor(
            tsk_order="full-2nd", top_n=top_n, n_output_buckets=6, random_state=seed,
        )
        without.fit(X, y)
        r2_without.append(_rsquared(y, without.predict(X)))

        with_detect = TribbleRegressor(
            tsk_order="full-2nd", top_n=top_n, n_output_buckets=6, random_state=seed,
            detect_interactions=True, select_interactions=True,
        )
        with_detect.fit(X, y)
        r2_with.append(_rsquared(y, with_detect.predict(X)))

    return {
        "problem": name,
        "top_n": top_n,
        "r2_without_mean": float(np.mean(r2_without)),
        "r2_with_mean": float(np.mean(r2_with)),
    }


def main() -> None:
    rows = [run_problem(name) for name in ("pure_interaction", "interaction_plus_dominant")]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
