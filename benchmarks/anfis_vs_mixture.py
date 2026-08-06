"""Reproduces the measured comparison table in `docs/anfis-design.md`.

Not a pytest test and not part of the checksum-verified `benchmarks.bench`
suite -- this measures *accuracy*, which is expected to differ across numpy/
sklearn versions and platforms, not a wall-clock number that must stay
bit-stable. Run directly: ``python -m benchmarks.anfis_vs_mixture``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tribblefis.anfis import ANFISRegressor
from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
from tribblefis.regression import _mse, _rsquared

_SEEDS = (0, 1, 2)
_MIX_BUCKETS = (6, 10, 15)
_MIX_ORDERS = ("1st", "2nd", "full-2nd")


def make_problem(name: str, seed: int, n: int = 1200) -> tuple[pd.DataFrame, np.ndarray]:
    """Three low-dimensional regression problems, chosen to probe different
    structure: purely radial, additively separable, and multiplicatively
    interacting (the case an implicit per-label rule base cannot represent
    without help -- see the design doc)."""
    rng = np.random.default_rng(seed)
    if name == "sinc2d":
        X = pd.DataFrame({"x0": rng.uniform(-6, 6, n), "x1": rng.uniform(-6, 6, n)})
        r = np.sqrt(X["x0"] ** 2 + X["x1"] ** 2) + 1e-9
        y = (np.sin(r) / r).to_numpy() + rng.normal(0, 0.01, n)
    elif name == "additive3d":
        X = pd.DataFrame({
            "x0": rng.uniform(-3, 3, n), "x1": rng.uniform(-3, 3, n), "x2": rng.uniform(-3, 3, n),
        })
        y = (np.sin(X["x0"]) + 0.5 * X["x1"] ** 2 - 0.3 * X["x2"]).to_numpy() + rng.normal(0, 0.05, n)
    elif name == "interaction2d":
        X = pd.DataFrame({"x0": rng.uniform(-3, 3, n), "x1": rng.uniform(-3, 3, n)})
        y = (X["x0"] * np.cos(X["x1"])).to_numpy() + rng.normal(0, 0.03, n)
    else:
        raise ValueError(name)
    return X, y


def _fit_and_score(estimator, X_tr, y_tr, X_te, y_te) -> tuple[float, float, float]:
    t0 = time.perf_counter()
    estimator.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    pred = estimator.predict(X_te)
    return _rsquared(y_te, pred), _mse(y_te, pred), fit_s


def run_problem(name: str) -> dict:
    anfis_r2s, anfis_fit_s = [], []
    mix_default_r2s = []
    mix_best_r2, mix_best_params = -np.inf, None

    for seed in _SEEDS:
        X, y = make_problem(name, seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)

        r2, _mse_val, fit_s = _fit_and_score(
            ANFISRegressor(n_terms=3, n_epochs=150, learning_rate=0.05, random_state=seed),
            X_tr, y_tr, X_te, y_te,
        )
        anfis_r2s.append(r2)
        anfis_fit_s.append(fit_s)

        r2, _mse_val, _fit_s = _fit_and_score(
            MixtureOfGaussiansFuzzyRegressor(n_output_buckets=6, tsk_order="1st", random_state=seed),
            X_tr, y_tr, X_te, y_te,
        )
        mix_default_r2s.append(r2)

        for buckets in _MIX_BUCKETS:
            for order in _MIX_ORDERS:
                r2, _mse_val, _fit_s = _fit_and_score(
                    MixtureOfGaussiansFuzzyRegressor(
                        n_output_buckets=buckets, tsk_order=order, random_state=seed,
                    ),
                    X_tr, y_tr, X_te, y_te,
                )
                if r2 > mix_best_r2:
                    mix_best_r2, mix_best_params = r2, (buckets, order)

    return {
        "problem": name,
        "anfis_r2_mean": float(np.mean(anfis_r2s)),
        "anfis_fit_s_mean": float(np.mean(anfis_fit_s)),
        "mix_r2_default_mean": float(np.mean(mix_default_r2s)),
        "mix_r2_best": float(mix_best_r2),
        "mix_best_params": mix_best_params,
    }


def main() -> None:
    rows = [run_problem(name) for name in ("sinc2d", "additive3d", "interaction2d")]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
