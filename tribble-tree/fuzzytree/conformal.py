"""Split-conformal prediction intervals for the tree regressors (#268).

Phase 0 of `docs/hierarchical-it2-design.md` (#147): calibrated intervals on
`DeconstructedHierarchicalRegressor` and `HierarchicalFuzzyExpertsRegressor`
without any type-2 machinery. The IT2/GT2 work (#149, #145) found that the
split-conformal margin does all of the calibration and the footprint of
uncertainty adds next to nothing, so the hierarchy gets the margin directly.

Mechanism (Vovk, Gammerman & Shafer 2005; the same one
`tribblefis.regression.conformal_calibration_margin` implements for the
IT2/GT2 regressors): hold out a calibration split *before* anything is fit,
fit on the rest, and take the ``ceil((n+1)(1-alpha))/n`` quantile of the
calibration split's absolute residuals as a symmetric margin. Passing the
point prediction as both bounds to `conformal_calibration_margin` reduces its
nonconformity score to exactly ``|y - y_hat|``.

The guarantee is marginal coverage ``>= 1 - alpha`` on exchangeable data, and
it holds only if the calibration rows never influenced the model -- not its
topology choice, not its gate structure, not EM. Each estimator enforces that
in its own ``fit``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.model_selection import train_test_split

from tribblefis.regression import conformal_calibration_margin


def validate(alpha: float, frac: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"conformal_alpha must be in (0, 1), got {alpha!r}")
    if not 0.0 < frac < 1.0:
        raise ValueError(f"conformal_calibration_frac must be in (0, 1), got {frac!r}")


def split_indices(n: int, frac: float, random_state) -> tuple[np.ndarray, np.ndarray]:
    """``(fit_idx, calib_idx)`` -- positional, shuffled, seeded, disjoint."""
    idx = np.arange(n)
    fit_idx, calib_idx = train_test_split(idx, test_size=frac, random_state=random_state)
    return np.sort(fit_idx), np.sort(calib_idx)


def margin(y_calib, y_pred_calib, alpha: float) -> float:
    """Symmetric split-conformal margin from calibration residuals.

    Warns when the calibration split is too small for the requested
    ``alpha``: below ``ceil(1/alpha) - 1`` points the conformal quantile
    index exceeds ``n`` and the margin silently degrades to the largest
    observed residual, which no longer carries the finite-sample guarantee.
    """
    y_calib = np.asarray(y_calib, dtype=float)
    y_pred_calib = np.asarray(y_pred_calib, dtype=float)
    n = len(y_calib)
    if n == 0:
        raise ValueError("conformal calibration split is empty; raise conformal_calibration_frac")
    if math.ceil((n + 1) * (1.0 - alpha)) > n:
        warnings.warn(
            f"conformal calibration split has {n} rows, too few for "
            f"conformal_alpha={alpha}: the margin falls back to the largest "
            "calibration residual and the coverage guarantee no longer holds. "
            "Use more data or a larger conformal_calibration_frac.",
            stacklevel=3,
        )
    return conformal_calibration_margin(y_calib, y_pred_calib, y_pred_calib, alpha)


def intervals(y_pred, margin_: float | None, estimator_name: str) -> tuple[np.ndarray, np.ndarray]:
    if margin_ is None:
        raise RuntimeError(
            f"{estimator_name}.predict_intervals() needs a calibrated margin: "
            "fit with conformal_calibration=True. Without it the model has no "
            "interval to report (returning a zero-width one would read as "
            "certainty)."
        )
    y_pred = np.asarray(y_pred, dtype=float)
    return y_pred - margin_, y_pred + margin_
