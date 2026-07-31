"""Leaf TSK consequent solver and prediction.

This replicates the closed-form ridge normal-equation solve of
``tribblefis.regression.solve_tsk_consequents`` (lines ~501-522) but consumes an
externally-supplied leaf firing-strength matrix rather than deriving firing from a
``GaussianMixtureModel``. That is the one thing the upstream function cannot do
(it calls ``tsk_firing_strengths`` internally), so we duplicate ~20 lines here and
reuse everything else by import:

    * ``_normalize_firing_strengths`` -- identical row-normalisation + zero-firing
      fallback, so uncovered points contribute nothing at fit and predict 0.
    * ``build_consequent_features`` -- identical poly/Legendre basis.

Because the leaf firing matrix already encodes the fuzzy path weights, no extra
sample weighting is needed: normalising it row-wise and stacking per-leaf design
blocks gives exactly the firing-weighted least-squares optimum, matching the flat
model's proven math.
"""

from __future__ import annotations

import numpy as np

from tribblefis.regression import (
    _normalize_firing_strengths,
    build_consequent_features,
)


def solve_leaf_consequents(
    X_feats: np.ndarray,
    y: np.ndarray,
    leaf_firing: np.ndarray,
    order: str = "0th",
    basis: str = "raw",
    l2_reg: float = 0.0,
    cross_pairs: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve every leaf's TSK consequent in one closed-form ridge least squares.

    Args:
        X_feats: (n_samples, n_features) consequent inputs (the selected features).
        y: (n_samples,) continuous target.
        leaf_firing: (n_samples, n_leaves) leaf firing strengths.
        order: TSK order ('0th'..'3rd', 'full-2nd').
        basis: 'raw' or 'orthogonal'.
        l2_reg: ridge strength (intercepts never penalised).
        cross_pairs: explicit interaction pairs for 'full-2nd'.

    Returns:
        (corr_terms, leaf_mean): coefficients shaped (n_leaves, n_terms) and
        (n_leaves,). For '0th' order, corr_terms has zero columns.
    """
    X_feats = np.asarray(X_feats, dtype=float)
    y = np.asarray(y, dtype=float)

    norm_fs = _normalize_firing_strengths(leaf_firing)
    n_leaves = norm_fs.shape[1]

    feats = build_consequent_features(X_feats, order, basis=basis, cross_pairs=cross_pairs)
    n_terms = feats.shape[1]
    n_coeffs = 1 + n_terms

    phi = np.hstack([np.ones((X_feats.shape[0], 1)), feats])  # (n, 1 + n_terms)
    design = (norm_fs[:, :, np.newaxis] * phi[:, np.newaxis, :]).reshape(
        X_feats.shape[0], n_leaves * n_coeffs
    )

    penalty = np.ones(n_leaves * n_coeffs)
    penalty[::n_coeffs] = 0.0  # never penalise the intercept of each leaf block
    gram = design.T @ design + l2_reg * np.diag(penalty)
    rhs = design.T @ y

    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(design, y, rcond=None)[0]

    coeffs = beta.reshape(n_leaves, n_coeffs)
    leaf_mean = coeffs[:, 0].copy()
    corr_terms = coeffs[:, 1:].copy() if n_terms > 0 else np.zeros((n_leaves, 0))
    return corr_terms, leaf_mean


def predict_leaves(
    X_feats: np.ndarray,
    leaf_firing: np.ndarray,
    leaf_mean: np.ndarray,
    corr_terms: np.ndarray,
    order: str = "0th",
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Firing-weighted TSK prediction (mirrors ``regression.predict_tsk``)."""
    X_feats = np.asarray(X_feats, dtype=float)
    norm_fs = _normalize_firing_strengths(leaf_firing)

    if order == "0th":
        return norm_fs @ np.asarray(leaf_mean)

    feats = build_consequent_features(X_feats, order, basis=basis, cross_pairs=cross_pairs)
    y_pred = np.zeros(X_feats.shape[0])
    for leaf in range(norm_fs.shape[1]):
        y_pred += (leaf_mean[leaf] + feats @ corr_terms[leaf, :]) * norm_fs[:, leaf]
    return y_pred
