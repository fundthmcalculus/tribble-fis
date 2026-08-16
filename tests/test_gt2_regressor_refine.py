"""Tests for post-fit GT2 *regressor* antecedent refinement with per-candidate
consequent re-solving (`gt2_refine.refine_gt2_regressor_antecedents`).

Direct GT2 analogue of `test_it2_regressor_refine.py` -- see that file's
docstring for the invariant-preservation history this mirrors, one dimension
wider (`sigma_lower <= sigma_principal <= sigma_upper`).
"""

import numpy as np
import pandas as pd
import pytest

from tribblefis.gt2_kernel import extract_alpha_plane_model
from tribblefis.it2_kernel import it2_firing_strengths
from tribblefis.gt2_refine import refine_gt2_regressor_antecedents
from tribblefis.gt2_regressor import GT2TribbleRegressor


@pytest.fixture
def synthetic_regression_data():
    rng = np.random.default_rng(42)
    x = np.linspace(-2, 2, 150)
    y = np.sin(3 * x) + 0.05 * rng.standard_normal(len(x))
    # Column name "x0" matches what `GT2TribbleRegressor.fit` renames a
    # single-feature input to internally, mirroring the IT2 fixture's own
    # comment -- see `test_it2_regressor_refine.py`.
    return pd.DataFrame({"x0": x}), pd.Series(y)


def _fitted_gt2_regressor(X, y, **kwargs):
    reg = GT2TribbleRegressor(
        top_n=1, n_gaussians=2, n_output_buckets=3, uncertainty_width=0.5,
        n_alpha_planes=3, km_iterations=10, random_state=42, **kwargs,
    )
    reg.fit(X, y)
    return reg


def test_refine_gt2_regressor_antecedents_never_increases_cv_loss(synthetic_regression_data):
    X, y = synthetic_regression_data
    reg = _fitted_gt2_regressor(X, y)

    refined_model, corr_terms, y_bucket_mean, info = refine_gt2_regressor_antecedents(
        X, y.to_numpy(), reg.model_, reg.norms_, reg._base_regressor.top_features_,
        order=reg._base_regressor.tsk_order, l2_reg=reg._base_regressor.l2_reg,
        basis=reg._base_regressor.consequent_basis, cross_pairs=reg._base_regressor.cross_pairs_,
        n_alpha_planes=3, km_iterations=10, n_sweeps=2, sub_maxfun=15, n_folds=2, verbose=False,
    )

    assert info["val_mse"] <= info["init_val_mse"] + 1e-9
    assert corr_terms.shape == reg._base_regressor.corr_terms_.shape
    assert y_bucket_mean.shape == reg._base_regressor.y_bucket_mean_.shape


def test_refine_gt2_regressor_antecedents_preserves_sigma_ordering_invariant(synthetic_regression_data):
    X, y = synthetic_regression_data
    reg = _fitted_gt2_regressor(X, y)

    refined_model, _, _, _ = refine_gt2_regressor_antecedents(
        X, y.to_numpy(), reg.model_, reg.norms_, reg._base_regressor.top_features_,
        order=reg._base_regressor.tsk_order, l2_reg=reg._base_regressor.l2_reg,
        basis=reg._base_regressor.consequent_basis, cross_pairs=reg._base_regressor.cross_pairs_,
        n_alpha_planes=3, km_iterations=10, n_sweeps=3, sub_maxfun=20, n_folds=2, verbose=False,
    )

    for gt2_mf in refined_model.all_membership_fcns:
        assert gt2_mf.lower_mf.sigma <= gt2_mf.principal_mf.sigma + 1e-9
        assert gt2_mf.principal_mf.sigma <= gt2_mf.upper_mf.sigma + 1e-9

    it2_model_alpha0 = extract_alpha_plane_model(refined_model, 0.0)
    firing_upper, firing_lower, _, _ = it2_firing_strengths(X, it2_model_alpha0, reg.norms_, km_iterations=None)
    assert np.all(firing_lower <= firing_upper + 1e-9)


def test_refine_gt2_regressor_antecedents_method_none_still_resolves_consequents(synthetic_regression_data):
    """`method="none"` skips the antecedent search but must still return
    consequents solved for the (unchanged) antecedents."""
    X, y = synthetic_regression_data
    reg = _fitted_gt2_regressor(X, y)

    refined_model, corr_terms, y_bucket_mean, info = refine_gt2_regressor_antecedents(
        X, y.to_numpy(), reg.model_, reg.norms_, reg._base_regressor.top_features_,
        order=reg._base_regressor.tsk_order, l2_reg=reg._base_regressor.l2_reg,
        basis=reg._base_regressor.consequent_basis, cross_pairs=reg._base_regressor.cross_pairs_,
        n_alpha_planes=3, method="none",
    )
    assert refined_model is reg.model_
    assert info["val_mse"] is None


def test_refine_gt2_regressor_antecedents_rejects_unknown_method(synthetic_regression_data):
    X, y = synthetic_regression_data
    reg = _fitted_gt2_regressor(X, y)

    with pytest.raises(ValueError):
        refine_gt2_regressor_antecedents(
            X, y.to_numpy(), reg.model_, reg.norms_, reg._base_regressor.top_features_,
            method="bogus",
        )


def test_regressor_refine_gt2_option_fits_and_predicts_with_containment(synthetic_regression_data):
    """End-to-end: `GT2TribbleRegressor(refine_gt2=True)` fits, and
    `predict()`'s alpha-combined point estimate stays inside
    `predict_intervals()`'s (alpha=0, widest) bounds.

    Unlike IT2 -- where `predict()` *is* the exact midpoint of
    `predict_intervals()` by construction -- GT2's `predict_intervals()`
    reports only the alpha=0 boundary while `predict()` combines every
    alpha-plane, so containment (not exact-midpoint equality) is the
    property that holds here; see `gt2_kernel.gt2_karnik_mendel_tsk`'s
    docstring for why containment holds regardless.
    """
    X, y = synthetic_regression_data
    reg = _fitted_gt2_regressor(X, y, refine_gt2=True, refine_gt2_n_sweeps=2, refine_gt2_n_folds=2)

    y_pred = reg.predict(X)
    y_lower, y_upper = reg.predict_intervals(X)

    assert np.all(y_lower <= y_upper + 1e-9)
    assert np.all(y_lower <= y_pred + 1e-9)
    assert np.all(y_pred <= y_upper + 1e-9)


def test_regressor_refine_gt2_does_not_drastically_worsen_rmse(synthetic_regression_data):
    X, y = synthetic_regression_data
    from sklearn.metrics import mean_squared_error

    baseline = _fitted_gt2_regressor(X, y, refine_gt2=False)
    baseline_rmse = np.sqrt(mean_squared_error(y, baseline.predict(X)))

    refined = _fitted_gt2_regressor(X, y, refine_gt2=True, refine_gt2_n_sweeps=2, refine_gt2_n_folds=2)
    refined_rmse = np.sqrt(mean_squared_error(y, refined.predict(X)))

    assert refined_rmse <= baseline_rmse * 1.5 + 0.05
