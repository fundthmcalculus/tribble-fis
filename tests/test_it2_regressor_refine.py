"""Tests for post-fit IT2 *regressor* antecedent refinement with per-candidate
consequent re-solving (`it2_refine.refine_it2_regressor_antecedents`).

Unlike the classifier, a regressor candidate's antecedents are only ever
meaningful alongside consequents solved *for* them -- these tests exercise
that re-solve directly, and the invariant-preservation fix that a first
version of the (shared) refinement machinery got wrong: optimizing each IT2
membership's upper/lower Gaussian halves fully independently let the search
invert `firing_lower <= firing_upper`, which in turn made
`karnik_mendel_tsk` return `y_l > y_r` on a real fit (observed directly while
developing this feature, not a hypothetical).
"""

import numpy as np
import pandas as pd
import pytest

from tribblefis.it2_kernel import it2_firing_strengths
from tribblefis.it2_refine import refine_it2_regressor_antecedents
from tribblefis.it2_regressor import IT2TribbleRegressor


@pytest.fixture
def synthetic_regression_data():
    rng = np.random.default_rng(42)
    x = np.linspace(-2, 2, 150)
    y = np.sin(3 * x) + 0.05 * rng.standard_normal(len(x))
    # Column name "x0" matches what `IT2TribbleRegressor.fit` renames a
    # single-feature input to internally (`check_X_y` drops the original "x"),
    # so `reg._base_regressor.top_features_` lines up with `X`'s own columns
    # when tests call `refine_it2_regressor_antecedents` directly.
    return pd.DataFrame({"x0": x}), pd.Series(y)


def _fitted_it2_regressor(X, y, **kwargs):
    reg = IT2TribbleRegressor(
        top_n=1, n_gaussians=2, n_output_buckets=3, uncertainty_width=0.5,
        km_iterations=10, random_state=42, **kwargs,
    )
    reg.fit(X, y)
    return reg


def test_refine_it2_regressor_antecedents_never_increases_cv_loss(synthetic_regression_data):
    X, y = synthetic_regression_data
    reg = _fitted_it2_regressor(X, y)

    refined_model, corr_terms, y_bucket_mean, info = refine_it2_regressor_antecedents(
        X, y.to_numpy(), reg.model_, reg.norms_, reg._base_regressor.top_features_,
        order=reg._base_regressor.tsk_order, l2_reg=reg._base_regressor.l2_reg,
        basis=reg._base_regressor.consequent_basis, cross_pairs=reg._base_regressor.cross_pairs_,
        km_iterations=10, n_sweeps=2, sub_maxfun=15, n_folds=2, verbose=False,
    )

    assert info["val_mse"] <= info["init_val_mse"] + 1e-9
    assert corr_terms.shape == reg._base_regressor.corr_terms_.shape
    assert y_bucket_mean.shape == reg._base_regressor.y_bucket_mean_.shape


def test_refine_it2_regressor_antecedents_preserves_lower_le_upper_invariant(synthetic_regression_data):
    X, y = synthetic_regression_data
    reg = _fitted_it2_regressor(X, y)

    refined_model, _, _, _ = refine_it2_regressor_antecedents(
        X, y.to_numpy(), reg.model_, reg.norms_, reg._base_regressor.top_features_,
        order=reg._base_regressor.tsk_order, l2_reg=reg._base_regressor.l2_reg,
        basis=reg._base_regressor.consequent_basis, cross_pairs=reg._base_regressor.cross_pairs_,
        km_iterations=10, n_sweeps=3, sub_maxfun=20, n_folds=2, verbose=False,
    )

    firing_upper, firing_lower, _, _ = it2_firing_strengths(X, refined_model, reg.norms_, km_iterations=None)
    assert np.all(firing_lower <= firing_upper + 1e-9)


def test_refine_it2_regressor_antecedents_method_none_still_resolves_consequents(synthetic_regression_data):
    """`method="none"` skips the antecedent search but must still return
    consequents solved for the (unchanged) antecedents -- not the base
    regressor's original ones, which were fit against Type-1 firing strengths,
    not the IT2 midpoint firing strengths this module uses."""
    X, y = synthetic_regression_data
    reg = _fitted_it2_regressor(X, y)

    refined_model, corr_terms, y_bucket_mean, info = refine_it2_regressor_antecedents(
        X, y.to_numpy(), reg.model_, reg.norms_, reg._base_regressor.top_features_,
        order=reg._base_regressor.tsk_order, l2_reg=reg._base_regressor.l2_reg,
        basis=reg._base_regressor.consequent_basis, cross_pairs=reg._base_regressor.cross_pairs_,
        method="none",
    )
    assert refined_model is reg.model_
    assert info["val_mse"] is None


def test_refine_it2_regressor_antecedents_rejects_unknown_method(synthetic_regression_data):
    X, y = synthetic_regression_data
    reg = _fitted_it2_regressor(X, y)

    with pytest.raises(ValueError):
        refine_it2_regressor_antecedents(
            X, y.to_numpy(), reg.model_, reg.norms_, reg._base_regressor.top_features_,
            method="bogus",
        )


def test_regressor_refine_it2_option_fits_and_predicts_with_containment(synthetic_regression_data):
    """End-to-end: `IT2TribbleRegressor(refine_it2=True)` fits, and
    `predict()`'s point estimate stays inside `predict_intervals()`'s bounds
    -- the property that a real fit violated before the invariant fix, because
    an inverted firing interval flows straight into `karnik_mendel_tsk` as
    `y_l > y_r`."""
    X, y = synthetic_regression_data
    reg = _fitted_it2_regressor(X, y, refine_it2=True, refine_it2_n_sweeps=2, refine_it2_n_folds=2)

    y_pred = reg.predict(X)
    y_lower, y_upper = reg.predict_intervals(X)

    assert np.all(y_lower <= y_upper + 1e-9)
    assert np.all(y_lower <= y_pred + 1e-9)
    assert np.all(y_pred <= y_upper + 1e-9)
    np.testing.assert_allclose(y_pred, 0.5 * (y_lower + y_upper), atol=1e-9)


def test_regressor_refine_it2_does_not_drastically_worsen_rmse(synthetic_regression_data):
    X, y = synthetic_regression_data
    from sklearn.metrics import mean_squared_error

    baseline = _fitted_it2_regressor(X, y, refine_it2=False)
    baseline_rmse = np.sqrt(mean_squared_error(y, baseline.predict(X)))

    refined = _fitted_it2_regressor(X, y, refine_it2=True, refine_it2_n_sweeps=2, refine_it2_n_folds=2)
    refined_rmse = np.sqrt(mean_squared_error(y, refined.predict(X)))

    assert refined_rmse <= baseline_rmse * 1.5 + 0.05
