"""Tests for the analytic bilevel gradient used by antecedent refinement
(src/tribblefis/refine.py, issue #43) and the pre-extracted feature-array
threading through the refinement hot path (issue #42).
"""

import unittest

import numpy as np
import pandas as pd

from tribblefis.gauss_data import NormPair
from tribblefis.gauss_math import create_gaussian_membership_dict, tsk_firing_strengths
from tribblefis.regression import partition_output, solve_tsk_consequents, predict_tsk
from tribblefis.refine import (
    refine_antecedents_coordinate,
    extract_gaussian_params,
    apply_gaussian_params,
    _iter_gaussian_slots,
    _prepare_folds,
    _make_folds,
    _fold_mse_and_grad,
)

_PROBABILITY_NORMS = NormPair(t_norm="probability", t_conorm="probability")


def _synthetic_regression_data(seed=0, n=150):
    """Two-feature regression data with a smooth, mildly nonlinear target -- just
    enough structure for the heuristic model's antecedents to have room to move."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.uniform(-3, 3, size=n),
        "b": rng.uniform(-3, 3, size=n),
    })
    y = np.sin(X["a"]) + 0.3 * X["b"] + rng.normal(0, 0.05, size=n)
    return X, pd.Series(y, name="y_value")


def _build_model_and_folds(seed=0, n_output_buckets=3):
    X, y = _synthetic_regression_data(seed=seed)
    y_partitioned, y_bucket_mean = partition_output(n_output_buckets, y)
    model = create_gaussian_membership_dict(
        X, y_partitioned["y_bucket"], top_n_var_names=["a", "b"], n_gaussians=1
    )
    folds = _make_folds(len(X), n_folds=2, val_fraction=0.2, random_state=seed)
    prepared = _prepare_folds(X, y_partitioned, folds)
    return model, X, y_partitioned, prepared


class TestAnalyticGradientMatchesFiniteDifference(unittest.TestCase):
    """The acceptance bar from issue #43: the analytic gradient must agree with a
    finite-difference estimate of the *same* CV fitness that L-BFGS-B optimizes,
    away from the min/max kink (hence "probability" norms here)."""

    def _relative_error(self, analytic, numeric):
        return abs(analytic - numeric) / max(abs(numeric), 1e-8)

    def test_gradient_check_mu_and_sigma(self):
        model, X, y_partitioned, prepared = _build_model_and_folds()
        slots = list(_iter_gaussian_slots(model))
        self.assertGreater(len(slots), 0)

        x0 = extract_gaussian_params(model)
        top_n_todo = ["a", "b"]
        order, l2_reg, basis, cross_pairs = "full-2nd", 1e-2, "raw", None

        def cv_mse(vec):
            candidate = apply_gaussian_params(model, vec)
            total = 0.0
            for X_tr, y_tr, fa_tr, X_val, y_val_true, fa_val in prepared:
                corr, means = solve_tsk_consequents(
                    X_tr, candidate, top_n_todo, np.zeros(3), y_tr,
                    n_output_buckets=3, order=order, l2_reg=l2_reg, basis=basis,
                    cross_pairs=cross_pairs, pin_extremes=False, norms=_PROBABILITY_NORMS,
                    feature_arrays=fa_tr, verbose=False,
                )
                y_hat = predict_tsk(
                    X_val, candidate, top_n_todo, means, corr, order=order, basis=basis,
                    cross_pairs=cross_pairs, norms=_PROBABILITY_NORMS, feature_arrays=fa_val,
                )
                total += np.mean((y_val_true - y_hat) ** 2)
            return total / len(prepared)

        # Check a couple of blocks (membership functions), not just the first.
        for block_idx in (0, min(2, len(slots) - 1)):
            target_feature, target_label, target_mf_index, _mf = slots[block_idx]
            idx = np.array([2 * block_idx, 2 * block_idx + 1])

            total_g = np.zeros(2)
            for X_tr, y_tr, fa_tr, X_val, y_val_true, fa_val in prepared:
                candidate = apply_gaussian_params(model, x0)
                _, g = _fold_mse_and_grad(
                    fa_tr, y_tr, fa_val, y_val_true, candidate, top_n_todo,
                    order, l2_reg, basis, cross_pairs, target_feature, target_label, target_mf_index,
                )
                total_g += g
            analytic_grad = total_g / len(prepared)

            h = 1e-5
            for k in (0, 1):
                vec_plus = x0.copy()
                vec_minus = x0.copy()
                vec_plus[idx[k]] += h
                vec_minus[idx[k]] -= h
                numeric = (cv_mse(vec_plus) - cv_mse(vec_minus)) / (2 * h)
                rel_err = self._relative_error(analytic_grad[k], numeric)
                self.assertLess(
                    rel_err, 5e-3,
                    msg=f"block {block_idx} param {k}: analytic={analytic_grad[k]!r} "
                        f"numeric={numeric!r} rel_err={rel_err!r}",
                )


class TestCoordinateRefinementWithAnalyticGradient(unittest.TestCase):
    """End-to-end smoke test: the analytic-gradient path through
    `refine_antecedents_coordinate` must run cleanly and keep the never-worse-
    than-heuristic guarantee."""

    def test_never_worse_than_heuristic(self):
        model, X, y_partitioned, _ = _build_model_and_folds()
        refined, info = refine_antecedents_coordinate(
            model, X, y_partitioned, ["a", "b"], n_output_buckets=3,
            order="full-2nd", l2_reg=1e-2, n_folds=2, n_sweeps=2, sub_maxfun=10,
            norms=_PROBABILITY_NORMS, seed=0,
        )
        self.assertLessEqual(info["val_mse"], info["init_val_mse"] + 1e-9)
        self.assertEqual(refined.n_membership_functions, model.n_membership_functions)

    def test_default_norms_unaffected(self):
        """min/max (the default) must keep using the finite-difference path --
        this is an opt-in feature, not a change to existing behavior."""
        model, X, y_partitioned, _ = _build_model_and_folds()
        refined, info = refine_antecedents_coordinate(
            model, X, y_partitioned, ["a", "b"], n_output_buckets=3,
            order="full-2nd", l2_reg=1e-2, n_folds=2, n_sweeps=1, sub_maxfun=10, seed=0,
        )
        self.assertLessEqual(info["val_mse"], info["init_val_mse"] + 1e-9)


class TestFeatureArraysThreading(unittest.TestCase):
    """Issue #42: pre-extracted feature arrays must be a pure perf optimization --
    identical results with or without them."""

    def test_tsk_firing_strengths_matches(self):
        model, X, y_partitioned, _ = _build_model_and_folds()
        fs_default, labels_default = tsk_firing_strengths(X, model)
        feature_arrays = {c: X[c].to_numpy() for c in X.columns}
        fs_pre, labels_pre = tsk_firing_strengths(X, model, feature_arrays=feature_arrays)
        np.testing.assert_allclose(fs_default, fs_pre)
        self.assertEqual(labels_default, labels_pre)

    def test_solve_and_predict_match(self):
        model, X, y_partitioned, _ = _build_model_and_folds()
        top_n_todo = ["a", "b"]
        feature_arrays = {c: X[c].to_numpy() for c in X.columns}

        corr_a, means_a = solve_tsk_consequents(
            X, model, top_n_todo, np.zeros(3), y_partitioned, n_output_buckets=3,
            order="2nd", l2_reg=1e-2, verbose=False,
        )
        corr_b, means_b = solve_tsk_consequents(
            X, model, top_n_todo, np.zeros(3), y_partitioned, n_output_buckets=3,
            order="2nd", l2_reg=1e-2, verbose=False, feature_arrays=feature_arrays,
        )
        np.testing.assert_allclose(corr_a, corr_b)
        np.testing.assert_allclose(means_a, means_b)

        y_hat_a = predict_tsk(X, model, top_n_todo, means_a, corr_a, order="2nd")
        y_hat_b = predict_tsk(X, model, top_n_todo, means_b, corr_b, order="2nd",
                               feature_arrays=feature_arrays)
        np.testing.assert_allclose(y_hat_a, y_hat_b)


if __name__ == "__main__":
    unittest.main()
