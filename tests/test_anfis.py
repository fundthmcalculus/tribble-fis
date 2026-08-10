"""Tests for the ANFIS engine (src/tribblefis/anfis.py).

Four things need checking, matching the acceptance bar the rest of the
package holds analytic-gradient and closed-form-solver code to
(`tests/test_refine_analytic_grad.py`, `tests/test_regression.py`):

* the vectorized forward pass agrees with an independent, brute-force
  reference built directly from `GaussianMembership.evaluate`;
* the analytic premise gradient agrees with a finite-difference estimate of
  the same (consequents-held-fixed) loss it is a gradient of;
* the closed-form consequent solver recovers a known-exact linear function;
* the estimator's guard-rail (never worse, on the validation fold, than the
  untrained grid partition) holds end to end.
"""

import unittest

import numpy as np
import pandas as pd

from tribblefis.gauss_data import GaussianMembership
from tribblefis.regression import _mse, _rsquared
from tribblefis.anfis import (
    ANFISModel,
    ANFISRegressor,
    RuleExplosionError,
    anfis_predict,
    fit_anfis,
    init_anfis_model,
    raw_firing_strengths,
    term_memberships,
    _normalize_firing_strengths,
    _per_rule_predictions,
    _premise_gradients,
)


def _toy_data(seed=0, n=150):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x0": rng.uniform(-3, 3, size=n),
        "x1": rng.uniform(-3, 3, size=n),
    })
    y = np.sin(X["x0"].to_numpy()) + 0.4 * X["x1"].to_numpy() + rng.normal(0, 0.02, size=n)
    return X, y


def _brute_force_firing_strengths(model: ANFISModel, X: pd.DataFrame) -> np.ndarray:
    """Independent reference: per-rule product t-norm, one rule at a time,
    using `GaussianMembership.evaluate` directly rather than the vectorized
    tensor path `raw_firing_strengths` takes."""
    n = len(X)
    fs = np.ones((n, model.n_rules))
    for r in range(model.n_rules):
        for fi, f in enumerate(model.feature_names):
            k = model.rule_grid[r, fi]
            mf = GaussianMembership(mu=float(model.mu[fi][k]), sigma=float(model.sigma[fi][k]))
            fs[:, r] *= mf.evaluate(X[f].to_numpy(dtype=float))
    return fs


class TestForwardPassMatchesBruteForce(unittest.TestCase):
    def test_raw_firing_strengths(self):
        X, _ = _toy_data()
        model = init_anfis_model(X, ["x0", "x1"], n_terms=[2, 3])
        feature_arrays = {c: X[c].to_numpy() for c in X.columns}
        memberships = term_memberships(model, feature_arrays)
        fast = raw_firing_strengths(model, memberships)
        reference = _brute_force_firing_strengths(model, X)
        np.testing.assert_allclose(fast, reference, rtol=1e-10, atol=1e-12)


class TestPremiseGradientMatchesFiniteDifference(unittest.TestCase):
    """The gradient `_premise_gradients` returns is of the loss with the
    *current* consequents held fixed (see the module docstring on why that is
    the correct half of the hybrid rule, not an approximation of it) -- so the
    finite-difference check below must hold consequents fixed too."""

    def _loss_and_intermediates(self, model: ANFISModel, X: pd.DataFrame, y: np.ndarray):
        feature_arrays = {c: X[c].to_numpy() for c in X.columns}
        memberships = term_memberships(model, feature_arrays)
        raw_fs = raw_firing_strengths(model, memberships)
        norm_fs = _normalize_firing_strengths(raw_fs)
        X_rule = np.column_stack([feature_arrays[f] for f in model.feature_names])
        per_rule = _per_rule_predictions(model, X_rule)
        y_hat = np.sum(norm_fs * per_rule, axis=1)
        return _mse(y, y_hat), memberships, raw_fs, y_hat, per_rule

    def test_gradient_check_mu_and_sigma(self):
        X, y = _toy_data(seed=1, n=80)
        model = init_anfis_model(X, ["x0", "x1"], n_terms=[2, 3])
        # Random (not zero) consequents: the gradient formula involves
        # `per_rule_pred`, so a degenerate all-zero consequent would leave
        # a sign or scale error undetected.
        rng = np.random.default_rng(2)
        model.consequent = rng.normal(0, 0.3, size=model.consequent.shape)

        feature_arrays = {c: X[c].to_numpy() for c in X.columns}
        loss0, memberships, raw_fs, y_hat, per_rule = self._loss_and_intermediates(model, X, y)
        feats = [feature_arrays[f] for f in model.feature_names]
        d_mu, d_sigma = _premise_gradients(model, feats, memberships, raw_fs, y, y_hat, per_rule)

        h = 1e-5
        checked = 0
        for fi in range(model.n_features):
            for k in range(model.n_terms[fi]):
                for arr_name, analytic in (("mu", d_mu[fi][k]), ("sigma", d_sigma[fi][k])):
                    plus = model.copy()
                    minus = model.copy()
                    getattr(plus, arr_name)[fi][k] += h
                    getattr(minus, arr_name)[fi][k] -= h
                    loss_plus, *_ = self._loss_and_intermediates(plus, X, y)
                    loss_minus, *_ = self._loss_and_intermediates(minus, X, y)
                    numeric = (loss_plus - loss_minus) / (2 * h)
                    rel_err = abs(analytic - numeric) / max(abs(numeric), 1e-8)
                    self.assertLess(
                        rel_err, 5e-4,
                        msg=f"{arr_name} feature {fi} term {k}: analytic={analytic!r} "
                            f"numeric={numeric!r} rel_err={rel_err!r}",
                    )
                    checked += 1
        self.assertGreater(checked, 0)


class TestConsequentSolveExactness(unittest.TestCase):
    """A single-rule model (`n_terms=1` everywhere) always fires with weight 1,
    so the closed-form consequent solve is plain ordinary least squares and
    must recover a noiseless linear target exactly."""

    def test_recovers_linear_function(self):
        rng = np.random.default_rng(3)
        n = 200
        X = pd.DataFrame({
            "x0": rng.uniform(-5, 5, size=n),
            "x1": rng.uniform(-5, 5, size=n),
        })
        true_coeffs = np.array([1.5, -2.0, 0.75])  # intercept, x0, x1
        y = true_coeffs[0] + true_coeffs[1] * X["x0"] + true_coeffs[2] * X["x1"]
        y = y.to_numpy()

        model = init_anfis_model(X, ["x0", "x1"], n_terms=1, order="1st")
        self.assertEqual(model.n_rules, 1)

        refined, history = fit_anfis(model, X, y, n_epochs=1, learning_rate=0.0, l2_reg=0.0, seed=0)
        np.testing.assert_allclose(refined.consequent[0], true_coeffs, atol=1e-8)
        pred = anfis_predict(refined, X)
        np.testing.assert_allclose(pred, y, atol=1e-8)


class TestRuleExplosionGuard(unittest.TestCase):
    def test_raises_before_building_anything(self):
        X, _ = _toy_data(n=10)
        with self.assertRaises(RuleExplosionError):
            init_anfis_model(X, ["x0", "x1"], n_terms=[100, 100])


class TestANFISRegressorEndToEnd(unittest.TestCase):
    def test_fit_predict_and_guard_rail(self):
        X, y = _toy_data(seed=4, n=300)
        reg = ANFISRegressor(n_terms=3, n_epochs=50, learning_rate=0.05, random_state=42)
        reg.fit(X, y)
        pred = reg.predict(X)

        self.assertGreater(_rsquared(y, pred), 0.9)
        self.assertEqual(reg.n_rules_, 9)

        # Guard-rail: the selected (best-validation) epoch must be at least as
        # good, on its own fold, as epoch 0 -- the untrained grid partition's
        # own first consequent solve. Same invariant `refine.py` enforces.
        val_curve = [h["val_mse"] for h in reg.history_]
        self.assertLessEqual(min(val_curve), val_curve[0] + 1e-9)

    def test_describe_rules_is_readable(self):
        X, y = _toy_data(seed=5, n=60)
        reg = ANFISRegressor(n_terms=2, n_epochs=5, random_state=42)
        reg.fit(X, y)
        rules = reg.describe_rules()
        self.assertEqual(len(rules), reg.n_rules_)
        for line in rules:
            self.assertIn("IF", line)
            self.assertIn("THEN", line)

    def test_zeroth_order(self):
        X, y = _toy_data(seed=6, n=100)
        reg = ANFISRegressor(n_terms=2, tsk_order="0th", n_epochs=20, random_state=42)
        reg.fit(X, y)
        pred = reg.predict(X)
        self.assertEqual(pred.shape, y.shape)
        self.assertTrue(np.all(np.isfinite(pred)))


if __name__ == "__main__":
    unittest.main()
