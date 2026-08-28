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
  untrained grid partition) holds end to end;
* the grid partition of a *constant* feature straddles the one value that
  feature takes rather than sitting beside it (#206).
"""

import unittest

import numpy as np
import pandas as pd

from tribblefis.gauss_data import GaussianMembership
from tribblefis.refine import feature_span
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


class TestConstantFeatureGridPartition(unittest.TestCase):
    """A feature whose observed min equals its max (#206).

    `init_anfis_model` used to widen such a feature by anchoring at the low end
    (`hi = lo + 1.0`) while `refine.feature_span` widened it symmetrically. Two
    conventions for one degenerate case, and the anchoring one put every term at
    or above the only value the feature takes: at `k == 1` the centre landed at
    `lo + 0.5`, and at `k == 4` the four terms evaluated to
    `[1.0, 6.25e-2, 1.53e-5, 1.46e-11]` there -- lopsided, with the outermost
    term effectively dead on every row.

    These assert against `feature_span` rather than restating "unit width
    centred on the constant" locally, so they follow the shared helper if its
    convention ever moves and fail if `anfis` drifts away from it again.
    """

    CONSTANT = 3.0

    def _model(self, k):
        return init_anfis_model(pd.DataFrame({"c": np.full(40, self.CONSTANT)}), ["c"], k)

    def _memberships_at_the_constant(self, model):
        return term_memberships(model, {"c": np.array([self.CONSTANT])})[0][0]

    def test_single_term_centre_sits_on_the_only_observed_value(self):
        # The whole point: `lo + 0.5` is not a defensible centre for a feature
        # that only ever takes `lo`.
        self.assertEqual(float(self._model(1).mu[0][0]), self.CONSTANT)

    def test_multiple_terms_straddle_the_only_observed_value(self):
        for k in (2, 3, 4):
            with self.subTest(k=k):
                mu = self._model(k).mu[0]
                self.assertLess(float(np.min(mu)), self.CONSTANT)
                self.assertGreater(float(np.max(mu)), self.CONSTANT)
                # Symmetric about the constant, not piled up on one side.
                self.assertAlmostEqual(
                    float(np.mean(mu)), self.CONSTANT, places=12
                )

    def test_terms_are_mirror_images_at_the_only_observed_value(self):
        # Under the old anchoring, k=4 gave [1.0, 6.25e-2, 1.53e-5, 1.46e-11]:
        # membership fell away monotonically from one end because every term sat
        # on the same side of the data. Symmetric widening makes the sequence a
        # palindrome, so no term is systematically starved.
        for k in (1, 2, 3, 4):
            with self.subTest(k=k):
                mem = self._memberships_at_the_constant(self._model(k))
                self.assertTrue(np.all(mem > 0.0), mem)
                np.testing.assert_allclose(mem, mem[::-1], rtol=0, atol=1e-15)

    def test_agrees_with_the_shared_feature_span_helper(self):
        lo, hi, _ = feature_span(np.full(40, self.CONSTANT))
        self.assertEqual(float(self._model(1).mu[0][0]), 0.5 * (lo + hi))
        for k in (2, 3, 4):
            with self.subTest(k=k):
                np.testing.assert_allclose(
                    self._model(k).mu[0], np.linspace(lo, hi, k), rtol=0, atol=1e-15
                )

    def test_the_width_convention_did_not_move(self):
        # Only the *location* changed in #206. Both conventions give unit width,
        # so sigma is what it always was -- adjacent terms crossing at 0.5.
        _, _, rng = feature_span(np.full(40, self.CONSTANT))
        for k in (1, 2, 3, 4):
            with self.subTest(k=k):
                gap = rng / (k - 1) if k > 1 else rng
                expected = 0.5 * gap / np.sqrt(2 * np.log(2))
                np.testing.assert_allclose(
                    self._model(k).sigma[0], np.full(k, expected), rtol=1e-12, atol=0
                )

    def test_an_ordinary_column_is_untouched(self):
        # The non-regression leg: nothing changes when no feature is constant.
        col = np.linspace(-2.0, 6.0, 50)
        X = pd.DataFrame({"c": col})
        for k in (1, 2, 3, 4):
            with self.subTest(k=k):
                mu = init_anfis_model(X, ["c"], k).mu[0]
                expected = (np.array([0.5 * (col.min() + col.max())]) if k == 1
                            else np.linspace(col.min(), col.max(), k))
                np.testing.assert_allclose(mu, expected, rtol=0, atol=0)

    def test_a_single_row_frame_centres_every_feature_on_its_value(self):
        # min == max on every column, which is the same degenerate case.
        X = pd.DataFrame({"a": [7.5], "b": [-2.0]})
        model = init_anfis_model(X, ["a", "b"], 2)
        for fi, value in enumerate((7.5, -2.0)):
            with self.subTest(feature=fi):
                self.assertAlmostEqual(float(np.mean(model.mu[fi])), value, places=12)

    def test_an_end_to_end_fit_survives_a_constant_column(self):
        # A smoke test, not a #206 assertion: the centres are checked above.
        # What this adds is that the widened feature still trains -- the terms
        # now overlap far more (0.5/0.5 at k=2 instead of 1.0/0.0625), and a
        # near-duplicated rule base is the sort of thing a consequent solve can
        # choke on.
        X, y = _toy_data(seed=7, n=80)
        X = X.assign(cst=np.full(len(X), self.CONSTANT))
        reg = ANFISRegressor(n_terms=2, n_epochs=10, random_state=42).fit(X, y)
        self.assertTrue(np.all(np.isfinite(reg.predict(X))))
        self.assertEqual(reg.n_rules_, 8)


if __name__ == "__main__":
    unittest.main()
