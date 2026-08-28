"""Tests for ``tsk_order="auto"`` (issue #120).

A full-2nd consequent fits ``1 + 2*n_features + C(n_features, 2)``
coefficients per rule, which overfits catastrophically once training rows
undercut that count by roughly 5x -- the caller previously had to know this
and hand-pick a safer order. ``tsk_order="auto"`` runs
``regression.select_consequent_hyperparams``'s k-fold CV over a fixed set of
candidate orders (basis/l2_reg held fixed) and fits the winner instead.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.regression import _rsquared


def _linear_problem(n, seed, n_features=8):
    """y is a clean linear function of X -- full-2nd's extra coefficients
    are pure overfitting risk here, with nothing genuine for them to fit."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, n_features))
    coefs = rng.uniform(0.5, 1.5, size=n_features)
    y = X @ coefs + rng.normal(0, 0.05, n)
    cols = [f"x{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), pd.Series(y)


class TestAutoOrderAvoidsOverfit(unittest.TestCase):
    """On small-sample data, full-2nd overfits badly relative to order 1;
    auto should recover order 1's accuracy rather than full-2nd's."""

    def test_auto_matches_best_order_not_full2nd(self):
        X_train, y_train = _linear_problem(40, seed=1)
        X_test, y_test = _linear_problem(400, seed=2)

        r2 = {}
        resolved = {}
        for order in ("1st", "full-2nd", "auto"):
            reg = TribbleRegressor(
                tsk_order=order, top_n=-1, n_output_buckets=2,
                l2_reg=0.0, random_state=0,
            )
            reg.fit(X_train, y_train)
            r2[order] = _rsquared(y_test, reg.predict(X_test))
            resolved[order] = reg.tsk_order_

        # full-2nd measurably overfits relative to order 1 on this small sample.
        self.assertLess(r2["full-2nd"], r2["1st"] - 0.1)
        # auto must not silently inherit full-2nd's overfit.
        self.assertEqual(resolved["auto"], "1st")
        self.assertAlmostEqual(r2["auto"], r2["1st"], places=6)

    def test_auto_resolves_tsk_order_and_records_selection(self):
        X, y = _linear_problem(200, seed=5)
        reg = TribbleRegressor(tsk_order="auto", top_n=-1, n_output_buckets=2, random_state=0)
        reg.fit(X, y)

        self.assertIn(reg.tsk_order_, reg.auto_order_candidates)
        self.assertIsNotNone(reg.consequent_selection_)
        self.assertEqual(reg.consequent_selection_["order"], reg.tsk_order_)
        self.assertIn("val_r2", reg.consequent_selection_)

        # predict() must use the resolved order, not the literal "auto".
        preds = reg.predict(X)
        self.assertEqual(len(preds), len(y))

    def test_tsk_order_param_itself_is_untouched_by_fit(self):
        """sklearn's clone()/get_params() require __init__ args to round-trip
        unmodified -- fit() must resolve "auto" into tsk_order_, not mutate
        self.tsk_order."""
        reg = TribbleRegressor(tsk_order="auto", top_n=-1, n_output_buckets=2, random_state=0)
        X, y = _linear_problem(200, seed=7)
        reg.fit(X, y)
        self.assertEqual(reg.tsk_order, "auto")
        self.assertIn(reg.tsk_order_, ("0th", "1st", "2nd", "full-2nd", "3rd"))

    def test_non_auto_order_sets_matching_tsk_order_(self):
        X, y = _linear_problem(200, seed=9)
        reg = TribbleRegressor(tsk_order="2nd", top_n=-1, n_output_buckets=2, random_state=0)
        reg.fit(X, y)
        self.assertEqual(reg.tsk_order_, "2nd")
        self.assertIsNone(reg.consequent_selection_)


class TestAutoOrderWithInteractionDetection(unittest.TestCase):
    """auto + detect_interactions must prepare cross_pairs_ the same way
    full-2nd does, since auto may resolve to full-2nd, and must not raise
    the select_interactions-without-full-2nd warning while doing so."""

    def _interaction_only_problem(self, seed=3, n=1500):
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(-1, 1, n)
        x1 = rng.uniform(-1, 1, n)
        noise = rng.uniform(-1, 1, n)
        y = x0 * x1 + rng.normal(0, 0.02, n)
        X = pd.DataFrame({"x0": x0, "x1": x1, "noise": noise})
        return X, y

    def test_auto_with_detect_and_select_interactions_does_not_warn(self):
        X, y = self._interaction_only_problem()
        reg = TribbleRegressor(
            tsk_order="auto", top_n=1, n_output_buckets=6, random_state=42,
            detect_interactions=True, select_interactions=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reg.fit(X, y)
        self.assertFalse(any("select_interactions" in str(w.message) for w in caught))

    def test_auto_resolving_to_full2nd_keeps_cross_pairs(self):
        X, y = self._interaction_only_problem()
        reg = TribbleRegressor(
            tsk_order="auto", top_n=1, n_output_buckets=6, random_state=42,
            detect_interactions=True,
        )
        reg.fit(X, y)
        if reg.tsk_order_ == "full-2nd":
            self.assertEqual(reg.cross_pairs_, [(0, 1)])


if __name__ == "__main__":
    unittest.main()
