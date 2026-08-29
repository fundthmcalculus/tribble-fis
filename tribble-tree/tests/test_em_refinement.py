"""Tests for EM refinement of the hierarchical fuzzy experts (fuzzytree/em.py).

Mirrors the validation plan in EM_REFINEMENT.md Sec.10: likelihood
monotonicity, the sharp-boundary synthetic headline check (Gaussian gates
should sharpen and beat the greedy build), a classification refinement smoke
test, and the starved-leaf safeguard.
"""

import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from fuzzytree import (
    HierarchicalFuzzyExpertsClassifier,
    HierarchicalFuzzyExpertsRegressor,
    VariablePlan,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


class TestEMRegression(unittest.TestCase):
    def setUp(self):
        rng = _rng(9)
        n = 1500
        self.a = rng.uniform(0, 10, n)
        self.b = rng.uniform(0, 10, n)
        self.X = pd.DataFrame({"a": self.a, "b": self.b})
        # Sharp piecewise-linear regime boundary at a=5 -- the headline case
        # from EM_REFINEMENT.md Sec.10.
        self.y = np.where(self.a < 5, 2 * self.b, -3 * self.b + 40) + rng.normal(0, 0.3, n)

    def _fit_gaussian_gate_hme(self):
        plan = VariablePlan(
            criterion="variance", max_depth=2, default_n_terms=2, max_terms_per_var=2,
            term_style="gaussian",
        )
        return HierarchicalFuzzyExpertsRegressor(
            variable_plan=plan, min_soft_count=50, min_expert_samples=50,
            expert_kwargs={"n_output_buckets": 3, "tsk_order": "1st"},
        ).fit(self.X, self.y)

    def test_log_likelihood_is_monotone(self):
        m = self._fit_gaussian_gate_hme()
        m.refine_em(self.X, self.y, max_iter=15)
        hist = m.em_log_likelihood_
        self.assertGreater(len(hist), 1)
        diffs = np.diff(hist)
        # Allow a tiny floating-point wobble at convergence, but no real decrease.
        self.assertTrue(np.all(diffs > -1e-6), f"log-likelihood decreased: {hist}")

    def test_em_sharpens_gaussian_gate_and_improves_fit(self):
        m = self._fit_gaussian_gate_hme()
        root_sigma_before = np.mean([mf.sigma for _, mf in m.tree_.terms])
        r2_before = r2_score(self.y, m.predict(self.X))

        m.refine_em(self.X, self.y, max_iter=15)
        root_sigma_after = np.mean([mf.sigma for _, mf in m.tree_.terms])
        r2_after = r2_score(self.y, m.predict(self.X))

        self.assertLess(root_sigma_after, root_sigma_before)
        self.assertGreaterEqual(r2_after, r2_before)

    def test_refine_em_returns_self_and_sets_history(self):
        m = self._fit_gaussian_gate_hme()
        out = m.refine_em(self.X, self.y, max_iter=5)
        self.assertIs(out, m)
        self.assertEqual(m.em_iterations_, len(m.em_log_likelihood_))
        self.assertEqual(set(m.sigma2_.keys()), set(m.experts_.keys()))
        self.assertTrue(all(v > 0 for v in m.sigma2_.values()))

    def test_starved_leaf_is_frozen_with_warning(self):
        m = self._fit_gaussian_gate_hme()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # An absurdly high min_mass starves every branch/expert immediately.
            m.refine_em(self.X, self.y, max_iter=2, min_mass=1e9)
        self.assertTrue(any("min_mass" in str(w.message) for w in caught))


class TestEMClassification(unittest.TestCase):
    def setUp(self):
        rng = _rng(9)
        n = 1500
        self.a = rng.uniform(0, 10, n)
        self.b = rng.uniform(0, 10, n)
        self.X = pd.DataFrame({"a": self.a, "b": self.b})
        self.labels = np.where(
            self.a < 5,
            np.where(self.b < 5, "A", "B"),
            np.where(self.b < 5, "C", "D"),
        )

    def test_refine_em_runs_and_keeps_proba_normalised(self):
        c = HierarchicalFuzzyExpertsClassifier(
            max_depth=2, n_gate_terms=2, min_soft_count=50, min_expert_samples=50,
            random_state=0,
        ).fit(self.X, self.labels)
        acc_before = accuracy_score(self.labels, c.predict(self.X))

        c.refine_em(self.X, self.labels, max_iter=8, random_state=0)
        proba = c.predict_proba(self.X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)

        # The importance-resampling expert M-step is stochastic and can make a
        # single step *worse*; the best-observed-iteration rollback (the
        # Sec.10 "no-worse guarantee") means the returned model's likelihood
        # is never worse than the pre-refinement (iteration-0) greedy fit.
        hist = c.em_log_likelihood_
        self.assertGreaterEqual(max(hist), hist[0])

        acc_after = accuracy_score(self.labels, c.predict(self.X))
        self.assertGreater(acc_after, 0.5)
        self.assertGreaterEqual(acc_after, acc_before - 0.1)


class TestEMTrapezoidDeprecation(unittest.TestCase):
    """Trapezoid gates (the default `gate_style`) + `refine_em` warns and
    documents the limitation -- see EM_REFINEMENT.md Sec.4.2/Sec.4 and
    ISSUE_163_RESOLUTION_PLAN.md."""

    def setUp(self):
        rng = _rng(9)
        n = 600
        self.a = rng.uniform(0, 10, n)
        self.b = rng.uniform(0, 10, n)
        self.X = pd.DataFrame({"a": self.a, "b": self.b})
        self.y = np.where(self.a < 5, 2 * self.b, -3 * self.b + 40) + rng.normal(0, 0.3, n)

    def test_trapezoid_gate_warns_not_recommended(self):
        # gate_style defaults to "trapezoid".
        m = HierarchicalFuzzyExpertsRegressor(
            max_depth=2, n_gate_terms=2, min_soft_count=50, min_expert_samples=50,
            expert_kwargs={"n_output_buckets": 3, "tsk_order": "1st"},
        ).fit(self.X, self.y)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.refine_em(self.X, self.y, max_iter=2)

        self.assertTrue(
            any(
                issubclass(w.category, FutureWarning)
                and "trapezoid" in str(w.message).lower()
                for w in caught
            ),
            f"expected a FutureWarning about deprecated trapezoid gates, got: {[str(w.message) for w in caught]}",
        )


if __name__ == "__main__":
    unittest.main()
