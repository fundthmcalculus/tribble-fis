"""Tests for post-hoc split-gain pruning (fuzzytree/prune.py)."""

import unittest

import numpy as np
import pandas as pd

from fuzzytree import FuzzyClassificationTree, FuzzyRegressionTree


def _rng(seed=0):
    return np.random.default_rng(seed)


class TestPruning(unittest.TestCase):
    def setUp(self):
        rng = _rng(4)
        n = 800
        self.X = pd.DataFrame({f"x{i}": rng.uniform(0, 1, n) for i in range(4)})
        self.z = self.X.sum(axis=1).to_numpy() + rng.normal(0, 0.01, n)

    def test_zero_alpha_is_a_no_op(self):
        m0 = FuzzyRegressionTree(max_depth=3, n_terms=2, min_soft_count=5, min_gain=1e-4).fit(
            self.X, self.z
        )
        m_alpha0 = FuzzyRegressionTree(
            max_depth=3, n_terms=2, min_soft_count=5, min_gain=1e-4, ccp_alpha=0.0
        ).fit(self.X, self.z)
        self.assertEqual(m0.n_leaves_, m_alpha0.n_leaves_)
        np.testing.assert_allclose(m0.predict(self.X), m_alpha0.predict(self.X))

    def test_leaf_count_is_nonincreasing_in_alpha(self):
        leaf_counts = []
        for alpha in [0.0, 0.05, 0.1, 0.2, 0.5]:
            m = FuzzyRegressionTree(
                max_depth=3, n_terms=2, min_soft_count=5, min_gain=1e-4, ccp_alpha=alpha
            ).fit(self.X, self.z)
            leaf_counts.append(m.n_leaves_)
        self.assertEqual(leaf_counts, sorted(leaf_counts, reverse=True))
        self.assertGreater(leaf_counts[0], leaf_counts[-1])  # some pruning actually occurred

    def test_pruned_predictions_stay_finite_and_leaves_renumbered(self):
        m = FuzzyRegressionTree(
            max_depth=3, n_terms=2, min_soft_count=5, min_gain=1e-4, ccp_alpha=0.1
        ).fit(self.X, self.z)
        pred = m.predict(self.X)
        self.assertTrue(np.all(np.isfinite(pred)))
        leaf_ids = sorted(n.leaf_id for n in m.tree_.iter_leaves())
        self.assertEqual(leaf_ids, list(range(m.n_leaves_)))
        self.assertEqual(m.corr_terms_.shape[0], m.n_leaves_)

    def test_classification_pruning_collapses_to_one_leaf_at_high_alpha(self):
        rng = _rng(6)
        n = 600
        a, b = rng.uniform(0, 10, n), rng.uniform(0, 10, n)
        X = pd.DataFrame({"a": a, "b": b})
        y = np.where(a < 5, np.where(b < 5, "LL", "LH"), np.where(b < 5, "HL", "HH"))
        clf = FuzzyClassificationTree(
            criterion="ambiguity", max_depth=2, n_terms=2, min_soft_count=10, ccp_alpha=5.0
        ).fit(X, y)
        self.assertEqual(clf.n_leaves_, 1)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
