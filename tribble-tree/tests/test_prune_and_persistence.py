"""Tests for cost-complexity-style pruning (fuzzytree/prune.py) and model
persistence (fuzzytree/persistence.py)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from fuzzytree import (
    FuzzyClassificationTree,
    FuzzyRegressionTree,
    HierarchicalFuzzyExpertsRegressor,
    load_model,
    save_model,
)


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


class TestPersistence(unittest.TestCase):
    def test_regression_tree_round_trip(self):
        rng = _rng(2)
        n = 400
        x = rng.uniform(-3, 3, n)
        X = pd.DataFrame({"x": x})
        z = x / (x**2 + 1) + rng.normal(0, 0.01, n)
        m = FuzzyRegressionTree(tsk_order="1st", max_depth=2, n_terms=3, min_soft_count=10).fit(X, z)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tree.pkl"
            save_model(m, path)
            loaded = load_model(path)
            np.testing.assert_allclose(m.predict(X), loaded.predict(X))
            self.assertEqual(type(loaded), type(m))

    def test_hme_round_trip(self):
        rng = _rng(9)
        n = 800
        a, b = rng.uniform(0, 10, n), rng.uniform(0, 10, n)
        X = pd.DataFrame({"a": a, "b": b})
        y = np.where(a < 5, 2 * b, -3 * b + 40) + rng.normal(0, 0.3, n)
        m = HierarchicalFuzzyExpertsRegressor(
            max_depth=2, n_gate_terms=2, min_soft_count=50, min_expert_samples=50,
            expert_kwargs={"n_output_buckets": 3, "tsk_order": "1st"},
        ).fit(X, y)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "hme.pkl"
            save_model(m, path)
            loaded = load_model(path)
            np.testing.assert_allclose(m.predict(X), loaded.predict(X))
            self.assertEqual(r2_score(y, loaded.predict(X)), r2_score(y, m.predict(X)))

    def test_save_unfitted_raises(self):
        m = FuzzyRegressionTree()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                save_model(m, Path(d) / "unfit.pkl")

    def test_load_non_model_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "notamodel.pkl"
            import pickle

            with open(path, "wb") as f:
                pickle.dump({"foo": "bar"}, f)
            with self.assertRaises(ValueError):
                load_model(path)


if __name__ == "__main__":
    unittest.main()
