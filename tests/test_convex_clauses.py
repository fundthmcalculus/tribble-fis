"""Tests for the `convex_clauses_only` option (issue #91).

Covers both producers of explicit `Rule` objects:
* `GaussianMixtureModel.to_simple_model` (flat/cascade path, interval-based
  convexity via `gauss_data.split_convex_clauses`).
* `RuspiniPartitionModel.to_simple_model` (shared-knot path, cheaper
  contiguous-index convexity via `ruspini._split_contiguous_runs`).

In both cases a clause covering two disjoint (non-touching) regions of one
feature is deliberately constructed, and splitting it must not change any
prediction -- only spread the same logical clause across more rules.
"""

import unittest

import numpy as np
import pandas as pd

from tribblefis.gauss_data import (
    GaussianMembership,
    GaussianMixtureModel,
    FeatureModel,
    LabelModel,
    mf_interval,
    split_convex_clauses,
)
from tribblefis.gauss_math import simple_gaussian_predict
from tribblefis.ruspini import RuspiniPartitionModel, build_triangular_partition


class TestMfInterval(unittest.TestCase):
    def test_gaussian_interval_default_k(self):
        mf = GaussianMembership.create(mu=10.0, sigma=2.0)
        lo, hi = mf_interval(mf)
        self.assertAlmostEqual(lo, 4.0)
        self.assertAlmostEqual(hi, 16.0)


class TestSplitConvexClausesInterval(unittest.TestCase):
    def test_disjoint_clause_splits_into_two(self):
        near = GaussianMembership.create(mu=0.0, sigma=0.1)
        far = GaussianMembership.create(mu=100.0, sigma=0.1)
        lookup = {near.id: near, far.id: far}
        combos = split_convex_clauses({"x": [near.id, far.id]}, lookup)
        self.assertEqual(len(combos), 2)
        groups = sorted(tuple(sorted(c["x"], key=str)) for c in combos)
        self.assertIn((near.id,), groups)
        self.assertIn((far.id,), groups)

    def test_touching_clause_stays_single_combination(self):
        left = GaussianMembership.create(mu=0.0, sigma=1.0)
        right = GaussianMembership.create(mu=1.0, sigma=1.0)
        lookup = {left.id: left, right.id: right}
        combos = split_convex_clauses({"x": [left.id, right.id]}, lookup)
        self.assertEqual(len(combos), 1)

    def test_multi_feature_cartesian_product(self):
        a1 = GaussianMembership.create(mu=0.0, sigma=0.1)
        a2 = GaussianMembership.create(mu=100.0, sigma=0.1)
        b1 = GaussianMembership.create(mu=0.0, sigma=1.0)
        lookup = {a1.id: a1, a2.id: a2, b1.id: b1}
        combos = split_convex_clauses({"x": [a1.id, a2.id], "y": [b1.id]}, lookup)
        # feature x splits into 2 convex groups, feature y stays as 1 -> 2 combos.
        self.assertEqual(len(combos), 2)
        for combo in combos:
            self.assertEqual(combo["y"], [b1.id])


class TestGaussianMixtureConvexClausesOnly(unittest.TestCase):
    def _model(self):
        # Feature "x": label 0 has two far-apart Gaussians (disjoint clause);
        # label 1 has a single Gaussian in between (already convex).
        near = GaussianMembership.create(mu=0.0, sigma=0.1)
        far = GaussianMembership.create(mu=100.0, sigma=0.1)
        middle = GaussianMembership.create(mu=50.0, sigma=0.1)
        return GaussianMixtureModel(
            feature_models={
                "x": FeatureModel(
                    label_models={
                        0: LabelModel(memberships=[near, far]),
                        1: LabelModel(memberships=[middle]),
                    }
                )
            }
        )

    def test_default_is_unchanged(self):
        gm = self._model()
        simple = gm.to_simple_model()
        self.assertEqual(len(simple.rules), 2)

    def test_convex_clauses_only_splits_disjoint_label(self):
        gm = self._model()
        simple = gm.to_simple_model(convex_clauses_only=True)
        # Label 0's disjoint clause becomes 2 rules; label 1's stays 1 rule.
        rules_for_0 = [r for r in simple.rules if r.consequent == 0]
        rules_for_1 = [r for r in simple.rules if r.consequent == 1]
        self.assertEqual(len(rules_for_0), 2)
        self.assertEqual(len(rules_for_1), 1)
        for rule in simple.rules:
            ids = rule.antecedents["x"]
            mfs = simple.get_mfs(ids)
            los_his = sorted(mf_interval(mf) for mf in mfs)
            # every emitted clause is itself already convex (no gaps between
            # consecutive merged intervals within the same rule).
            for (lo1, hi1), (lo2, hi2) in zip(los_his, los_his[1:]):
                self.assertLessEqual(lo2, hi1)

    def test_predictions_unchanged_by_splitting(self):
        gm = self._model()
        X = pd.DataFrame({"x": [0.0, 100.0, 50.0]})
        baseline = simple_gaussian_predict(X, gm.to_simple_model())
        split = simple_gaussian_predict(X, gm.to_simple_model(convex_clauses_only=True))
        np.testing.assert_array_equal(baseline, split)


class TestRuspiniConvexClausesOnly(unittest.TestCase):
    def _model(self):
        apex = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        term_ids = build_triangular_partition(apex)
        return RuspiniPartitionModel(
            feature_order=["x"],
            apexes={"x": apex},
            term_ids={"x": [t.id for t in term_ids]},
            # Label 0: disjoint indices {0, 3}; label 1: contiguous {1, 2}.
            rules=[(0, {"x": [0, 3]}), (1, {"x": [1, 2]})],
        )

    def test_default_is_unchanged(self):
        rm = self._model()
        simple = rm.to_simple_model()
        self.assertEqual(len(simple.rules), 2)

    def test_convex_clauses_only_splits_disjoint_indices(self):
        rm = self._model()
        simple = rm.to_simple_model(convex_clauses_only=True)
        rules_for_0 = [r for r in simple.rules if r.consequent == 0]
        rules_for_1 = [r for r in simple.rules if r.consequent == 1]
        self.assertEqual(len(rules_for_0), 2)
        self.assertEqual(len(rules_for_1), 1)

    def test_predictions_unchanged_by_splitting(self):
        rm = self._model()
        X = pd.DataFrame({"x": [-1.0, 0.0, 1.5, 3.0, 5.0]})
        direct = rm.predict(X)
        baseline = simple_gaussian_predict(X, rm.to_simple_model())
        split = simple_gaussian_predict(X, rm.to_simple_model(convex_clauses_only=True))
        np.testing.assert_array_equal(np.asarray(direct), baseline)
        np.testing.assert_array_equal(baseline, split)


if __name__ == "__main__":
    unittest.main()
