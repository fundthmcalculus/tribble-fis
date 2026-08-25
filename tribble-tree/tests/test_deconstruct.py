"""Tests for fuzzytree.topology / fuzzytree.deconstruct (unittest, run via pytest)."""

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from fuzzytree import DeconstructedHierarchicalRegressor, TopologyNode, parse_topology


def _rng(seed=0):
    return np.random.default_rng(seed)


class TestParseTopology(unittest.TestCase):
    FEATURES = ["a", "b", "c", "d"]

    def test_simple_two_level_tree(self):
        topo = {"ROOT": ["G1", "G2"], "G1": ["a", "b"], "G2": ["c", "d"]}
        root = parse_topology(topo, self.FEATURES)
        self.assertIsInstance(root, TopologyNode)
        self.assertEqual(root.name, "ROOT")
        self.assertFalse(root.is_leaf)
        self.assertEqual({c.name for c in root.children}, {"G1", "G2"})
        leaves = {leaf.name: sorted(leaf.own_features) for leaf in root.iter_leaves()}
        self.assertEqual(leaves, {"G1": ["a", "b"], "G2": ["c", "d"]})

    def test_no_root_raises(self):
        topo = {"A": ["B"], "B": ["A"]}
        with self.assertRaises(ValueError):
            parse_topology(topo, self.FEATURES)

    def test_multiple_roots_raises(self):
        topo = {"A": ["a"], "B": ["b"]}
        with self.assertRaises(ValueError):
            parse_topology(topo, self.FEATURES)

    def test_unknown_child_raises(self):
        topo = {"ROOT": ["a", "nonexistent"]}
        with self.assertRaises(ValueError):
            parse_topology(topo, self.FEATURES)

    def test_mixed_branch_and_leaf_children_raises(self):
        topo = {"ROOT": ["G1", "a"], "G1": ["b", "c"]}
        with self.assertRaises(ValueError):
            parse_topology(topo, self.FEATURES)

    def test_unreachable_node_raises(self):
        topo = {"ROOT": ["a", "b"], "ORPHAN": ["c", "d"]}
        with self.assertRaises(ValueError):
            parse_topology(topo, self.FEATURES)

    def test_node_name_colliding_with_feature_raises(self):
        topo = {"a": ["b", "c"]}
        with self.assertRaises(ValueError):
            parse_topology(topo, self.FEATURES)

    def test_diamond_reuse_raises(self):
        topo = {"ROOT": ["G1", "G2"], "G1": ["H"], "G2": ["H"], "H": ["a", "b"]}
        with self.assertRaises(ValueError):
            parse_topology(topo, self.FEATURES)


class TestDeconstructedHierarchicalRegressor(unittest.TestCase):
    def setUp(self):
        rng = _rng(3)
        n = 2000
        self.a = rng.uniform(0, 10, n)
        self.b = rng.uniform(0, 10, n)
        self.c = rng.uniform(0, 10, n)
        self.d = rng.uniform(0, 10, n)
        self.X = pd.DataFrame({"a": self.a, "b": self.b, "c": self.c, "d": self.d})
        # No cross-group interaction: G1 = f(a, b), G2 = g(c, d), y = G1 + G2.
        self.g1 = np.sin(self.a / 2) * 5 + 0.5 * self.b
        self.g2 = np.cos(self.c / 3) * 4 - 0.3 * self.d
        self.y = self.g1 + self.g2 + rng.normal(0, 0.2, n)
        self.topology = {"ROOT": ["G1", "G2"], "G1": ["a", "b"], "G2": ["c", "d"]}

    def test_leaf_only_sees_its_own_feature_group(self):
        m = DeconstructedHierarchicalRegressor(
            flat_regressor_kwargs={"n_output_buckets": 4},
        ).fit(self.X, self.y, self.topology)
        g1_state = m.node_state_["G1"]
        g2_state = m.node_state_["G2"]
        self.assertEqual(g1_state["kind"], "leaf")
        self.assertEqual(g2_state["kind"], "leaf")
        self.assertEqual(set(g1_state["top_n_todo"]), {"a", "b"})
        self.assertEqual(set(g2_state["top_n_todo"]), {"c", "d"})

    def test_end_to_end_fit_predict_shape_and_fit_quality(self):
        m = DeconstructedHierarchicalRegressor(
            flat_regressor_kwargs={"n_output_buckets": 4},
        ).fit(self.X, self.y, self.topology)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (len(self.X),))
        self.assertGreater(r2_score(self.y, pred), 0.5)

    def test_branch_combiner_recovers_affine_combination(self):
        # Root's branch combiner should recover an affine map of its two
        # children's outputs close to the true (unit-weight) sum they were
        # generated from.
        m = DeconstructedHierarchicalRegressor(
            flat_regressor_kwargs={"n_output_buckets": 4},
        ).fit(self.X, self.y, self.topology)
        root_state = m.node_state_["ROOT"]
        self.assertEqual(root_state["kind"], "branch")
        a_coeffs = root_state["corr_terms"][0]
        # Two children -> two affine coefficients, both should be positive
        # and roughly comparable in scale since y = 1*G1 + 1*G2.
        self.assertEqual(a_coeffs.shape, (2,))
        self.assertTrue(np.all(a_coeffs > 0.3))

    def test_leaf_target_override_is_used(self):
        # Supervise G1's leaf directly on a known target instead of y; its
        # own fitted output should then track that target, not y.
        m = DeconstructedHierarchicalRegressor(
            flat_regressor_kwargs={"n_output_buckets": 4},
        ).fit(self.X, self.y, self.topology, leaf_targets={"G1": self.g1})
        # Recompute G1's leaf output directly to compare against g1 (the
        # override target) rather than against the blended y.
        g1_node = next(n for n in m.root_.children if n.name == "G1")
        g1_pred = m._predict_node(g1_node, self.X)
        self.assertGreater(r2_score(self.g1, g1_pred), 0.5)

    def test_leaf_with_no_surviving_features_falls_back_to_constant(self):
        # y depends only on a, b; with top_n=2 the flat model's own feature
        # selection should drop c and d entirely, starving G2's leaf.
        y_ab_only = 3 * self.a - 2 * self.b
        m = DeconstructedHierarchicalRegressor(
            flat_regressor_kwargs={"n_output_buckets": 4, "top_n": 2},
        ).fit(self.X, y_ab_only, self.topology)
        self.assertEqual(m.node_state_["G2"]["kind"], "constant")
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (len(self.X),))
        self.assertTrue(np.all(np.isfinite(pred)))


if __name__ == "__main__":
    unittest.main()
