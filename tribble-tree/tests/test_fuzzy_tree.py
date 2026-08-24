"""Tests for the fuzzy tree module (unittest, run via pytest)."""

import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from fuzzytree import (
    AUTO,
    FuzzyClassificationTree,
    FuzzyRegressionTree,
    HierarchicalFuzzyExpertsClassifier,
    HierarchicalFuzzyExpertsRegressor,
    MimoFuzzyTreeRegressor,
    NodePin,
    VariablePlan,
    render_hme_text,
    render_tree_text,
)
from fuzzytree.firing import compute_leaf_firing
from fuzzytree.hme import compute_responsibilities
from fuzzytree.plan import resolve_split_variable
from fuzzytree.terms import build_split_terms


def _rng(seed=0):
    return np.random.default_rng(seed)


class TestTerms(unittest.TestCase):
    def test_terms_are_ordered_and_labelled(self):
        rng = _rng(1)
        vals = rng.uniform(0, 10, 300)
        w = np.ones_like(vals)
        terms = build_split_terms(vals, w, n_terms=3, labels=("Low", "Med", "High"))
        self.assertEqual([lbl for lbl, _ in terms], ["Low", "Med", "High"])

    def test_open_shoulders_cover_out_of_range(self):
        """Points beyond the training range must still get membership 1.0 on the
        extreme terms, so firing never collapses to all-zero."""
        vals = np.linspace(0, 10, 200)
        w = np.ones_like(vals)
        terms = build_split_terms(vals, w, n_terms=3)
        low_mf = terms[0][1]
        high_mf = terms[-1][1]
        self.assertAlmostEqual(float(low_mf.evaluate(np.array([-1000.0]))[0]), 1.0)
        self.assertAlmostEqual(float(high_mf.evaluate(np.array([1000.0]))[0]), 1.0)

    def test_degenerate_variable_returns_no_terms(self):
        vals = np.full(100, 3.0)
        w = np.ones_like(vals)
        self.assertEqual(build_split_terms(vals, w, n_terms=3), [])


class TestRegression(unittest.TestCase):
    def setUp(self):
        rng = _rng(2)
        n = 600
        x = rng.uniform(-3, 3, n)
        self.X = pd.DataFrame({"x": x})
        self.z = x / (x**2 + 1) + rng.normal(0, 0.01, n)

    def test_fit_predict_shapes_and_accuracy(self):
        m = FuzzyRegressionTree(tsk_order="1st", max_depth=2, n_terms=3, min_soft_count=10)
        m.fit(self.X, self.z)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (len(self.X),))
        self.assertGreater(r2_score(self.z, pred), 0.9)

    def test_first_order_beats_zeroth(self):
        r0 = FuzzyRegressionTree(tsk_order="0th", max_depth=2, min_soft_count=10).fit(self.X, self.z)
        r1 = FuzzyRegressionTree(tsk_order="1st", max_depth=2, min_soft_count=10).fit(self.X, self.z)
        self.assertGreaterEqual(
            r2_score(self.z, r1.predict(self.X)),
            r2_score(self.z, r0.predict(self.X)),
        )

    def test_out_of_range_prediction_is_finite(self):
        m = FuzzyRegressionTree(tsk_order="0th", max_depth=2, min_soft_count=10).fit(self.X, self.z)
        far = pd.DataFrame({"x": [-100.0, 100.0]})
        pred = m.predict(far)
        self.assertTrue(np.all(np.isfinite(pred)))
        # With open shoulders, an extrapolated point must actually fire a leaf
        # (non-zero), not fall through to the zero-firing fallback.
        self.assertTrue(np.all(np.abs(pred) > 0))


class TestVariablePlan(unittest.TestCase):
    def setUp(self):
        rng = _rng(3)
        n = 600
        self.a = rng.uniform(0, 10, n)
        self.b = rng.uniform(0, 10, n)
        self.X = pd.DataFrame({"a": self.a, "b": self.b})
        self.z = 2 * self.a - 3 * self.b + rng.normal(0, 0.1, n)

    def test_level_order_controls_split_variables(self):
        plan = VariablePlan(
            level_order=("a", "b"), max_depth=2, default_n_terms=2, max_terms_per_var=2
        )
        m = FuzzyRegressionTree(variable_plan=plan, tsk_order="1st", min_soft_count=10).fit(
            self.X, self.z
        )
        self.assertEqual(m.tree_.split_var, "a")
        for child in m.tree_.children:
            self.assertEqual(child.split_var, "b")

    def test_node_pin_places_variable_at_path(self):
        plan = VariablePlan(
            level_order=("a", None),
            pins=(NodePin(path=("Low",), variable="b"),),
            max_depth=2,
            default_n_terms=2,
            max_terms_per_var=2,
        )
        m = FuzzyRegressionTree(variable_plan=plan, tsk_order="1st", min_soft_count=10).fit(
            self.X, self.z
        )
        self.assertEqual(m.tree_.split_var, "a")
        # The "Low" child (terms sorted so index 0 is Low) is pinned to 'b'.
        self.assertEqual(m.tree_.children[0].split_var, "b")

    def test_exclude_is_honored(self):
        plan = VariablePlan(exclude=frozenset({"b"}), max_depth=2, default_n_terms=2)
        m = FuzzyRegressionTree(variable_plan=plan, tsk_order="1st", min_soft_count=10).fit(
            self.X, self.z
        )
        split_vars = {n.split_var for n in m.tree_.iter_nodes() if not n.is_leaf}
        self.assertNotIn("b", split_vars)

    def test_linear_plan_recovers_linear_target(self):
        plan = VariablePlan(level_order=("a", "b"), max_depth=2, default_n_terms=2, max_terms_per_var=2)
        m = FuzzyRegressionTree(variable_plan=plan, tsk_order="1st", min_soft_count=10).fit(
            self.X, self.z
        )
        self.assertGreater(r2_score(self.z, m.predict(self.X)), 0.95)


class TestCapsAndFiring(unittest.TestCase):
    def test_max_depth_and_max_leaves(self):
        rng = _rng(4)
        n = 800
        X = pd.DataFrame({f"x{i}": rng.uniform(0, 1, n) for i in range(4)})
        z = X.sum(axis=1).to_numpy() + rng.normal(0, 0.01, n)
        m = FuzzyRegressionTree(
            max_depth=2, n_terms=2, max_leaves=3, min_soft_count=1, min_gain=0.0
        ).fit(X, z)
        self.assertLessEqual(m.n_leaves_, 3)
        depths = [nd.depth for nd in m.tree_.iter_nodes()]
        self.assertLessEqual(max(depths), 2)

    def test_leaf_firing_matrix_shape(self):
        rng = _rng(5)
        n = 300
        X = pd.DataFrame({"x": rng.uniform(0, 1, n)})
        z = X["x"].to_numpy()
        m = FuzzyRegressionTree(max_depth=2, min_soft_count=5).fit(X, z)
        fs = compute_leaf_firing(m.tree_, X[m.top_features_], m.n_leaves_, m.t_norm)
        self.assertEqual(fs.shape, (n, m.n_leaves_))
        self.assertTrue(np.all(fs >= 0))


class TestClassification(unittest.TestCase):
    def setUp(self):
        rng = _rng(6)
        n = 600
        self.a = rng.uniform(0, 10, n)
        self.b = rng.uniform(0, 10, n)
        self.X = pd.DataFrame({"a": self.a, "b": self.b})
        self.y = np.where(
            self.a < 5,
            np.where(self.b < 5, "LL", "LH"),
            np.where(self.b < 5, "HL", "HH"),
        )

    def test_ambiguity_classifier(self):
        clf = FuzzyClassificationTree(
            criterion="ambiguity", max_depth=2, n_terms=2, min_soft_count=10
        ).fit(self.X, self.y)
        self.assertGreater(accuracy_score(self.y, clf.predict(self.X)), 0.85)

    def test_proba_rows_sum_to_one(self):
        clf = FuzzyClassificationTree(max_depth=2, n_terms=2, min_soft_count=10).fit(
            self.X, self.y
        )
        proba = clf.predict_proba(self.X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)

    def test_info_gain_classifier(self):
        # Fits and predicts without error; a much looser bound than the
        # "ambiguity"/default criterion since info_gain picks split *thresholds*
        # by fuzzy entropy reduction rather than the ambiguity criterion this
        # dataset was tuned against (observed ~0.73 on this fixture).
        clf = FuzzyClassificationTree(
            criterion="info_gain", max_depth=2, n_terms=2, min_soft_count=10
        ).fit(self.X, self.y)
        self.assertGreater(accuracy_score(self.y, clf.predict(self.X)), 0.6)

    def test_differentiation_classifier(self):
        # "differentiation" scores variable *relevance*, not partition quality
        # (see its docstring: "a prefilter, not a sole splitter"), so as the
        # sole split criterion it's expected to underperform ambiguity/info_gain
        # here (observed ~0.52, vs. 0.25 random-chance baseline for 4 classes).
        clf = FuzzyClassificationTree(
            criterion="differentiation", max_depth=2, n_terms=2, min_soft_count=10
        ).fit(self.X, self.y)
        self.assertGreater(accuracy_score(self.y, clf.predict(self.X)), 0.4)


class TestVariablePlanSerialization(unittest.TestCase):
    """`VariablePlan.to_dict`/`from_dict` round-trip, used by config-file
    front ends per the class docstring."""

    def test_round_trip_default_plan(self):
        plan = VariablePlan()
        restored = VariablePlan.from_dict(plan.to_dict())
        self.assertEqual(restored, plan)

    def test_round_trip_with_pins_and_overrides(self):
        plan = VariablePlan(
            level_order=("a", None, "b"),
            pins=(NodePin(path=("Low",), variable="b", n_terms=2),),
            criterion="ambiguity",
            exclude=frozenset({"c", "d"}),
            max_depth=4,
            default_n_terms=4,
            max_terms_per_var=5,
            term_labels=("Lo", "Hi"),
            no_reuse_on_path=True,
            term_style="gaussian",
        )
        restored = VariablePlan.from_dict(plan.to_dict())
        self.assertEqual(restored, plan)

    def test_from_dict_fills_defaults_for_missing_keys(self):
        restored = VariablePlan.from_dict({})
        self.assertEqual(restored, VariablePlan())


class TestResolveSplitVariableFallbacks(unittest.TestCase):
    """`resolve_split_variable`'s three warn-and-fall-back-to-auto branches:
    a pin naming an excluded variable, a pin naming an unavailable variable,
    and a level_order slot naming an unavailable variable."""

    def test_pin_naming_excluded_variable_falls_back_to_auto(self):
        plan = VariablePlan(
            pins=(NodePin(path=(), variable="b"),), exclude=frozenset({"b"})
        )
        with self.assertWarns(UserWarning):
            decision, _ = resolve_split_variable(
                plan, path=(), depth=0, available_vars=["a", "b"], var_by_path={}
            )
        self.assertEqual(decision, AUTO)

    def test_pin_naming_unavailable_variable_falls_back_to_auto(self):
        plan = VariablePlan(pins=(NodePin(path=(), variable="c"),))
        with self.assertWarns(UserWarning):
            decision, _ = resolve_split_variable(
                plan, path=(), depth=0, available_vars=["a", "b"], var_by_path={}
            )
        self.assertEqual(decision, AUTO)

    def test_level_order_naming_unavailable_variable_falls_back_to_auto(self):
        plan = VariablePlan(level_order=("c",))
        with self.assertWarns(UserWarning):
            decision, _ = resolve_split_variable(
                plan, path=(), depth=0, available_vars=["a", "b"], var_by_path={}
            )
        self.assertEqual(decision, AUTO)

    def test_valid_pin_is_honored_without_warning(self):
        plan = VariablePlan(pins=(NodePin(path=(), variable="b", n_terms=2),))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            decision, n_terms = resolve_split_variable(
                plan, path=(), depth=0, available_vars=["a", "b"], var_by_path={}
            )
        self.assertEqual(decision, "b")
        self.assertEqual(n_terms, 2)


class TestMimoAndRender(unittest.TestCase):
    def test_mimo_regressor(self):
        rng = _rng(7)
        n = 500
        X = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)})
        Y = pd.DataFrame({"o1": X["a"] + X["b"], "o2": X["a"] - X["b"]})
        m = MimoFuzzyTreeRegressor(tsk_order="1st", max_depth=2, min_soft_count=10).fit(X, Y)
        pred = m.predict(X)
        self.assertEqual(list(pred.columns), ["o1", "o2"])
        self.assertEqual(pred.shape, (n, 2))

    def test_render_text_nonempty(self):
        rng = _rng(8)
        n = 300
        X = pd.DataFrame({"x": rng.uniform(-2, 2, n)})
        z = X["x"].to_numpy() ** 2
        m = FuzzyRegressionTree(max_depth=2, min_soft_count=10).fit(X, z)
        text = render_tree_text(m)
        self.assertIn("FuzzyTree", text)
        self.assertIn("IF x is", text)


class TestHME(unittest.TestCase):
    """Hierarchical mixture of fuzzy experts."""

    def setUp(self):
        rng = _rng(9)
        n = 1500
        self.a = rng.uniform(0, 10, n)
        self.b = rng.uniform(0, 10, n)
        self.X = pd.DataFrame({"a": self.a, "b": self.b})
        # Piecewise-linear regime target: a routes, b drives the local expert.
        self.y = np.where(self.a < 5, 2 * self.b, -3 * self.b + 40) + rng.normal(0, 0.3, n)
        self.labels = np.where(
            self.a < 5,
            np.where(self.b < 5, "A", "B"),
            np.where(self.b < 5, "C", "D"),
        )

    def test_regressor_fits_and_blends(self):
        m = HierarchicalFuzzyExpertsRegressor(
            max_depth=2, n_gate_terms=2, min_soft_count=50, min_expert_samples=50,
            expert_kwargs={"n_output_buckets": 3, "tsk_order": "1st"},
        ).fit(self.X, self.y)
        self.assertGreaterEqual(m.n_leaves_, 2)
        self.assertGreater(r2_score(self.y, m.predict(self.X)), 0.6)
        # Every leaf has an expert.
        self.assertEqual(len(m.experts_), m.n_leaves_)

    def test_responsibilities_sum_to_one(self):
        m = HierarchicalFuzzyExpertsRegressor(
            max_depth=2, n_gate_terms=2, min_soft_count=50, min_expert_samples=50
        ).fit(self.X, self.y)
        R = compute_responsibilities(m.tree_, self.X[m.gate_features_], m.n_leaves_)
        self.assertEqual(R.shape, (len(self.X), m.n_leaves_))
        np.testing.assert_allclose(R.sum(axis=1), 1.0, rtol=1e-6)

    def test_classifier(self):
        c = HierarchicalFuzzyExpertsClassifier(
            max_depth=2, n_gate_terms=2, min_soft_count=50, min_expert_samples=50
        ).fit(self.X, self.labels)
        self.assertGreater(accuracy_score(self.labels, c.predict(self.X)), 0.8)
        proba = c.predict_proba(self.X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)

    def test_user_pinned_gate_variable(self):
        plan = VariablePlan(level_order=("a", "b"), max_depth=2, default_n_terms=2, max_terms_per_var=2)
        m = HierarchicalFuzzyExpertsRegressor(
            variable_plan=plan, min_soft_count=50, min_expert_samples=50
        ).fit(self.X, self.y)
        self.assertEqual(m.tree_.split_var, "a")  # root routes on the pinned variable

    def test_render_hme_text(self):
        m = HierarchicalFuzzyExpertsRegressor(
            max_depth=2, n_gate_terms=2, min_soft_count=50, min_expert_samples=50
        ).fit(self.X, self.y)
        text = render_hme_text(m)
        self.assertIn("HierarchicalFuzzyExperts", text)
        self.assertIn("ROUTE", text)
        self.assertIn("expert", text)


if __name__ == "__main__":
    unittest.main()
