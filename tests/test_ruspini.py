"""Tests for Ruspini triangular-partition models (src/tribblefis/ruspini.py).

Covers the partition-of-unity property of the triangular terms, the implicit ->
explicit "TRIBBLE" conversion, knot round-tripping, the explicit-model faithful
serialization, the sklearn estimator, and the never-worse knot refinement.
"""

import io
import contextlib
import unittest
import uuid

import numpy as np
import pandas as pd

from tribblefis.gauss_data import TriangularMembership
from tribblefis.gauss_math import create_gaussian_membership_dict, simple_gaussian_predict
from tribblefis.ruspini import (
    build_triangular_partition,
    complete_ruspini_partition,
    ruspinize_model,
    verify_partition_of_unity,
    RuspiniPartitionModel,
    RuspiniFuzzyClassifier,
)
from tribblefis.refine import refine_ruspini_partition
from tribblefis.triangle_fit import GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH


def _two_class_blobs(seed=0, n=160):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    x0 = rng.normal([0.0, 0.0], [1.0, 1.5], size=(n0, 2))
    x1 = rng.normal([3.0, 1.0], [1.0, 1.5], size=(n - n0, 2))
    X = pd.DataFrame(np.vstack([x0, x1]), columns=["a", "b"])
    y = np.array([0] * n0 + [1] * (n - n0))
    perm = rng.permutation(len(X))
    return X.iloc[perm].reset_index(drop=True), y[perm]


class TestTriangularPartition(unittest.TestCase):

    def test_partition_of_unity(self):
        for apexes in ([0.0, 1.0, 2.0, 5.0], [-3.0, 0.0, 4.0], [1.0, 2.0], [0.0, 1.0]):
            terms = build_triangular_partition(apexes)
            xs = np.linspace(min(apexes) - 5, max(apexes) + 5, 401)
            total = np.sum([t.evaluate(xs) for t in terms], axis=0)
            np.testing.assert_allclose(total, 1.0, atol=1e-9)

    def test_shoulders(self):
        terms = build_triangular_partition([0.0, 1.0, 2.0])
        # First term is a left shoulder (1 far left), last is a right shoulder.
        self.assertAlmostEqual(float(terms[0].evaluate(np.array([-100.0]))[0]), 1.0)
        self.assertAlmostEqual(float(terms[-1].evaluate(np.array([100.0]))[0]), 1.0)
        # Apex of the middle triangle fires 1 at its centre.
        self.assertAlmostEqual(float(terms[1].evaluate(np.array([1.0]))[0]), 1.0)

    def test_single_apex_is_constant_one(self):
        terms = build_triangular_partition([2.0])
        xs = np.linspace(-10, 10, 51)
        np.testing.assert_allclose(terms[0].evaluate(xs), 1.0, atol=1e-12)

    def test_verify_partition_of_unity_true_for_any_knots(self):
        for apexes in ([0.0, 1.0, 2.0, 5.0], [-3.0, 0.0, 4.0], [1.0, 2.0]):
            terms = build_triangular_partition(apexes)
            xs = np.linspace(min(apexes) - 5, max(apexes) + 5, 401)
            self.assertTrue(verify_partition_of_unity(terms, xs))

    def test_verify_partition_of_unity_false_when_missing_a_term(self):
        terms = build_triangular_partition([0.0, 1.0, 2.0])
        xs = np.linspace(-5, 5, 101)
        self.assertFalse(verify_partition_of_unity(terms[:-1], xs))


class TestRuspinize(unittest.TestCase):

    def _model(self, seed=0):
        X, y = _two_class_blobs(seed)
        gm = create_gaussian_membership_dict(X, pd.Series(y), top_n_var_names=["a", "b"], n_gaussians=1)
        return X, y, gm

    def test_convert_builds_valid_explicit_model(self):
        X, y, gm = self._model()
        rm = ruspinize_model(gm, X, y)
        self.assertGreaterEqual(rm.n_terms_total, 2)
        # One rule per class, referencing valid term indices.
        self.assertEqual(len(rm.rules), len(np.unique(y)))
        preds = rm.predict(X)
        self.assertTrue(set(np.unique(preds)).issubset(set(np.unique(y))))

    def test_knot_roundtrip(self):
        X, y, gm = self._model()
        rm = ruspinize_model(gm, X, y)
        knots = rm.extract_knots()
        rebuilt = rm.with_knots(knots)
        # Knots come back unchanged (they were already sorted per feature), and the
        # identity round-trip does not change predictions.
        np.testing.assert_allclose(rebuilt.extract_knots(), knots)
        np.testing.assert_array_equal(rm.predict(X), rebuilt.predict(X))

    def test_explicit_serialization_matches_predict(self):
        # to_simple_model + simple_gaussian_predict must match the model's predict.
        X, y, gm = self._model()
        rm = ruspinize_model(gm, X, y)
        direct = rm.predict(X)
        via_explicit = simple_gaussian_predict(X, rm.to_simple_model())
        np.testing.assert_array_equal(np.asarray(direct), np.asarray(via_explicit))

    def test_default_sigma_knots_is_mae_optimal_fit_width(self):
        import inspect

        self.assertEqual(
            inspect.signature(ruspinize_model).parameters["sigma_knots"].default,
            GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH,
        )

    def test_default_sigma_knots_adds_gaussian_width_knots_and_stays_ruspini(self):
        X, y, gm = self._model()
        centres_only = ruspinize_model(gm, X, y, sigma_knots=0.0)
        with_width = ruspinize_model(gm, X, y)  # default sigma_knots
        # Encoding each Gaussian's spread as extra knots should not shrink the
        # partition, and the result must still be a valid Ruspini partition.
        self.assertGreaterEqual(with_width.n_terms_total, centres_only.n_terms_total)
        for f in with_width.feature_order:
            terms = with_width.feature_terms()[f]
            xs = np.linspace(X[f].min() - 5, X[f].max() + 5, 201)
            self.assertTrue(verify_partition_of_unity(terms, xs))

    def test_predict_still_works_with_default_sigma_knots(self):
        X, y, gm = self._model()
        rm = ruspinize_model(gm, X, y)
        preds = rm.predict(X)
        self.assertTrue(set(np.unique(preds)).issubset(set(np.unique(y))))


class TestCompleteRuspiniPartition(unittest.TestCase):
    def test_fills_wide_gap_and_preserves_partition_of_unity(self):
        X = pd.DataFrame({"a": [0.0, 1.0, 9.0, 10.0]})
        rm = RuspiniPartitionModel(
            feature_order=["a"],
            apexes={"a": np.array([0.0, 10.0])},
            term_ids={"a": [uuid.uuid4(), uuid.uuid4()]},
            rules=[(0, {"a": [0]}), (1, {"a": [1]})],
        )
        filled = complete_ruspini_partition(rm, X, min_gap_frac=0.2)
        self.assertGreater(len(filled.apexes["a"]), len(rm.apexes["a"]))
        terms = filled.feature_terms()["a"]
        xs = np.linspace(-5, 15, 401)
        self.assertTrue(verify_partition_of_unity(terms, xs))

    def test_small_gap_is_left_alone(self):
        X = pd.DataFrame({"a": [0.0, 1.0]})
        rm = RuspiniPartitionModel(
            feature_order=["a"],
            apexes={"a": np.array([0.0, 1.0])},
            term_ids={"a": [uuid.uuid4(), uuid.uuid4()]},
            rules=[(0, {"a": [0]}), (1, {"a": [1]})],
        )
        filled = complete_ruspini_partition(rm, X, min_gap_frac=2.0)
        np.testing.assert_allclose(filled.apexes["a"], rm.apexes["a"])

    def test_new_terms_are_reachable_by_a_rule(self):
        X = pd.DataFrame({"a": [0.0, 1.0, 9.0, 10.0]})
        rm = RuspiniPartitionModel(
            feature_order=["a"],
            apexes={"a": np.array([0.0, 10.0])},
            term_ids={"a": [uuid.uuid4(), uuid.uuid4()]},
            rules=[(0, {"a": [0]}), (1, {"a": [1]})],
        )
        filled = complete_ruspini_partition(rm, X, min_gap_frac=0.2)
        referenced = {i for _, ant in filled.rules for i in ant["a"]}
        self.assertEqual(referenced, set(range(len(filled.apexes["a"]))))

    def test_predictions_are_not_disrupted(self):
        X, y, gm = TestRuspinize()._model()
        rm = ruspinize_model(gm, X, y)
        before = rm.predict(X)
        filled = complete_ruspini_partition(rm, X, min_gap_frac=0.5)
        after = filled.predict(X)
        # Filling resolution gaps should not flip predictions on the training
        # data the gaps were measured against.
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


class TestRuspiniClassifierAndRefine(unittest.TestCase):

    def test_estimator_fit_predict(self):
        X, y = _two_class_blobs()
        with contextlib.redirect_stdout(io.StringIO()):
            clf = RuspiniFuzzyClassifier(top_p=1.0, n_gaussians=1, refine=False)
            clf.fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(len(preds), len(y))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

    def test_refine_never_worsens_val_coordinate(self):
        X, y = _two_class_blobs()
        gm = create_gaussian_membership_dict(X, pd.Series(y), top_n_var_names=["a", "b"], n_gaussians=1)
        rm = ruspinize_model(gm, X, y)
        _, info = refine_ruspini_partition(rm, X, y, method="coordinate", seed=0, verbose=False)
        if info["refined"]:
            self.assertGreaterEqual(info["val_acc"], info["init_val_acc"])

    def test_refine_optimizers_backend_runs(self):
        X, y = _two_class_blobs()
        gm = create_gaussian_membership_dict(X, pd.Series(y), top_n_var_names=["a", "b"], n_gaussians=1)
        rm = ruspinize_model(gm, X, y)
        refined, info = refine_ruspini_partition(
            rm, X, y, method="optimizers", optimizer_method="ga", local_grad_optim="perturb",
            population_size=16, num_generations=5, seed=0, verbose=False,
        )
        # Same structure (knot count preserved); partition-of-unity intact.
        self.assertEqual(refined.n_terms_total, rm.n_terms_total)
        if info["refined"]:
            self.assertGreaterEqual(info["val_acc"], info["init_val_acc"])

    def test_refine_flag_on_estimator(self):
        X, y = _two_class_blobs()
        with contextlib.redirect_stdout(io.StringIO()):
            clf = RuspiniFuzzyClassifier(top_p=1.0, n_gaussians=1, refine=True, refine_method="coordinate")
            clf.fit(X, y)
        self.assertIsNotNone(clf.refine_info_)
        self.assertEqual(len(clf.predict(X)), len(y))


if __name__ == "__main__":
    unittest.main()
