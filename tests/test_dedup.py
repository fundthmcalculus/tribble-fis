"""Tests for membership-function deduplication (see issue #85).

Scope: `GaussianMixtureModel`'s dedup machinery in `gauss_data.py`, and its
tolerance knobs and wiring on `TribbleClassifier`. The cascade
(`TribbleSequenceClassifier`) and `TribbleRegressor` are intentionally not
covered here -- see the TODO(#85) comments in `gaussian_classifier.py` and
`gaussian_regressor.py` for why.
"""

import unittest

import numpy as np
import pandas as pd

from tribblefis.gauss_data import (
    FeatureModel,
    GaussianMembership,
    GaussianMixtureModel,
    LabelModel,
)
from tribblefis.gauss_math import simple_gaussian_predict, tsk_predict
from tribblefis.gaussian_classifier import TribbleClassifier


def _model_with_duplicate() -> GaussianMixtureModel:
    """A tiny 1-feature, 2-label model where label 0 carries an exact duplicate
    Gaussian alongside a distinct one."""
    dup_a = GaussianMembership.create(mu=0.0, sigma=1.0)
    dup_b = GaussianMembership.create(mu=0.0, sigma=1.0)  # exact duplicate of dup_a
    distinct = GaussianMembership.create(mu=1.0, sigma=1.0)
    other_label_mf = GaussianMembership.create(mu=5.0, sigma=1.0)

    return GaussianMixtureModel(
        feature_models={
            "f": FeatureModel(
                label_models={
                    0: LabelModel(memberships=[dup_a, dup_b, distinct]),
                    1: LabelModel(memberships=[other_label_mf]),
                }
            )
        }
    )


def _blobs(seed=0, n=160):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    x0 = rng.normal([0.0, 0.0], [1.0, 1.5], size=(n0, 2))
    x1 = rng.normal([3.0, 1.0], [1.0, 1.5], size=(n - n0, 2))
    X = pd.DataFrame(np.vstack([x0, x1]), columns=["a", "b"])
    y = np.array([0] * n0 + [1] * (n - n0))
    perm = rng.permutation(len(X))
    return X.iloc[perm].reset_index(drop=True), y[perm]


class TestGaussianMixtureModelDedup(unittest.TestCase):
    def test_identify_duplicate_membership_fcns_finds_exact_duplicate(self):
        model = _model_with_duplicate()
        duplicates = model.identify_duplicate_membership_fcns(rtol=0.0, atol=0.0)
        self.assertEqual(len(duplicates), 1)
        feature_name, label, dup_mf, src_mf = duplicates[0]
        self.assertEqual(feature_name, "f")
        self.assertEqual(label, 0)

    def test_remove_duplicate_membership_fcns_removes_one_and_reports_count(self):
        model = _model_with_duplicate()
        self.assertEqual(model.n_membership_functions, 4)
        removed = model.remove_duplicate_membership_fcns(rtol=0.0, atol=0.0)
        self.assertEqual(removed, 1)
        self.assertEqual(model.n_membership_functions, 3)

    def test_exact_tolerance_dedup_never_changes_predictions(self):
        """rtol=atol=0 dedup only ever drops bit-identical duplicates within a
        single (feature, label)'s conorm fold, so tsk_predict must be unchanged."""
        model = _model_with_duplicate()
        X = pd.DataFrame({"f": np.linspace(-3.0, 8.0, 25)})

        before = tsk_predict(X, model)
        model.remove_duplicate_membership_fcns(rtol=0.0, atol=0.0)
        after = tsk_predict(X, model)

        np.testing.assert_array_equal(before, after)

    def test_rtol_atol_knobs_are_threaded_through(self):
        """A near-duplicate pair (mu differs by 0.05) is only found once the
        tolerance is loosened enough to cover the gap -- proving `rtol`/`atol`
        actually reach `_is_close`, not just the exact-match default."""
        a = GaussianMembership.create(mu=0.0, sigma=1.0)
        b = GaussianMembership.create(mu=0.05, sigma=1.0)
        model = GaussianMixtureModel(
            feature_models={"f": FeatureModel(label_models={0: LabelModel(memberships=[a, b])})}
        )

        self.assertEqual(len(model.identify_duplicate_membership_fcns(rtol=0.0, atol=0.0)), 0)
        self.assertEqual(len(model.identify_duplicate_membership_fcns(rtol=0.0, atol=0.1)), 1)

    def test_to_simple_model_matches_tsk_predict_at_exact_tolerance(self):
        """`to_simple_model` + `simple_gaussian_predict` is a different code path
        (explicit rules over deduplicated ids) but must reproduce `tsk_predict`
        exactly when dedup is exact-tolerance only."""
        model = _model_with_duplicate()
        X = pd.DataFrame({"f": np.linspace(-3.0, 8.0, 25)})

        direct = tsk_predict(X, model)
        via_simple = simple_gaussian_predict(X, model.to_simple_model(rtol=0.0, atol=0.0))

        np.testing.assert_array_equal(direct, np.asarray(via_simple))


class TestTribbleClassifierDedup(unittest.TestCase):
    def _fitted(self):
        X, y = _blobs()
        clf = TribbleClassifier(top_p=1.0, n_gaussians=1, random_state=0)
        clf.fit(X, y)
        return clf, X

    def test_deduplicate_at_exact_tolerance_never_changes_predictions(self):
        clf, X = self._fitted()
        before = clf.predict(X)

        # Force a real duplicate into the fitted model so this test does not
        # depend on one emerging naturally from the fit.
        first_feature = next(iter(clf.model_.feature_models))
        label_model = clf.model_.feature_models[first_feature].label_models[clf.classes_[0]]
        label_model.memberships.append(label_model.memberships[0])
        n_before = clf.model_.n_membership_functions

        removed = clf.deduplicate(rtol=0.0, atol=0.0)

        self.assertEqual(removed, 1)
        self.assertEqual(clf.model_.n_membership_functions, n_before - 1)
        self.assertEqual(clf.n_deduplicated_membership_functions_, 1)
        np.testing.assert_array_equal(before, clf.predict(X))

    def test_to_simple_model_matches_predict_at_exact_tolerance(self):
        clf, X = self._fitted()
        direct = clf.predict(X)
        via_simple = simple_gaussian_predict(X, clf.to_simple_model(rtol=0.0, atol=0.0))
        np.testing.assert_array_equal(np.asarray(direct), np.asarray(via_simple))


if __name__ == "__main__":
    unittest.main()
