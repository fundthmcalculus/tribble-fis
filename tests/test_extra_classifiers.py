"""Coverage for the previously-untested classifier modules:
``ensemble_fuzzy_classifier.py`` (BaggedFuzzyClassifier),
``calibrated_fuzzy_classifier.py`` (CalibratedGaussianFuzzyClassifier), and
``bsp_fuzzy_classifier.py`` (BSPFuzzyTreeClassifier).
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB

from tribblefis.calibrated_fuzzy_classifier import CalibratedGaussianFuzzyClassifier
from tribblefis.ensemble_fuzzy_classifier import BaggedFuzzyClassifier
from tribblefis.bsp_fuzzy_classifier import BSPFuzzyTreeClassifier


def _blobs(n=180, seed=0, n_classes=3, sep=3.0):
    """Well-separated Gaussian blobs, one per class, on 2 features.

    Both features are informative (class centers vary in both dims), so a
    random-subspace model that only sees one feature -- as BaggedFuzzyClassifier's
    default `max_features="sqrt"` does with 2 features -- still has real signal
    regardless of which feature it's given.
    """
    rng = np.random.default_rng(seed)
    per = n // n_classes
    xs, ys = [], []
    for c in range(n_classes):
        center = [c * sep, c * sep]
        xs.append(rng.normal(center, 1.0, size=(per, 2)))
        ys.append(np.full(per, c))
    X = pd.DataFrame(np.vstack(xs), columns=["a", "b"])
    y = np.concatenate(ys)
    perm = rng.permutation(len(X))
    return X.iloc[perm].reset_index(drop=True), y[perm]


class TestCalibratedGaussianFuzzyClassifier(unittest.TestCase):

    def test_fit_predict_accuracy_on_separated_blobs(self):
        X, y = _blobs()
        clf = CalibratedGaussianFuzzyClassifier(n_gaussians=1, top_p=1.0)
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = float(np.mean(preds == y))
        self.assertGreater(acc, 0.9)

    def test_predict_proba_is_a_valid_distribution(self):
        X, y = _blobs()
        clf = CalibratedGaussianFuzzyClassifier(n_gaussians=1, top_p=1.0).fit(X, y)
        proba = clf.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), len(clf.classes_)))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)
        self.assertTrue(np.all(proba >= 0.0))

    def test_matches_gaussian_naive_bayes_with_one_component(self):
        """With one Gaussian per feature and product t-norm, the docstring
        claims this is provably equivalent to GaussianNB -- check predictions
        agree on well-separated data where both should be near-perfect."""
        X, y = _blobs(sep=6.0)
        clf = CalibratedGaussianFuzzyClassifier(n_gaussians=1, top_p=1.0).fit(X, y)
        gnb = GaussianNB().fit(X, y)
        agreement = float(np.mean(clf.predict(X) == gnb.predict(X)))
        self.assertGreater(agreement, 0.95)

    def test_get_params_and_clone(self):
        clf = CalibratedGaussianFuzzyClassifier(n_gaussians=2, use_priors=False)
        cloned = clone(clf)
        self.assertEqual(cloned.n_gaussians, 2)
        self.assertFalse(cloned.use_priors)


class TestBaggedFuzzyClassifier(unittest.TestCase):

    def test_fit_predict_soft_voting(self):
        X, y = _blobs()
        clf = BaggedFuzzyClassifier(n_estimators=5, voting="soft", random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(len(preds), len(X))
        acc = float(np.mean(preds == y))
        self.assertGreater(acc, 0.8)

    def test_fit_predict_hard_voting(self):
        X, y = _blobs()
        clf = BaggedFuzzyClassifier(n_estimators=5, voting="hard", random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(len(preds), len(X))
        acc = float(np.mean(preds == y))
        self.assertGreater(acc, 0.8)

    def test_predict_proba_is_a_valid_distribution(self):
        X, y = _blobs()
        clf = BaggedFuzzyClassifier(n_estimators=5, random_state=0).fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)

    def test_feature_usage_tracks_bootstrap_draws(self):
        X, y = _blobs()
        clf = BaggedFuzzyClassifier(n_estimators=10, max_features=1, random_state=0).fit(X, y)
        self.assertEqual(set(clf.feature_usage_), {"a", "b"})
        self.assertEqual(sum(clf.feature_usage_.values()), 10)  # 1 feature x 10 estimators

    def test_max_features_variants(self):
        X, y = _blobs()
        for mf in ("sqrt", "all", 1, 0.5):
            clf = BaggedFuzzyClassifier(n_estimators=3, max_features=mf, random_state=0)
            clf.fit(X, y)
            self.assertEqual(len(clf.predict(X)), len(X))


class TestBSPFuzzyTreeClassifier(unittest.TestCase):

    def test_stays_a_single_leaf_when_accuracy_threshold_is_trivial(self):
        X, y = _blobs()
        clf = BSPFuzzyTreeClassifier(accuracy_threshold=0.0, n_gaussians=1, top_p=1.0)
        clf.fit(X, y)
        self.assertEqual(clf.n_leaves_, 1)
        self.assertEqual(clf.max_depth_reached_, 0)

    def test_splits_when_a_single_leaf_is_not_accurate_enough(self):
        X, y = _blobs(sep=1.2, n=400)  # overlapping classes -> single leaf underfits
        clf = BSPFuzzyTreeClassifier(
            accuracy_threshold=0.99, max_depth=4, min_samples_split=20,
            n_gaussians=1, top_p=1.0,
        )
        clf.fit(X, y)
        self.assertGreaterEqual(clf.n_leaves_, 1)
        self.assertLessEqual(clf.max_depth_reached_, 4)

    def test_predict_covers_every_row(self):
        X, y = _blobs()
        clf = BSPFuzzyTreeClassifier(accuracy_threshold=0.5, n_gaussians=1, top_p=1.0).fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(len(preds), len(X))
        self.assertTrue(all(p in clf.classes_ for p in preds))

    def test_describe_returns_readable_structure(self):
        X, y = _blobs()
        clf = BSPFuzzyTreeClassifier(accuracy_threshold=0.5, n_gaussians=1, top_p=1.0).fit(X, y)
        text = clf.describe()
        self.assertIsInstance(text, str)
        self.assertIn("leaf(", text)


if __name__ == "__main__":
    unittest.main()
