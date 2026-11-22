"""Issue #252: the redundancy filter must not misreport or hide what it does.

- `top_n_actual_` is the number of features actually kept (it used to echo `top_n`).
- Dropped features are recorded in `redundant_features_` with their partner.
- `correlation_threshold >= 1.0` disables the filter and records nothing.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.gaussian_regressor import TribbleRegressor


def _shortage_data(seed=0, n=400):
    """Four near-copies of one signal plus pure noise: top_n=4 cannot be filled uncorrelated."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    base = rng.normal(size=n) + 2.0 * y
    X = pd.DataFrame({f"c{i}": base + 0.01 * rng.normal(size=n) for i in range(4)})
    X["noise"] = rng.normal(size=n)
    return X, y


class TestRedundancyRecordClassifier(unittest.TestCase):
    def test_shortage_reports_kept_count_not_requested(self):
        X, y = _shortage_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = TribbleClassifier(top_n=4, scoring_method="higher_order",
                                    correlation_threshold=0.85, random_state=0).fit(X, y)
        self.assertEqual(len(clf.top_features_), 2)
        self.assertEqual(clf.top_n_actual_, len(clf.top_features_))

    def test_dropped_features_are_recorded_with_partner(self):
        X, y = _shortage_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = TribbleClassifier(top_n=4, scoring_method="higher_order",
                                    correlation_threshold=0.85, random_state=0).fit(X, y)
        copies = {"c0", "c1", "c2", "c3"}
        dropped = {r["feature"] for r in clf.redundant_features_}
        self.assertEqual(dropped, copies - set(clf.top_features_))
        self.assertEqual(len(dropped), 3)
        for r in clf.redundant_features_:
            self.assertIn(r["partner"], clf.top_features_)
            self.assertGreaterEqual(abs(r["correlation"]), 0.85)
            self.assertNotIn(r["feature"], clf.top_features_)

    def test_backfill_fills_top_n_on_breast_cancer(self):
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split

        X, y = load_breast_cancer(return_X_y=True)
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = TribbleClassifier(top_n=5, scoring_method="higher_order",
                                    correlation_threshold=0.85, random_state=0).fit(Xtr, ytr)
        self.assertEqual(len(clf.top_features_), 5)
        self.assertEqual(clf.top_n_actual_, 5)
        self.assertGreaterEqual(len(clf.redundant_features_), 1)

    def test_threshold_one_disables_filter(self):
        X, y = _shortage_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = TribbleClassifier(top_n=4, scoring_method="higher_order",
                                    correlation_threshold=1.0, random_state=0).fit(X, y)
        self.assertEqual(len(clf.top_features_), 4)
        self.assertEqual(clf.top_n_actual_, 4)
        self.assertEqual(clf.redundant_features_, [])

    def test_redundant_features_resets_each_fit(self):
        X, y = _shortage_data()
        clf = TribbleClassifier(top_n=4, scoring_method="higher_order",
                                correlation_threshold=0.85, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y)
            first = len(clf.redundant_features_)
            clf.fit(X, y)
        self.assertEqual(len(clf.redundant_features_), first)


class TestRedundancyRecordRegressor(unittest.TestCase):
    def test_shortage_reports_kept_count_and_records_drops(self):
        X, y = _shortage_data()
        y = y + 0.1 * np.random.default_rng(1).normal(size=len(y))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg = TribbleRegressor(top_n=4, scoring_method="higher_order",
                                   correlation_threshold=0.85, random_state=0).fit(X, y)
        self.assertEqual(reg.top_n_actual_, len(reg.top_features_))
        self.assertLess(len(reg.top_features_), 4)
        copies = {"c0", "c1", "c2", "c3"}
        self.assertEqual({r["feature"] for r in reg.redundant_features_}, copies - set(reg.top_features_))


if __name__ == "__main__":
    unittest.main()
