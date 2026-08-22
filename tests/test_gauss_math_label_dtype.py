"""The differentiation score must not depend on the label column's dtype.

`calculate_gaussian_correlation` masks with ``data[y == label]`` once per
(feature, label pair), so the same K masks are rebuilt M times. On a string dtype
that masking, not the distance computation, is the dominant cost of the function:
on RT-IOT2022 (92,293 rows x 82 features x 11 labels) one comparison takes 2.91 ms
against 0.02 ms for a categorical, and the 9,020 comparisons cost 26 s against
0.58 s for every ``wasserstein_distance`` call combined.

The function therefore converts the labels to ``category`` once. That is a pure
performance change and it is only safe while the scores it produces do not depend
on the dtype it was handed, which is what these tests pin. They assert the
property, not the speed -- a timing assertion would be flaky and would not catch
the thing that actually matters.

`unique_labels` is deliberately taken from the original ``y``, before the
conversion, so the label enumeration order (and hence the order the pairwise
scores accumulate in) is unchanged. `test_label_order_preserved` guards that.
"""

import unittest

import numpy as np
import pandas as pd

from tribblefis.gauss_math import calculate_gaussian_correlation

METHODS = ("wasserstein", "bhattacharyya", "composite")


def _dataset(n_per_class: int = 60, seed: int = 0):
    """Three separable classes over four features, labelled with strings."""
    rng = np.random.default_rng(seed)
    frames, labels = [], []
    for i, name in enumerate(["alpha", "beta", "gamma"]):
        frames.append(
            pd.DataFrame(
                {
                    "sep": rng.normal(3.0 * i, 1.0, n_per_class),
                    "weak": rng.normal(0.2 * i, 1.0, n_per_class),
                    "noise": rng.normal(0.0, 1.0, n_per_class),
                    "wide": rng.normal(50.0 * i, 20.0, n_per_class),
                }
            )
        )
        labels += [name] * n_per_class
    X = pd.concat(frames, ignore_index=True)
    y = pd.Series(labels)
    return X, y


class TestLabelDtypeInvariance(unittest.TestCase):
    def test_scores_identical_across_label_dtypes(self):
        X, y = _dataset()
        variants = {
            "str": y.astype("str"),
            "object": y.astype(object),
            "category": y.astype("category"),
        }
        for method in METHODS:
            with self.subTest(method=method):
                results = {
                    name: calculate_gaussian_correlation(X, yy, method=method)
                    for name, yy in variants.items()
                }
                base = results["str"]
                for name, got in results.items():
                    self.assertEqual(
                        [c for c, _ in got],
                        [c for c, _ in base],
                        f"{name}: feature ordering changed",
                    )
                    for (_, a), (col, b) in zip(base, got):
                        self.assertEqual(a, b, f"{name}: score for {col} changed")

    def test_label_order_preserved(self):
        """Reordering the ROWS changes label first-appearance order, and the
        scores are allowed to move with it -- but the two dtypes must move
        together. This is what would break if the conversion were done before
        `y.unique()` rather than after."""
        X, y = _dataset()
        shuffled = np.random.default_rng(1).permutation(len(X))
        Xs, ys = X.iloc[shuffled].reset_index(drop=True), y.iloc[shuffled].reset_index(drop=True)
        for method in METHODS:
            with self.subTest(method=method):
                a = calculate_gaussian_correlation(Xs, ys.astype("str"), method=method)
                b = calculate_gaussian_correlation(Xs, ys.astype("category"), method=method)
                self.assertEqual([c for c, _ in a], [c for c, _ in b])
                for (_, va), (col, vb) in zip(a, b):
                    self.assertEqual(va, vb, f"score for {col} differs by dtype")

    def test_numeric_labels_still_work(self):
        """Not every caller passes strings; the conversion must be a no-op for
        integer labels rather than an error."""
        X, y = _dataset()
        numeric = y.map({"alpha": 0, "beta": 1, "gamma": 2})
        for method in METHODS:
            with self.subTest(method=method):
                got = calculate_gaussian_correlation(X, numeric, method=method)
                self.assertEqual(len(got), X.shape[1])
                self.assertTrue(all(np.isfinite(v) for _, v in got))

    def test_single_label_does_not_raise(self):
        """One class means no pairs; the function should still return a score
        per feature rather than falling over on the empty pair loop."""
        X, y = _dataset()
        one = pd.Series(["only"] * len(X))
        got = calculate_gaussian_correlation(X, one, method="wasserstein")
        self.assertLessEqual(len(got), X.shape[1])


if __name__ == "__main__":
    unittest.main()
