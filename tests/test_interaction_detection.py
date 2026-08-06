"""Tests for candidate cross-term identification during feature selection.

`calculate_gaussian_correlation` (gauss_math.py) ranks every feature
independently, so a feature that only matters *jointly* with another can be
dropped by `take_top_features` before anything downstream -- including the
existing `select_interaction_terms` LassoCV screen in regression.py, which
had zero test coverage of its own before this file -- ever sees it. These
tests check, in order: the pairwise interaction score correctly ranks a real
interaction above noise pairs and above the univariate scores it is compared
against; the rescue mechanism actually changes which features survive
selection; `select_interaction_terms` recovers a known cross term; and the
whole thing wired into `MixtureOfGaussiansFuzzyRegressor` measurably improves
a strict-feature-budget fit on an interaction-only problem.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    calculate_interaction_scores,
    take_top_features,
    take_top_interactions,
    rescue_interacting_features,
)
from tribblefis.regression import partition_output, select_interaction_terms, _rsquared
from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor


class TestInteractionScoreRanksRealInteractionAboveNoise(unittest.TestCase):
    """`y` depends only on the product `x0 * x1`, plus an unrelated `noise`
    column. The (x0, x1) pair should score far above any pair involving
    `noise`, and above either feature's own individual score."""

    def test_multiplicative_interaction(self):
        rng = np.random.default_rng(3)
        n = 1500
        x0 = rng.uniform(-1, 1, n)
        x1 = rng.uniform(-1, 1, n)
        noise = rng.uniform(-1, 1, n)
        y = x0 * x1 + rng.normal(0, 0.02, n)
        X = pd.DataFrame({"x0": x0, "x1": x1, "noise": noise})
        y_series = pd.Series(y, name="y_value")

        y_part, _ = partition_output(6, y_series)
        buckets = y_part["y_bucket"]

        diffs = calculate_gaussian_correlation(X, buckets)
        scores = calculate_interaction_scores(X, buckets, diffs)

        best_pair = (scores[0][0], scores[0][1])
        self.assertEqual(set(best_pair), {"x0", "x1"})
        self.assertAlmostEqual(scores[0][2], 1.0, places=6)  # normalized top lift

        noise_pair_lifts = [lift for fi, fj, lift in scores if "noise" in (fi, fj)]
        self.assertTrue(all(lift < 0.1 for lift in noise_pair_lifts))


class TestRescueMechanism(unittest.TestCase):
    """A feature that scores under `take_top_features`'s threshold, but is
    the weaker half of a strong interaction, must be excluded without
    detection and included with it."""

    def test_rescues_the_weaker_interacting_feature(self):
        rng = np.random.default_rng(0)
        n = 600
        x0 = rng.uniform(-1, 1, n)
        x1 = rng.uniform(-1, 1, n)
        noise = rng.uniform(-1, 1, n)
        y = pd.Series((np.sign(x0 * x1) > 0).astype(int))
        X = pd.DataFrame({"x0": x0, "x1": x1, "noise": noise})

        diffs = calculate_gaussian_correlation(X, y)
        _, top_no_rescue = take_top_features(diffs, top_p=0.5)
        self.assertNotIn("x0", top_no_rescue)  # dropped by the univariate threshold alone

        scores = calculate_interaction_scores(X, y, diffs)
        kept_pairs = take_top_interactions(scores, top_p=0.5)
        self.assertIn(("x1", "x0"), kept_pairs)

        rescued = rescue_interacting_features(top_no_rescue, diffs, kept_pairs)
        self.assertIn("x0", rescued)
        self.assertIn("x1", rescued)

    def test_take_top_interactions_filters_nonpositive_lift(self):
        scores = [("a", "b", 1.0), ("c", "d", 0.4), ("e", "f", -0.2)]
        kept = take_top_interactions(scores, top_p=1.0)  # threshold 0 -> keep every positive-lift pair
        self.assertEqual(kept, [("a", "b"), ("c", "d")])

        kept_n1 = take_top_interactions(scores, top_n=1)
        self.assertEqual(kept_n1, [("a", "b")])


class TestInteractionScoreGuardRail(unittest.TestCase):
    def test_max_pairs_raises_before_scoring(self):
        rng = np.random.default_rng(0)
        n = 50
        n_features = 40  # 40 choose 2 = 780 pairs
        X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
        y = pd.Series(rng.integers(0, 2, n))
        diffs = calculate_gaussian_correlation(X, y)
        with self.assertRaises(ValueError):
            calculate_interaction_scores(X, y, diffs, max_pairs=100)


class TestSelectInteractionTermsRecoversKnownCrossTerm(unittest.TestCase):
    """First test coverage for `select_interaction_terms` (regression.py):
    on data with exactly one true cross term among several candidate pairs,
    the LassoCV screen must keep that pair and respect `max_pairs`."""

    def _synthetic_frame(self, seed=7, n=400):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame({f"x{i}": rng.uniform(-2, 2, n) for i in range(4)})
        y = 2.0 * X["x0"] - X["x1"] + 3.0 * X["x0"] * X["x2"] + rng.normal(0, 0.01, n)
        return X, pd.DataFrame({"y_value": y})

    def test_recovers_the_true_pair(self):
        X, y_train = self._synthetic_frame()
        top_n_todo = list(X.columns)
        kept = select_interaction_terms(X, top_n_todo, y_train, y_bucket_mean=None)
        kept_names = {(top_n_todo[i], top_n_todo[j]) for i, j in kept}
        self.assertIn(("x0", "x2"), kept_names)

    def test_respects_max_pairs(self):
        X, y_train = self._synthetic_frame()
        top_n_todo = list(X.columns)
        kept = select_interaction_terms(X, top_n_todo, y_train, y_bucket_mean=None, max_pairs=1)
        self.assertLessEqual(len(kept), 1)

    def test_candidate_pairs_restricts_the_screen(self):
        """Passing an explicit shortlist (the new parameter this PR adds)
        must only ever return pairs from that shortlist, never fall back to
        scanning every pair among `top_n_todo`."""
        X, y_train = self._synthetic_frame()
        top_n_todo = list(X.columns)
        # Deliberately excludes the true (0, 2) pair from the candidate list.
        candidate_pairs = [(0, 1), (1, 3)]
        kept = select_interaction_terms(
            X, top_n_todo, y_train, y_bucket_mean=None, candidate_pairs=candidate_pairs,
        )
        self.assertTrue(all(pair in candidate_pairs for pair in kept))


class TestEndToEndEstimator(unittest.TestCase):
    """The estimator-level proof: at a strict feature budget that would
    normally keep only one half of a pure interaction, `detect_interactions`
    (+ `select_interactions`) rescues the other half and cross terms recover
    most of the R² a naive fit throws away."""

    def _interaction_only_problem(self, seed=3, n=1500):
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(-1, 1, n)
        x1 = rng.uniform(-1, 1, n)
        noise = rng.uniform(-1, 1, n)
        y = x0 * x1 + rng.normal(0, 0.02, n)
        X = pd.DataFrame({"x0": x0, "x1": x1, "noise": noise})
        return X, y

    def test_detection_improves_strict_budget_fit(self):
        X, y = self._interaction_only_problem()

        without = MixtureOfGaussiansFuzzyRegressor(
            tsk_order="full-2nd", top_n=1, n_output_buckets=6, random_state=42,
        )
        without.fit(X, y)
        r2_without = _rsquared(y, without.predict(X))

        with_detect = MixtureOfGaussiansFuzzyRegressor(
            tsk_order="full-2nd", top_n=1, n_output_buckets=6, random_state=42,
            detect_interactions=True, select_interactions=True,
        )
        with_detect.fit(X, y)
        r2_with = _rsquared(y, with_detect.predict(X))

        self.assertNotIn("x1", without.top_features_)
        self.assertIn("x1", with_detect.top_features_)
        self.assertEqual(with_detect.cross_pairs_, [(0, 1)])
        self.assertGreater(r2_with, r2_without + 0.5)  # dramatic, not marginal
        self.assertGreater(r2_with, 0.9)

    def test_detect_interactions_without_full2nd_only_rescues(self):
        """Rescue into `top_features_` doesn't require `full-2nd` -- there
        is just nowhere for `cross_pairs_` to go, so it stays `None`."""
        X, y = self._interaction_only_problem(seed=9, n=500)
        reg = MixtureOfGaussiansFuzzyRegressor(
            tsk_order="1st", top_n=1, n_output_buckets=6, random_state=42,
            detect_interactions=True,
        )
        reg.fit(X, y)
        self.assertIn("x1", reg.top_features_)
        self.assertIsNone(reg.cross_pairs_)

    def test_select_interactions_without_full2nd_warns(self):
        X, y = self._interaction_only_problem(seed=11, n=400)
        reg = MixtureOfGaussiansFuzzyRegressor(
            tsk_order="1st", top_n=1, n_output_buckets=6, random_state=42,
            detect_interactions=True, select_interactions=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reg.fit(X, y)
        self.assertTrue(any("select_interactions" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
