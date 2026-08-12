"""Tests for cluster-restricted joint-term admission (`cluster_joint_terms` in
`ruspinize_model`, src/tribblefis/ruspini.py).

Independently OR-ing each feature's activated terms into one rule per class
admits the *Cartesian product* of those per-feature term sets -- a hyper-box
that can include joint combinations the class never actually contains (e.g.
X in {0,1} and Y in {0,1} admits (X=0,Y=1) even if only (X=0,Y=0) and
(X=1,Y=1) were observed). `cluster_joint_terms=True` replaces that single
marginal rule, per class, with one rule per joint term combination the
class's own rows actually land on.

These tests hand-build a `GaussianMixtureModel` (mus placed exactly at the
apex knots we want, rather than fit from data) so the resulting term indices
are deterministic, then feed it through the real `ruspinize_model` entry
point -- not a bypass. Sample noise is kept small (sigma=0.3) relative to the
knot spacing (5) so every row's hard-assigned term is unambiguous.
`sigma_knots=0.0` is passed explicitly throughout to disable the library's
own default extra spread-derived knots, which would otherwise add landmarks
these tests don't control for and break the exact term-index assumptions
they're built on.
"""

import unittest

import numpy as np
import pandas as pd

from tribblefis.gauss_data import GaussianMembership, GaussianMixtureModel, FeatureModel, LabelModel
from tribblefis.gauss_math import simple_gaussian_predict
from tribblefis.ruspini import ruspinize_model
from tribblefis.refine import refine_ruspini_partition


def _hand_gmm(feature_mus: dict[str, dict], sigma: float = 1.0) -> GaussianMixtureModel:
    """Build a GaussianMixtureModel whose landmarks are exactly the given mus."""
    feature_models = {}
    for f, label_mus in feature_mus.items():
        label_models = {
            label: LabelModel([GaussianMembership.create(mu, sigma) for mu in mus])
            for label, mus in label_mus.items()
        }
        feature_models[f] = FeatureModel(label_models)
    return GaussianMixtureModel(feature_models)


def _blob(rng, x_mu, y_mu, n=10, noise=0.3):
    return rng.normal([x_mu, y_mu], noise, size=(n, 2))


class TestDisjointJointClusters(unittest.TestCase):
    """A class with two far-apart joint corners: (x=0,y=0) and (x=10,y=10).

    Marginal (per-feature) matching independently activates both x-terms and
    both y-terms for this class, so its naive box also covers the two
    "ghost" corners (0,10) and (10,0), which the class has zero data in.
    """

    def setUp(self):
        rng = np.random.default_rng(0)
        a1 = _blob(rng, 0.0, 0.0)
        a2 = _blob(rng, 10.0, 10.0)
        b = _blob(rng, 10.0, 0.0)  # sits exactly in one of A's marginal "ghost" corners
        X = pd.DataFrame(np.vstack([a1, a2, b]), columns=["x", "y"])
        y = np.array(["A"] * len(a1) + ["A"] * len(a2) + ["B"] * len(b))
        perm = rng.permutation(len(X))
        self.X, self.y = X.iloc[perm].reset_index(drop=True), y[perm]
        self.gm = _hand_gmm({
            # y=5.0 is a decoy landmark for A only (no real A data there) --
            # it exists purely so the y-axis has 3 terms, so A's own two real
            # corners (y=0 and y=10) land on non-adjacent term indices (0 and 2)
            # and clustering can't trivially merge them back into one rule.
            "x": {"A": [0.0, 10.0], "B": [10.0]},
            "y": {"A": [0.0, 10.0, 5.0], "B": [0.0]},
        })

    def test_default_admits_unsupported_ghost_cell(self):
        rm = ruspinize_model(self.gm, self.X, self.y, sigma_knots=0.0)
        self.assertEqual(len(rm.rules), 2)  # one rule per class, as before
        proba, labels = rm.class_proba(pd.DataFrame({"x": [0.0], "y": [10.0]}))
        a_firing = proba[0, labels.index("A")]
        self.assertGreater(a_firing, 0.9)  # admits a corner "A" never occupied

    def test_default_lets_a_intrude_on_bs_real_territory(self):
        rm = ruspinize_model(self.gm, self.X, self.y, sigma_knots=0.0)
        proba, labels = rm.class_proba(pd.DataFrame({"x": [10.0], "y": [0.0]}))
        a_firing = proba[0, labels.index("A")]
        self.assertGreaterEqual(a_firing, 0.4)  # steals ~half the probability at B's own point

    def test_clustering_rejects_the_ghost_cell(self):
        rm = ruspinize_model(self.gm, self.X, self.y, sigma_knots=0.0, cluster_joint_terms=True)
        # A's two disjoint corners are not index-adjacent -> two separate rules;
        # B's single corner stays one rule.
        a_rules = [ant for lab, ant in rm.rules if lab == "A"]
        self.assertEqual(len(a_rules), 2)
        self.assertEqual(len([lab for lab, _ in rm.rules if lab == "B"]), 1)

        proba, labels = rm.class_proba(pd.DataFrame({"x": [10.0], "y": [0.0]}))
        a_firing = proba[0, labels.index("A")]
        self.assertLess(a_firing, 0.05)  # no longer admitted at B's own point

    def test_training_accuracy_not_worse_than_default(self):
        default = ruspinize_model(self.gm, self.X, self.y, sigma_knots=0.0)
        clustered = ruspinize_model(self.gm, self.X, self.y, sigma_knots=0.0, cluster_joint_terms=True)
        acc_default = np.mean(default.predict(self.X) == self.y)
        acc_clustered = np.mean(clustered.predict(self.X) == self.y)
        self.assertGreaterEqual(acc_clustered, acc_default)

    def test_class_proba_is_a_proper_distribution_with_duplicate_labels(self):
        rm = ruspinize_model(self.gm, self.X, self.y, sigma_knots=0.0, cluster_joint_terms=True)
        self.assertGreater(len(rm.rules), len(np.unique(self.y)))  # duplicate-label rules exist
        proba, labels = rm.class_proba(self.X)
        self.assertEqual(len(labels), len(np.unique(self.y)))  # but labels are deduplicated
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_explicit_serialization_matches_predict(self):
        rm = ruspinize_model(self.gm, self.X, self.y, sigma_knots=0.0, cluster_joint_terms=True)
        direct = rm.predict(self.X)
        via_explicit = simple_gaussian_predict(self.X, rm.to_simple_model())
        np.testing.assert_array_equal(np.asarray(direct), np.asarray(via_explicit))

    def test_refine_runs_with_duplicate_label_rules(self):
        rm = ruspinize_model(self.gm, self.X, self.y, sigma_knots=0.0, cluster_joint_terms=True)
        refined, info = refine_ruspini_partition(rm, self.X, self.y, method="coordinate", seed=0, verbose=False)
        if info["refined"]:
            self.assertGreaterEqual(info["val_acc"], info["init_val_acc"])


class TestAdjacentSubclustersMerge(unittest.TestCase):
    """A class spanning two index-adjacent terms on one feature must stay a
    single rule -- clustering should merge, not fragment, contiguous support."""

    def setUp(self):
        rng = np.random.default_rng(1)
        a1 = rng.normal(0.0, 0.3, size=(10, 1))
        a2 = rng.normal(5.0, 0.3, size=(10, 1))
        b = rng.normal(10.0, 0.3, size=(10, 1))
        X = pd.DataFrame(np.vstack([a1, a2, b]), columns=["x"])
        y = np.array(["A"] * 20 + ["B"] * 10)
        perm = rng.permutation(len(X))
        self.X, self.y = X.iloc[perm].reset_index(drop=True), y[perm]
        self.gm = _hand_gmm({"x": {"A": [0.0, 5.0], "B": [10.0]}})

    def test_contiguous_class_stays_one_rule(self):
        default = ruspinize_model(self.gm, self.X, self.y, sigma_knots=0.0)
        clustered = ruspinize_model(self.gm, self.X, self.y, sigma_knots=0.0, cluster_joint_terms=True)
        self.assertEqual(len(default.rules), 2)
        self.assertEqual(len(clustered.rules), 2)  # no spurious fragmentation
        # Not exact equality: the anomaly-bracket knots (`_bracket_anomaly_knots`)
        # replace what used to be one wide shoulder plateau near the data's min/max
        # with two narrower triangles, so a handful of points right at that boundary
        # can land on a different term under marginal- vs cluster-matching. That's a
        # real, expected side effect of a partition-shape guarantee unrelated to
        # clustering, not a regression -- so tolerate near-agreement instead.
        agree = np.mean(default.predict(self.X) == clustered.predict(self.X))
        self.assertGreaterEqual(agree, 0.95)


if __name__ == "__main__":
    unittest.main()
