"""Tests for fuzzytree.auto_topology and the `topology="auto"` path (#226).

`DeconstructedHierarchicalRegressor` required a fully-specified topology dict
and had no fallback, so a dataset with no domain structure could not use it
without someone inventing a grouping by hand. These cover the three strategies
#226 asks for, the selection between them, and -- the part most likely to rot --
that a derived topology is subject to exactly the same validation a
hand-authored one is, rather than getting a private path around it.
"""

import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from fuzzytree import (
    DeconstructedHierarchicalRegressor,
    affinity_topology,
    candidate_topologies,
    cluster_features,
    feature_affinity,
    parse_topology,
    per_feature_topology,
    select_topology,
)


def _two_factor_frame(n=200, seed=0, noise=0.05):
    """Four features generated from two latent factors, two features each.

    The right answer at k=2 is known by construction -- {a, b} and {c, d} --
    which is what makes this a test of the clustering rather than a
    demonstration of it. Anything that groups a with c has not worked.
    """
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a": f1 + noise * rng.normal(size=n),
            "b": f1 + noise * rng.normal(size=n),
            "c": f2 + noise * rng.normal(size=n),
            "d": f2 + noise * rng.normal(size=n),
        }
    )
    y = pd.Series(2.0 * f1 - f2 + 0.1 * rng.normal(size=n))
    return X, y


def _groups(topology):
    """The feature partition a topology dict describes, ignoring node names."""
    root = next(iter(topology))
    return {frozenset(topology[child]) for child in topology[root]}


class TestFeatureAffinity(unittest.TestCase):
    def test_affinity_recovers_the_latent_block_structure(self):
        X, _ = _two_factor_frame()
        affinity = feature_affinity(X)
        # Within-factor pairs must beat every across-factor pair, by a lot --
        # this is the signal the clustering has to see.
        self.assertGreater(min(affinity.loc["a", "b"], affinity.loc["c", "d"]), 0.9)
        self.assertLess(max(affinity.loc["a", "c"], affinity.loc["b", "d"]), 0.3)

    def test_diagonal_is_one_and_matrix_is_symmetric(self):
        """Both are preconditions for the precomputed-distance clustering.

        A non-unit diagonal gives a non-zero self-distance, and asymmetry makes
        `metric="precomputed"` invalid outright. `.abs()` on a correlation frame
        can also hand back a read-only array, which is how the first version of
        this failed -- in place assignment on a view.
        """
        X, _ = _two_factor_frame(n=50)
        affinity = feature_affinity(X).to_numpy()
        np.testing.assert_allclose(np.diag(affinity), 1.0)
        np.testing.assert_allclose(affinity, affinity.T, atol=1e-12)

    def test_constant_feature_has_no_affinity_to_anything(self):
        """A column with no variance carries no evidence about what it belongs
        with. `corr` gives NaN; guessing would be worse than isolating it."""
        X, _ = _two_factor_frame(n=50)
        X["flat"] = 3.0
        affinity = feature_affinity(X)
        self.assertEqual(affinity.loc["flat", "a"], 0.0)
        self.assertEqual(affinity.loc["flat", "flat"], 1.0)
        self.assertFalse(affinity.isna().any().any())


class TestClusterFeatures(unittest.TestCase):
    def test_two_groups_recover_the_two_factors(self):
        X, _ = _two_factor_frame()
        groups = cluster_features(X, 2)
        self.assertEqual(
            {frozenset(g) for g in groups}, {frozenset("ab"), frozenset("cd")}
        )

    def test_degenerate_cut_counts(self):
        X, _ = _two_factor_frame(n=50)
        self.assertEqual(cluster_features(X, 1), [list(X.columns)])
        self.assertEqual(cluster_features(X, 4), [[c] for c in X.columns])

    def test_group_order_is_deterministic_and_follows_column_order(self):
        """Node names land in `node_state_` and in any plot of the tree, so a
        topology that renames its nodes between runs on the same data is
        unreadable."""
        X, _ = _two_factor_frame(n=80)
        self.assertEqual(cluster_features(X, 2), cluster_features(X, 2))
        self.assertEqual(cluster_features(X, 2)[0][0], "a")

    def test_out_of_range_cut_counts_are_rejected(self):
        X, _ = _two_factor_frame(n=30)
        with self.assertRaises(ValueError):
            cluster_features(X, 0)
        with self.assertRaises(ValueError):
            cluster_features(X, 5)


class TestGeneratedTopologiesAreValid(unittest.TestCase):
    """Every derived topology must survive `parse_topology` untouched.

    #226 is explicit that an auto-derived topology "should still go through
    `parse_topology`'s existing validation [...] it shouldn't need a second
    validation path." These assert the generator satisfies the validator rather
    than the validator being relaxed to admit the generator.
    """

    def test_affinity_topologies_parse(self):
        X, _ = _two_factor_frame(n=60)
        for k in (1, 2, 3, 4):
            root = parse_topology(affinity_topology(X, k), list(X.columns))
            leaves = list(root.iter_leaves())
            self.assertEqual(len(leaves), k)
            owned = [f for leaf in leaves for f in leaf.own_features]
            self.assertCountEqual(owned, list(X.columns))

    def test_per_feature_topology_parses_and_owns_every_column(self):
        X, _ = _two_factor_frame(n=60)
        root = parse_topology(per_feature_topology(X), list(X.columns))
        leaves = list(root.iter_leaves())
        self.assertEqual(len(leaves), X.shape[1])
        self.assertTrue(all(len(leaf.own_features) == 1 for leaf in leaves))

    def test_generated_names_dodge_colliding_feature_columns(self):
        """`parse_topology` rejects a node name that is also a column, rightly.

        A dataset with a column literally called "root" or "group_0" is not
        exotic enough to be worth failing on, so generated names take an
        underscore until they are free.
        """
        X = pd.DataFrame(
            {"root": [1.0, 2, 3, 4], "group_0": [2.0, 1, 4, 3], "z": [0.0, 1, 0, 1]}
        )
        topology = affinity_topology(X, 2)
        parse_topology(topology, list(X.columns))  # must not raise
        self.assertFalse(set(topology) & set(X.columns))


class TestCandidateSet(unittest.TestCase):
    def test_cut_counts_above_the_feature_count_are_dropped(self):
        """A default sweep of (2, 3, 4) must not explode on a 2-column frame."""
        X = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [4.0, 3, 2, 1]})
        names = set(candidate_topologies(X))
        self.assertIn("affinity_k2", names)
        self.assertNotIn("affinity_k3", names)
        self.assertNotIn("affinity_k4", names)

    def test_duplicate_groupings_collapse(self):
        """At k == n_features the affinity cut *is* `per_feature`.

        Scoring both spends a full k-fold sweep learning that a topology ties
        with itself, and then reports the winner under whichever name sorted
        first -- so "affinity_k4 won" would be describing the no-knowledge
        floor under an assumed name.
        """
        X, _ = _two_factor_frame(n=40)
        candidates = candidate_topologies(X, n_groups=(2, 4))
        partitions = [_groups(t) for t in candidates.values()]
        self.assertEqual(len(partitions), len({frozenset(p) for p in partitions}))
        self.assertNotIn("per_feature", candidates)  # affinity_k4 got there first

    def test_floor_is_always_present_when_not_duplicated(self):
        X, _ = _two_factor_frame(n=40)
        self.assertIn("per_feature", candidate_topologies(X, n_groups=(2,)))


class TestSelectTopology(unittest.TestCase):
    def test_selection_returns_a_candidate_and_all_scores(self):
        """The spread matters as much as the winner.

        A caller who sees only the winning name cannot tell "structure found"
        from "arbitrary pick among ties", so `select_topology` returns every
        score and this pins that it does.
        """
        X, y = _two_factor_frame(n=120)
        # A cheap stand-in for the real fit: score each candidate by how well a
        # per-group mean predicts. Keeps the test about the *selection* rather
        # than about the hierarchical regressor, and keeps it fast.
        def fit_score(X_tr, y_tr, X_va, y_va, topology):
            root = next(iter(topology))
            groups = [topology[child] for child in topology[root]]
            preds = np.column_stack([X_va[g].mean(axis=1) for g in groups])
            coef, *_ = np.linalg.lstsq(
                np.column_stack(
                    [np.column_stack([X_tr[g].mean(axis=1) for g in groups]),
                     np.ones(len(X_tr))]
                ),
                y_tr.to_numpy(),
                rcond=None,
            )
            fitted = preds @ coef[:-1] + coef[-1]
            return -float(np.mean((y_va.to_numpy() - fitted) ** 2))

        name, topology, scores = select_topology(X, y, fit_score, n_splits=3)
        self.assertIn(name, scores)
        self.assertEqual(set(scores), set(candidate_topologies(X)))
        self.assertEqual(scores[name], max(scores.values()))
        parse_topology(topology, list(X.columns))

    def test_selection_is_deterministic_for_a_fixed_random_state(self):
        X, y = _two_factor_frame(n=90)
        calls = []

        def fit_score(X_tr, y_tr, X_va, y_va, topology):
            calls.append(tuple(sorted(X_va.index)))
            return float(len(_groups(topology)))

        first, _, _ = select_topology(X, y, fit_score, random_state=0)
        folds_first = list(calls)
        calls.clear()
        second, _, _ = select_topology(X, y, fit_score, random_state=0)
        self.assertEqual(first, second)
        self.assertEqual(folds_first, calls)


class TestAutoTopologyOnTheEstimator(unittest.TestCase):
    def _fit(self, X, y, **kwargs):
        model = DeconstructedHierarchicalRegressor(
            flat_regressor_kwargs={"n_gaussians": 2}, auto_n_groups=(2, 3)
        )
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X, y, **kwargs)
        return model

    def test_auto_fits_predicts_and_records_its_provenance(self):
        """`topology_source_` exists so a downstream table can never report a
        derived grouping as a domain one."""
        X, y = _two_factor_frame(n=120)
        model = self._fit(X, y, topology="auto")

        self.assertEqual(model.topology_source_, "auto")
        self.assertIn(model.topology_name_, candidate_topologies(X, (2, 3)))
        self.assertEqual(set(model.topology_scores_), set(candidate_topologies(X, (2, 3))))
        predictions = model.predict(X)
        self.assertEqual(predictions.shape, (len(X),))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_omitting_the_topology_means_auto(self):
        X, y = _two_factor_frame(n=90)
        self.assertEqual(self._fit(X, y).topology_source_, "auto")

    def test_a_supplied_topology_is_used_verbatim_and_marked_as_such(self):
        """A caller who supplied a topology gets exactly that topology.

        The auto path must not second-guess a domain grouping -- that would
        invert the premise the whole deconstruction approach rests on.
        """
        X, y = _two_factor_frame(n=90)
        supplied = {"R": ["L1", "L2"], "L1": ["a", "c"], "L2": ["b", "d"]}
        model = self._fit(X, y, topology=supplied)

        self.assertEqual(model.topology_source_, "supplied")
        self.assertEqual(model.topology_, supplied)
        self.assertIsNone(model.topology_scores_)
        # ...including a grouping that cuts straight across the latent factors,
        # which no affinity strategy would ever propose.
        self.assertEqual(_groups(model.topology_), {frozenset("ac"), frozenset("bd")})

    def test_a_bad_topology_argument_is_rejected(self):
        X, y = _two_factor_frame(n=40)
        model = DeconstructedHierarchicalRegressor()
        with self.assertRaises(ValueError):
            with contextlib.redirect_stdout(io.StringIO()):
                model.fit(X, y, topology="automatic")

    def test_single_feature_frame_skips_selection(self):
        """One feature cannot be grouped, and the k-fold sweep would spend
        three fits discovering that."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=80)})
        y = pd.Series(2.0 * X["a"] + 0.1 * rng.normal(size=80))
        model = self._fit(X, y, topology="auto")
        self.assertEqual(model.topology_name_, "per_feature")
        self.assertIsNone(model.topology_scores_)

    def test_candidate_scoring_does_not_leak_into_the_final_fit(self):
        """Selection fits clones, not `self`.

        The first version scored candidates on `self`, and the flat model from
        the last candidate survived into the real fit -- so the returned
        estimator was fitted on a training fold rather than on all of X, and
        nothing about its predictions said so.
        """
        X, y = _two_factor_frame(n=120)
        model = self._fit(X, y, topology="auto")

        # The flat model has to have been fitted on every row, not on a fold.
        self.assertEqual(len(model.flat_.model_.feature_models), X.shape[1])
        self.assertEqual(len(model.flat_.predict(X)), len(X))

        # And every node of the *chosen* topology -- not some earlier
        # candidate's -- has state, which is what would be missing if a clone's
        # fit had been mistaken for the real one.
        parsed = parse_topology(model.topology_, list(X.columns))
        self.assertEqual({node.name for node in parsed.iter_nodes()}, set(model.node_state_))


if __name__ == "__main__":
    unittest.main()
