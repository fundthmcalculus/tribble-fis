"""Tests for classifier antecedent refinement (src/tribblefis/refine.py).

A zeroth-order TSK classifier has no consequents, so its Gaussian ``(mu, sigma)``
antecedents *are* the whole model. These tests cover the two refinement backends
(``coordinate`` and the `optimizers`-package ``optimizers`` search), the
acceptance guards, and the ``refine=True`` wiring on the public classifier API.

A note on what is safe to assert about the ``optimizers`` backend. Two things
make the obvious assertions wrong there:

* **The default guard is ``"none"``.** Since #66 a refinement is accepted
  unconditionally, so ``info["refined"]`` says nothing about whether held-out
  accuracy improved. Assertions of that shape belong on ``guard="legacy"``.
* **The search path is not something to assert on.** Reproducibility here is a
  property of the pinned ``optimizers`` revision rather than of this repository.
  At the revision this test was written against, ``set_seed(seed)`` did not
  reach the initial population at all -- it came from ``np.random.default_rng()``
  with no argument, i.e. fresh OS entropy -- so a seeded refinement returned a
  different model every call, and a pure deterministic fitness produced a
  different evaluation count every run. Upstream ``3a57f91`` fixes that for the
  ``n_jobs=1`` path this code uses, and above ``n_jobs=1`` the workers still
  share a ``numpy.random.Generator`` and race on it.

  So the reproducibility of these calls can change under the suite without any
  change here. Tests in this file assert invariants that hold whatever path the
  search takes, never a specific result, which is correct either way.
"""

import io
import contextlib
import unittest

import numpy as np
import pandas as pd

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
from tribblefis.gauss_math import create_gaussian_membership_dict
from tribblefis.refine import (
    refine_classifier_antecedents,
    extract_gaussian_params,
    _classifier_accuracy,
)


def _blobs(seed=0, n=120):
    """Two 2-D Gaussian classes whose per-class marginal fit is deliberately a
    poor discriminator (overlapping, unequal spreads), so discriminative
    refinement has room to help."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    x0 = rng.normal([0.0, 0.0], [1.0, 2.5], size=(n0, 2))
    x1 = rng.normal([2.0, 0.0], [1.0, 2.5], size=(n - n0, 2))
    X = pd.DataFrame(np.vstack([x0, x1]), columns=["a", "b"])
    y = np.array([0] * n0 + [1] * (n - n0))
    perm = rng.permutation(len(X))
    return X.iloc[perm].reset_index(drop=True), y[perm]


def _quiet_fit(**kwargs):
    X, y = _blobs()
    with contextlib.redirect_stdout(io.StringIO()):
        clf = MixtureOfGaussiansFuzzyClassifier(top_p=1.0, n_gaussians=1, **kwargs)
        clf.fit(X, y)
    return clf, X, y


class TestClassifierRefinement(unittest.TestCase):

    def _heuristic_model(self):
        X, y = _blobs()
        model = create_gaussian_membership_dict(
            X, pd.Series(y), top_n_var_names=["a", "b"], n_gaussians=1
        )
        return X, y, model

    def test_coordinate_accept_implies_val_improvement(self):
        X, y, model = self._heuristic_model()
        _, info = refine_classifier_antecedents(
            model, X, y, method="coordinate", l2_shrink=0.05, seed=0, verbose=False
        )
        # If the refinement was accepted it must have improved held-out accuracy
        # (or tied it with a strictly lower cross-entropy).
        if info["refined"]:
            self.assertGreaterEqual(info["val_acc"], info["init_val_acc"])
        self.assertIn("init_train_obj", info)

    def test_structure_preserved(self):
        X, y, model = self._heuristic_model()
        refined, _ = refine_classifier_antecedents(
            model, X, y, method="coordinate", seed=0, verbose=False
        )
        # Refinement only moves (mu, sigma); it must not add/remove memberships.
        self.assertEqual(
            refined.n_membership_functions, model.n_membership_functions
        )
        self.assertEqual(
            len(extract_gaussian_params(refined)), len(extract_gaussian_params(model))
        )

    def test_rejected_returns_heuristic_unchanged(self):
        # A huge shrinkage weight pins the candidate to the heuristic, so no move
        # can beat it on validation -> the original model object is returned.
        X, y, model = self._heuristic_model()
        out, info = refine_classifier_antecedents(
            model, X, y, method="coordinate", l2_shrink=1e6, seed=0, verbose=False
        )
        if not info["refined"]:
            self.assertIs(out, model)

    def _optimizers_refine(self, **kwargs):
        X, y, model = self._heuristic_model()
        params = dict(
            method="optimizers", optimizer_method="ga", local_grad_optim="perturb",
            population_size=16, num_generations=5, local_scale=0.25,
            l2_shrink=0.05, seed=0, verbose=False,
        )
        params.update(kwargs)
        refined, info = refine_classifier_antecedents(model, X, y, **params)
        return model, refined, info

    def test_optimizers_backend_never_returns_a_worse_training_objective(self):
        """The invariant `_run_optimizer_search` actually provides.

        It seeds the heuristic into the solution archive and falls back to it
        explicitly if the search does not beat it, so `fit <= init_fit` holds on
        the objective the search minimises -- the k-fold cross-entropy, on the
        rows it trained on.

        This is deliberately *not* an assertion about `val_acc`. That is a
        different metric on different rows, and with the default `guard="none"`
        nothing referees it; see the two tests below. Nor is it an assertion
        about the search *path* -- whether a seeded run reproduces depends on the
        pinned `optimizers` revision (see the module docstring), so this asserts
        an invariant that holds whatever the search does.
        """
        model, refined, info = self._optimizers_refine()

        self.assertEqual(refined.n_membership_functions, model.n_membership_functions)
        self.assertIn("fit", info)
        self.assertLessEqual(info["fit"], info["init_fit"])

    def test_no_guard_promises_nothing_about_held_out_accuracy(self):
        """`guard="none"` is the default and accepts unconditionally.

        Dropping the acceptance guard (#66) was measured as worth +1.5 points in
        expectation, and the price is exactly this: `refined` is True whether or
        not held-out accuracy improved. An earlier version of this suite
        asserted `val_acc >= init_val_acc` whenever `refined` was True, which
        the code stopped guaranteeing at that commit -- it passed only because
        the search usually does improve, and failed roughly one run in six.

        Asserting the *absence* of a guarantee keeps that assertion from coming
        back on the strength of a green run.
        """
        _, _, info = self._optimizers_refine()

        self.assertTrue(info["refined"])
        self.assertEqual(info["guard"], "none")
        # Both metrics are still reported -- they are diagnostics, not contracts.
        self.assertIn("val_acc", info)
        self.assertIn("init_val_acc", info)

    def test_legacy_guard_does_promise_held_out_improvement(self):
        """The contract the old assertion belonged to, tested where it holds.

        `guard="legacy"` accepts only on a better held-out accuracy, or an equal
        one with lower cross-entropy. That decision is a deterministic function
        of the two models, so it is testable regardless of how the search got
        there.
        """
        _, _, info = self._optimizers_refine(guard="legacy")

        self.assertEqual(info["guard"], "legacy")
        if info["refined"]:
            self.assertGreaterEqual(info["val_acc"], info["init_val_acc"])
            if info["val_acc"] == info["init_val_acc"]:
                self.assertLess(info["val_ce"], info["init_val_ce"])
        else:
            self.assertLessEqual(info["val_acc"], info["init_val_acc"])

    def test_optimizers_backend_is_reproducible(self):
        """A seeded refinement must return the same model every time.

        This is a property of the pinned `optimizers` revision, not of
        `refine.py`. Before optimizers 3a57f91 the initial population came from
        `np.random.default_rng()` with no argument -- fresh OS entropy -- so
        `set_seed(seed)` did nothing and this loop produced three different
        answers in eight tries. The test exists to make a rollback of that pin
        fail here rather than surface as an intermittently red suite.
        """
        X, y, model = self._heuristic_model()
        results = []
        for _ in range(3):
            refined, info = refine_classifier_antecedents(
                model, X, y, method="optimizers", optimizer_method="ga",
                local_grad_optim="perturb", population_size=16, num_generations=5,
                local_scale=0.25, l2_shrink=0.05, seed=0, verbose=False,
            )
            results.append(
                (round(info["fit"], 12), tuple(np.round(extract_gaussian_params(refined), 12)))
            )
        self.assertEqual(len(set(results)), 1, f"seeded refinement varied: {results}")

    def test_refine_flag_on_classifier_api(self):
        clf, X, y = _quiet_fit(refine=True)
        self.assertTrue(clf.is_fitted_)
        self.assertIsNotNone(clf.refine_info_)
        preds = clf.predict(X)
        # Predictions stay within the trained label set.
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

    def test_refine_does_not_hurt_training_fit(self):
        # On the (in-sample) data the antecedents were tuned against, the refined
        # classifier should be at least as accurate as the heuristic.
        base, X, y = _quiet_fit(refine=False)
        ref, _, _ = _quiet_fit(refine=True)
        base_acc = _classifier_accuracy(X, y, base.model_)
        ref_acc = _classifier_accuracy(X, y, ref.model_)
        self.assertGreaterEqual(ref_acc, base_acc - 1e-9)


if __name__ == "__main__":
    unittest.main()
