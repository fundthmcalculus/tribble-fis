"""Tests for classifier antecedent refinement (src/tribblefis/refine.py).

A zeroth-order TSK classifier has no consequents, so its Gaussian ``(mu, sigma)``
antecedents *are* the whole model. These tests cover the two refinement backends
(``coordinate`` and the `optimizers`-package ``optimizers`` search), the
never-worse-on-validation acceptance guard, and the ``refine=True`` wiring on the
public classifier API.
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

    def test_optimizers_backend_runs_and_guards(self):
        X, y, model = self._heuristic_model()
        refined, info = refine_classifier_antecedents(
            model, X, y, method="optimizers", optimizer_method="ga",
            local_grad_optim="perturb", population_size=16, num_generations=5,
            local_scale=0.25, l2_shrink=0.05, seed=0, verbose=False,
        )
        self.assertEqual(refined.n_membership_functions, model.n_membership_functions)
        if info["refined"]:
            self.assertGreaterEqual(info["val_acc"], info["init_val_acc"])

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
