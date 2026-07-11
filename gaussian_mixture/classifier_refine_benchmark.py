"""Benchmark: fuzzy-classifier antecedent refinement vs the heuristic baseline.

A zeroth-order TSK classifier predicts ``argmax`` of the per-class Gaussian firing
strengths, so its ``(mu, sigma)`` antecedents *are* the whole model. The heuristic
membership fit (KMeans + ``norm.fit`` per class) only matches each feature/label
marginal -- it never looks at the classification objective. ``refine=True`` tunes
those antecedents against a cross-entropy loss (with a ridge shrinkage toward the
heuristic and a held-out acceptance guard, so it can never make things worse).

This script reports mean held-out test accuracy over several seeds for:
  * baseline   -- the heuristic Gaussian classifier;
  * coordinate -- per-membership block coordinate descent (fast, default);
  * optimizers -- the population + local-polish search from the `optimizers`
                  package (github.com/fundthmcalculus/optimizers).

Run:  python gaussian_mixture/classifier_refine_benchmark.py
Data: sklearn built-in datasets (always available) + darwin.csv if present.
"""

import os
import io
import contextlib
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
from tribblefis.refine import refine_classifier_antecedents

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2, 3, 4]


def _sk(loader, **kw):
    d = loader(as_frame=True)
    return d.data, pd.Series(d.target), kw


def _darwin(**kw):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "darwin.csv")
    df = pd.read_csv(path).dropna()
    y = df["class"]
    X = df.drop(["class"], axis=1).select_dtypes(include=[np.number])
    return X, y, kw


DATASETS = {
    "iris": lambda: _sk(load_iris, top_p=1.0, n_gaussians=1),
    "wine": lambda: _sk(load_wine, top_n=8, n_gaussians=1),
    "breast_cancer": lambda: _sk(load_breast_cancer, top_n=10, n_gaussians=1),
    "darwin": lambda: _darwin(top_n=20, n_gaussians=1),
}


def _eval(X, y, kw, method, seed, **extra):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.33, random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    Xtr = Xtr.reset_index(drop=True)
    ytr = np.asarray(ytr)
    with contextlib.redirect_stdout(io.StringIO()):
        clf = MixtureOfGaussiansFuzzyClassifier(random_state=seed, **kw)
        clf.fit(Xtr, ytr)
        if method != "baseline":
            clf.model_, _ = refine_classifier_antecedents(
                clf.model_, Xtr, ytr, method=method, l2_shrink=0.05,
                seed=seed, verbose=False, **extra,
            )
    return accuracy_score(yte, clf.predict(Xte))


def main():
    opt_kw = dict(optimizer_method="ga", local_grad_optim="perturb",
                  population_size=24, num_generations=10)
    print(f"{'dataset':16s}{'baseline':>10s}{'coordinate':>12s}{'optimizers':>12s}")
    print("-" * 50)
    for name, loader in DATASETS.items():
        try:
            X, y, kw = loader()
        except FileNotFoundError:
            print(f"{name:16s}{'(data not present -- skipped)':>34s}")
            continue
        base = np.mean([_eval(X, y, kw, "baseline", s) for s in SEEDS])
        coor = np.mean([_eval(X, y, kw, "coordinate", s) for s in SEEDS])
        opti = np.mean([_eval(X, y, kw, "optimizers", s, **opt_kw) for s in SEEDS])
        print(f"{name:16s}{base:>10.4f}{coor:>12.4f}{opti:>12.4f}   "
              f"(+{coor - base:.4f} / +{opti - base:.4f})")


if __name__ == "__main__":
    main()
