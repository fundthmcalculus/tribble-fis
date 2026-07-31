"""Benchmark: Ruspini-partitioned triangular fuzzy classifiers.

Demonstrates the "TRIBBLE a strong initial candidate, then refine" flow:

  1. Fit the ordinary Gaussian fuzzy classifier (the implicit layout).
  2. TRIBBLE it -- convert to an explicit, Ruspini-partitioned *triangular* model
     (``ruspinize_model``): each feature is tiled by triangular terms whose
     memberships sum to exactly 1 (a partition of unity), and each class gets an
     explicit rule matched to the terms its data occupies.
  3. Refine the partition's apex knots against cross-entropy with the `optimizers`
     package (or the coordinate line-search), preserving partition-of-unity by
     construction.

Reports mean held-out test accuracy over several seeds:
    gaussian            -- the source Gaussian fuzzy classifier
    ruspini (TRIBBLE)   -- the triangular Ruspini candidate, unrefined
    ruspini + coord     -- knots refined by coordinate line-search
    ruspini + optimizers-- knots refined by the optimizers-package search

Run:  python gaussian_mixture/ruspini_benchmark.py
"""

import io
import contextlib
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
from tribblefis.ruspini import RuspiniFuzzyClassifier

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2, 3, 4]

DATASETS = {
    "iris": (load_iris, dict(top_p=1.0, n_gaussians=1)),
    "wine": (load_wine, dict(top_n=8, n_gaussians=1)),
    "breast_cancer": (load_breast_cancer, dict(top_n=10, n_gaussians=1)),
}


def _acc(model_factory, Xtr, ytr, Xte, yte):
    with contextlib.redirect_stdout(io.StringIO()):
        m = model_factory()
        m.fit(Xtr, ytr)
        return accuracy_score(yte, m.predict(Xte))


def main():
    print(f"{'dataset':16s}{'gaussian':>10s}{'TRIBBLE':>10s}{'+coord':>10s}{'+optim':>10s}")
    print("-" * 56)
    for name, (loader, kw) in DATASETS.items():
        d = loader(as_frame=True)
        X, y = d.data, pd.Series(d.target)
        g, t, c, o = [], [], [], []
        for s in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=s, stratify=y)
            Xtr = Xtr.reset_index(drop=True); ytr = np.asarray(ytr)
            g.append(_acc(lambda: MixtureOfGaussiansFuzzyClassifier(random_state=s, **kw), Xtr, ytr, Xte, yte))
            t.append(_acc(lambda: RuspiniFuzzyClassifier(random_state=s, refine=False, **kw), Xtr, ytr, Xte, yte))
            c.append(_acc(lambda: RuspiniFuzzyClassifier(random_state=s, refine=True, refine_method="coordinate", **kw), Xtr, ytr, Xte, yte))
            o.append(_acc(lambda: RuspiniFuzzyClassifier(random_state=s, refine=True, refine_method="optimizers", **kw), Xtr, ytr, Xte, yte))
        print(f"{name:16s}{np.mean(g):>10.4f}{np.mean(t):>10.4f}{np.mean(c):>10.4f}{np.mean(o):>10.4f}   "
              f"(refine +{np.mean(c)-np.mean(t):.3f}/+{np.mean(o)-np.mean(t):.3f})")


if __name__ == "__main__":
    main()
