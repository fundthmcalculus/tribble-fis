"""Benchmark of candidate classifier modifications for the iris_v2 goal.

Compares, on the extended_flower_morphometrics dataset:

  * base            -- MixtureOfGaussiansFuzzyClassifier (min/max, as shipped)
  * base+product    -- same, with the product t-norm (one-line change)
  * calibrated      -- CalibratedGaussianFuzzyClassifier (candidate #3)
  * ensemble        -- BaggedFuzzyClassifier                (candidate #1)
  * bsp-tree        -- BSPFuzzyTreeClassifier               (candidate #2)

Run:  python -m gaussian_mixture.iris_v2_candidates
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
from tribblefis.calibrated_fuzzy_classifier import CalibratedGaussianFuzzyClassifier
from tribblefis.ensemble_fuzzy_classifier import BaggedFuzzyClassifier
from tribblefis.bsp_fuzzy_classifier import BSPFuzzyTreeClassifier


def load_data():
    data_path = "extended_flower_morphometrics.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)
    df = pd.read_csv(data_path).dropna()
    return df.drop("species", axis=1), df["species"]


def cv_accuracy(make_clf, X, y, n_splits=3, seed=42):
    """Mean +/- std accuracy over stratified folds (<= 3 folds per the goal)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = make_clf()
        clf.fit(X.iloc[tr], y.iloc[tr])
        accs.append(accuracy_score(y.iloc[te], clf.predict(X.iloc[te])))
    return float(np.mean(accs)), float(np.std(accs))


CANDIDATES = {
    "base (min/max)":     lambda: MixtureOfGaussiansFuzzyClassifier(top_p=1.0, n_gaussians=1, norm_conorm="min/max"),
    "base (product)":     lambda: MixtureOfGaussiansFuzzyClassifier(top_p=1.0, n_gaussians=1, norm_conorm="probability"),
    "calibrated (#3)":    lambda: CalibratedGaussianFuzzyClassifier(n_gaussians=1, top_p=1.0),
    "ensemble (#1)":      lambda: BaggedFuzzyClassifier(n_estimators=25, base="calibrated", max_features="sqrt"),
    "bsp-tree (#2)":      lambda: BSPFuzzyTreeClassifier(max_depth=10, accuracy_threshold=0.90),
}


def main():
    X, y = load_data()
    print(f"Dataset: {len(X)} rows, {X.shape[1]} features, {y.nunique()} classes\n")

    # Fixed hold-out split (matches iris_v2.py) for a single headline number.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"{'candidate':22s} {'holdout':>8s}   {'3-fold CV (mean+/-std)':>24s}   {'fit+pred s':>10s}")
    print("-" * 74)
    for name, make in CANDIDATES.items():
        t0 = time.time()
        clf = make()
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
        dt = time.time() - t0
        mean, std = cv_accuracy(make, X, y, n_splits=3)
        print(f"{name:22s} {acc:8.4f}   {mean:18.4f} +/- {std:5.4f}   {dt:10.2f}")

    # Detail on the best interpretable candidate.
    print("\n" + "=" * 74)
    print("Calibrated classifier -- hold-out classification report")
    print("=" * 74)
    clf = CalibratedGaussianFuzzyClassifier(n_gaussians=1, top_p=1.0).fit(X_tr, y_tr)
    print(classification_report(y_te, clf.predict(X_te)))


if __name__ == "__main__":
    main()
