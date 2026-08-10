"""End-to-end demo: fuzzy tree vs. the flat TRIBBLE classifier on phishing URLs.

Uses the PhiUSIIL Phishing URL dataset shipped with the repo
(``gaussian_mixture/phishing_data/PhiUSIIL_Phishing_URL_Dataset.csv``): numeric
URL/page features label each site as "legit" or "phish". Fits a
``FuzzyClassificationTree`` and the flat ``TribbleClassifier``
baseline, prints accuracy for both, renders the tree as human-readable IF-THEN
rules (with thresholds in raw feature units), and saves a matplotlib tree diagram.

The full dataset is ~236k rows; the demo subsamples for speed. Quantile-based
linguistic terms make the tree robust to the heavy skew of the count features.

Run:
    uv run python tribble-tree/demo_phishing.py
"""

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")  # quiet upstream KMeans/scipy fit warnings

sys.path.insert(0, os.path.dirname(__file__))

from fuzzytree import (
    FuzzyClassificationTree,
    HierarchicalFuzzyExpertsClassifier,
    VariablePlan,
    plot_fuzzy_tree,
    plot_hme,
    render_hme_text,
    render_tree_text,
)
from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.scaling import StandardFuzzyScalar, UnitFuzzyScalar

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "gaussian_mixture",
    "phishing_data",
    "PhiUSIIL_Phishing_URL_Dataset.csv",
)

SAMPLE_SIZE = 20000


def load_data(sample_size=SAMPLE_SIZE, random_state=42):
    """X = numeric URL/page features (raw units); y = 'legit'/'phish'."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig").dropna()
    df.columns = [c.strip() for c in df.columns]
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
    y = df["label"].map({0: "legit", 1: "phish"})
    X = df.drop(columns=["label"]).select_dtypes(include=[np.number]).astype(float)
    return X, y.to_numpy()


def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label="phish")
    print(f"  {name:<46} acc={acc:6.3f}   F1(phish)={f1:6.3f}")
    return acc, f1


def main():
    print("Loading PhiUSIIL phishing-URL data...")
    X, y = load_data()
    print(f"Samples: {len(X)}  ({(y == 'phish').mean():.1%} phishing)  "
          f"Numeric features: {X.shape[1]}\n")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("=" * 76)
    print("ACCURACY COMPARISON (classifying URLs as legit vs. phishing)")
    print("=" * 76)

    baseline = TribbleClassifier(top_n=5, random_state=42).fit(X_tr, y_tr)
    report("Flat TRIBBLE (TribbleClassifier)", y_te, baseline.predict(X_te))

    # Several of the URL/page count features here (e.g. NoOfSubDomain,
    # NoOfImage) span multiple orders of magnitude. StandardFuzzyScalar/UnitFuzzyScalar
    # are the opt-in preprocessing step for exactly that: both log-transform
    # the wide-dynamic-range columns first, then either z-score standardize
    # (StandardFuzzyScalar, mu=0/sigma=1) or min-max bound to [0, 1] (UnitFuzzyScalar).
    # Nothing in the estimator itself changes -- composing one via a Pipeline
    # is the only difference from the row above. Both rows are shown here for
    # comparison, but UnitFuzzyScalar is the recommended default: Gaussian
    # membership functions assume a bounded, non-negative domain, and the
    # unbounded centred output of StandardFuzzyScalar measurably hurts FIS
    # accuracy (see the StandardFuzzyScalar docstring for the measurement).
    standard_pipe = make_pipeline(
        StandardFuzzyScalar(), TribbleClassifier(top_n=5, random_state=42)
    ).fit(X_tr, y_tr)
    report("Flat TRIBBLE + StandardFuzzyScalar (Pipeline)", y_te, standard_pipe.predict(X_te))

    unit_pipe = make_pipeline(
        UnitFuzzyScalar(), TribbleClassifier(top_n=5, random_state=42)
    ).fit(X_tr, y_tr)
    report("Flat TRIBBLE + UnitFuzzyScalar (Pipeline)", y_te, unit_pipe.predict(X_te))

    tree = FuzzyClassificationTree(
        criterion="ambiguity", max_depth=3, n_terms=2, top_n=5, min_soft_count=50
    ).fit(X_tr, y_tr)
    report("Fuzzy tree (ambiguity splits)", y_te, tree.predict(X_te))

    tree_ig = FuzzyClassificationTree(
        criterion="info_gain", max_depth=3, n_terms=2, top_n=5, min_soft_count=50
    ).fit(X_tr, y_tr)
    report("Fuzzy tree (fuzzy info-gain splits)", y_te, tree_ig.predict(X_te))

    # A user-directed structure: HTTPS usage is a well-known phishing signal, so
    # pin IsHTTPS to the root and let the criterion choose deeper splits.
    if "IsHTTPS" in tree.top_features_:
        plan = VariablePlan(
            level_order=("IsHTTPS",), criterion="ambiguity", max_depth=3,
            default_n_terms=2, max_terms_per_var=2,
        )
        tree_plan = FuzzyClassificationTree(
            variable_plan=plan, top_n=5, min_soft_count=50
        ).fit(X_tr, y_tr)
        report("Fuzzy tree (IsHTTPS pinned to root)", y_te, tree_plan.predict(X_te))

    # Hierarchical mixture of fuzzy experts: gate (route) on the strongest
    # signals, and let a full fuzzy classifier sub-FIS decide within each region.
    hme = HierarchicalFuzzyExpertsClassifier(
        criterion="ambiguity", max_depth=2, n_gate_terms=2, top_n=5,
        min_soft_count=100, min_expert_samples=200,
        expert_kwargs={"top_n": 5},
    ).fit(X_tr, y_tr)
    report("Hierarchical fuzzy experts (gated sub-FIS)", y_te, hme.predict(X_te))

    print("\n" + "=" * 76)
    print("HUMAN-READABLE RULE TREE (ambiguity splits)")
    print("=" * 76)
    print(render_tree_text(tree))

    print("\n" + "=" * 76)
    print("HIERARCHICAL FUZZY EXPERTS (gates route to sub-FIS experts)")
    print("=" * 76)
    print(render_hme_text(hme))

    tree_png = os.path.join(os.path.dirname(__file__), "phishing_tree.png")
    plot_fuzzy_tree(tree, title="Phishing URL detection: fuzzy tree").savefig(tree_png, dpi=120)
    hme_png = os.path.join(os.path.dirname(__file__), "phishing_hme.png")
    plot_hme(hme, title="Phishing: hierarchical fuzzy experts").savefig(hme_png, dpi=120)
    print(f"\nSaved diagrams to {tree_png} and {hme_png}")


if __name__ == "__main__":
    main()
