"""End-to-end demo: fuzzy tree vs. the flat TRIBBLE regressor on concrete strength.

Uses the UCI Concrete Compressive Strength dataset shipped with the repo
(``gaussian_mixture/Concrete_Data.csv``): 8 mixture/age features predict the
compressive strength (MPa). Fits a ``FuzzyRegressionTree`` and the flat
``TribbleRegressor`` baseline, prints accuracy for both, renders
the tree as human-readable IF-THEN rules (with physically meaningful thresholds
in raw units), and saves a matplotlib tree diagram.

Run:
    uv run python tribble-tree/demo_concrete.py
"""

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")  # quiet upstream KMeans/scipy fit warnings

sys.path.insert(0, os.path.dirname(__file__))

from fuzzytree import (
    FuzzyRegressionTree,
    HierarchicalFuzzyExpertsRegressor,
    VariablePlan,
    plot_fuzzy_tree,
    plot_hme,
    render_hme_text,
    render_tree_text,
)
from tribblefis.gaussian_regressor import TribbleRegressor

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "gaussian_mixture", "Concrete_Data.csv"
)


def load_data():
    """X = 8 concrete mixture/age features (raw units); y = strength in MPa.

    Raw (unscaled) features are kept on purpose so the tree's split thresholds are
    physically meaningful, e.g. "Cement is High >= 350". If your own features need
    bounding or span multiple scales, compose ``tribblefis.scaling.UnitFuzzyScalar``
    ([0, 1] bounding, the recommended default for FIS estimators) in front of the
    estimator with ``sklearn.pipeline.make_pipeline`` instead -- see
    ``demo_phishing.py`` for an example.
    """
    df = pd.read_csv(DATA_PATH).dropna()
    df.columns = [c.strip() for c in df.columns]
    y = df["Strength"].astype(float)
    X = df.drop(columns=["Strength"]).select_dtypes(include=[np.number]).astype(float)
    return X, y.to_numpy()


def report(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {name:<44} R2={r2:6.3f}   RMSE={rmse:6.3f} MPa")
    return r2, rmse


def main():
    print("Loading UCI concrete compressive-strength data...")
    X, y = load_data()
    print(f"Samples: {len(X)}  Features: {list(X.columns)}\n")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    print("=" * 74)
    print("ACCURACY COMPARISON (predicting concrete compressive strength, MPa)")
    print("=" * 74)

    baseline = TribbleRegressor(
        n_output_buckets=3, tsk_order="1st", top_n=-1, random_state=42
    ).fit(X_tr, y_tr)
    report("Flat TRIBBLE (TribbleRegressor)", y_te, baseline.predict(X_te))

    tree0 = FuzzyRegressionTree(
        tsk_order="0th", criterion="variance", max_depth=3, n_terms=2, top_n=4, min_soft_count=20
    ).fit(X_tr, y_tr)
    report("Fuzzy tree (0th-order, constant leaves)", y_te, tree0.predict(X_te))

    tree1 = FuzzyRegressionTree(
        tsk_order="1st", criterion="variance", max_depth=3, n_terms=2, top_n=4, min_soft_count=20
    ).fit(X_tr, y_tr)
    report("Fuzzy tree (1st-order, linear leaves)", y_te, tree1.predict(X_te))

    # A user-directed structure: cement content is the dominant strength driver,
    # so pin Cement to the root and let the criterion choose the rest.
    plan = VariablePlan(
        level_order=("Cement",),
        criterion="variance", max_depth=3,
        default_n_terms=2, max_terms_per_var=2,
    )
    tree_plan = FuzzyRegressionTree(
        variable_plan=plan, tsk_order="1st", top_n=4, min_soft_count=20
    ).fit(X_tr, y_tr)
    report("Fuzzy tree (1st-order, Cement pinned to root)", y_te, tree_plan.predict(X_te))

    # Hierarchical mixture of fuzzy experts: gate (route) on the strongest
    # features, and let a full TSK sub-FIS predict within each region.
    hme = HierarchicalFuzzyExpertsRegressor(
        criterion="variance", max_depth=2, n_gate_terms=2, top_n=4,
        min_soft_count=40, min_expert_samples=60,
        expert_kwargs={"n_output_buckets": 3, "tsk_order": "1st"},
    ).fit(X_tr, y_tr)
    report("Hierarchical fuzzy experts (gated sub-FIS)", y_te, hme.predict(X_te))

    print("\n" + "=" * 74)
    print("HUMAN-READABLE RULE TREE (1st-order fuzzy tree, auto structure)")
    print("=" * 74)
    print(render_tree_text(tree1))

    print("\n" + "=" * 74)
    print("HIERARCHICAL FUZZY EXPERTS (gates route to sub-FIS experts)")
    print("=" * 74)
    print(render_hme_text(hme))

    tree_png = os.path.join(os.path.dirname(__file__), "concrete_tree.png")
    plot_fuzzy_tree(tree1, title="Concrete compressive strength: fuzzy tree").savefig(tree_png, dpi=120)
    hme_png = os.path.join(os.path.dirname(__file__), "concrete_hme.png")
    plot_hme(hme, title="Concrete: hierarchical fuzzy experts").savefig(hme_png, dpi=120)
    print(f"\nSaved diagrams to {tree_png} and {hme_png}")


if __name__ == "__main__":
    main()
