"""Stage A sanity/recovery check for `fuzzytree.deconstruct`.

Synthetic regression with a known group structure (3 feature groups, no
cross-group interaction) so the "true" topology is known by construction.
Compares:

    (i)   flat TribbleRegressor on all features (no structure at all)
    (ii)  HierarchicalFuzzyExpertsRegressor, auto topology (today's top-down
          "fit each leaf from scratch on a row subset" approach)
    (iii) DeconstructedHierarchicalRegressor, given the TRUE topology (fit one
          flat model, then deconstruct it per feature group)

and reports whether (iii) actually recovers each group's own latent
contribution (per-leaf R^2 against the known noiseless group signal), not
just the final blended prediction.

Run:
    uv run python tribble-tree/demo_deconstruct_synthetic.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from demo_utils import evaluate_model, regressor_report

# Retain report for internal leaf prediction, which doesn't go through evaluate_model
def report(name, y_true, y_pred):
    return regressor_report(name, y_true, y_pred)
from fuzzytree import DeconstructedHierarchicalRegressor, HierarchicalFuzzyExpertsRegressor
from tribblefis.gaussian_regressor import TribbleRegressor

TOPOLOGY = {
    "TOTAL": ["G1", "G2", "G3"],
    "G1": ["a", "b"],
    "G2": ["c", "d"],
    "G3": ["e", "f"],
}
TRUE_WEIGHTS = {"G1": 1.0, "G2": 0.7, "G3": 1.4}


def make_data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {name: rng.uniform(0, 10, n) for name in ["a", "b", "c", "d", "e", "f"]}
    )
    g1 = np.sin(X["a"] / 2) * 5 + 0.5 * X["b"]
    g2 = np.cos(X["c"] / 3) * 4 - 0.3 * X["d"]
    g3 = np.tanh((X["e"] - 5) / 2) * 6 + 0.2 * X["f"]
    groups = {"G1": g1, "G2": g2, "G3": g3}
    y = sum(TRUE_WEIGHTS[k] * v for k, v in groups.items()) + rng.normal(0, 0.3, n)
    return X, y.to_numpy(), groups


def main():
    print("Synthetic 3-group dataset (a,b -> G1; c,d -> G2; e,f -> G3; no cross-group interaction)\n")
    X, y, groups = make_data()
    idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)
    X_tr, X_te = X.iloc[idx_tr].reset_index(drop=True), X.iloc[idx_te].reset_index(drop=True)
    y_tr, y_te = y[idx_tr], y[idx_te]

    print("=" * 78)
    print("ACCURACY COMPARISON (predicting y, held-out test split)")
    print("=" * 78)

    flat = TribbleRegressor(n_output_buckets=5, tsk_order="1st", top_n=-1, random_state=42).fit(X_tr, y_tr)
    evaluate_model(flat, X_te, y_te, "Flat TRIBBLE (TribbleRegressor, no structure)", regressor_report)

    hme = HierarchicalFuzzyExpertsRegressor(
        criterion="variance", max_depth=2, n_gate_terms=2, top_n=4,
        min_soft_count=40, min_expert_samples=60,
        expert_kwargs={"n_output_buckets": 4, "tsk_order": "1st"},
    ).fit(X_tr, y_tr)
    evaluate_model(hme, X_te, y_te, "HME (auto topology, fits sub-FIS per leaf row subset)", regressor_report)

    deconstructed = DeconstructedHierarchicalRegressor(
        flat_regressor_kwargs={"n_output_buckets": 5, "top_n": -1, "random_state": 42},
    ).fit(X_tr, y_tr, TOPOLOGY)
    evaluate_model(deconstructed, X_te, y_te, "Deconstructed tree (true topology, flat-then-deconstruct)", regressor_report)

    print("\n" + "=" * 78)
    print("PER-LEAF RECOVERY (deconstructed tree only): does each leaf's own fitted")
    print("output track its group's TRUE noiseless latent contribution?")
    print("=" * 78)
    for leaf in deconstructed.root_.iter_leaves():
        leaf_pred = deconstructed._predict_node(leaf, X_te)
        true_contribution = groups[leaf.name].to_numpy()[idx_te]
        report(f"  leaf {leaf.name} vs. true latent contribution", true_contribution, leaf_pred)

    print("\n" + "=" * 78)
    print("ROOT BRANCH COMBINER: fitted affine weights vs. true generating weights")
    print("=" * 78)
    root_state = deconstructed.node_state_["TOTAL"]
    for name, coeff in zip(root_state["children"], root_state["corr_terms"][0]):
        print(f"  {name}: fitted={coeff:6.3f}   true={TRUE_WEIGHTS[name]:6.3f}")

    # Fold-1-style ablation: supervise each leaf directly on its own TRUE
    # latent group contribution (leaf_targets), instead of letting every leaf
    # independently try to predict the whole blended y from only its own
    # features (the default above). This is the whiteboard's "once theta_m is
    # assigned, train RUL" vs. "sensor -> leaf" distinction.
    print("\n" + "=" * 78)
    print("FOLD-1 ABLATION: leaves supervised directly on their TRUE latent")
    print("group contribution (leaf_targets), instead of on the blended y")
    print("=" * 78)
    leaf_targets_tr = {name: groups[name].to_numpy()[idx_tr] for name in ("G1", "G2", "G3")}
    deconstructed_oracle_leaves = DeconstructedHierarchicalRegressor(
        flat_regressor_kwargs={"n_output_buckets": 5, "top_n": -1, "random_state": 42},
    ).fit(X_tr, y_tr, TOPOLOGY, leaf_targets=leaf_targets_tr)
    evaluate_model(deconstructed_oracle_leaves, X_te, y_te, "Deconstructed tree (leaves supervised on true group signal)", regressor_report)
    for leaf in deconstructed_oracle_leaves.root_.iter_leaves():
        leaf_pred = deconstructed_oracle_leaves._predict_node(leaf, X_te)
        true_contribution = groups[leaf.name].to_numpy()[idx_te]
        report(f"  leaf {leaf.name} vs. true latent contribution", true_contribution, leaf_pred)
    root_state_oracle = deconstructed_oracle_leaves.node_state_["TOTAL"]
    for name, coeff in zip(root_state_oracle["children"], root_state_oracle["corr_terms"][0]):
        print(f"  {name}: fitted={coeff:6.3f}   true={TRUE_WEIGHTS[name]:6.3f}")


if __name__ == "__main__":
    main()
