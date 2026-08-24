"""Evaluate error-driven adaptive rule partitioning against the uniform qcut
partition on the Concrete dataset.

`MixtureOfGaussiansFuzzyRegressor(bucket_strategy="uniform")` splits y into a
fixed number of equal-frequency buckets. `bucket_strategy="adaptive"` instead
grows the rule set from a single rule, splitting the worst-fitting rule
(lowest local R^2) until every rule clears a threshold or a rule budget is
hit (see `tribblefis.adaptive_partition.grow_adaptive_partition`). This
script fits both at matched rule counts on a held-out test split and reports
whether the adaptive partition is worth its extra fitting cost.
"""

import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concrete import load_data  # noqa: E402

from tribblefis.gauss_math import detect_and_apply_log_transform, standard_transform  # noqa: E402
from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor  # noqa: E402


def main():
    X, y_raw = load_data()
    y_raw = standard_transform(y_raw)

    X, log_transformed_features = detect_and_apply_log_transform(X, min_dynamic_range=2)
    X = standard_transform(X, column=X.columns)
    if log_transformed_features:
        print(f"Auto-detected log transform for: {log_transformed_features}")

    X_train, X_test, y_train, y_test = train_test_split(X, y_raw, test_size=0.2, random_state=42)
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    tsk_order = "1st"
    max_rules = 8
    r2_threshold = 0.9
    min_bucket_samples = 20

    print("\nUniform partition sweep (bucket_strategy='uniform'):")
    print("=" * 60)
    # k=1 is skipped: with a single output bucket there is only one label, so the
    # feature-differentiation ranking (which compares distributions across
    # labels) scores every feature 0 and selects none -- a pre-existing property
    # of the uniform strategy's feature-selection step, not of the model itself.
    uniform_rows = []
    for k in range(2, max_rules + 1):
        reg = MixtureOfGaussiansFuzzyRegressor(
            n_output_buckets=k, tsk_order=tsk_order, bucket_strategy="uniform",
        )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        uniform_rows.append((k, r2, rmse))
        print(f"  n_rules={k:2d}  test R2={r2:7.4f}  test RMSE={rmse:7.4f}")

    adaptive_results = {}
    for split_method in ["median", "sse"]:
        print(f"\nAdaptive partition (bucket_strategy='adaptive', split_method='{split_method}'):")
        print("=" * 60)
        adaptive_reg = MixtureOfGaussiansFuzzyRegressor(
            tsk_order=tsk_order, bucket_strategy="adaptive",
            max_rules=max_rules, bucket_r2_threshold=r2_threshold,
            min_bucket_samples=min_bucket_samples, adaptive_split_method=split_method,
        )
        adaptive_reg.fit(X_train, y_train)
        y_pred_adaptive = adaptive_reg.predict(X_test)
        adaptive_r2 = r2_score(y_test, y_pred_adaptive)
        adaptive_rmse = root_mean_squared_error(y_test, y_pred_adaptive)
        adaptive_n_rules = adaptive_reg.n_rules_
        adaptive_results[split_method] = (adaptive_n_rules, adaptive_r2, adaptive_rmse)

        print(f"  Growth trace (train R2 and edges per iteration):")
        for entry in adaptive_reg.partition_history_:
            print(
                f"    n_rules={entry['n_rules']:2d}  train R2={entry['train_r2']:7.4f}  "
                f"edges={[round(e, 3) for e in entry['edges']]}  bucket_r2={{"
                + ", ".join(f"{k}: {v:.3f}" for k, v in entry['bucket_r2'].items()) + "}}"
            )
        print(f"  Final: n_rules={adaptive_n_rules}  test R2={adaptive_r2:.4f}  test RMSE={adaptive_rmse:.4f}")

    print("\nHead-to-head at matched rule count:")
    print("=" * 60)
    for split_method, (adaptive_n_rules, adaptive_r2, adaptive_rmse) in adaptive_results.items():
        matched = next((row for row in uniform_rows if row[0] == adaptive_n_rules), None)
        if matched is not None:
            print(f"  uniform             n_rules={matched[0]:2d}  test R2={matched[1]:7.4f}  test RMSE={matched[2]:7.4f}")
        print(f"  adaptive ({split_method:>6}) n_rules={adaptive_n_rules:2d}  test R2={adaptive_r2:7.4f}  test RMSE={adaptive_rmse:7.4f}")

    ks = [row[0] for row in uniform_rows]
    r2s = [row[1] for row in uniform_rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, r2s, marker="o", label="uniform (qcut)")
    markers = {"median": ("*", "red"), "sse": ("D", "green")}
    for split_method, (adaptive_n_rules, adaptive_r2, _) in adaptive_results.items():
        marker, color = markers[split_method]
        ax.scatter([adaptive_n_rules], [adaptive_r2], marker=marker, s=200, color=color,
                   label=f"adaptive ({split_method})", zorder=5)
    ax.set_xlabel("Number of rules")
    ax.set_ylabel("Test R²")
    ax.set_title("Concrete: uniform vs. adaptive rule partitioning")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("concrete_adaptive_comparison.png", dpi=150)
    print("\nSaved plot to concrete_adaptive_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
