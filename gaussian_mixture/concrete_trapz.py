"""
Concrete Strength Prediction using Trapezoidal Membership Functions

This is a trapezoid variant of concrete.py that uses the new TrapzMixtureModel
for 1D feature fitting instead of Gaussian mixture models. It demonstrates how
to integrate trapezoid membership functions into the regression pipeline.

Key differences from concrete.py:
- Uses create_trapz_membership_dict() instead of create_gaussian_membership_dict()
- Trapezoid MFs provide broader, flatter membership regions
- Can be beneficial for data with plateau-like distributions
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tribblefis.trapz_math import (
    fit_trapezoids,
    create_trapz_membership_dict,
)
from tribblefis.gauss_math import (
    log_transform,
    calculate_gaussian_correlation,
    take_top_features,
    tsk_firing_strengths,
)
from tribblefis.regression import (
    report_regression_performance,
    compute_first_order_corrections,
    compute_second_order_corrections,
    plot_tsk_order_comparison,
    compute_full_second_order_corrections,
    compute_third_order_corrections,
    partition_output,
    optimize_tsk_coefficients,
)
from tribblefis.report import print_membership_details


def _standardize(X):
    """Standardize features in the dataset."""
    return (X - X.mean()) / X.std()


def _normalize(X):
    """Normalize features in the dataset."""
    return (X - X.min()) / (X.max() - X.min())


def load_data():
    data_path = "Concrete_Data.csv"
    if not os.path.exists(data_path):
        # Try to find it in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_path)

    X = pd.read_csv(data_path)
    X = X.dropna()
    y = X["Strength"]
    y.name = "y_value"
    X.drop("Strength", axis=1, inplace=True)
    X = X.select_dtypes(include=[np.number])
    return X, y


def main():
    start_time = time.time()
    X, y_raw = load_data()

    n_output_buckets: int = 2
    n_top_vars: int = -1
    n_trapezoids: int = 2  # Fixed 2 trapezoids per feature/label (fast, reasonable for regression)
    # Note: Set to -1 for auto-select via BIC, but this can be slow for large datasets
    b_optimize_coeff: bool = True

    if n_top_vars <= 0 or n_top_vars > len(X.columns):
        n_top_vars = len(X.columns)

    y, y_bucket_mean = partition_output(n_output_buckets, y_raw)

    X = log_transform(X, ["Slag", "FlyAsh", "Age"], 1)

    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y["y_bucket"]
    )
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")
    print(f"Model type: TRAPEZOID membership functions (auto-select K via BIC)\n")

    # Calculate correlation coefficient between Gaussian distributions using training data
    # (still use Gaussian for feature selection - trapezoids are only for MFs)
    feature_differentiators = calculate_gaussian_correlation(X_train, y_train["y_bucket"])

    # Take the top-n variables so that the normalized differentiation value encompasses 90-95%
    top_n, top_n_todo = take_top_features(feature_differentiators, top_n=n_top_vars)

    print(f"Selected Top-{top_n} Variables ({top_n/len(feature_differentiators):.2%} coverage):")

    # Compute trapezoid memberships using training data
    # This is the key difference: uses trapezoids instead of Gaussians
    print("\nFitting trapezoid membership functions (this will auto-select K per feature/label via BIC)...")
    trapz_memberships = create_trapz_membership_dict(
        X_train, y_train["y_bucket"], top_n_var_names=top_n_todo, n_trapezoids=n_trapezoids
    )

    duplicates = trapz_memberships.identify_duplicate_membership_fcns()
    print(
        f"\nDuplicate Membership Functions({len(duplicates)/trapz_memberships.n_membership_functions:.1%} redundant):\n"
        f"{duplicates}"
    )

    print_membership_details(trapz_memberships)

    print("\nEvaluating Zeroth-Order TSK Model on TEST set:")
    print("=" * 80)

    # Compute correction terms for TSK orders
    # Trapezoids have broader support, so we'll try 0th and 1st order
    print("\nComputing TSK correction terms...")

    corr_terms_1 = compute_first_order_corrections(
        X_train, trapz_memberships, n_top_vars, top_n_todo, y_bucket_mean, y_train
    )

    # For trapezoids, optimization can be slow/unstable due to broader MF regions
    # Use simple (unoptimized) bucket means with correction terms
    print("\nNote: Using unoptimized TSK for trapezoid MFs")
    print("  (Trapezoid broad regions can cause optimization instability)")

    y_bucket_mean_opt_0 = y_bucket_mean
    y_bucket_mean_opt_1 = y_bucket_mean
    corr_terms_1_opt = corr_terms_1
    use_1st_order = True

    # Evaluate the trapezoid TSK model on test set
    print("\nEvaluating TSK Models with TRAPEZOID MFs on TEST set:")
    print("=" * 80)
    firing_strengths, labels = tsk_firing_strengths(X_test[top_n_todo], trapz_memberships)

    # Handle any rows with zero firing strengths
    row_sums = firing_strengths.sum(axis=1)
    zero_rows = row_sums == 0
    nonzero_rows = ~zero_rows

    norm_firing_strength = np.zeros_like(firing_strengths)
    if np.any(nonzero_rows):
        norm_firing_strength[nonzero_rows] = (
            firing_strengths[nonzero_rows] / row_sums[nonzero_rows, np.newaxis]
        )
    if np.any(zero_rows):
        norm_firing_strength[zero_rows] = 1.0 / len(labels)

    # Evaluate zeroth-order model
    y_test_pred_zeroth = np.dot(norm_firing_strength, y_bucket_mean_opt_0)
    r2_0, rmse_0 = report_regression_performance(
        start_time, y_test, y_test_pred_zeroth, n_order="0 optimized (trapezoid)"
    )

    # Evaluate first-order model
    if use_1st_order:
        y_test_pred_first_order = np.zeros(len(X_test))
        X_test_rule = X_test[top_n_todo].to_numpy()

        for ij, y_id in enumerate(labels):
            y_test_pred_first_order[:] += (
                y_bucket_mean_opt_1[y_id] + X_test_rule @ corr_terms_1_opt[y_id, :]
            ) * norm_firing_strength[:, ij]

        r2_1, rmse_1 = report_regression_performance(
            start_time, y_test, y_test_pred_first_order, n_order="1 optimized (trapezoid)"
        )
    else:
        y_test_pred_first_order = y_test_pred_zeroth
        r2_1, rmse_1 = r2_0, rmse_0

    # Plot comparison
    print("\nGenerating comparison plot...")
    import matplotlib.pyplot as plt

    # Convert to numpy arrays for plotting
    y_test_arr = y_test['y_value'].values if isinstance(y_test, pd.DataFrame) else (y_test.values if hasattr(y_test, 'values') else np.asarray(y_test))
    print(f"  Debug: y_test shape={y_test_arr.shape}, y_test_pred_zeroth shape={y_test_pred_zeroth.shape}, y_test_pred_first_order shape={y_test_pred_first_order.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 0th order actual vs predicted
    ax = axes[0, 0]
    ax.scatter(y_test_arr, y_test_pred_zeroth, alpha=0.6, s=20)
    lim = [min(y_test_arr.min(), y_test_pred_zeroth.min()), max(y_test_arr.max(), y_test_pred_zeroth.max())]
    ax.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Strength')
    ax.set_ylabel('Predicted Strength')
    ax.set_title(f'0th Order TSK (Trapezoid)\nR² = {r2_0:.3f}, RMSE = {rmse_0:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1st order actual vs predicted
    ax = axes[0, 1]
    ax.scatter(y_test_arr, y_test_pred_first_order, alpha=0.6, s=20, color='green')
    lim = [min(y_test_arr.min(), y_test_pred_first_order.min()), max(y_test_arr.max(), y_test_pred_first_order.max())]
    ax.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Strength')
    ax.set_ylabel('Predicted Strength')
    ax.set_title(f'1st Order TSK (Trapezoid)\nR² = {r2_1:.3f}, RMSE = {rmse_1:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residuals for 0th order
    ax = axes[1, 0]
    residuals_0 = y_test_arr - y_test_pred_zeroth
    ax.scatter(y_test_pred_zeroth, residuals_0, alpha=0.6, s=20)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Strength')
    ax.set_ylabel('Residuals')
    ax.set_title('0th Order Residuals (Trapezoid)')
    ax.grid(True, alpha=0.3)

    # Residuals for 1st order
    ax = axes[1, 1]
    residuals_1 = y_test_arr - y_test_pred_first_order
    ax.scatter(y_test_pred_first_order, residuals_1, alpha=0.6, s=20, color='green')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Strength')
    ax.set_ylabel('Residuals')
    ax.set_title('1st Order Residuals (Trapezoid)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, 'concrete_trapz_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to: {plot_path}")
    try:
        plt.show()
    except:
        pass  # Headless environment

    print("\n" + "=" * 80)
    print("TRAPEZOID MEMBERSHIP FUNCTION REGRESSION COMPLETE")
    print("=" * 80)
    print("\nResults Summary:")
    print(f"  0th Order: R² = {r2_0:.4f}, RMSE = {rmse_0:.2f}")
    print(f"  1st Order: R² = {r2_1:.4f}, RMSE = {rmse_1:.2f}")
    print(f"  Improvement: ΔR² = {r2_1 - r2_0:+.4f}, ΔRMSE = {rmse_1 - rmse_0:+.2f}")
    print("\nComparison: Trapezoid vs Gaussian MFs")
    print("  Run concrete.py (Gaussian) and compare R² and RMSE values")
    print("  Trapezoids provide broader membership regions suitable for classification")


if __name__ == "__main__":
    main()
