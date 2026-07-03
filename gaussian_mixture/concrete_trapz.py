"""
Concrete Strength Prediction: Gaussian vs Trapezoidal Membership Functions

Unified comparison of Gaussian and Trapezoid membership function models.
- Gaussian MFs: Smooth, bell-curve shaped membership regions
- Trapezoid MFs: Broader, flatter membership regions with sharper transitions
Both models support 0th-3rd order TSK with optimized coefficients.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tribblefis.gauss_math import (
    log_transform,
    calculate_gaussian_correlation,
    create_gaussian_membership_dict,
    take_top_features,
    tsk_firing_strengths,
)
from tribblefis.regression import (
    report_regression_performance,
    compute_first_order_corrections,
    compute_second_order_corrections,
    compute_third_order_corrections,
    compute_full_second_order_corrections,
    optimize_tsk_coefficients,
    plot_tsk_order_comparison,
    partition_output,
)
from tribblefis.report import print_membership_details
from tribblefis.trapz_math import create_trapz_membership_dict
from tribblefis.trapz_math_fast import create_trapz_membership_dict_fast


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
    X = _standardize(X)
    X = _normalize(X)
    return X, y


def run_model(model_type, X_train, X_test, y_train, y_test, y_bucket_mean, top_n_todo, n_top_vars, start_time, n_output_buckets):
    """Run either Gaussian or Trapz model with full optimization."""
    print(f"\n{'=' * 80}")
    print(f"EVALUATING {model_type.upper()} MEMBERSHIP FUNCTION MODEL")
    print(f"{'=' * 80}")

    # Create memberships
    if model_type == "gaussian":
        memberships = create_gaussian_membership_dict(
            X_train, y_train["y_bucket"], top_n_var_names=top_n_todo, n_gaussians=-1
        )
    elif model_type == "trapz-fast":
        memberships = create_trapz_membership_dict_fast(
            X_train, y_train["y_bucket"], top_n_var_names=top_n_todo,
        )
    elif model_type == "trapz":
        memberships = create_trapz_membership_dict(
            X_train, y_train["y_bucket"], top_n_var_names=top_n_todo,
        )

    duplicates = memberships.identify_duplicate_membership_fcns()
    print(
        f"Duplicate Membership Functions ({len(duplicates)/memberships.n_membership_functions:.1%} redundant):\n"
        f"{duplicates}\n"
    )
    print_membership_details(memberships)

    # Compute correction terms (used as initial guess for coefficient optimization)
    print("\nComputing TSK correction terms...")
    corr_terms_1 = compute_first_order_corrections(
        X_train, memberships, n_top_vars, top_n_todo, y_bucket_mean, y_train
    )
    corr_terms_2 = compute_second_order_corrections(
        X_train, memberships, n_top_vars, top_n_todo, y_bucket_mean, y_train
    )
    corr_terms_3 = compute_third_order_corrections(
        X_train, memberships, n_top_vars, top_n_todo, y_bucket_mean, y_train
    )
    corr_terms_2f = compute_full_second_order_corrections(
        X_train, memberships, n_top_vars, top_n_todo, y_bucket_mean, y_train
    )

    # Jointly optimize the output (consequent) coefficients against the
    # firing-strength-weighted training error. This tunes both the constant
    # bucket-mean terms and the per-rule correction terms, rather than fixing
    # the constants at the raw bucket means.
    #
    # Trapezoids have bounded support, so their high-order least-squares initial
    # guess is ill-conditioned (coefficients ~1e4) and overfits badly without a
    # mild ridge penalty. Gaussian MFs have infinite support and generalize best
    # unregularized, so only regularize the trapezoid consequents.
    l2_reg = 1e-4 if model_type in ("trapz", "trapz-fast") else 0.0
    print("\nOptimizing TSK output coefficients...")
    corr_terms_1, y_bucket_mean_1 = optimize_tsk_coefficients(
        X_train, memberships, top_n_todo, y_bucket_mean, y_train,
        n_output_buckets=n_output_buckets, initial_corr_terms=corr_terms_1, order="1st", l2_reg=l2_reg,
    )
    corr_terms_2, y_bucket_mean_2 = optimize_tsk_coefficients(
        X_train, memberships, top_n_todo, y_bucket_mean, y_train,
        n_output_buckets=n_output_buckets, initial_corr_terms=corr_terms_2, order="2nd", l2_reg=l2_reg,
    )
    corr_terms_2f, y_bucket_mean_2f = optimize_tsk_coefficients(
        X_train, memberships, top_n_todo, y_bucket_mean, y_train,
        n_output_buckets=n_output_buckets, initial_corr_terms=corr_terms_2f, order="full-2nd", l2_reg=l2_reg,
    )
    corr_terms_3, y_bucket_mean_3 = optimize_tsk_coefficients(
        X_train, memberships, top_n_todo, y_bucket_mean, y_train,
        n_output_buckets=n_output_buckets, initial_corr_terms=corr_terms_3, order="3rd", l2_reg=l2_reg,
    )
    _, y_bucket_mean_0 = optimize_tsk_coefficients(
        X_train, memberships, top_n_todo, y_bucket_mean, y_train,
        n_output_buckets=n_output_buckets, order="0th", l2_reg=l2_reg,
    )

    # Evaluate on test set
    print("\nEvaluating on TEST set...")
    firing_strengths, labels = tsk_firing_strengths(X_test[top_n_todo], memberships)

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

    # 0th order
    y_test_pred_0 = np.dot(norm_firing_strength, y_bucket_mean_0)
    r2_0, rmse_0 = report_regression_performance(
        start_time, y_test, y_test_pred_0, n_order=f"0 optimized ({model_type})"
    )

    # 1st order
    y_test_pred_1 = np.zeros(len(X_test))
    # 2nd order
    y_test_pred_2 = np.zeros(len(X_test))
    # 2nd full order
    y_test_pred_2f = np.zeros(len(X_test))
    # 3rd order
    y_test_pred_3 = np.zeros(len(X_test))

    X_test_rule = X_test[top_n_todo].to_numpy()
    X_test_rule2 = np.hstack([X_test_rule, X_test_rule**2])
    X_test_rule3 = np.hstack([X_test_rule, X_test_rule**2, X_test_rule**3])

    # Build cross-terms for full 2nd order
    cross_terms = []
    for i in range(X_test_rule.shape[1]):
        for j in range(i + 1, X_test_rule.shape[1]):
            cross_terms.append(X_test_rule[:, i] * X_test_rule[:, j])
    if cross_terms:
        cross_terms = np.column_stack(cross_terms)
        X_test_rule2f = np.hstack([X_test_rule2, cross_terms])
    else:
        X_test_rule2f = X_test_rule2

    for ij, y_id in enumerate(labels):
        y_test_pred_1[:] += (
            y_bucket_mean_1[y_id] + X_test_rule @ corr_terms_1[y_id, :]
        ) * norm_firing_strength[:, ij]
        y_test_pred_2[:] += (
            y_bucket_mean_2[y_id] + X_test_rule2 @ corr_terms_2[y_id, :]
        ) * norm_firing_strength[:, ij]
        y_test_pred_2f[:] += (
            y_bucket_mean_2f[y_id] + X_test_rule2f @ corr_terms_2f[y_id, :]
        ) * norm_firing_strength[:, ij]
        y_test_pred_3[:] += (
            y_bucket_mean_3[y_id] + X_test_rule3 @ corr_terms_3[y_id, :]
        ) * norm_firing_strength[:, ij]

    r2_1, rmse_1 = report_regression_performance(
        start_time, y_test, y_test_pred_1, n_order=f"1 optimized ({model_type})"
    )
    r2_2, rmse_2 = report_regression_performance(
        start_time, y_test, y_test_pred_2, n_order=f"2 optimized ({model_type})"
    )
    r2_2f, rmse_2f = report_regression_performance(
        start_time, y_test, y_test_pred_2f, n_order=f"2-full optimized ({model_type})"
    )
    r2_3, rmse_3 = report_regression_performance(
        start_time, y_test, y_test_pred_3, n_order=f"3 optimized ({model_type})"
    )

    results = {
        'model_type': model_type,
        'r2': [r2_0, r2_1, r2_2, r2_2f, r2_3],
        'rmse': [rmse_0, rmse_1, rmse_2, rmse_2f, rmse_3],
        'predictions': [y_test_pred_0, y_test_pred_1, y_test_pred_2, y_test_pred_2f, y_test_pred_3],
        'labels': ["0 Optimized", "1 Optimized", "2 Optimized", "2-full Optimized", "3 Optimized"],
    }

    plot_tsk_order_comparison(results['r2'], results['rmse'], y_test, results['predictions'], results['labels'])

    return results


def main():
    start_time = time.time()
    X, y_raw = load_data()

    n_output_buckets = 5
    n_top_vars = -1

    if n_top_vars <= 0 or n_top_vars > len(X.columns):
        n_top_vars = len(X.columns)

    y, y_bucket_mean = partition_output(n_output_buckets, y_raw)

    X = log_transform(X, ["Slag", "FlyAsh", "Age"], 1)

    # Split dataset once for fair comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y["y_bucket"]
    )
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}\n")

    # Feature selection
    feature_differentiators = calculate_gaussian_correlation(X_train, y_train["y_bucket"])
    top_n, top_n_todo = take_top_features(feature_differentiators, top_n=n_top_vars)
    print(f"Selected Top-{top_n} Variables ({top_n/len(feature_differentiators):.2%} coverage):\n")

    # Run both models
    gaussian_results = run_model("gaussian", X_train, X_test, y_train, y_test, y_bucket_mean, top_n_todo, n_top_vars, start_time, n_output_buckets)
    trapz_results = run_model("trapz", X_train, X_test, y_train, y_test, y_bucket_mean, top_n_todo, n_top_vars, start_time, n_output_buckets)

    # Summary comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print("\nGAUSSIAN MODEL:")
    for i, label in enumerate(gaussian_results['labels']):
        print(f"  {label:20s}: R² = {gaussian_results['r2'][i]:.4f}, RMSE = {gaussian_results['rmse'][i]:.2f}")

    print("\nTRAPEZOID MODEL:")
    for i, label in enumerate(trapz_results['labels']):
        print(f"  {label:20s}: R² = {trapz_results['r2'][i]:.4f}, RMSE = {trapz_results['rmse'][i]:.2f}")

    print("\nPERFORMANCE DELTA (Trapz - Gaussian):")
    for i, label in enumerate(gaussian_results['labels']):
        r2_delta = trapz_results['r2'][i] - gaussian_results['r2'][i]
        rmse_delta = trapz_results['rmse'][i] - gaussian_results['rmse'][i]
        r2_sign = "+" if r2_delta >= 0 else ""
        rmse_sign = "+" if rmse_delta >= 0 else ""
        print(f"  {label:20s}: ΔR² = {r2_sign}{r2_delta:+.4f}, ΔRMSE = {rmse_sign}{rmse_delta:+.2f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
