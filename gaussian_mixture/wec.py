import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    create_gaussian_membership_dict,
    take_top_features,
    tsk_firing_strengths,
)
from tribblefis.report import print_membership_details
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


def load_data():
    data_path = "WEC_Perth_49.csv"
    if not os.path.exists(data_path):
        # Try to find it in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_path)

    X = pd.read_csv(data_path)
    X = X.dropna()
    y = X["Total_Power"]
    y.name = "y_value"
    X.drop(columns=[col for col in X.columns if "Power" in col], inplace=True)
    X.drop("qW", axis=1, inplace=True)
    X = X.select_dtypes(include=[np.number])
    return X, y


def main():
    start_time = time.time()
    X, y_raw = load_data()

    n_output_buckets: int = 3
    n_top_vars: int = -1
    n_gaussians: int = 2
    b_optimize_coeff: bool = False

    if n_top_vars <= 0 or n_top_vars > len(X.columns):
        n_top_vars = len(X.columns)

    y, y_bucket_mean = partition_output(n_output_buckets, y_raw)

    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y["y_bucket"])
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    # Calculate correlation coefficient between Gaussian distributions using training data
    feature_differentiators = calculate_gaussian_correlation(X_train, y_train["y_bucket"])

    # Take the top-n variables so that the normalized differentiation value encompasses 90-95%
    top_n, top_n_todo = take_top_features(feature_differentiators, top_n=n_top_vars)

    print(f"Selected Top-{top_n} Variables ({top_n/len(feature_differentiators):.2%} coverage):")

    # Compute memberships using training data
    gaussian_memberships = create_gaussian_membership_dict(
        X_train, y_train["y_bucket"], top_n_var_names=top_n_todo, n_gaussians=n_gaussians
    )

    duplicates = gaussian_memberships.identify_duplicate_membership_fcns()
    print(
        f"\nDuplicate Membership Functions({len(duplicates)/gaussian_memberships.n_membership_functions:.1%} redundant):\n"
        f"{duplicates}"
    )

    print_membership_details(gaussian_memberships)

    print("\nEvaluating Zeroth-Order TSK Model on TEST set:")
    print("=" * 80)

    # Allocate the corrections terms (1st-order TSK)
    corr_terms_1 = compute_first_order_corrections(
        X_train, gaussian_memberships, n_top_vars, top_n_todo, y_bucket_mean, y_train
    )
    corr_terms_2 = compute_second_order_corrections(
        X_train, gaussian_memberships, n_top_vars, top_n_todo, y_bucket_mean, y_train
    )
    corr_terms_3 = compute_third_order_corrections(
        X_train, gaussian_memberships, n_top_vars, top_n_todo, y_bucket_mean, y_train
    )
    corr_terms_2f = compute_full_second_order_corrections(
        X_train, gaussian_memberships, n_top_vars, top_n_todo, y_bucket_mean, y_train
    )

    if b_optimize_coeff:
        _, y_bucket_mean_opt_0 = optimize_tsk_coefficients(
            X_train,
            gaussian_memberships,
            top_n_todo,
            y_bucket_mean,
            y_train,
            n_output_buckets=n_output_buckets,
            order="0th",
        )
        corr_terms_1_opt, y_bucket_mean_opt_1 = optimize_tsk_coefficients(
            X_train,
            gaussian_memberships,
            top_n_todo,
            y_bucket_mean,
            y_train,
            n_output_buckets=n_output_buckets,
            initial_corr_terms=corr_terms_1,
            order="1st",
        )
        corr_terms_2_opt, y_bucket_mean_opt_2 = optimize_tsk_coefficients(
            X_train,
            gaussian_memberships,
            top_n_todo,
            y_bucket_mean,
            y_train,
            n_output_buckets=n_output_buckets,
            initial_corr_terms=corr_terms_2,
            order="2nd",
        )
    else:
        y_bucket_mean_opt_0 = y_bucket_mean
        corr_terms_1_opt = corr_terms_1
        y_bucket_mean_opt_1 = y_bucket_mean
        corr_terms_2_opt = corr_terms_2
        y_bucket_mean_opt_2 = y_bucket_mean
        # TODO - Handle 3rd-order optimization?

    # Now, we need to evaluate the model
    print("\nEvaluating Multi-Order TSK Model on TEST set:")
    print("=" * 80)
    firing_strengths, labels = tsk_firing_strengths(X_test[top_n_todo], gaussian_memberships)
    norm_firing_strength = firing_strengths / firing_strengths.sum(axis=1)[:, np.newaxis]

    # Evaluate optimized order-0 model
    y_test_pred_zeroth_opt = np.dot(norm_firing_strength, y_bucket_mean_opt_0)

    # Calculate correction term for each sample
    y_test_pred_first_order_opt = np.zeros(len(X_test))
    y_test_pred_second_order_opt = np.zeros(len(X_test))
    y_test_pred_second_order_full = np.zeros(len(X_test))
    y_test_pred_third_order = np.zeros(len(X_test))
    for ij, y_id in enumerate(labels):
        X_test_rule = X_test[top_n_todo].to_numpy()
        X_test_rule2 = np.hstack([X_test_rule, X_test_rule**2])
        X_test_rule3 = np.hstack([X_test_rule, X_test_rule**2, X_test_rule**3])
        # Augment with the second-order terms, including cross-power terms
        cross_terms = []
        for i in range(X_test_rule.shape[1]):
            for j in range(i + 1, X_test_rule.shape[1]):
                cross_terms.append(X_test_rule[:, i] * X_test_rule[:, j])
        if cross_terms:
            cross_terms = np.column_stack(cross_terms)
            X_test_rule2f = np.hstack([X_test_rule2, cross_terms])
        else:
            X_test_rule2f = X_test_rule2

        y_test_pred_first_order_opt[:] += (
            y_bucket_mean_opt_1[y_id] + X_test_rule @ corr_terms_1_opt[y_id, :]
        ) * norm_firing_strength[:, ij]
        y_test_pred_second_order_opt[:] += (
            y_bucket_mean_opt_2[y_id] + X_test_rule2 @ corr_terms_2_opt[y_id, :]
        ) * norm_firing_strength[:, ij]
        y_test_pred_third_order[:] += (
            y_bucket_mean[y_id] + X_test_rule3 @ corr_terms_3[y_id, :]
        ) * norm_firing_strength[:, ij]
        y_test_pred_second_order_full[:] += (
            y_bucket_mean[y_id] + X_test_rule2f @ corr_terms_2f[y_id, :]
        ) * norm_firing_strength[:, ij]

    r2_0opt, rmse_0opt = report_regression_performance(
        start_time, y_test, y_test_pred_zeroth_opt, n_order="0 optimized"
    )
    r2_1opt, rmse_1opt = report_regression_performance(
        start_time, y_test, y_test_pred_first_order_opt, n_order="1 optimized"
    )
    r2_2opt, rmse_2opt = report_regression_performance(
        start_time, y_test, y_test_pred_second_order_opt, n_order="2 optimized"
    )
    plot_tsk_order_comparison(
        [r2_0opt, r2_1opt, r2_2opt],
        [rmse_0opt, rmse_1opt, rmse_2opt],
        y_test,
        [
            y_test_pred_zeroth_opt,
            y_test_pred_first_order_opt,
            y_test_pred_second_order_opt,
        ],
        [
            "0 Optimized",
            "1 Optimized",
            "2 Optimized",
        ],
    )


if __name__ == "__main__":
    main()
