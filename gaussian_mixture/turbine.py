import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    create_gaussian_membership_dict,
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


def load_all_data(folder: str = "train") -> tuple[pd.DataFrame, pd.Series]:
    # Determine the base path (script directory or current working directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path_base = os.path.join(script_dir, "gas_turbine", folder)

    if not os.path.exists(train_path_base):
        # Try relative to current working directory
        train_path_base = os.path.join("gas_turbine", folder)

    if not os.path.isdir(train_path_base):
        raise FileNotFoundError(f"Train directory not found at {train_path_base}")

    all_files = sorted([f for f in os.listdir(train_path_base) if f.endswith(".csv")])
    if not all_files:
        raise ValueError(f"No CSV files found in train directory: {train_path_base}")

    X_dataframes = []
    y_series = []
    for filename in all_files:
        filepath = os.path.join(train_path_base, filename)
        X, y = load_data(filepath)
        X_dataframes.append(X)
        y_series.append(y)

    X = pd.concat(X_dataframes, ignore_index=True, axis=0)
    y: pd.Series = pd.concat(y_series, ignore_index=True)
    return X, y


def load_data(data_path: str = "gas_turbine/test/ex_4.csv"):
    if not os.path.exists(data_path):
        # Try to find it in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_path)

    X = pd.read_csv(data_path)
    X = X.dropna()
    y = X["el_power"]
    y.name = "y_value"
    X.drop(["el_power"], axis=1, inplace=True)
    X["time"] -= X["time"][0]
    X = X.select_dtypes(include=[np.number])
    # Create Fibonacci-delayed features
    X = create_fibonacci_lagged_features(X)
    return X, y


def create_fibonacci_lagged_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create lagged features using Fibonacci sequence delays."""
    fibonacci_delays = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # fibonacci_delays = [1, 100, 1000]
    df_list = [X.copy()]

    for delay in fibonacci_delays:
        lagged_df = X.shift(delay, fill_value=X.iloc[0, 0])
        lagged_df.drop(["time"], axis=1, inplace=True)
        lagged_df.columns = [f"{col}_lag_{delay}" for col in lagged_df.columns]
        df_list.append(lagged_df)

    # Concatenate all features
    result = pd.concat(df_list, axis=1)
    return result


def main():
    start_time = time.time()
    X, y = load_all_data("train")

    n_output_buckets: int = 5
    n_top_vars: int = -1
    n_gaussians: int = 1
    b_optimize_coeff: bool = False

    if n_top_vars <= 0 or n_top_vars > len(X.columns):
        n_top_vars = len(X.columns)

    y, y_bucket_mean = partition_output(n_output_buckets, y)

    # Split dataset into train/test
    X_train = X
    y_train = y
    # HACK - Test data is separate!
    X_test, y_test_raw = load_all_data("test")
    y_test, y_bucket_mean = partition_output(n_output_buckets, y_test_raw)
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

    fig, ax1 = plt.subplots()
    y_plot = pd.DataFrame()
    y_plot["y_test"] = y_test_raw
    y_plot["y_test_pred_zeroth_opt"] = y_test_pred_zeroth_opt
    y_plot["y_test_pred_first_order_opt"] = y_test_pred_first_order_opt
    y_plot["y_test_pred_second_order_opt"] = y_test_pred_second_order_opt
    ax1.plot(X_test["time"], y_plot, label=y_plot.columns)
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot(X_test["time"], X_test["input_voltage"], label="test_input_voltage")
    ax2.set_ylabel("Raw Input")
    ax2.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
