import time
import typing

import numpy as np
import pandas as pd
from itertools import combinations
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.optimize import minimize

from tribblefis.gauss_data import GaussianMixtureModel
from tribblefis.gauss_math import tsk_firing_strengths


def plot_tsk_order_comparison(
    r2: list[float],
    rmse: list[float],
    y_test: pd.DataFrame,
    y_test_pred: list[ndarray],
    order_names: list[str] | None = None,
):
    # Plot comparison of actual vs predicted values
    fig, axes = plt.subplots(int(np.ceil(len(r2) / 2)), 2, figsize=(8, 3 * len(r2)))

    # Calculate the min and max values from the true output
    y_min = y_test["y_value"].min()
    y_max = y_test["y_value"].max()

    # Zeroth-order TSK model
    for ij, y_test_pred_vec in enumerate(y_test_pred):
        order_name = order_names[ij] if order_names is not None else f"{ij}-Order"
        axes[ij // 2, ij % 2].scatter(y_test["y_value"], y_test_pred_vec, alpha=0.5, edgecolors="k")
        axes[ij // 2, ij % 2].plot(
            [y_test["y_value"].min(), y_test["y_value"].max()],
            [y_test["y_value"].min(), y_test["y_value"].max()],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )
        axes[ij // 2, ij % 2].set_xlabel("Actual", fontsize=12)
        axes[ij // 2, ij % 2].set_ylabel("Predicted", fontsize=12)
        axes[ij // 2, ij % 2].set_title(
            f"{order_name} TSK Model\nActual vs Predicted\nRMSE={rmse[ij]:.4f}, R²={r2[ij]:.4f}", fontsize=14
        )
        axes[ij // 2, ij % 2].legend()
        axes[ij // 2, ij % 2].grid(True, alpha=0.3)
        axes[ij // 2, ij % 2].set_xlim(y_min, y_max)
        axes[ij // 2, ij % 2].set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()


def _rsquared(y_t: pd.Series | np.ndarray, y_p: pd.Series | np.ndarray) -> float:
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    return 1 - ss_res / ss_tot


def _mse(y_t: pd.Series | np.ndarray, y_p: pd.Series | np.ndarray) -> float:
    return float(np.mean((y_t - y_p) ** 2))


def _mae(y_t: pd.Series | np.ndarray, y_p: pd.Series | np.ndarray) -> float:
    return float(np.mean(np.abs(y_t - y_p)))


def partition_output(
    n_output_buckets: int, y_raw: pd.Series | pd.DataFrame | typing.Any
) -> tuple[pd.DataFrame, typing.Any]:
    # Partition y into n_output_buckets, but ensure one bucket is essentially at each end of the range.
    y_part = pd.cut(y_raw, bins=n_output_buckets, labels=False, include_lowest=True)
    # y_part = pd.qcut(y_raw, q=n_output_buckets, labels=False)
    y_part.name = "y_bucket"
    # Build a full-length array indexed by bucket label (0..n_output_buckets-1).
    # groupby silently drops empty buckets, so reconstruct with correct label alignment
    # and fill any gaps via linear interpolation so downstream indexing by rule_id is safe.
    grouped = y_raw.groupby(y_part).mean()
    y_bucket_mean = np.full(n_output_buckets, np.nan)
    for label, val in grouped.items():
        y_bucket_mean[int(label)] = val
    y_bucket_mean = pd.Series(y_bucket_mean).interpolate(method='linear', limit_direction='both').values.copy()
    # For the extreme endpoint buckets, use the min and max
    y_bucket_mean[0] = float(y_raw.min())
    y_bucket_mean[-1] = float(y_raw.max())

    y = pd.concat([y_part, y_raw], axis=1)
    return y, y_bucket_mean


def report_regression_performance(
    start_time: float, y_test, y_test_pred_first_order: ndarray, n_order: int | str
) -> tuple[float, float]:
    # HACK: Remove the handful of nan outputs
    keep_rows = ~np.isnan(y_test_pred_first_order)
    y_test = y_test.loc[keep_rows]
    y_test_pred_first_order = y_test_pred_first_order[keep_rows]
    # Print accuracy metrics (e.g., RMSE, MAE)
    rmse = np.sqrt(_mse(y_test["y_value"], y_test_pred_first_order))
    mae = _mae(y_test["y_value"], y_test_pred_first_order)
    r2 = _rsquared(y_test["y_value"], y_test_pred_first_order)

    print(f"{n_order}-Order TSK Model Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    return r2, rmse


def compute_first_order_corrections(
    X_train: pd.DataFrame,
    gaussian_memberships: GaussianMixtureModel,
    n_top_vars: int,
    top_n_todo: list[str],
    y_bucket_mean,
    y_train,
) -> ndarray:
    n_slots = max(gaussian_memberships.rule_ids) + 1
    corr_terms = np.zeros(shape=(n_slots, n_top_vars))
    # Loop through each rule
    for rule_id in gaussian_memberships.rule_ids:
        # Find the X_train, y_train which belong to this output bucket
        rule_mask = y_train["y_bucket"] == rule_id
        X_train_rule = X_train[rule_mask][top_n_todo].to_numpy()
        y_train_rule_err = y_train[rule_mask]["y_value"] - y_bucket_mean[rule_id]
        A = X_train_rule
        b = y_train_rule_err
        # A \ b
        corr_terms[rule_id, :] = np.linalg.pinv(A) @ b
    return corr_terms


def compute_second_order_corrections(
    X_train: pd.DataFrame,
    gaussian_memberships: GaussianMixtureModel,
    n_top_vars: int,
    top_n_todo: list[str],
    y_bucket_mean,
    y_train,
) -> ndarray:
    n_slots = max(gaussian_memberships.rule_ids) + 1
    corr_terms = np.zeros(shape=(n_slots, n_top_vars * 2))
    # Loop through each rule
    for rule_id in gaussian_memberships.rule_ids:
        # Find the X_train, y_train which belong to this output bucket
        rule_mask = y_train["y_bucket"] == rule_id
        X_train_rule = X_train[rule_mask][top_n_todo].to_numpy()
        # Augment with the second-order terms, excluding cross-power terms
        X_train_rule_squared = X_train_rule**2
        A = np.hstack([X_train_rule, X_train_rule_squared])
        y_train_rule_err = y_train[rule_mask]["y_value"] - y_bucket_mean[rule_id]
        b = y_train_rule_err
        # A \ b
        corr_terms[rule_id, :] = np.linalg.pinv(A) @ b
    return corr_terms


def compute_third_order_corrections(
    X_train: pd.DataFrame,
    gaussian_memberships: GaussianMixtureModel,
    n_top_vars: int,
    top_n_todo: list[str],
    y_bucket_mean,
    y_train,
) -> ndarray:
    n_slots = max(gaussian_memberships.rule_ids) + 1
    corr_terms = np.zeros(shape=(n_slots, n_top_vars * 3))
    # Loop through each rule
    for rule_id in gaussian_memberships.rule_ids:
        # Find the X_train, y_train which belong to this output bucket
        rule_mask = y_train["y_bucket"] == rule_id
        X_train_rule = X_train[rule_mask][top_n_todo].to_numpy()
        # Augment with the second-order terms, excluding cross-power terms
        A = np.hstack([X_train_rule, X_train_rule**2, X_train_rule**3])
        y_train_rule_err = y_train[rule_mask]["y_value"] - y_bucket_mean[rule_id]
        b = y_train_rule_err
        # A \ b
        corr_terms[rule_id, :] = np.linalg.pinv(A) @ b
    return corr_terms


def compute_full_second_order_corrections(
    X_train: pd.DataFrame,
    gaussian_memberships: GaussianMixtureModel,
    n_top_vars: int,
    top_n_todo: list[str],
    y_bucket_mean,
    y_train,
) -> ndarray:
    # Calculate number of cross-power terms
    n_cross_terms = len(list(combinations(range(n_top_vars), 2)))
    total_terms = n_top_vars + n_top_vars + n_cross_terms  # linear + squared + cross
    n_slots = max(gaussian_memberships.rule_ids) + 1
    corr_terms = np.zeros(shape=(n_slots, total_terms))
    # Loop through each rule
    for rule_id in gaussian_memberships.rule_ids:
        # Find the X_train, y_train which belong to this output bucket
        rule_mask = y_train["y_bucket"] == rule_id
        X_train_rule = X_train[rule_mask][top_n_todo].to_numpy()
        # Augment with the second-order terms, including cross-power terms
        X_train_rule_squared = X_train_rule**2
        # Compute cross-power terms
        cross_terms = []
        for i, j in combinations(range(n_top_vars), 2):
            cross_terms.append(X_train_rule[:, i] * X_train_rule[:, j])
        X_train_rule_cross = np.column_stack(cross_terms) if cross_terms else np.empty((X_train_rule.shape[0], 0))

        A = np.hstack([X_train_rule, X_train_rule_squared, X_train_rule_cross])
        y_train_rule_err = y_train[rule_mask]["y_value"] - y_bucket_mean[rule_id]
        b = y_train_rule_err
        # A \ b
        corr_terms[rule_id, :] = np.linalg.lstsq(A, b, rcond=None)[0]
    return corr_terms


def optimize_tsk_coefficients(
    X_train: pd.DataFrame,
    gaussian_memberships: GaussianMixtureModel,
    top_n_todo: list[typing.Any],
    y_bucket_mean: pd.Series,
    y_train: pd.Series,
    n_output_buckets: int,
    initial_corr_terms: ndarray | None = None,
    order: typing.Literal["0th", "1st", "2nd", "3rd", "full-2nd"] = "2nd",
) -> tuple[typing.Any, typing.Any]:
    """
    Optimize TSK coefficients for different polynomial orders.

    Args:
        order: One of '0th', '1st', '2nd', '3rd', or 'full-2nd'
            - '0th': Only constant terms
            - '1st': Constant + linear terms
            - '2nd': Constant + linear + squared terms (no cross-products)
            - 'full-2nd': Constant + linear + squared + cross-product terms
            - '3rd': Constant + linear + sqquared + cubic terms (no cross-products)
    """
    # Optimize coefficients based on order
    print(f"\nOptimizing {order.capitalize()} Order TSK Coefficients...")
    print("=" * 80)

    # Compute training firing strengths
    firing_strengths_train, labels_train = tsk_firing_strengths(X_train[top_n_todo], gaussian_memberships)
    norm_firing_strength_train = firing_strengths_train / firing_strengths_train.sum(axis=1)[:, np.newaxis]
    X_train_rule = X_train[top_n_todo].to_numpy()

    n_top_vars = len(top_n_todo)

    # Build feature matrix based on order
    if order == "0th":
        X_train_features = None
        n_terms = 0
    elif order == "1st":
        X_train_features = X_train_rule
        n_terms = n_top_vars
    elif order == "2nd":
        X_train_features = np.hstack([X_train_rule, X_train_rule**2])
        n_terms = 2 * n_top_vars
    elif order == "3rd":
        X_train_features = np.hstack([X_train_rule, X_train_rule**2, X_train_rule**3])
        n_terms = 3 * n_top_vars
    elif order == "full-2nd":
        # Add cross-product terms
        cross_terms = []
        for i, j in combinations(range(n_top_vars), 2):
            cross_terms.append(X_train_rule[:, i] * X_train_rule[:, j])
        X_train_cross = np.column_stack(cross_terms) if cross_terms else np.empty((X_train_rule.shape[0], 0))
        X_train_features = np.hstack([X_train_rule, X_train_rule**2, X_train_cross])
        n_cross_terms = len(cross_terms)
        n_terms = n_top_vars + n_top_vars + n_cross_terms
    else:
        raise ValueError(f"Unknown order: {order}")

    # Flatten initial coefficients
    if order == "0th":
        initial_coeffs_flat = y_bucket_mean.copy()
    else:
        assert initial_corr_terms is not None
        initial_coeffs = np.column_stack([y_bucket_mean, initial_corr_terms])
        initial_coeffs_flat = initial_coeffs.flatten()

    def predict(coeffs_flat, X_data, norm_fs, rule_labels, n_rules, n_feat_terms):
        """Compute predictions from flattened coefficients."""
        if order == "0th":
            # Only constant terms
            y_pred = np.dot(norm_fs, coeffs_flat[rule_labels])
        else:
            coeffs = coeffs_flat.reshape(n_rules, n_feat_terms + 1)
            y_pred = np.zeros(X_data.shape[0])
            for ij, y_id in enumerate(rule_labels):
                y_pred[:] += (coeffs[y_id, 0] + X_data @ coeffs[y_id, 1:]) * norm_fs[:, ij]
        return y_pred

    def objective(coeffs_flat):
        """RMSE objective function."""
        y_pred = predict(
            coeffs_flat, X_train_features, norm_firing_strength_train, labels_train, n_output_buckets, n_terms
        )
        return _mse(y_train["y_value"].values, y_pred)

    # Optimize using L-BFGS-B
    result = minimize(objective, initial_coeffs_flat, method="L-BFGS-B", options={"maxiter": 1000})

    # Extract optimized coefficients
    if order == "0th":
        y_bucket_mean_opt = result.x
        corr_terms_opt = np.zeros((n_output_buckets, 0))
    else:
        optimized_coeffs = result.x.reshape(n_output_buckets, n_terms + 1)
        y_bucket_mean_opt = optimized_coeffs[:, 0]
        corr_terms_opt = optimized_coeffs[:, 1:]

    print(f"Optimization completed. Final RMSE on training set: {result.fun:.4f}")
    return corr_terms_opt, y_bucket_mean_opt
