import time
import typing
import warnings

import numpy as np
import pandas as pd
from itertools import combinations
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.optimize import minimize

from tribblefis.gauss_data import GaussianMixtureModel
from tribblefis.gauss_data import NormPair
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


# A bucket below this holds too little to fit a rule from, and the aggregate error will
# not show it. Three is the smallest count that admits any spread at all.
_STARVED_BUCKET = 3


def _index_of(y_raw):
    """The index to carry onto a rebuilt label series, so concat aligns."""
    return getattr(y_raw, "index", pd.RangeIndex(len(y_raw)))


def _warn_starved_buckets(y_part, n_output_buckets, method) -> None:
    """Warn when a bucket is too thin to fit a rule from.

    Under equal-width boundaries a skewed target starves its extreme buckets, and that is
    the one failure mode uniform partitioning has. It is invisible to aggregate error --
    a starved bucket barely moves an average -- so it is worth a warning rather than a
    silent rule fitted on two points. The remedy is named in the message because it is a
    call-site decision: transform the target, do not switch the partition.
    """
    counts = pd.Series(y_part).value_counts()
    thin = {int(k): int(v) for k, v in counts.items() if v < _STARVED_BUCKET}
    missing = [b for b in range(n_output_buckets) if b not in set(counts.index)]
    if not thin and not missing:
        return
    parts = []
    if missing:
        parts.append(f"empty: {missing}")
    if thin:
        parts.append("under-filled: "
                     + ", ".join(f"{k} holds {v}" for k, v in sorted(thin.items())))
    warnings.warn(
        f"partition_output(method={method!r}) left {'; '.join(parts)} out of "
        f"{n_output_buckets} buckets. Those rules are fitted on almost no data and the "
        f"aggregate error will not show it. If the target range is discontinuous or "
        f"badly non-uniform, apply a monotone transform (log, Box-Cox, rank) before "
        f"fitting and invert for reporting; a monotone map preserves bucket order, so "
        f"nothing downstream changes.",
        RuntimeWarning,
        stacklevel=3,
    )


def partition_output(
    n_output_buckets: int,
    y_raw: pd.Series | pd.DataFrame | typing.Any,
    method: str = "uniform",
    pin_extremes: bool | None = None,
) -> tuple[pd.DataFrame, typing.Any]:
    """Cut the target into buckets and return (frame with bucket labels, bucket centroids).

    ``method="uniform"`` -- equal-WIDTH boundaries, the default. ``"quantile"`` --
    equal-FREQUENCY boundaries, which is what this function used to do unconditionally.

    Why uniform is the default
    --------------------------
    Equal-frequency boundaries guarantee every bucket is occupied, which sounds like the
    safe choice and is not. Under target skew quantile does not become less accurate, it
    becomes UNSTABLE: a sweep over increasing skew measured its seed-to-seed deviation
    growing to +/-0.99, +/-4.45 and +/-21.2 while uniform's mean decayed smoothly toward
    zero with a bounded spread. A few catastrophic splits drag the quantile mean; uniform
    fails predictably instead. For a component inside a larger pipeline the predictable
    failure is the one to prefer, because it can be detected and bounded.

    Uniform has its own failure mode and it is the reverse: on a skewed target the extreme
    buckets STARVE, since equal-width bins in a sparse tail may catch almost nothing. That
    is bounded, visible, and fixable at the call site -- which is the point. The fix is to
    transform the target before fitting, not to change the partition: if the output range
    is discontinuous or badly non-uniform, apply a monotone transform (log, Box-Cox, rank)
    first, fit in the transformed space, and invert for reporting. A monotone transform
    leaves the ordering intact, so nothing downstream that reasons about bucket order
    changes.

    Starvation is warned about rather than left silent. A bucket holding fewer than
    ``_STARVED_BUCKET`` samples means a rule fitted on almost no data, and the aggregate
    error barely moves when that happens -- exactly the failure an accuracy-only check
    cannot see.

    pin_extremes
    ------------
    ``None`` (the default) pins the two extreme centroids to the observed min and max
    ONLY for ``method="quantile"``. That keeps each method's default identical to the arm
    that was actually measured: uniform with data means, quantile with data means, and
    quantile with pinned extremes, which is the hybrid this function used to ship. Pass
    True or False to override.

    Pinning exists to compensate for quantile's wide sparse-tail buckets, whose means
    badly under-resolve the extremes. Equal-width buckets are narrow by construction, so
    their means already sit near the ends of the range, and pinning them to a single most
    extreme observation would only add outlier sensitivity.
    """
    if method not in ("uniform", "quantile"):
        raise ValueError(
            f"method must be 'uniform' or 'quantile', got {method!r}")
    if pin_extremes is None:
        pin_extremes = method == "quantile"

    if method == "uniform":
        # Equal-width edges over the observed range. The left edge is nudged below the
        # minimum so the smallest value lands in bucket 0 rather than becoming NaN --
        # `include_lowest` alone is not enough once the edges are supplied explicitly.
        lo, hi = float(np.min(y_raw)), float(np.max(y_raw))
        if hi <= lo:                     # degenerate target: one bucket holds everything
            y_part = pd.Series(np.zeros(len(y_raw), dtype=int), index=_index_of(y_raw))
        else:
            edges = np.linspace(lo, hi, n_output_buckets + 1)
            edges[0] -= 1e-9
            y_part = pd.cut(y_raw, bins=edges, labels=False, include_lowest=True)
            y_part = pd.Series(np.asarray(y_part, dtype=float),
                               index=_index_of(y_raw)).fillna(0).astype(int)
    else:
        y_part = pd.qcut(y_raw, q=n_output_buckets, labels=False)
    y_part.name = "y_bucket"

    _warn_starved_buckets(y_part, n_output_buckets, method)
    # Build a full-length array indexed by bucket label (0..n_output_buckets-1).
    # groupby silently drops empty buckets, so reconstruct with correct label alignment
    # and fill any gaps via linear interpolation so downstream indexing by rule_id is safe.
    grouped = y_raw.groupby(y_part).mean()
    y_bucket_mean = np.full(n_output_buckets, np.nan)
    for label, val in grouped.items():
        y_bucket_mean[int(label)] = val
    y_bucket_mean = pd.Series(y_bucket_mean).interpolate(method='linear', limit_direction='both').values.copy()
    if pin_extremes:
        # For the extreme endpoint buckets, use the min and max. See the docstring: this
        # compensates for quantile's wide sparse-tail buckets and is off under uniform.
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
    """Compute first-order (linear) corrections per rule.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    gaussian_memberships : GaussianMixtureModel
        Membership model with rule_ids.
    n_top_vars : int
        Number of top variables.
    top_n_todo : list[str]
        Feature column names.
    y_bucket_mean : array-like
        Per-rule output means.
    y_train : pd.DataFrame
        Training target with 'y_bucket' and 'y_value' columns.

    Returns
    -------
    ndarray
        Correction terms shape (n_slots, n_top_vars).
    """
    n_slots = max(gaussian_memberships.rule_ids) + 1
    corr_terms = np.zeros(shape=(n_slots, n_top_vars))
    for rule_id in gaussian_memberships.rule_ids:
        rule_mask = y_train["y_bucket"] == rule_id
        X_train_rule = X_train[rule_mask][top_n_todo].to_numpy()
        y_train_rule_err = y_train[rule_mask]["y_value"] - y_bucket_mean[rule_id]
        corr_terms[rule_id, :] = np.linalg.pinv(X_train_rule) @ y_train_rule_err
    return corr_terms


def compute_second_order_corrections(
    X_train: pd.DataFrame,
    gaussian_memberships: GaussianMixtureModel,
    n_top_vars: int,
    top_n_todo: list[str],
    y_bucket_mean,
    y_train,
) -> ndarray:
    """Compute second-order (quadratic) corrections per rule: [X, X²].

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    gaussian_memberships : GaussianMixtureModel
        Membership model with rule_ids.
    n_top_vars : int
        Number of top variables.
    top_n_todo : list[str]
        Feature column names.
    y_bucket_mean : array-like
        Per-rule output means.
    y_train : pd.DataFrame
        Training target with 'y_bucket' and 'y_value' columns.

    Returns
    -------
    ndarray
        Correction terms shape (n_slots, n_top_vars * 2).
    """
    n_slots = max(gaussian_memberships.rule_ids) + 1
    corr_terms = np.zeros(shape=(n_slots, n_top_vars * 2))
    for rule_id in gaussian_memberships.rule_ids:
        rule_mask = y_train["y_bucket"] == rule_id
        X_train_rule = X_train[rule_mask][top_n_todo].to_numpy()
        A = np.hstack([X_train_rule, X_train_rule**2])
        y_train_rule_err = y_train[rule_mask]["y_value"] - y_bucket_mean[rule_id]
        corr_terms[rule_id, :] = np.linalg.pinv(A) @ y_train_rule_err
    return corr_terms


def compute_third_order_corrections(
    X_train: pd.DataFrame,
    gaussian_memberships: GaussianMixtureModel,
    n_top_vars: int,
    top_n_todo: list[str],
    y_bucket_mean,
    y_train,
) -> ndarray:
    """Compute third-order (cubic) corrections per rule: [X, X², X³].

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    gaussian_memberships : GaussianMixtureModel
        Membership model with rule_ids.
    n_top_vars : int
        Number of top variables.
    top_n_todo : list[str]
        Feature column names.
    y_bucket_mean : array-like
        Per-rule output means.
    y_train : pd.DataFrame
        Training target with 'y_bucket' and 'y_value' columns.

    Returns
    -------
    ndarray
        Correction terms shape (n_slots, n_top_vars * 3).
    """
    n_slots = max(gaussian_memberships.rule_ids) + 1
    corr_terms = np.zeros(shape=(n_slots, n_top_vars * 3))
    for rule_id in gaussian_memberships.rule_ids:
        rule_mask = y_train["y_bucket"] == rule_id
        X_train_rule = X_train[rule_mask][top_n_todo].to_numpy()
        A = np.hstack([X_train_rule, X_train_rule**2, X_train_rule**3])
        y_train_rule_err = y_train[rule_mask]["y_value"] - y_bucket_mean[rule_id]
        corr_terms[rule_id, :] = np.linalg.pinv(A) @ y_train_rule_err
    return corr_terms


def compute_full_second_order_corrections(
    X_train: pd.DataFrame,
    gaussian_memberships: GaussianMixtureModel,
    n_top_vars: int,
    top_n_todo: list[str],
    y_bucket_mean,
    y_train,
) -> ndarray:
    """Compute full second-order corrections: [X, X², cross-products].

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    gaussian_memberships : GaussianMixtureModel
        Membership model with rule_ids.
    n_top_vars : int
        Number of top variables.
    top_n_todo : list[str]
        Feature column names.
    y_bucket_mean : array-like
        Per-rule output means.
    y_train : pd.DataFrame
        Training target with 'y_bucket' and 'y_value' columns.

    Returns
    -------
    ndarray
        Correction terms shape (n_slots, n_terms) where n_terms includes cross-products.
    """
    n_cross_terms = len(list(combinations(range(n_top_vars), 2)))
    total_terms = n_top_vars + n_top_vars + n_cross_terms
    n_slots = max(gaussian_memberships.rule_ids) + 1
    corr_terms = np.zeros(shape=(n_slots, total_terms))
    for rule_id in gaussian_memberships.rule_ids:
        rule_mask = y_train["y_bucket"] == rule_id
        X_train_rule = X_train[rule_mask][top_n_todo].to_numpy()
        X_train_rule_squared = X_train_rule**2
        cross_terms = [X_train_rule[:, i] * X_train_rule[:, j]
                       for i, j in combinations(range(n_top_vars), 2)]
        X_train_rule_cross = (np.column_stack(cross_terms)
                              if cross_terms else np.empty((X_train_rule.shape[0], 0)))
        A = np.hstack([X_train_rule, X_train_rule_squared, X_train_rule_cross])
        y_train_rule_err = y_train[rule_mask]["y_value"] - y_bucket_mean[rule_id]
        corr_terms[rule_id, :] = np.linalg.lstsq(A, y_train_rule_err, rcond=None)[0]
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
    norms: NormPair | None = None,
    l2_reg: float = 1e-6,
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
        l2_reg: Ridge penalty on the correction (non-constant) coefficients. Guards
            against ill-conditioned high-order fits (e.g. cubic consequents on
            standardized features) whose least-squares initial guess can have
            coefficients of magnitude ~1e4 that overfit badly. The constant
            bucket-mean terms are never penalized. Set to 0 to disable.
    """
    # Optimize coefficients based on order
    print(f"\nOptimizing {order.capitalize()} Order TSK Coefficients...")
    print("=" * 80)

    # Compute training firing strengths
    firing_strengths_train, labels_train = tsk_firing_strengths(
        X_train[top_n_todo], gaussian_memberships, norms=norms
    )
    # Create mask for rows where sum > 1e-6
    row_sums = firing_strengths_train.sum(axis=1)
    valid_rows = row_sums > 1e-6

    # Initialize with zeros
    norm_firing_strength_train = np.zeros_like(firing_strengths_train)

    # Only normalize valid rows
    norm_firing_strength_train[valid_rows] = (
        firing_strengths_train[valid_rows] / row_sums[valid_rows, np.newaxis]
    )
    # Rows with no firing (common for bounded-support trapezoids) must match the
    # evaluation convention of a uniform blend; otherwise the optimizer sees those
    # rows as identically zero, leaving a null space that L-BFGS-B can wander into
    # and blow up the coefficients (near-zero training-error change, huge test error).
    n_labels = firing_strengths_train.shape[1]
    if np.any(~valid_rows) and n_labels > 0:
        norm_firing_strength_train[~valid_rows] = 1.0 / n_labels
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

    def _correction_penalty(coeffs_flat):
        """Ridge penalty on correction coefficients only (constants excluded)."""
        if l2_reg <= 0 or order == "0th":
            return 0.0
        coeffs = coeffs_flat.reshape(n_output_buckets, n_terms + 1)
        return l2_reg * float(np.sum(coeffs[:, 1:] ** 2))

    def objective(coeffs_flat):
        """Mean squared error plus a ridge penalty on the correction terms."""
        y_pred = predict(
            coeffs_flat, X_train_features, norm_firing_strength_train, labels_train, n_output_buckets, n_terms
        )
        return _mse(y_train["y_value"].values, y_pred) + _correction_penalty(coeffs_flat)

    # Optimize using L-BFGS-B
    initial_obj = objective(initial_coeffs_flat)
    result = minimize(objective, initial_coeffs_flat, method="L-BFGS-B", options={"maxiter": 1000})

    # Ill-conditioned problems (e.g. high-order trapezoid rules with sparse firing
    # strengths) can leave L-BFGS-B at a point worse than where it started. Never
    # return coefficients worse than the initial least-squares guess.
    if result.fun <= initial_obj:
        best_flat, best_obj = result.x, result.fun
    else:
        best_flat, best_obj = initial_coeffs_flat, initial_obj
        print("  Optimizer did not improve on the initial guess; keeping initial coefficients.")

    # Extract optimized coefficients
    if order == "0th":
        y_bucket_mean_opt = best_flat
        corr_terms_opt = np.zeros((n_output_buckets, 0))
    else:
        optimized_coeffs = best_flat.reshape(n_output_buckets, n_terms + 1)
        y_bucket_mean_opt = optimized_coeffs[:, 0]
        corr_terms_opt = optimized_coeffs[:, 1:]

    data_mse = best_obj - _correction_penalty(best_flat)
    print(f"Optimization completed. Final training MSE: {data_mse:.4f}")
    return corr_terms_opt, y_bucket_mean_opt


# ---------------------------------------------------------------------------
# Phase 1: closed-form consequent solver, pluggable basis, shared prediction.
#
# For fixed firing strengths the TSK output is *linear* in the consequent
# coefficients, so the optimal (ridge-regularized) consequents have a closed
# form. `solve_tsk_consequents` replaces the per-bucket least-squares init plus
# L-BFGS refinement with a single exact linear solve. `build_consequent_features`
# centralizes the polynomial feature expansion (previously duplicated across the
# `compute_*_order_corrections`, `optimize_tsk_coefficients`, and every caller's
# predict loop) and adds an orthogonal (Legendre) basis for better conditioning.
# `predict_tsk` is the single shared prediction path.
# ---------------------------------------------------------------------------

# Per-order polynomial degrees applied to each feature. "full-2nd" additionally
# includes pairwise cross-product (interaction) terms.
_ORDER_DEGREES: dict[str, list[int]] = {
    "1st": [1],
    "2nd": [1, 2],
    "3rd": [1, 2, 3],
    "full-2nd": [1, 2],
}


def _poly_features(X_rule: ndarray, degree: int, basis: str) -> ndarray:
    """Apply a degree-`degree` univariate polynomial to every column of X_rule.

    - basis="raw":        x ** degree (plain monomials, matches legacy behavior).
    - basis="orthogonal": the Legendre polynomial L_degree(x), evaluated
      elementwise. L_1(x) = x, L_2(x) = (3x^2 - 1)/2, ... These are far better
      conditioned than raw monomials, so the least-squares design matrix is not
      ill-posed at 2nd/3rd order (the raw basis produces ~1e4 coefficients that
      overfit). The mapping is stateless, so it is identical at fit and predict.
    """
    if basis == "raw":
        return X_rule ** degree
    if basis == "orthogonal":
        coeffs = np.zeros(degree + 1)
        coeffs[degree] = 1.0
        return np.polynomial.legendre.legval(X_rule, coeffs)
    raise ValueError(f"Unknown basis: {basis!r} (expected 'raw' or 'orthogonal')")


def _gaussian_rbf_features(X_rule: ndarray, centers: ndarray, gamma: float,
                           radius: float | None = None) -> ndarray:
    """Compute Gaussian RBF features with optional compact support.

    For each center point, compute: exp(-gamma * ||x - center||^2) if ||x - center|| <= radius, else 0.

    Args:
        X_rule: (n_samples, n_features) feature matrix.
        centers: (n_centers, n_features) RBF center points.
        gamma: Shape parameter controlling RBF width. Typical range: 0.1 to 10.
               Larger gamma = narrower, more localized RBFs.
        radius: Compact support radius. RBF is exactly zero outside this radius.
                If None, no truncation (infinite support). Typically set to ~0.5-1.0 in
                normalized feature space.

    Returns:
        (n_samples, n_centers) matrix of RBF evaluations.
    """
    # Compute squared Euclidean distances: (n_samples, n_centers)
    sq_distances = np.sum((X_rule[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
    distances = np.sqrt(sq_distances)

    # Compute Gaussian RBF values
    rbf = np.exp(-gamma * sq_distances)

    # Apply compact support (truncate to zero outside radius)
    if radius is not None:
        rbf[distances > radius] = 0.0

    return rbf


def build_consequent_features(
    X_rule: ndarray,
    order: str,
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
    rbf_centers: ndarray | None = None,
    rbf_gamma: float = 1.0,
    rbf_radius: float | None = None,
) -> ndarray:
    """Build the consequent design columns (without the intercept) for one order.

    Column layout is grouped by degree then interactions, i.e. for a raw
    full-2nd basis: ``[x_1..x_n, x_1^2..x_n^2, cross_pairs...]``. This matches the
    legacy ``np.hstack`` ordering exactly, so raw-basis coefficients are a
    drop-in replacement for the old code.

    For 'gaussian-rbf' basis, returns RBF evaluations: exp(-gamma * ||x - center||^2)
    with optional compact support (zero outside radius).

    Args:
        X_rule: (n_samples, n_features) feature matrix.
        order: One of '0th', '1st', '2nd', '3rd', 'full-2nd'. Ignored for 'gaussian-rbf'.
        basis: 'raw', 'orthogonal', or 'gaussian-rbf' (see `_poly_features`, `_gaussian_rbf_features`).
        cross_pairs: For 'full-2nd', the explicit list of (i, j) feature-index pairs.
        rbf_centers: (n_centers, n_features) RBF center points. Required for 'gaussian-rbf' basis.
        rbf_gamma: Shape parameter for Gaussian RBFs (default 1.0).
        rbf_radius: Compact support radius for RBFs. If None, RBFs have infinite support.
    """
    n_samples, n_features = X_rule.shape
    if order == "0th":
        return np.empty((n_samples, 0))

    if basis == "gaussian-rbf":
        if rbf_centers is None:
            raise ValueError("rbf_centers must be provided for 'gaussian-rbf' basis")
        return _gaussian_rbf_features(X_rule, rbf_centers, rbf_gamma, radius=rbf_radius)

    if order not in _ORDER_DEGREES:
        raise ValueError(f"Unknown order: {order!r}")

    cols = [_poly_features(X_rule, d, basis) for d in _ORDER_DEGREES[order]]

    if order == "full-2nd":
        if cross_pairs is None:
            cross_pairs = list(combinations(range(n_features), 2))
        cross = [X_rule[:, i] * X_rule[:, j] for i, j in cross_pairs]
        if cross:
            cols.append(np.column_stack(cross))

    return np.hstack(cols) if cols else np.empty((n_samples, 0))


def _normalize_firing_strengths(firing_strengths: ndarray) -> ndarray:
    """Row-normalize firing strengths using the canonical zero-firing convention.

    Rows whose total firing is <= 1e-6 are left as all-zero (no rule fires). This
    exact convention must be shared by the solver and prediction, or training and
    evaluation silently disagree. It is self-consistent for the closed-form solver:
    an all-zero design row contributes nothing to the ridge normal equations (so
    such training rows are effectively ignored by the fit), and at predict time the
    row yields 0 -- the graceful fallback for a point no rule covers. (Contrast the
    old L-BFGS path, which forced a uniform 1/n_labels blend only to stop the
    optimizer wandering the resulting null space; the regularized closed-form solve
    has no such null-space issue, and a uniform blend would multiply
    unbounded out-of-range consequents under extrapolation.)
    """
    row_sums = firing_strengths.sum(axis=1)
    valid = row_sums > 1e-6
    norm = np.zeros_like(firing_strengths)
    norm[valid] = firing_strengths[valid] / row_sums[valid, np.newaxis]
    return norm


def solve_tsk_consequents(
    X_train: pd.DataFrame,
    gaussian_memberships: GaussianMixtureModel,
    top_n_todo: list[typing.Any],
    y_bucket_mean: pd.Series | ndarray,
    y_train: pd.DataFrame,
    n_output_buckets: int,
    order: typing.Literal["0th", "1st", "2nd", "3rd", "full-2nd"] = "2nd",
    l2_reg: float = 1e-6,
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
    pin_extremes: bool = True,
    norms: NormPair | None = None,
    verbose: bool = True,
    feature_arrays: dict[str, np.ndarray] | None = None,
    rbf_centers: ndarray | None = None,
    rbf_gamma: float = 1.0,
    rbf_radius: float | None = None,
) -> tuple[ndarray, ndarray]:
    """Solve for the globally optimal TSK consequent coefficients in closed form.

    For fixed firing strengths, the prediction
    ``y_hat = sum_r w_r * (mean_r + basis(X) @ corr_r)`` is linear in the
    coefficients ``(mean_r, corr_r)``. Stacking a per-rule design block
    ``w_r[:, None] * [1 | basis(X)]`` across all rules and solving the ridge
    normal equations ``(Phi^T Phi + l2_reg * D) beta = Phi^T y`` (with D = 0 on
    the intercept/bucket-mean columns) yields the exact minimizer of the
    firing-weighted MSE + ridge penalty -- no iterative optimizer needed.

    ``feature_arrays``: optional pre-extracted ``{feature_name: ndarray}``
    mapping (see `tsk_firing_strengths`), also reused here to build `X_rule`
    without a pandas round-trip. `None` reproduces the previous behavior exactly.

    Computes the firing strengths from `gaussian_memberships` and delegates the
    actual solve to `solve_tsk_consequents_from_firing` -- see that function for
    the ``pin_extremes`` behavior and the ``(corr_terms_opt, y_bucket_mean_opt)``
    return value. Split out so callers that already have firing strengths from
    somewhere other than a single `GaussianMixtureModel` -- e.g. the interval
    type-2 antecedent refiner, which needs its own per-candidate consequent
    re-solve -- can reuse the identical ridge solve instead of reimplementing it.
    """
    if verbose:
        print(f"\nSolving {order} consequents in closed form (basis={basis}, l2={l2_reg:g})...")

    firing_strengths_train, labels_train = tsk_firing_strengths(
        X_train[top_n_todo], gaussian_memberships, norms=norms, feature_arrays=feature_arrays
    )
    return solve_tsk_consequents_from_firing(
        firing_strengths_train, labels_train, X_train, top_n_todo, y_bucket_mean, y_train,
        order=order, l2_reg=l2_reg, basis=basis, cross_pairs=cross_pairs, pin_extremes=pin_extremes,
        verbose=verbose, feature_arrays=feature_arrays, rbf_centers=rbf_centers,
        rbf_gamma=rbf_gamma, rbf_radius=rbf_radius,
    )


def solve_tsk_consequents_from_firing(
    firing_strengths_train: ndarray,
    labels_train: list[int],
    X_train: pd.DataFrame,
    top_n_todo: list[typing.Any],
    y_bucket_mean: pd.Series | ndarray,
    y_train: pd.DataFrame,
    order: typing.Literal["0th", "1st", "2nd", "3rd", "full-2nd"] = "2nd",
    l2_reg: float = 1e-6,
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
    pin_extremes: bool = True,
    verbose: bool = True,
    feature_arrays: dict[str, np.ndarray] | None = None,
    rbf_centers: ndarray | None = None,
    rbf_gamma: float = 1.0,
    rbf_radius: float | None = None,
) -> tuple[ndarray, ndarray]:
    """`solve_tsk_consequents`'s ridge solve, given firing strengths directly.

    ``pin_extremes`` holds the first and last rules' bucket means fixed at the
    values supplied in ``y_bucket_mean`` -- normally the observed min and max set
    by :func:`partition_output` -- instead of letting the solve re-derive them.
    Without it those pinned values are silently discarded, since this function
    returns its own ``y_bucket_mean_opt`` and the prediction path uses that; the
    model is then free to shrink its output range inward and can no longer reach
    the extremes of the target. The constraint is applied exactly, not as a
    penalty: the pinned columns are moved to the right-hand side and the
    remaining coefficients are solved against the residual, so the result is the
    exact minimizer subject to the constraint.

    The pin is skipped where it cannot be stated well-posedly: fewer than two
    rules (the two extremes are then the same coefficient), or a
    ``y_bucket_mean`` too short to index by rule, both of which fall back to the
    unconstrained solve. A non-finite value at one extreme skips only that end;
    the other stays pinned.

    Returns (corr_terms_opt, y_bucket_mean_opt), matching
    `optimize_tsk_coefficients`.
    """
    norm_fs = _normalize_firing_strengths(firing_strengths_train)
    n_rules = norm_fs.shape[1]

    if feature_arrays is not None:
        X_rule = np.column_stack([feature_arrays[c] for c in top_n_todo]) if top_n_todo \
            else np.empty((len(X_train), 0))
    else:
        X_rule = X_train[top_n_todo].to_numpy()
    feats = build_consequent_features(X_rule, order, basis=basis, cross_pairs=cross_pairs,
                                      rbf_centers=rbf_centers, rbf_gamma=rbf_gamma,
                                      rbf_radius=rbf_radius)
    n_terms = feats.shape[1]
    n_coeffs_per_rule = 1 + n_terms

    # Per-rule augmented features [1 | basis(X)] are identical across rules; only
    # the firing weights differ. Build the stacked design by broadcasting.
    phi = np.hstack([np.ones((X_rule.shape[0], 1)), feats])  # (n_samples, 1 + n_terms)
    design = (norm_fs[:, :, np.newaxis] * phi[:, np.newaxis, :]).reshape(
        X_rule.shape[0], n_rules * n_coeffs_per_rule
    )

    y = np.asarray(y_train["y_value"].values, dtype=float)

    # Ridge diagonal: never penalize the intercept (column 0 of each rule block).
    penalty = np.ones(n_rules * n_coeffs_per_rule)
    penalty[::n_coeffs_per_rule] = 0.0

    # Which columns, if any, are held fixed. Each rule block's column 0 IS that
    # rule's bucket mean, so pinning is a linear equality constraint on those
    # columns. Everything that could make the constraint ill-defined -- a single
    # rule (where "first" and "last" are the same column), a short or non-finite
    # y_bucket_mean -- drops out here, leaving the solve below to handle the
    # empty case as the ordinary unconstrained problem.
    pinned_cols: list[int] = []
    pinned_vals: list[float] = []
    if pin_extremes and n_rules >= 2 and y_bucket_mean is not None:
        ybm = np.asarray(y_bucket_mean, dtype=float).ravel()
        # Only apply constraints if y_bucket_mean is large enough to index all bucket labels
        max_bucket_label = int(np.max(labels_train))
        if ybm.size > max_bucket_label:
            for rule_idx in (0, n_rules - 1):
                bucket_label = labels_train[rule_idx]
                value = float(ybm[bucket_label])
                if np.isfinite(value):
                    pinned_cols.append(rule_idx * n_coeffs_per_rule)
                    pinned_vals.append(value)

    if pinned_cols:
        # Move the pinned columns' known contribution to the right-hand side and
        # solve the reduced system for the rest. That is exact -- the result is
        # the true minimizer subject to the constraint, not a penalty
        # approximation -- and it needs no iteration.
        pinned = np.asarray(pinned_cols, dtype=int)
        values = np.asarray(pinned_vals, dtype=float)
        free = np.setdiff1d(np.arange(design.shape[1]), pinned)
        residual = y - design[:, pinned] @ values
        design_free = design[:, free]

        # Use lstsq on the augmented design (with ridge penalty as rows) for better
        # conditioning. Near-singular designs return finite but huge coefficients
        # when solved via normal equations; lstsq on the design matrix has condition
        # number that is the square root of the Gram's and handles ill-conditioning
        # gracefully by truncating small singular values instead of inverting them.
        if l2_reg > 0:
            # Augment the design with ridge penalty rows: sqrt(l2_reg) * sqrt(D)
            sqrt_penalty = np.sqrt(l2_reg * penalty[free])
            design_aug = np.vstack([design_free, np.diag(sqrt_penalty)])
            residual_aug = np.hstack([residual, np.zeros_like(sqrt_penalty)])
            beta_free = np.linalg.lstsq(design_aug, residual_aug, rcond=None)[0]
        else:
            beta_free = np.linalg.lstsq(design_free, residual, rcond=None)[0]

        beta = np.zeros(design.shape[1])
        beta[pinned] = values
        beta[free] = beta_free
    else:
        # Use lstsq on the augmented design (with ridge penalty as rows) for better
        # conditioning. See comment above about why this is necessary.
        if l2_reg > 0:
            sqrt_penalty = np.sqrt(l2_reg * penalty)
            design_aug = np.vstack([design, np.diag(sqrt_penalty)])
            y_aug = np.hstack([y, np.zeros_like(sqrt_penalty)])
            beta = np.linalg.lstsq(design_aug, y_aug, rcond=None)[0]
        else:
            beta = np.linalg.lstsq(design, y, rcond=None)[0]

    coeffs = beta.reshape(n_rules, n_coeffs_per_rule)
    y_bucket_mean_opt = coeffs[:, 0].copy()
    corr_terms_opt = coeffs[:, 1:].copy() if n_terms > 0 else np.zeros((n_rules, 0))

    if verbose:
        # beta carries the pinned entries, so this is the fitted prediction in
        # both cases -- no need to reassemble it from the reduced solve.
        print(f"  Training MSE: {_mse(y, design @ beta):.4f}")
    return corr_terms_opt, y_bucket_mean_opt


def predict_tsk(
    X: pd.DataFrame,
    model: GaussianMixtureModel,
    top_n_todo: list[typing.Any],
    y_bucket_mean: ndarray,
    corr_terms: ndarray,
    order: str = "2nd",
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
    norms: NormPair | None = None,
    feature_arrays: dict[str, np.ndarray] | None = None,
    rbf_centers: ndarray | None = None,
    rbf_gamma: float = 1.0,
    rbf_radius: float | None = None,
) -> ndarray:
    """Shared TSK prediction path used by the solver's callers and CV.

    Uses the same firing-strength normalization and feature basis as
    `solve_tsk_consequents`, so fit and predict cannot silently diverge.

    ``feature_arrays``: optional pre-extracted ``{feature_name: ndarray}``
    mapping (see `tsk_firing_strengths`); `None` reproduces the previous
    behavior exactly.
    """
    firing_strengths, labels = tsk_firing_strengths(
        X[top_n_todo], model, norms=norms, feature_arrays=feature_arrays
    )
    return apply_tsk_consequents(
        X, top_n_todo, firing_strengths, labels, y_bucket_mean, corr_terms,
        order=order, basis=basis, cross_pairs=cross_pairs,
        feature_arrays=feature_arrays,
        rbf_centers=rbf_centers, rbf_gamma=rbf_gamma, rbf_radius=rbf_radius,
    )


def rule_consequent_values(
    X: pd.DataFrame,
    top_n_todo: list[typing.Any],
    labels: list[typing.Any],
    y_bucket_mean: ndarray,
    corr_terms: ndarray,
    order: str = "2nd",
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
    feature_arrays: dict[str, np.ndarray] | None = None,
    rbf_centers: ndarray | None = None,
    rbf_gamma: float = 1.0,
    rbf_radius: float | None = None,
) -> ndarray:
    """Per-rule TSK consequent output, *before* any firing-strength weighting.

    Returns an ``(n_samples, n_rules)`` array: column ``ij`` is rule ``labels[ij]``'s
    own crisp output (``y_bucket_mean[ij] + feats @ corr_terms[ij, :]``) for every
    row, independent of how strongly that rule fires.

    Factored out of `apply_tsk_consequents` so that callers who need to combine
    rules themselves -- interval type-2 Karnik-Mendel type reduction weights and
    sums these same per-rule values under its own switch-point search rather than
    a plain firing-strength weighted mean -- reuse the exact consequent
    evaluation instead of re-deriving it (see `it2_kernel.karnik_mendel_tsk`).
    """
    n_rules = len(labels)
    if order == "0th":
        return np.broadcast_to(np.asarray(y_bucket_mean, dtype=float), (len(X), n_rules)).copy()

    if feature_arrays is not None:
        X_rule = np.column_stack([feature_arrays[c] for c in top_n_todo]) if top_n_todo \
            else np.empty((len(X), 0))
    else:
        X_rule = X[top_n_todo].to_numpy()
    feats = build_consequent_features(X_rule, order, basis=basis, cross_pairs=cross_pairs,
                                      rbf_centers=rbf_centers, rbf_gamma=rbf_gamma,
                                      rbf_radius=rbf_radius)
    return np.asarray(y_bucket_mean, dtype=float)[np.newaxis, :] + feats @ corr_terms.T


def apply_tsk_consequents(
    X: pd.DataFrame,
    top_n_todo: list[typing.Any],
    firing_strengths: ndarray,
    labels: list[typing.Any],
    y_bucket_mean: ndarray,
    corr_terms: ndarray,
    order: str = "2nd",
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
    feature_arrays: dict[str, np.ndarray] | None = None,
    rbf_centers: ndarray | None = None,
    rbf_gamma: float = 1.0,
    rbf_radius: float | None = None,
) -> ndarray:
    """Evaluate TSK consequents against *supplied* firing strengths.

    Split out of `predict_tsk` so that inference schemes which compute firing
    strengths differently -- notably the interval type-2 path, whose strengths
    come out of type reduction rather than a single type-1 forward pass -- can
    reuse the identical consequent evaluation and normalization instead of
    reimplementing it. Sharing this is the point: the IT2 regressor previously
    had its own `predict`, and it silently diverged (see `it2_regressor`).
    """
    norm_fs = _normalize_firing_strengths(firing_strengths)
    rule_vals = rule_consequent_values(
        X, top_n_todo, labels, y_bucket_mean, corr_terms, order=order, basis=basis,
        cross_pairs=cross_pairs, feature_arrays=feature_arrays,
        rbf_centers=rbf_centers, rbf_gamma=rbf_gamma, rbf_radius=rbf_radius,
    )
    return np.sum(norm_fs * rule_vals, axis=1)


def compute_rbf_centers(X: ndarray, n_centers: int = 5) -> ndarray:
    """Compute RBF center points using quantiles of the feature space.

    Args:
        X: (n_samples, n_features) feature matrix.
        n_centers: Number of centers per feature (default 5, produces n_features * n_centers total).

    Returns:
        (n_centers_total, n_features) array of RBF centers.
    """
    n_samples, n_features = X.shape
    quantiles = np.linspace(0.1, 0.9, n_centers)
    centers = []
    for feat_idx in range(n_features):
        feat_quantiles = np.quantile(X[:, feat_idx], quantiles)
        centers.append(feat_quantiles)
    # Create a grid of all combinations (Cartesian product across features)
    return np.column_stack(np.meshgrid(*centers, indexing='ij')).reshape(-1, n_features)


def select_interaction_terms(
    X_train: pd.DataFrame,
    top_n_todo: list[typing.Any],
    y_train: pd.DataFrame,
    y_bucket_mean: pd.Series | ndarray,
    max_pairs: int | None = None,
    random_state: int = 42,
    candidate_pairs: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Screen full-2nd cross-product terms with a LassoCV, returning the pairs worth keeping.

    Cross terms grow as O(n_features^2); most are noise. We fit a cross-validated
    Lasso to the centered target using the standardized linear + squared + all
    cross features, then keep only the interaction pairs whose coefficient is
    non-zero (optionally capped at the `max_pairs` largest by magnitude). Pass the
    returned list as `cross_pairs` to `solve_tsk_consequents`/`predict_tsk`.

    Args:
        candidate_pairs: Restrict the screen to these (index-space) pairs
            instead of every pair among `top_n_todo` -- e.g. the shortlist
            `gauss_math.calculate_interaction_scores`/`take_top_interactions`
            already flagged as worth considering, turning this from an
            `O(n^2)` screen into a final sparsity pass over a handful of
            candidates. `None` (default) reproduces the original all-pairs
            behavior exactly.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    X_rule = X_train[top_n_todo].to_numpy()
    n_features = X_rule.shape[1]
    all_pairs = candidate_pairs if candidate_pairs is not None else list(combinations(range(n_features), 2))
    if not all_pairs:
        return []

    cross = np.column_stack([X_rule[:, i] * X_rule[:, j] for i, j in all_pairs])
    design = np.hstack([X_rule, X_rule ** 2, cross])
    y = np.asarray(y_train["y_value"].values, dtype=float)
    y = y - y.mean()

    design = StandardScaler().fit_transform(design)
    lasso = LassoCV(cv=5, random_state=random_state, max_iter=10_000).fit(design, y)

    # The cross-term coefficients live in the trailing block of the design.
    cross_coeffs = lasso.coef_[2 * n_features:]
    order_by_mag = np.argsort(np.abs(cross_coeffs))[::-1]
    kept = [all_pairs[k] for k in order_by_mag if cross_coeffs[k] != 0.0]
    if max_pairs is not None:
        kept = kept[:max_pairs]
    # Preserve canonical (ascending) pair ordering for a deterministic layout.
    kept.sort()
    print(f"  Sparse interactions: kept {len(kept)}/{len(all_pairs)} cross terms")
    return kept


def select_consequent_hyperparams(
    X_train: pd.DataFrame,
    gaussian_memberships: GaussianMixtureModel,
    top_n_todo: list[typing.Any],
    y_bucket_mean: pd.Series | ndarray,
    y_train: pd.DataFrame,
    n_output_buckets: int,
    candidate_orders: typing.Sequence[str] = ("1st", "2nd", "full-2nd", "3rd"),
    candidate_bases: typing.Sequence[str] = ("raw", "orthogonal"),
    candidate_l2: typing.Sequence[float] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1),
    n_folds: int = 5,
    pin_extremes: bool = True,
    random_state: int = 42,
    rbf_n_centers: int = 3,
    rbf_gamma: float = 1.0,
    rbf_radius: float | None = None,
) -> dict[str, typing.Any]:
    """Pick (order, basis, l2_reg) by k-fold cross-validated R² on X_train.

    The membership model (antecedents) is held fixed; only the consequents are
    re-solved per candidate per fold, so this is cheap (each fit is one linear
    solve). k-fold averaging (rather than a single validation split) is important
    here: a single fold is high-variance for ridge-strength selection and can pick
    an unregularized fit that overfits the test set. Selecting on held-out folds
    prevents the higher-order models from chasing the test set.

    Args:
        rbf_n_centers: Number of centers per feature for Gaussian RBF basis.
        rbf_gamma: Shape parameter for Gaussian RBF evaluations.
        rbf_radius: Compact support radius for RBF basis. If None, RBFs have infinite support.

    Returns a dict with keys: order, basis, l2_reg, val_r2 (mean across folds),
    val_mse (mean across folds).
    """
    from sklearn.model_selection import KFold

    # Pre-compute RBF centers if needed
    rbf_centers = None
    if "gaussian-rbf" in candidate_bases:
        X_array = X_train[top_n_todo].to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        rbf_centers = compute_rbf_centers(X_array, n_centers=rbf_n_centers)

    idx = np.arange(len(X_train))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = list(kf.split(idx))

    best = {"val_r2": -np.inf}
    for order in candidate_orders:
        for basis in candidate_bases:
            for l2 in candidate_l2:
                fold_r2, fold_mse = [], []
                for tr_idx, val_idx in folds:
                    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
                    y_val_true = y_val["y_value"].values

                    corr, means = solve_tsk_consequents(
                        X_tr, gaussian_memberships, top_n_todo, y_bucket_mean, y_tr,
                        n_output_buckets=n_output_buckets, order=order, l2_reg=l2, basis=basis,
                        pin_extremes=pin_extremes,
                        verbose=False,
                        rbf_centers=rbf_centers, rbf_gamma=rbf_gamma, rbf_radius=rbf_radius,
                    )
                    y_hat = predict_tsk(X_val, gaussian_memberships, top_n_todo, means, corr,
                                        order=order, basis=basis,
                                        rbf_centers=rbf_centers, rbf_gamma=rbf_gamma,
                                        rbf_radius=rbf_radius)
                    keep = ~np.isnan(y_hat)
                    fold_r2.append(_rsquared(y_val_true[keep], y_hat[keep]))
                    fold_mse.append(_mse(y_val_true[keep], y_hat[keep]))
                mean_r2 = float(np.mean(fold_r2))
                if mean_r2 > best["val_r2"]:
                    best = {"order": order, "basis": basis, "l2_reg": l2,
                            "val_r2": mean_r2, "val_mse": float(np.mean(fold_mse))}

    print(f"\nCV-selected consequent: order={best['order']}, basis={best['basis']}, "
          f"l2={best['l2_reg']:g} ({n_folds}-fold val R²={best['val_r2']:.4f}, "
          f"val MSE={best['val_mse']:.4f})")
    return best
