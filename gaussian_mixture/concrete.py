import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tribblefis.gauss_math import (
    detect_and_apply_log_transform,
    calculate_gaussian_correlation,
    create_gaussian_membership_dict,
    take_top_features,
    standard_transform,
)
from tribblefis.regression import (
    report_regression_performance,
    plot_tsk_order_comparison,
    partition_output,
    solve_tsk_consequents,
    predict_tsk,
    select_interaction_terms,
    select_consequent_hyperparams,
)
from tribblefis.refine import refine_antecedents_local
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
    X = X.select_dtypes(include=[np.number]).astype(np.float64)
    return X, y


def main():
    start_time = time.time()
    X, y_raw = load_data()

    y_raw = standard_transform(y_raw)

    n_output_buckets: int = 3
    n_top_vars: int = -1
    n_gaussians: int = -1
    # Phase 2 antecedent refinement (default path). Each order's Gaussian
    # (mu, sigma) are tuned for that order via local L-BFGS-B; consequents are the
    # closed-form ridge solve at basis="raw", l2=refine_l2, all cross terms.
    b_refine_antecedents: bool = True     # local (L-BFGS-B) tuning of the Gaussian (mu, sigma)
    refine_l2: float = 1e-2               # ridge on correction terms (constants unpenalized)
    # Phase 1 consequent-only path (used when b_refine_antecedents is False):
    b_sparse_interactions: bool = True    # LassoCV-select cross terms for full-2nd
    b_cv_hyperparams: bool = True         # CV-select the consequent (order, basis, l2)

    if n_top_vars <= 0 or n_top_vars > len(X.columns):
        n_top_vars = len(X.columns)

    y, y_bucket_mean = partition_output(n_output_buckets, y_raw)

    X, log_transformed_features = detect_and_apply_log_transform(X, min_dynamic_range=2)
    X = standard_transform(X, column=X.columns)
    if log_transformed_features:
        print(f"Auto-detected log transform for: {log_transformed_features}")

    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

    # Closed-form consequent solve for each polynomial order. Because the TSK
    # output is linear in the consequent coefficients for fixed firing strengths,
    # a single ridge least-squares solve yields the exact optimum -- no iterative
    # optimizer needed.
    #
    # Phase 2: when refinement is enabled, each order gets its OWN membership model
    # whose Gaussian (mu, sigma) are tuned (local L-BFGS-B, closed-form consequents
    # as the inner step) for that order's complexity. Refining a single model for a
    # high order and reusing it distorts the fuzzy sets for the others (e.g. it
    # wrecks the 0th-order firing-weighted mean), so refinement is per-order. The
    # refinement fitness uses ALL cross terms for full-2nd -- constraining it to the
    # sparse subset markedly weakens the antecedent search -- and the evaluation
    # config is matched to the refinement config so fit and test cannot diverge.
    print("\nEvaluating Multi-Order TSK Model on TEST set:")
    print("=" * 80)

    orders = ["0th", "1st", "2nd", "full-2nd", "3rd"]
    order_labels = ["0 Optimized", "1 Optimized", "2 Optimized", "2-full Optimized", "3 Optimized"]

    r2_list, rmse_list, pred_list = [], [], []
    for order, label in zip(orders, order_labels):
        model_o = gaussian_memberships
        basis, l2, pairs = "raw", refine_l2, None

        if b_refine_antecedents:
            # Per-order antecedent refinement (all cross terms, matched config).
            model_o, _ = refine_antecedents_local(
                gaussian_memberships, X_train, y_train, top_n_todo,
                n_output_buckets=n_output_buckets, order=order,
                l2_reg=l2, basis=basis, cross_pairs=pairs,
            )
        elif b_cv_hyperparams and order != "0th":
            # Phase 1 path: CV-select the consequent (basis, l2); sparse full-2nd.
            pairs = (select_interaction_terms(X_train, top_n_todo, y_train, y_bucket_mean)
                     if (b_sparse_interactions and order == "full-2nd") else None)
            sel = select_consequent_hyperparams(
                X_train, gaussian_memberships, top_n_todo, y_bucket_mean, y_train,
                n_output_buckets=n_output_buckets,
                candidate_orders=(order,), candidate_bases=("raw", "orthogonal"),
            )
            basis, l2 = sel["basis"], sel["l2_reg"]

        corr_terms, y_bucket_mean_opt = solve_tsk_consequents(
            X_train, model_o, top_n_todo, y_bucket_mean, y_train,
            n_output_buckets=n_output_buckets, order=order,
            l2_reg=l2, basis=basis, cross_pairs=pairs,
        )
        y_test_pred = predict_tsk(
            X_test, model_o, top_n_todo, y_bucket_mean_opt, corr_terms,
            order=order, basis=basis, cross_pairs=pairs,
        )
        r2, rmse = report_regression_performance(start_time, y_test, y_test_pred, n_order=label)
        r2_list.append(r2)
        rmse_list.append(rmse)
        pred_list.append(y_test_pred)

    plot_tsk_order_comparison(r2_list, rmse_list, y_test, pred_list, order_labels)


if __name__ == "__main__":
    main()
