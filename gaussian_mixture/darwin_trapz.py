"""
Darwin Handwriting Classification using Trapezoidal Membership Functions

This is a trapezoid variant of darwin.py that uses the new trapezoid membership
functions for the Darwin handwriting dataset. It demonstrates how trapezoids
perform on a real classification task and allows comparison with Gaussian MFs.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
from tribblefis.gauss_plot import report_figures_of_merit


def load_data():
    data_path = "darwin.csv"
    if not os.path.exists(data_path):
        # Try to find it in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_path)

    X = pd.read_csv(data_path, delimiter=",")
    X = X.dropna()
    y = X["class"]
    X.drop(["class"], axis=1, inplace=True)
    X = X.select_dtypes(include=[np.number])
    return X, y


def main():
    from sklearn.metrics import accuracy_score

    X, y = load_data()

    # Get the number of unique values in y
    n_unique = y.nunique()
    print(f"Number of unique values in y: {n_unique}")

    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    print("\n" + "="*80)
    print("TRAPEZOID FITTING COMPARISON: EM vs Fast Method")
    print("="*80)

    # Train using EM method
    print("\n--- Training with EM Method ---")
    start_em = time.time()
    clf_em = MixtureOfGaussiansFuzzyClassifier(member_function="trap", trapz_method="em")
    clf_em.fit(X_train, y_train)
    time_em = time.time() - start_em

    y_test_pred_em = clf_em.predict(X_test)
    acc_em = accuracy_score(y_test, y_test_pred_em)
    print(f"EM Method: {time_em:.2f}s training, {acc_em:.4f} test accuracy")

    # Train using Fast method
    print("\n--- Training with Fast Histogram Method ---")
    start_fast = time.time()
    clf_fast = MixtureOfGaussiansFuzzyClassifier(member_function="trap", trapz_method="fast")
    clf_fast.fit(X_train, y_train)
    time_fast = time.time() - start_fast

    y_test_pred_fast = clf_fast.predict(X_test)
    acc_fast = accuracy_score(y_test, y_test_pred_fast)
    print(f"Fast Method: {time_fast:.2f}s training, {acc_fast:.4f} test accuracy")

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    speedup = time_em / time_fast if time_fast > 0 else float('inf')
    print(f"\nSpeedup: {speedup:.1f}x")
    print(f"EM:   {time_em:.2f}s,  {acc_em:.4f} accuracy")
    print(f"Fast: {time_fast:.2f}s,  {acc_fast:.4f} accuracy")

    if abs(acc_em - acc_fast) < 0.0001:
        print(f"\n✓ Fast method matches EM accuracy while being {speedup:.0f}x faster!")
    else:
        print(f"\nAccuracy difference: {abs(acc_em - acc_fast):.4f}")

    # Use fast method for final reporting
    print("\n" + "="*80)
    print("DETAILED EVALUATION (Fast Method)")
    print("="*80)
    top_n_todo = clf_fast.top_features_
    trapz_memberships = clf_fast.model_

    report_figures_of_merit(X_test, y_test, trapz_memberships, n_unique, time.time(), top_n_todo, label="test (trapezoid-fast)")

    # Now, plot a set of distributions for the most-differentiating variables
    # plot_var_gauss_dist(X_train, y_train, top_n_todo, trapz_memberships)


if __name__ == "__main__":
    main()
