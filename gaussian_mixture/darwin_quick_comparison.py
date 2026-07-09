"""
Darwin Quick Comparison: Gaussian vs Trapezoid (Fast Version)

Uses pre-selected features to avoid slow feature ranking on 450-feature dataset.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier


def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "darwin.csv")
    X = pd.read_csv(data_path, delimiter=",")
    X = X.dropna()
    y = X["class"]
    X.drop(["class"], axis=1, inplace=True)
    X = X.select_dtypes(include=[np.number])
    return X, y


def main():
    print("\n" + "=" * 80)
    print("DARWIN HANDWRITING: Gaussian vs Trapezoid (Quick Comparison)")
    print("=" * 80)

    # Load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"\nDataset: Train={len(X_train)}, Test={len(X_test)}, Classes={y.nunique()}")

    results = {}

    # Test configurations: (name, member_function, trapz_method)
    configs = [
        ("Gaussian (EM)", "gaussian", None),
        ("Trapezoid (EM)", "trap", "em"),
        ("Trapezoid (Fast)", "trap", "fast"),
    ]

    for config_name, mf_type, trapz_method in configs:
        print(f"\n{'-' * 80}")
        print(f"Training {config_name}...")
        print(f"{'-' * 80}")

        start = time.time()
        if trapz_method is None:
            clf = MixtureOfGaussiansFuzzyClassifier(
                member_function=mf_type,
                top_n=10,  # Use only top 10 features for speed
                n_gaussians=1,
            )
        else:
            clf = MixtureOfGaussiansFuzzyClassifier(
                member_function=mf_type,
                trapz_method=trapz_method,
                top_n=10,  # Use only top 10 features for speed
            )
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        # Evaluate
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        cm = confusion_matrix(y_test, y_test_pred)

        print(f"Training time: {train_time:.2f}s")
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Test accuracy:  {test_acc:.4f}")
        print(f"Confusion matrix:\n{cm}")

        results[config_name] = {
            "time": train_time,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "cm": cm,
        }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - Three Methods Comparison")
    print("=" * 80)
    print(f"\n{'Metric':<25} {'Gaussian':<20} {'Trap (EM)':<20} {'Trap (Fast)':<20}")
    print("-" * 85)
    print(f"{'Training Time (sec)':<25} {results['Gaussian (EM)']['time']:>19.3f} {results['Trapezoid (EM)']['time']:>19.3f} {results['Trapezoid (Fast)']['time']:>19.3f}")
    print(f"{'Train Accuracy':<25} {results['Gaussian (EM)']['train_acc']:>19.4f} {results['Trapezoid (EM)']['train_acc']:>19.4f} {results['Trapezoid (Fast)']['train_acc']:>19.4f}")
    print(f"{'Test Accuracy':<25} {results['Gaussian (EM)']['test_acc']:>19.4f} {results['Trapezoid (EM)']['test_acc']:>19.4f} {results['Trapezoid (Fast)']['test_acc']:>19.4f}")

    # Speedup calculation
    em_time = results['Trapezoid (EM)']['time']
    fast_time = results['Trapezoid (Fast)']['time']
    speedup = em_time / fast_time if fast_time > 0 else float('inf')
    print(f"\n{'Speedup (Fast vs EM)':<25} {'N/A':>19} {'N/A':>19} {f'{speedup:.1f}x':>19}")

    # Winner analysis
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)

    gauss_acc = results['Gaussian (EM)']['test_acc']
    em_acc = results['Trapezoid (EM)']['test_acc']
    fast_acc = results['Trapezoid (Fast)']['test_acc']

    print(f"\nAccuracy Comparison:")
    print(f"  Gaussian:       {gauss_acc:.4f}")
    print(f"  Trapezoid (EM): {em_acc:.4f}")
    print(f"  Trapezoid (Fast): {fast_acc:.4f}")

    print(f"\nSpeed Comparison:")
    print(f"  Gaussian:       {results['Gaussian (EM)']['time']:.3f}s")
    print(f"  Trapezoid (EM): {results['Trapezoid (EM)']['time']:.3f}s")
    print(f"  Trapezoid (Fast): {results['Trapezoid (Fast)']['time']:.3f}s")

    print(f"\nKey Finding:")
    print(f"  Fast trapezoid is {speedup:.0f}x faster than EM trapezoid")
    print(f"  Accuracy difference (Fast vs EM): {abs(fast_acc - em_acc):.4f}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
