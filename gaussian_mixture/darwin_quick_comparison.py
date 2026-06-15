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

    for mf_type in ["gaussian", "trap"]:
        print(f"\n{'-' * 80}")
        print(f"Training {mf_type.upper()} Classifier...")
        print(f"{'-' * 80}")

        start = time.time()
        clf = MixtureOfGaussiansFuzzyClassifier(
            member_function=mf_type,
            top_n=10,  # Use only top 10 features for speed
            n_gaussians=1 if mf_type == "gaussian" else 1,
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

        results[mf_type] = {
            "time": train_time,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "cm": cm,
        }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<25} {'Gaussian':>20} {'Trapezoid':>20}")
    print("-" * 65)
    print(f"{'Training Time (sec)':<25} {results['gaussian']['time']:>20.2f} {results['trap']['time']:>20.2f}")
    print(f"{'Train Accuracy':<25} {results['gaussian']['train_acc']:>20.4f} {results['trap']['train_acc']:>20.4f}")
    print(f"{'Test Accuracy':<25} {results['gaussian']['test_acc']:>20.4f} {results['trap']['test_acc']:>20.4f}")

    # Winner
    print("\n" + "=" * 80)
    gauss_acc = results['gaussian']['test_acc']
    trap_acc = results['trap']['test_acc']

    if gauss_acc > trap_acc:
        diff = (gauss_acc - trap_acc) * 100
        print(f"✓ GAUSSIAN wins: {gauss_acc:.4f} vs {trap_acc:.4f} (+{diff:.2f}%)")
    elif trap_acc > gauss_acc:
        diff = (trap_acc - gauss_acc) * 100
        print(f"✓ TRAPEZOID wins: {trap_acc:.4f} vs {gauss_acc:.4f} (+{diff:.2f}%)")
    else:
        print(f"✓ TIE at {gauss_acc:.4f} accuracy")

    print("\nInterpretation:")
    print("• Gaussian: Sharp peaks → precise feature discrimination → better for complex tasks")
    print("• Trapezoid: Broader regions → more interpretable → better for simple/plateau data")
    print("• Darwin (450 handwriting features): Gaussian expected to win")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
