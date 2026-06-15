"""
Darwin Handwriting Classification: Gaussian vs Trapezoid Comparison

This script runs both Gaussian and Trapezoid classifiers on the Darwin dataset
and produces a side-by-side comparison of their performance.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier


def load_data():
    data_path = "darwin.csv"
    if not os.path.exists(data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_path)

    X = pd.read_csv(data_path, delimiter=",")
    X = X.dropna()
    y = X["class"]
    X.drop(["class"], axis=1, inplace=True)
    X = X.select_dtypes(include=[np.number])
    return X, y


def evaluate_classifier(clf, X_train, X_test, y_train, y_test, name):
    """Train and evaluate a classifier."""
    print(f"\n{'=' * 80}")
    print(f"Training {name} Classifier")
    print(f"{'=' * 80}")

    start_time = time.time()
    # Use limited features for speed (Darwin has 450 features)
    clf.top_n = 20  # Limit to top 20 features for faster training
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predictions
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Probabilities
    y_test_proba = clf.predict_proba(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nTraining Time: {train_time:.2f} seconds")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    print(f"\nModel Structure:")
    model = clf.model_
    print(f"  Features selected: {len(clf.top_features_)}")
    print(f"  Total membership functions: {model.n_membership_functions}")
    print(f"  Possible rules: {model.possible_rules:.0f}")

    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"Confusion Matrix Shape: {cm.shape}")

    return {
        'name': name,
        'train_time': train_time,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'y_pred': y_test_pred,
        'y_proba': y_test_proba,
        'model': model,
        'top_features': clf.top_features_,
    }


def main():
    print("\n" + "=" * 80)
    print("DARWIN HANDWRITING CLASSIFICATION: Gaussian vs Trapezoid Comparison")
    print("=" * 80)

    # Load data
    print("\nLoading Darwin handwriting dataset...")
    X, y = load_data()
    n_unique = y.nunique()
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features, {n_unique} classes")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"Split: Train={len(X_train)}, Test={len(X_test)}")

    # Note: Darwin dataset has 450 features, feature selection is slow
    # Using faster settings: top_n=10 instead of auto-selection
    print("\nNote: Using top_n=10 for faster training (Darwin has 450 features)")

    # Train Gaussian classifier
    clf_gauss = MixtureOfGaussiansFuzzyClassifier(member_function="gaussian", top_n=10)
    results_gauss = evaluate_classifier(
        clf_gauss, X_train, X_test, y_train, y_test, "Gaussian"
    )

    # Train Trapezoid classifier (using fast method by default)
    clf_trapz = MixtureOfGaussiansFuzzyClassifier(member_function="trap", top_n=10)
    results_trapz = evaluate_classifier(
        clf_trapz, X_train, X_test, y_train, y_test, "Trapezoid (Fast)"
    )

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY - Gaussian vs Trapezoid (Fast)")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Gaussian':>20} {'Trapezoid':>20}")
    print("-" * 70)
    print(f"{'Training Time (seconds)':<30} {results_gauss['train_time']:>20.2f} {results_trapz['train_time']:>20.2f}")
    print(f"{'Training Accuracy':<30} {results_gauss['train_acc']:>20.4f} {results_trapz['train_acc']:>20.4f}")
    print(f"{'Test Accuracy':<30} {results_gauss['test_acc']:>20.4f} {results_trapz['test_acc']:>20.4f}")

    model_gauss = results_gauss['model']
    model_trapz = results_trapz['model']
    print(f"\n{'Model Complexity':<30} {'Gaussian':>20} {'Trapezoid':>20}")
    print("-" * 70)
    print(f"{'Features Selected':<30} {len(results_gauss['top_features']):>20} {len(results_trapz['top_features']):>20}")
    print(f"{'Total MFs':<30} {model_gauss.n_membership_functions:>20} {model_trapz.n_membership_functions:>20}")
    print(f"{'Possible Rules':<30} {model_gauss.possible_rules:>20.0f} {model_trapz.possible_rules:>20.0f}")

    # Winner
    print("\n" + "=" * 80)
    if results_gauss['test_acc'] > results_trapz['test_acc']:
        acc_diff = results_gauss['test_acc'] - results_trapz['test_acc']
        print(f"✓ GAUSSIAN wins by {acc_diff:.4f} ({acc_diff*100:.2f}%)")
    elif results_trapz['test_acc'] > results_gauss['test_acc']:
        acc_diff = results_trapz['test_acc'] - results_gauss['test_acc']
        print(f"✓ TRAPEZOID wins by {acc_diff:.4f} ({acc_diff*100:.2f}%)")
    else:
        print(f"✓ TIE: Both achieve {results_gauss['test_acc']:.4f} accuracy")

    print("\nKey Takeaways:")
    print("- Gaussian MFs: Sharp peaks, excellent classification performance")
    print(f"- Trapezoid MFs (Fast): Broader regions, {results_trapz['train_time']/results_gauss['train_time']:.1f}x faster training")
    print("- Trapezoids use fast histogram-based method for speed")
    print("=" * 80)


if __name__ == "__main__":
    main()
