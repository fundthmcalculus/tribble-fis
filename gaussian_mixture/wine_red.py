import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzySequenceClassifier
from tribblefis.gauss_math import log_transform
from tribblefis.gauss_plot import (
    report_figures_of_merit,
    plot_confusion_matrix,
    plot_classification_report,
)


def load_data():
    data_path = "winequality-red.csv"
    if not os.path.exists(data_path):
        # Try to find it in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_path)

    X = pd.read_csv(data_path, delimiter=";")
    X = X.dropna()
    y = X["quality"].astype(str)
    X.drop(["quality"], axis=1, inplace=True)
    X = X.select_dtypes(include=[np.number])

    return X, y


def report_sequence(clf: MixtureOfGaussiansFuzzySequenceClassifier, X, y, label: str):
    """Report cascade accuracy and per-class metrics for the full sequence."""
    y_pred = clf.predict(X)
    accuracy = np.mean(y_pred == y.values.astype(object))

    print(f"\nCascade ({clf.n_layers} layers) on {label.upper()} set:")
    print("=" * 80)
    print(f"Cascade Accuracy ({label}): {accuracy:.4f}")
    labels = list(np.unique(np.concatenate([y.values.astype(object), y_pred])))
    print(f"\nConfusion Matrix ({label}):")
    print(confusion_matrix(y, y_pred, labels=labels))
    print(f"\nClassification Report ({label}):")
    print(classification_report(y, y_pred))
    print("=" * 80)
    return y_pred


def plot_sequence(y_true, y_pred, label: str = "cascade"):
    """Plot confusion matrix and classification report for the cascade."""
    plot_confusion_matrix(y_true, y_pred, title=f"Cascade Confusion Matrix ({label} Set)")
    plot_classification_report(y_true, y_pred, title=f"Cascade Classification Report ({label} Set)")


def main():
    start_time = time.time()
    X, y = load_data()

    # Get the number of unique values in y
    n_unique = y.nunique()
    print(f"Number of unique values in y: {n_unique}")

    X = log_transform(X, ["total sulfur dioxide", "free sulfur dioxide", "chlorides"], 1)

    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    # Fit a layered (cascade) classifier. The primary model is trained on all
    # data; each subsequent specialist is a binary model keyed to the single
    # most-confused (predicted, true) class pair, trained to peel the true-class
    # rows out of that confused region, unless it flags the input as an anomaly
    # (outside its trained region), in which case the cascade stops.
    clf = MixtureOfGaussiansFuzzySequenceClassifier(
        max_layers=5,
        anomaly_threshold=0.95,
        norm_conorm="probability",
    )
    clf.fit(X_train, y_train)
    print(f"\nFit cascade with {clf.n_layers} layer(s).")
    print("Specialists keyed to (predicted -> true) confusion pairs: "
          f"{[f'{p} -> {t}' for p, t in clf.confused_pairs_]}")

    # Report figures of merit for the PRIMARY model alone (layer 0) using the
    # existing plotting/reporting helpers.
    primary = clf.layers_[0]
    report_figures_of_merit(
        X_train, y_train, primary.model_, n_unique, start_time, primary.top_features_, label="train (primary)"
    )
    report_figures_of_merit(
        X_test, y_test, primary.model_, n_unique, start_time, primary.top_features_, label="test (primary)"
    )

    # Report the full cascade (primary + specialists) for comparison.
    y_pred_train = report_sequence(clf, X_train, y_train, label="train (cascade)")
    y_pred_test = report_sequence(clf, X_test, y_test, label="test (cascade)")

    # Plot cascade metrics.
    plot_sequence(y_train.values.astype(object), y_pred_train, label="train (cascade)")
    plot_sequence(y_test.values.astype(object), y_pred_test, label="test (cascade)")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
