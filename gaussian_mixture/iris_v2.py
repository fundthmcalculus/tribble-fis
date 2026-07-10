import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tribblefis.gaussian_classifier import (
    MixtureOfGaussiansFuzzyClassifier, MixtureOfGaussiansFuzzySequenceClassifier,
)


def load_data():
    data_path = "extended_flower_morphometrics.csv"
    if not os.path.exists(data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_path)

    df = pd.read_csv(data_path)
    df = df.dropna()
    X = df.drop("species", axis=1)
    y = df["species"]
    return X, y


def identify_poorly_recognized_classes(y_true, y_pred):
    """Identify classes with low recall (poorly recognized)."""
    classes = np.unique(y_true)
    poor_classes = []

    for cls in classes:
        mask = y_true == cls
        recall = np.sum((y_pred[mask] == cls)) / np.sum(mask)
        if recall < 0.8:
            poor_classes.append((cls, recall))

    return poor_classes


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}\n")

    # Primary classifier
    print("=" * 80)
    print("PRIMARY CLASSIFIER")
    print("=" * 80)
    # clf = MixtureOfGaussiansFuzzySequenceClassifier(top_p=0.95, n_gaussians=0, norm_conorm='probability')
    clf = MixtureOfGaussiansFuzzyClassifier(top_p=0.95, n_gaussians=3, norm_conorm='probability')
    clf.fit(X_train, y_train)

    # Training data confusion matrix to identify poorly recognized classes
    print("\nTraining Data Confusion Matrix:")
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Train Accuracy: {train_accuracy:.4f}\n")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))
    print()

    poor_classes = identify_poorly_recognized_classes(y_train.values, y_train_pred)
    if poor_classes:
        print(f"Poorly recognized classes (recall < 0.8):")
        for cls, recall in poor_classes:
            print(f"  {cls}: {recall:.2%} recall")
    else:
        print("All classes recognized well (recall >= 0.8)")

    # Test set evaluation
    print("\n" + "=" * 80)
    print("PRIMARY CLASSIFIER - TEST SET")
    print("=" * 80)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
