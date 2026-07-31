import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from tribblefis.gaussian_classifier import (
    MixtureOfGaussiansFuzzySequenceClassifier, )


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
    # Confirmed best with GridCV search.
    # clf = MixtureOfGaussiansFuzzyClassifier(top_p=0.95, n_gaussians=1, norm_conorm='min/max')
    clf = MixtureOfGaussiansFuzzySequenceClassifier(top_p=0.95, n_gaussians=1, norm_conorm='min/max', min_confused=400)
    clf.fit(X_train, y_train)
    # Plot membership functions for the base classifier per-label
    # print("\nPlotting membership functions for base classifier...")
    # plot_var_gauss_dist(X_train, y_train, features_to_plot=clf.top_features_, model=clf.model_)

    # Training data confusion matrix to identify poorly recognized classes
    print("\nTraining Data Confusion Matrix:")
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Train Accuracy: {train_accuracy:.4f}\n")

    # Get confused training samples
    print("\nAnalyzing confused samples...")
    confused_mask = y_train != y_train_pred
    confused_indices = np.where(confused_mask)[0]
    print(f"Number of confused samples: {len(confused_indices)} out of {len(y_train)}")
    
    if len(confused_indices) > 0:
        # Get confused samples data
        if isinstance(X_train, pd.DataFrame):
            X_confused = X_train.iloc[confused_indices]
            y_confused_true = y_train.iloc[confused_indices]
            y_confused_pred = pd.Series(y_train_pred[confused_indices], index=y_confused_true.index)
        else:
            X_confused = X_train[confused_indices]
            y_confused_true = y_train[confused_indices]
            y_confused_pred = y_train_pred[confused_indices]
        
        # Print confusion details
        print("\nConfused sample breakdown:")
        for true_label in np.unique(y_confused_true):
            mask = y_confused_true == true_label
            if isinstance(mask, pd.Series):
                mask = mask.values
            pred_labels = y_confused_pred[mask]
            print(f"  True class '{true_label}':")
            unique, counts = np.unique(pred_labels, return_counts=True)
            for pred_label, count in zip(unique, counts):
                print(f"    -> Predicted as '{pred_label}': {count} samples")
        
        # Plot data distribution for confused samples
        print("\nPlotting data distribution for confused samples...")
        features_to_plot = clf.top_features_ if hasattr(clf, 'top_features_') else X_train.columns.tolist()
        
        # Create a combined label showing true vs predicted
        if isinstance(y_confused_true, pd.Series):
            y_confused_combined = y_confused_true.astype(str) + " → " + y_confused_pred.astype(str)
        else:
            y_confused_combined = np.array([f"{true} → {pred}" for true, pred in zip(y_confused_true, y_confused_pred)])
        
        # plot_var_gauss_dist(X_confused, y_confused_combined, features_to_plot=features_to_plot, model=None)

    # Test set evaluation
    print("\n" + "=" * 80)
    print("PRIMARY CLASSIFIER - TEST SET")
    print("=" * 80)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
