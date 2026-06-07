import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
from tribblefis.gauss_math import log_transform
from tribblefis.gauss_plot import report_figures_of_merit


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


def main():
    n_gaussians = 2

    start_time = time.time()
    X, y = load_data()

    # Get the number of unique values in y
    n_unique = y.nunique()
    print(f"Number of unique values in y: {n_unique}")

    X = log_transform(X, ["total sulfur dioxide", "free sulfur dioxide", "chlorides"], 1)

    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    # Initialize and fit the Gaussian Mixture Classifier
    clf = MixtureOfGaussiansFuzzyClassifier(n_gaussians=n_gaussians)
    clf.fit(X_train, y_train)

    top_n_todo = clf.top_features_
    gaussian_memberships = clf.model_

    cm_train, top_confusion_train, confused_data_train = report_figures_of_merit(
        X_train, y_train, gaussian_memberships, n_unique, start_time, top_n_todo, label="train"
    )

    for (true_class, confused_class), confusion_data in confused_data_train.items():
        X_local_train, y_local_train = confusion_data["X"], confusion_data["y"]
        # Augment the existing classifier
        clf.augment(X_local_train, y_local_train)

    # Update references after augmentation
    gaussian_memberships = clf.model_

    cm_test, top_confusion_test, confused_data_test = report_figures_of_merit(
        X_test, y_test, gaussian_memberships, n_unique, start_time, top_n_todo, label="test"
    )


if __name__ == "__main__":
    main()
