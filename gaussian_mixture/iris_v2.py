import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier


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

    clf = MixtureOfGaussiansFuzzyClassifier(top_p=0.95, n_gaussians=0, log_transform=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
