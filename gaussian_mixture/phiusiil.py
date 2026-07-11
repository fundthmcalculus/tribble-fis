import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
from tribblefis.gauss_math import log_transform, simple_gaussian_predict
from tribblefis.gauss_plot import report_figures_of_merit, plot_confusion_matrix, plot_classification_report, \
    plot_membership_functions


def load_data():
    data_path = "phishing_data/PhiUSIIL_Phishing_URL_Dataset.csv"
    if not os.path.exists(data_path):
        # Try to find it in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_path)

    X = pd.read_csv(data_path)
    X = X.dropna()
    y = X["label"]

    y = y.map({0: "legit", 1: "phish"})
    X = X.drop(columns=["label", "FILENAME"])
    X = X.select_dtypes(include=[np.number])

    return X, y


def main():
    start_time = time.time()
    X, y = load_data()

    # Get the number of unique values in y
    n_unique = y.nunique()
    print(f"Number of unique values in y: {n_unique}")

    X = log_transform(
        X,
        [
            "URLLength",
            "DomainLength",
            "NoOfAmpersandInURL",
            "NoOfObfuscatedChar",
            "LineOfCode",
            "LargestLineLength",
            "NoOfPopup",
            "NoOfiFrame",
            "NoOfLettersInURL",
            "NoOfDegitsInURL",
            "NoOfImage",
            "NoOfCSS",
            "NoOfJS",
            "NoOfSelfRef",
            "NoOfEmptyRef",
            "NoOfExternalRef",
            "TLDLength",
        ],
        1,
    )
    X = log_transform(
        X, ["SpacialCharRatioInURL", "DegitRatioInURL", "ObfuscationRatio", "CharContinuationRate"], 0.0001
    )

    # Split dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    # Initialize and fit the Gaussian Mixture Classifier
    clf = MixtureOfGaussiansFuzzyClassifier(
        top_n=5,
        n_gaussians={
            "LineOfCode": 3,
            "NoOfExternalRef": 2,
        }
    )
    clf.fit(X_train, y_train)

    top_n_todo = clf.top_features_
    gaussian_memberships = clf.model_

    cm_train, top_confusion_train, confused_data_train = report_figures_of_merit(
        X_train, y_train, gaussian_memberships, n_unique, start_time, top_n_todo, label="train"
    )

    print("1-pass Total Model Stats:")
    print("=" * 80)
    print(f"N_rules={gaussian_memberships.n_rules}")
    print(f"N_memberships={gaussian_memberships.n_membership_functions}")
    print(f"Possible rules={gaussian_memberships.possible_rules}")

    # for (true_class, confused_class), confusion_data in confused_data_train.items():
    #     X_local_train, y_local_train = confusion_data["X"], confusion_data["y"]
    #     new_gaussian_memberships = create_gaussian_membership_dict(
    #         X_local_train, y_local_train, top_n_var_names=top_n_todo
    #     )
    #     # Now, we need to augment the existing gaussian memberships
    #     gaussian_memberships = gaussian_memberships.augment(new_gaussian_memberships)

    cm_test, top_confusion_test, confused_data_test = report_figures_of_merit(
        X_test, y_test, gaussian_memberships, n_unique, start_time, top_n_todo, label="test"
    )

    # print("2-pass Total Model Stats:")
    # print("=" * 80)
    # print(f"N_rules={gaussian_memberships.n_rules}")
    # print(f"N_memberships={gaussian_memberships.n_membership_functions}")
    # print(f"Possible rules={gaussian_memberships.possible_rules}")

    # Create simple gaussian model from GaussianMixtureModel
    simple_model = gaussian_memberships.to_simple_model(None)

    print("\nSimple Gaussian Classifier Model Stats:")
    print("=" * 80)
    print(f"N_rules={len(simple_model.rules)}")
    print(f"N_memberships={len(simple_model.input_mfs)}")

    # Compare results on test set
    y_pred_simple = simple_gaussian_predict(X_test[top_n_todo], simple_model)
    simple_accuracy = np.mean(y_pred_simple == y_test)
    print(f"Simple Model Accuracy (test): {simple_accuracy:.4f}")
    plot_confusion_matrix(y_test, y_pred_simple, title=f"TSK Model Confusion Matrix (Simple Set)")
    plot_classification_report(y_test, y_pred_simple, title=f"TSK Model Classification Report (Simple Set)")

    # Plot membership functions
    plot_membership_functions(gaussian_memberships)
    # Plot membership functions of the simple model
    plot_membership_functions(simple_model)



if __name__ == "__main__":
    main()
