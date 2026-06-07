import time

import numpy as np
import pandas as pd

from tribblefis.gauss_data import AnomalyParameters
from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    create_gaussian_membership_dict,
    take_top_features,
    log_transform, simple_gaussian_predict,
)
from tribblefis.gauss_plot import report_figures_of_merit, plot_anomaly_threshold_sweep, plot_membership_functions, \
    plot_confusion_matrix, plot_classification_report


def load_data(test: bool = False):
    # Load the benign traffic for training
    data = pd.read_csv("beth_data/labelled_training_data.csv")
    data["Traffic_type"] = "regular"
    if test:
        data = pd.read_csv("beth_data/labelled_testing_data.csv")
        data["Traffic_type"] = "regular"
        # Only update the "evil" as anomaly.
        data.loc[data["evil"] == 1, "Traffic_type"] = "anomaly"

    X = data

    y = X["Traffic_type"].copy()
    X = X.drop(columns=["sus", "evil", "Traffic_type", "args", "timestamp"])
    # data (as pandas dataframes)
    X = X.select_dtypes(include=[np.number])
    X = log_transform(X, ["processId", "mountNamespace", "eventId", "userId"], 1)
    return X, y


def _train_test_split(X, y):
    X_train, y_train = load_data(False)
    X_test, y_test = load_data(True)
    return X_train, X_test, y_train, y_test


def main():
    X, y = load_data()
    start_time = time.time()

    # Get the number of unique values in y
    n_unique = 2
    print(f"Number of unique values in y: {n_unique}")

    # Split dataset into train/test
    X_train, X_test, y_train, y_test = _train_test_split(X, y)
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    # Calculate correlation coefficient between Gaussian distributions using training data
    feature_differentiators = calculate_gaussian_correlation(X_train, y_train)

    # Take the top-n variables so that the normalized differentiation value encompasses 90-95%
    top_n, top_n_todo = take_top_features(feature_differentiators, top_p=1.0)

    print(f"Selected Top-{top_n} Variables ({top_n/len(feature_differentiators):.2%} coverage):")

    # Compute memberships using training data
    gaussian_memberships = create_gaussian_membership_dict(
        X_train,
        y_train,
        top_n_var_names=top_n_todo,
    )
    anomaly_details = AnomalyParameters(
        include_anomaly=True, threshold=0.99, label="anomaly", norm_conorm="hamacher", member_function="gaussian"
    )

    for fom_pass in range(1):
        cm_train, top_confusion_train, confused_data_train = report_figures_of_merit(
            X_train,
            y_train,
            gaussian_memberships,
            n_unique,
            start_time,
            top_n_todo,
            label="train",
            anomaly_details=anomaly_details,
        )

        print(f"{fom_pass}-pass Total Model Stats:")
        print("=" * 80)
        print(f"N_rules={gaussian_memberships.n_rules}")
        print(f"N_memberships={gaussian_memberships.n_membership_functions}")
        print(f"Possible rules={gaussian_memberships.possible_rules}")

        for (true_class, confused_class), confusion_data in confused_data_train.items():
            X_local_train, y_local_train = confusion_data["X"], confusion_data["y"]
            new_gaussian_memberships = create_gaussian_membership_dict(
                X_local_train, y_local_train, top_n_var_names=top_n_todo
            )
            # Now, we need to augment the existing gaussian memberships
            gaussian_memberships = gaussian_memberships.augment(new_gaussian_memberships)

    cm_test, top_confusion_test, confused_data_test = report_figures_of_merit(
        X_test,
        y_test,
        gaussian_memberships,
        n_unique,
        start_time,
        top_n_todo,
        label="test",
        anomaly_details=anomaly_details,
    )

    print(f"Test-pass Total Model Stats:")
    print("=" * 80)
    print(f"N_rules={gaussian_memberships.n_rules}")
    print(f"N_memberships={gaussian_memberships.n_membership_functions}")
    print(f"Possible rules={gaussian_memberships.possible_rules}")

    # Plot anomaly threshold sweep
    plot_anomaly_threshold_sweep(
        X_test,
        y_test,
        gaussian_memberships,
        top_n_todo,
        anomaly_label=anomaly_details.label,
    )

    # Create simple gaussian model from GaussianMixtureModel
    simple_model = gaussian_memberships.to_simple_model(anomaly_details)

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
