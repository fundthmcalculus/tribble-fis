import time

import numpy as np
import pandas as pd
from tribblefis import gauss_math
from sklearn.model_selection import train_test_split

from tribblefis.gauss_data import AnomalyParameters
from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    create_gaussian_membership_dict,
    take_top_features,
    log_transform, simple_gaussian_predict,
)
from tribblefis.gauss_plot import report_figures_of_merit, plot_anomaly_threshold_sweep, plot_confusion_matrix, \
    plot_classification_report

gauss_math.NORM_CONORM = "min/max"
gauss_math.MEMBER_FCN = "triangular"


def load_data():
    # Load the benign traffic for training
    benign_doorbell = pd.read_csv("iot-botnet/Danmini_Doorbell/benign_traffic.csv")
    benign_doorbell["Traffic_type"] = "regular"

    attack_doorbell = pd.read_csv("iot-botnet/Danmini_Doorbell/gafgyt_attacks/combo.csv")
    attack_doorbell["Traffic_type"] = "anomaly"

    X = pd.concat([benign_doorbell, attack_doorbell], ignore_index=True)

    y = X["Traffic_type"].copy()
    X = X.drop(columns=["Traffic_type"])
    # data (as pandas dataframes)
    X = X.select_dtypes(include=[np.number])

    return X, y


def _train_test_split(X, y):
    # For this case, because it is anomaly detection, we stratify as 2/3 of regular data for training, 1/3 regular + 3/3 attack data
    X_benign = X[y == "regular"]
    y_benign = y[y == "regular"]
    X_attack = X[y == "anomaly"]
    y_attack = y[y == "anomaly"]

    # Split regular data: 2/3 for training, 1/3 for testing
    X_benign_train, X_benign_test, y_benign_train, y_benign_test = train_test_split(
        X_benign, y_benign, test_size=0.3, random_state=42
    )

    # Training set: only regular data (2/3)
    X_train = X_benign_train
    y_train = y_benign_train

    # Test set: remaining regular (1/3) + all attack data
    X_test = pd.concat([X_benign_test, X_attack], ignore_index=True)
    y_test = pd.concat([y_benign_test, y_attack], ignore_index=True)

    print(f"Training set - regular: {len(y_train)}, anomaly: 0")
    print(f"Test set - regular: {len(y_benign_test)}, anomaly: {len(y_attack)}")

    # This `.copy()` is to defragment the data frame.
    return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()


def main():
    X, y = load_data()
    start_time = time.time()

    # Apply log transformation to all *_variance columns
    weight_cols = [col for col in X.columns if col.endswith("_weight")]
    mean_cols = [col for col in X.columns if col.endswith("_mean")]
    variance_cols = [col for col in X.columns if col.endswith("_variance")]
    log_transform(X, variance_cols, 1)

    # Get the number of unique values in y
    n_unique = y.nunique()
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
    gaussian_details = {}
    for w in weight_cols:
        gaussian_details[w] = 5
    for m in mean_cols:
        gaussian_details[m] = 2
    for v in variance_cols:
        gaussian_details[v] = 2
    gaussian_memberships = create_gaussian_membership_dict(
        X_train, y_train, top_n_var_names=top_n_todo, n_gaussians=gaussian_details
    )
    anomaly_details = AnomalyParameters(include_anomaly=True, threshold=0.95, label="anomaly")

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

    print("1-pass Total Model Stats:")
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

    print("2-pass Total Model Stats:")
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

    # Plot membership functions of the simple model
    # plot_membership_functions(simple_model)




if __name__ == "__main__":
    main()
