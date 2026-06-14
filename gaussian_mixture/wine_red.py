import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
from tribblefis.gauss_math import log_transform
from tribblefis.regression import (
    report_regression_performance,
    plot_tsk_order_comparison,
)


def load_data():
    data_path = "winequality-white.csv"
    if not os.path.exists(data_path):
        # Try to find it in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_path)

    X = pd.read_csv(data_path, delimiter=";")
    X = X.dropna()
    # Quality is a continuous score, not a class label, so keep it numeric.
    y = X["quality"].astype(float)
    y.name = "y_value"
    X.drop(["quality"], axis=1, inplace=True)
    X = X.select_dtypes(include=[np.number])

    return X, y


def main():
    start_time = time.time()
    X, y = load_data()

    print(f"Target 'quality' range: [{y.min():.2f}, {y.max():.2f}], "
          f"mean={y.mean():.3f}, std={y.std():.3f}")

    X = log_transform(X, ["total sulfur dioxide", "free sulfur dioxide", "chlorides"], 1)

    # Split dataset into train/test. Stratify on coarse quality buckets so both
    # splits span the full continuous range.
    strata = pd.cut(y, bins=5, labels=False, include_lowest=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=strata
    )
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

    # Fit a TSK Gaussian-mixture regressor at several polynomial orders and
    # compare how much each correction order helps.
    orders = ["0th", "1st", "2nd"]
    order_labels = ["0 Optimized", "1 Optimized", "2 Optimized"]

    # report_regression_performance expects the truth as a frame with a
    # "y_value" column; align indices with the positionally-ordered predictions.
    y_test_frame = pd.DataFrame({"y_value": y_test.to_numpy()})

    r2_list, rmse_list, pred_list = [], [], []
    for order, label in zip(orders, order_labels):
        print(f"\nFitting {order}-order TSK regressor...")
        reg = MixtureOfGaussiansFuzzyRegressor(
            n_output_buckets=5,
            tsk_order=order,
            optimize_coefficients=True,
            random_state=42,
        )
        reg.fit(X_train, y_train)

        # Because the output is rounded, we do the same.
        y_pred_test = np.round(reg.predict(X_test))
        print(f"\n{order}-order on TEST set:")
        print("=" * 80)
        r2, rmse = report_regression_performance(start_time, y_test_frame, y_pred_test, n_order=label)

        r2_list.append(r2)
        rmse_list.append(rmse)
        pred_list.append(y_pred_test)

    # Comparison baseline: a Random Forest regressor on the same split.
    print("\nFitting RandomForest regressor (comparison baseline)...")
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    # Round to match the TSK predictions for an apples-to-apples comparison.
    y_pred_rf = np.round(rf.predict(X_test))
    print("\nRandomForest on TEST set:")
    print("=" * 80)
    r2_rf, rmse_rf = report_regression_performance(start_time, y_test_frame, y_pred_rf, n_order="RandomForest")

    r2_list.append(r2_rf)
    rmse_list.append(rmse_rf)
    pred_list.append(y_pred_rf)
    order_labels = order_labels + ["RandomForest"]

    # Plot actual-vs-predicted scatter for each model on the test set.
    plot_tsk_order_comparison(r2_list, rmse_list, y_test_frame, pred_list, order_labels)

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
