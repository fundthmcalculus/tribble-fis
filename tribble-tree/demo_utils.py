"""Shared utilities for demo scripts.

Provides common report functions and model evaluation patterns
to avoid duplication across demo_concrete.py, demo_phishing.py,
demo_deconstruct_synthetic.py, and cmapss_deconstruct_eval.py.
"""

from typing import Callable, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


def regressor_report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Report R² and RMSE for a regression model.

    Args:
        name: Model name for display
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        (r2, rmse) tuple
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {name:<44} R2={r2:6.3f}   RMSE={rmse:6.3f}")
    return r2, rmse


def classifier_report(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, pos_label: str = None
) -> Tuple[float, float]:
    """Report accuracy and F1 score for a classification model.

    Args:
        name: Model name for display
        y_true: Ground truth labels
        y_pred: Predicted labels
        pos_label: Label to use as positive class for F1 (optional)

    Returns:
        (accuracy, f1) tuple
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    if pos_label:
        print(f"  {name:<46} acc={acc:6.3f}   F1({pos_label})={f1:6.3f}")
    else:
        print(f"  {name:<46} acc={acc:6.3f}   F1={f1:6.3f}")
    return acc, f1


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
    report_fn: Callable,
    **report_kwargs
) -> Tuple[float, float]:
    """Evaluate a fitted model and report results.

    Convenience wrapper that generates predictions and reports metrics
    in a single call.

    Args:
        model: Fitted scikit-learn compatible estimator
        X_test: Test features
        y_test: Test labels/targets
        name: Model name for display
        report_fn: Function to report metrics (e.g., regressor_report or classifier_report)
        **report_kwargs: Additional keyword arguments for report_fn

    Returns:
        Tuple of metrics from report_fn
    """
    y_pred = model.predict(X_test)
    return report_fn(name, y_test, y_pred, **report_kwargs)
