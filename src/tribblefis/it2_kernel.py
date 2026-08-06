"""Interval Type-2 FIS kernel with Karnik-Mendel type reduction."""

import numpy as np
import pandas as pd

from .gauss_data import (
    IT2GaussianMixtureModel,
    IT2GaussianMembership,
    IT2TrapezoidMembership,
    IT2TriangularMembership,
    NormPair,
)
from .gauss_math import tsk_firing_strengths, GaussianMixtureModel, FeatureModel, LabelModel


def it2_firing_strengths(
    X: pd.DataFrame,
    model: IT2GaussianMixtureModel,
    norms: NormPair,
    km_iterations: int | None = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Compute IT2 firing strengths with type reduction.

    Computes both upper and lower membership function evaluations, then performs
    type reduction to get a crisp output.

    Args:
        X: Input feature matrix (n_samples, n_features)
        model: IT2GaussianMixtureModel
        norms: (t_norm, t_conorm) pair for fuzzy operations
        km_iterations: Number of KM algorithm iterations (None = simple averaging)

    Returns:
        firing_upper: (n_samples, n_labels) upper bound firing strengths
        firing_lower: (n_samples, n_labels) lower bound firing strengths
        firing_crisp: (n_samples, n_labels) type-reduced crisp outputs
        labels: list of output labels
    """
    n_samples = len(X)
    n_classes = model.n_classes
    labels = sorted(model.all_output_labels)

    # Extract upper and lower Type-1 models from IT2 model
    upper_model = _extract_upper_model(model)
    lower_model = _extract_lower_model(model)

    # Compute firing strengths for both upper and lower
    firing_upper = tsk_firing_strengths(X, upper_model, norms=norms)[0]
    firing_lower = tsk_firing_strengths(X, lower_model, norms=norms)[0]

    # Type reduction
    if km_iterations is None or km_iterations == 0:
        # Simple averaging: center-of-sets type reduction
        firing_crisp = 0.5 * (firing_upper + firing_lower)
    else:
        # Karnik-Mendel iterative algorithm
        firing_crisp = karnik_mendel_type_reduction(
            firing_upper, firing_lower, max_iterations=km_iterations
        )

    return firing_upper, firing_lower, firing_crisp, labels


def _extract_upper_model(model: IT2GaussianMixtureModel) -> GaussianMixtureModel:
    """Extract the upper bound Type-1 model from an IT2 model."""
    feature_models = {}
    for feature_name, it2_feature_model in model.feature_models.items():
        label_models = {}
        for label, it2_label_model in it2_feature_model.label_models.items():
            # Extract upper MFs from all IT2 memberships (works for any IT2 type)
            upper_mfs = [mf.upper_mf for mf in it2_label_model.memberships]
            label_models[label] = LabelModel(upper_mfs)
        feature_models[feature_name] = FeatureModel(label_models)

    # Use the same anomaly_params as the original model
    anomaly_params = getattr(model, 'anomaly_params', None)
    return GaussianMixtureModel(feature_models, anomaly_params=anomaly_params)


def _extract_lower_model(model: IT2GaussianMixtureModel) -> GaussianMixtureModel:
    """Extract the lower bound Type-1 model from an IT2 model."""
    feature_models = {}
    for feature_name, it2_feature_model in model.feature_models.items():
        label_models = {}
        for label, it2_label_model in it2_feature_model.label_models.items():
            # Extract lower MFs from all IT2 memberships (works for any IT2 type)
            lower_mfs = [mf.lower_mf for mf in it2_label_model.memberships]
            label_models[label] = LabelModel(lower_mfs)
        feature_models[feature_name] = FeatureModel(label_models)

    # Use the same anomaly_params as the original model
    anomaly_params = getattr(model, 'anomaly_params', None)
    return GaussianMixtureModel(feature_models, anomaly_params=anomaly_params)


def karnik_mendel_type_reduction(
    firing_upper: np.ndarray,
    firing_lower: np.ndarray,
    max_iterations: int = 10,
) -> np.ndarray:
    """Karnik-Mendel algorithm for type-2 type reduction.

    For each output, finds the crisp value that minimizes the distance between
    upper and lower bound firing strengths.

    Args:
        firing_upper: (n_samples, n_outputs) upper bound firing strengths
        firing_lower: (n_samples, n_outputs) lower bound firing strengths
        max_iterations: Number of KM iterations

    Returns:
        firing_crisp: (n_samples, n_outputs) type-reduced crisp outputs
    """
    n_samples, n_outputs = firing_upper.shape
    firing_crisp = np.zeros_like(firing_upper)

    for i in range(n_samples):
        for j in range(n_outputs):
            # For this sample and output, find the crisp output via KM
            y_l, y_r = _km_single(
                firing_upper[i, j],
                firing_lower[i, j],
                max_iterations=max_iterations,
            )
            # Output is the average of left and right switch points
            firing_crisp[i, j] = 0.5 * (y_l + y_r)

    return firing_crisp


def _km_single(
    y_upper: float,
    y_lower: float,
    max_iterations: int = 10,
) -> tuple[float, float]:
    """Single KM step for one output dimension.

    Simplified implementation: iteratively refine left and right switch points.
    For a scalar output (single firing strength value), this converges to the
    center point that minimizes spread.

    Args:
        y_upper: Upper bound for this output
        y_lower: Lower bound for this output
        max_iterations: Number of refinement iterations

    Returns:
        (y_left, y_right) - the computed output interval
    """
    # Initialize
    y_left = y_lower
    y_right = y_upper

    for iteration in range(max_iterations):
        # Midpoint
        y_mid = 0.5 * (y_left + y_right)

        # Update bounds (simple averaging on both sides)
        # In full KM, this would involve weighted switch points; here we simplify
        y_left = 0.5 * (y_lower + y_mid)
        y_right = 0.5 * (y_upper + y_mid)

        # Check convergence
        if abs(y_right - y_left) < 1e-6:
            break

    return y_left, y_right
