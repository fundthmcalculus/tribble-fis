"""
Fast histogram-based trapezoidal membership function fitting.

This module provides a simplified alternative to the EM-based trapz_math.py
that directly converts histogram bins into trapezoid membership functions.
Sequential bins with >0 frequency are grouped into single trapezoid MFs.

Key differences from trapz_math.py:
- No EM algorithm (much faster: O(n) instead of O(n*k*iter))
- No constrained optimization
- Direct histogram-to-MF conversion
- No parameter tuning needed (reproducible output)
"""

import numpy as np
import pandas as pd
from .gauss_data import TrapezoidMembership, GaussianMixtureModel, FeatureModel, LabelModel


def fit_trapezoids_fast(data_1d: np.ndarray, n_bins: int = 50) -> tuple[list[TrapezoidMembership], np.ndarray]:
    """Fast histogram-based trapezoid fitting without EM.

    Algorithm:
    1. Create histogram with n_bins
    2. Find contiguous bins with count > 0
    3. For each contiguous region, create one trapezoid:
       - [a, d] spans the bin region
       - [b, c] is the inner plateau (slightly inset from bin edges)

    Args:
        data_1d: 1D array of data points
        n_bins: Number of histogram bins (default: 50)

    Returns:
        (trapezoids, weights) where:
        - trapezoids: list of TrapezoidMembership objects
        - weights: normalized weights for each trapezoid (equal by default)
    """
    data_1d = np.asarray(data_1d, dtype=float)

    if len(data_1d) == 0:
        return [], np.array([])

    # Create histogram
    counts, bin_edges = np.histogram(data_1d, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Identify contiguous regions with count > 0
    active_bins = counts > 0
    trapezoids = []

    # Find contiguous regions
    regions = _find_contiguous_regions(active_bins)

    for start_idx, end_idx in regions:
        # Get the bin range for this region
        a = bin_edges[start_idx]  # Left edge of first bin
        d = bin_edges[end_idx + 1]  # Right edge of last bin

        # Inner plateau: slightly inset from outer edges
        inset = bin_width * 0.15  # 15% inset
        b = a + inset
        c = d - inset

        # Ensure valid trapezoid (b <= c)
        if b > c:
            b = c = (a + d) / 2

        trapezoids.append(TrapezoidMembership(a=a, b=b, c=c, d=d))

    if not trapezoids:
        # Fallback: single trapezoid covering data range
        a = data_1d.min()
        d = data_1d.max()
        if a == d:
            a = d - 0.5
        b = a + (d - a) * 0.15
        c = d - (d - a) * 0.15
        trapezoids = [TrapezoidMembership(a=a, b=b, c=c, d=d)]

    # Equal weights for all trapezoids
    weights = np.ones(len(trapezoids)) / len(trapezoids)

    return trapezoids, weights


def _find_contiguous_regions(active: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous regions of True values in a boolean array.

    Returns list of (start_idx, end_idx) tuples (inclusive on both ends).
    """
    regions = []
    in_region = False
    start = 0

    for i, is_active in enumerate(active):
        if is_active and not in_region:
            # Start of new region
            start = i
            in_region = True
        elif not is_active and in_region:
            # End of region
            regions.append((start, i - 1))
            in_region = False

    # Handle region that extends to the end
    if in_region:
        regions.append((start, len(active) - 1))

    return regions


def trapz_pdf_fast(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Same as trapz_math.trapz_pdf but included here for convenience."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)

    ab_width = b - a
    if ab_width > 0:
        mask = (x > a) & (x < b)
        y[mask] = (x[mask] - a) / ab_width

    y[(x >= b) & (x <= c)] = 1.0

    cd_width = d - c
    if cd_width > 0:
        mask = (x > c) & (x < d)
        y[mask] = (d - x[mask]) / cd_width

    area = (b - a) / 2 + (c - b) + (d - c) / 2
    if area > 0:
        y = y / area

    return y


def create_trapz_membership_dict_fast(
    X: pd.DataFrame,
    y: pd.Series,
    top_n_var_names: list[str],
) -> GaussianMixtureModel:
    """Create trapezoid membership model using fast histogram method.

    Analogous to create_trapz_membership_dict from trapz_math.py but uses
    the fast histogram-based method instead of EM for rapid fitting.

    Args:
        X: DataFrame of features
        y: Series of labels
        top_n_var_names: List of feature names to use

    Returns:
        GaussianMixtureModel with TrapezoidMembership objects
    """
    feature_models = {}

    for feature_name in top_n_var_names:
        label_models = {}

        for label_value in y.unique():
            # Get data for this label
            mask = y == label_value
            feature_data = X[feature_name][mask].values

            # Fit trapezoids using fast method
            trapezoids, weights = fit_trapezoids_fast(feature_data, n_bins=50)

            # Create label model
            label_model = LabelModel(memberships=trapezoids)
            label_models[label_value] = label_model

        # Create feature model
        feature_model = FeatureModel(label_models=label_models)
        feature_models[feature_name] = feature_model

    # Create and return the overall model
    return GaussianMixtureModel(feature_models=feature_models)
