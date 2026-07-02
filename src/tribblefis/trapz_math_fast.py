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


def fit_trapezoids_fast(data_1d: np.ndarray, n_bins: int = 50, ramp_width_ratio: float = 0.1, merge_width_ratio: float = 0.2) -> tuple[list[TrapezoidMembership], np.ndarray]:
    """Fast histogram-based trapezoid fitting without EM.

    Algorithm:
    1. Create histogram with n_bins
    2. Find contiguous bins with count > 0
    3. Merge regions separated by fewer empty bins than merge_width
    4. For each region, create one trapezoid:
       - [a, d] spans the bin region
       - [b, c] is the inner plateau (ramp_width inset from edges)

    Args:
        data_1d: 1D array of data points
        n_bins: Number of histogram bins (default: 50)
        ramp_width_ratio: Ramp width as fraction of total bin count (default: 0.1 = 10%)
        merge_width_ratio: Merge width as fraction of total bin count (default: 0.2 = 2x ramp_width)

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

    # Calculate ramp and merge widths based on bin count
    ramp_width_bins = max(1, int(n_bins * ramp_width_ratio))
    merge_width_bins = max(1, int(n_bins * merge_width_ratio))

    # Identify contiguous regions with count > 0
    active_bins = counts > 0
    trapezoids = []

    # Find contiguous regions and merge those separated by fewer empty bins than merge_width
    regions = _find_contiguous_regions(active_bins)
    regions = _merge_nearby_regions(regions, max_gap=merge_width_bins - 1)

    for start_idx, end_idx in regions:
        # Get the bin range for this region
        a = bin_edges[start_idx]  # Left edge of first bin
        d = bin_edges[end_idx + 1]  # Right edge of last bin

        # Inner plateau: ramp down over ramp_width bins
        ramp_width = bin_width * ramp_width_bins
        b = a + ramp_width
        c = d - ramp_width

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
        ramp_width = (d - a) * ramp_width_ratio
        b = a + ramp_width
        c = d - ramp_width
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


def _merge_nearby_regions(regions: list[tuple[int, int]], max_gap: int = 2) -> list[tuple[int, int]]:
    """Merge regions separated by max_gap or fewer empty bins.

    Args:
        regions: List of (start_idx, end_idx) tuples
        max_gap: Maximum number of empty bins to allow between regions for merging

    Returns:
        Merged list of (start_idx, end_idx) tuples
    """
    if not regions:
        return regions

    merged = [regions[0]]

    for current_start, current_end in regions[1:]:
        last_start, last_end = merged[-1]
        gap = current_start - last_end - 1

        if gap <= max_gap:
            # Merge regions
            merged[-1] = (last_start, current_end)
        else:
            # Add as new region
            merged.append((current_start, current_end))

    return merged


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
    n_bins: int = 50,
    ramp_width_ratio: float = 0.1,
    merge_width_ratio: float = 0.2,
) -> GaussianMixtureModel:
    """Create trapezoid membership model using fast histogram method.

    Analogous to create_trapz_membership_dict from trapz_math.py but uses
    the fast histogram-based method instead of EM for rapid fitting.

    Args:
        X: DataFrame of features
        y: Series of labels
        top_n_var_names: List of feature names to use
        n_bins: Number of histogram bins (default: 50)
        ramp_width_ratio: Ramp width as fraction of total bin count (default: 0.1 = 10%)
        merge_width_ratio: Merge width as fraction of total bin count (default: 0.2 = 2x ramp_width)

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
            trapezoids, weights = fit_trapezoids_fast(
                feature_data,
                n_bins=n_bins,
                ramp_width_ratio=ramp_width_ratio,
                merge_width_ratio=merge_width_ratio,
            )

            # Create label model
            label_model = LabelModel(memberships=trapezoids)
            label_models[label_value] = label_model

        # Create feature model
        feature_model = FeatureModel(label_models=label_models)
        feature_models[feature_name] = feature_model

    # Create and return the overall model
    return GaussianMixtureModel(feature_models=feature_models)
