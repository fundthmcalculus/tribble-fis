"""
Trapezoidal membership function fitting using histogram-based Expectation Maximization.

This module provides fitting routines for mixtures of trapezoidal membership functions,
analogous to the Gaussian fitting in gauss_math.py but designed for 1D histogram data.
"""

from typing import Optional

import numpy as np
from scipy import signal, ndimage
from scipy.optimize import minimize

from .gauss_data import TrapezoidMembership


def trapz_pdf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Compute normalized trapezoidal probability density function.

    The trapezoidal membership function is piecewise linear:
    - 0 outside [a, d]
    - Rises linearly from 0 to 1 over [a, b]
    - Constant 1 over [b, c]
    - Falls linearly from 1 to 0 over [c, d]

    The normalized PDF is: trapz_mf(x) / area
    where area = (b-a)/2 + (c-b) + (d-c)/2

    Args:
        x: Array of points at which to evaluate the PDF
        a, b, c, d: Trapezoid parameters (must satisfy a <= b <= c <= d)

    Returns:
        Array of normalized PDF values
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)

    # Rising slope [a, b]
    ab_width = b - a
    if ab_width > 0:
        mask = (x > a) & (x < b)
        y[mask] = (x[mask] - a) / ab_width

    # Flat top [b, c]
    y[(x >= b) & (x <= c)] = 1.0

    # Falling slope [c, d]
    cd_width = d - c
    if cd_width > 0:
        mask = (x > c) & (x < d)
        y[mask] = (d - x[mask]) / cd_width

    # Compute area (normalization constant)
    area = (b - a) / 2 + (c - b) + (d - c) / 2
    if area <= 0:
        # Degenerate (zero-width) trapezoid: density is undefined rather than a
        # spike of unnormalized height 1 at the single point b==c.
        return np.zeros_like(x, dtype=float)
    y = y / area

    return y


def _init_trapz_from_histogram(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    n_components: int,
    data_min: float,
    data_max: float,
) -> tuple[list[tuple[float, float, float, float]], np.ndarray]:
    """Initialize trapezoid parameters from histogram peaks.

    Strategy:
    1. Smooth histogram with Gaussian kernel
    2. Find K peaks using scipy.signal.find_peaks
    3. For each peak, set [b,c] at half-power width
    4. Set [a,d] at valley edges or data boundaries
    5. Apply minimum-width guards

    Args:
        bin_centers: Histogram bin centers
        bin_counts: Counts per bin
        n_components: Number of trapezoid components
        data_min, data_max: Data range for bounds

    Returns:
        (params_list, weights) where params_list is [(a,b,c,d), ...]
    """
    # Normalize histogram to density
    density = bin_counts / np.sum(bin_counts) if np.sum(bin_counts) > 0 else bin_counts

    # Smooth with Gaussian kernel (sigma = 1 bin)
    smoothed = ndimage.gaussian_filter1d(density, sigma=1.0)

    # Find peaks
    if n_components <= 0 or n_components > 10:
        n_components = max(1, min(3, n_components if n_components > 0 else 1))

    peaks, properties = signal.find_peaks(smoothed, prominence=np.max(smoothed) * 0.05)

    # If fewer peaks found than requested, use what we found; if more, take largest
    if len(peaks) == 0:
        peaks = np.array([np.argmax(smoothed)])
    elif len(peaks) > n_components:
        prominences = properties["prominences"]
        peaks = peaks[np.argsort(prominences)[-n_components:]]
        peaks = np.sort(peaks)

    params_list = []
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0

    for peak_idx in peaks:
        peak_x = bin_centers[peak_idx]
        peak_height = smoothed[peak_idx]

        # Find half-power width (where density falls to 0.5 * peak height)
        half_height = peak_height / 2

        # Search left from peak
        b_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if smoothed[i] < half_height:
                b_idx = i
                break
        else:
            b_idx = 0
        b = bin_centers[b_idx]

        # Search right from peak
        c_idx = peak_idx
        for i in range(peak_idx + 1, len(bin_centers)):
            if smoothed[i] < half_height:
                c_idx = i
                break
        else:
            c_idx = len(bin_centers) - 1
        c = bin_centers[c_idx]

        # Find valley edges
        # Left valley: minimum between this peak and the previous, or data_min
        if len(params_list) == 0:
            a = data_min
        else:
            prev_peak_idx = peaks[len(params_list) - 1]
            valley_region = smoothed[prev_peak_idx:peak_idx]
            if len(valley_region) > 0:
                valley_idx = prev_peak_idx + np.argmin(valley_region)
                a = bin_centers[valley_idx]
            else:
                a = (bin_centers[prev_peak_idx] + bin_centers[peak_idx]) / 2

        # Right valley: minimum between this peak and the next, or data_max
        if len(params_list) == len(peaks) - 1:
            d = data_max
        else:
            next_peak_idx = peaks[len(params_list) + 1]
            valley_region = smoothed[peak_idx:next_peak_idx]
            if len(valley_region) > 0:
                valley_idx = peak_idx + np.argmin(valley_region)
                d = bin_centers[valley_idx]
            else:
                d = (bin_centers[peak_idx] + bin_centers[next_peak_idx]) / 2

        # Apply minimum-width guard
        min_bc_width = bin_width
        if c - b < min_bc_width:
            mid = (b + c) / 2
            b = mid - min_bc_width / 2
            c = mid + min_bc_width / 2

        min_ad_width = 3 * (c - b)
        if d - a < min_ad_width:
            mid = (a + d) / 2
            span = min_ad_width / 2
            a = mid - span
            d = mid + span

        # Clamp to data range
        a = max(a, data_min)
        d = min(d, data_max)
        b = np.clip(b, a, d)
        c = np.clip(c, b, d)

        params_list.append((a, b, c, d))

    # Initialize weights uniformly
    weights = np.ones(len(params_list)) / len(params_list)

    return params_list, weights


def _trapz_log_likelihood(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    params_list: list[tuple[float, float, float, float]],
    weights: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """Compute log-likelihood of histogram under trapezoid mixture.

    Args:
        bin_centers: Histogram bin centers
        bin_counts: Counts per bin
        params_list: [(a, b, c, d), ...] for each component
        weights: Mixing weights (shape (n_components,), sums to 1)
        eps: Small epsilon to avoid log(0)

    Returns:
        Total weighted log-likelihood
    """
    # Vectorized mixture PDF over all bin centers at once.
    mixture_pdf = np.zeros_like(bin_centers, dtype=float)
    for k, (a, b, c, d) in enumerate(params_list):
        mixture_pdf += weights[k] * trapz_pdf(bin_centers, a, b, c, d)

    # Avoid log(0)
    mixture_pdf = np.maximum(mixture_pdf, eps)
    return float(np.sum(bin_counts * np.log(mixture_pdf)))


def _em_e_step(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    params_list: list[tuple[float, float, float, float]],
    weights: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """E-step: compute responsibilities r[i,k] = P(z=k | x_i).

    Args:
        bin_centers: Histogram bin centers
        bin_counts: Counts per bin
        params_list: Trapezoid parameters
        weights: Mixing weights
        eps: Small epsilon for numerical stability

    Returns:
        Responsibilities matrix of shape (n_bins, n_components)
    """
    n_bins = len(bin_centers)
    n_components = len(params_list)

    # Weighted PDF for each component, vectorized over all bins: shape (n_bins, K)
    densities = np.empty((n_bins, n_components))
    for k, (a, b, c, d) in enumerate(params_list):
        densities[:, k] = weights[k] * trapz_pdf(bin_centers, a, b, c, d)

    # Normalize each row to get responsibilities.
    denom = densities.sum(axis=1)
    valid = denom > eps
    responsibilities = np.empty((n_bins, n_components))
    # Where the mixture density is non-negligible, normalize; otherwise use uniform.
    responsibilities[valid] = densities[valid] / denom[valid, None]
    responsibilities[~valid] = 1.0 / n_components

    return responsibilities


def _em_m_step_weights(
    responsibilities: np.ndarray,
    bin_counts: np.ndarray,
) -> np.ndarray:
    """M-step for mixing weights.

    Args:
        responsibilities: Shape (n_bins, n_components)
        bin_counts: Counts per bin

    Returns:
        Updated weights, shape (n_components,)
    """
    n_components = responsibilities.shape[1]
    weights = np.zeros(n_components)

    total_count = np.sum(bin_counts)
    for k in range(n_components):
        weights[k] = np.sum(responsibilities[:, k] * bin_counts) / total_count

    # Normalize
    weights = weights / np.sum(weights)
    return weights


def _em_m_step_params(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    responsibilities: np.ndarray,
    params_list: list[tuple[float, float, float, float]],
    data_min: float,
    data_max: float,
) -> list[tuple[float, float, float, float]]:
    """M-step for trapezoid parameters using constrained optimization.

    For each component k, minimize the negative weighted log-likelihood using SLSQP.

    Args:
        bin_centers: Histogram bin centers
        bin_counts: Counts per bin
        responsibilities: Shape (n_bins, n_components)
        params_list: Current trapezoid parameters
        data_min, data_max: Data range for bounds

    Returns:
        Updated trapezoid parameters
    """
    n_components = responsibilities.shape[1]
    new_params = []

    bin_centers = np.asarray(bin_centers, dtype=float)
    bin_counts = np.asarray(bin_counts, dtype=float)

    for k in range(n_components):
        a_k, b_k, c_k, d_k = params_list[k]
        # Per-bin coefficient c_i = responsibility * count, precomputed once so the
        # SLSQP objective (called many times per iteration) is a single vectorized pass.
        coeff_k = responsibilities[:, k] * bin_counts

        def objective(params, _coeff=coeff_k):
            a, b, c, d = params
            pdf_vals = np.maximum(trapz_pdf(bin_centers, a, b, c, d), 1e-10)
            return -np.dot(_coeff, np.log(pdf_vals))

        # Constraints: a <= b <= c <= d (with analytic linear Jacobians)
        constraints = [
            {'type': 'ineq', 'fun': lambda p: p[1] - p[0],
             'jac': lambda p: np.array([-1.0, 1.0, 0.0, 0.0])},  # b >= a
            {'type': 'ineq', 'fun': lambda p: p[2] - p[1],
             'jac': lambda p: np.array([0.0, -1.0, 1.0, 0.0])},  # c >= b
            {'type': 'ineq', 'fun': lambda p: p[3] - p[2],
             'jac': lambda p: np.array([0.0, 0.0, -1.0, 1.0])},  # d >= c
        ]

        # Bounds: all parameters within data range
        bounds = [(data_min, data_max)] * 4

        # Initial guess
        x0 = [a_k, b_k, c_k, d_k]

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 100}
        )

        if result.success and result.x is not None:
            a_new, b_new, c_new, d_new = result.x
        else:
            a_new, b_new, c_new, d_new = x0

        new_params.append((a_new, b_new, c_new, d_new))

    return new_params


def fit_trapezoids_em(
    data_1d: np.ndarray,
    n_components: int,
    n_bins: int = 50,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> tuple[list[TrapezoidMembership], np.ndarray, float]:
    """Run EM to fit a mixture of trapezoids to 1D data.

    Args:
        data_1d: 1D array of observations
        n_components: Number of trapezoid components
        n_bins: Number of histogram bins
        max_iter: Maximum EM iterations
        tol: Convergence tolerance on log-likelihood relative change
        random_state: Random seed (currently unused, for API compatibility)

    Returns:
        (trapezoids, weights, log_likelihood): List of fitted TrapezoidMembership objects,
        their mixing weights, and final log-likelihood
    """
    if random_state is not None:
        np.random.seed(random_state)

    data_1d = np.asarray(data_1d, dtype=float)
    data_min, data_max = data_1d.min(), data_1d.max()

    if data_min == data_max:
        # Degenerate case: all data identical
        mid = data_min
        trapz = TrapezoidMembership.create(mid, mid, mid, mid)
        return [trapz], np.array([1.0]), 0.0

    # Compute histogram
    bin_counts, bin_edges = np.histogram(data_1d, bins=n_bins, range=(data_min, data_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initialize parameters from histogram peaks
    params_list, weights = _init_trapz_from_histogram(
        bin_centers, bin_counts, n_components, data_min, data_max
    )

    # Prune components with zero responsibility
    params_list = params_list[:n_components]
    weights = weights[:len(params_list)]
    weights = weights / np.sum(weights)

    # EM loop
    prev_ll = -np.inf
    best_ll = -np.inf
    best_params = params_list
    best_weights = weights

    for iteration in range(max_iter):
        # E-step
        responsibilities = _em_e_step(bin_centers, bin_counts, params_list, weights)

        # M-step (weights)
        weights = _em_m_step_weights(responsibilities, bin_counts)

        # M-step (parameters)
        params_list = _em_m_step_params(
            bin_centers, bin_counts, responsibilities, params_list, data_min, data_max
        )

        # Compute log-likelihood
        ll = _trapz_log_likelihood(bin_centers, bin_counts, params_list, weights)

        # Track best
        if ll > best_ll:
            best_ll = ll
            best_params = params_list
            best_weights = weights

        # Check convergence
        if prev_ll > -np.inf:
            rel_change = abs(ll - prev_ll) / (abs(prev_ll) + 1e-10)
            if rel_change < tol:
                break

        prev_ll = ll

    # Convert to TrapezoidMembership objects
    trapezoids = [
        TrapezoidMembership.create(a, b, c, d)
        for a, b, c, d in best_params
    ]

    return trapezoids, best_weights, best_ll


def find_optimal_trapezoids(
    data_1d: np.ndarray,
    max_components: int = 4,
    n_bins: int = 50,
) -> int:
    """Select number of trapezoid components using BIC.

    BIC = n_params * log(N) - 2 * log_likelihood
    For K components: n_params = 5K - 1 (4 params per trapezoid + K-1 free weights)

    Args:
        data_1d: 1D array of observations
        max_components: Maximum number of components to try
        n_bins: Number of histogram bins

    Returns:
        Optimal number of components (1 <= K <= max_components)
    """
    data_1d = np.asarray(data_1d, dtype=float)
    N = len(data_1d)

    bics = []
    for k in range(1, max_components + 1):
        trapezoids, weights, ll = fit_trapezoids_em(
            data_1d, n_components=k, n_bins=n_bins, max_iter=100
        )

        n_params = 5 * k - 1
        bic = n_params * np.log(N) - 2 * ll
        bics.append(bic)

    optimal_k = np.argmin(bics) + 1
    return optimal_k


def fit_trapezoids(
    X,
    y,
    column: str,
    label_value: int,
    n_trapezoids: int = 0,
    max_samples: int = 20_000,
) -> list[TrapezoidMembership]:
    """Fit multiple trapezoidal MFs to a single variable filtered by label.

    Analogue of gauss_math.fit_gaussians() but for trapezoids.

    Args:
        X: Feature dataframe
        y: Label series
        column: Column name to fit
        label_value: Class label to filter by
        n_trapezoids: Number of trapezoid components (0 for automatic BIC selection)
        max_samples: Maximum samples to use

    Returns:
        List of fitted TrapezoidMembership objects
    """
    data = X[column][y == label_value].dropna().values
    data = data[:max_samples]

    if len(data) == 0:
        return []

    # Determine number of trapezoids if not specified
    if n_trapezoids <= 0:
        n_trapezoids = find_optimal_trapezoids(data, max_components=4)
        print(f"  Automatically selected {n_trapezoids} trapezoids for {column} (label {label_value})")

    # Fit EM
    trapezoids, weights, ll = fit_trapezoids_em(
        data, n_components=n_trapezoids, n_bins=50, max_iter=100, tol=1e-4
    )

    return trapezoids


def create_trapz_membership_dict(
    X, y, top_n_var_names: list[str], n_trapezoids: int | dict[str, int] = 0
) -> "GaussianMixtureModel":
    """Create a trapezoid membership model for top-n variables across all class labels.

    Analogue of gauss_math.create_gaussian_membership_dict() but uses trapezoids.

    Returns the same GaussianMixtureModel container type — the model is agnostic
    about whether LabelModel.memberships contains Gaussian or Trapezoid objects.

    Args:
        X: Feature dataframe
        y: Label series
        top_n_var_names: List of feature names to fit
        n_trapezoids: Number of trapezoids (0 for automatic, or dict per feature)

    Returns:
        GaussianMixtureModel containing fitted trapezoid MFs
    """
    from concurrent.futures.thread import ThreadPoolExecutor
    from .gauss_data import GaussianMixtureModel, FeatureModel, LabelModel

    unique_labels = y.unique()

    def process_feature(feature_name: str) -> tuple[str, FeatureModel]:
        """Process a single feature across all labels"""
        label_models = {}

        # Determine number of trapezoids for this feature
        if isinstance(n_trapezoids, dict):
            feature_n_trapezoids = n_trapezoids.get(feature_name, 0)
        else:
            feature_n_trapezoids = n_trapezoids

        for label_value in unique_labels:
            label_n_trapezoids = 0
            if isinstance(n_trapezoids, dict):
                label_n_trapezoids = n_trapezoids.get(label_value, 0)
            if label_n_trapezoids > 0:
                feature_n_trapezoids = label_n_trapezoids

            trapz_params = fit_trapezoids(X, y, feature_name, label_value, feature_n_trapezoids)
            label_models[label_value] = LabelModel(memberships=trapz_params)

        return feature_name, FeatureModel(label_models=label_models)

    feature_models = {}

    # Use ThreadPoolExecutor for parallel fitting
    with ThreadPoolExecutor() as executor:
        result = executor.map(process_feature, top_n_var_names)
        for r in result:
            feature_models[r[0]] = r[1]

    return GaussianMixtureModel(feature_models=feature_models)


class TrapzMixtureModel:
    """Fits a mixture of trapezoidal membership functions to 1D histogram data.

    This class provides a simple API analogous to scikit-learn for fitting
    trapezoid MFs to 1D data using histogram-based EM.

    Attributes:
        n_components: Number of components (0 = auto-select via BIC)
        max_components: Maximum number of components for BIC search
        n_bins: Number of histogram bins
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
        random_state: Random seed

    After calling fit(), access:
        trapezoids_: List of fitted TrapezoidMembership objects
        weights_: Mixing weights
        log_likelihood_: Final log-likelihood
        bic_: BIC of the fitted model
    """

    def __init__(
        self,
        n_components: int = 0,
        max_components: int = 4,
        n_bins: int = 50,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.max_components = max_components
        self.n_bins = n_bins
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Set after fit
        self.trapezoids_: Optional[list[TrapezoidMembership]] = None
        self.weights_: Optional[np.ndarray] = None
        self.log_likelihood_: Optional[float] = None
        self.bic_: Optional[float] = None

    def fit(self, data_1d: np.ndarray) -> "TrapzMixtureModel":
        """Fit trapezoidal MFs to 1D data.

        Args:
            data_1d: 1D array of observations

        Returns:
            self
        """
        data_1d = np.asarray(data_1d, dtype=float)

        # Auto-select number of components if n_components <= 0
        n_comp = self.n_components
        if n_comp <= 0:
            n_comp = find_optimal_trapezoids(
                data_1d, max_components=self.max_components, n_bins=self.n_bins
            )

        # Fit EM
        trapezoids, weights, ll = fit_trapezoids_em(
            data_1d,
            n_components=n_comp,
            n_bins=self.n_bins,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        self.trapezoids_ = trapezoids
        self.weights_ = weights
        self.log_likelihood_ = ll

        # Compute BIC
        N = len(data_1d)
        K = len(trapezoids)
        n_params = 5 * K - 1
        self.bic_ = n_params * np.log(N) - 2 * ll

        return self
