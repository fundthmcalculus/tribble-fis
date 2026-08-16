"""
Trapezoidal (and triangular) membership function fitting using histogram-based
Expectation Maximization.

This module provides fitting routines for mixtures of trapezoidal membership
functions, analogous to the Gaussian fitting in gauss_math.py but designed for
1D histogram data.

A triangular membership function is not a separate algorithm: it is the
degenerate trapezoid whose plateau ``[b, c]`` has collapsed to a single apex
point (``b == c``). Every function below that does real EM work -- the
histogram init, the E-step, the M-step, the log-likelihood, the BIC
selection -- takes a ``shape`` argument (``"trapezoid"`` or ``"triangle"``)
and is the *only* place that logic lives; there is no parallel triangle
implementation to keep in sync. Only the M-step's optimization actually
branches on ``shape`` (3 free parameters instead of 4, since the plateau is a
point rather than an independent interval); the histogram init, E-step,
weight M-step, and log-likelihood are shape-agnostic by construction, because
they only ever consume/produce ``(a, b, c, d)`` 4-tuples and evaluate them
through :func:`trapz_pdf` -- for a triangle that 4-tuple simply always has
``b == c``.

The ``fit_triangles_em`` / ``TriangleMixtureModel`` / etc. names at the
bottom of this module are thin, triangle-flavored entry points for callers
who don't want to pass ``shape="triangle"`` everywhere; each is a one-line
forward into the shared engine above it.
"""

from typing import Literal, Optional

import numpy as np
from scipy import signal, ndimage

from .gauss_data import TrapezoidMembership, TriangularMembership
from .optimizer_utils import optimizers_sub_solve

Shape = Literal["trapezoid", "triangle"]


def _solve_ordered_params(objective, x0: np.ndarray, data_min: float, data_max: float):
    """Minimize `objective(params)` over `params` constrained to be
    non-decreasing (``params[0] <= params[1] <= ... <= params[-1]``), each
    within ``[data_min, data_max]``.

    This used to be `scipy.optimize.minimize(method="SLSQP", constraints=...)`
    with explicit ``a <= b <= c <= d``-style inequality constraints.
    `optimizer_utils.optimizers_sub_solve`'s local search only supports box
    bounds, not general inequality constraints, so the ordering is folded into
    the parameterization instead: search over (first value, non-negative gaps
    between consecutive values) rather than the ordered values directly, which
    turns "non-decreasing" into plain per-gap box bounds. `to_ordered` also
    clips the reconstructed values into ``[data_min, data_max]`` -- a clip is
    monotonic, so it cannot undo the ordering the cumulative sum already
    guarantees.

    Returns ``(ordered_params, objective_value)``.
    """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    span = data_max - data_min if data_max > data_min else 1.0
    gaps0 = np.clip(np.diff(x0), 0.0, span)
    z0 = np.concatenate([[x0[0]], gaps0])

    def to_ordered(z):
        first = z[0]
        gaps = np.clip(z[1:], 0.0, None)
        raw = first + np.concatenate([[0.0], np.cumsum(gaps)])
        return np.clip(raw, data_min, data_max)

    def objective_z(z):
        return objective(to_ordered(z))

    bounds = [(data_min, data_max)] + [(0.0, span)] * (n - 1)
    result = optimizers_sub_solve(objective_z, z0, bounds)
    return to_ordered(result.x), float(result.fun)


def trapz_pdf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Compute normalized trapezoidal probability density function.

    The trapezoidal membership function is piecewise linear:
    - 0 outside [a, d]
    - Rises linearly from 0 to 1 over [a, b]
    - Constant 1 over [b, c]
    - Falls linearly from 1 to 0 over [c, d]

    The normalized PDF is: trapz_mf(x) / area
    where area = (b-a)/2 + (c-b) + (d-c)/2

    Zero-width trapezoids (where a == b == c == d) have no support and thus
    carry no mass; the PDF is zero everywhere.

    Args:
        x: Array of points at which to evaluate the PDF
        a, b, c, d: Trapezoid parameters (must satisfy a <= b <= c <= d)

    Returns:
        Array of normalized PDF values
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)

    # Compute area (normalization constant)
    area = (b - a) / 2 + (c - b) + (d - c) / 2
    if area == 0:
        # Zero-width trapezoid: no mass
        return y

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

    # Normalize by area
    y = y / area

    return y


def _init_trapz_from_histogram(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    n_components: int,
    data_min: float,
    data_max: float,
    shape: Shape = "trapezoid",
) -> tuple[list[tuple[float, float, float, float]], np.ndarray]:
    """Initialize trapezoid (or triangle) parameters from histogram peaks.

    Strategy:
    1. Smooth histogram with Gaussian kernel
    2. Find K peaks using scipy.signal.find_peaks
    3. For each peak, set [b,c] at half-power width
    4. Set [a,d] at valley edges or data boundaries
    5. Apply minimum-width guards
    6. If shape="triangle", collapse [b,c] to their midpoint -- a single
       apex -- so every downstream 4-tuple already satisfies b == c before
       the EM loop's first E-step.

    Args:
        bin_centers: Histogram bin centers
        bin_counts: Counts per bin
        n_components: Number of trapezoid components
        data_min, data_max: Data range for bounds
        shape: "trapezoid" (default) keeps the full [b,c] plateau; "triangle"
            collapses it to a single apex point.

    Returns:
        (params_list, weights) where params_list is [(a,b,c,d), ...] (with
        b == c when shape="triangle")
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

        if shape == "triangle":
            apex = (b + c) / 2
            b = c = apex

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
    shape: Shape = "trapezoid",
) -> list[tuple[float, float, float, float]]:
    """M-step for trapezoid (or triangle) parameters using constrained optimization.

    For each component k, minimize the negative weighted log-likelihood via
    `_solve_ordered_params`. This is the one place shape actually changes what
    gets optimized:
    a trapezoid has 4 free parameters (independent [b,c] plateau); a triangle
    has 3 (the plateau is a single apex, optimized directly rather than fit
    as two independent shoulders and averaged afterwards). Either way the
    result is handed back as a 4-tuple -- b == c for a triangle -- so every
    other function in this module (the E-step, the log-likelihood, the
    weight M-step) stays shape-agnostic.

    Args:
        bin_centers: Histogram bin centers
        bin_counts: Counts per bin
        responsibilities: Shape (n_bins, n_components)
        params_list: Current trapezoid parameters (b == c per component when
            shape="triangle")
        data_min, data_max: Data range for bounds
        shape: "trapezoid" (default) optimizes all 4 parameters independently;
            "triangle" optimizes (a, apex, d) and returns (a, apex, apex, d).

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
        # objective (called many times per iteration) is a single vectorized pass.
        coeff_k = responsibilities[:, k] * bin_counts

        if shape == "triangle":
            def objective(params, _coeff=coeff_k):
                a, apex, d = params
                pdf_vals = np.maximum(trapz_pdf(bin_centers, a, apex, apex, d), 1e-10)
                return -np.dot(_coeff, np.log(pdf_vals))

            x0 = np.array([a_k, b_k, d_k])  # b_k == c_k already, one apex value
            init_obj = objective(x0)
            solved, solved_obj = _solve_ordered_params(objective, x0, data_min, data_max)
            a_new, apex_new, d_new = solved if solved_obj <= init_obj else x0

            new_params.append((a_new, apex_new, apex_new, d_new))
            continue

        def objective(params, _coeff=coeff_k):
            a, b, c, d = params
            pdf_vals = np.maximum(trapz_pdf(bin_centers, a, b, c, d), 1e-10)
            return -np.dot(_coeff, np.log(pdf_vals))

        # Initial guess
        x0 = np.array([a_k, b_k, c_k, d_k])

        # Optimize: a <= b <= c <= d is enforced by `_solve_ordered_params`'s
        # gap reparametrization rather than explicit inequality constraints
        # (see that function's docstring -- this used to be
        # scipy.optimize.minimize(method="SLSQP", constraints=[...])).
        init_obj = objective(x0)
        solved, solved_obj = _solve_ordered_params(objective, x0, data_min, data_max)
        a_new, b_new, c_new, d_new = solved if solved_obj <= init_obj else x0

        new_params.append((a_new, b_new, c_new, d_new))

    return new_params


def fit_trapezoids_em(
    data_1d: np.ndarray,
    n_components: int,
    n_bins: int = 50,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    shape: Shape = "trapezoid",
) -> "tuple[list, np.ndarray, float]":
    """Run EM to fit a mixture of trapezoids -- or triangles -- to 1D data.

    Args:
        data_1d: 1D array of observations
        n_components: Number of components
        n_bins: Number of histogram bins
        max_iter: Maximum EM iterations
        tol: Convergence tolerance on log-likelihood relative change
        random_state: Random seed (currently unused, for API compatibility)
        shape: "trapezoid" (default) returns TrapezoidMembership objects with
            an independently-fit [b,c] plateau; "triangle" returns
            TriangularMembership objects (apex fit directly as one free
            parameter -- see :func:`_em_m_step_params`).

    Returns:
        (memberships, weights, log_likelihood): List of fitted membership
        objects (TrapezoidMembership or TriangularMembership per ``shape``),
        their mixing weights, and final log-likelihood
    """
    if random_state is not None:
        np.random.seed(random_state)

    data_1d = np.asarray(data_1d, dtype=float)
    data_min, data_max = data_1d.min(), data_1d.max()

    if data_min == data_max:
        # Degenerate case: all data identical
        mid = data_min
        if shape == "triangle":
            degenerate = TriangularMembership.create(mid, mid, mid)
        else:
            degenerate = TrapezoidMembership.create(mid, mid, mid, mid)
        return [degenerate], np.array([1.0]), 0.0

    # Compute histogram
    bin_counts, bin_edges = np.histogram(data_1d, bins=n_bins, range=(data_min, data_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initialize parameters from histogram peaks
    params_list, weights = _init_trapz_from_histogram(
        bin_centers, bin_counts, n_components, data_min, data_max, shape=shape
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
            bin_centers, bin_counts, responsibilities, params_list, data_min, data_max, shape=shape
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

    # Convert to membership objects
    if shape == "triangle":
        memberships = [
            TriangularMembership.create(a, b, d)
            for a, b, c, d in best_params
        ]
    else:
        memberships = [
            TrapezoidMembership.create(a, b, c, d)
            for a, b, c, d in best_params
        ]

    return memberships, best_weights, best_ll


def fit_trapezoid_mixture_1d(
    data_1d: np.ndarray,
    n_trapezoids: int = 0,
    max_components: int = 4,
    n_bins: int = 50,
    shape: Shape = "trapezoid",
) -> "tuple[list, int]":
    """Fit a 1-D trapezoid (or triangle) mixture, choosing the component count by BIC.

    Returns ``(memberships, n_selected)`` -- both, so the caller does not refit.
    :func:`find_optimal_trapezoids` used to run :func:`fit_trapezoids_em` at
    every candidate k, return only the winning *count*, and leave the caller to
    run the same EM again at that k. The winning fit was already in hand.

    BIC = n_params * log(N) - 2 * log_likelihood, with n_params = 5K - 1 for a
    trapezoid (4 parameters per component plus K-1 free weights) or 4K - 1 for
    a triangle (3 parameters per component -- one fewer, since the plateau is
    a single apex rather than an independent interval).
    """
    data_1d = np.asarray(data_1d, dtype=float)
    params_per_component = 3 if shape == "triangle" else 4

    if n_trapezoids > 0:
        memberships, _weights, _ll = fit_trapezoids_em(
            data_1d, n_components=n_trapezoids, n_bins=n_bins, max_iter=100, tol=1e-4, shape=shape
        )
        return memberships, n_trapezoids

    N = len(data_1d)
    best = (np.inf, [], 0)
    for k in range(1, max_components + 1):
        memberships, _weights, ll = fit_trapezoids_em(
            data_1d, n_components=k, n_bins=n_bins, max_iter=100, shape=shape
        )
        bic = ((params_per_component + 1) * k - 1) * np.log(N) - 2 * ll
        if bic < best[0]:
            best = (bic, memberships, k)

    return best[1], best[2]


def find_optimal_trapezoids(
    data_1d: np.ndarray,
    max_components: int = 4,
    n_bins: int = 50,
    shape: Shape = "trapezoid",
) -> int:
    """Number of trapezoid (or triangle) components the data supports, by BIC.

    Thin wrapper over :func:`fit_trapezoid_mixture_1d`, kept because it is
    public. Prefer that function directly: it returns the fit the count came
    from rather than discarding it.
    """
    return fit_trapezoid_mixture_1d(
        data_1d, n_trapezoids=0, max_components=max_components, n_bins=n_bins, shape=shape
    )[1]


def fit_trapezoids(
    X,
    y,
    column: str,
    label_value: int,
    n_trapezoids: int = 0,
    max_samples: int | None = None,
    random_state: int = 42,
    verbose: bool = False,
    shape: Shape = "trapezoid",
) -> list:
    """Fit multiple trapezoidal (or triangular) MFs to a single variable filtered by label.

    Analogue of gauss_math.fit_gaussians() but for trapezoids/triangles.

    Args:
        X: Feature dataframe
        y: Label series
        column: Column name to fit
        label_value: Class label to filter by
        n_trapezoids: Number of components (0 for automatic BIC selection)
        max_samples: Cap on the rows used for the fit; ``None`` -- the default --
            uses every row. When a cap is given the rows are drawn at random
            without replacement, seeded by ``random_state``. This defaulted to
            the first 20,000 rows, which is a biased sample on ordered data and
            an invisible one from the caller's side.
        random_state: Seeds the subsample draw.
        verbose: Print the automatically-selected component count.
        shape: "trapezoid" (default) or "triangle" -- see :func:`fit_trapezoids_em`.

    Returns:
        List of fitted membership objects (TrapezoidMembership or
        TriangularMembership per ``shape``)
    """
    data = X[column][y == label_value].dropna().values

    if max_samples is not None and 0 < max_samples < len(data):
        rng = np.random.default_rng(random_state)
        data = data[rng.choice(len(data), size=max_samples, replace=False)]

    if len(data) == 0:
        return []

    memberships, n_selected = fit_trapezoid_mixture_1d(
        data, n_trapezoids=n_trapezoids, max_components=4, shape=shape
    )
    if verbose and n_trapezoids <= 0:
        noun = "triangles" if shape == "triangle" else "trapezoids"
        print(f"  Automatically selected {n_selected} {noun} for {column} (label {label_value})")

    return memberships


def create_trapz_membership_dict(
    X,
    y,
    top_n_var_names: list[str],
    n_trapezoids: int | dict[str, int] = 0,
    max_samples: int | None = None,
    random_state: int = 42,
    verbose: bool = False,
    shape: Shape = "trapezoid",
) -> "GaussianMixtureModel":
    """Create a trapezoid (or triangle) membership model for top-n variables
    across all class labels.

    Analogue of gauss_math.create_gaussian_membership_dict() but uses
    trapezoids/triangles.

    Returns the same GaussianMixtureModel container type — the model is agnostic
    about whether LabelModel.memberships contains Gaussian, Trapezoid, or
    Triangular objects.

    Args:
        X: Feature dataframe
        y: Label series
        top_n_var_names: List of feature names to fit
        n_trapezoids: Number of components (0 for automatic, or dict per feature)
        max_samples: Rows per (feature, label) used for the fit; ``None`` uses
            all of them. See :func:`fit_trapezoids`.
        random_state: Seeds the subsample draw.
        verbose: Print each automatically-selected component count.
        shape: "trapezoid" (default) or "triangle" -- see :func:`fit_trapezoids_em`.

    Returns:
        GaussianMixtureModel containing fitted trapezoid/triangle MFs
    """
    from concurrent.futures.thread import ThreadPoolExecutor
    from .gauss_data import GaussianMixtureModel, FeatureModel, LabelModel

    unique_labels = y.unique()

    def process_feature(feature_name: str) -> tuple[str, FeatureModel]:
        """Process a single feature across all labels"""
        label_models = {}

        # Determine number of components for this feature
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

            trapz_params = fit_trapezoids(
                X,
                y,
                feature_name,
                label_value,
                feature_n_trapezoids,
                max_samples=max_samples,
                random_state=random_state,
                verbose=verbose,
                shape=shape,
            )
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
    """Fits a mixture of trapezoidal (or triangular) membership functions to
    1D histogram data.

    This class provides a simple API analogous to scikit-learn for fitting
    trapezoid/triangle MFs to 1D data using histogram-based EM.

    Attributes:
        n_components: Number of components (0 = auto-select via BIC)
        max_components: Maximum number of components for BIC search
        n_bins: Number of histogram bins
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
        random_state: Random seed
        shape: "trapezoid" (default) or "triangle" -- see :func:`fit_trapezoids_em`.

    After calling fit(), access:
        trapezoids_: List of fitted membership objects (TrapezoidMembership or
            TriangularMembership per ``shape``)
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
        shape: Shape = "trapezoid",
    ):
        self.n_components = n_components
        self.max_components = max_components
        self.n_bins = n_bins
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.shape = shape

        # Set after fit
        self.trapezoids_: Optional[list] = None
        self.weights_: Optional[np.ndarray] = None
        self.log_likelihood_: Optional[float] = None
        self.bic_: Optional[float] = None

    def fit(self, data_1d: np.ndarray) -> "TrapzMixtureModel":
        """Fit trapezoidal (or triangular) MFs to 1D data.

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
                data_1d, max_components=self.max_components, n_bins=self.n_bins, shape=self.shape
            )

        # Fit EM
        trapezoids, weights, ll = fit_trapezoids_em(
            data_1d,
            n_components=n_comp,
            n_bins=self.n_bins,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            shape=self.shape,
        )

        self.trapezoids_ = trapezoids
        self.weights_ = weights
        self.log_likelihood_ = ll

        # Compute BIC
        N = len(data_1d)
        K = len(trapezoids)
        params_per_component = 3 if self.shape == "triangle" else 4
        n_params = (params_per_component + 1) * K - 1
        self.bic_ = n_params * np.log(N) - 2 * ll

        return self


def triangle_pdf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Normalized triangular probability density function.

    Piecewise linear: 0 outside [a, c], rises linearly from 0 to 1 over [a, b],
    falls linearly from 1 to 0 over [b, c]. Apex ``b`` satisfies ``a <= b <= c``.

    This is exactly the trapezoidal PDF with its plateau collapsed to a point:
    ``triangle_pdf(x, a, b, c) == trapz_pdf(x, a, b, b, c)``.
    """
    return trapz_pdf(x, a, b, b, c)


def fit_triangles_em(
    data_1d: np.ndarray,
    n_components: int,
    n_bins: int = 50,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> tuple[list[TriangularMembership], np.ndarray, float]:
    """Run EM to fit a mixture of triangles to 1D data.

    Thin, triangle-named entry point over :func:`fit_trapezoids_em` --
    ``shape="triangle"`` pinned. See that function's docstring for the
    algorithm; there is no separate triangle implementation.
    """
    return fit_trapezoids_em(
        data_1d, n_components, n_bins=n_bins, max_iter=max_iter, tol=tol,
        random_state=random_state, shape="triangle",
    )


def fit_triangle_mixture_1d(
    data_1d: np.ndarray,
    n_triangles: int = 0,
    max_components: int = 4,
    n_bins: int = 50,
) -> tuple[list[TriangularMembership], int]:
    """Fit a 1-D triangle mixture, choosing the component count by BIC.

    Thin, triangle-named entry point over :func:`fit_trapezoid_mixture_1d`.
    """
    return fit_trapezoid_mixture_1d(
        data_1d, n_trapezoids=n_triangles, max_components=max_components, n_bins=n_bins, shape="triangle",
    )


def find_optimal_triangles(
    data_1d: np.ndarray,
    max_components: int = 4,
    n_bins: int = 50,
) -> int:
    """Number of triangle components the data supports, by BIC.

    Thin, triangle-named entry point over :func:`find_optimal_trapezoids`.
    """
    return find_optimal_trapezoids(data_1d, max_components=max_components, n_bins=n_bins, shape="triangle")


def fit_triangles(
    X,
    y,
    column: str,
    label_value: int,
    n_triangles: int = 0,
    max_samples: int | None = None,
    random_state: int = 42,
    verbose: bool = False,
) -> list[TriangularMembership]:
    """Fit multiple triangular MFs to a single variable filtered by label.

    Thin, triangle-named entry point over :func:`fit_trapezoids`.
    """
    return fit_trapezoids(
        X, y, column, label_value, n_trapezoids=n_triangles, max_samples=max_samples,
        random_state=random_state, verbose=verbose, shape="triangle",
    )


def create_triangle_membership_dict(
    X,
    y,
    top_n_var_names: list[str],
    n_triangles: int | dict[str, int] = 0,
    max_samples: int | None = None,
    random_state: int = 42,
    verbose: bool = False,
) -> "GaussianMixtureModel":
    """Create a triangle membership model for top-n variables across all class labels.

    Thin, triangle-named entry point over :func:`create_trapz_membership_dict`.
    """
    return create_trapz_membership_dict(
        X, y, top_n_var_names, n_trapezoids=n_triangles, max_samples=max_samples,
        random_state=random_state, verbose=verbose, shape="triangle",
    )


class TriangleMixtureModel(TrapzMixtureModel):
    """Fits a mixture of triangular membership functions to 1D histogram data.

    Thin, triangle-named subclass of :class:`TrapzMixtureModel` with
    ``shape="triangle"`` pinned; ``triangles_`` is an alias for the parent's
    ``trapezoids_`` attribute, since the underlying fit is identical.
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
        super().__init__(
            n_components=n_components, max_components=max_components, n_bins=n_bins,
            max_iter=max_iter, tol=tol, random_state=random_state, shape="triangle",
        )

    @property
    def triangles_(self) -> Optional[list[TriangularMembership]]:
        return self.trapezoids_

    @triangles_.setter
    def triangles_(self, value: Optional[list[TriangularMembership]]) -> None:
        self.trapezoids_ = value
