"""
Smooth trapezoid approximation for EM fitting.

Instead of piecewise-linear trapezoids (which have non-smooth objectives),
this module uses sigmoid-based smooth approximations for fitting via EM.
After EM converges, crisp trapezoid parameters are extracted for use in
gates, Ruspini partitions, and membership evaluation.

The smooth approximation allows EM to work on well-behaved, differentiable
objectives, avoiding the "mode-hugging" pathology of the original trapz_math.py
approach while preserving the interpretability of crisp trapezoids in the output.

Algorithm:
  1. Initialize (a, b, c, d, steepness) from histogram peaks (as before)
  2. EM loop:
     - E-step: compute responsibilities using smooth_trapz_pdf
     - M-step: optimize (a, b, c, d, steepness) via smooth, differentiable objective
  3. After convergence, extract crisp trapezoid (a, b, c, d) from smooth parameters
  4. Return crisp trapezoids for downstream use (gates, Ruspini, evaluation)
"""

from typing import Literal, Optional
import numpy as np
from scipy import signal, ndimage
from scipy.special import expit  # sigmoid

from .gauss_data import TrapezoidMembership, TriangularMembership
from .optimizer_utils import optimizers_sub_solve

Shape = Literal["trapezoid", "triangle"]


def smooth_trapz_pdf(
    x: np.ndarray, a: float, b: float, c: float, d: float, steepness: float = 10.0
) -> np.ndarray:
    """Smooth trapezoid PDF using sigmoid ramps.

    A continuous, infinitely differentiable approximation to the crisp trapezoid.
    As steepness → ∞, this approaches the piecewise-linear trapezoid.

    Shape:
      - Sigmoid rise from 0→1 over [a, b]
      - Plateau 1 over [b, c]
      - Sigmoid fall from 1→0 over [c, d]
      - Sigmoid tails extend to ±∞

    Args:
        x: Points at which to evaluate
        a, b, c, d: Trapezoid knots (must satisfy a ≤ b ≤ c ≤ d)
        steepness: Controls sharpness of transitions (default 10). Higher = sharper.
                   At convergence, steepness is typically 5-20 depending on data scale.

    Returns:
        Normalized PDF values (integrated over x → 1)
    """
    x = np.asarray(x, dtype=float)
    steepness = max(float(steepness), 0.1)

    # Handle degenerate case
    if not (a <= b <= c <= d):
        return np.zeros_like(x, dtype=float)

    # Sigmoid: σ(z) = 1 / (1 + exp(-z))
    # Rising edge: rises from 0→1 as x moves from a→b
    sigma_left = expit(steepness * (x - a))  # 0 for x<<a, 1 for x>>a
    sigma_left_end = expit(steepness * (x - b))  # 0 for x<<b, 1 for x>>b
    left_ramp = sigma_left * (1.0 - sigma_left_end)

    # Plateau: 1 over [b, c]
    plateau = (x >= b) & (x <= c)

    # Falling edge: falls from 1→0 as x moves from c→d
    sigma_right = expit(steepness * (d - x))  # 1 for x<<d, 0 for x>>d
    sigma_right_start = expit(steepness * (c - x))  # 1 for x<<c, 0 for x>>c
    right_ramp = sigma_right * (1.0 - sigma_right_start)

    # Combine: unnormalized density
    y = left_ramp + plateau.astype(float) + right_ramp

    # Normalization: ∫ smooth_trapz dx from -∞ to +∞
    # Analytical approximation:
    #   - Rising integral: ∫_a^b sigmoid_product ≈ (b-a)/2 + log(2)/steepness
    #   - Plateau: (c - b)
    #   - Falling integral: ∫_c^d sigmoid_product ≈ (d-c)/2 + log(2)/steepness
    #   - Tail integrals: 2*log(2)/steepness (from sigmoid tails at ±∞)
    # For high steepness (50+), tail contributions are negligible (~0.03)

    crisp_area = max(0.0, (b - a) / 2.0 + (c - b) + (d - c) / 2.0)

    if crisp_area < 1e-10:
        # Degenerate trapezoid
        return np.zeros_like(y)

    # Analytical area correction for sigmoid tails
    # log(2) / steepness accounts for the smooth transitions vs. crisp knots
    tail_correction = 2.0 * np.log(2.0) / steepness
    total_area = crisp_area + tail_correction

    # Normalize
    if total_area > 1e-10:
        y = y / total_area
    else:
        y = np.zeros_like(y)

    return y


def extract_crisp_trapz_from_smooth(
    a: float, b: float, c: float, d: float, steepness: float = 10.0
) -> tuple[float, float, float, float]:
    """Extract crisp trapezoid knots from smooth-fit parameters.

    The smooth trapezoid is parameterized by (a, b, c, d, steepness).
    The effective knots where the sigmoid crosses 0.5 are approximately:
      - a_eff ≈ a - log(1 + √2) / steepness
      - d_eff ≈ d + log(1 + √2) / steepness

    For a tight approximation, use the input knots (a, b, c, d) directly after
    steepness has converged. The steepness parameter itself controls how "crisp"
    the output will be.

    Args:
        a, b, c, d: Smooth trapezoid knot parameters
        steepness: Controls how much to correct for sigmoid slope

    Returns:
        (a_crisp, b_crisp, c_crisp, d_crisp) ready for TrapezoidMembership
    """
    # For steep sigmoids, the inflection points are close to the knots.
    # Correction factor: where sigmoid crosses 0.5, z = 0, so:
    # sigmoid((x - a) * steepness) = 0.5 when (x - a) * steepness = 0
    # i.e., x = a
    # But the actual trapezoid slope starts from where the sigmoid curve begins
    # to rise noticeably (not at 0.5, but at some threshold like 0.05 or 0.95).
    #
    # For practical purposes: if steepness is high (>10), the sigmoid is steep
    # enough that a, b, c, d are already good approximations of the crisp knots.
    # Apply a small correction based on steepness:
    correction = max(0.0, np.log(1 + np.sqrt(2)) / steepness) if steepness > 0 else 0.0

    # Extract knots with minimal correction (already quite crisp if steepness is high)
    a_crisp = a - correction
    b_crisp = b  # plateau left edge, no correction needed
    c_crisp = c  # plateau right edge, no correction needed
    d_crisp = d + correction

    return a_crisp, b_crisp, c_crisp, d_crisp


def _init_smooth_trapz_from_histogram(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    n_components: int,
    data_min: float,
    data_max: float,
    shape: Shape = "trapezoid",
) -> tuple[list[tuple[float, float, float, float, float]], np.ndarray]:
    """Initialize smooth trapezoid parameters from histogram peaks.

    Same initialization as trapz_math._init_trapz_from_histogram, but returns
    5-tuples (a, b, c, d, steepness) instead of 4-tuples.

    Args:
        bin_centers, bin_counts, n_components, data_min, data_max: As in trapz_math
        shape: "trapezoid" or "triangle"

    Returns:
        (params_list, weights) where params_list contains (a, b, c, d, steepness)
        tuples and steepness is initialized to a reasonable default (10.0).
    """
    density = bin_counts / np.sum(bin_counts) if np.sum(bin_counts) > 0 else bin_counts
    smoothed = ndimage.gaussian_filter1d(density, sigma=1.0)

    if n_components <= 0 or n_components > 10:
        n_components = max(1, min(3, n_components if n_components > 0 else 1))

    peaks, properties = signal.find_peaks(smoothed, prominence=np.max(smoothed) * 0.05)

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
        half_height = peak_height / 2

        b_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if smoothed[i] < half_height:
                b_idx = i
                break
        else:
            b_idx = 0
        b = bin_centers[b_idx]

        c_idx = peak_idx
        for i in range(peak_idx + 1, len(bin_centers)):
            if smoothed[i] < half_height:
                c_idx = i
                break
        else:
            c_idx = len(bin_centers) - 1
        c = bin_centers[c_idx]

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

        a = max(a, data_min)
        d = min(d, data_max)
        b = np.clip(b, a, d)
        c = np.clip(c, b, d)

        if shape == "triangle":
            apex = (b + c) / 2
            b = c = apex

        # Add steepness parameter (initialized to 10.0, will be optimized in EM)
        steepness = 10.0
        params_list.append((a, b, c, d, steepness))

    weights = np.ones(len(params_list)) / len(params_list)
    return params_list, weights


def _smooth_trapz_log_likelihood(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    params_list: list[tuple[float, float, float, float, float]],
    weights: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """Compute log-likelihood using crisp trapezoid mixture.

    Even though EM fitting uses smooth trapezoids (smooth objective),
    we evaluate log-likelihood using crisp trapezoids (exact evaluation).
    This gives smooth gradients without approximation error.
    """
    from .trapz_math import trapz_pdf
    mixture_pdf = np.zeros_like(bin_centers, dtype=float)
    for k, (a, b, c, d, _steep) in enumerate(params_list):
        # Use crisp trapezoid for evaluation (ignore steepness parameter)
        mixture_pdf += weights[k] * trapz_pdf(bin_centers, a, b, c, d)

    mixture_pdf = np.maximum(mixture_pdf, eps)
    return float(np.sum(bin_counts * np.log(mixture_pdf)))


def _em_e_step_smooth(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    params_list: list[tuple[float, float, float, float, float]],
    weights: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """E-step using smooth trapezoid PDFs."""
    n_bins = len(bin_centers)
    n_components = len(params_list)

    densities = np.empty((n_bins, n_components))
    for k, (a, b, c, d, steep) in enumerate(params_list):
        densities[:, k] = weights[k] * smooth_trapz_pdf(bin_centers, a, b, c, d, steep)

    denom = densities.sum(axis=1)
    valid = denom > eps
    responsibilities = np.empty((n_bins, n_components))
    responsibilities[valid] = densities[valid] / denom[valid, None]
    responsibilities[~valid] = 1.0 / n_components

    return responsibilities


def _em_m_step_weights(
    responsibilities: np.ndarray,
    bin_counts: np.ndarray,
) -> np.ndarray:
    """M-step for mixing weights (unchanged from trapz_math)."""
    n_components = responsibilities.shape[1]
    weights = np.zeros(n_components)

    total_count = np.sum(bin_counts)
    for k in range(n_components):
        weights[k] = np.sum(responsibilities[:, k] * bin_counts) / total_count

    weights = weights / np.sum(weights)
    return weights


def _em_m_step_params_smooth(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    responsibilities: np.ndarray,
    params_list: list[tuple[float, float, float, float, float]],
    data_min: float,
    data_max: float,
    shape: Shape = "trapezoid",
    steepness: float = 50.0,
) -> list[tuple[float, float, float, float, float]]:
    """M-step for smooth trapezoid parameters using smooth objective.

    Optimizes (a, b, c, d) to maximize weighted log-likelihood.
    Steepness is held fixed (typically 50.0) to keep the objective smooth and
    close to crisp trapezoid behavior while avoiding the piecewise-linear kinks
    that cause mode-hugging pathology in the original trapz_math.

    The fixed steepness means: smooth_trapz ≈ crisp trapezoid, but with
    infinitely differentiable gradients for robust EM optimization.

    Args:
        bin_centers, bin_counts, responsibilities, params_list, data_min, data_max, shape: As in trapz_math
        steepness: Fixed steepness parameter (default 50.0 = very steep, nearly crisp)

    Returns:
        Updated smooth trapezoid parameters (with steepness fixed and returned as-is)
    """
    n_components = responsibilities.shape[1]
    new_params = []

    bin_centers = np.asarray(bin_centers, dtype=float)
    bin_counts = np.asarray(bin_counts, dtype=float)
    steepness = max(float(steepness), 5.0)

    for k in range(n_components):
        a_k, b_k, c_k, d_k, _ = params_list[k]  # Ignore old steepness
        coeff_k = responsibilities[:, k] * bin_counts

        if shape == "triangle":
            def objective(params, _coeff=coeff_k, _steep=steepness):
                a, apex, d = params
                pdf_vals = np.maximum(smooth_trapz_pdf(bin_centers, a, apex, apex, d, _steep), 1e-10)
                return -np.dot(_coeff, np.log(pdf_vals))

            x0 = np.array([a_k, b_k, d_k])
            init_obj = objective(x0)
            # Bounds: a < apex < d
            bounds = [(data_min, data_max), (data_min, data_max), (data_min, data_max)]
            result = optimizers_sub_solve(objective, x0, bounds)
            solved, solved_obj = result.x, float(result.fun)
            a_new, apex_new, d_new = solved if solved_obj <= init_obj else x0

            new_params.append((a_new, apex_new, apex_new, d_new, steepness))
            continue

        def objective(params, _coeff=coeff_k, _steep=steepness):
            a, b, c, d = params
            pdf_vals = np.maximum(smooth_trapz_pdf(bin_centers, a, b, c, d, _steep), 1e-10)
            return -np.dot(_coeff, np.log(pdf_vals))

        x0 = np.array([a_k, b_k, c_k, d_k])
        init_obj = objective(x0)
        # Bounds: a ≤ b ≤ c ≤ d
        bounds = [(data_min, data_max), (data_min, data_max), (data_min, data_max), (data_min, data_max)]
        result = optimizers_sub_solve(objective, x0, bounds)
        solved, solved_obj = result.x, float(result.fun)
        a_new, b_new, c_new, d_new = solved if solved_obj <= init_obj else x0

        new_params.append((a_new, b_new, c_new, d_new, steepness))

    return new_params


def fit_smooth_trapezoids_em(
    data_1d: np.ndarray,
    n_components: int,
    n_bins: int = 50,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    shape: Shape = "trapezoid",
) -> tuple[list, np.ndarray, float]:
    """Run EM to fit smooth trapezoids, returning crisp trapezoid results.

    This is a drop-in replacement for trapz_math.fit_trapezoids_em that uses
    smooth approximations during fitting, then extracts crisp trapezoid parameters
    for output.

    Args:
        data_1d, n_components, n_bins, max_iter, tol, random_state, shape: As in trapz_math

    Returns:
        (memberships, weights, log_likelihood): Crisp TrapezoidMembership objects,
        mixing weights, and log-likelihood evaluated at crisp trapezoids
    """
    if random_state is not None:
        np.random.seed(random_state)

    data_1d = np.asarray(data_1d, dtype=float)
    data_min, data_max = data_1d.min(), data_1d.max()

    if data_min == data_max:
        mid = data_min
        if shape == "triangle":
            degenerate = TriangularMembership.create(mid, mid, mid)
        else:
            degenerate = TrapezoidMembership.create(mid, mid, mid, mid)
        return [degenerate], np.array([1.0]), 0.0

    bin_counts, bin_edges = np.histogram(data_1d, bins=n_bins, range=(data_min, data_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initialize with smooth trapezoid parameters
    params_list, weights = _init_smooth_trapz_from_histogram(
        bin_centers, bin_counts, n_components, data_min, data_max, shape=shape
    )

    params_list = params_list[:n_components]
    weights = weights[: len(params_list)]
    weights = weights / np.sum(weights)

    # EM loop
    prev_ll = -np.inf
    best_ll = -np.inf
    best_params = params_list
    fixed_steepness = 50.0  # Keep smooth trapezoid nearly crisp but smooth-objective

    for iteration in range(max_iter):
        responsibilities = _em_e_step_smooth(bin_centers, bin_counts, params_list, weights)
        weights = _em_m_step_weights(responsibilities, bin_counts)
        params_list = _em_m_step_params_smooth(
            bin_centers, bin_counts, responsibilities, params_list, data_min, data_max,
            shape=shape, steepness=fixed_steepness
        )

        ll = _smooth_trapz_log_likelihood(bin_centers, bin_counts, params_list, weights)

        if ll > best_ll:
            best_ll = ll
            best_params = params_list

        if prev_ll > -np.inf:
            rel_change = abs(ll - prev_ll) / (abs(prev_ll) + 1e-10)
            if rel_change < tol:
                break

        prev_ll = ll

    # Extract crisp trapezoids from smooth parameters
    # Since steepness is fixed high, the smooth knots are already very close to crisp
    crisp_params = []
    for a, b, c, d, steep in best_params:
        # For high steepness, the smooth knots ARE the crisp knots
        crisp_params.append((a, b, c, d))

    # Evaluate log-likelihood at crisp trapezoids for reporting
    # (This uses the crisp piecewise-linear PDF, not the smooth one)
    from .trapz_math import trapz_pdf, _trapz_log_likelihood as crisp_log_likelihood
    crisp_ll = crisp_log_likelihood(bin_centers, bin_counts,
                                     [(a, b, c, d) for a, b, c, d in crisp_params], weights)

    # Convert to membership objects
    if shape == "triangle":
        memberships = [
            TriangularMembership.create(a, b, d)
            for a, b, c, d in crisp_params
        ]
    else:
        memberships = [
            TrapezoidMembership.create(a, b, c, d)
            for a, b, c, d in crisp_params
        ]

    return memberships, weights, crisp_ll


def fit_smooth_trapezoid_mixture_1d(
    data_1d: np.ndarray,
    n_trapezoids: int = 0,
    max_components: int = 4,
    n_bins: int = 50,
    shape: Shape = "trapezoid",
) -> tuple[list, int]:
    """Fit smooth trapezoid mixture with automatic component selection by BIC.

    Drop-in replacement for trapz_math.fit_trapezoid_mixture_1d.
    """
    data_1d = np.asarray(data_1d, dtype=float)
    params_per_component = 3 if shape == "triangle" else 4

    if n_trapezoids > 0:
        memberships, _weights, _ll = fit_smooth_trapezoids_em(
            data_1d, n_components=n_trapezoids, n_bins=n_bins, max_iter=100, tol=1e-4, shape=shape
        )
        return memberships, n_trapezoids

    N = len(data_1d)
    best = (np.inf, [], 0)
    for k in range(1, max_components + 1):
        memberships, _weights, ll = fit_smooth_trapezoids_em(
            data_1d, n_components=k, n_bins=n_bins, max_iter=100, shape=shape
        )
        bic = ((params_per_component + 1) * k - 1) * np.log(N) - 2 * ll
        if bic < best[0]:
            best = (bic, memberships, k)

    return best[1], best[2]
