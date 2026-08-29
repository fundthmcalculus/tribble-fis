"""
Smooth trapezoid approximation for EM fitting.

Instead of piecewise-linear trapezoids (whose area-normalized-density M-step
objective structurally rewards shrinking the support onto the data mode --
the "mode-hugging" pathology described in `docs/antecedent-membership-function-evaluation.md`
and `.github/ISSUE_163_RESOLUTION_PLAN.md`), this module fits a genuinely
differentiable relaxation of the trapezoid shape and anneals it toward the
crisp shape as EM converges, while carrying the same `width_reg` support-width
regularizer as `trapz_math.py` to counter the mode-collapse incentive itself
-- smoothing the optimization landscape alone does not remove that incentive
(see the module docstring history below), so this module treats the two as
separate problems and addresses both.

Revision note: the original version of this module (#195) built its "smooth
ramp" out of a product of two full-height sigmoid bumps
(`sigma_left * (1 - sigma_left_end)`), which is a smoothed *indicator of the
whole [a, b] interval*, not a rising ramp from 0 to 1 -- it does not reduce to
a trapezoid shape at any steepness. Its declared normalization
(`(b-a)/2 + (c-b) + (d-c)/2 + 2*log(2)/steepness`) was correspondingly wrong:
integrating the actual unnormalized shape numerically gives a value close to
`d - a` (a boxcar over the whole support), not the trapezoid's true area --
confirmed by direct quadrature during this revision (errors of 30-50%, not
the few-percent sigmoid-tail correction the formula implied). That bug means
the #195 prototype's "10-30% worse quality" finding was measured against a
mixture of *box* densities, not smooth trapezoids, so it does not settle
whether a correctly-built smooth relaxation can close the gap.

This revision replaces the shape function with softplus-based smoothed ramps
(`_rising_ramp`/`_falling_ramp` below) that are the actual smoothed
`clip((x-a)/(b-a), 0, 1)` -- linear in the middle, rounded only at the two
corners, converging to the crisp ramp pointwise as steepness -> infinity --
combined via `min` exactly as a crisp trapezoid combines its two ramps.
Normalization is computed by numerical quadrature over the shape's own
support (`_smooth_trapz_area`) rather than a closed-form guess, so it is
always correct regardless of steepness or how narrow a component is.

Algorithm:
  1. Initialize (a, b, c, d) from histogram peaks (as in trapz_math.py)
  2. EM loop, steepness annealed from `steepness_start` (a soft, wide-basin
     relaxation early on) to `steepness_end` (near-crisp) on a geometric
     schedule over `max_iter` iterations:
     - E-step: responsibilities from `smooth_trapz_pdf` at the current steepness
     - M-step: optimize (a, b, c, d) -- ordering-constrained via the same
       `_solve_ordered_params` reparametrization `trapz_math.py` uses -- against
       the smooth, differentiable-in-shape objective, with the same
       `width_reg` penalty against support collapse
  3. Return the converged (a, b, c, d) directly as crisp trapezoid knots --
     no post-hoc "extraction" correction is needed, because the smooth ramps
     are already anchored exactly at those knots (see the docstring above).
"""

from typing import Literal, Optional

import numpy as np
from scipy import signal, ndimage

from .gauss_data import TrapezoidMembership, TriangularMembership
from .optimizer_utils import optimizers_sub_solve
from .trapz_math import _solve_ordered_params, _init_trapz_from_histogram

Shape = Literal["trapezoid", "triangle"]

_MIN_RAMP_WIDTH = 1e-9


def _softplus(z: np.ndarray) -> np.ndarray:
    """Numerically stable ``log(1 + exp(z))``."""
    return np.logaddexp(0.0, z)


def _rising_ramp(x: np.ndarray, a: float, b: float, steepness: float) -> np.ndarray:
    """Smooth version of ``clip((x - a) / (b - a), 0, 1)``.

    Linear in ``(a, b)``, rounded only at the two corners (width ~
    ``1/steepness``); converges pointwise to the crisp ramp as
    ``steepness -> infinity``. ``b <= a`` degenerates to a step at ``a``.
    """
    width = b - a
    if width <= _MIN_RAMP_WIDTH:
        return np.where(x > a, 1.0, np.where(x < a, 0.0, 0.5))
    return (_softplus(steepness * (x - a)) - _softplus(steepness * (x - b))) / (steepness * width)


def _falling_ramp(x: np.ndarray, c: float, d: float, steepness: float) -> np.ndarray:
    """Smooth version of ``clip((d - x) / (d - c), 0, 1)``; see :func:`_rising_ramp`."""
    width = d - c
    if width <= _MIN_RAMP_WIDTH:
        return np.where(x < c, 1.0, np.where(x > c, 0.0, 0.5))
    return (_softplus(steepness * (d - x)) - _softplus(steepness * (c - x))) / (steepness * width)


def _smooth_trapz_shape(
    x: np.ndarray, a: float, b: float, c: float, d: float, steepness: float
) -> np.ndarray:
    """Unnormalized smooth trapezoid shape: the pointwise minimum of the
    rising and falling ramps, exactly as a crisp trapezoid combines them.
    """
    return np.minimum(
        _rising_ramp(x, a, b, steepness), _falling_ramp(x, c, d, steepness)
    )


def _smooth_trapz_area(a: float, b: float, c: float, d: float, steepness: float, n_grid: int = 161) -> float:
    """Normalization constant: ``integral(shape(x) dx)`` by quadrature.

    The shape has no closed-form antiderivative once combined via ``min``, so
    this integrates numerically over a grid wide enough to capture the
    sigmoid tails on both sides (``~20/steepness`` past each end), using
    `numpy.trapezoid` -- cheap (a couple hundred points) and accurate because
    the integrand is smooth by construction.
    """
    span = max(d - a, 1e-9)
    pad = max(20.0 / max(steepness, 1e-6), span * 0.05)
    lo, hi = a - pad, d + pad
    if hi <= lo:
        hi = lo + 1e-6
    xs = np.linspace(lo, hi, n_grid)
    ys = _smooth_trapz_shape(xs, a, b, c, d, steepness)
    return float(np.trapezoid(ys, xs))


def smooth_trapz_pdf(
    x: np.ndarray, a: float, b: float, c: float, d: float, steepness: float = 10.0
) -> np.ndarray:
    """Normalized smooth trapezoid PDF.

    A continuous, differentiable-in-shape approximation to the crisp
    trapezoid PDF (:func:`tribblefis.trapz_math.trapz_pdf`): linear ramps with
    corners rounded over a window of width ``~1/steepness``, converging to it
    pointwise as ``steepness -> infinity``. See the module docstring for why
    this differs from the original (#195) formula.
    """
    x = np.asarray(x, dtype=float)
    steepness = max(float(steepness), 0.1)

    if not (a <= b <= c <= d):
        return np.zeros_like(x, dtype=float)

    area = _smooth_trapz_area(a, b, c, d, steepness)
    if area < 1e-10:
        return np.zeros_like(x, dtype=float)

    return _smooth_trapz_shape(x, a, b, c, d, steepness) / area


def _init_smooth_trapz_from_histogram(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    n_components: int,
    data_min: float,
    data_max: float,
    shape: Shape = "trapezoid",
) -> tuple[list[tuple[float, float, float, float]], np.ndarray]:
    """Same histogram-peak initialization as
    :func:`tribblefis.trapz_math._init_trapz_from_histogram` -- there is
    nothing smoothing-specific about picking the starting knots, only about
    how they are refined, so this is a thin pass-through kept for import
    compatibility.
    """
    return _init_trapz_from_histogram(
        bin_centers, bin_counts, n_components, data_min, data_max, shape=shape
    )


def _smooth_trapz_log_likelihood(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    params_list: list[tuple[float, float, float, float]],
    weights: np.ndarray,
    steepness: float,
    eps: float = 1e-10,
) -> float:
    """Log-likelihood of the histogram under the smooth mixture at the
    current annealing steepness (comparable across iterations only at a
    fixed steepness -- see the EM loop, which tracks the best crisp-evaluated
    likelihood instead for model selection).
    """
    mixture_pdf = np.zeros_like(bin_centers, dtype=float)
    for k, (a, b, c, d) in enumerate(params_list):
        mixture_pdf += weights[k] * smooth_trapz_pdf(bin_centers, a, b, c, d, steepness)
    mixture_pdf = np.maximum(mixture_pdf, eps)
    return float(np.sum(bin_counts * np.log(mixture_pdf)))


def _em_e_step_smooth(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    params_list: list[tuple[float, float, float, float]],
    weights: np.ndarray,
    steepness: float,
    eps: float = 1e-10,
) -> np.ndarray:
    """E-step using the smooth trapezoid PDF at the current steepness."""
    n_bins = len(bin_centers)
    n_components = len(params_list)

    densities = np.empty((n_bins, n_components))
    for k, (a, b, c, d) in enumerate(params_list):
        densities[:, k] = weights[k] * smooth_trapz_pdf(bin_centers, a, b, c, d, steepness)

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
    """M-step for mixing weights (identical to `trapz_math.py`)."""
    n_components = responsibilities.shape[1]
    weights = np.zeros(n_components)
    total_count = np.sum(bin_counts)
    for k in range(n_components):
        weights[k] = np.sum(responsibilities[:, k] * bin_counts) / total_count
    return weights / np.sum(weights)


def _em_m_step_params_smooth(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    responsibilities: np.ndarray,
    params_list: list[tuple[float, float, float, float]],
    data_min: float,
    data_max: float,
    steepness: float,
    shape: Shape = "trapezoid",
    width_reg: float = 0.0,
) -> list[tuple[float, float, float, float]]:
    """M-step for smooth trapezoid (or triangle) parameters.

    Mirrors `trapz_math._em_m_step_params`: ordering (``a <= b <= c <= d``)
    is enforced via the same `_solve_ordered_params` gap-reparametrization
    (the original #195 M-step used plain independent box bounds per
    parameter, so nothing stopped e.g. ``b > c`` -- an invalid trapezoid that
    `smooth_trapz_pdf`'s ``a <= b <= c <= d`` guard would then just zero out,
    silently discarding the step). ``width_reg`` penalizes narrow support the
    same way, since annealing the steepness fixes the optimization surface,
    not the density-normalization incentive to shrink support onto the mode.
    """
    n_components = responsibilities.shape[1]
    new_params = []

    bin_centers = np.asarray(bin_centers, dtype=float)
    bin_counts = np.asarray(bin_counts, dtype=float)

    for k in range(n_components):
        a_k, b_k, c_k, d_k = params_list[k]
        coeff_k = responsibilities[:, k] * bin_counts
        _tot = float(coeff_k.sum())

        if shape == "triangle":
            def objective(params, _coeff=coeff_k, _tot=_tot, _steep=steepness):
                a, apex, d = params
                pdf_vals = np.maximum(smooth_trapz_pdf(bin_centers, a, apex, apex, d, _steep), 1e-10)
                nll = -np.dot(_coeff, np.log(pdf_vals))
                if width_reg:
                    nll -= width_reg * _tot * np.log(max(d - a, 1e-6))
                return nll

            x0 = np.array([a_k, b_k, d_k])
            init_obj = objective(x0)
            solved, solved_obj = _solve_ordered_params(objective, x0, data_min, data_max)
            a_new, apex_new, d_new = solved if solved_obj <= init_obj else x0
            new_params.append((a_new, apex_new, apex_new, d_new))
            continue

        def objective(params, _coeff=coeff_k, _tot=_tot, _steep=steepness):
            a, b, c, d = params
            pdf_vals = np.maximum(smooth_trapz_pdf(bin_centers, a, b, c, d, _steep), 1e-10)
            nll = -np.dot(_coeff, np.log(pdf_vals))
            if width_reg:
                nll -= width_reg * _tot * np.log(max(d - a, 1e-6))
            return nll

        x0 = np.array([a_k, b_k, c_k, d_k])
        init_obj = objective(x0)
        solved, solved_obj = _solve_ordered_params(objective, x0, data_min, data_max)
        a_new, b_new, c_new, d_new = solved if solved_obj <= init_obj else x0
        new_params.append((a_new, b_new, c_new, d_new))

    return new_params


def _steepness_schedule(max_iter: int, steepness_start: float, steepness_end: float) -> np.ndarray:
    """Geometric anneal from `steepness_start` to `steepness_end` over
    `max_iter` iterations -- a soft, wide-basin objective early (to move
    knots past bad local optima the crisp piecewise-linear objective cannot
    escape) tightening toward the crisp trapezoid by the last iteration.
    """
    if max_iter <= 1:
        return np.array([steepness_end])
    ratio = steepness_end / steepness_start
    return steepness_start * ratio ** (np.arange(max_iter) / (max_iter - 1))


def fit_smooth_trapezoids_em(
    data_1d: np.ndarray,
    n_components: int,
    n_bins: int = 50,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    shape: Shape = "trapezoid",
    width_reg: float = 0.0,
    steepness_start: float = 4.0,
    steepness_end: float = 60.0,
) -> tuple[list, np.ndarray, float]:
    """Run EM with an annealed smooth-trapezoid relaxation, returning crisp
    trapezoid (or triangle) results.

    Drop-in replacement for `trapz_math.fit_trapezoids_em` with the same
    return shape. `steepness` anneals geometrically from `steepness_start`
    to `steepness_end` across `max_iter` iterations (see
    :func:`_steepness_schedule`); pass `steepness_start == steepness_end` to
    fit at one fixed steepness throughout (the #195 behavior, with the
    shape/ordering/normalization fixes still applied).

    Returns:
        (memberships, weights, log_likelihood): crisp TrapezoidMembership (or
        TriangularMembership) objects, mixing weights, and the log-likelihood
        of the best iterate evaluated under the *crisp* trapz_pdf (so it is
        directly comparable to `trapz_math.fit_trapezoids_em`'s output and to
        BIC-based model selection).
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

    params_list, weights = _init_smooth_trapz_from_histogram(
        bin_centers, bin_counts, n_components, data_min, data_max, shape=shape
    )
    params_list = params_list[:n_components]
    weights = weights[: len(params_list)]
    weights = weights / np.sum(weights)

    from .trapz_math import _trapz_log_likelihood as crisp_log_likelihood

    schedule = _steepness_schedule(max_iter, steepness_start, steepness_end)

    prev_ll = -np.inf
    best_ll = -np.inf
    best_params = params_list

    for iteration in range(max_iter):
        steepness = float(schedule[iteration])

        responsibilities = _em_e_step_smooth(bin_centers, bin_counts, params_list, weights, steepness)
        weights = _em_m_step_weights(responsibilities, bin_counts)
        params_list = _em_m_step_params_smooth(
            bin_centers, bin_counts, responsibilities, params_list, data_min, data_max,
            steepness, shape=shape, width_reg=width_reg,
        )

        # Track the best iterate by its *crisp* likelihood: steepness changes
        # across iterations, so the smooth likelihood alone is not comparable
        # iteration-to-iteration, but the crisp evaluation always is.
        crisp_ll = crisp_log_likelihood(bin_centers, bin_counts, params_list, weights)
        if crisp_ll > best_ll:
            best_ll = crisp_ll
            best_params = params_list

        smooth_ll = _smooth_trapz_log_likelihood(bin_centers, bin_counts, params_list, weights, steepness)
        if prev_ll > -np.inf:
            rel_change = abs(smooth_ll - prev_ll) / (abs(prev_ll) + 1e-10)
            if rel_change < tol and steepness >= steepness_end * 0.999:
                break
        prev_ll = smooth_ll

    if shape == "triangle":
        memberships = [TriangularMembership.create(a, b, d) for a, b, c, d in best_params]
    else:
        memberships = [TrapezoidMembership.create(a, b, c, d) for a, b, c, d in best_params]

    return memberships, weights, best_ll


def fit_smooth_trapezoid_mixture_1d(
    data_1d: np.ndarray,
    n_trapezoids: int = 0,
    max_components: int = 4,
    n_bins: int = 50,
    shape: Shape = "trapezoid",
    width_reg: float = 0.0,
) -> tuple[list, int]:
    """Fit a smooth-EM trapezoid mixture with automatic component selection
    by BIC. Drop-in replacement for `trapz_math.fit_trapezoid_mixture_1d`.
    """
    data_1d = np.asarray(data_1d, dtype=float)
    params_per_component = 3 if shape == "triangle" else 4

    if n_trapezoids > 0:
        memberships, _weights, _ll = fit_smooth_trapezoids_em(
            data_1d, n_components=n_trapezoids, n_bins=n_bins, max_iter=100, tol=1e-4,
            shape=shape, width_reg=width_reg,
        )
        return memberships, n_trapezoids

    N = len(data_1d)
    best = (np.inf, [], 0)
    for k in range(1, max_components + 1):
        memberships, _weights, ll = fit_smooth_trapezoids_em(
            data_1d, n_components=k, n_bins=n_bins, shape=shape, width_reg=width_reg,
        )
        bic = ((params_per_component + 1) * k - 1) * np.log(N) - 2 * ll
        if bic < best[0]:
            best = (bic, memberships, k)

    return best[1], best[2]
