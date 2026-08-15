"""
Triangular membership function fitting using histogram-based Expectation Maximization.

A triangular membership function is the degenerate case of a trapezoid whose
plateau collapses to a single apex point -- ``TrapezoidMembership(a, b, c, d)``
with ``b == c`` is exactly ``TriangularMembership(a, b, d)``. Rather than fit a
full trapezoid and average its two shoulders together afterwards, this module
fits that degenerate shape directly: the apex is a single free parameter
optimized in the M-step (3 free parameters per component instead of 4), and
every density evaluation is literally ``trapz_math.trapz_pdf(x, a, b, b, c)``.

The E-step, log-likelihood, and histogram-peak initialization are structurally
identical to :mod:`tribblefis.trapz_math`'s (this module imports and reuses the
shared pieces directly); only the M-step's parameter count and constraints
differ, since a triangle has one fewer degree of freedom than a trapezoid.

See ``docs/triangle-em-resolution-evaluation.md`` for why this stays
histogram-based (rather than fitting directly against raw, unbinned samples):
in short, histogram binning is not the resolution bottleneck for the
well-separated, roughly-unimodal-per-label components this is meant to fit,
and a naive unbinned substitution breaks the peak-detection initialization
outright (it implicitly assumes evenly-spaced bins).
"""

from typing import Optional

import numpy as np
from scipy.optimize import minimize

from .gauss_data import TriangularMembership
from .trapz_math import trapz_pdf, _init_trapz_from_histogram, _em_m_step_weights


def triangle_pdf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Normalized triangular probability density function.

    Piecewise linear: 0 outside [a, c], rises linearly from 0 to 1 over [a, b],
    falls linearly from 1 to 0 over [b, c]. Apex ``b`` satisfies ``a <= b <= c``.

    This is exactly the trapezoidal PDF with its plateau collapsed to a point:
    ``triangle_pdf(x, a, b, c) == trapz_pdf(x, a, b, b, c)``.
    """
    return trapz_pdf(x, a, b, b, c)


def _init_triangles_from_histogram(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    n_components: int,
    data_min: float,
    data_max: float,
) -> tuple[list[tuple[float, float, float]], np.ndarray]:
    """Initialize triangle parameters from histogram peaks.

    Reuses :func:`trapz_math._init_trapz_from_histogram`'s peak/valley
    detection to place each component's ``(a, d)`` support and half-power
    ``(b, c)`` shoulders, then collapses the plateau ``[b, c]`` to a single
    apex at its midpoint -- the EM's M-step subsequently moves that apex
    freely, this just picks where it starts.

    Returns:
        (params_list, weights) where params_list is [(a, apex, c), ...]
    """
    trapz_params, weights = _init_trapz_from_histogram(
        bin_centers, bin_counts, n_components, data_min, data_max
    )
    params_list = [(a, (b + c) / 2, d) for (a, b, c, d) in trapz_params]
    return params_list, weights


def _triangle_log_likelihood(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    params_list: list[tuple[float, float, float]],
    weights: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """Compute log-likelihood of histogram under triangle mixture."""
    mixture_pdf = np.zeros_like(bin_centers, dtype=float)
    for k, (a, b, c) in enumerate(params_list):
        mixture_pdf += weights[k] * triangle_pdf(bin_centers, a, b, c)

    mixture_pdf = np.maximum(mixture_pdf, eps)
    return float(np.sum(bin_counts * np.log(mixture_pdf)))


def _em_e_step(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    params_list: list[tuple[float, float, float]],
    weights: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """E-step: compute responsibilities r[i,k] = P(z=k | x_i)."""
    n_bins = len(bin_centers)
    n_components = len(params_list)

    densities = np.empty((n_bins, n_components))
    for k, (a, b, c) in enumerate(params_list):
        densities[:, k] = weights[k] * triangle_pdf(bin_centers, a, b, c)

    denom = densities.sum(axis=1)
    valid = denom > eps
    responsibilities = np.empty((n_bins, n_components))
    responsibilities[valid] = densities[valid] / denom[valid, None]
    responsibilities[~valid] = 1.0 / n_components

    return responsibilities


def _em_m_step_params(
    bin_centers: np.ndarray,
    bin_counts: np.ndarray,
    responsibilities: np.ndarray,
    params_list: list[tuple[float, float, float]],
    data_min: float,
    data_max: float,
) -> list[tuple[float, float, float]]:
    """M-step for triangle parameters using constrained optimization.

    For each component k, minimize the negative weighted log-likelihood over
    ``(a, b, c)`` -- one fewer free parameter than the trapezoid M-step, since
    the plateau is a single point rather than an independent interval.
    """
    n_components = responsibilities.shape[1]
    new_params = []

    bin_centers = np.asarray(bin_centers, dtype=float)
    bin_counts = np.asarray(bin_counts, dtype=float)

    for k in range(n_components):
        a_k, b_k, c_k = params_list[k]
        coeff_k = responsibilities[:, k] * bin_counts

        def objective(params, _coeff=coeff_k):
            a, b, c = params
            pdf_vals = np.maximum(triangle_pdf(bin_centers, a, b, c), 1e-10)
            return -np.dot(_coeff, np.log(pdf_vals))

        # Constraints: a <= b <= c (with analytic linear Jacobians)
        constraints = [
            {'type': 'ineq', 'fun': lambda p: p[1] - p[0],
             'jac': lambda p: np.array([-1.0, 1.0, 0.0])},  # b >= a
            {'type': 'ineq', 'fun': lambda p: p[2] - p[1],
             'jac': lambda p: np.array([0.0, -1.0, 1.0])},  # c >= b
        ]

        bounds = [(data_min, data_max)] * 3
        x0 = [a_k, b_k, c_k]

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 100}
        )

        if result.success and result.x is not None:
            a_new, b_new, c_new = result.x
        else:
            a_new, b_new, c_new = x0

        new_params.append((a_new, b_new, c_new))

    return new_params


def fit_triangles_em(
    data_1d: np.ndarray,
    n_components: int,
    n_bins: int = 50,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> tuple[list[TriangularMembership], np.ndarray, float]:
    """Run EM to fit a mixture of triangles to 1D data.

    Analogue of :func:`trapz_math.fit_trapezoids_em`, fitting the degenerate
    (plateau-collapsed) shape directly rather than fitting a trapezoid and
    averaging ``b, c`` together afterwards.

    Args:
        data_1d: 1D array of observations
        n_components: Number of triangle components
        n_bins: Number of histogram bins
        max_iter: Maximum EM iterations
        tol: Convergence tolerance on log-likelihood relative change
        random_state: Random seed (currently unused, for API compatibility)

    Returns:
        (triangles, weights, log_likelihood): List of fitted TriangularMembership
        objects, their mixing weights, and final log-likelihood
    """
    if random_state is not None:
        np.random.seed(random_state)

    data_1d = np.asarray(data_1d, dtype=float)
    data_min, data_max = data_1d.min(), data_1d.max()

    if data_min == data_max:
        # Degenerate case: all data identical
        mid = data_min
        tri = TriangularMembership.create(mid, mid, mid)
        return [tri], np.array([1.0]), 0.0

    bin_counts, bin_edges = np.histogram(data_1d, bins=n_bins, range=(data_min, data_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    params_list, weights = _init_triangles_from_histogram(
        bin_centers, bin_counts, n_components, data_min, data_max
    )

    params_list = params_list[:n_components]
    weights = weights[:len(params_list)]
    weights = weights / np.sum(weights)

    prev_ll = -np.inf
    best_ll = -np.inf
    best_params = params_list
    best_weights = weights

    for iteration in range(max_iter):
        responsibilities = _em_e_step(bin_centers, bin_counts, params_list, weights)
        weights = _em_m_step_weights(responsibilities, bin_counts)
        params_list = _em_m_step_params(
            bin_centers, bin_counts, responsibilities, params_list, data_min, data_max
        )

        ll = _triangle_log_likelihood(bin_centers, bin_counts, params_list, weights)

        if ll > best_ll:
            best_ll = ll
            best_params = params_list
            best_weights = weights

        if prev_ll > -np.inf:
            rel_change = abs(ll - prev_ll) / (abs(prev_ll) + 1e-10)
            if rel_change < tol:
                break

        prev_ll = ll

    triangles = [
        TriangularMembership.create(a, b, c)
        for a, b, c in best_params
    ]

    return triangles, best_weights, best_ll


def fit_triangle_mixture_1d(
    data_1d: np.ndarray,
    n_triangles: int = 0,
    max_components: int = 4,
    n_bins: int = 50,
) -> tuple[list[TriangularMembership], int]:
    """Fit a 1-D triangle mixture, choosing the component count by BIC.

    Returns ``(triangles, n_selected)`` -- both, so the caller does not refit,
    matching :func:`trapz_math.fit_trapezoid_mixture_1d`'s convention.

    BIC = n_params * log(N) - 2 * log_likelihood, with n_params = 4K - 1
    (3 parameters per triangle plus K-1 free weights).
    """
    data_1d = np.asarray(data_1d, dtype=float)

    if n_triangles > 0:
        triangles, _weights, _ll = fit_triangles_em(
            data_1d, n_components=n_triangles, n_bins=n_bins, max_iter=100, tol=1e-4
        )
        return triangles, n_triangles

    N = len(data_1d)
    best = (np.inf, [], 0)
    for k in range(1, max_components + 1):
        triangles, _weights, ll = fit_triangles_em(
            data_1d, n_components=k, n_bins=n_bins, max_iter=100
        )
        bic = (4 * k - 1) * np.log(N) - 2 * ll
        if bic < best[0]:
            best = (bic, triangles, k)

    return best[1], best[2]


def find_optimal_triangles(
    data_1d: np.ndarray,
    max_components: int = 4,
    n_bins: int = 50,
) -> int:
    """Number of triangle components the data supports, by BIC.

    Thin wrapper over :func:`fit_triangle_mixture_1d`, kept because it is
    public. Prefer that function directly: it returns the fit the count came
    from rather than discarding it.
    """
    return fit_triangle_mixture_1d(
        data_1d, n_triangles=0, max_components=max_components, n_bins=n_bins
    )[1]


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

    Analogue of trapz_math.fit_trapezoids() but for triangles.

    Args:
        X: Feature dataframe
        y: Label series
        column: Column name to fit
        label_value: Class label to filter by
        n_triangles: Number of triangle components (0 for automatic BIC selection)
        max_samples: Cap on the rows used for the fit; ``None`` -- the default --
            uses every row. When a cap is given the rows are drawn at random
            without replacement, seeded by ``random_state``.
        random_state: Seeds the subsample draw.
        verbose: Print the automatically-selected component count.

    Returns:
        List of fitted TriangularMembership objects
    """
    data = X[column][y == label_value].dropna().values

    if max_samples is not None and 0 < max_samples < len(data):
        rng = np.random.default_rng(random_state)
        data = data[rng.choice(len(data), size=max_samples, replace=False)]

    if len(data) == 0:
        return []

    triangles, n_selected = fit_triangle_mixture_1d(
        data, n_triangles=n_triangles, max_components=4
    )
    if verbose and n_triangles <= 0:
        print(f"  Automatically selected {n_selected} triangles for {column} (label {label_value})")

    return triangles


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

    Analogue of trapz_math.create_trapz_membership_dict() but uses triangles.

    Returns the same GaussianMixtureModel container type -- the model is
    agnostic about whether LabelModel.memberships contains Gaussian, Trapezoid,
    or Triangular objects.

    Args:
        X: Feature dataframe
        y: Label series
        top_n_var_names: List of feature names to fit
        n_triangles: Number of triangles (0 for automatic, or dict per feature)
        max_samples: Rows per (feature, label) used for the fit; ``None`` uses
            all of them. See :func:`fit_triangles`.
        random_state: Seeds the subsample draw.
        verbose: Print each automatically-selected component count.

    Returns:
        GaussianMixtureModel containing fitted triangle MFs
    """
    from concurrent.futures.thread import ThreadPoolExecutor
    from .gauss_data import GaussianMixtureModel, FeatureModel, LabelModel

    unique_labels = y.unique()

    def process_feature(feature_name: str) -> tuple[str, FeatureModel]:
        """Process a single feature across all labels"""
        label_models = {}

        if isinstance(n_triangles, dict):
            feature_n_triangles = n_triangles.get(feature_name, 0)
        else:
            feature_n_triangles = n_triangles

        for label_value in unique_labels:
            label_n_triangles = 0
            if isinstance(n_triangles, dict):
                label_n_triangles = n_triangles.get(label_value, 0)
            if label_n_triangles > 0:
                feature_n_triangles = label_n_triangles

            tri_params = fit_triangles(
                X,
                y,
                feature_name,
                label_value,
                feature_n_triangles,
                max_samples=max_samples,
                random_state=random_state,
                verbose=verbose,
            )
            label_models[label_value] = LabelModel(memberships=tri_params)

        return feature_name, FeatureModel(label_models=label_models)

    feature_models = {}

    with ThreadPoolExecutor() as executor:
        result = executor.map(process_feature, top_n_var_names)
        for r in result:
            feature_models[r[0]] = r[1]

    return GaussianMixtureModel(feature_models=feature_models)


class TriangleMixtureModel:
    """Fits a mixture of triangular membership functions to 1D histogram data.

    Analogue of :class:`trapz_math.TrapzMixtureModel`, but for the 3-parameter
    degenerate (plateau-collapsed) shape.

    After calling fit(), access:
        triangles_: List of fitted TriangularMembership objects
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

        self.triangles_: Optional[list[TriangularMembership]] = None
        self.weights_: Optional[np.ndarray] = None
        self.log_likelihood_: Optional[float] = None
        self.bic_: Optional[float] = None

    def fit(self, data_1d: np.ndarray) -> "TriangleMixtureModel":
        """Fit triangular MFs to 1D data.

        Args:
            data_1d: 1D array of observations

        Returns:
            self
        """
        data_1d = np.asarray(data_1d, dtype=float)

        n_comp = self.n_components
        if n_comp <= 0:
            n_comp = find_optimal_triangles(
                data_1d, max_components=self.max_components, n_bins=self.n_bins
            )

        triangles, weights, ll = fit_triangles_em(
            data_1d,
            n_components=n_comp,
            n_bins=self.n_bins,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        self.triangles_ = triangles
        self.weights_ = weights
        self.log_likelihood_ = ll

        N = len(data_1d)
        K = len(triangles)
        n_params = 4 * K - 1
        self.bic_ = n_params * np.log(N) - 2 * ll

        return self
