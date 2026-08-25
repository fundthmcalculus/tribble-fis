"""Performant numpy versions of scipy.stats functions.

Reduces dependency on scipy by providing:
- norm_fit: Fit normal distribution (mean and std)
- norm_pdf: Normal probability density function
- jensenshannon_distance: Jensen-Shannon divergence
- wasserstein_distance: 1D Wasserstein distance
"""

import numpy as np


def norm_fit(data: np.ndarray) -> tuple[float, float]:
    """Fit a normal distribution to data.

    Args:
        data: 1-D array of samples

    Returns:
        (mu, sigma) where mu is mean and sigma is standard deviation
    """
    data = np.asarray(data, dtype=float).ravel()
    if len(data) == 0:
        return 0.0, 1e-6
    mu = float(np.mean(data))
    sigma = float(np.std(data))
    return mu, sigma


def norm_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Normal probability density function.

    Args:
        x: Value(s) at which to evaluate the PDF
        mu: Mean of the distribution
        sigma: Standard deviation of the distribution

    Returns:
        Probability density at x
    """
    x = np.asarray(x, dtype=float)
    sigma = max(sigma, 1e-10)
    inv_sigma = 1.0 / sigma
    return np.exp(-0.5 * ((x - mu) * inv_sigma) ** 2) * inv_sigma * np.sqrt(0.5 / np.pi)


def jensenshannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability distributions.

    The Jensen-Shannon distance is defined as the square root of the
    Jensen-Shannon divergence, which is the average of the KL divergences
    of p and q from their average distribution m = (p + q) / 2.

    Args:
        p: First probability distribution (must sum to 1)
        q: Second probability distribution (must sum to 1)

    Returns:
        Jensen-Shannon distance (between 0 and 1)
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()

    if len(p) != len(q):
        raise ValueError(f"p and q must have same length, got {len(p)} and {len(q)}")

    p = np.clip(p, 1e-300, 1.0)
    q = np.clip(q, 1e-300, 1.0)

    m = 0.5 * (p + q)
    kl_pm = _kl_divergence(p, m)
    kl_qm = _kl_divergence(q, m)
    js_div = 0.5 * (kl_pm + kl_qm)
    return float(np.sqrt(max(js_div, 0.0)))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence: sum(p * log(p / q))."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    p = np.clip(p, 1e-300, 1.0)
    q = np.clip(q, 1e-300, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))



def wasserstein_distance(u: np.ndarray, v: np.ndarray) -> float:
    r"""Compute the 1-D Wasserstein distance between two samples.

    .. math::
        W_1(u, v) = \int \left| F_u(x) - F_v(x) \right| \, dx

    The integral is taken **with respect to x**, so each gap between the CDFs is
    weighted by the width of the interval it spans. That weighting is what makes
    the result a distance in the units of the data: scale the samples by `k` and
    the distance scales by `k`.

    Args:
        u: First sample of observations
        v: Second sample of observations

    Returns:
        Wasserstein distance, in the units of the input.
    """
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()

    if len(u) == 0 or len(v) == 0:
        return 0.0

    u_sorted = np.sort(u)
    v_sorted = np.sort(v)

    n_u = len(u_sorted)
    n_v = len(v_sorted)

    # Distinct support points, in order. Both CDFs are constant on each interval
    # between consecutive points, so the integral is an exact finite sum over
    # them -- no quadrature error, and nothing outside [q_0, q_m] contributes
    # (below q_0 both CDFs are 0, above q_m both are 1).
    all_quantiles = np.union1d(u_sorted, v_sorted)
    if len(all_quantiles) < 2:
        return 0.0

    u_cdf = np.searchsorted(u_sorted, all_quantiles[:-1], side="right") / n_u
    v_cdf = np.searchsorted(v_sorted, all_quantiles[:-1], side="right") / n_v
    widths = np.diff(all_quantiles)

    return float(np.sum(np.abs(u_cdf - v_cdf) * widths))
