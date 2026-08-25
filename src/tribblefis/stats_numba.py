"""Performant numpy/numba accelerated versions of scipy.stats functions.

Reduces dependency on scipy and sklearn.metrics by providing:
- norm_fit: Fit normal distribution (mean and std)
- norm_pdf: Normal probability density function
- jensenshannon_distance: Jensen-Shannon divergence
- wasserstein_distance: 1D Wasserstein distance
- silhouette_score: Silhouette coefficient for clustering
- kmeans_1d: Simple 1-D k-means clustering
"""

import numpy as np
from numba import jit, prange


# TODO(sklearn-review): candidate for sklearn.cluster.KMeans now that
# scikit-learn is a core dependency, used by gauss_math._kmeans_labels_1d to
# seed a BIC-selected Gaussian-mixture fit. NOT a mechanical swap: this uses
# random-sample initialization (sorted) rather than k-means++, which affects
# which local optimum is found and thus the exact BIC/selected-k on
# borderline cases; would need explicit init="random", n_init=1 to
# approximate current behavior, and existing mixture-output tests (e.g.
# tests/test_gauss_math.py) checked for brittleness before swapping.
def kmeans_1d(data: np.ndarray, n_clusters: int, random_state: int = 42, max_iter: int = 100) -> np.ndarray:
    """Simple k-means clustering for 1-D data.

    Args:
        data: 1-D array of samples
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        max_iter: Maximum number of iterations

    Returns:
        Cluster labels for each sample
    """
    data = np.asarray(data, dtype=float).ravel()

    if n_clusters <= 1:
        return np.zeros(len(data), dtype=int)

    if len(data) == 0:
        return np.array([], dtype=int)

    n_clusters = min(n_clusters, len(data))

    rng = np.random.default_rng(random_state)
    init_indices = rng.choice(len(data), size=n_clusters, replace=False)
    centers = data[init_indices].copy()
    centers.sort()

    for _ in range(max_iter):
        distances = np.abs(data[:, np.newaxis] - centers[np.newaxis, :])
        labels = np.argmin(distances, axis=1)

        new_centers = np.empty(n_clusters, dtype=float)
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = np.mean(data[mask])
            else:
                new_centers[k] = centers[k]

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    distances = np.abs(data[:, np.newaxis] - centers[np.newaxis, :])
    labels = np.argmin(distances, axis=1)

    return labels


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


@jit(nopython=True, parallel=True)
def _silhouette_sample_jit(
    labels: np.ndarray,
    distances: np.ndarray,
    unique_labels: np.ndarray,
) -> np.ndarray:
    """Compute silhouette coefficient for each sample (numba-jitted).

    Args:
        labels: Cluster label for each sample
        distances: Pre-computed pairwise distances (n_samples, n_samples)
        unique_labels: Unique cluster labels

    Returns:
        Silhouette coefficient for each sample
    """
    n_samples = len(labels)
    silhouette = np.empty(n_samples, dtype=np.float64)

    for i in prange(n_samples):
        label_i = labels[i]
        same_cluster = labels == label_i
        same_cluster[i] = False
        n_same = np.sum(same_cluster)

        if n_same == 0:
            silhouette[i] = 0.0
            continue

        a_i = 0.0
        for j in range(n_samples):
            if same_cluster[j]:
                a_i += distances[i, j]
        if n_same > 0:
            a_i /= n_same

        b_i = np.inf
        for label in unique_labels:
            if label != label_i:
                other_cluster = labels == label
                n_other = np.sum(other_cluster)
                if n_other > 0:
                    avg_dist = 0.0
                    for j in range(n_samples):
                        if other_cluster[j]:
                            avg_dist += distances[i, j]
                    avg_dist /= n_other
                    if avg_dist < b_i:
                        b_i = avg_dist

        if b_i == np.inf:
            silhouette[i] = 0.0
        else:
            max_val = max(a_i, b_i)
            if max_val == 0.0:
                silhouette[i] = 0.0
            else:
                silhouette[i] = (b_i - a_i) / max_val

    return silhouette


# TODO(sklearn-review): hand-rolled reimplementation of
# sklearn.metrics.silhouette_score, written when this module's docstring goal
# was avoiding a hard sklearn dependency. scikit-learn is now a core
# dependency (needed transitively by `optimizers`), and this function has no
# remaining call sites anywhere in the repo (only imported, unused, by
# gauss_math.py) -- candidate for outright deletion, or a straight swap to
# sklearn.metrics.silhouette_score if a caller reappears.
def silhouette_score(X: np.ndarray, labels: np.ndarray, metric: str = 'euclidean') -> float:
    """Compute the mean silhouette coefficient for all samples.

    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster label for each sample
        metric: Distance metric ('euclidean' or 'precomputed')

    Returns:
        Mean silhouette coefficient
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int).ravel()

    n_samples = len(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    if len(unique_labels) == n_samples:
        return 0.0

    if metric == 'precomputed':
        distances = X
    elif metric == 'euclidean':
        distances = _euclidean_distances(X)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    silhouette_vals = _silhouette_sample_jit(labels, distances, unique_labels)
    return float(np.mean(silhouette_vals))


@jit(nopython=True, parallel=True)
def _euclidean_distances(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances (numba-jitted).

    Args:
        X: Feature matrix (n_samples, n_features)

    Returns:
        Distance matrix (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    distances = np.empty((n_samples, n_samples), dtype=np.float64)

    for i in prange(n_samples):
        for j in range(n_samples):
            dist = 0.0
            for k in range(n_features):
                diff = X[i, k] - X[j, k]
                dist += diff * diff
            distances[i, j] = np.sqrt(dist)

    return distances
