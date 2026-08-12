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


@jit(nopython=True)
def _wasserstein_distance_jit(u_sorted: np.ndarray, v_sorted: np.ndarray) -> float:
    """Wasserstein distance for 1D distributions (numba-jitted).

    Assumes both inputs are sorted in ascending order.
    """
    n_u = len(u_sorted)
    n_v = len(v_sorted)
    u_idx = 0
    v_idx = 0
    u_cdf = 0.0
    v_cdf = 0.0
    distance = 0.0

    while u_idx < n_u and v_idx < n_v:
        u_val = u_sorted[u_idx]
        v_val = v_sorted[v_idx]
        u_next_cdf = (u_idx + 1.0) / n_u
        v_next_cdf = (v_idx + 1.0) / n_v

        if u_val <= v_val:
            distance += abs(v_cdf - u_next_cdf) * (u_val - max(u_sorted[u_idx - 1] if u_idx > 0 else u_val, v_sorted[v_idx - 1] if v_idx > 0 else v_val))
            u_cdf = u_next_cdf
            u_idx += 1
        else:
            distance += abs(u_cdf - v_next_cdf) * (v_val - max(u_sorted[u_idx - 1] if u_idx > 0 else u_val, v_sorted[v_idx - 1] if v_idx > 0 else v_val))
            v_cdf = v_next_cdf
            v_idx += 1

    distance += abs(v_cdf - u_cdf)
    return distance


def wasserstein_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Compute the Wasserstein distance between two 1-D distributions.

    Args:
        u: First sample of observations
        v: Second sample of observations

    Returns:
        Wasserstein distance
    """
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()

    if len(u) == 0 or len(v) == 0:
        return 0.0

    u_sorted = np.sort(u)
    v_sorted = np.sort(v)

    n_u = len(u_sorted)
    n_v = len(v_sorted)

    u_cdf = np.arange(1, n_u + 1, dtype=float) / n_u
    v_cdf = np.arange(1, n_v + 1, dtype=float) / n_v

    all_quantiles = np.union1d(u_sorted, v_sorted)
    u_quantile_vals = np.searchsorted(u_sorted, all_quantiles, side='right') / n_u
    v_quantile_vals = np.searchsorted(v_sorted, all_quantiles, side='right') / n_v

    return float(np.sum(np.abs(u_quantile_vals - v_quantile_vals)) / len(all_quantiles))


@jit(nopython=True, parallel=True)
def _silhouette_sample_jit(
    X: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Compute silhouette coefficient for each sample (numba-jitted).

    Args:
        X: Distance matrix (n_samples, n_samples)
        labels: Cluster label for each sample
        distances: Pre-computed pairwise distances

    Returns:
        Silhouette coefficient for each sample
    """
    n_samples = len(labels)
    silhouette = np.empty(n_samples, dtype=np.float64)

    for i in prange(n_samples):
        label_i = labels[i]
        same_cluster = labels == label_i
        diff_cluster = labels != label_i

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
        unique_labels = np.unique(labels)
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

    silhouette_vals = _silhouette_sample_jit(X, labels, distances)
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
