from concurrent.futures.thread import ThreadPoolExecutor
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from tribbleclustering import IVATMeans, FuzzyCMeans

from . import kernel
from .gauss_data import *  # noqa: F401, F403
from .stats_numba import norm_fit, norm_pdf, jensenshannon_distance, wasserstein_distance, silhouette_score, kmeans_1d

# Numeric thresholds for numerical stability
_SIGMA_FLOOR = 1e-6  # Minimum variance/sigma to avoid numerical issues
_SMALL_THRESHOLD = 1e-12  # Threshold for near-zero denominators in norm calculations

#: Variance floor for BIC scoring: scale-relative (fraction of feature variance)
#: to avoid hard floors on different units. Prevents single-point components.
BIC_VARIANCE_FLOOR_FRAC = 1e-6

# A larger component count is accepted only when it improves BIC by more than this
# relative margin. Degenerate higher-k fits routinely tie the best BIC to machine
# precision (a redundant component lands on a duplicate value under the variance
# floor), so a plain strict-argmin selection is decided by sub-ULP noise -- which
# flips the selected integer k across CPUs/BLAS/numba builds and, through a fragile
# downstream partition, moves a whole model. Requiring a clear margin resolves ties
# to the smaller, more parsimonious k deterministically. The genuine "prefer more
# components" decisions clear this margin by orders of magnitude. See tribble-fis
# reproducibility issue on cross-platform BIC tie flips.
BIC_SELECTION_REL_MARGIN = 1e-6


def _bic_improves(bic: float, best_bic: float) -> bool:
    """Whether a candidate BIC is a clear enough improvement to be selected.

    `k` ascends in the selection loop, so this is only ever asked of an equal-or-
    larger component count than the running best. A tie or near-tie within
    ``BIC_SELECTION_REL_MARGIN`` is rejected, so the smaller (parsimonious) k wins
    deterministically rather than on sub-ULP float noise. See BIC_SELECTION_REL_MARGIN.
    """
    if not np.isfinite(best_bic):
        return True
    return bic < best_bic - abs(best_bic) * BIC_SELECTION_REL_MARGIN


def _hard_partition_gaussians(data: np.ndarray, labels: np.ndarray, n_clusters: int):
    """MLE Gaussian mixture from hard k-means partition: (mu, sigma, weight) per cluster.

    Parameters
    ----------
    data : np.ndarray
        1-D array of data points.
    labels : np.ndarray
        Hard cluster assignments (0 to n_clusters-1).
    n_clusters : int
        Number of clusters.

    Returns
    -------
    list[tuple[float, float, float]]
        List of (mu, sigma, weight) tuples for each cluster with samples.
    """
    n = len(data)
    out = []
    for i in range(n_clusters):
        vals = data[labels == i]
        if vals.size == 0:
            continue
        mu = float(vals.mean())
        sd = float(vals.std()) if vals.size > 1 else 0.0
        if np.isfinite(mu) and np.isfinite(sd):
            out.append((mu, sd, vals.size / n))
    return out


def _mixture_bic(data: np.ndarray, components, var_floor: float) -> float:
    """BIC of a 1-D Gaussian mixture, scored on every point.

    Parameters
    ----------
    data : np.ndarray
        1-D array of data points.
    components : list[tuple[float, float, float]]
        List of (mu, sigma, weight) Gaussian components.
    var_floor : float
        Variance floor to prevent numerical issues.

    Returns
    -------
    float
        BIC score (n_params * log(N) - 2 * log_likelihood); lower is better.

    Notes
    -----
    Uses hard-assignment MLE instead of EM optimum, evaluated on full mixture
    density at every point, making scores comparable across different k values.
    """
    if not components:
        return np.inf
    density = np.zeros(len(data), dtype=float)
    for mu, sd, weight in components:
        var = sd * sd + var_floor
        density += weight * np.exp(-0.5 * (data - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)
    log_likelihood = float(np.log(np.maximum(density, 1e-300)).sum())
    n_params = 3 * len(components) - 1
    return n_params * np.log(len(data)) - 2.0 * log_likelihood


def _kmeans_labels_1d(data: np.ndarray, k: int, random_state: int) -> np.ndarray:
    """1-D k-means clustering.

    Parameters
    ----------
    data : np.ndarray
        1-D array of data points.
    k : int
        Number of clusters.
    random_state : int
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Cluster labels (0 to k-1) for each data point.
    """
    return kmeans_1d(data, k, random_state=random_state)


def fit_gaussian_mixture_1d(
    data,
    n_gaussians: int = 0,
    max_gaussians: int = 4,
    random_state: int = 42,
) -> tuple[list, int]:
    """Fit 1-D Gaussian mixture by BIC-selected k-means.

    Parameters
    ----------
    data : array-like
        1-D data to fit.
    n_gaussians : int
        Force this many components (0 = auto-select by BIC).
    max_gaussians : int
        Maximum components to try when auto-selecting.
    random_state : int
        RNG seed for k-means.

    Returns
    -------
    memberships : list[GaussianMembership]
        Fitted Gaussian components.
    n_selected : int
        Number of components selected.

    Notes
    -----
    Returns both components and count to avoid refitting k-means after BIC selection.
    """
    data = np.asarray(data, dtype=float).ravel()
    if len(data) == 0:
        return [], 0

    var_floor = BIC_VARIANCE_FLOOR_FRAC * max(float(data.var()), np.finfo(float).tiny)
    n_distinct = len(np.unique(data))  # Limit clusters to distinct values.

    if n_gaussians > 0:
        k = min(n_gaussians, len(data), n_distinct)
        components = _hard_partition_gaussians(data, _kmeans_labels_1d(data, k, random_state), k)
    elif len(data) < 2:
        components = _hard_partition_gaussians(data, np.zeros(len(data), dtype=int), 1)
    else:
        best_bic, components = np.inf, []
        for k in range(1, min(max_gaussians, n_distinct) + 1):
            candidate = _hard_partition_gaussians(data, _kmeans_labels_1d(data, k, random_state), k)
            bic = _mixture_bic(data, candidate, var_floor)
            # Accept a larger k only on a clear improvement (see _bic_improves):
            # a bare `bic < best_bic` is decided by sub-ULP noise on the frequent
            # ties, which flips the chosen k across platforms. k ascends, so ties
            # keep the smaller (already-selected) k.
            if _bic_improves(bic, best_bic):
                best_bic, components = bic, candidate

    return (
        [GaussianMembership.create(mu=mu, sigma=sd) for mu, sd, _w in components],
        len(components),
    )


def find_optimal_gaussians(data, max_gaussians: int = 4, random_state: int = 42) -> int:
    """Number of Gaussians the data supports, by BIC.

    Parameters
    ----------
    data : array-like
        1-D data to fit.
    max_gaussians : int, default=4
        Maximum components to try.
    random_state : int, default=42
        RNG seed for k-means.

    Returns
    -------
    int
        Optimal number of components selected by BIC.

    Notes
    -----
    Wrapper over :func:`fit_gaussian_mixture_1d`; prefer calling that directly
    to get both components and count instead of refitting.
    """
    if len(data) < 2:
        return 1
    return fit_gaussian_mixture_1d(
        data, n_gaussians=0, max_gaussians=max_gaussians, random_state=random_state
    )[1]


def fit_gaussians(
    X,
    y,
    column: str,
    label_value: int,
    n_gaussians: int = 0,
    max_samples: int | None = None,
    random_state: int = 42,
    verbose: bool = False,
):
    """Fit 1-D Gaussians to one variable, filtered to one label, by k-means.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data.
    y : pd.Series
        Labels.
    column : str
        Feature name to fit.
    label_value : int
        Label value to filter to.
    n_gaussians : int, default=0
        Number of components (0 = auto-select by BIC).
    max_samples : int | None, default=None
        Row cap for fitting; None uses all rows. Random rows without replacement.
    random_state : int, default=42
        RNG seed for k-means and subsampling.
    verbose : bool, default=False
        Print auto-selected component count.

    Returns
    -------
    list[GaussianMembership]
        Fitted Gaussian components.
    """

    series = X[column][y == label_value].dropna()

    # Check if the column is categorical (string or object)
    if (
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_bool_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
    ):
        # Strings and integers are treated as class labels. Encode them as integers.
        # For these labels, there should be a number of membership functions equal to the number of class labels.
        # You can approximate them as "gaussians" with mu=index, sigma=0.00001.
        unique_values = sorted(X[column].unique())

        gaussians = []
        for i, val in enumerate(unique_values):
            gaussians.append(GaussianMembership.create(mu=float(i), sigma=0.00001))
        return gaussians

    data = series.to_numpy(dtype=float)

    if max_samples is not None and 0 < max_samples < len(data):
        rng = np.random.default_rng(random_state)
        data = data[rng.choice(len(data), size=max_samples, replace=False)]

    if len(data) == 0:
        return []

    gaussians, n_selected = fit_gaussian_mixture_1d(
        data, n_gaussians=n_gaussians, random_state=random_state
    )
    if verbose and n_gaussians <= 0:
        print(f"  Automatically selected {n_selected} Gaussians for {column} (label {label_value})")

    return gaussians


_VALID_DIFFERENTIATION_METHODS = {"bhattacharyya", "wasserstein", "composite"}


def _pairwise_label_distance(
    data_label_ij: np.ndarray, data_label_jk: np.ndarray, method: str, data_min: float, data_max: float
) -> float:
    """One label-pair's distance for a numeric 1-D series, under `method`.

    Parameters
    ----------
    data_label_ij : np.ndarray
        Data values for first label.
    data_label_jk : np.ndarray
        Data values for second label.
    method : str
        Distance metric: 'bhattacharyya', 'wasserstein', or 'composite'.
    data_min : float
        Overall series minimum for PDF grid.
    data_max : float
        Overall series maximum for PDF grid.

    Returns
    -------
    float
        Distance between the two distributions.
    """
    if method in ("bhattacharyya", "composite"):
        # Fit Gaussian distributions
        mu_ij, std_ij = norm_fit(data_label_ij)
        mu_jk, std_jk = norm_fit(data_label_jk)

        # Create probability distributions over same range
        x_range = np.linspace(data_min, data_max, 100)
        pdf_ij = norm_pdf(x_range, mu_ij, std_ij)
        pdf_jk = norm_pdf(x_range, mu_jk, std_jk)

        # Normalize PDFs to sum to 1 for proper probability distributions
        pdf_ij = pdf_ij / np.sum(pdf_ij)
        pdf_jk = pdf_jk / np.sum(pdf_jk)

        # Bhattacharyya distance = 1 - Bhattacharyya coefficient
        bhattacharyya_coeff = np.sum(np.sqrt(pdf_ij * pdf_jk))
        bhatta_diff = 1 - bhattacharyya_coeff

    if method in ("wasserstein", "composite"):
        # Wasserstein distance (non-parametric, no distribution assumption)
        w_distance = wasserstein_distance(data_label_ij, data_label_jk)
        # Normalize by pooled standard deviation for scale invariance
        pooled_std = np.sqrt((np.var(data_label_ij) + np.var(data_label_jk)) / 2)
        if pooled_std > 1e-10:
            w_distance = w_distance / pooled_std

    if method == "bhattacharyya":
        return bhatta_diff

    if method == "wasserstein":
        return w_distance

    # method == "composite"
    # Jensen-Shannon distance (0 = identical, 1 = completely different)
    js_diff = jensenshannon_distance(pdf_ij, pdf_jk)
    # Overlap coefficient, converted to a "higher = more different" scale
    overlap_diff = 1 - np.sum(np.minimum(pdf_ij, pdf_jk))
    # Squash the unbounded pooled-std-normalized wasserstein distance
    # onto the same [0, 1) scale as the other three measures so it
    # doesn't dominate or get drowned out in the blend.
    wasserstein_diff = w_distance / (1 + w_distance)

    # Deliberately excludes the removed histogram-correlation term
    # (different scale, crashed on zero-variance features).
    diff_vals = np.array([bhatta_diff, js_diff, overlap_diff, wasserstein_diff])
    diff_vals = diff_vals[np.isfinite(diff_vals)]

    if len(diff_vals) == 0:
        return 0.0
    arithmetic_mean = np.mean(diff_vals)
    geometric_mean = np.prod(diff_vals) ** (1 / len(diff_vals))
    # Average of both means: geometric term requires every measure to agree,
    # so one measure's blind spot can't alone drive the score high.
    return (arithmetic_mean + geometric_mean) / 2


def _differentiation_score(data: pd.Series, y: pd.Series, unique_labels, method: str) -> float:
    """Sum `_pairwise_label_distance` over every pair of labels in `unique_labels`.

    Parameters
    ----------
    data : pd.Series
        Feature values.
    y : pd.Series
        Labels.
    unique_labels : array-like
        Unique label values to score pairwise.
    method : str
        Distance metric: 'bhattacharyya', 'wasserstein', or 'composite'.

    Returns
    -------
    float
        Sum of pairwise distances across all label pairs.
    """
    data_min, data_max = data.min(), data.max()
    score = 0.0
    for ij in range(len(unique_labels)):
        for jk in range(ij + 1, len(unique_labels)):
            data_label_ij = data[y == unique_labels[ij]].values
            data_label_jk = data[y == unique_labels[jk]].values
            score += _pairwise_label_distance(data_label_ij, data_label_jk, method, data_min, data_max)
    return score


def _encode_if_categorical(series: pd.Series, full_column: pd.Series) -> pd.Series:
    """Map a categorical/string/integer column to integer codes; pass numeric data through.

    Parameters
    ----------
    series : pd.Series
        Series to encode (may be subset).
    full_column : pd.Series
        Full column used to determine unique values and ordering.

    Returns
    -------
    pd.Series
        Encoded series (integer codes) or original if numeric.
    """
    if (
        series.dtype == "object"
        or pd.api.types.is_string_dtype(series.dtype)
        or pd.api.types.is_integer_dtype(series.dtype)
    ):
        unique_values = sorted(full_column.unique())
        value_to_index = {val: i for i, val in enumerate(unique_values)}
        return series.map(value_to_index)
    return series


def calculate_gaussian_correlation(X, y, method: str = "wasserstein", top_n: int = -1) -> list[tuple[Any, Any]]:
    """Calculate distance metric between distributions for each feature across different labels.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data.
    y : pd.Series
        Labels.
    method : {'wasserstein', 'bhattacharyya', 'composite'}, default='wasserstein'
        Distance metric. Wasserstein is non-parametric; bhattacharyya is parametric
        (cheaper, works for approximately Gaussian data); composite blends both.
    top_n : int, default=-1
        If > 0, return only top N features; if <= 0, return all sorted descending.

    Returns
    -------
    list[tuple[str, float]]
        (feature_name, normalized_score) sorted by score descending; scores in [0, 1].

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    if method not in _VALID_DIFFERENTIATION_METHODS:
        raise ValueError(f"method must be one of {_VALID_DIFFERENTIATION_METHODS}, got {method!r}")

    # Taken from the ORIGINAL y, before the conversion below, so the label
    # enumeration order -- which decides the order the pairwise scores are
    # accumulated in -- is exactly what it was.
    unique_labels = y.unique()

    # `_differentiation_score` masks with `data[y == unique_labels[i]]` once per
    # (feature, label pair), so the same K masks are rebuilt M times. On a string
    # dtype that is the dominant cost of this function -- not the distance
    # computations it exists for. Measured on RT-IOT2022 (92,293 rows x 82
    # features x 11 labels, so 9,020 comparisons):
    #
    #     y dtype     one comparison    x 9,020
    #     str              2.91 ms       26.2 s
    #     object           2.59 ms       23.4 s
    #     category         0.02 ms        0.1 s
    #
    # against 0.58 s for every wasserstein_distance call combined. Converting
    # once makes the whole function 11.5x faster with bit-identical scores
    # (max |diff| 0.0 across all 82 features, identical ranking).
    if not isinstance(y.dtype, pd.CategoricalDtype):
        y = y.astype("category")

    def process_column(column):
        """Process a single column and return its differentiation score"""
        series = X[column].dropna()
        data = _encode_if_categorical(series, X[column])
        return column, _differentiation_score(data, y, unique_labels, method)

    # Use ThreadPoolExecutor to process columns in parallel
    with ThreadPoolExecutor(max_workers=1) as executor:
        feature_differentiators = list(executor.map(process_column, X.columns))

    # Remove nan and inf
    feature_differentiators = [(col, diff_s) for (col, diff_s) in feature_differentiators if np.isfinite(diff_s)]
    # Sort features by differentiation score (descending)
    feature_differentiators.sort(key=lambda x: x[1], reverse=True)

    # When top_n is specified, only keep the top N before normalization
    # This reduces downstream processing when only a subset is needed
    if top_n > 0:
        feature_differentiators = feature_differentiators[:top_n]

    # Normalize feature differentiators
    if feature_differentiators:
        max_score = feature_differentiators[0][1]
        if max_score > 0:
            feature_differentiators = [(col, diff_s / max_score) for (col, diff_s) in feature_differentiators]

    print("\nFeatures Ranked by Differentiation Strength:")
    print("=" * 80)
    for rank, (feature, score) in enumerate(feature_differentiators, 1):
        print(f"{rank:2d}. {feature:30s} - Score: {score:.4f}")
    print("=" * 80)

    return feature_differentiators


def calculate_interaction_scores(
    X: pd.DataFrame,
    y: pd.Series,
    feature_differentiators: list[tuple[Any, Any]],
    method: str = "wasserstein",
    candidate_pool: list[Any] | None = None,
    max_pairs: int = 2000,
) -> list[tuple[Any, Any, float]]:
    """Score every candidate feature *pair* for interaction "lift" beyond either alone.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data (all features, not just top-selected).
    y : pd.Series
        Labels.
    feature_differentiators : list[tuple[str, float]]
        Output of `calculate_gaussian_correlation` with individual scores.
    method : {'wasserstein', 'bhattacharyya', 'composite'}, default='wasserstein'
        Distance metric (must match `calculate_gaussian_correlation`).
    candidate_pool : list[str] | None, default=None
        Feature subset for pairs; None considers all in `feature_differentiators`.
    max_pairs : int, default=2000
        Maximum allowed pair count; raises ValueError if exceeded.

    Returns
    -------
    list[tuple[str, str, float]]
        (feature_i, feature_j, lift) sorted by lift descending, normalized to [0,1]
        for positive lifts. Non-positive lifts included.

    Raises
    ------
    ValueError
        If method invalid or candidate pool produces more than max_pairs pairs.

    Notes
    -----
    Lift = score(z_i * z_j) - max(score(i), score(j)), measuring joint separation
    beyond either feature alone. See Friedman's H-statistic for similar concept.
    """
    if method not in _VALID_DIFFERENTIATION_METHODS:
        raise ValueError(f"method must be one of {_VALID_DIFFERENTIATION_METHODS}, got {method!r}")

    individual_score = {col: score for col, score in feature_differentiators}
    pool = list(candidate_pool) if candidate_pool is not None else list(individual_score)
    pool = [f for f in pool if f in individual_score]

    pairs = list(combinations(pool, 2))
    if len(pairs) > max_pairs:
        raise ValueError(
            f"{len(pool)} candidate features produce {len(pairs)} pairs, exceeding "
            f"max_pairs={max_pairs}. Interaction scoring is O(n_pairs) distance "
            f"computations, not O(n) like univariate ranking -- narrow "
            f"`candidate_pool` (e.g. to features already close to the "
            f"`take_top_features` threshold) or raise `max_pairs` explicitly if "
            f"you mean to pay for the wider search."
        )

    unique_labels = y.unique()
    results: list[tuple[Any, Any, float]] = []
    for fi, fj in pairs:
        zi = _encode_if_categorical(X[fi].dropna(), X[fi])
        zj = _encode_if_categorical(X[fj].dropna(), X[fj])
        common = zi.index.intersection(zj.index)
        if len(common) < 2:
            continue
        zi, zj, y_common = zi.loc[common], zj.loc[common], y.loc[common]

        std_i, std_j = zi.std(), zj.std()
        if std_i <= _SMALL_THRESHOLD or std_j <= _SMALL_THRESHOLD:
            continue  # a constant feature has no interaction to offer
        product = zi * zj

        joint_score = _differentiation_score(product, y_common, unique_labels, method)
        lift = joint_score - max(individual_score[fi], individual_score[fj])
        if np.isfinite(lift):
            results.append((fi, fj, lift))

    results.sort(key=lambda t: t[2], reverse=True)
    if results:
        max_lift = results[0][2]
        if max_lift > 0:
            results = [(fi, fj, lift / max_lift) for fi, fj, lift in results]
    return results


def create_gaussian_membership_dict(
    X,
    y,
    top_n_var_names: list[str],
    n_gaussians: int | dict[str, int] = 0,
    max_samples: int | None = None,
    random_state: int = 42,
    verbose: bool = False,
) -> GaussianMixtureModel:
    """Create Gaussian input memberships for top-n variables across all class labels.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data.
    y : pd.Series
        Labels.
    top_n_var_names : list[str]
        Features to fit Gaussians for.
    n_gaussians : int | dict[str, int], default=0
        Components per feature-label pair (0 = auto-select by BIC).
        Can be dict mapping feature names or label values to component counts.
    max_samples : int | None, default=None
        Row cap per (feature, label) pair; None uses all rows.
    random_state : int, default=42
        RNG seed for k-means and subsampling.
    verbose : bool, default=False
        Print auto-selected component counts.

    Returns
    -------
    GaussianMixtureModel
        Fitted Gaussian components per feature and label.
    """
    import os

    # Taken from the ORIGINAL y, before the conversion below. These become the
    # task list, so their order decides the order the label models are built in
    # -- and therefore the order the membership functions reach the dedup scan,
    # which is order-sensitive. Converting first would be a silent behaviour
    # change dressed as a speedup.
    unique_labels = y.unique()

    # `fit_gaussians` slices with `X[column][y == label_value]` once per
    # (feature, label) pair -- 902 comparisons on RT-IOT2022's 82 features and 11
    # labels. On a string dtype each costs ~2.9 ms against ~0.02 ms categorical,
    # so this is the same masking cost already removed from
    # `calculate_gaussian_correlation`, in the other function that pays it.
    if not isinstance(y.dtype, pd.CategoricalDtype):
        y = y.astype("category")

    def process_feature_label_pair(args):
        """Process a single (feature, label) pair and fit Gaussians"""
        feature_name, label_value = args

        # Determine number of gaussians for this feature/label
        if isinstance(n_gaussians, dict):
            feature_n_gaussians = n_gaussians.get(feature_name, 0)
            label_n_gaussians = n_gaussians.get(label_value, 0)
            if label_n_gaussians > 0:
                feature_n_gaussians = label_n_gaussians
        else:
            feature_n_gaussians = n_gaussians

        gaussians_params = fit_gaussians(
            X,
            y,
            feature_name,
            label_value,
            feature_n_gaussians,
            max_samples=max_samples,
            random_state=random_state,
            verbose=verbose,
        )
        return feature_name, label_value, LabelModel(memberships=gaussians_params)

    # Create list of (feature, label) pairs to process
    tasks = [(fname, lval) for fname in top_n_var_names for lval in unique_labels]

    # Determine number of workers based on CPU count, but respect environment overrides
    # Use at most cpu_count - 2 to avoid overwhelming the system
    max_workers = min(
        int(os.environ.get('TRIBBLE_GAUSSIAN_WORKERS', 0)) or (os.cpu_count() or 1) - 2,
        len(tasks)
    )
    max_workers = max(1, max_workers)  # Ensure at least 1 worker

    feature_models = {}

    # Use ThreadPoolExecutor to parallelize per-(feature, label) Gaussian fitting
    ordered_models = {}
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_feature_label_pair, tasks))

        # Reconstruct nested dict structure from results
        for feature_name, label_value, label_model in results:
            if feature_name not in feature_models:
                feature_models[feature_name] = {}
            feature_models[feature_name][label_value] = label_model

        # Convert dict of dicts to FeatureModel objects, maintaining feature order
        for feature_name in top_n_var_names:
            if feature_name in feature_models:
                ordered_models[feature_name] = FeatureModel(label_models=feature_models[feature_name])
            else:
                # Fallback: if feature missing from parallel results, compute serially
                label_models = {}
                for label_value in unique_labels:
                    _, _, label_model = process_feature_label_pair((feature_name, label_value))
                    label_models[label_value] = label_model
                ordered_models[feature_name] = FeatureModel(label_models=label_models)

    except Exception as e:
        # Log the error but fall back to serial processing
        print(f"Error during parallel processing: {e}")
        import traceback
        traceback.print_exc()

        # Fall back to serial processing - process each feature completely
        ordered_models = {}
        for feature_name in top_n_var_names:
            label_models = {}
            for label_value in unique_labels:
                _, _, label_model = process_feature_label_pair((feature_name, label_value))
                label_models[label_value] = label_model
            ordered_models[feature_name] = FeatureModel(label_models=label_models)

    return GaussianMixtureModel(feature_models=ordered_models)


def t_norm(x, y, selected_norm: NormConorm | None = None):
    """T-norm (AND) function for fuzzy logic operations.

    Parameters
    ----------
    x : np.ndarray | float
        First operand.
    y : np.ndarray | float | None
        Second operand; None triggers column-wise aggregation of x.
    selected_norm : NormConorm | None, default=None
        Norm family to use (min/max, probability, luk, hamacher, einstein).

    Returns
    -------
    np.ndarray | float
        T-norm result under the selected norm family.
    """
    # None means "unspecified"; any other value is validated below rather than
    # being swapped for the default by a falsy test.
    selected_norm = selected_norm if selected_norm is not None else DefaultNormCornorm

    if y is None:
        z = np.ones(x.shape[0])
        for ij in range(0, x.shape[1]):
            z = t_norm(z, x[:, ij], selected_norm)
        return z

    if selected_norm == "min/max":
        return np.minimum(x, y)
    elif selected_norm == "probability":
        return x * y
    elif selected_norm == "luk":
        return np.maximum(0, x + y - 1)
    elif selected_norm == "hamacher":
        den = x + y - x * y
        out = np.zeros_like(np.asarray(x, dtype=float))
        ok = np.abs(den) > _SMALL_THRESHOLD
        np.divide(x * y, den, out=out, where=ok)
        return out
    elif selected_norm == "einstein":
        # Einstein product. x + y - xy = 1 - (1-x)(1-y) lies in [0, 1] for inputs
        # in [0, 1], so the denominator lies in [1, 2] and never vanishes -- no
        # singularity to guard, unlike the Hamacher product.
        return (x * y) / (2.0 - (x + y - x * y))
    else:
        raise ValueError(f"Invalid NORM_CORNOM value: {selected_norm}")


def t_conorm(x, y, selected_norm: NormConorm | None = None):
    """T-conorm (OR) function for fuzzy logic operations.

    Parameters
    ----------
    x : np.ndarray | float
        First operand.
    y : np.ndarray | float | None
        Second operand; None triggers column-wise aggregation of x.
    selected_norm : NormConorm | None, default=None
        Norm family to use (min/max, probability, luk, hamacher, einstein).

    Returns
    -------
    np.ndarray | float
        T-conorm result under the selected norm family.
    """
    selected_norm = selected_norm if selected_norm is not None else DefaultNormCornorm

    if y is None:
        z = np.zeros(x.shape[0])
        for ij in range(0, x.shape[1]):
            z = t_conorm(z, x[:, ij], selected_norm)
        return z

    if selected_norm == "min/max":
        return np.maximum(x, y)
    elif selected_norm == "probability":
        return x + y - x * y
    elif selected_norm == "luk":
        return np.minimum(1, x + y)
    elif selected_norm == "einstein":
        # Einstein sum, the De Morgan dual of the Einstein product. The
        # denominator lies in [1, 2] for inputs in [0, 1], so it is likewise
        # singularity-free.
        return (x + y) / (1.0 + x * y)
    elif selected_norm == "hamacher":
        num = x + y - 2.0 * x * y
        den = 1.0 - x * y
        out = np.ones_like(np.asarray(x, dtype=float))
        ok = np.abs(den) > _SMALL_THRESHOLD
        np.divide(num, den, out=out, where=ok)
        return out
    else:
        raise ValueError(f"Invalid NORM_CORNOM value: {selected_norm}")


def t_complement(x):
    """T-complement function for fuzzy logic operations.

    Parameters
    ----------
    x : np.ndarray | float
        Fuzzy membership value(s).

    Returns
    -------
    np.ndarray | float
        Complement (1 - x).
    """
    return 1 - x


def membership(x, mu, sigma, default_member: MemberFunction | None = None):
    """Membership function for fuzzy logic operations.

    Parameters
    ----------
    x : np.ndarray | float
        Input values.
    mu : float
        Center of membership function.
    sigma : float
        Spread parameter.
    default_member : MemberFunction | None, default=None
        Type of membership function (gaussian or triangular).

    Returns
    -------
    np.ndarray | float
        Membership degree(s) in [0, 1].
    """
    member_fn: MemberFunction = default_member or DefaultMemberFunction
    sigma = max(sigma, _SIGMA_FLOOR)
    if member_fn == "gaussian":
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    elif member_fn == "triangular":
        # MAE-optimal fit of a triangle to this Gaussian -- see
        # tribblefis.triangle_fit for the derivation.
        from .triangle_fit import GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH

        return np.maximum(0, 1 - np.abs((x - mu) / (GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH * sigma)))
    else:
        raise ValueError(f"Invalid MEMBER_FCN value: {member_fn}")


def tsk_firing_strengths(
    X: pd.DataFrame,
    model: GaussianMixtureModel,
    anomaly_details: AnomalyParameters | None = None,
    norms: NormPair | None = None,
    feature_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[Any]]:
    """Calculate firing strengths for each label in a Zeroth-order TSK fuzzy model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data (input variables).
    model : GaussianMixtureModel
        Fitted Gaussian components per feature and label.
    anomaly_details : AnomalyParameters | None, default=None
        Anomaly detection settings (label, threshold, norms).
    norms : NormPair | None, default=None
        Explicit (t-norm, t-conorm) pair; overrides anomaly_details.
    feature_arrays : dict[str, np.ndarray] | None, default=None
        Pre-extracted feature columns for efficiency; None re-extracts each call.

    Returns
    -------
    tuple[np.ndarray, list]
        - Firing strengths array (n_samples x n_labels).
        - Label values corresponding to firing_strengths columns.
    """
    if norms is None:
        norms = anomaly_details.norms() if anomaly_details else resolve_norm_pair()
    n_samples = len(X)
    # Get unique labels from any of the feature models
    first_feature_model = next(iter(model.feature_models.values()))
    unique_labels: list[int | str] = list(first_feature_model.ordered_keys)
    if anomaly_details and anomaly_details.include_anomaly:
        unique_labels = unique_labels + [anomaly_details.label]

    # Initialize firing strengths for each label/class
    # We'll treat each class as having its own rule: IF (x1 is A1) AND (x2 is A2) ... THEN class = L
    firing_strengths = np.zeros((n_samples, len(unique_labels)))

    # Pull every feature column out of the DataFrame ONCE, before the label loop
    # (unless the caller already did so and passed `feature_arrays`).
    # `X[name].values` used to sit in the inner loop, so the same column was
    # re-extracted for every label -- n_labels x n_features pandas lookups per
    # call instead of n_features. Under coordinate refinement, which evaluates
    # this tens of thousands of times on an unchanging frame, that dominated the
    # runtime: pandas __getitem__ accounted for 148s of a 257s profile.
    if feature_arrays is None:
        feature_arrays = {
            name: np.asarray(X[name].values)
            for name in model.feature_models
            if name in X
        }

    # Fast path: hand the whole class-membership block to the compiled kernel.
    # It only applies to models the flat layout can hold exactly (all-Gaussian,
    # every feature carrying every label) and produces bit-identical output, so
    # it is a pure substitution -- see `tribblefis.kernel`. The anomaly column,
    # when requested, is still derived here from the class columns exactly as
    # below, because it is a function of them rather than of the memberships.
    if kernel.HAVE_CYTHON_KERNEL:
        n_class_labels = len(unique_labels) - (
            1 if anomaly_details and anomaly_details.include_anomaly else 0
        )
        try:
            compiled = kernel.compile_model(model, list(feature_arrays))
        except kernel.NotCompilable:
            compiled = None
        if compiled is not None:
            firing_strengths[:, :n_class_labels] = kernel.firing_strengths(
                compiled, compiled.feature_matrix(feature_arrays), norms
            )
            if n_class_labels < len(unique_labels):
                boosted = np.clip(
                    firing_strengths[:, :n_class_labels] + anomaly_details.threshold,
                    0.0, 1.0,
                )
                firing_strengths[:, -1] = t_complement(
                    t_conorm(boosted, None, norms.t_conorm)
                )
            return firing_strengths, unique_labels

    for label_idx, label_value in enumerate(unique_labels):
        if anomaly_details and label_value == anomaly_details.label:
            # Anomaly label is treated as a special case
            # We'll use a complementary membership function for it
            boosted = np.clip(firing_strengths[:, :-1] + anomaly_details.threshold, 0.0, 1.0)
            firing_strengths[:, label_idx] = t_complement(
                t_conorm(boosted, None, norms.t_conorm)
            )
            continue

        # Initialize membership for this label across all features
        # Using product as T-norm (AND operator)
        label_membership = np.ones(n_samples)

        for feature_name, feature_model in model.feature_models.items():
            if label_value not in feature_model.label_models:
                continue

            feature_data = feature_arrays.get(feature_name)
            if feature_data is None:
                continue
            label_model = feature_model.label_models[label_value]

            # If multiple membership functions exist for a feature-label pair,
            # we combine them (e.g., using OR/max or weighted sum)
            feature_membership = np.zeros(n_samples)
            for mf in label_model.memberships:
                # Evaluate membership function (works for both Gaussian and Trapezoid)
                # Logic-OR
                feature_membership = t_conorm(
                    feature_membership, mf.evaluate(feature_data), norms.t_conorm
                )

            # Logic-AND
            label_membership = t_norm(label_membership, feature_membership, norms.t_norm)

        firing_strengths[:, label_idx] = label_membership

    # For zeroth-order TSK classification, the output is typically the class
    # with the maximum firing strength (defuzzification)
    return firing_strengths, unique_labels


# ---------------------------------------------------------------------------
# Analytic gradient of one Gaussian membership function's (mu, sigma) through
# the raw (pre-normalization) firing strengths, under "probability" norms.
#
# See issue #43. Each output label owns an entirely independent set of Gaussian
# antecedents, so at this raw stage only the *targeted* label's firing-strength
# column depends on the targeted membership function -- every other column's
# derivative is exactly zero. That is what makes a per-membership-function
# gradient cheap: `refine_antecedents_coordinate`'s coordinate-descent block is
# exactly one Gaussian's (mu, sigma) (`block=2`), so only one label's column
# and one feature's contribution to it ever need differentiating.
#
# Restricted to the "probability" t-norm/t-conorm family (product / probabilistic
# sum) because it is genuinely smooth everywhere; "min/max" is piecewise smooth
# (a kink where the min/max argument switches) and an analytic derivative there
# would be a subgradient, which finite differences already recover.
# ---------------------------------------------------------------------------

def _conorm_fold_probability(feature_data: np.ndarray, memberships: list) -> np.ndarray:
    """Probability t-conorm fold (``a S b = a + b - ab``) over a label's memberships.

    Parameters
    ----------
    feature_data : np.ndarray
        1-D or 2-D feature values.
    memberships : list[GaussianMembership]
        Membership functions to fold.

    Returns
    -------
    np.ndarray
        Result of t-conorm fold over all memberships.
    """
    z = np.zeros_like(feature_data, dtype=float)
    for mf in memberships:
        g = mf.evaluate(feature_data)
        z = z + g - z * g
    return z


def _conorm_fold_probability_with_grad(
    feature_data: np.ndarray, memberships: list, target_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probability t-conorm fold plus gradient w.r.t. target membership parameters.

    Parameters
    ----------
    feature_data : np.ndarray
        1-D or 2-D feature values.
    memberships : list[GaussianMembership]
        Membership functions to fold.
    target_index : int
        Index of membership function to differentiate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - z: t-conorm fold result.
        - dz_mu: Gradient w.r.t. target membership's mu.
        - dz_sigma: Gradient w.r.t. target membership's sigma.
    """
    z = np.zeros_like(feature_data, dtype=float)
    dz_mu = np.zeros_like(feature_data, dtype=float)
    dz_sigma = np.zeros_like(feature_data, dtype=float)
    for j, mf in enumerate(memberships):
        g = mf.evaluate(feature_data)
        if j == target_index:
            sigma = max(mf.sigma, 1e-6)
            diff = feature_data - mf.mu
            dg_mu = g * diff / (sigma ** 2)
            dg_sigma = g * (diff ** 2) / (sigma ** 3)
        else:
            dg_mu = 0.0
            dg_sigma = 0.0
        dz_mu, dz_sigma = dz_mu * (1.0 - g) + (1.0 - z) * dg_mu, dz_sigma * (1.0 - g) + (1.0 - z) * dg_sigma
        z = z + g - z * g
    return z, dz_mu, dz_sigma


def firing_strengths_and_mf_grad(
    feature_arrays: dict[str, np.ndarray],
    model: GaussianMixtureModel,
    target_feature: str,
    target_label: Any,
    target_mf_index: int,
) -> tuple[np.ndarray, list[Any], np.ndarray, np.ndarray]:
    """Raw firing strengths under "probability" norms plus gradient.

    Parameters
    ----------
    feature_arrays : dict[str, np.ndarray]
        Pre-extracted feature columns.
    model : GaussianMixtureModel
        Candidate model with trial parameters.
    target_feature : str
        Feature name of targeted membership.
    target_label : Any
        Output label of targeted membership.
    target_mf_index : int
        Index in that label's membership list.

    Returns
    -------
    tuple[np.ndarray, list, np.ndarray, np.ndarray]
        - Firing strengths array (n_samples x n_labels).
        - Label values.
        - Derivative w.r.t. target column's mu.
        - Derivative w.r.t. target column's sigma.

    Notes
    -----
    Only target label's column derivatives are nonzero; others are exactly zero.
    """
    first_feature_model = next(iter(model.feature_models.values()))
    unique_labels: list[Any] = list(first_feature_model.ordered_keys)
    n_samples = len(next(iter(feature_arrays.values())))
    firing_strengths = np.zeros((n_samples, len(unique_labels)))
    dF_target_dmu = np.zeros(n_samples)
    dF_target_dsigma = np.zeros(n_samples)

    for label_idx, label_value in enumerate(unique_labels):
        label_membership = np.ones(n_samples)
        is_target_label = label_value == target_label
        target_dz_mu = target_dz_sigma = None
        other_product = np.ones(n_samples) if is_target_label else None

        for feature_name, feature_model in model.feature_models.items():
            if label_value not in feature_model.label_models:
                continue
            feature_data = feature_arrays.get(feature_name)
            if feature_data is None:
                continue
            label_model = feature_model.label_models[label_value]

            if is_target_label and feature_name == target_feature:
                feature_membership, target_dz_mu, target_dz_sigma = _conorm_fold_probability_with_grad(
                    feature_data, label_model.memberships, target_mf_index
                )
            else:
                feature_membership = _conorm_fold_probability(feature_data, label_model.memberships)
                if is_target_label:
                    other_product = other_product * feature_membership

            label_membership = label_membership * feature_membership  # probability t-norm

        firing_strengths[:, label_idx] = label_membership
        if is_target_label:
            dF_target_dmu = other_product * target_dz_mu
            dF_target_dsigma = other_product * target_dz_sigma

    return firing_strengths, unique_labels, dF_target_dmu, dF_target_dsigma


def tsk_predict(X: pd.DataFrame, model: GaussianMixtureModel,  anomaly_details: AnomalyParameters | None = None) -> np.ndarray:
    """Zeroth-order TSK fuzzy model for classification.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data.
    model : GaussianMixtureModel
        Fitted Gaussian components per feature and label.
    anomaly_details : AnomalyParameters | None, default=None
        Anomaly detection settings.

    Returns
    -------
    np.ndarray
        Predicted class labels for each sample.
    """
    firing_strengths, unique_labels = tsk_firing_strengths(X, model, anomaly_details)
    predictions = np.argmax(firing_strengths, axis=1)
    # Map back to original label values if they weren't 0 and 1
    return np.array([unique_labels[i] for i in predictions])


def simple_gaussian_predict(X: pd.DataFrame, model: SimpleGaussianClassifierModel) -> np.ndarray:
    """Predict labels using SimpleGaussianClassifierModel.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data.
    model : SimpleGaussianClassifierModel
        Fitted classifier with rules and membership functions.

    Returns
    -------
    np.ndarray
        Predicted class labels for each sample.
    """
    anomaly_details = model.anomaly_params
    norms = anomaly_details.norms() if anomaly_details else resolve_norm_pair()

    rule_firing, unique_labels = _class_rule_firing(X, model, norms)
    return _anomaly_argmax(rule_firing, unique_labels, anomaly_details, norms)


def _class_rule_firing(X, model, norms):
    """The theta-independent half of prediction: each class rule's firing.

    Returns ``(rule_firing, unique_labels)`` where ``rule_firing`` is
    ``(n_samples, n_rules)`` with a trailing all-ones column reserved for the
    anomaly rule when one is configured -- so ``_anomaly_argmax`` can fill it
    without resizing. Nothing here depends on the anomaly threshold, which is why
    it can be computed once and reused across a theta sweep.
    """
    n_samples = len(X)
    n_rules = len(model.rules)

    anomaly_details = model.anomaly_params
    if anomaly_details and anomaly_details.include_anomaly:
        n_rules += 1
    rule_firing = np.ones((n_samples, n_rules))

    for i, rule in enumerate(model.rules):
        for feature_name, mf_ids in rule.antecedents.items():
            if feature_name not in X.columns:
                continue
            matched_mfs = model.get_mfs(mf_ids)
            local_vals = np.zeros(n_samples)
            for mf in matched_mfs:
                local_vals = t_conorm(local_vals, mf.evaluate(X[feature_name].values), norms.t_conorm)
            rule_firing[:, i] = t_norm(local_vals, rule_firing[:, i], norms.t_norm)

    unique_labels = [rule.consequent for rule in model.rules]
    if anomaly_details and anomaly_details.include_anomaly:
        unique_labels.append(anomaly_details.label)
    return rule_firing, unique_labels


def _anomaly_argmax(rule_firing, unique_labels, anomaly_details, norms, threshold=None):
    """The theta-dependent half: fill the anomaly column, then argmax to labels.

    ``threshold`` overrides ``anomaly_details.threshold`` (used by the sweep). The
    class firing is not mutated -- the anomaly column is written into a copy -- so
    the same ``rule_firing`` can be reused for another threshold.
    """
    if anomaly_details and anomaly_details.include_anomaly:
        rule_firing = rule_firing.copy()
        th = anomaly_details.threshold if threshold is None else threshold
        # Anomaly is the complement of the conorm of all other class firings.
        boosted = np.clip(rule_firing[:, :-1] + th, 0.0, 1.0)
        rule_firing[:, -1] = t_complement(t_conorm(boosted, None, norms.t_conorm))

    predictions_idx = np.argmax(rule_firing, axis=1)
    return np.array([unique_labels[rule_idx] for rule_idx in predictions_idx])


def simple_gaussian_predict_sweep(X, model, thresholds):
    """Predict at each anomaly threshold, reusing one class rule_firing.

    Bit-identical to calling :func:`simple_gaussian_predict` once per threshold
    with that threshold swapped into ``model.anomaly_params`` -- it runs the same
    two helpers -- but the theta-independent class firing (the expensive per-rule,
    per-feature, per-MF loop) is computed once instead of once per threshold. This
    is the operating-curve / theta-sweep path; a single threshold gains nothing.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data.
    model : SimpleGaussianClassifierModel
        Fitted classifier; must carry an enabled anomaly rule for the threshold
        to matter (otherwise every entry is the same class prediction).
    thresholds : Iterable[float]
        Anomaly boost values to evaluate.

    Returns
    -------
    dict[float, np.ndarray]
        Threshold -> predicted labels for each sample.
    """
    anomaly_details = model.anomaly_params
    norms = anomaly_details.norms() if anomaly_details else resolve_norm_pair()
    rule_firing, unique_labels = _class_rule_firing(X, model, norms)
    return {
        th: _anomaly_argmax(rule_firing, unique_labels, anomaly_details, norms, threshold=th)
        for th in thresholds
    }


def take_top_features(
    feature_differentiators: list[tuple[Any, Any]], top_p: float = 0.95, top_n: int = -1
) -> tuple[int, list[Any]]:
    """Select features from a differentiation-score ranking.

    Parameters
    ----------
    feature_differentiators : list[tuple[str, float]]
        (feature_name, normalized_score) pairs, sorted descending.
    top_p : float, default=0.95
        Per-feature threshold; keep if score >= (1 - top_p). Ignored if top_n > 0.
    top_n : int, default=-1
        If > 0, keep exactly top_n features; if <= 0, use top_p threshold.

    Returns
    -------
    tuple[int, list[str]]
        Number of features kept and list of feature names.
    """
    if top_n > 0:
        return top_n, [s for s, v in feature_differentiators[:top_n]]

    top_n = sum(v >= (1 - top_p) for _, v in feature_differentiators)
    top_n_vars = feature_differentiators[:top_n]
    top_n_todo = [s for s, v in top_n_vars]
    return top_n, top_n_todo


def take_top_interactions(
    interaction_scores: list[tuple[Any, Any, float]], top_p: float = 0.95, top_n: int = -1
) -> list[tuple[Any, Any]]:
    """Select interacting pairs from a `calculate_interaction_scores` ranking.

    Parameters
    ----------
    interaction_scores : list[tuple[str, str, float]]
        (feature_i, feature_j, lift) triples, sorted descending by lift.
    top_p : float, default=0.95
        Per-pair threshold; keep if lift >= (1 - top_p). Ignored if top_n > 0.
    top_n : int, default=-1
        If > 0, keep top_n highest-lift pairs; if <= 0, use top_p threshold.

    Returns
    -------
    list[tuple[str, str]]
        Feature pairs with positive lift meeting selection criteria.

    Notes
    -----
    Only pairs with positive lift are returned, regardless of threshold.
    """
    positive = [(fi, fj, lift) for fi, fj, lift in interaction_scores if lift > 0]
    if top_n > 0:
        return [(fi, fj) for fi, fj, _ in positive[:top_n]]
    kept_n = sum(lift >= (1 - top_p) for _, _, lift in positive)
    return [(fi, fj) for fi, fj, _ in positive[:kept_n]]


def rescue_interacting_features(
    top_features: list[Any], feature_differentiators: list[tuple[Any, Any]], kept_pairs: list[tuple[Any, Any]]
) -> list[Any]:
    """Union any feature in a kept interacting pair into the selected feature list.

    Parameters
    ----------
    top_features : list[str]
        Features already selected by `take_top_features`.
    feature_differentiators : list[tuple[str, float]]
        Full ranking for ordering merged result.
    kept_pairs : list[tuple[str, str]]
        Interacting pairs from `take_top_interactions`.

    Returns
    -------
    list[str]
        Union of top_features and rescued features, in feature_differentiators order.

    Notes
    -----
    Rescues features in kept pairs that scored below `take_top_features` threshold,
    fixing the univariate blind spot where joint interactions weren't discoverable.
    """
    rescued = {f for pair in kept_pairs for f in pair}
    keep = set(top_features) | rescued
    return [f for f, _ in feature_differentiators if f in keep]


def calculate_top_k_accuracy(y_true, firing_strengths, labels, max_k: int = 5):
    """Calculate top-k accuracy for different values of k.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    firing_strengths : np.ndarray
        Firing strengths (n_samples x n_labels).
    labels : list
        Label values corresponding to firing_strengths columns.
    max_k : int, default=5
        Maximum k value to compute accuracy for.

    Returns
    -------
    dict[int, float]
        Top-k accuracy (fraction of samples with true label in top-k predictions).
    """
    max_k = min(max_k, len(labels))

    # Sort indices by firing strength in descending order. Stable sort so tied
    # firing strengths (common with bounded-support MFs saturating at 1.0) give a
    # platform- and version-independent top-k ordering.
    sorted_indices = np.argsort(firing_strengths, axis=1, kind="stable")[:, ::-1]

    # Map labels to their index
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y_true_idx = np.array([label_to_idx[y] for y in y_true])

    top_k_accuracies = {}
    for k in range(1, max_k + 1):
        # Check if true index is in top-k predicted indices
        correct = np.any(sorted_indices[:, :k] == y_true_idx.reshape(-1, 1), axis=1)
        top_k_accuracies[k] = np.mean(correct)

    return top_k_accuracies


def generate_synthetic_data(
    X: pd.DataFrame,
    y: pd.Series,
    model: GaussianMixtureModel,
    target_count: int = -1,
    classes_to_augment: list[Any] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic data to improve parity for underrepresented classes.

    Parameters
    ----------
    X : pd.DataFrame
        Original feature data.
    y : pd.Series
        Original labels.
    model : GaussianMixtureModel
        Fitted Gaussian components for sampling.
    target_count : int, default=-1
        Target samples per class (-1 = mean count across all classes).
    classes_to_augment : list[Any] | None, default=None
        Specific classes to augment (None = all underrepresented classes).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Augmented feature data and labels.
    """
    rng = np.random.default_rng(random_state)
    counts = y.value_counts()
    if target_count == -1:
        target_count = int(counts.mean())

    if classes_to_augment is None:
        classes_to_augment = counts[counts < target_count].index.tolist()

    X_synthetic_list = []
    y_synthetic_list = []

    for label in classes_to_augment:
        n_to_generate = target_count - counts.get(label, 0)
        if n_to_generate <= 0:
            continue

        print(f"  Generating {n_to_generate} synthetic samples for class {label}")

        label_samples = pd.DataFrame(index=range(n_to_generate), columns=X.columns)

        for feature_name, feature_model in model.feature_models.items():
            if label not in feature_model.label_models:
                # If no model for this label, we can't generate data for this feature
                # Fill with mean of the feature from X if available, else 0
                label_samples[feature_name] = X[feature_name].mean()
                continue

            label_model = feature_model.label_models[label]
            memberships = label_model.memberships

            if not memberships:
                label_samples[feature_name] = X[feature_name].mean()
                continue

            # Check if all memberships are Gaussian (sampling is only implemented for Gaussian)
            from .gauss_data import GaussianMembership
            has_non_gaussian = any(not isinstance(mf, GaussianMembership) for mf in memberships)
            if has_non_gaussian:
                # Placeholder: fall back to mean for non-Gaussian (e.g., Trapezoid)
                label_samples[feature_name] = X[feature_name].mean()
                continue

            # Sample from the mixture of Gaussians
            choices = rng.integers(len(memberships), size=n_to_generate)

            feature_values = np.zeros(n_to_generate)
            for i, g in enumerate(memberships):
                mask = choices == i
                if np.any(mask):
                    n_samples_g = np.sum(mask)
                    # Use a small epsilon for sigma if it's 0 to allow some variation
                    safe_sigma = max(g.sigma, 1e-9)
                    feature_values[mask] = rng.normal(g.mu, safe_sigma, size=n_samples_g)

            label_samples[feature_name] = feature_values

        X_synthetic_list.append(label_samples)
        y_synthetic_list.extend([label] * n_to_generate)

    if not X_synthetic_list:
        return X.copy(), y.copy()

    X_augmented = pd.concat([X] + X_synthetic_list, ignore_index=True)
    y_augmented = pd.concat([y, pd.Series(y_synthetic_list)], ignore_index=True)

    return X_augmented, y_augmented
