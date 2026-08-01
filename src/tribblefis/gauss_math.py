from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from tribbleclustering import IVATMeans, FuzzyCMeans

from .gauss_data import *  # noqa: F401, F403


def log_transform(X: pd.DataFrame, column: str | list[str], offset=0):
    """Apply log transformation to a column in the DataFrame with an offset to avoid log(0)."""
    X[column] = np.log(X[column] + offset)
    return X


def standard_transform(X: pd.DataFrame | pd.Series, column: str | list[str] = "") -> pd.DataFrame:
    if isinstance(X, pd.Series):
        X = (X - X.min()) / (X.max() - X.min())
    else:
        X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min())
    return X


def detect_and_apply_log_transform(X: pd.DataFrame, min_dynamic_range: float= 3.0) -> tuple[pd.DataFrame, list[str]]:
    """
    Detect and apply log transformation to features with dynamic range

    A feature qualifies for log transformation if its dynamic range
    (log10(max_abs / min_abs) of non-zero values) exceeds dynamic_range.

    Args:
        X: DataFrame of features
        min_dynamic_range: Dynamic range threshold for log transformation

    Returns:
        Tuple of (transformed_X, features_to_transform)
    """
    features_to_transform = []
    for col in X.columns:
        vals = X[col].dropna()
        if len(vals) == 0:
            continue

        non_zero_vals = vals[vals != 0]
        if len(non_zero_vals) == 0:
            continue

        abs_vals = np.abs(non_zero_vals)
        max_abs = abs_vals.max()
        min_abs = abs_vals.min()

        dynamic_range = np.log10(max_abs / min_abs)
        if dynamic_range >= min_dynamic_range:
            features_to_transform.append(col)
            print(f"  Suggesting log-transform for feature '{col}' (dynamic range: {dynamic_range:.2f})")
            # Offset by minimum value to avoid log(0)
            X[col] = np.log1p(X[col] - vals.min())

    return X, features_to_transform


def find_optimal_gaussians(data, max_gaussians: int = 4):
    """Find the optimal number of Gaussians using Bayesian Information Criterion (BIC)"""
    if len(data) < 2:
        return 1

    n_samples = len(data)
    max_components = min(max_gaussians, n_samples)

    bics = []
    n_components_range = range(1, max_components + 1)

    # Use IVat Means to estimate optimal number of Gaussians?
    # ivat_means = IVATMeans(random_state=42)
    # ivat_means.fit(data.copy())
    # return ivat_means.cluster_centers_.shape[0]

    for n in n_components_range:
        # TO DO - Use Fuzzy C Means to pick the mu and then compute the sigma
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(data)
        if not gmm.converged_:
            continue
        bics.append(gmm.bic(data))

    optimal_n = n_components_range[np.argmin(bics)]
    return optimal_n


def fit_gaussians(X, y, column: str, label_value: int, n_gaussians: int = 0, max_samples: int = 20_000):
    """Fit multiple Gaussian distributions to a single variable filtered by label value using K-means

    If n_gaussians <= 0, the optimal number of Gaussians is determined automatically.
    """

    # Filter data by label, and take a maximum of 20K samples
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

    data = series.values.reshape(-1, 1)
    data = data[:max_samples]

    if len(data) == 0:
        return []

    if n_gaussians <= 0:
        n_gaussians = find_optimal_gaussians(data)
        print(f"  Automatically selected {n_gaussians} Gaussians for {column} (label {label_value})")

    # Use Fuzzy C Means to find cluster centers
    # TODO - Buffer source array is read-only
    n_clusters = min(n_gaussians, len(data))
    # ivat_means = IVATMeans(random_state=42, n_clusters=n_clusters)
    # cluster_labels_ivat = ivat_means.fit_predict(data.copy())
    # cluster_labels = cluster_labels_ivat
    # TODO - Fuzzy C Means?
    fc_means = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = fc_means.fit_predict(data.copy())

    # Fit Gaussian to each cluster
    gaussians = []
    for i in range(n_gaussians):
        cluster_data = data[cluster_labels == i].flatten()
        mu, std = stats.norm.fit(cluster_data)
        if np.isfinite(mu) and np.isfinite(std):
            gaussians.append(GaussianMembership.create(mu=mu, sigma=std))

    return gaussians


def calculate_gaussian_correlation(X, y) -> list[tuple[Any, Any]]:
    """Calculate correlation coefficient between Gaussian distributions for each feature across different labels"""
    unique_labels = y.unique()

    def process_column(column):
        """Process a single column and return its differentiation score"""
        series = X[column].dropna()
        if (
            series.dtype == "object"
            or pd.api.types.is_string_dtype(series.dtype)
            or pd.api.types.is_integer_dtype(series.dtype)
        ):
            # For categorical (strings or integers), encode them as indices
            unique_values = sorted(X[column].unique())
            value_to_index = {val: i for i, val in enumerate(unique_values)}
            data = series.map(value_to_index)
        else:
            data = series

        differentiation_score = 0

        for ij in range(len(unique_labels)):
            for jk in range(ij + 1, len(unique_labels)):
                # Get data for each label
                data_label_ij = data[y == unique_labels[ij]]
                data_label_jk = data[y == unique_labels[jk]]

                # Fit Gaussian distributions
                mu_ij, std_ij = stats.norm.fit(data_label_ij)
                mu_jk, std_jk = stats.norm.fit(data_label_jk)

                # Create probability distributions over same range
                x_range = np.linspace(data.min(), data.max(), 100)
                pdf_ij = stats.norm.pdf(x_range, mu_ij, std_ij)
                pdf_jk = stats.norm.pdf(x_range, mu_jk, std_jk)

                # Normalize PDFs to sum to 1 for proper probability distributions
                pdf_ij = pdf_ij / np.sum(pdf_ij)
                pdf_jk = pdf_jk / np.sum(pdf_jk)

                # Calculate Bhattacharyya coefficient (correlation between distributions)
                bhattacharyya_coeff = np.sum(np.sqrt(pdf_ij * pdf_jk))

                # Calculate Jensen-Shannon distance (0 = identical, 1 = completely different)
                js_distance = jensenshannon(pdf_ij, pdf_jk)

                # Calculate overlap coefficient (simpler measure)
                overlap = np.sum(np.minimum(pdf_ij, pdf_jk))

                # Calculate differentiation score
                # Convert metrics to "higher = more different" scale
                bhatta_diff = 1 - bhattacharyya_coeff  # 1=completely different, 0=identical
                js_diff = js_distance  # already 1=completely different, 0=identical
                overlap_diff = 1 - overlap  # 1=completely different, 0=identical

                # Compute histogram in 1000 bins for each dataset
                hist_ij, _ = np.histogram(data_label_ij, bins=100, range=(data.min(), data.max()), density=True)
                hist_jk, _ = np.histogram(data_label_jk, bins=100, range=(data.min(), data.max()), density=True)
                hist_corr = np.corrcoef(hist_ij, hist_jk)[0, 1]
                corr_diff = 1 - hist_corr

                # Arithmetic mean
                diff_vals = np.array([bhatta_diff, js_diff, overlap_diff, corr_diff])
                diff_vals = diff_vals[np.isfinite(diff_vals)]

                arithmetic_mean = np.mean(diff_vals)

                # Geometric mean
                geometric_mean = np.prod(diff_vals) ** (1 / len(diff_vals))

                # Combined differentiation score (average of both means)
                differentiation_score += (arithmetic_mean + geometric_mean) / 2

        return column, differentiation_score

    # Use ThreadPoolExecutor to process columns in parallel
    with ThreadPoolExecutor() as executor:
        feature_differentiators = list(executor.map(process_column, X.columns))

    # Remove nan and inf
    feature_differentiators = [(col, diff_s) for (col, diff_s) in feature_differentiators if np.isfinite(diff_s)]
    # Sort features by differentiation score (descending)
    feature_differentiators.sort(key=lambda x: x[1], reverse=True)

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


def create_gaussian_membership_dict(
    X, y, top_n_var_names: list[str], n_gaussians: int | dict[str, int] = 0
) -> GaussianMixtureModel:
    """Create a dictionary of Gaussian input memberships for top-n variables across all class labels

    Args:
        X: Feature dataframe
        y: Label series
        top_n_var_names: List of feature names
        n_gaussians: Number of Gaussians to fit per feature per label (0 for automatic).
                     Can also be a dictionary mapping feature names to their respective number of Gaussians.

    Returns:
        GaussianMixtureModel containing the fit Gaussian membership functions
    """

    unique_labels = y.unique()

    def process_feature(feature_name: str) -> tuple[str, FeatureModel]:
        """Process a single feature across all labels"""
        label_models = {}

        # Determine number of gaussians for this feature
        if isinstance(n_gaussians, dict):
            feature_n_gaussians = n_gaussians.get(feature_name, 0)
        else:
            feature_n_gaussians = n_gaussians

        for label_value in unique_labels:
            label_n_gaussians = 0
            if isinstance(n_gaussians, dict):
                label_n_gaussians = n_gaussians.get(label_value, 0)
            if label_n_gaussians > 0:
                feature_n_gaussians = label_n_gaussians
            gaussians_params = fit_gaussians(X, y, feature_name, label_value, feature_n_gaussians)
            label_models[label_value] = LabelModel(memberships=gaussians_params)

        return feature_name, FeatureModel(label_models=label_models)

    feature_models = {}

    # Use ProcessPoolExecutor for CPU-bound work
    # TODO - This hangs on some linux machines without max_workers=1!
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Submit all tasks
        futures = [executor.submit(process_feature, name) for name in top_n_var_names]
        
        # Collect results as they complete
        from concurrent.futures import as_completed
        for future in as_completed(futures):
            try:
                feature_name, feature_model = future.result()
                feature_models[feature_name] = feature_model
            except Exception as e:
                # Log the error but continue processing other features
                print(f"Error processing feature: {e}")
                import traceback
                traceback.print_exc()

    return GaussianMixtureModel(feature_models=feature_models)


def t_norm(x, y, selected_norm: NormConorm | None = None):
    """T-norm (AND) function for fuzzy logic operations."""
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
        ok = np.abs(den) > 1e-12
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
    """T-conorm (OR) function for fuzzy logic operations."""
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
        ok = np.abs(den) > 1e-12
        np.divide(num, den, out=out, where=ok)
        return out
    else:
        raise ValueError(f"Invalid NORM_CORNOM value: {selected_norm}")


def t_complement(x):
    """T-complement function for fuzzy logic operations."""
    return 1 - x


def membership(x, mu, sigma, default_member: MemberFunction | None = None):
    member_fn: MemberFunction = default_member or DefaultMemberFunction
    # Add a small epsilon to sigma to avoid division by zero
    sigma = max(sigma, 1e-6)
    """Membership function for fuzzy logic operations."""
    if member_fn == "gaussian":
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    elif member_fn == "triangular":
        return np.maximum(0, 1 - np.abs((x - mu) / (2.3756 * sigma)))
    else:
        raise ValueError(f"Invalid MEMBER_FCN value: {member_fn}")


def tsk_firing_strengths(
    X: pd.DataFrame,
    model: GaussianMixtureModel,
    anomaly_details: AnomalyParameters | None = None,
    norms: NormPair | None = None,
) -> tuple[np.ndarray, list[Any]]:
    """Calculate firing strengths for each label in a Zeroth-order TSK fuzzy model.

    Args:
        X: Feature dataframe (input variables)
        model: GaussianMixtureModel containing labels and their Gaussian parameters
        anomaly_details: Whether to include the anomaly label in the firing strengths
        norms: Explicit (t-norm, t-conorm) pair. Takes precedence over
            `anomaly_details`; when both are absent the default family is used.
            Regression has no `anomaly_details` to carry the selection, so this
            argument is the only way a regressor can choose its operators.

    Returns:
        tuple containing:
            - np.ndarray of firing strengths (n_samples, n_labels)
            - list of label values corresponding to the columns in firing_strengths
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

            feature_data = X[feature_name].values
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


def tsk_predict(X: pd.DataFrame, model: GaussianMixtureModel,  anomaly_details: AnomalyParameters | None = None) -> np.ndarray:
    """Zeroth-order TSK fuzzy model for classification.

    Args:
        X: Feature dataframe (input variables)
        model: GaussianMixtureModel containing labels and their Gaussian parameters
        anomaly_details: AnomalyParameters containing anomaly detection details

    Returns:
        Array of predicted class labels (0 or 1)
    """
    firing_strengths, unique_labels = tsk_firing_strengths(X, model, anomaly_details)
    predictions = np.argmax(firing_strengths, axis=1)
    # Map back to original label values if they weren't 0 and 1
    return np.array([unique_labels[i] for i in predictions])


def simple_gaussian_predict(X: pd.DataFrame, model: SimpleGaussianClassifierModel) -> np.ndarray:
    """Predict labels using SimpleGaussianClassifierModel.

    Args:
        X: Feature dataframe (input variables)
        model: SimpleGaussianClassifierModel containing input MFs and rules

    Returns:
        Array of predicted labels
    """
    n_samples = len(X)
    n_rules = len(model.rules)

    anomaly_details = model.anomaly_params
    norms = anomaly_details.norms() if anomaly_details else resolve_norm_pair()
    member_fcn = anomaly_details.member_function if anomaly_details else DefaultMemberFunction


    if anomaly_details and anomaly_details.include_anomaly:
        n_rules += 1
    rule_firing = np.ones((n_samples, n_rules))

    for i, rule in enumerate(model.rules):
        for feature_name, mf_ids in rule.antecedents.items():
            if feature_name not in X.columns:
                continue
            matched_mfs = model.get_mfs(mf_ids)
            local_vals = np.zeros(n_samples)
            for j, mf in enumerate(matched_mfs):
                local_vals = t_conorm(local_vals, mf.evaluate(X[feature_name].values), norms.t_conorm)
            rule_firing[:, i] = t_norm(local_vals, rule_firing[:, i], norms.t_norm)

    # Aggregate rule firing strengths by consequent label
    unique_labels = [rule.consequent for rule in model.rules]

    if anomaly_details and anomaly_details.include_anomaly:
        unique_labels.append(anomaly_details.label)

    if anomaly_details and anomaly_details.include_anomaly:
        # Anomaly is the complement of the conorm of all other class firings
        # We use a similar hack as in tsk_firing_strengths
        boosted = np.clip(rule_firing[:,:-1] + anomaly_details.threshold, 0.0, 1.0)
        rule_firing[:, -1] = t_complement(
            t_conorm(boosted, None, norms.t_conorm)
        )

    predictions_idx = np.argmax(rule_firing, axis=1)
    
    return np.array([unique_labels[rule_idx] for rule_idx in predictions_idx])


def take_top_features(
    feature_differentiators: list[tuple[Any, Any]], top_p: float = 0.95, top_n: int = -1
) -> tuple[int, list[Any]]:
    if top_n > 0:
        return top_n, [s for s, v in feature_differentiators[:top_n]]

    top_n = sum(v >= (1 - top_p) for _, v in feature_differentiators)
    top_n_vars = feature_differentiators[:top_n]
    top_n_todo = [s for s, v in top_n_vars]
    return top_n, top_n_todo


def calculate_top_k_accuracy(y_true, firing_strengths, labels, max_k: int = 5):
    """Calculate top-k accuracy for different values of k.

    Args:
        y_true: True class labels.
        firing_strengths: Firing strengths for each label.
        labels: List of label values corresponding to firing_strengths columns.
        max_k: Maximum k to calculate top-k accuracy for.

    Returns:
        Dictionary mapping k to accuracy.
    """
    max_k = min(max_k, len(labels))

    # Sort indices by firing strength in descending order
    sorted_indices = np.argsort(firing_strengths, axis=1)[:, ::-1]

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
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic data to improve parity for underrepresented classes.

    Args:
        X: Original feature dataframe.
        y: Original labels series.
        model: GaussianMixtureModel containing the fitted Gaussian parameters.
        target_count: The number of samples to aim for in each class. If -1, uses the mean count.
        classes_to_augment: List of specific classes to augment. If None, augments underrepresented classes.

    Returns:
        tuple containing the augmented X and y.
    """
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
            choices = np.random.choice(len(memberships), size=n_to_generate)

            feature_values = np.zeros(n_to_generate)
            for i, g in enumerate(memberships):
                mask = choices == i
                if np.any(mask):
                    n_samples_g = np.sum(mask)
                    # Use a small epsilon for sigma if it's 0 to allow some variation
                    safe_sigma = max(g.sigma, 1e-9)
                    feature_values[mask] = np.random.normal(g.mu, safe_sigma, size=n_samples_g)

            label_samples[feature_name] = feature_values

        X_synthetic_list.append(label_samples)
        y_synthetic_list.extend([label] * n_to_generate)

    if not X_synthetic_list:
        return X.copy(), y.copy()

    X_augmented = pd.concat([X] + X_synthetic_list, ignore_index=True)
    y_augmented = pd.concat([y, pd.Series(y_synthetic_list)], ignore_index=True)

    return X_augmented, y_augmented
