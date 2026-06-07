import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from gauss_data import GaussianMixtureModel, AnomalyParameters, SimpleGaussianClassifierModel
from gauss_math import tsk_firing_strengths, calculate_top_k_accuracy


def plot_fit_gaussians(column: str, data, gaussians: list[dict], label_value: int, n_gaussians: int):
    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    ax.hist(data, bins=100, density=True, alpha=0.4, edgecolor="black", label="Data")

    # Plot each Gaussian component
    x_range = np.linspace(data.min(), data.max(), 1000)
    mixture_pdf = np.zeros_like(x_range)

    for i, g in enumerate(gaussians):
        component_pdf = g["weight"] * stats.norm.pdf(x_range, g["mu"], g["sigma"])
        mixture_pdf += component_pdf
        ax.plot(
            x_range,
            component_pdf,
            "--",
            linewidth=2,
            label=f"Component {i + 1} (μ={g['mu']:.2f}, σ={g['sigma']:.2f}, weight={g['weight']:.2f})",
        )

    # Plot mixture
    ax.plot(x_range, mixture_pdf, "k-", linewidth=3, alpha=0.4, label="Mixture")

    ax.set_title(f"{column} - {n_gaussians} Gaussian Mixture (label={label_value})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_var_gauss_dist(X: pd.DataFrame, y, features_to_plot: list[str] = None, model: GaussianMixtureModel = None):
    """Plots Gaussian distributions for each feature column based upon label selections"""
    if not features_to_plot:
        features_to_plot = list(X.columns)
    unique_labels = y.unique() if y is not None else ["<<ALL>>"]
    n_features = len(features_to_plot)
    n_cols = min(4, max(1, n_features // 2))
    n_rows = (n_features + n_cols - 1) // n_cols

    for y_value in unique_labels:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, n_rows * 4))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Plots feature distributions conditioned on label values
        for gf_idx, column in enumerate(features_to_plot):
            ax = axes[gf_idx]
            data = X[column].dropna()
            data_label = data[y == y_value] if y is not None else data

            # Plot histogram
            ax.hist(data_label, bins=100, density=True, alpha=0.4, edgecolor="black", label=f"$y={y_value}$")

            # Fit and plot gaussian density
            if model is None or column not in model.feature_models:
                mu, std = stats.norm.fit(data_label)
                x_range = np.linspace(data.min(), data.max(), 100)
                gaussian_density = stats.norm.pdf(x_range, mu, std)
                ax.plot(x_range, gaussian_density, "-", linewidth=2, label=f"Gaussian (μ={mu:.2f}, σ={std:.2f})")
            else:
                feature_model = model.feature_models[column]
                if y_value not in feature_model.label_models:
                    continue
                label_model = feature_model.label_models[y_value]
                gauss_fits = label_model.gaussians
                x_range = np.linspace(data.min(), data.max(), 100)
                for gf_idx, gf in enumerate(gauss_fits):
                    if gf.sigma == 0:
                        ax.axvline(
                            x=gf.mu,
                            linewidth=2,
                            label=f"Gaussian {gf_idx + 1} (μ={gf.mu:.2f}, σ={gf.sigma:.2f})",
                        )
                    else:
                        gaussian_density = stats.norm.pdf(x_range, gf.mu, gf.sigma)
                        ax.plot(
                            x_range,
                            gaussian_density,
                            "-",
                            linewidth=2,
                            label=f"Gaussian {gf_idx + 1} (μ={gf.mu:.2f}, σ={gf.sigma:.2f})",
                        )

            ax.set_title(column)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for gf_idx in range(n_features, len(axes)):
            axes[gf_idx].axis("off")

        plt.tight_layout()
        plt.show()


def plot_membership_functions(model: GaussianMixtureModel | SimpleGaussianClassifierModel):
    """Plots all Gaussian membership functions in the model, sorted by mean, in a single plot with vertical subplots."""
    if isinstance(model, SimpleGaussianClassifierModel):
        n_features = model.n_features
        feature_data = [
            (
                feature_name,
                sorted(
                    [
                        ("", g)
                        for g in model.get_mfs_for_feature(feature_name)
                    ],
                    key=lambda x: x[1].mu,
                ),
            )
            for feature_name in model.all_features
        ]
    else:
        n_features = len(model.feature_models)
        feature_data = [
            (
                feature_name,
                sorted(
                    [
                        (label, g)
                        for label, label_model in feature_model.label_models.items()
                        for g in label_model.gaussians
                    ],
                    key=lambda x: x[1].mu,
                ),
            )
            for feature_name, feature_model in model.feature_models.items()
        ]

    if n_features == 0:
        print("No features in model to plot.")
        return


    fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features), squeeze=False)

    feature_data = sorted(feature_data, key=lambda x: x[0])

    for i, (feature_name, all_gaussians) in enumerate(feature_data):
        ax = axes[i, 0]

        # Determine plot range
        mus = [g.mu for label, g in all_gaussians]
        sigmas = [g.sigma for label, g in all_gaussians]
        if not mus:
            ax.set_title(f"Feature: {feature_name} (No Gaussians)")
            continue

        x_min = min(mus) - 3 * max(sigmas)
        x_max = max(mus) + 3 * max(sigmas)
        x = np.linspace(x_min, x_max, 1000)

        for label, g in all_gaussians:
            # Gaussian membership function is exp(-0.5 * ((x - mu) / sigma) ** 2)
            # We use a small epsilon for sigma as in gauss_math.membership
            if g.sigma == 0:
                ax.axvline(
                    x=g.mu,
                    linewidth=2,
                    label=f"Label {label} (μ={g.mu:.2f}, σ={g.sigma:.2f})",
                )
            else:
                sigma = max(g.sigma, 1e-6)
                y = np.exp(-0.5 * ((x - g.mu) / sigma) ** 2)
                ax.plot(x, y, label=f"Label {label} (μ={g.mu:.2f}, σ={g.sigma:.2f})")

        ax.set_title(f"Feature: {feature_name}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Membership")
        ax.legend(loc="upper right", fontsize="small")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_elbow_method(k_range, kmeans_silhouettes, kmeans_inertia):
    """Plot elbow method results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot silhouette scores
    axes[0].plot(list(k_range), kmeans_silhouettes, "bo-", label="K-means", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Clusters (k)", fontsize=12)
    axes[0].set_ylabel("Silhouette Score", fontsize=12)
    axes[0].set_title("Elbow Method - Silhouette Score", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), kmeans_inertia, "rs-", label="K-means", linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of Clusters (k)", fontsize=12)
    axes[1].set_ylabel("Inertia", fontsize=12)
    axes[1].set_title("Elbow Method - Inertia", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_classification_report(y_true, y_pred, title="Classification Report"):
    """Plot the classification report as a heatmap."""
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    metrics_df = report_df  # .drop(index="accuracy")

    # Normalize the support column by the largest value
    max_support = metrics_df["support"].max()
    if max_support > 0:
        metrics_df["support"] = metrics_df["support"] / max_support

    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_df, annot=True, cmap="RdYlGn", fmt=".4f", vmin=0, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot the confusion matrix."""
    labels = np.union1d(np.unique(y_true), np.unique(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="Blues", ax=ax, values_format=".0%")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels diagonally
    plt.tight_layout()
    plt.show()


def plot_top_k_accuracy(top_k_accuracies: dict[int, float], title: str = "Top-k Accuracy", n_classes: int = None):
    """Plot the top-k accuracy as a bar chart."""
    ks = list(top_k_accuracies.keys())
    accuracies = [top_k_accuracies[k] for k in ks]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(ks, accuracies, color="skyblue", edgecolor="navy")
    plt.xlabel("k (Top-k)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(ks)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add accuracy labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    # Add random guessing baseline
    if n_classes is not None:
        k = np.r_[1 : n_classes + 1]
        random_probs = 1.0 - np.cumprod(n_classes - k) / n_classes**k
        plt.plot(ks, random_probs[: len(ks)], "r--", linewidth=2, label="Random Guessing", marker="o")
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_anomaly_threshold_sweep(
    X: pd.DataFrame,
    y: pd.Series,
    model: GaussianMixtureModel,
    top_n_todo: list[Any],
    anomaly_label: str = "anomaly",
    thresholds: np.ndarray = None,
):
    """Plot FPR and FNR as a function of the anomaly threshold.

    Args:
        X: Feature dataframe
        y: True labels
        model: GaussianMixtureModel
        top_n_todo: List of features to use
        anomaly_label: Label used for anomalies
        thresholds: Array of thresholds to sweep. Defaults to np.linspace(0.9, 0.9999, 50)
    """
    if thresholds is None:
        thresholds = 1.0 - np.logspace(-5, 0, 15)

    fpr_list = []
    fnr_list = []

    print(f"\nSweeping {len(thresholds)} anomaly thresholds...")

    for threshold in thresholds:
        anomaly_params = AnomalyParameters(include_anomaly=True, threshold=threshold, label=anomaly_label)
        firing_strengths, labels = tsk_firing_strengths(X[top_n_todo], model, anomaly_details=anomaly_params)
        y_pred = np.array([labels[i] for i in np.argmax(firing_strengths, axis=1)])

        # Calculate confusion matrix components
        # We care about "regular" vs "anomaly"
        # FPR = FP / (FP + TN)  (Regular predicted as Anomaly / All Regular)
        # FNR = FN / (FN + TP)  (Anomaly predicted as Regular / All Anomaly)

        is_anomaly_true = y == anomaly_label
        is_regular_true = ~is_anomaly_true

        is_anomaly_pred = y_pred == anomaly_label
        is_regular_pred = ~is_anomaly_pred

        tp = np.sum(is_anomaly_true & is_anomaly_pred)
        fp = np.sum(is_regular_true & is_anomaly_pred)
        tn = np.sum(is_regular_true & is_regular_pred)
        fn = np.sum(is_anomaly_true & is_regular_pred)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        fpr_list.append(fpr)
        fnr_list.append(fnr)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogx(-thresholds + 1.0, fpr_list, "b", label="False Positive Rate (FPR)", linewidth=2)
    plt.semilogx(-thresholds + 1.0, fnr_list, "r", label="False Negative Rate (FNR)", linewidth=2)
    plt.xlabel("Anomaly Fraction")
    plt.ylabel("Rate")
    plt.title("Anomaly Fraction Sweep: FPR vs FNR")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return thresholds, fpr_list, fnr_list


def report_figures_of_merit(
    X,
    y,
    gaussian_memberships: GaussianMixtureModel,
    n_unique: int,
    start_time: float,
    top_n_todo: list[Any],
    label: str = "test",
    anomaly_details: AnomalyParameters = None,
) -> tuple[np.ndarray, list, dict]:
    # Create the actual fuzzy model and predict
    print(f"\nEvaluating Zeroth-Order TSK Model on {label.upper()} set:")
    print("=" * 80)
    firing_strengths, labels = tsk_firing_strengths(
        X[top_n_todo], gaussian_memberships, anomaly_details=anomaly_details
    )
    y_pred = np.array([labels[i] for i in np.argmax(firing_strengths, axis=1)])

    # Calculate top-k accuracy
    top_k_acc = calculate_top_k_accuracy(y, firing_strengths, labels, max_k=n_unique)
    print(f"\nTop-k Accuracy ({label}):")
    for k, acc in top_k_acc.items():
        print(f"  Top-{k}: {acc:.4f}")

    # Calculate accuracy
    accuracy = np.mean(y_pred == y)
    print(f"Model Accuracy ({label}): {accuracy:.4f}")

    # Confusion matrix and Classification report
    print(f"\nConfusion Matrix ({label}):")
    cm = confusion_matrix(y, y_pred, labels=labels)
    print(cm)

    # Analyze confusion matrix to identify top confusions
    top_confusions = analyze_confusion_matrix(cm, labels)

    # Extract confused class data
    confused_data = extract_confused_class_data(X, y, y_pred, top_confusions)

    print(f"\nClassification Report ({label}):")
    print(classification_report(y, y_pred))
    print("=" * 80)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

    # Plot results
    plot_confusion_matrix(y, y_pred, title=f"TSK Model Confusion Matrix ({label} Set)")
    plot_classification_report(y, y_pred, title=f"TSK Model Classification Report ({label} Set)")
    plot_top_k_accuracy(top_k_acc, title=f"TSK Model Top-k Accuracy ({label} Set)", n_classes=n_unique)

    return cm, top_confusions, confused_data


def analyze_confusion_matrix(cm: np.ndarray, labels: list, top_n: int = -1):
    """Analyze confusion matrix to find the highest cross-class confusions."""
    confusions = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j:  # off-diagonal
                if cm[i, j] > 0:
                    confusions.append((labels[i], labels[j], cm[i, j]))

    if top_n == -1:
        top_n = len(confusions)

    # Sort by confusion count (descending)
    confusions.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop {top_n} Cross-Class Confusions:")
    print("-" * 60)
    for true_label, pred_label, count in confusions[:top_n]:
        print(f"  True: {true_label:20s} -> Predicted: {pred_label:20s} | Count: {count}")
    print("-" * 60)

    return confusions[:top_n]


def extract_confused_class_data(X: pd.DataFrame, y: pd.Series, y_pred: pd.Series, confusion_pairs: list) -> dict:
    """Extract subsets of data for confused class pairs.

    Args:
        X: Feature dataset
        y: Labels
        y_pred: Predicted labels
        confusion_pairs: List of tuples (true_label, pred_label, count)

    Returns:
        Dictionary mapping confusion pair to filtered data
    """
    confused_data = {}

    for true_label, pred_label, count in confusion_pairs:
        # Get indices where label is either true_label or pred_label
        mask = (y == true_label) & (y_pred == pred_label)
        X_subset = X[mask]
        y_subset = y[mask]

        confused_data[(true_label, pred_label)] = {
            "X": X_subset,
            "y": y_subset,
        }

    return confused_data
