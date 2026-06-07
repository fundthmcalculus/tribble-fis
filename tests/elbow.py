from tqdm.std import tqdm


def elbow_method(X, top_n_vars: list[tuple[Any, Any]]) -> int:
    # Now, filter the data to only the top-columns selected for performance
    X_filtered: pd.DataFrame = X[[s for s, v in top_n_vars]]

    # Perform elbow method analysis
    k_range, kmeans_silhouettes, kmeans_inertia = elbow_method_analysis(
        X_filtered[: len(X_filtered) // 10].to_numpy(), max_k=15
    )

    # Plot elbow method results
    plot_elbow_method(k_range, kmeans_silhouettes, kmeans_inertia)

    # Determine optimal k (can be adjusted based on visual inspection of elbow plot - somewhere between 5 and 10)
    optimal_k = 10
    return optimal_k


def elbow_method_analysis(X, max_k=10):
    """Perform elbow method analysis for both K-means and K-medoids"""
    k_range = range(2, max_k + 1)
    kmeans_silhouettes = []
    kmeans_inertia = []

    p_bar = tqdm(k_range, desc="Elbow Method Analysis")

    for k in p_bar:
        # K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        kmeans_silhouettes.append(silhouette_score(X, kmeans_labels))
        kmeans_inertia.append(kmeans.inertia_)

        p_bar.set_postfix({"silhouette": f"{kmeans_silhouettes[-1]:.4f}", "inertia": f"{kmeans_inertia[-1]:.2f}"})

    return k_range, kmeans_silhouettes, kmeans_inertia
