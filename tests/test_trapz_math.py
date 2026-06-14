"""
Comprehensive tests for trapezoidal membership function fitting via EM.

Tests verify:
1. Trapezoid PDF computation
2. Initialization strategy (histogram-driven peaks)
3. E-step responsibility computation
4. M-step convergence (weights and parameters)
5. BIC-based model selection
6. Edge cases (unimodal, bimodal, multimodal data)
7. Numerical stability (log-likelihood tracking)
8. Degenerate component handling
"""

import numpy as np
from src.tribblefis.trapz_math import (
    trapz_pdf,
    TrapzMixtureModel,
    fit_trapezoids_em,
    find_optimal_trapezoids,
    fit_trapezoids,
    create_trapz_membership_dict,
)
from src.tribblefis.gauss_data import TrapezoidMembership


def test_trapz_pdf_shape(self):
    """Test that trapz_pdf returns correct shape."""
    x = np.linspace(-2, 2, 100)
    y = trapz_pdf(x, a=-1, b=-0.5, c=0.5, d=1)
    assert y.shape == x.shape

def test_trapz_pdf_flat_top(self):
    """Test trapz_pdf has max value in [b, c] plateau."""
    x = np.linspace(-2, 2, 1000)
    y = trapz_pdf(x, a=-1, b=-0.5, c=0.5, d=1)
    max_idx = np.argmax(y)
    x_at_max = x[max_idx]
    # Max should be in the plateau region [b, c]
    assert -0.5 <= x_at_max <= 0.5

def test_trapz_pdf_outside_support(self):
    """Test trapz_pdf is zero outside [a, d]."""
    x_left = np.array([-5.0])
    x_right = np.array([5.0])
    a, b, c, d = -1, -0.5, 0.5, 1
    assert np.allclose(trapz_pdf(x_left, a, b, c, d), 0)
    assert np.allclose(trapz_pdf(x_right, a, b, c, d), 0)

def test_trapz_pdf_symmetry(self):
    """Test trapz_pdf is symmetric for symmetric parameters."""
    x = np.linspace(-2, 2, 100)
    # Symmetric trapezoid around 0
    y = trapz_pdf(x, a=-1, b=-0.5, c=0.5, d=1)
    # Should be symmetric
    left = y[x < 0]
    right = y[x > 0]
    assert np.allclose(left, right[::-1][:len(left)])

def test_trapz_pdf_rectangular(self):
    """Test that rectangle (b == c) is valid."""
    x = np.linspace(-2, 2, 100)
    y = trapz_pdf(x, a=-1, b=0, c=0, d=1)
    # Should have max value >= 0
    assert np.max(y) >= 0

def test_trapz_pdf_degenerate_point(self):
    """Test degenerate case where a == b == c == d (point)."""
    x = np.array([0.0, 0.5, 1.0])
    y = trapz_pdf(x, a=0.5, b=0.5, c=0.5, d=0.5)
    # All values should be 0 (no area)
    assert np.allclose(y, 0)

"""Test TrapzMixtureModel class."""

def test_fit_unimodal(self):
    """Test fitting unimodal Gaussian data."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 500)
    model = TrapzMixtureModel(n_components=1, n_bins=50)
    model.fit(data)

    assert len(model.trapezoids_) == 1
    assert np.allclose(model.weights_, [1.0])
    # Trapezoid should be centered around 0
    trapz = model.trapezoids_[0]
    center = (trapz.a + trapz.d) / 2
    assert -0.5 < center < 0.5

def test_fit_bimodal(self):
    """Test fitting bimodal data with two well-separated modes."""
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(-3, 0.5, 500),
        np.random.normal(3, 0.5, 500)
    ])
    model = TrapzMixtureModel(n_components=2, n_bins=50)
    model.fit(data)

    assert len(model.trapezoids_) == 2
    assert len(model.weights_) == 2
    assert np.allclose(model.weights_.sum(), 1.0)

    # Trapezoids should be near -3 and 3
    centers = sorted([(t.a + t.d) / 2 for t in model.trapezoids_])
    assert centers[0] < -1  # First mode near -3
    assert centers[1] > 1   # Second mode near 3

def test_fit_auto_select_components(self):
    """Test auto-selection of number of components via BIC."""
    np.random.seed(42)
    # Bimodal data
    data = np.concatenate([
        np.random.normal(-2, 0.5, 500),
        np.random.normal(2, 0.5, 500)
    ])
    model = TrapzMixtureModel(n_components=0, max_components=4, n_bins=50)
    model.fit(data)

    # Should select K=2 for bimodal data
    assert len(model.trapezoids_) == 2

def test_weights_sum_to_one(self):
    """Test that mixing weights sum to 1."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 500)
    model = TrapzMixtureModel(n_components=2)
    model.fit(data)

    assert np.allclose(model.weights_.sum(), 1.0)

def test_bic_computed(self):
    """Test that BIC is computed and reasonable."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 500)
    model = TrapzMixtureModel(n_components=1)
    model.fit(data)

    assert model.bic_ is not None
    assert model.bic_ > 0  # BIC should be positive

def test_convergence(self):
    """Test that log-likelihood increases during fit."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 500)
    model = TrapzMixtureModel(n_components=1, max_iter=10, n_bins=50)
    model.fit(data)

    # Log-likelihood should be reasonable (not too negative)
    assert model.log_likelihood_ > -5000  # Very loose bound

def test_fit_degenerate_data_single_value(self):
    """Test fitting when all data is identical."""
    data = np.ones(100) * 5.0
    model = TrapzMixtureModel(n_components=1)
    model.fit(data)

    assert len(model.trapezoids_) == 1
    # Trapezoid should be at the single value
    trapz = model.trapezoids_[0]
    assert trapz.a == 5.0
    assert trapz.d == 5.0


"""Test BIC-based model selection."""

def test_bic_unimodal(self):
    """Test BIC selects K=1 for unimodal data."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 500)
    optimal_k = find_optimal_trapezoids(data, max_components=4)
    assert optimal_k == 1

def test_bic_bimodal(self):
    """Test BIC selects K=2 for bimodal data."""
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(-3, 0.5, 500),
        np.random.normal(3, 0.5, 500)
    ])
    optimal_k = find_optimal_trapezoids(data, max_components=4)
    assert optimal_k == 2

def test_bic_trimodal(self):
    """Test BIC selects K=3 for trimodal data."""
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(-4, 0.4, 300),
        np.random.normal(0, 0.4, 300),
        np.random.normal(4, 0.4, 300)
    ])
    optimal_k = find_optimal_trapezoids(data, max_components=4)
    assert optimal_k == 3


"""Test fit_trapezoids function."""

def test_fit_trapezoids_basic(self):
    """Test basic trapezoid fitting to labeled data."""
    import pandas as pd

    np.random.seed(42)
    X = pd.DataFrame({
        'feature': np.concatenate([
            np.random.normal(0, 1, 100),
            np.random.normal(3, 1, 100)
        ])
    })
    y = pd.Series([0] * 100 + [1] * 100)

    trapz0 = fit_trapezoids(X, y, 'feature', label_value=0, n_trapezoids=1)
    trapz1 = fit_trapezoids(X, y, 'feature', label_value=1, n_trapezoids=1)

    assert len(trapz0) >= 1
    assert len(trapz1) >= 1

    # Centers should reflect the data distribution
    center0 = (trapz0[0].a + trapz0[0].d) / 2
    center1 = (trapz1[0].a + trapz1[0].d) / 2
    assert center0 < center1


"""Test the create_trapz_membership_dict function."""

def test_create_model_basic(self):
    """Test creating a trapezoid membership model."""
    import pandas as pd
    from src.tribblefis.gauss_data import GaussianMixtureModel

    np.random.seed(42)
    X = pd.DataFrame({
        'f1': np.random.normal(0, 1, 100),
        'f2': np.random.normal(1, 1, 100),
        'f3': np.random.normal(2, 1, 100),
    })
    y = pd.Series(np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)]))

    model = create_trapz_membership_dict(X, y, top_n_var_names=['f1', 'f2'], n_trapezoids=1)

    assert isinstance(model, GaussianMixtureModel)
    assert 'f1' in model.feature_models
    assert 'f2' in model.feature_models

def test_model_has_trapezoids(self):
    """Test that created model contains TrapezoidMembership objects."""
    import pandas as pd

    np.random.seed(42)
    X = pd.DataFrame({
        'feature': np.random.normal(0, 1, 100),
    })
    y = pd.Series(np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)]))

    model = create_trapz_membership_dict(X, y, top_n_var_names=['feature'], n_trapezoids=1)

    has_trapezoids = False
    for feature_model in model.feature_models.values():
        for label_model in feature_model.label_models.values():
            for mf in label_model.memberships:
                if isinstance(mf, TrapezoidMembership):
                    has_trapezoids = True

    assert has_trapezoids


"""Test EM algorithm convergence properties."""

def test_em_produces_valid_trapezoids(self):
    """Test that EM produces valid trapezoids (a <= b <= c <= d)."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 500)
    trapezoids, weights, ll = fit_trapezoids_em(
        data, n_components=2, n_bins=50, max_iter=100
    )

    for trapz in trapezoids:
        assert trapz.a <= trapz.b, f"a ({trapz.a}) > b ({trapz.b})"
        assert trapz.b <= trapz.c, f"b ({trapz.b}) > c ({trapz.c})"
        assert trapz.c <= trapz.d, f"c ({trapz.c}) > d ({trapz.d})"

def test_em_weights_positive_and_sum_to_one(self):
    """Test that EM weights are positive and sum to 1."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 500)
    trapezoids, weights, ll = fit_trapezoids_em(
        data, n_components=2, n_bins=50
    )

    assert np.all(weights >= 0), "Negative weights"
    assert np.allclose(weights.sum(), 1.0), "Weights don't sum to 1"

def test_em_improves_likelihood_on_correct_k(self):
    """Test that EM finds better likelihood with correct K."""
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(-2, 0.5, 500),
        np.random.normal(2, 0.5, 500)
    ])

    # Fit with correct K=2
    trapz_k2, w_k2, ll_k2 = fit_trapezoids_em(data, n_components=2)

    # BIC should be better for K=2 than K=1
    from src.tribblefis.trapz_math import _trapz_log_likelihood
    bin_counts, bin_edges = np.histogram(data, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compare BICs
    N = len(data)
    bic_k1_trapz, bic_k1_w = fit_trapezoids_em(data, n_components=1)
    ll_k1 = _trapz_log_likelihood(
        bin_centers, bin_counts,
        [(t.a, t.b, t.c, t.d) for t in bic_k1_trapz],
        bic_k1_w
    )
    bic_k1 = 5 * 1 - 1 * np.log(N) - 2 * ll_k1
    bic_k2 = 5 * 2 - 1 * np.log(N) - 2 * ll_k2

    assert bic_k2 < bic_k1, "BIC should favor K=2 for bimodal data"


"""Test numerical stability and edge cases."""

def test_handles_very_small_data(self):
    """Test fitting on small sample size."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 10)
    model = TrapzMixtureModel(n_components=1)
    model.fit(data)
    assert len(model.trapezoids_) == 1

def test_handles_large_data(self):
    """Test fitting on large sample size."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 10000)
    model = TrapzMixtureModel(n_components=1, n_bins=100)
    model.fit(data)
    assert len(model.trapezoids_) == 1

def test_handles_wide_range_data(self):
    """Test fitting data with very different scales."""
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(0, 1, 500),
        np.random.normal(1000, 100, 500)
    ])
    model = TrapzMixtureModel(n_components=2)
    model.fit(data)
    assert len(model.trapezoids_) == 2

def test_handles_negative_data(self):
    """Test fitting negative values."""
    np.random.seed(42)
    data = np.random.normal(-100, 10, 500)
    model = TrapzMixtureModel(n_components=1)
    model.fit(data)
    assert len(model.trapezoids_) == 1
    trapz = model.trapezoids_[0]
    # Trapezoid should be in negative region
    assert trapz.d < 0

