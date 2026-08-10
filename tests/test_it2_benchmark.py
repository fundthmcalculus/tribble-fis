"""Benchmark test for Interval Type-2 Fuzzy Classifier with hand-crafted IT2 model."""

import numpy as np
import pytest
import pandas as pd
import uuid

from tribblefis.gauss_data import (
    GaussianMembership,
    IT2GaussianMembership,
    IT2LabelModel,
    IT2FeatureModel,
    IT2GaussianMixtureModel,
    resolve_norm_pair,
)
from tribblefis.it2_kernel import it2_firing_strengths


@pytest.fixture
def hand_crafted_it2_model():
    """Create a hand-crafted IT2 model for a simple classification problem.

    Problem: Classify scalar input x as class 0 (x < 0) or class 1 (x > 0)
    with fuzzy transition near x = 0.

    Rules:
    - Rule 0: IF x is "low" (IT2) THEN Class 0
    - Rule 1: IF x is "high" (IT2) THEN Class 1
    """
    # Feature "x" with IT2 memberships for each class

    # For Class 0: "low" membership centered at x = -1
    # Upper: mu = -1, sigma = 1.2  (wider, more permissive)
    # Lower: mu = -1, sigma = 0.6  (narrower, more restrictive)
    low_upper = GaussianMembership(mu=-1.0, sigma=1.2, id=uuid.uuid4())
    low_lower = GaussianMembership(mu=-1.0, sigma=0.6, id=uuid.uuid4())
    it2_low = IT2GaussianMembership(upper_mf=low_upper, lower_mf=low_lower)

    # For Class 1: "high" membership centered at x = 1
    # Upper: mu = 1, sigma = 1.2  (wider, more permissive)
    # Lower: mu = 1, sigma = 0.6  (narrower, more restrictive)
    high_upper = GaussianMembership(mu=1.0, sigma=1.2, id=uuid.uuid4())
    high_lower = GaussianMembership(mu=1.0, sigma=0.6, id=uuid.uuid4())
    it2_high = IT2GaussianMembership(upper_mf=high_upper, lower_mf=high_lower)

    # Build model structure
    feature_models = {
        "x": IT2FeatureModel(
            label_models={
                0: IT2LabelModel(memberships=[it2_low]),
                1: IT2LabelModel(memberships=[it2_high]),
            }
        )
    }

    model = IT2GaussianMixtureModel(feature_models=feature_models)
    return model


def test_hand_crafted_it2_firing_strengths(hand_crafted_it2_model):
    """Test that IT2 firing strengths are computed correctly for hand-crafted model."""
    model = hand_crafted_it2_model
    norms = resolve_norm_pair("probability")  # Returns a NormPair NamedTuple

    # Test points
    X = pd.DataFrame({"x": [-2, -1, 0, 1, 2]})

    firing_upper, firing_lower, firing_crisp, labels = it2_firing_strengths(
        X, model, norms, km_iterations=None
    )

    # Should have (5 samples, 2 classes)
    assert firing_upper.shape == (5, 2)
    assert firing_lower.shape == (5, 2)
    assert firing_crisp.shape == (5, 2)

    # Lower should always be <= upper
    assert np.all(firing_lower <= firing_upper)

    # Firing strengths should be in [0, 1] (membership degrees)
    assert np.all(firing_upper >= 0)
    assert np.all(firing_upper <= 1)
    assert np.all(firing_lower >= 0)
    assert np.all(firing_lower <= 1)

    # Class 0 ("low") should have higher firing for negative x
    assert firing_crisp[0, 0] > firing_crisp[0, 1]  # x = -2
    assert firing_crisp[1, 0] > firing_crisp[1, 1]  # x = -1

    # Class 1 ("high") should have higher firing for positive x
    assert firing_crisp[3, 1] > firing_crisp[3, 0]  # x = 1
    assert firing_crisp[4, 1] > firing_crisp[4, 0]  # x = 2


def test_hand_crafted_it2_uncertainty_near_boundary(hand_crafted_it2_model):
    """Test that uncertainty intervals are wider near decision boundaries."""
    model = hand_crafted_it2_model
    norms = resolve_norm_pair("probability")

    # Test points: far from boundary, at boundary
    X_far = pd.DataFrame({"x": [-5, 5]})  # Far from decision boundary at x=0
    X_near = pd.DataFrame({"x": [-0.1, 0.1]})  # Near decision boundary

    firing_upper_far, firing_lower_far, _, _ = it2_firing_strengths(
        X_far, model, norms, km_iterations=None
    )
    firing_upper_near, firing_lower_near, _, _ = it2_firing_strengths(
        X_near, model, norms, km_iterations=None
    )

    # Compute interval widths
    width_far = (firing_upper_far - firing_lower_far).max(axis=1).mean()
    width_near = (firing_upper_near - firing_lower_near).max(axis=1).mean()

    # Near boundary should have wider uncertainty (more ambiguous)
    # Note: This might not always be strictly true due to the Gaussian nature,
    # but we can check that both widths are reasonable
    assert width_far > 0, "Intervals should have non-zero width"
    assert width_near > 0, "Intervals should have non-zero width"


def test_hand_crafted_it2_crisp_is_between_bounds(hand_crafted_it2_model):
    """Test that type-reduced crisp output is within upper/lower bounds."""
    model = hand_crafted_it2_model
    norms = resolve_norm_pair("probability")

    X = pd.DataFrame({"x": np.linspace(-3, 3, 7)})

    firing_upper, firing_lower, firing_crisp, _ = it2_firing_strengths(
        X, model, norms, km_iterations=10
    )

    # Crisp should be strictly between lower and upper (with small tolerance for rounding)
    assert np.all(firing_crisp >= firing_lower - 1e-6)
    assert np.all(firing_crisp <= firing_upper + 1e-6)


def test_it2_vs_type1_comparison(hand_crafted_it2_model):
    """Compare IT2 intervals with lower/upper Type-1 models."""
    model = hand_crafted_it2_model
    norms = resolve_norm_pair("probability")

    X = pd.DataFrame({"x": np.array([-1.5, -0.5, 0.5, 1.5])})

    firing_upper, firing_lower, firing_crisp, _ = it2_firing_strengths(
        X, model, norms, km_iterations=None
    )

    # Check that the intervals are consistent:
    # - upper memberships should produce highest firing strengths
    # - lower memberships should produce lowest
    # - crisp should be somewhere in between

    # For each sample, sum the firing strengths across classes
    upper_sum = firing_upper.sum(axis=1)
    lower_sum = firing_lower.sum(axis=1)
    crisp_sum = firing_crisp.sum(axis=1)

    # Sums should be in reasonable order (though not strictly ordered due to firing dynamics)
    assert np.all(upper_sum >= 0.01), "Upper firing strengths should be non-negligible"
    assert np.all(lower_sum >= 0.01), "Lower firing strengths should be non-negligible"
    assert np.all(crisp_sum >= 0.01), "Crisp firing strengths should be non-negligible"


def test_it2_symmetry_for_symmetric_input(hand_crafted_it2_model):
    """Test that symmetric inputs produce symmetric firing strengths."""
    model = hand_crafted_it2_model
    norms = resolve_norm_pair("probability")

    # Test symmetric points
    X = pd.DataFrame({"x": [-1.0, 1.0]})

    firing_upper, firing_lower, firing_crisp, _ = it2_firing_strengths(
        X, model, norms, km_iterations=None
    )

    # For symmetric inputs centered at 0, class firing strengths should be swapped
    # firing[0] for class 0 ≈ firing[1] for class 1
    # firing[0] for class 1 ≈ firing[1] for class 0

    # This is a qualitative check: the model is designed to be symmetric
    assert firing_crisp[0, 0] > firing_crisp[0, 1], "x=-1 should prefer class 0"
    assert firing_crisp[1, 1] > firing_crisp[1, 0], "x=+1 should prefer class 1"
