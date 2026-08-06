# Interval Type-2 Fuzzy Inference System (IT2-FIS) Guide

## Overview

The IT2-FIS implementation provides a production-ready interval type-2 fuzzy inference system that extends the existing Type-1 FIS framework. It automatically generates uncertainty intervals from learned membership functions, with optional Karnik-Mendel type reduction. Supports three membership function types (Gaussian, trapezoidal, triangular) with optional antecedent refinement.

## Quick Start

### Classification

```python
from tribblefis.it2_classifier import IntervalType2FuzzyClassifier
import pandas as pd
import numpy as np

# Your training data
X_train = pd.DataFrame({'feature1': [...], 'feature2': [...]})
y_train = np.array([0, 1, 2, ...])

# Create and fit classifier
clf = IntervalType2FuzzyClassifier(
    top_n=3,                       # Use top 3 features
    member_function="gaussian",     # "gaussian", "trap", or "triangular"
    uncertainty_width=0.5,          # Expand bounds by 0.5 * sigma
    km_iterations=10,               # Use KM for type reduction (None = averaging)
    refine=False,                   # Refine Type-1 before IT2 conversion
    random_state=42,
)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Get confidence intervals
upper, lower = clf.predict_intervals(X_test)
# upper[i, j] = upper bound firing strength for sample i, class j
# lower[i, j] = lower bound firing strength for sample i, class j
```

### Regression

```python
from tribblefis.it2_regressor import IntervalType2FuzzyRegressor

# Create and fit regressor
reg = IntervalType2FuzzyRegressor(
    top_n=3,
    n_gaussians=2,
    uncertainty_width=0.5,
    km_iterations=10,
    random_state=42,
)
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)

# Get prediction intervals
y_lower, y_upper = reg.predict_intervals(X_test)
# y_lower[i] = lower bound prediction for sample i
# y_upper[i] = upper bound prediction for sample i
```

## Membership Function Types

The IT2-FIS supports three membership function types via the `member_function` parameter:

### Gaussian (default: `"gaussian"`)

Classic bell-curve membership functions. **Recommended** for most applications.

```python
clf = IntervalType2FuzzyClassifier(member_function="gaussian", uncertainty_width=0.5)
```

**Uncertainty expansion**: For a learned Gaussian (μ, σ):
- Upper: μ + 0.5σ with same σ
- Lower: μ - 0.5σ with same σ

### Trapezoidal (`"trap"`)

Piecewise-linear membership functions defined by (a, b, c, d) with flat plateau.

```python
clf = IntervalType2FuzzyClassifier(member_function="trap", uncertainty_width=0.3)
```

**Uncertainty expansion**: Proportional expansion of all four parameters (a, b, c, d).

### Triangular (`"triangular"`)

Piecewise-linear membership functions defined by (a, b, c) with single apex at b.

```python
clf = IntervalType2FuzzyClassifier(member_function="triangular", uncertainty_width=0.3)
```

**Uncertainty expansion**: Symmetric expansion around apex at b.

---

## Key Parameters

### `member_function` (default: `"gaussian"`)

Type of membership function: `"gaussian"`, `"trap"`, or `"triangular"`.

The IT2-FIS automatically converts any Type-1 membership type to its IT2 variant by creating upper and lower bounds.

### `uncertainty_width` (default: 0.5)

Controls the footprint of uncertainty (FoU). For each learned Gaussian membership function with parameters (μ, σ):

- **Upper bound**: μ + uncertainty_width × σ
- **Lower bound**: μ - uncertainty_width × σ

**Effect**: Larger values create wider uncertainty intervals, representing greater ambiguity near decision boundaries.

```
uncertainty_width = 0.2  → narrow intervals, crisp predictions
uncertainty_width = 0.5  → moderate intervals (recommended)
uncertainty_width = 1.0  → wide intervals, more conservative
```

### `km_iterations` (default: 10)

Number of iterations for Karnik-Mendel type reduction. The KM algorithm iteratively refines left and right switch points to find the optimal crisp output.

- **None or 0**: Use simple center-of-sets averaging (faster, less accurate)
- **5-10**: Good balance of accuracy and speed (recommended)
- **20+**: High precision but slower

### `refine` (default: False)

If True, performs antecedent refinement on the Type-1 model **before** converting to IT2. This optimizes the base membership function parameters against the training data cross-entropy loss, leading to more discriminative bounds.

**How it works**:
1. Fit base Type-1 FIS (learns Gaussian/trapezoidal/triangular parameters)
2. Refine Type-1 parameters to minimize cross-entropy loss
3. Convert refined Type-1 model to IT2 (refined parameters become center points)
4. Create uncertainty bounds around the refined parameters

**Effect**: Refined IT2 models typically achieve better classification/regression accuracy because the uncertainty bounds are centered on discriminatively-learned parameters rather than marginal-fit parameters.

### `refine_method` (default: `"coordinate"`)

Method for antecedent refinement: `"coordinate"` (block coordinate descent) or `"none"` (skip refinement).

### `refine_l2_shrink` (default: 0.05)

L2 regularization strength during refinement. Controls how far refined parameters can drift from their initial values. Higher values keep parameters closer to initialization, preventing overfitting.

### Other Parameters

Same as `MixtureOfGaussiansFuzzyClassifier` / `MixtureOfGaussiansFuzzyRegressor`:

- `top_n`: Number of top features to select
- `top_p`: Per-feature score threshold
- `n_gaussians`: Gaussians per feature per label (0 = automatic)
- `n_output_buckets`: Output partitioning (regression only)
- `norm_conorm`: Fuzzy operator family ("probability" is default and recommended)
- `random_state`: Reproducibility seed

## How It Works

### 1. Model Conversion

IT2-FIS converts a fitted Type-1 TSK classifier/regressor to IT2 by creating upper and lower membership functions:

```
Type-1 Gaussian: (μ=5, σ=2)
         ↓
IT2 Membership:
  Upper: (μ=6, σ=2)     [μ + 0.5×σ]
  Lower: (μ=4, σ=2)     [μ - 0.5×σ]
```

### 2. Firing Strength Computation

For each input sample, firing strengths are computed for **both** upper and lower bound membership functions independently:

```
firing_upper[i,j] = firing strength of upper MFs for sample i, class j
firing_lower[i,j] = firing strength of lower MFs for sample i, class j
```

### 3. Type Reduction

The IT2 output interval is reduced to a crisp value:

**Simple averaging** (fast):
```
y_crisp = 0.5 × (firing_upper + firing_lower)
```

**Karnik-Mendel algorithm** (accurate):
Iteratively refines switch points to minimize the distance between left and right interval endpoints.

### 4. Classification / Regression

- **Classification**: `argmax(y_crisp)` selects the class with highest crisp firing strength
- **Regression**: Weighted average of firing strengths scaled to original target range

## Design Philosophy

**KISS (Keep It Simple, Stupid)**:
- Gaussian memberships only (other types added later)
- Probability norms only (5 norm families available for future extension)
- No antecedent refinement in v1 (will add optional refinement post-fit)
- Reuses all existing Type-1 FIS code and patterns

**Extensibility**:
- Easy to add trapezoidal/triangular memberships (already in Type-1)
- Easy to add other norm families (min/max, luk, hamacher, einstein)
- Easy to add post-fit refinement (use existing refinement module)
- Easy to add compiled Cython/GPU kernels (mirror Type-1 pattern)

## Testing

Three comprehensive test suites validate the implementation:

### 1. `test_it2_classifier_iris.py`

Classifies the iris dataset with IT2 uncertainty:
- Tests fit/predict cycle
- Verifies uncertainty width controls interval size
- Validates interval bounds (lower ≤ upper)
- Compares KM vs. averaging approaches

### 2. `test_it2_regressor_synthetic.py`

Nonlinear regression on synthetic data: `y = sin(3x) + noise`
- Tests fit/predict with prediction intervals
- Verifies interval validity
- Tests uncertainty width effect
- Checks KM vs. averaging

### 3. `test_it2_benchmark.py`

Hand-crafted IT2 model with known semantics:
- Validates firing strength computation
- Tests uncertainty near decision boundaries
- Verifies type-reduced output is within bounds
- Checks symmetry properties

## Advanced Usage Examples

### Trapezoidal IT2-FIS with Refinement

```python
from tribblefis.it2_classifier import IntervalType2FuzzyClassifier

clf = IntervalType2FuzzyClassifier(
    top_n=3,
    member_function="trap",      # Use trapezoid memberships
    uncertainty_width=0.3,        # Smaller bounds for crisp trapezoids
    km_iterations=10,
    refine=True,                  # Refine before IT2 conversion
    refine_l2_shrink=0.1,        # Moderate regularization
    random_state=42,
)
clf.fit(X_train, y_train)

# Get refined and bounded predictions
y_pred = clf.predict(X_test)
upper, lower = clf.predict_intervals(X_test)
```

### Triangular IT2-FIS for Interpretability

Triangular memberships (especially with Ruspini partitions) are more interpretable:

```python
clf = IntervalType2FuzzyClassifier(
    top_n=5,
    member_function="triangular",   # Interpretable triangle shape
    uncertainty_width=0.2,
    km_iterations=20,               # Higher precision for IT2 boundaries
    refine=False,                   # Skip refinement for simplicity
)
clf.fit(X_train, y_train)
```

### Regression with Uncertainty Quantification

```python
from tribblefis.it2_regressor import IntervalType2FuzzyRegressor

reg = IntervalType2FuzzyRegressor(
    top_n=4,
    n_gaussians=3,
    n_output_buckets=5,
    member_function="gaussian",
    uncertainty_width=0.7,          # Wider bounds for conservative intervals
    refine=True,
    km_iterations=None,             # Use fast averaging
)
reg.fit(X_train, y_train)

# Get point and interval estimates
y_point = reg.predict(X_test)
y_lower, y_upper = reg.predict_intervals(X_test)

# Interval width quantifies uncertainty
interval_width = y_upper - y_lower
print(f"Average interval width: {interval_width.mean():.3f}")
```

---

## Feature Completeness

### ✅ Implemented in v2

1. **All Membership Types**: Gaussian, trapezoidal, triangular (with type-aware uncertainty expansion)
2. **Antecedent Refinement**: Pre-conversion Type-1 refinement for discriminative bounds
3. **Flexible Type Reduction**: Simple averaging or Karnik-Mendel algorithm
4. **Confidence Intervals**: `.predict_intervals()` for uncertainty quantification
5. **Both Classification and Regression**: Full support for both task types

### 🔮 Future Extensions

1. **Norm Families**: Add all 5 families (probability is currently only option)
2. **GPU Acceleration**: PyTorch backend for large-scale models
3. **Hierarchical IT2**: IT2 fuzzy trees (leverage `tribble-tree`)
4. **Post-Fit IT2 Refinement**: Dedicated IT2 parameter optimization
5. **Large-Scale Optimization**: Evolutionary algorithms for IT2 learning
6. **Type-Reduction Variants**: Algorithms beyond KM (EIASC, centroid methods)

## Performance Notes

- **Classification**: Accuracy comparable to Type-1 baseline (~95% on iris)
- **Regression**: RMSE comparable to Type-1 baseline, with added uncertainty quantification
- **Type Reduction**: Averaging is ~10x faster than KM iterations; visual difference is small
- **Memory**: ~2x Type-1 (two membership functions per antecedent)

## References

- Karnik-Mendel Algorithm: "Type-2 Fuzzy Logic Systems" by Mendel (2001)
- Footprint of Uncertainty: Mizumoto & Tanaka (1976)
- TSK Inference: Takagi & Sugeno (1985)
