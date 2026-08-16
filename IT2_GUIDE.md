# Interval Type-2 Fuzzy Inference System (IT2-FIS) Guide

## Overview

The IT2-FIS implementation provides a production-ready interval type-2 fuzzy inference system that extends the existing Type-1 FIS framework. It automatically generates uncertainty intervals from learned membership functions, with optional Karnik-Mendel type reduction. Supports three membership function types (Gaussian, trapezoidal, triangular) with optional antecedent refinement.

## Quick Start

### Classification

```python
from tribblefis.it2_classifier import T2TribbleClassifier
import pandas as pd
import numpy as np

# Your training data
X_train = pd.DataFrame({'feature1': [...], 'feature2': [...]})
y_train = np.array([0, 1, 2, ...])

# Create and fit classifier
clf = T2TribbleClassifier(
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

The IT2-FIS supports Gaussian membership functions. This is the standard and recommended choice for most applications.

```python
clf = T2TribbleClassifier(uncertainty_width=0.5)
```

**Uncertainty expansion**: For a learned Gaussian (μ, σ), the IT2-FIS creates upper and lower bounds:
- Upper: μ with σ × (1 + 0.5) = σ × 1.5 (wider, more permissive)
- Lower: μ with σ × max(0.1, 1 - 0.5) = σ × 0.5 (narrower, more restrictive)

---

## Key Parameters

### `uncertainty_width` (default: 0.5)

Controls the footprint of uncertainty (FoU) by expanding the sigma of upper/lower membership functions:

- **Upper membership**: σ × (1 + uncertainty_width) — wider, more permissive
- **Lower membership**: σ × max(0.1, 1 - uncertainty_width) — narrower, more restrictive

**Effect**: Larger values create wider uncertainty intervals, representing greater ambiguity.

```
uncertainty_width = 0.2  → narrow intervals, tighter bounds
uncertainty_width = 0.5  → moderate intervals (recommended)
uncertainty_width = 1.0  → wide intervals, more conservative
```

### `km_iterations` (default: 10)

Regressor: caps the switch-point search in the real Karnik-Mendel algorithm
(`it2_kernel.karnik_mendel_tsk`), which combines every rule's own TSK
consequent value under its interval firing-strength weight to find the
type-reduced output interval `[y_l, y_r]`. The search provably converges in at
most `n_rules` refinements, so with the handful of output-bucket rules typical
here it usually settles in far fewer than the default cap.

Classifier: has no effect on the *result* (only on which code path runs). Each
class's own firing-strength interval is reduced independently, and the
centroid of a single interval under a uniform secondary membership function is
provably its midpoint -- there is no cross-rule switch point to search for a
lone interval (see `karnik_mendel_type_reduction`'s docstring for the proof).
`None`/`0` and any positive integer therefore compute the same
`firing_crisp` for the classifier; the option exists for API symmetry with the
regressor and to make that equivalence explicit rather than silent.

- **None or 0**: fast path -- regressor: plain weighted average per bound
  (no cross-rule optimization, not guaranteed to bracket the crisp estimate);
  classifier: same output as any other setting.
- **10-50**: regressor: exact Karnik-Mendel output interval (recommended).

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

### `refine_it2` (default: False)

If True, runs a **second**, post-conversion refinement pass directly on the
IT2 upper/lower Gaussian antecedents (`it2_refine.refine_it2_antecedents`),
distinct from `refine` above (which only ever touches the pre-conversion
Type-1 model and never sees the footprint of uncertainty it becomes).

**How it works**: cycles through one Gaussian half (upper or lower MF of one
IT2 membership) at a time and runs a small bounded local solve on just its
`(mu, sigma)`, holding everything else fixed, for `refine_it2_n_sweeps`
sweeps -- the IT2 analogue of the Type-1 `"coordinate"` method. Each
sub-problem's objective is the cross-entropy of the type-reduced,
row-normalized firing strengths (the classifier has no consequents beyond the
firing strength itself, the same reasoning `refine.py` gives for why refining
antecedents *is* the whole Type-1 classifier). A candidate replaces the
running best only on a strict training-loss improvement, so refinement never
returns a model worse than its starting point.

Regression has no `refine_it2` counterpart yet: the base regressor's
consequents are fixed at conversion time, and refining antecedents alone
without re-solving those consequents for each candidate is a materially
different (and currently unimplemented) undertaking -- see
`it2_refine.py`'s module docstring.

### `refine_it2_n_sweeps` (default: 3) / `refine_it2_l2_shrink` (default: 0.05)

Sweep count and L2 anchor strength for `refine_it2`'s coordinate descent.

### Other Parameters

Same as `TribbleClassifier` / `TribbleRegressor`:

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

Two different reductions happen depending on what's being combined:

**Classifier -- per-class interval midpoint** (always, regardless of `km_iterations`):
```
firing_crisp[:, j] = 0.5 × (firing_upper[:, j] + firing_lower[:, j])
```
Each class's own firing-strength interval is independent of every other
class's, and the centroid of one interval under a uniform secondary
membership function is provably its midpoint -- there is nothing to search.

**Regressor -- Karnik-Mendel over rule consequents** (`km_iterations` set):
The regressor's crisp output is a weighted average of every *rule's own* TSK
consequent value, and type-2 uncertainty makes each rule's weight an interval
rather than a single number. `karnik_mendel_tsk` finds the exact minimum and
maximum of that weighted average by sorting rules by consequent value and
searching for the switch point separating which rules get their lower vs.
upper weight (Karnik & Mendel, 2001) -- this is where a genuine iterative
search matters, and where it now actually runs (numba-compiled, parallel
across samples).
```
y_l, y_r = karnik_mendel_tsk(rule_values, firing_lower, firing_upper)
y_crisp  = 0.5 × (y_l + y_r)   # guaranteed to lie in [y_l, y_r]
```
`km_iterations=None`/`0` skips the search for a faster, approximate interval
(each bound is the plain weighted average using that bound's own raw firing
strengths, with no cross-rule optimization and no containment guarantee).

### 4. Classification / Regression

- **Classification**: `argmax(y_crisp)` selects the class with highest crisp firing strength
- **Regression**: `karnik_mendel_tsk` combines every rule's own TSK consequent value under
  its interval firing-strength weight into `[y_l, y_r]`; the prediction is `0.5 * (y_l + y_r)`

## Design Philosophy

**KISS (Keep It Simple, Stupid)**:
- Gaussian memberships only (other types added later)
- Probability norms only (5 norm families available for future extension)
- Post-fit antecedent refinement now implemented for the classifier (`refine_it2`);
  regression's version -- which would need to re-solve consequents per candidate -- remains
  future work (see Future Extensions)
- Reuses all existing Type-1 FIS code and patterns

**Extensibility**:
- Easy to add trapezoidal/triangular memberships (already in Type-1)
- Easy to add other norm families (min/max, luk, hamacher, einstein)
- Easy to add post-fit refinement (use existing refinement module)
- Easy to add compiled Cython/GPU kernels (mirror Type-1 pattern)

## Testing

Five test suites validate the implementation:

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

### 4. `test_it2_karnik_mendel.py`

Correctness of the real Karnik-Mendel search (`karnik_mendel_tsk`):
- Matches an independent brute-force grid-search oracle across random rule sets
- Single-rule and zero-firing degenerate cases
- Per-row independence under batching (parallel search doesn't cross-contaminate rows)
- `predict()` is the exact midpoint of `predict_intervals()`, and the KM path
  structurally guarantees containment (the property the old two-stage pipeline violated
  on ~3% of rows)

### 5. `test_it2_refine.py`

Post-fit IT2 antecedent refinement (`refine_it2_antecedents`, `refine_it2`):
- Refinement never increases training cross-entropy
- Refinement actually moves antecedent parameters (guards against the previous
  always-`0.001`-gradient stub silently doing nothing)
- `method="none"` is an identity no-op; an unknown method raises
- The `T2TribbleClassifier(refine_it2=True)` option fits, predicts, and
  doesn't materially hurt training accuracy

## Advanced Usage Examples

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

### ✅ Implemented in v3

1. **All Membership Types**: Gaussian, trapezoidal, triangular (with type-aware uncertainty expansion)
2. **Antecedent Refinement**: Pre-conversion Type-1 refinement for discriminative bounds
3. **Real Karnik-Mendel Type Reduction**: `karnik_mendel_tsk` runs the actual switch-point
   search over rule consequents (regressor), numba-compiled and parallelized across
   samples; the classifier's per-class interval midpoint is closed-form (see above)
4. **Post-Fit IT2 Refinement**: `refine_it2` runs block coordinate descent directly on the
   IT2 upper/lower Gaussian antecedents, after conversion (classifier only -- see below)
5. **Confidence Intervals**: `.predict_intervals()`, with the regressor's KM path
   *guaranteeing* containment of `.predict()`'s point estimate by construction
6. **Both Classification and Regression**: Full support for both task types

### 🔮 Future Extensions

1. **Norm Families**: Add all 5 families (probability is currently only option)
2. **GPU Acceleration**: PyTorch backend for large-scale models
3. **Hierarchical IT2**: IT2 fuzzy trees (leverage `tribble-tree`)
4. **Regression `refine_it2`**: post-conversion antecedent refinement against held-out
   MSE, re-solving consequents per candidate the way `refine.py`'s Type-1 regressor
   path does (the classifier's `refine_it2` doesn't need this, since it has no
   consequents to re-solve)
5. **Large-Scale Optimization**: Evolutionary algorithms for IT2 learning
6. **EIASC and other KM variants**: `karnik_mendel_tsk` implements the classic
   Karnik-Mendel switch-point search; faster variants (EIASC, Wu-Mendel closed forms)
   remain a possible follow-up, though with the small rule counts typical here (a
   handful of output buckets) the classic search already converges in a few iterations

## Performance Notes

- **Classification**: Accuracy comparable to Type-1 baseline (~95% on iris)
- **Regression**: RMSE comparable to Type-1 baseline, with added uncertainty quantification
- **Type Reduction**: measured at ~7M rows/s (5 rules/row, post-JIT-warmup) for the
  regressor's real Karnik-Mendel search -- it replaced a pure-Python
  `for sample: for rule:` loop that called an iterative-but-never-converging
  function that always just returned the interval midpoint anyway (see
  `karnik_mendel_type_reduction`'s docstring); the classifier's per-class reduction
  is now the closed-form midpoint directly, with no iteration at all
- **Memory**: ~2x Type-1 (two membership functions per antecedent)

## References

- Karnik-Mendel Algorithm: "Type-2 Fuzzy Logic Systems" by Mendel (2001)
- Footprint of Uncertainty: Mizumoto & Tanaka (1976)
- TSK Inference: Takagi & Sugeno (1985)
