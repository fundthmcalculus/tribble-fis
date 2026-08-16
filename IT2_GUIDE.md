# Interval Type-2 Fuzzy Inference System (IT2-FIS) Guide

## Overview

The IT2-FIS implementation provides a production-ready interval type-2 fuzzy inference system that extends the existing Type-1 FIS framework. It automatically generates uncertainty intervals from learned membership functions, with optional Karnik-Mendel type reduction. Supports three membership function types (Gaussian, trapezoidal, triangular) with optional antecedent refinement.

## Quick Start

### Classification

```python
from tribblefis.it2_classifier import IT2TribbleClassifier
import pandas as pd
import numpy as np

# Your training data
X_train = pd.DataFrame({'feature1': [...], 'feature2': [...]})
y_train = np.array([0, 1, 2, ...])

# Create and fit classifier
clf = IT2TribbleClassifier(
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
# NOTE: this width is antecedent-boundary ambiguity, not a calibrated
# confidence score -- see "Interval Calibration" below before using it as one.
```

### Regression

```python
from tribblefis.it2_regressor import IT2TribbleRegressor

# Create and fit regressor
reg = IT2TribbleRegressor(
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
# NOTE: raw coverage plateaus well under any real target regardless of
# uncertainty_width -- pass conformal_calibration=True to fix this,
# see "Interval Calibration" below.
```

## Membership Function Types

The IT2-FIS supports Gaussian membership functions. This is the standard and recommended choice for most applications.

```python
clf = IT2TribbleClassifier(uncertainty_width=0.5)
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
IT2 upper/lower Gaussian antecedents (`it2_refine`), distinct from `refine`
above (which only ever touches the pre-conversion Type-1 model and never sees
the footprint of uncertainty it becomes).

**Classifier** (`IT2TribbleClassifier.refine_it2`,
`it2_refine.refine_it2_antecedents`): cycles through one IT2 Gaussian
membership at a time and runs a small bounded local solve on its
`(mu, sigma_lower, sigma_upper)` -- `mu` shared between the upper and lower
halves, `sigma_upper >= sigma_lower` enforced by construction so the search
cannot invert `firing_lower <= firing_upper` -- holding everything else
fixed, for `refine_it2_n_sweeps` sweeps. Each sub-problem's objective is the
cross-entropy of the type-reduced, row-normalized firing strengths (the
classifier has no consequents beyond the firing strength itself, the same
reasoning `refine.py` gives for why refining antecedents *is* the whole
Type-1 classifier). A candidate replaces the running best only on a strict
training-loss improvement, so refinement never returns a model worse than its
starting point.

**Regressor** (`IT2TribbleRegressor.refine_it2`,
`it2_refine.refine_it2_regressor_antecedents`): the same coordinate descent,
but a regressor's antecedents are only ever meaningful alongside consequents
solved *for* them, so every candidate evaluated during the search re-solves
the TSK consequents in closed form (ridge regression, weighted by each rule's
midpoint firing strength) before scoring held-out MSE through the full
Karnik-Mendel prediction path -- mirroring `refine.py`'s Type-1 regressor
coordinate descent (antecedents outer, LSE-fit consequents inner). The final
`y_bucket_mean_`/`corr_terms_` used by `predict`/`predict_intervals` are then
re-solved once more against the *full* training set for the refined
antecedents.

### `refine_it2_n_sweeps` (default: 3) / `refine_it2_l2_shrink` (default: 0.05)

Sweep count and L2 anchor strength for the classifier's `refine_it2`
coordinate descent. The regressor's `refine_it2_n_sweeps` plays the same
role; it has no `l2_shrink` (its ridge penalty is `l2_reg`, matching the base
regressor's own consequent solve) but does add `refine_it2_km_iterations`
(Karnik-Mendel iterations for the *search* objective; `None` falls back to
`km_iterations`, or 15) and `refine_it2_n_folds` (cross-validation folds for
the held-out MSE objective).

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
IT2 Membership (uncertainty_width=0.5), μ unchanged:
  Upper: (μ=5, σ=3.0)     [σ × (1 + 0.5)]     -- wider, more permissive
  Lower: (μ=5, σ=1.0)     [σ × max(0.1, 1 - 0.5)]  -- narrower, more restrictive
```
Both bounds share the same `μ`, only `σ` changes -- which means upper and
lower **always evaluate to exactly 1 at `x=μ` regardless of
`uncertainty_width`**. See "Interval Calibration" below for why this matters.

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
- Post-fit antecedent refinement (`refine_it2`) now implemented for both classification and
  regression, the latter re-solving TSK consequents in closed form per candidate
- Reuses all existing Type-1 FIS code and patterns

**Extensibility**:
- Easy to add trapezoidal/triangular memberships (already in Type-1)
- Easy to add other norm families (min/max, luk, hamacher, einstein)
- Easy to add post-fit refinement (use existing refinement module)
- Easy to add compiled Cython/GPU kernels (mirror Type-1 pattern)

## Interval Calibration

`predict_intervals()` means something different for the two tasks, and
neither is "a confidence score" out of the box. See
`docs/t1-it2-gt2-tradeoff.md` for the full investigation (issue #149).

**Classification**: the interval is antecedent-boundary ambiguity, not
correctness confidence, and it is fundamentally **non-monotonic** in the
predicted class's own firing strength. Because upper/lower share `μ` (see
above), `width = upper - lower` is exactly 0 at `x=μ` for *any*
`uncertainty_width`, rises as `x` moves away from `μ`, then falls back
toward 0 again as both bounds decay toward 0 in the tails -- a hump, not a
ramp. Depending on where a dataset's typical points land on that hump,
"correct" and "incorrect" predictions can end up on either side of it, so
interval width should not be read as a proxy for "how likely is this
prediction to be right." There is no `conformal_calibration` fix for this --
fixing the *sign* would require a different secondary-membership
representation (see #145), not a recalibration of this one.

**Regression**: the interval's width is bounded by how much the *rules*
disagree about which one should fire, weighted by their firing-strength
interval -- never by any actual residual/aleatoric noise estimate, because
each rule's own TSK consequent is a crisp point value. Empirically, this
caps coverage well under any real target (e.g. ~70% on a synthetic
benchmark) **no matter how high `uncertainty_width` is set** -- the ceiling
is fixed by the base Type-1 model's own per-rule consequent spread, not by
the antecedent footprint. Pass `conformal_calibration=True` to
`IT2TribbleRegressor`/`GT2TribbleRegressor` to fix this: it holds out
`conformal_calibration_frac` (default 0.2) of the training rows, and pads
both bounds with an additive split-conformal margin sized to hit
`1 - conformal_alpha` coverage (default target 90%) on unseen data. This
margin is not subject to the antecedent-disagreement ceiling, since it pads
the output directly instead of widening the underlying antecedents further.

```python
reg = IT2TribbleRegressor(
    conformal_calibration=True,
    conformal_alpha=0.1,   # target 90% coverage
)
reg.fit(X_train, y_train)
y_lower, y_upper = reg.predict_intervals(X_test)  # now actually ~90%-covering
```

## Testing

Six test suites validate the implementation:

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

Post-fit IT2 classifier antecedent refinement (`refine_it2_antecedents`, `refine_it2`):
- Refinement never increases training cross-entropy
- Refinement actually moves antecedent parameters (guards against the previous
  always-`0.001`-gradient stub silently doing nothing)
- Refinement preserves `firing_lower <= firing_upper` (guards against the
  independent-halves bug described above)
- `method="none"` is an identity no-op; an unknown method raises
- The `IT2TribbleClassifier(refine_it2=True)` option fits, predicts, and
  doesn't materially hurt training accuracy

### 6. `test_it2_regressor_refine.py`

Post-fit IT2 regressor antecedent refinement with per-candidate consequent
re-solving (`refine_it2_regressor_antecedents`, `refine_it2`):
- Refinement never increases held-out cross-validated MSE
- Refinement preserves `firing_lower <= firing_upper`
- `method="none"` still re-solves consequents for the (unchanged) antecedents
  rather than returning stale ones; an unknown method raises
- The `IT2TribbleRegressor(refine_it2=True)` option fits and predicts
  with the containment guarantee (`y_lower <= predict() <= y_upper`) intact,
  and doesn't drastically worsen RMSE

## Advanced Usage Examples

### Regression with Uncertainty Quantification

```python
from tribblefis.it2_regressor import IT2TribbleRegressor

reg = IT2TribbleRegressor(
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

### ✅ Implemented in v4

1. **All Membership Types**: Gaussian, trapezoidal, triangular (with type-aware uncertainty expansion)
2. **Antecedent Refinement**: Pre-conversion Type-1 refinement for discriminative bounds
3. **Real Karnik-Mendel Type Reduction**: `karnik_mendel_tsk` runs the actual switch-point
   search over rule consequents (regressor), numba-compiled and parallelized across
   samples; the classifier's per-class interval midpoint is closed-form (see above)
4. **Post-Fit IT2 Refinement**: `refine_it2` runs block coordinate descent directly on the
   IT2 upper/lower Gaussian antecedents, after conversion -- for both classification
   (cross-entropy objective) and regression (held-out MSE with a per-candidate closed-form
   consequent re-solve, `it2_refine.refine_it2_regressor_antecedents`)
5. **Confidence Intervals**: `.predict_intervals()`, with the regressor's KM path
   *guaranteeing* containment of `.predict()`'s point estimate by construction
6. **Both Classification and Regression**: Full support for both task types

### 🔮 Future Extensions

1. **GPU Acceleration**: PyTorch backend for large-scale models. Not just an
   IT2 gap -- IT2/GT2 call the plain-numpy `tsk_firing_strengths`, not the
   GPU-capable `kernel.firing_strengths` path `TribbleClassifier`/
   `TribbleRegressor` already use. See issue #146.
2. **Hierarchical IT2**: IT2 fuzzy trees (leverage `tribble-tree`).
   `tribble-tree` is Type-1 only today; this needs a design pass on how an
   IT2 footprint attaches to a tree leaf/gate before any implementation.
   See issue #147.
3. **Large-Scale Optimization**: Evolutionary algorithms for IT2 learning
4. **EIASC and other KM variants**: `karnik_mendel_tsk` implements the classic
   Karnik-Mendel switch-point search; faster variants (EIASC, Wu-Mendel closed forms)
   remain a possible follow-up, though with the small rule counts typical here (a
   handful of output buckets) the classic search already converges in a few iterations

Done, despite once being listed here: all 5 norm families (`min/max`,
`probability`, `luk`, `hamacher`, `einstein`) already work -- IT2/GT2 route
through the shared Type-1 kernel path, which has always supported them. The
`scipy.optimize` → `optimizers` package swap is also done (#119/#134/#136);
no `src/tribblefis` module imports `scipy.optimize` directly anymore.

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
