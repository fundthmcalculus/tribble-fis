# General Type-2 Fuzzy Inference System (GT2-FIS) Guide

## Overview

The GT2-FIS implementation extends this package's interval type-2 (IT2) FIS
to general type-2 fuzzy sets via the **alpha-plane representation** (Mendel,
Liu 2008): each antecedent's secondary membership grade -- which point in its
footprint of uncertainty is more or less likely, not just the footprint's
boundary -- is decomposed into a finite number of alpha-cuts, each an
ordinary IT2 set. Every forward-inference and type-reduction call this
package needs is therefore `it2_kernel.it2_firing_strengths`/
`karnik_mendel_tsk`, run **unchanged**, once per alpha-plane; the only new
computation is combining the planes with an alpha-weighted average. See
`docs/gt2-evaluation.md` for the research spike (issue #122) this
implements, including the measured cost of the alpha-plane approach and the
architectural reuse assessment against IT2.

This is not a separate system from IT2 so much as IT2 with one more
membership per antecedent (`principal_mf`, the single most-likely membership
within the footprint) and a loop around IT2's own kernel. If that reads as
underwhelming, that is the point: the alpha-plane approach exists precisely
so GT2 does not need its own inference or type-reduction algorithm.

## Quick Start

### Classification

```python
from tribblefis.gt2_classifier import GT2TribbleClassifier
import pandas as pd
import numpy as np

X_train = pd.DataFrame({'feature1': [...], 'feature2': [...]})
y_train = np.array([0, 1, 2, ...])

clf = GT2TribbleClassifier(
    top_n=3,
    uncertainty_width=0.5,   # footprint of uncertainty, same meaning as IT2
    n_alpha_planes=5,        # number of alpha-planes to combine
    random_state=42,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# The widest (alpha=0, IT2-equivalent) footprint boundary:
upper, lower = clf.predict_intervals(X_test)
```

### Regression

```python
from tribblefis.gt2_regressor import GT2TribbleRegressor

reg = GT2TribbleRegressor(
    top_n=3,
    n_gaussians=2,
    uncertainty_width=0.5,
    n_alpha_planes=5,
    km_iterations=10,   # Karnik-Mendel iterations, per alpha-plane
    random_state=42,
)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
y_lower, y_upper = reg.predict_intervals(X_test)  # alpha=0 boundary
```

## Key Parameters

### `uncertainty_width` (default: 0.5)

Identical meaning to `IT2TribbleClassifier`/`IT2TribbleRegressor`: for a
learned Gaussian `(mu, sigma)`, the alpha=0 (widest) boundary is
`upper_mf = sigma * (1 + uncertainty_width)`,
`lower_mf = sigma * max(0.1, 1 - uncertainty_width)`. GT2 additionally
carries the *original* Type-1 sigma through unchanged as `principal_mf` --
the single most-likely membership, at alpha=1.

### `n_alpha_planes` (default: 5)

Number of alpha-cuts to combine (evenly spaced in `(0, 1]`; see
`gt2_kernel.default_alpha_levels`). Cost is linear in this parameter with no
combination overhead (measured in `docs/gt2-evaluation.md`): a forward pass
or Karnik-Mendel search at `n_alpha_planes=K` costs almost exactly `K` times
what the same call costs at `K=1`. 5-10 is the recommended range for this
library's typical model sizes (a handful of output-bucket rules); increase
it if the alpha-weighted combination needs to resolve a finer secondary
grade, decrease it (down to 1) to trade fidelity for speed.

`alpha=0` itself (today's plain IT2 footprint) always carries zero weight in
the combination -- it is the "no confidence information used" boundary case,
not a plane that participates in the weighted average -- so it is never one
of the `n_alpha_planes` evaluated; use `IT2TribbleClassifier`/
`IT2TribbleRegressor` directly (or `gt2_kernel.extract_alpha_plane_model(model, 0.0)`)
if that boundary alone is what you want.

### `km_iterations` (default: 10)

Passed through to *each* alpha-plane's own type reduction, exactly as in
`IT2TribbleClassifier`/`IT2TribbleRegressor`:

- **Classifier**: has no effect on the result, for the same reason it has
  none for IT2 -- each plane's per-class reduction is provably the midpoint
  of that plane's own interval (see `it2_kernel.karnik_mendel_type_reduction`).
- **Regressor**: `None`/`0` skips each plane's Karnik-Mendel switch-point
  search for a faster, approximate per-plane interval (plain weighted
  average); `10-50` runs the real search per plane before the
  alpha-combination.

### `refine_gt2` (default: False)

Post-conversion antecedent refinement directly on the GT2
`(upper_mf, lower_mf, principal_mf)` triple -- the GT2 analogue of
`IT2TribbleClassifier.refine_it2`/`IT2TribbleRegressor.refine_it2`. One
dimension wider than IT2's own coordinate descent:

**Classifier** (`gt2_refine.refine_gt2_antecedents`): cycles through one GT2
membership at a time (any of Gaussian, trapezoidal, or triangular -- #144),
searching its shared peak plus two independent non-negative spread gaps per
side (Gaussian: `(mu, sigma_lower, sigma_principal, sigma_upper)`, `mu`
shared across all three, `sigma_lower <= sigma_principal <= sigma_upper`
enforced by construction -- the GT2 analogue of IT2's own
`sigma_upper >= sigma_lower` invariant, needed for the same reason:
`GT2GaussianMembership.alpha_cut`'s narrowing property depends on that
ordering holding; trapezoid/triangular slots follow the identical pattern
one level wider, see `gt2_refine.py`'s module docstring). The objective is
the cross-entropy of the alpha-combined, row-normalized firing strengths.

**Regressor** (`gt2_refine.refine_gt2_regressor_antecedents`): the same
coordinate descent, but -- as with IT2 -- a regressor's antecedents are only
ever meaningful alongside consequents solved *for* them, so every candidate
re-solves the TSK consequents in closed form, weighted by the
alpha-weighted average of each plane's own midpoint firing strength, before
scoring held-out MSE through the full alpha-combined Karnik-Mendel path.

### `refine_gt2_n_sweeps` (default: 3) / `refine_gt2_l2_shrink` (default: 0.05)

Sweep count and L2 anchor strength, same role as IT2's own
`refine_it2_n_sweeps`/`refine_it2_l2_shrink`. The regressor additionally has
`refine_gt2_km_iterations` (per-plane Karnik-Mendel iterations for the
search objective; `None` falls back to `km_iterations`, or 15) and
`refine_gt2_n_folds` (cross-validation folds).

### Other Parameters

Same as `IT2TribbleClassifier`/`IT2TribbleRegressor`: `top_n`, `top_p`,
`n_gaussians`, `n_output_buckets`, `norm_conorm`, `random_state`.

## How It Works

### 1. Model Conversion

Identical to IT2's own conversion, plus one field (shown here for Gaussian;
trapezoid/triangular follow the same shared-peak, scaled-spread pattern --
see `gauss_data.widen_membership`/`GT2TrapezoidMembership`/
`GT2TriangularMembership`):

```
Type-1 Gaussian: (mu=5, sigma=2)
         v
GT2 Membership:
  upper_mf:     (mu=5, sigma=3.0)   [sigma * (1 + 0.5)]
  lower_mf:     (mu=5, sigma=1.0)   [sigma * max(0.1, 1 - 0.5)]
  principal_mf: (mu=5, sigma=2.0)   [the original Type-1 sigma, unchanged]
```

### 2. Alpha-Plane Decomposition

Each membership's secondary grade is modeled as **triangular over sigma**
(apex at `principal_mf.sigma`, base spanning
`[lower_mf.sigma, upper_mf.sigma]`) -- the simplest closed-form shape
consistent with this package's KISS design philosophy. Its alpha-cut at
level `alpha` is a linear interpolation from each side toward the apex
(`GT2GaussianMembership.alpha_cut`):

```
sigma_lo(alpha) = lower_mf.sigma + alpha * (principal_mf.sigma - lower_mf.sigma)
sigma_hi(alpha) = upper_mf.sigma - alpha * (upper_mf.sigma - principal_mf.sigma)
```

`alpha=0` recovers `[lower_mf.sigma, upper_mf.sigma]` exactly -- today's
plain IT2 footprint. `alpha=1` collapses both bounds onto
`principal_mf.sigma` -- a crisp Type-1 evaluation.

### 3. Forward Inference and Type Reduction

For `n_alpha_planes` levels in `(0, 1]`:

```
firing_crisp_alpha = it2_firing_strengths(X, alpha_cut(model, alpha), norms)   # classifier
(y_l_alpha, y_r_alpha) = karnik_mendel_tsk(rule_values, firing_lower_alpha, firing_upper_alpha)  # regressor
```

then combined with the Mendel/Liu alpha-weighted average
(`gt2_kernel.alpha_weighted_average`):

```
result = sum(alpha_k * result_alpha_k) / sum(alpha_k)
```

**Containment.** Every alpha-plane's footprint is a *subset* of the
`alpha=0` plane's own (widest) footprint, and `karnik_mendel_tsk` is
monotonic in its firing-bound arguments, so every plane's own output lies
inside the `alpha=0` plane's -- and since the combined output is a convex
combination of those nested results, it inherits the same containment. This
is what `predict()`'s alpha-combined estimate staying inside
`predict_intervals()`'s (alpha=0) bounds relies on; see
`tests/test_gt2_kernel.py` for the direct check and
`gt2_kernel.gt2_karnik_mendel_tsk`'s docstring for the argument.

### 4. Classification / Regression

- **Classification**: `argmax(firing_crisp)` where `firing_crisp` is the
  alpha-combined per-class score.
- **Regression**: the alpha-combined `(y_l, y_r)`; the prediction is
  `0.5 * (y_l + y_r)`.

## Design Philosophy

**KISS, same as IT2**:
- Gaussian, trapezoidal, or triangular memberships (#144 -- the alpha-plane
  kernel (`gt2_kernel.py`) is fully type-agnostic; only each type's own
  `alpha_cut` (`gauss_data.py`) needed type-specific math)
- Triangular secondary membership grade only (the simplest closed-form
  alpha-cut; other shapes, e.g. Gaussian-on-Gaussian, are a possible
  follow-up if a real workload needs the extra expressiveness)
- Reuses every IT2 kernel function verbatim, once per alpha-plane

**Extensibility**: mirrors IT2's own list -- other norm families, GPU
acceleration, and swapping `scipy.optimize` for the project's own
`optimizers` package all apply identically here, since the per-plane kernel
calls are exactly IT2's.

## Testing

Six test suites, mirroring IT2's own (`test_it2_*`) one-for-one:

1. **`test_gt2_data_model.py`** -- `GT2GaussianMembership.alpha_cut`
   containment (alpha=0 recovers IT2 exactly, alpha=1 collapses to
   `principal_mf`) and monotonic narrowing.
2. **`test_gt2_kernel.py`** -- hand-crafted GT2 models; the
   alpha-weighted-combination containment and convergence properties,
   validated directly against `it2_kernel.karnik_mendel_tsk` per plane.
3. **`test_gt2_classifier.py`** -- iris fit/predict, interval validity,
   uncertainty-width effect, `km_iterations`' documented no-op on
   predictions, alpha-plane-count convergence.
4. **`test_gt2_regressor.py`** -- synthetic nonlinear regression,
   interval/containment validity, uncertainty-width effect, and the
   convergence-to-Type-1-as-footprint-vanishes invariant IT2 itself is held
   to.
5. **`test_gt2_refine.py`** -- classifier refinement never increases
   training loss, actually moves parameters, preserves the
   `sigma_lower <= sigma_principal <= sigma_upper` ordering invariant.
6. **`test_gt2_regressor_refine.py`** -- regressor refinement with
   per-candidate consequent re-solving; same invariant preservation, plus
   containment of `predict()`'s point estimate in `predict_intervals()`'s
   bounds (not exact-midpoint equality, unlike IT2 -- see that test's
   docstring for why the two differ here).

## Performance Notes

Measured in `docs/gt2-evaluation.md` (`benchmarks/gt2_alpha_plane_probe.py`)
before this implementation existed, using the exact IT2 kernel calls this
module makes: cost is linear in `n_alpha_planes` to within a few percent,
with no measurable combination overhead, from `n_alpha_planes=1` through
`20` on both the forward pass and the Karnik-Mendel search.

## References

- Mendel, J.M., Liu, F., Zhai, D. (2009). "Alpha-Plane Representation for
  Type-2 Fuzzy Sets: Theory and Applications." IEEE Trans. Fuzzy Systems.
- Karnik-Mendel Algorithm: "Type-2 Fuzzy Logic Systems" by Mendel (2001)
- `docs/gt2-evaluation.md`: the research spike this implementation follows.
- `IT2_GUIDE.md`: the IT2 design this extends.
