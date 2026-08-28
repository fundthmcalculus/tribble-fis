# Issue #164: Direct Weighted Fitting for Smooth Trapezoid EM

**Status:** Follow-up to #163 (Smooth Trapezoid EM Deployment)  
**Priority:** Medium (optimization, not blocking)  
**Effort:** 1.5-2 hours  

## Summary

The current smooth trapezoid EM implementation uses **importance resampling** to handle weighted samples during gate refinement (samples weighted by gate responsibilities). This introduces stochastic variance in the M-step.

A better approach: **Direct weighted fitting** that incorporates sample weights directly into the EM objective, avoiding resampling variance.

## Current Approach (Issue #163)

```python
# Importance resampling (stochastic)
p = gamma / total_gamma
idx_resampled = rng.choice(len(col), size=n_resample, replace=True, p=p)
col_resampled = col[idx_resampled]
memberships, weights, _ = fit_smooth_trapezoids_em(col_resampled, ...)
```

**Pros:** Simple, works, reuses existing `fit_smooth_trapezoids_em`  
**Cons:** Stochastic (resampling variance), not true weighted MLE

## Proposed Approach

**Direct weighted fitting** in `fit_smooth_trapezoids_em`:

```python
def fit_smooth_trapezoids_em(
    data_1d: np.ndarray,
    n_components: int,
    sample_weights: Optional[np.ndarray] = None,  # ← NEW
    n_bins: int = 50,
    max_iter: int = 100,
    ...
):
    # Initialize weighted histogram
    if sample_weights is not None:
        bin_counts, bin_edges = np.histogram(data_1d, bins=n_bins, weights=sample_weights)
    else:
        bin_counts, bin_edges = np.histogram(data_1d, bins=n_bins)
    
    # E-step and M-step handle weighted data naturally
    # (already support weighted responsibilities)
    
    return memberships, weights, log_likelihood
```

## Implementation Tasks

### Phase 1: Core Weighted Fitting
- [ ] Add `sample_weights` parameter to `fit_smooth_trapezoids_em()`
- [ ] Update histogram initialization to use `np.histogram(..., weights=sample_weights)`
- [ ] Verify E-step/M-step work correctly with weighted data
- [ ] Test with synthetic weighted samples

**Effort:** ~45 minutes

### Phase 2: Integration
- [ ] Update `_rebuild_gate_tree()` in `em.py` to pass weights instead of resampling
  ```python
  # Instead of resampling:
  p = gamma / total_gamma
  idx_resampled = rng.choice(len(col), size=n_resample, replace=True, p=p)
  col_resampled = col[idx_resampled]
  
  # Use direct weighting:
  memberships, weights, _ = fit_smooth_trapezoids_em(
      col, n_components=n_terms, sample_weights=gamma, ...
  )
  ```
- [ ] Remove resampling code and RNG dependency
- [ ] Test that gate refinement still works

**Effort:** ~30 minutes

### Phase 3: Validation
- [ ] Benchmark: direct weighting vs current resampling
  - Quality comparison (log-likelihood)
  - Variance comparison (multiple runs)
  - Determinism check
- [ ] Add unit test for weighted fitting
- [ ] Compare convergence speed

**Effort:** ~30 minutes

### Phase 4: Documentation
- [ ] Update docstrings with weight parameter
- [ ] Add note to `EM_REFINEMENT.md` about direct weighting
- [ ] Document performance improvement (if any)

**Effort:** ~15 minutes

## Expected Benefits

| Aspect | Current | Direct Weighting | Gain |
|--------|---------|------------------|------|
| **Determinism** | Stochastic | Deterministic | ✓ Exact reproduction |
| **Variance** | Resampling variance | None | ✓ More stable |
| **Quality** | Good | Slightly better | ✓ Truer to data |
| **Code complexity** | Simple | Slightly more complex | (acceptable) |

## Acceptance Criteria

- [x] `fit_smooth_trapezoids_em()` supports `sample_weights` parameter
- [x] Integration into `_rebuild_gate_tree()` complete
- [x] All tests pass (existing + new)
- [x] Benchmark shows equal or better quality vs resampling
- [x] Deterministic behavior verified
- [x] Documentation updated

## Testing Plan

```python
# Unit test: weighted fitting
data = np.array([1, 2, 3, 4, 5])
weights = np.array([0.5, 1.0, 1.0, 1.0, 0.5])  # Emphasize middle values

result1 = fit_smooth_trapezoids_em(data, n_components=2, sample_weights=weights)
result2 = fit_smooth_trapezoids_em(data, n_components=2, sample_weights=weights)
assert np.allclose(result1[0][0].a, result2[0][0].a)  # Deterministic

# Benchmark: gate refinement with direct weights
# Before: relies on resampling RNG
# After: deterministic, no variance across runs
```

## Notes

- This is a **pure improvement**, no risk to current functionality
- Current implementation works fine; this is optimization only
- Can be done independently of other EM enhancements
- Low-priority but good to have for production robustness

## Related

- Closes: (This issue resolves the "direct weighting" part of) #163 follow-up
- Depends on: #163 (Smooth Trapezoid EM)
- See also: `EM_REFINEMENT.md` Sec.5 (gate M-step design)

## Labels

- `optimization`
- `enhancement`
- `em-refinement`
- `triangular-fis`

## Assignee

(Open to contributors)
