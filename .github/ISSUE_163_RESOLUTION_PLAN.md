# Issue #163: Resolve EM Trapezoid Gate Ineffectiveness

## Problem Statement

The EM-based trapezoid fitting approach in `trapz_math.py` suffers from fundamental mathematical limitations that make it unsuitable for fuzzy gate refinement in HierarchicalFuzzyExperts:

1. **Non-smooth objectives** (piecewise-linear trapezoid PDF) cause optimization to get stuck in local optima
2. **Mode-hugging pathology** (bounded-support incentive to concentrate mass) produces poor partitions
3. **No true MLE** (uses quantile-based knots instead of optimizing the likelihood)
4. **No sharpening capability** (unlike Gaussians which can sharpen via σ reduction during EM)

Benchmark: EM trapz achieves R²=0.43 vs. Gaussian R²=0.83 vs. Fast method R²=0.70 on standard datasets.

## Solution Approach

### Phase 1: Prototype & Validation (COMPLETED)
Explored smooth sigmoid-based trapezoid approximation to provide smooth gradients:
- **Result:** 2-16x faster, but 10-30% worse quality (inverted trade-off)
- **Finding:** Smooth approximation creates different objective, not smoother version of same objective
- **Conclusion:** Approximation-based approach doesn't work; removal is better path

### Phase 2: Deprecation (IN PROGRESS)

#### 2a. Add Deprecation Warning
**File:** `tribble-tree/fuzzytree/em.py`  
**Change:** Add warning in `_rebuild_gate_tree()` when trapezoid MFs provided:
```python
if not isinstance(old_mfs[0], GaussianMembership):
    warnings.warn(
        "EM refinement of trapezoid gates is deprecated and not recommended. "
        "Trapezoid MFs lack properties needed for effective EM optimization. "
        "Use Gaussian gates instead via gate_style='gaussian'. "
        "For trapezoid interpretability without EM, use the fast trapezoid method.",
        FutureWarning
    )
```

#### 2b. Update Documentation
**File:** `tribble-tree/EM_REFINEMENT.md`  
**Change:** Remove trapezoid gate option (Section 5, Option B):
- Section 4.2: Remove trapezoid MLE discussion
- Section 5: Remove "Option B" for trapezoid gates
- Add migration section: Recommend Gaussian gates or fast trapezoid method

**File:** `tribble-tree/fuzzytree/em.py`  
**Change:** Update docstrings in `refine_em_regressor` and `refine_em_classifier`:
```
If trapezoid gates are provided, a warning is issued and they are frozen
(not refined) during EM. For EM refinement, use gate_style='gaussian' 
in HierarchicalFuzzyExpertsRegressor/Classifier.
```

#### 2c. Add Test
**File:** `tribble-tree/tests/test_em_refinement.py`  
**Change:** Add test class `TestEMTrapezoidDeprecation`:
```python
def test_trapezoid_gate_warns_not_recommended(self):
    """Trapezoid+EM warns and documents the limitation."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Create HME with trapezoid gates, run EM
        model = HierarchicalFuzzyExpertsRegressor(
            variable_plan=VariablePlan(..., term_style="trapezoid"),
            ...
        ).fit(X, y)
        model.refine_em(X, y, max_iter=5)
        
        # Verify warning issued
        self.assertTrue(any("EM refinement of trapezoid" in str(w_.message) for w_ in w))
```

### Phase 3: Code Cleanup (FUTURE)
After deprecation period:
- [ ] Remove trapezoid branch in `_rebuild_gate_tree()`
- [ ] Simplify gate M-step logic (~50-100 LOC reduction)
- [ ] Keep `trapz_math.py` for non-EM use (backward compatibility)

### Phase 4: Migration Messaging (grad-school repo)
Document in research materials:
- Why trapezoid+EM doesn't work (mathematical limitations)
- How smooth approximations were explored and failed (prototype findings)
- Better alternatives (Gaussian gates for EM, fast method for trapezoids)
- When/if this might be revisited (via JAX-based smooth gradients)

## Migration Guide for Users

If you're using trapezoid gates with EM refinement:

### Option 1: Use Gaussian Gates (Recommended)
```python
model = HierarchicalFuzzyExpertsRegressor(
    variable_plan=VariablePlan(..., term_style="gaussian"),  # Changed from "trapezoid"
    ...
).fit(X, y)
model.refine_em(X, y)  # Now works well
```
**Benefits:** True MLE, sharpening works, proven quality

### Option 2: Use Fast Trapezoids (No EM)
```python
model = TribbleRegressor(
    member_function="trapezoid",  # Use trapezoid
    trapz_method="fast",  # No EM
).fit(X, y)
```
**Benefits:** Crisp shoulders, 5900x faster than EM, sufficient quality

### Option 3: Hybrid (Trapezoids + Expert-Only EM)
```python
model = HierarchicalFuzzyExpertsRegressor(
    variable_plan=VariablePlan(..., term_style="trapezoid"),
    ...
).fit(X, y)
# Note: Gates stay fixed, only experts are refined via EM
```
**Benefits:** Crisp gate shoulders, expert-level refinement

## Files Changed

- [ ] `tribble-tree/fuzzytree/em.py` — Add deprecation warning
- [ ] `tribble-tree/EM_REFINEMENT.md` — Remove trapezoid option
- [ ] `tribble-tree/tests/test_em_refinement.py` — Add deprecation test
- [ ] `README.md` — Add note about trapezoid+EM deprecation (if applicable)

## Backward Compatibility

**Breaking changes:** None yet (warning only). Trapezoid+EM still works but warns.

**Future break (3 major versions later):**
- Remove trapezoid branch from `_rebuild_gate_tree()`
- Raise error if trapezoid gates used with `refine_em()`

## Testing

Current tests should not be affected:
- All EM tests use Gaussian gates (not trapezoid)
- Trapezoid tests use fast method (not EM)
- No existing code combines trapezoid+EM in tests

## Related References

- Memory: `em-trapz-vectorization.md` — 10x EM speedup, still bad quality
- Memory: `trapz-performance-benchmark.md` — Fast method 5900x faster
- Memory: `trapz-fast-integration.md` — Fast method details
- Issue: #167 — Width regularization (partial workaround, not full fix)

## Future Options

If this needs to be revisited:

### Option A: JAX-Based Smooth Gradients
Use automatic differentiation to compute smooth gradients of exact (crisp) trapezoid objective.
- **Effort:** 2-3 days
- **Dependencies:** Add JAX
- **Quality:** Should match original EM or better
- **Status:** Deferred to future if needed

### Option B: Custom Analytic Gradient Smoothing
Hand-derive smooth gradient approximations.
- **Effort:** 1-2 days for derivation
- **Risk:** Subtle bugs hard to spot
- **Status:** Lower priority than JAX approach

## Sign-Off

**Technical Decision:** Remove EM support for trapezoid gates, document migration path.

**Rationale:**
- Mathematical limitations cannot be easily fixed
- Prototype confirmed smooth approximations don't work
- Better alternatives exist (Gaussian gates, fast method)
- Zero production uses found in codebase
- All tests use Gaussian gates already

**Risk:** Low (no breaking change, clear error messages, alternatives provided)

**Maintenance benefit:** ~5-10% reduction in EM code complexity
