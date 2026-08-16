# Should this library extend to general type-2 (GT2) fuzzy inference?

**Answer: go, as an opt-in extension of the existing IT2 architecture, not a
parallel system.** The standard alpha-plane decomposition (Mendel, Liu 2008)
reuses `it2_kernel.karnik_mendel_tsk` and `it2_firing_strengths` completely
unchanged, called K times instead of once. Measured at this repo's own
benchmark shapes, that costs almost exactly K times today's IT2 cost with no
combinatorial blowup -- K=20 alpha-planes on a 50k-row forward pass is 4.6s
and on the Karnik-Mendel search is 156ms, both linear in K to within a few
percent. This is issue #122's research spike; no production code changes
result from it.

## What alpha-plane GT2 actually computes

A GT2 secondary membership grade is a full type-1 fuzzy set over `[0, 1]` at
every primary point, not IT2's flat interval. The alpha-plane trick avoids
inventing new inference machinery for that: for a finite set of levels
`alpha in {alpha_1, ..., alpha_K}`, take the alpha-cut of the secondary grade
at each level. For any secondary MF that is convex in its own membership
value (Gaussian-on-Gaussian, triangular -- the only families worth
considering per the KISS design philosophy in `IT2_GUIDE.md`), that alpha-cut
is a single interval, so **each alpha-plane is an ordinary IT2 set** -- an
`(upper_mf, lower_mf)` pair, structurally identical to today's
`IT2GaussianMembership`. Level `alpha -> 0` recovers today's IT2 set (the
full footprint of uncertainty); level `alpha -> 1` collapses to the
"principal" membership function, the single most-likely one.

So GT2 inference is: decompose into K alpha-planes, run today's *unmodified*
IT2 pipeline on each, then combine the K results with an alpha-weighted
average (weight = alpha itself, the standard rule from Mendel/Liu -- higher
alpha-levels are more certain and are weighted more).

## Reuse assessment (issue #122, ask 1)

| Component | Today (IT2) | Under GT2 (alpha-plane) | Reuse |
|---|---|---|---|
| Antecedent type | `IT2GaussianMembership(upper_mf, lower_mf)` -- 3 free params per slot (`mu`, `sigma_lower`, `sigma_upper`, shared `mu` -- see `it2_refine._slot_x0_and_bounds`) | New `GT2GaussianMembership`: the same 3 IT2 params plus **one** extra shape parameter controlling how sigma interpolates from `sigma_lower` (alpha=1) to `sigma_upper` (alpha->0) -- not K independent pairs | Wraps the existing type; +1 field |
| Alpha-plane extraction | n/a | New, small: `_extract_alpha_plane(gt2_mf, alpha) -> IT2GaussianMembership`, closed-form given the one shape parameter | New, ~10 lines, no search |
| Forward inference per plane | `it2_firing_strengths(X, it2_model, norms, ...)` | Same function, called once per plane on that plane's `IT2GaussianMixtureModel` | 100% reused, unmodified |
| Type reduction per plane | `karnik_mendel_tsk(rule_values, firing_lower, firing_upper)` | Same function, called once per plane | 100% reused, unmodified |
| Cross-plane combination | n/a | New: alpha-weighted average of the K planes' `firing_crisp` (classifier) or `(y_l, y_r)` (regressor) | New, ~10 lines, vectorized, not the bottleneck (see below) |
| Classifier predict | `argmax` over per-class crisp score | Same `argmax`, over the alpha-combined crisp score | 100% reused |
| Regressor predict | midpoint of `karnik_mendel_tsk`'s `(y_l, y_r)` | Same midpoint, over the alpha-combined `(y_l, y_r)` | 100% reused |
| Refinement loop shape | `it2_refine._iter_it2_gaussian_slots` / `_slot_x0_and_bounds` / `_apply_slot_params`: one slot per whole membership, bounded L-BFGS-B, L2 anchor to the sweep start, "never worse than the heuristic" guard, CV fitness via `refine._make_folds`/`_prepare_folds` | Same loop, same guard, same CV plumbing; only the per-slot parameter count grows (3 -> 4) | ~90% reused -- only the slot's x0/bounds/apply functions change |
| Regressor consequent re-solve | `_solve_it2_consequents`: closed-form ridge using the plane's midpoint firing strength as the design weight | Same closed-form solve, using the alpha-combined midpoint firing strength | 100% reused |

The critical design decision this table is built on: **K is an inference-time
constant, not a stored free parameter.** Refinement searches the same order
of magnitude of parameters as IT2 does today (3 per slot -> 4), not `K x`
more -- the K-plane loop only appears at evaluation time, inside the fitness
function, exactly the way `karnik_mendel_tsk`'s own iteration count
(`max_iterations`) already does not multiply the parameter count it searches
over.

## Cost multiplier (issue #122, ask 2)

Measured with `benchmarks/gt2_alpha_plane_probe.py`, which calls today's
exact `it2_firing_strengths`/`karnik_mendel_tsk` K times at this repo's own
benchmark shapes (`forward-large`'s 50k x 20 features x 8 labels x 4 MF for
the forward pass; 50k samples x 8 rules -- "a handful of output buckets" per
`IT2_GUIDE.md` -- for the Karnik-Mendel search):

| K | forward pass (`it2_firing_strengths`) | vs K=1 | per-plane overhead | KM search (`karnik_mendel_tsk`) | vs K=1 | per-plane overhead |
|---|---|---|---|---|---|---|
| 1  | 214.46 ms | 1.00x | -- | 7.23 ms | 1.00x | -- |
| 3  | 643.85 ms | 3.00x | 1.001x | 24.87 ms | 3.44x | 1.147x |
| 5  | 1092.34 ms | 5.09x | 1.019x | 39.13 ms | 5.41x | 1.083x |
| 10 | 2134.55 ms | 9.95x | 0.995x | 75.89 ms | 10.50x | 1.050x |
| 20 | 4587.05 ms | 21.39x | 1.069x | 155.93 ms | 21.57x | 1.079x |

Two findings drive the recommendation:

1. **The cost model is exactly what the mechanism predicts: linear in K, no
   hidden combination cost.** Per-plane overhead sits at 1.0-1.15x across two
   orders of magnitude of K, on two structurally different kernels. The
   alpha-weighted combination step (summing K arrays) is not a measurable
   fraction of the total even at K=20 -- it is dominated entirely by the K
   repeated calls into the existing (already-optimized) IT2 kernel.
2. **Absolute cost stays comfortably practical at this library's design
   point.** `IT2_GUIDE.md` is explicit that this codebase targets "small rule
   counts... a handful of output buckets" -- exactly the KM benchmark's
   shape, where even K=20 is 156ms for 50k samples. The forward pass is the
   more expensive of the two (dominated by `GaussianMembership.evaluate`, per
   the existing profiling in `benchmarks/README.md`), but a K=5-10 default
   -- 1.0-2.1s for 50k rows -- is still well inside what this project already
   accepts for `refine-*` training workloads, and nothing here would need a
   new kernel: it is the same Cython/numba-compiled path IT2 already has,
   called more times.

No numba/parallelization work is *required* to make this practical -- the
existing kernels already carry the compiled/parallel implementation, and the
K-loop itself parallelizes trivially (independent per-plane calls) if it
ever needs to.

## Minimal validation benchmark (issue #122, ask 3)

Recommended design, mirroring `test_it2_benchmark.py` (hand-crafted model)
and `test_it2_karnik_mendel.py` (brute-force oracle) -- not yet implemented,
since this spike commits to no production code:

A single-feature GT2 set with a **Gaussian primary MF and triangular
secondary MF** -- the standard textbook minimal nontrivial GT2 case (apex of
the triangle, i.e. alpha=1, at the principal/embedded Gaussian; alpha=0 at
the FOU boundary, exactly matching today's `IT2GaussianMembership`'s
`upper_mf`/`lower_mf`). Its centroid has three independent computations that
must agree, the same triangulation `test_it2_karnik_mendel.py` used for IT2:

1. **Ground truth**: direct numerical double integration over the primary
   domain and the secondary-grade domain, per Mendel's defining centroid
   formula for a GT2 set -- no alpha-planes involved, just brute-force
   quadrature.
2. **Alpha-plane sum, K -> large**: the standard formula,
   `centroid = sum(alpha_k * c_IT2(alpha_k)) / sum(alpha_k)`, where
   `c_IT2(alpha_k)` is that plane's ordinary IT2 centroid via
   `karnik_mendel_tsk` -- must converge to (1) as K grows, the same
   convergence check `test_it2_karnik_mendel.py` already runs for its own
   iteration count.
3. **The eventual implementation at a modest, fixed K** (e.g. 10) -- must
   land within a stated tolerance of (1), the same role
   `test_it2_karnik_mendel.py`'s brute-force oracle plays for `karnik_mendel_tsk`
   today.

This benchmark is what should exist *before* `karnik_mendel_tsk`'s GT2
analogue is trusted, exactly as the IT2 brute-force oracle existed before
`karnik_mendel_tsk` itself was.

## Recommendation and phased plan (issue #122, ask 4)

**Go**, phased the same way IT2 itself was built (#89 data model -> #103
forward inference -> #120 type reduction -> #121 refinement), each phase
gated behind the previous one's validation:

1. **Data model**: `GT2GaussianMembership` (IT2 params + 1 shape parameter)
   and `_extract_alpha_plane`. Validate containment only: alpha=0 recovers
   today's IT2 set exactly (regression test against existing
   `IT2GaussianMembership` output), and the extracted interval narrows
   monotonically as alpha increases.
2. **Forward inference**: `gt2_firing_strengths` looping
   `it2_firing_strengths` over K planes, classifier `predict` wired through
   the alpha-weighted combination + `argmax`. Validate against the
   hand-crafted analytic-centroid benchmark above (classifier side: the
   per-class alpha-combined score, not yet the full KM path).
3. **Type reduction**: loop `karnik_mendel_tsk` per plane for regression,
   combine into `(y_l, y_r)`. Validate against the same benchmark's full
   centroid, plus a containment test analogous to
   `test_it2_karnik_mendel.py`'s (the GT2 midpoint must lie in the widest,
   alpha=0 plane's interval).
4. **Refinement**: extend `it2_refine`'s slot to 4 parameters
   (`_slot_x0_and_bounds`/`_apply_slot_params` only); the coordinate-descent
   loop, L2 anchor, "never worse than heuristic" guard, and CV-fold plumbing
   in `refine.py` carry over unchanged.

Ship as opt-in (`n_alpha_planes` parameter, unset/`1` behaving exactly like
today's IT2 -- a K=1 alpha-plane decomposition literally *is* IT2, which is
also a useful invariant test at every phase). Recommended default K, from
the cost table above: **5-10** -- inside 2x the fully-refined
`refine-classifier-wide` benchmark's own training cost, per plane, at this
library's typical model sizes.

## What this does not settle

- The parametric secondary-membership family (triangular-in-sigma-space vs
  Gaussian-on-Gaussian) is a real design choice with no data behind it yet --
  triangular is recommended here only because it is the simplest closed-form
  option consistent with `IT2_GUIDE.md`'s stated KISS philosophy, the same
  reasoning that kept IT2 to Gaussian memberships in v1.
- Whether GT2's added expressiveness (non-uniform confidence within the FOU)
  actually improves accuracy or interval calibration over IT2 on real data is
  an empirical question phases 2-3's benchmarks do not answer -- they
  validate correctness of the *computation*, not whether the computation is
  worth its cost on real problems. That is a phase-3/4 evaluation, not this
  spike's.

## Reproducing

```bash
python -m benchmarks.gt2_alpha_plane_probe                    # default K = 3, 5, 10, 20
python -m benchmarks.gt2_alpha_plane_probe -k 2 4 8 --repeats 8
```
