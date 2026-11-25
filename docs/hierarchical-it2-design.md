# Where does an IT2 footprint of uncertainty attach to a `tribble-tree` model?

> **Status: design decided, implementation not started.** Issue #147 step 1
> (the attachment-point question). Research-spike format, after
> `docs/gt2-evaluation.md`. No production code changes.

**Answer: at the leaves of `DeconstructedHierarchicalRegressor` first,
propagated exactly through the branch combiners by interval arithmetic; at
the gates of `HierarchicalFuzzyExpertsRegressor` second, via a recursive
per-node Karnik-Mendel reduction.** GT2 then rides on either via the usual
K-alpha-plane loop over the whole tree. But the user-facing value a
hierarchical model would want from type-2 -- a calibrated interval -- is
available **now, with no type-2 at all**, by putting the split-conformal
wrapper #149 shipped at the root. Ship that first; build type-2 trees only
if a workload needs antecedent-level uncertainty specifically.

## The two tree models are structurally different

| | `DeconstructedHierarchicalRegressor` (`deconstruct.py`) | `HierarchicalFuzzyExpertsRegressor` (`hme.py`) |
|---|---|---|
| where memberships live | leaves only -- each leaf reuses the flat `TribbleRegressor`'s per-feature Gaussian models for its feature group (`_leaf_gmm`) | gates (`terms.build_split_terms`, trapezoid or Gaussian) **and** inside each leaf's own `TribbleRegressor` |
| leaf model | TSK, consequents re-solved by ridge (`solve_tsk_consequents`, `predict_tsk`) | a full `TribbleRegressor` / `TribbleClassifier` (hard-coded class, no factory) |
| branch / gate | affine combiner `a·child_outputs + a0`, firing fixed at 1 -- no fuzzy weighting | per-node partition of unity `M / M.sum()`; leaf weight = product along the path (`compute_responsibilities`) |
| output | linear up the tree | soft mixture `Σ R[:, leaf] · expert.predict(X)` |
| interval output today | none | none (`em.py`'s `sigma2_` is learned but not exposed) |

## Candidate attachment points

| option | where the FOU lives | how intervals propagate | recovers Type-1 at `uncertainty_width=0`? | reuse | verdict |
|---|---|---|---|---|---|
| **A. Deconstructed, leaf FOU** | each leaf's slice of the flat GMM, widened by `widen_membership` (same as `IT2TribbleRegressor`) | leaf: `karnik_mendel_tsk` -> `[y_l, y_r]`. Branch (order 0/1, affine): `lo = a0 + Σ(a_i⁺·lo_i + a_i⁻·hi_i)`, `hi` symmetric -- exact, not a bound | yes, to floating-point tolerance (KM on equal bounds = T1 TSK; interval arithmetic on degenerate intervals = point arithmetic) | `widen_membership`, `it2_firing_strengths`/`_extract_*_model`, `rule_consequent_values`, `karnik_mendel_tsk` -- unchanged | **phase 1** |
| B. HME, gate FOU, global KM | gate terms widened | leaves become KM "rules": path products of upper/lower raw gate memberships as firing bounds, `expert.predict` as rule values, one cross-leaf KM | **no** -- global normalisation of raw path products ≠ product of per-node normalised gates | high | rejected: breaks the T1 invariant every other type-2 path in this repo keeps |
| **C. HME, gate FOU, recursive per-node KM** | gate terms widened | bottom-up: each node's interval = KM over its children with gate firing bounds as weights and children's intervals as rule values (`y_l` from children's lower ends, `y_r` from upper -- KM is monotone in rule values) | yes -- with equal bounds each node's KM is exactly that node's partition-of-unity mixture | `karnik_mendel_tsk` per node; `compute_responsibilities` needs an interval twin | **phase 2** |
| D. HME, leaf FOU (IT2 experts) | inside each leaf's `IT2TribbleRegressor` | crisp responsibilities x expert intervals: a convex combination of intervals | yes | needs an expert-factory hook in `hme.py`; `em.py` reads T1 internals (`top_features_`, `model_`, `corr_terms_`, ...) | optional add-on to C; blocked on the EM coupling |
| E. Per-split FOU propagated as input uncertainty | split variable itself | would require interval-valued inputs to child nodes | -- | none | rejected: no machinery anywhere in either repo |

### Why A is first

- The deconstructed model has **no gates**, so "per leaf vs per split" is
  not a choice there -- leaves are the only place memberships exist.
- Its leaves already *are* slices of the flat model, so IT2-ising the flat
  model once (one `IT2GaussianMixtureModel`) gives every leaf its FOU with no
  per-leaf fitting. That is the same Type-1 -> widen conversion
  `IT2TribbleRegressor` uses.
- Branch propagation is exact and trivial for the default `order="1st"`
  (and `"0th"`). For `"2nd"`+ the combiner is not monotone in its inputs;
  use the min/max over interval corners (2^children, small at this
  library's fan-outs) or restrict IT2 to order ≤ 1. Recommend the
  restriction for v1.

### The #226 concern is resolved by A

#147's status comment worried that a per-leaf footprint "means something
different" depending on whether the topology was hand-authored or derived
by #226's auto-topology (now merged, #230). Under option A the footprint is
attached to the **flat model's** memberships *before* any topology is
applied; leaves only select which features' memberships they use. So the
FOU is topology-independent, and `topology="auto"` works unchanged --
including its k-fold candidate scoring, which should score candidates on
the midpoint prediction exactly as it scores T1 today.

## The caveat that sets priority

Everything #149 and #145 found about flat IT2/GT2 intervals carries into the
tree, because option A's leaf intervals are the same KM rule-disagreement
intervals:

- raw coverage is capped well below any target, regardless of
  `uncertainty_width`;
- interval width does not track error (ρ(width, |resid|) ≈ 0,
  `docs/gt2-secondary-family-evaluation.md`).

So option A alone would produce an interval a caller should not trust. The
supported answer for calibrated hierarchical intervals is therefore a
**root-level split-conformal margin** -- the exact mechanism of
`IT2TribbleRegressor(conformal_calibration=True)`, applied to
`DeconstructedHierarchicalRegressor.predict`. It is model-agnostic, needs no
type-2 code, and works with every topology source.

## Phased plan

0. **Conformal intervals for the tree models (no type-2).**
   `conformal_calibration` / `conformal_alpha` / `conformal_calibration_frac`
   on `DeconstructedHierarchicalRegressor` and
   `HierarchicalFuzzyExpertsRegressor`, plus `predict_intervals`. Validate
   coverage on the existing Stage A (synthetic) / Stage B (N-CMAPSS)
   protocol in `DECONSTRUCTED_TREE_FINDINGS.md`. **Done, #268** (Stage D:
   coverage within 0.013 of target on Stage A; Stage B pending a run on
   N-CMAPSS).
1. **Option A** (`IT2DeconstructedHierarchicalRegressor` or an
   `uncertainty_width` parameter): IT2 flat model -> leaf KM intervals ->
   exact affine propagation; `order` ≤ "1st". Invariant tests: width=0
   recovers T1 to floating-point tolerance; root interval contains the root midpoint; branch
   interval arithmetic matches brute-force corner enumeration. Combine with
   phase 0's conformal margin for calibrated output.
2. **Option C** for HME: interval twin of `compute_responsibilities` +
   per-node KM; EM refinement stays T1 (refine on the midpoint) in v1.
3. **GT2**: run the phase-1/2 IT2 tree once per alpha-plane
   (`gt2_kernel.extract_alpha_plane_model` on the flat model) and combine
   with `alpha_weighted_average` -- no tree-specific GT2 code.

## Recommendation

- Treat #147 step 1 as answered by this doc.
- Open phase 0 as its own issue now: it delivers calibrated intervals on
  hierarchical models immediately and is the yardstick phases 1-3 must beat.
- Keep phases 1-3 deprioritised until a workload asks for hierarchical
  **and** antecedent-level (not output-level) uncertainty, per the issue's
  own priority note.
