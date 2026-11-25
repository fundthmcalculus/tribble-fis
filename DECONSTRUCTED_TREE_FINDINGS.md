# Findings: Deconstructing a Flat TSK Model into a User-Specified Tree

> Evaluation note. Status: **kept, recommended for real use on problems with a
> known physical/domain topology** — see "Disposition" below. Implementation:
> `tribble-tree/fuzzytree/topology.py`, `tribble-tree/fuzzytree/deconstruct.py`
> (`DeconstructedHierarchicalRegressor`), `tribble-tree/tests/test_deconstruct.py`,
> `tribble-tree/demo_deconstruct_synthetic.py`, `tribble-tree/cmapss_deconstruct_eval.py`,
> `tribble-tree/fuzzytree/conformal.py` / `tribble-tree/demo_conformal_intervals.py` (intervals, #268).

## Motivation

`fuzzytree`'s existing `HierarchicalFuzzyExpertsRegressor` (HME) builds
structure top-down: infer or pin a gate topology, then fit a *brand new*
`TribbleRegressor` from scratch on each leaf's row subset. This evaluates a
different construction proposed on a whiteboard: fit **one flat
`TribbleRegressor` on all features first**, let the **user supply the tree
topology** directly (no row subsetting, no re-fit antecedents), and
**deconstruct** the flat model's already-fitted rule base into the tree by
slicing its antecedents per feature group, then re-solving only the
consequents (closed-form ridge, leaf and branch alike).

Two questions this evaluation set out to answer:
1. Does the deconstructed tree's *branch combiner* correctly recover
   structure when it exists (a mechanism sanity check)?
2. On real, hard, physically-structured sensor data, does giving the model
   the *true* topology up front beat both the flat baseline and today's
   auto-discovered HME topology?

## Stage A — synthetic, known group structure

`demo_deconstruct_synthetic.py`: `y = 1.0*g(a,b) + 0.7*g(c,d) + 1.4*g(e,f)` +
noise, no cross-group interaction, so the true topology and true branch
weights are known by construction.

| Model | Test R² |
|---|---|
| Flat `TribbleRegressor` (no structure) | 0.938 |
| HME (auto topology) | 0.971 |
| Deconstructed tree, true topology, leaves supervised on blended `y` | 0.971 |
| Deconstructed tree, true topology, leaves supervised on **true group signal** | **0.985** |

The default mode (leaves supervised only on the root's `y`, since no
`leaf_targets` override was given) already matches HME. But its *leaves*
turned out not to be physically meaningful in that mode: G1's leaf tracked
its own true latent contribution well (R²=0.931), G2's did not (R²=**-2.42**)
even though the branch combiner still blended everything into a good final
`y`. This makes sense in hindsight — with no override, every leaf is
independently trying to predict the *whole* `y` from only its own features,
so a leaf's output is not "this group's isolated contribution," it's "this
group's best guess at everything." The branch combiner's fitted weights
confirmed this: all three came out ≈1.0 regardless of the true generating
weights (1.0, 0.7, 1.4) — consistent with leaves that already each encode a
full (if noisy) guess at `y`, not an isolated per-group signal.

Passing `leaf_targets` (each leaf supervised directly on its own true group
contribution) fixed this cleanly: per-leaf R² rose to 0.951 / 0.991 / 0.998,
the branch combiner recovered the true weights almost exactly (0.997 / 0.708
/ 1.396 vs. true 1.0 / 0.7 / 1.4), **and** the final `y` prediction improved
further to R²=0.985 — the best of every model tried. This is the mechanism
check passing: the deconstruction + affine branch combiner is mathematically
sound and does the right thing when leaves are told what they're actually
supposed to represent. It's also the empirical justification for the
whiteboard's two-fold ablation being two *genuinely different* things, not
a redundant check.

## Stage B — NASA N-CMAPSS DS02 (real turbofan RUL)

`cmapss_deconstruct_eval.py`, 40k train / 10k test rows subsampled from
`N-CMAPSS_DS02-006.h5`. Topology: `RUL -> {HP, LP} -> component -> flow/eff
leaf -> sensors`, exactly matching the second whiteboard photo, with sensors
assigned to components by turbofan station number (a domain-informed
starting proposal — see the caveat in the sensor assignment section below).

**Inputs are the 18 REAL, physically-measured channels only** — the 14 `X_s`
sensors plus the 4 `W` flight-condition channels. The 14 `X_v` *virtual*
sensors (model-derived/estimated quantities, not actual instrumentation —
e.g. unmeasured internal flows and stall margins) are excluded on purpose:
a deployed system wouldn't have them either, and (see the channel-set note
below) they were also quietly propping up the earlier numbers.

**Fold 1 (sensor → leaf specificity)** — each leaf vs. its own true
(normally-unobservable) health-parameter column:

| Leaf | Test R² |
|---|---|
| FAN_flow, FAN_eff, LPC_flow, LPC_eff, HPC_flow, HPC_eff | **1.000** |
| HPT_flow | 1.000 |
| HPT_eff | 0.751 |
| LPT_eff | 0.683 |
| LPT_flow | **0.297** |

Seven of ten leaves recover their true health parameter essentially exactly.
LPT is now the clear weak point — LPT_flow in particular is a poor fit with
only `T50, P50` to work with, suggesting the LPT leaf needs a better
real-sensor grouping (or that LPT flow degradation genuinely leaves a
fainter signature in the 2 real channels assigned to it than elsewhere).

**Fold 2 (leaf → root RUL)**:

| Model | Test R² | Test RMSE |
|---|---|---|
| Flat `TribbleRegressor` on all 18 real channels | 0.370 | 14.94 |
| HME (auto topology) | 0.405 | 14.52 |
| Deconstructed tree, leaves supervised only on RUL | **0.593** | 12.01 |
| Deconstructed tree, leaves supervised on true health params | **0.594** | 12.00 |

Restricting to real channels only is still the headline result: the
deconstructed tree beats both baselines by a wide margin (≈0.19 R² over
HME, ≈0.22 over flat). Skipping the intermediate health-parameter
supervision made effectively no difference here (0.593 vs. 0.594) — with
the virtual channels gone, the earlier "skipping supervision helps a
little" edge disappeared along with them.

### Channel-set ablation: real-only vs. all 32 channels

An earlier pass (including the 14 `X_v` virtual sensors, 32 channels total)
found flat and HME collapsing much harder:

| Model | R², 32 channels (real+virtual) | R², 18 channels (real only) |
|---|---|---|
| Flat `TribbleRegressor` | **-0.063** | **0.370** |
| HME (auto topology) | 0.191 | 0.405 |
| Deconstructed tree (RUL-only leaves) | 0.628 | 0.593 |

Dropping the virtual channels is a large, direct win for both baselines
(flat: +0.43 R²; HME: +0.21 R²) and confirms the mechanism suspected
earlier: several virtual columns are near-duplicates of each other
(`W22`/`W25`/`W31`/`W32` carried identical differentiation scores in the
diagnostic output), and fitting one global antecedent structure across all
of them at once landed in exactly the degenerate-firing regime
`ZERO_FIRING_THRESHOLD`/normalization code elsewhere in this repo already
guards against. The deconstructed tree, which never fits one global
antecedent block over all channels at once, was far less exposed to this in
the first place — its own real-channel-only number (0.593) is close to its
32-channel number (0.628), a small give, not a collapse. Net effect:
removing the collinear virtual channels closes most but not all of the gap
to the baselines — the deconstructed tree still leads by a wide margin, but
this ablation shows part of its earlier edge really was "baselines tripping
over degenerate collinear inputs" rather than purely "structure helps,"
and that distinction matters for how strong a claim to make going forward.

### Sensor → component assignment (starting proposal, not verified)

```
FAN: P2, P15, P21, Nf
LPC: T24, P24
HPC: T30, Ps30, Nc
HPT: T48, Wf, P40
LPT: T50, P50
```
14 real `X_s` sensors, one component each, from turbofan station numbers,
with flight-condition columns (`alt, Mach, TRA, T2`) fed into every leaf as
shared covariates. `P40` (HPC/HPT boundary) was assigned by judgment call,
not verified against N-CMAPSS's own documentation/model description. LPT's
weak Fold-1 score is the first place to look if this grouping gets
revisited — it has only 2 real channels to work with, fewer than any other
component.

## Stage C — derived topology when no domain topology exists (#226)

`DeconstructedHierarchicalRegressor.fit` required a fully-specified topology
dict and had no fallback, so a dataset with no known physical structure could
not use the estimator at all without someone inventing a grouping by hand —
which is what `grad-school`'s `table_tribbletree_tsk_order.py` had to do for
Concrete and Body Fat, with no automated alternative to compare against.
`fuzzytree.auto_topology` adds the three strategies #226 asks for, reachable as
`fit(X, y, topology="auto")`.

`demo_auto_topology.py` re-runs the Stage A generator above with those arms
added (N=3000, 75/25 split, 3 seeds, `n_gaussians=3`):

| Model | Test R² |
|---|---|
| Flat `TribbleRegressor` (no structure) | 0.879 ± 0.008 |
| Deconstructed, **true** topology | 0.917 ± 0.014 |
| Deconstructed, affinity `k=2` | 0.917 ± 0.012 |
| Deconstructed, affinity `k=3` | 0.916 ± 0.013 |
| Deconstructed, affinity `k=4` | 0.923 ± 0.020 |
| Deconstructed, `per_feature` (zero-knowledge floor) | 0.933 ± 0.011 |
| Deconstructed, **`topology="auto"`** (selected) | **0.934 ± 0.011** |

### Stage A cannot validate a topology-derivation strategy, and that is the finding

The generator draws all six features as independent uniforms, and its group
functions are additively separable in their own members
(`g1 = 5·sin(a/2) + 0.5·b`, and likewise for `g2`, `g3`). So the whole of `y`
is additively separable across all six features, and **the "true" topology
`G1=[a,b] G2=[c,d] G3=[e,f]` is a labelling convention from how the generator
was written — it leaves no trace in the data.** Measured max off-diagonal
`|corr|` among the six features: **0.042**, i.e. noise.

No derivation strategy can recover that grouping, and none should be judged on
whether it does. It also means the correlation-affinity strategy (#226's
option 1) is being run here on a dataset with no affinity structure to find,
which is the honest reading of why `affinity_k2`/`k3` land exactly on the true
topology's number rather than above it: on this data the benefit is the
*deconstruction itself* — per-leaf TSK plus an affine combiner — not the
grouping being right.

### Three things the table does say

1. **Every deconstructed arm beats flat** (0.879), grouping or no grouping.
2. **The zero-knowledge floor lands above the hand-authored topology**
   (0.933 vs 0.917) — indicative, not established: over 3 seeds with spreads of
   ±0.011 and ±0.014, that gap is about one standard deviation and the bars
   overlap. The *reason* to expect it is the stronger evidence than the number:
   `y` is additively separable, so one leaf per feature under an affine
   combiner *is* the generating form, and grouping features together only
   constrains it. A domain topology is a hypothesis about structure, and on a
   problem with no group structure it is a wrong one. Flat is the only arm
   whose gap to the others clears the seed spread.
3. **Selection does its job.** `topology="auto"` lands at 0.934 — at or above
   every fixed strategy, including the floor — by picking per seed
   (`per_feature`, `affinity_k4`, `per_feature`) rather than committing to a
   `k` in advance. That is the whole argument for option 2 over option 1: `k`
   is not knowable a priori, and on this data the right answer is "no grouping".

### What this does not show

**Nothing here demonstrates that a derived topology approaches a domain one
where a real domain topology exists.** Stage B (N-CMAPSS, R² 0.593 with the
turbofan-station topology against 0.405 flat) is the case that matters for that
claim, and it has not been re-run with `topology="auto"` — the dataset is not
available from this repository. Until it is, the conservative reading stands:
**derive a topology when you have no domain knowledge, not instead of using it.**
`fit` records which was used in `topology_source_` (`"supplied"` / `"auto"`) so
a downstream table can never report one as the other.

## Stage D — calibrated prediction intervals (#268)

Phase 0 of `docs/hierarchical-it2-design.md` (#147). Both tree regressors
take `conformal_calibration=True`: `fit` holds out
`conformal_calibration_frac` (default 0.2) of the rows *before* topology
selection, gate building or any expert fit, and sizes a split-conformal
margin on them (`fuzzytree/conformal.py`, reusing
`tribblefis.regression.conformal_calibration_margin`). `predict_intervals`
returns `predict ∓ conformal_margin_`. HME's `refine_em()` in that mode runs
on the stored fit split only and re-sizes the margin afterward.

`demo_conformal_intervals.py`: Stage A generator and split, 5 seeds.

| estimator | alpha | target | coverage (mean, min–max) | width | width / range | R² |
|---|--:|--:|--:|--:|--:|--:|
| Deconstructed (true topology) | 0.1 | 0.90 | 0.905 (0.875–0.914) | 4.232 | 0.118 | 0.974 |
| Deconstructed (true topology) | 0.2 | 0.80 | 0.813 (0.789–0.838) | 3.308 | 0.092 | 0.974 |
| Deconstructed (`topology="auto"`) | 0.1 | 0.90 | 0.902 (0.888–0.922) | 3.452 | 0.096 | 0.983 |
| Deconstructed (`topology="auto"`) | 0.2 | 0.80 | 0.797 (0.763–0.847) | 2.684 | 0.075 | 0.983 |
| HME (gaussian gates) | 0.1 | 0.90 | 0.899 (0.876–0.928) | 4.230 | 0.118 | 0.974 |
| HME (gaussian gates) | 0.2 | 0.80 | 0.791 (0.758–0.818) | 3.181 | 0.088 | 0.974 |

- **Coverage is on target**: every mean is within 0.013 of `1 - alpha`
  (#268's acceptance bound is ±0.03), for both estimators and both topology
  sources.
- **Width tracks model accuracy**: the more accurate `"auto"` topology
  (R² 0.983) gets an interval ~18% narrower than the hand-authored one
  (0.974) at the same coverage -- a better model is rewarded with a tighter
  interval, which a fixed-footprint interval cannot do.
- **Cost**: holding out 20% costs no visible R² (0.974 mean over 5 seeds vs.
  Stage A's single-seed full-data 0.971).
- **Limitation**: width is constant per model, not per sample. It says how
  wrong the model tends to be, not where. See the adaptive variant below.

### Adaptive (per-sample) width for HME

`HierarchicalFuzzyExpertsRegressor(conformal_mode="adaptive")` uses
normalized split conformal: scores are `|y - ŷ| / s(x)` and intervals are
`ŷ ∓ q·s(x)`, where `s(x)` (`predict_scale`) is the gated mixture's
predictive std (law of total variance over leaves: within-leaf residual
variance plus between-expert disagreement). Leaf variances come from EM's
`sigma2_` after `refine_em`, else `leaf_sigma2_`, estimated on the fit split
at `fit`. Same marginal guarantee as `"constant"`.

`demo_conformal_intervals.py --adaptive`: Stage A signal, noise sd rising
linearly 0.05 → 2.55 along one feature, 5 seeds, target 0.90.

| noise along | mode | coverage | cov. low-noise ⅓ | cov. high-noise ⅓ | width low ⅓ | width high ⅓ | mean width | ρ(width, \|resid\|) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| `e` (gate split var) | constant | 0.897 | 0.991 | 0.772 | 6.23 | 6.23 | 6.23 | +0.039 |
| `e` (gate split var) | adaptive | 0.900 | 0.968 | 0.802 | 4.82 | 6.66 | 5.89 | +0.286 |
| `a` (rarely split) | constant | 0.903 | 0.969 | 0.831 | 6.44 | 6.44 | 6.44 | +0.007 |
| `a` (rarely split) | adaptive | 0.905 | 0.970 | 0.825 | 6.60 | 6.58 | 6.57 | +0.136 |

- **It works only along the gate partition.** When noise varies with the
  variable the gate tree splits on (`e`), adaptive mode moves width to where
  the noise is: high-noise coverage 0.772 → 0.802, low-noise over-coverage
  0.991 → 0.968, mean width −5%, ρ 0.04 → 0.29. When noise varies along a
  variable the tree barely splits on (`a`), nothing changes.
- **Why it is modest even when aligned.** `s(x)` has one value per leaf
  (blended at gate boundaries), so it can only be as fine as the partition
  (8 leaves here). And per-leaf residual variance mixes expert *bias* with
  noise: leaf variances here are 5–20 against noise variance 0.003–6.5, so
  most of the signal `s(x)` carries is "where the experts fit badly", not
  "where the data is noisy".
- **Recommendation.** Keep `"constant"` as the default. Use `"adaptive"`
  when heteroscedasticity is expected along variables the gates route on.
  Marginal coverage holds either way. A finer scale model, such as a
  separate regressor on `|resid|`, would adapt along any feature, but it
  is a different design from per-leaf variance.

**Stage B is not yet measured.** `cmapss_deconstruct_eval.py` now ends with a
`conformal_coverage` block (HME and the RUL-only deconstructed tree, target
0.90); it was smoke-tested on a synthetic stand-in frame only, because the
N-CMAPSS file is not available in CI.

## Disposition

Kept, unlike the (rejected) adaptive-partitioning experiment this findings
note is styled after. `DeconstructedHierarchicalRegressor` is a genuine win
over both the flat baseline and today's HME on the one real dataset tested,
and the synthetic check confirms the mechanism itself (branch combiner
weight recovery) is sound. Recommended for problems where the caller
actually has domain knowledge of the right topology. It is no longer
unusable without one — `topology="auto"` (Stage C, #226) derives a grouping
and, on the Stage A generator, beats the hand-authored topology — but a derived
topology is a fallback, not an equivalent, and Stage B has not been re-run with
it.

Not yet done, and the natural next steps if this is picked up again:
- Classification (`TribbleClassifier`-based leaves) — out of scope here,
  the affine branch combiner doesn't generalize to class probabilities
  without a separate design.
- Cross-validated / multi-seed evaluation — Stage B is a single 40k/10k
  subsample and split; the RUL numbers above should be treated as
  indicative, not final, until repeated across seeds and larger samples.
- Revisit the LPT sensor assignment given the weak LPT_flow Fold-1 score
  (R² = 0.297) — it has only 2 real channels (`T50, P50`) to work with.
- The channel-set ablation shows part of the deconstructed tree's earlier
  margin over flat/HME was those baselines tripping over collinear virtual
  sensors, not purely structure helping — real numbers to cite are the
  18-real-channel ones above (0.593 vs. 0.405 vs. 0.370), not the original
  32-channel run.
- The leaf-bucket-dedup enhancement sketched in `deconstruct.py`'s design
  (merging buckets that project to identical clauses on a leaf's own
  features) was not needed to get these results and was not built.
- **Run Stage D's coverage block on N-CMAPSS** (`cmapss_deconstruct_eval.py`
  now includes it) to confirm the ±0.03 coverage bound on real data.
- **Re-run Stage B with `topology="auto"`.** Stage C shows a derived topology
  beating a hand-authored one on a generator with no group structure to find,
  which is the easy case. N-CMAPSS is the one with real structure, and until
  the auto path is measured there, "derive when you have no domain knowledge"
  is a recommendation resting on a benchmark that cannot test the alternative.
- A *supervised* affinity for the clustering strategy. The correlation matrix
  groups features that move together; what a leaf actually wants is features
  whose joint effect on the target is not separable, which is what
  `gauss_math.calculate_interaction_scores` measures. Not built, because Stage
  A is additively separable end to end and would have shown a lift of zero for
  it too — demonstrating it needs a benchmark with real interactions.
