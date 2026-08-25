# Findings: Deconstructing a Flat TSK Model into a User-Specified Tree

> Evaluation note. Status: **kept, recommended for real use on problems with a
> known physical/domain topology** — see "Disposition" below. Implementation:
> `tribble-tree/fuzzytree/topology.py`, `tribble-tree/fuzzytree/deconstruct.py`
> (`DeconstructedHierarchicalRegressor`), `tribble-tree/tests/test_deconstruct.py`,
> `tribble-tree/demo_deconstruct_synthetic.py`, `tribble-tree/cmapss_deconstruct_eval.py`.

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
starting proposal — see the caveat in the script's docstring and the sensor
assignment section below).

**Fold 1 (sensor → leaf specificity)** — each leaf vs. its own true
(normally-unobservable) health-parameter column:

| Leaf | Test R² |
|---|---|
| FAN_flow, FAN_eff, LPC_flow, LPC_eff, HPC_flow, HPC_eff | **1.000** |
| HPT_flow | 0.711 |
| HPT_eff | 0.733 |
| LPT_flow | 0.711 |
| LPT_eff | **0.345** |

Six of ten leaves recover their true health parameter essentially exactly.
HPT and LPT are visibly harder — LPT_eff in particular is a weak fit,
suggesting either the LPT-side sensor grouping needs revisiting or LPT
efficiency degradation genuinely leaves a fainter sensor signature in this
station region than the other four component/mode combinations.

**Fold 2 (leaf → root RUL)**:

| Model | Test R² | Test RMSE |
|---|---|---|
| Flat `TribbleRegressor` on all 32 sensors | **-0.063** | 19.40 |
| HME (auto topology) | 0.191 | 16.93 |
| Deconstructed tree, leaves supervised only on RUL | **0.628** | 11.48 |
| Deconstructed tree, leaves supervised on true health params | 0.607 | 11.79 |

This is the headline result: giving the model the real physical topology
and deconstructing one flat fit into it beats both baselines by a wide
margin on the actual task — not a small improvement, a qualitative one (the
flat model doesn't even beat predicting the mean). Skipping the
intermediate health-parameter supervision (leaves trained directly toward
RUL) came out slightly ahead of the "oracle" leaves (0.628 vs. 0.607) — a
mild edge, not a decisive one at this sample size, but it does NOT support
the assumption that physically-motivated intermediate targets are free
wins; asking a leaf to match an unobservable label that is itself only
loosely coupled to RUL can cost a little accuracy on the thing that
actually matters.

A plausible mechanism for why flat collapses so badly here: several of the
32 raw sensor/flight columns are near-duplicates in this dataset (e.g.
`W22`/`W25`/`W31`/`W32` carried identical differentiation scores across
every bucket in the diagnostic output), and fitting one global antecedent
structure across all of them at once is exactly the degenerate-firing regime
`ZERO_FIRING_THRESHOLD`/normalization code elsewhere in this repo already
has to guard against. Splitting the same sensors into 5-6-feature
per-component groups, each fit and solved independently, sidesteps that
collapse — which is the whole premise of the whiteboard's design, now with
a concrete real-data example of it mattering.

### Sensor → component assignment (starting proposal, not verified)

```
FAN: P2, P15, P21, Nf, SmFan, W21
LPC: T24, P24, W22, W25, SmLPC
HPC: T30, Ps30, P30, Nc, W31, SmHPC
HPT: T48, T40, Wf, W32, P40
LPT: P50, T50, P45, W48, W50, phi
```
from turbofan station numbers, with flight-condition columns (`alt, Mach,
TRA, T2`) fed into every leaf as shared covariates. Several of these sit at
component boundaries (`P40`/`T40`/`W32` near the HPC/HPT interface,
`P45`/`W48` near HPT/LPT) and were assigned by judgment call, not
verified against N-CMAPSS's own documentation/model description. The weak
LPT_eff Fold-1 result is the first place to look if this grouping gets
revisited.

## Disposition

Kept, unlike the (rejected) adaptive-partitioning experiment this findings
note is styled after. `DeconstructedHierarchicalRegressor` is a genuine win
over both the flat baseline and today's HME on the one real dataset tested,
and the synthetic check confirms the mechanism itself (branch combiner
weight recovery) is sound. Recommended for problems where the caller
actually has domain knowledge of the right topology — it is not a
replacement for HME's auto-discovery when no such knowledge exists.

Not yet done, and the natural next steps if this is picked up again:
- Classification (`TribbleClassifier`-based leaves) — out of scope here,
  the affine branch combiner doesn't generalize to class probabilities
  without a separate design.
- Cross-validated / multi-seed evaluation — Stage B is a single 40k/10k
  subsample and split; the RUL numbers above should be treated as
  indicative, not final, until repeated across seeds and larger samples.
- Revisit the HPT/LPT sensor assignment given the weak LPT_eff Fold-1 score.
- The leaf-bucket-dedup enhancement sketched in `deconstruct.py`'s design
  (merging buckets that project to identical clauses on a leaf's own
  features) was not needed to get these results and was not built.
