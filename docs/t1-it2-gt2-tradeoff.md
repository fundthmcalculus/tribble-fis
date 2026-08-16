# T1 vs IT2 vs GT2: is the extra machinery worth its construction cost?

> **Status: extended (#143), superseding the single-dataset first pass.**
> Six datasets, five seeds on the flagship pair, and an interval-calibration
> metric now back this doc. Still not a powered study -- see "What this does
> not settle" -- but the headline verdict is now backed by more than one
> dataset and one seed.

**Answer: no, not for point-estimate accuracy -- and the uncertainty the
footprint claims to carry does not hold up either.** Across six datasets,
sweeping GT2's `n_alpha_planes` moved point-estimate performance by at most
0.017 (usually far less). Seed robustness confirms the single-seed read
wasn't noise. And the new finding: `predict_intervals()`'s empirical
coverage ranges 4-57% depending on dataset -- nowhere near reliable --
and classification's interval width is, if anything, *inversely* related to
correctness (smaller on errors, not larger) on every dataset tested. The
only knob that reliably helps either metric is post-fit refinement, which
costs 85-1400x the construction time.

## Setup

- **Classification**: `wine` (178x13, 3 classes), `breast_cancer` (569x30,
  2 classes), `digits` (1797x64, 10 classes) -- all sklearn built-ins, no
  network fetch.
- **Regression**: `make_friedman1` (500x10, 5 informative, nonlinear),
  `diabetes` (442x10, real data), `make_regression` (500x10, 5 informative,
  linear-ish) -- likewise no network fetch.
- **Sweep** (per dataset, seed 42): `n_gaussians` 1-5 for T1/IT2/GT2; GT2
  additionally sweeps `n_alpha_planes` in {1,2,3,5,8,10} at `n_gaussians=3`.
- **Refine**: `refine_it2=True` only on the flagship pair (`wine`,
  `friedman1`) -- its cost was already characterized as large and highly
  variable in the first pass, so it isn't repeated per new dataset or seed.
- **Seed robustness**: T1/IT2/GT2 at `n_gaussians=2` (no refine) repeated
  over 5 seeds on `wine`/`friedman1`.
- **Calibration**: regression gets empirical coverage (fraction of test
  points inside `predict_intervals()`) and mean interval width relative to
  the target's observed range. Classification's `predict_intervals()`
  returns per-class *firing-strength* bounds, not a label interval, so
  there's no coverage analogue there -- instead, mean width of the
  predicted class's interval, split by correct vs. incorrect predictions.
  Full sweep (152 fits) in `results/t1-it2-gt2-tradeoff.json`.

## Results

### 1. Alpha-plane count still barely moves point-estimate performance

GT2 `n_alpha_planes` 1→10 spread in final performance, by dataset:

| dataset | metric spread (max − min across K) |
|---|---|
| classification/wine | 0.0000 |
| classification/breast_cancer | 0.0058 |
| classification/digits | 0.0167 |
| regression/make_regression | 0.0003 |
| regression/friedman1 | 0.0038 |
| regression/diabetes | 0.0054 |

Largest observed spread is 0.017 accuracy points, on `digits` -- still
nowhere near enough to change a deployment decision, and every other
dataset is smaller still. This generalizes the first pass's single-dataset
"K is nearly inert" finding.

### 2. Seed robustness confirms the type-1-parity finding, and adds a wrinkle

Mean ± population stdev over 5 seeds, `n_gaussians=2`, no refine:

| task | family | performance | fit time |
|---|---|---|---|
| classification (wine) | T1 | 0.841 ± 0.054 | 40.1 ± 2.8 ms |
| classification (wine) | IT2 | 0.833 ± 0.056 | 43.2 ± 2.6 ms |
| classification (wine) | GT2 | 0.841 ± 0.054 | 43.0 ± 3.3 ms |
| regression (friedman1) | T1 | 0.784 ± 0.002 | 23.6 ± 0.7 ms |
| regression (friedman1) | IT2 | 0.682 ± 0.023 | 24.6 ± 1.1 ms |
| regression (friedman1) | GT2 | 0.771 ± 0.007 | 23.6 ± 2.1 ms |

Classification: T1 and GT2 are identical to three decimal places, IT2 a
hair behind -- consistent with "no accuracy win," not a single-seed fluke.

Regression has a wrinkle worth stating precisely: **GT2's alpha-weighted
combination lands much closer to T1 than plain IT2 does, and is more seed-
stable than IT2** (std 0.007 vs. IT2's 0.023). GT2 doesn't beat T1, but
alpha-combining IT2's raw footprint recovers most of what the footprint
alone gives up. That's a genuine, non-obvious benefit of the alpha-plane
average -- just not the "better than IT2" story `docs/gt2-evaluation.md`
was checking for, and not one that changes the top-line verdict (T1 is
still on the frontier, faster, for effectively the same number).

### 3. Interval coverage is unreliable, and nobody claimed otherwise

`uncertainty_width` is documented as a heuristic sigma-widening factor
(`upper_mf.sigma = sigma * (1 + uncertainty_width)`), not a calibrated
nominal coverage target -- so there was never a specific number to hit. What
this measures is how far an intuitive default (`uncertainty_width=0.5`)
actually lands from anything usable as a confidence interval:

| dataset | family | coverage | mean relative width |
|---|---|---|---|
| friedman1 | IT2 | 0.573 (range 0.36-0.86 across configs) | 0.256 |
| friedman1 | GT2 | 0.276 (range 0.00-0.74) | 0.079 |
| diabetes | IT2 | 0.189 (range 0.14-0.22) | 0.088 |
| diabetes | GT2 | 0.053 (range 0.00-0.08) | 0.024 |
| make_regression | IT2 | 0.133 (range 0.10-0.15) | 0.010 |
| make_regression | GT2 | 0.042 (range 0.00-0.07) | 0.003 |

Coverage never exceeds 57%, and on two of three datasets it's under 20%.
**GT2 is consistently *less* covered than IT2, at a narrower width** --
the alpha-weighted average pulls the combined interval in from IT2's own
(alpha=0) footprint, which trades away coverage for a tighter-looking band
with nothing backing up the extra confidence. If `predict_intervals()` is
being read as "the true value is probably in here," this data says don't,
at the current default width, for either family.

Classification's version of the same question: does interval width track
error likelihood? Mean predicted-class interval width, correct vs.
incorrect predictions:

| dataset | family | width (correct) | width (incorrect) |
|---|---|---|---|
| wine | IT2 | 0.240 | 0.199 |
| wine | GT2 | 0.250 | 0.216 |
| breast_cancer | IT2 | 0.380 | 0.343 |
| breast_cancer | GT2 | 0.403 | 0.375 |
| digits | IT2 | 0.318 | 0.287 |
| digits | GT2 | 0.328 | 0.296 |

Every single row runs backwards: incorrect predictions get a *narrower*
interval than correct ones, on all three datasets, both families. A
calibrated uncertainty signal should do the opposite -- wider on the ones
it gets wrong. This isn't noise (six for six), and it isn't a training-time
concern already flagged elsewhere in this doc; it's the interval itself
carrying a wrong-signed correlation with the thing it's supposed to signal.

### 4. Refinement is still the whole story, and still expensive

| task | unrefined | refined | delta | time multiplier |
|---|---|---|---|---|
| classification (wine) | 0.926 acc | 0.982 acc | +0.056 | 85x |
| regression (friedman1) | 0.639 R² | 0.880 R² | +0.241 | 1388x |

Consistent with the first pass's 82-89x / 1325-1460x (small run-to-run
timing noise, same order of magnitude both times).

## What this does not settle

- Coverage/width were measured at a single `uncertainty_width=0.5`; whether
  a different width, or a genuinely calibrated wrapper (conformal
  prediction over the existing footprint, e.g.), fixes coverage is
  untested. See #149, the tracking issue this finding produced.
- Still no dataset where GT2 clearly beats IT2 on either metric or on
  calibration -- the "when does the extra machinery earn its cost" question
  remains open, now with more evidence pointing toward "rarely, on tabular
  data like this."
- `refine_it2`'s cost multiplier is still characterized on one dataset each;
  not repeated across the newly added datasets (see Setup).

## Next steps

Tracked as GitHub issues, in priority order:

0. ~~Fix stale claims in `IT2_GUIDE.md`~~ -- done, #142.
1. ~~Extend this benchmark~~ -- done, #143 (this update).
2. **New, from this update's calibration finding**: investigate why
   `predict_intervals()` coverage is so unreliable and why classification's
   interval width runs backwards from correctness -- before spending effort
   on anything downstream that assumes the footprint means something.
   See #149.
3. Close the IT2/GT2 membership-type gap (#144) -- still gated on finding a
   workload where GT2 earns its keep; this update didn't find one.
4. Resolve the secondary-membership-family question (#145) -- still gated,
   and now doubly so: if the footprint's own calibration doesn't hold up,
   changing its secondary shape isn't the next lever to pull.
5. GPU acceleration (#146) -- no evidence yet of a construction-time
   bottleneck at any scale tested here.
6. Hierarchical GT2 via `tribble-tree` (#147) -- largest, most open item;
   unchanged by this update.

## Reproducing

```bash
python -m benchmarks.t1_it2_gt2_tradeoff
python -m benchmarks.t1_it2_gt2_tradeoff -o benchmarks/results/mine.json
```
