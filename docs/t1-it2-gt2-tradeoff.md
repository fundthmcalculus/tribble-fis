# T1 vs IT2 vs GT2: is the extra machinery worth its construction cost?

> **Status: extended (#143, #144), superseding the single-dataset first pass.**
> Six datasets, five seeds on the flagship pair, an interval-calibration
> metric, and a `member_function` sweep (gaussian/trap/triangular, #144) now
> back this doc. Still not a powered study -- see "What this does not
> settle" -- but the headline verdict is now backed by more than one
> dataset, one seed, and one antecedent shape.

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
- **member_function** (#144): T1/IT2/GT2 at `n_gaussians=2`, no refine,
  repeated with `member_function` in {gaussian, trap, triangular} on the
  flagship pair -- same performance and calibration metrics as above.
  Full sweep (170 fits) in `results/t1-it2-gt2-tradeoff.json`.

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

### 5. Antecedent shape (#144) doesn't change the accuracy story, and makes the calibration story worse

`member_function` in {gaussian, trap, triangular}, `n_gaussians=2`, no refine:

| task | member_function | T1 | IT2 | GT2 | fit time (T1/IT2/GT2) |
|---|---|---|---|---|---|
| classification (wine) | gaussian | 0.926 | 0.926 | 0.926 | 0.04 / 0.04 / 0.04s |
| classification (wine) | trap | 0.741 | 0.852 | 0.833 | 0.02 / 0.02 / 0.02s |
| classification (wine) | triangular | 0.704 | 0.833 | 0.815 | 2.28 / 2.25 / 2.21s |
| regression (friedman1) | gaussian | 0.785 | 0.639 | 0.768 | 0.015 / 0.015 / 0.020s |
| regression (friedman1) | trap | 0.350 | 0.719 | 0.674 | 0.010 / 0.014 / 0.013s |
| regression (friedman1) | triangular | 0.160 | 0.460 | 0.476 | 0.87 / 0.90 / 0.97s |

Two findings, neither of which was expected going in:

- **No shape makes GT2 clearly beat IT2 or T1** -- the "no accuracy win"
  verdict holds for all three antecedent shapes, not just Gaussian.
- **`triangular` costs 60-100x more construction time than gaussian/trap at
  matched config**, because `member_function="triangular"` always routes
  through the EM-based, BIC-searching fit (`trapz_math.create_trapz_
  membership_dict(shape="triangle")`) -- unlike `"trap"`, there is no
  `trapz_method="fast"` histogram alternative for triangular shapes. That is
  a real, avoidable cost difference between two membership types that both
  ship today, not a fundamental one.
- **A genuine wrinkle, not fully explained here**: T1's own trap/triangular
  fits are markedly worse than its Gaussian fit at this `n_gaussians`
  (0.926->0.741->0.704 classification; 0.785->0.350->0.160 regression), yet
  IT2/GT2's widened footprint recovers much of that gap relative to their
  *own* (weak) T1 base. Plausibly the EM fit lands in a worse local optimum
  at this small `n_gaussians` and the extra flexibility from widening
  partially compensates -- untested here, and not something to read as "IT2
  beats T1" since both are being compared to a base fit that itself wasn't
  well-tuned for shape.

Calibration, both metrics, same sweep:

| task | member_function | IT2 width (correct/incorrect) | GT2 width (correct/incorrect) |
|---|---|---|---|
| classification (wine) | gaussian | 0.274 / 0.241 (1.14x) | 0.274 / 0.241 (1.14x) |
| classification (wine) | trap | 0.186 / 0.113 (1.64x) | 0.190 / 0.101 (1.89x) |
| classification (wine) | triangular | 0.085 / 0.018 (4.85x) | 0.087 / 0.016 (5.51x) |

| task | member_function | IT2 coverage | GT2 coverage |
|---|---|---|---|
| regression (friedman1) | gaussian | 0.700 | 0.393 |
| regression (friedman1) | trap | 0.213 | 0.093 |
| regression (friedman1) | triangular | 0.540 | 0.347 |

This is the predicted mechanism from #149 playing out, more sharply:
Gaussian's IT2/GT2 conversion holds `mu` fixed and scales `sigma`, so
`upper == lower == 1` at exactly one point. Trapezoid's conversion holds
the flat top `[b, c]` fixed and scales the outer slopes
(`gauss_data.widen_membership`), so `upper == lower == 1` across an entire
*interval*, not a point -- a strictly larger zero-width region for the
"backwards" mechanism to produce from. The classification width ratio
(correct-width / incorrect-width, a proxy for how badly backwards the
signal runs) climbs from 1.14x (gaussian) to 1.89x (trap) to 5.51x
(triangular); regression coverage falls further from gaussian's already-poor
baseline for trap, recovering partway for triangular. **Antecedent shape is
not a fix for #149 -- for classification it makes the miscalibration worse,
not better.**

## What this does not settle

- Coverage/width were measured at a single `uncertainty_width=0.5`; whether
  a different width fixes coverage is untested. A genuinely calibrated
  wrapper (`conformal_calibration` on `IT2TribbleRegressor`/
  `GT2TribbleRegressor`) shipped in #149's fix -- it's opt-in, not the
  default this doc's numbers reflect, so the coverage table above still
  describes what a caller gets without asking for calibration.
- Classification's backwards-width finding has no equivalent fix (#149's
  investigation concluded it's structural, not a tuning problem -- see
  `IT2TribbleClassifier.predict_intervals`'s docstring).
- Still no dataset or antecedent shape where GT2 clearly beats IT2 on either
  metric or on calibration -- the "when does the extra machinery earn its
  cost" question remains open, now with more evidence pointing toward
  "rarely, on tabular data like this."
- `refine_it2`'s cost multiplier is still characterized on one dataset each;
  not repeated across the newly added datasets or member_function values
  (see Setup).

## Next steps

Tracked as GitHub issues, in priority order:

0. ~~Fix stale claims in `IT2_GUIDE.md`~~ -- done, #142.
1. ~~Extend this benchmark~~ -- done, #143 (this update).
2. ~~Investigate why `predict_intervals()` coverage is unreliable and why
   classification's interval width runs backwards from correctness~~ --
   done, #149 (regression: split-conformal calibration, opt-in; classification:
   structural, documented as non-fixable by recalibration).
3. ~~Close the IT2/GT2 membership-type gap~~ -- done, #144 (this update):
   trapezoid/triangular now work end-to-end for both classification and
   regression, IT2 and GT2, including refinement. No accuracy win from any
   shape, and calibration gets *worse* for non-Gaussian shapes (see above).
4. Resolve the secondary-membership-family question (#145) -- still gated,
   and now doubly so: if the footprint's own calibration doesn't hold up
   for any antecedent shape tested, changing its secondary shape isn't the
   next lever to pull.
5. GPU acceleration (#146) -- no evidence yet of a construction-time
   bottleneck at any scale tested here, though `triangular`'s EM-only fit
   path is now a known, unrelated cost outlier worth a look on its own.
6. Hierarchical GT2 via `tribble-tree` (#147) -- largest, most open item;
   unchanged by this update.

## Reproducing

```bash
python -m benchmarks.t1_it2_gt2_tradeoff
python -m benchmarks.t1_it2_gt2_tradeoff -o benchmarks/results/mine.json
```
