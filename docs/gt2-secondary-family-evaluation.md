# Should GT2 offer a Gaussian-on-Gaussian secondary membership family?

> **Status: decided -- no.** Issue #145. Backed by
> `benchmarks/gt2_secondary_family_probe.py`; no production code changes.

**Answer: do not add a `secondary_family` parameter.** Under this library's
alpha-plane GT2, the secondary family only chooses a *width schedule* -- how
far each alpha-plane's sigma interval sits from the principal sigma. Swapping
triangular for a truncated Gaussian-on-Gaussian moves GT2's regression
interval somewhere between today's GT2 and IT2, changes nothing a caller can
see once `conformal_calibration=True` is on, and *cannot* touch the
classification interval at all, because `GT2TribbleClassifier.predict_intervals`
returns the alpha=0 plane by construction. The issue's own gate (a workload
where the uniform/triangular profile is the binding constraint) is not met,
and the probe shows it cannot be met by this knob.

## What a secondary family actually controls here

Every family worth considering (convex in its own membership value) has
nested alpha-cuts, so each one reduces to a schedule
`f(alpha) in [0, 1]`: the fraction of each side's FOU span the plane keeps.

| family | `f(alpha)` | plane at alpha = 0.2 / 0.6 |
|---|---|---|
| triangular (shipped) | `1 - alpha` | 0.80 / 0.40 |
| Gaussian-on-Gaussian, truncated at 2 sd | `min(1, sqrt(-2 ln alpha) / 2)` | 0.90 / 0.51 |
| Gaussian-on-Gaussian, truncated at 3 sd | `min(1, sqrt(-2 ln alpha) / 3)` | 0.60 / 0.34 |

The truncation point `c` is not a property of "Gaussian"; it is a free shape
parameter, and it moves the schedule both above and below triangular. That is
the first finding: **"Gaussian-on-Gaussian vs triangular" is not a binary
choice of model, it is a one-parameter reshaping of the same width
schedule.** `GT2GaussianMembership.principal_mf.sigma` already gives one
such knob (where the apex sits), and refinement already searches it.

## Measured (3 seeds, `n_gaussians=3`, `n_alpha_planes=5`)

Regression:

| dataset | conformal | family | R² | coverage | rel. width | ρ(width, \|resid\|) |
|---|---|---|--:|--:|--:|--:|
| friedman1 | off | IT2 | 0.710 | 0.436 | 0.137 | +0.062 |
| friedman1 | off | triangular | 0.739 | 0.213 | 0.058 | +0.007 |
| friedman1 | off | gauss c=2 | 0.737 | 0.258 | 0.070 | +0.020 |
| friedman1 | off | gauss c=3 | 0.740 | 0.198 | 0.054 | +0.005 |
| friedman1 | on | IT2 | 0.718 | 0.918 | 0.439 | +0.038 |
| friedman1 | on | triangular | 0.739 | 0.909 | 0.405 | +0.030 |
| friedman1 | on | gauss c=2 | 0.737 | 0.913 | 0.413 | +0.030 |
| friedman1 | on | gauss c=3 | 0.739 | 0.913 | 0.405 | +0.036 |
| diabetes | off | IT2 | 0.477 | 0.221 | 0.089 | -0.108 |
| diabetes | off | triangular | 0.492 | 0.060 | 0.026 | +0.012 |
| diabetes | off | gauss c=2 | 0.492 | 0.073 | 0.033 | -0.007 |
| diabetes | off | gauss c=3 | 0.492 | 0.048 | 0.023 | +0.061 |
| diabetes | on | IT2 | 0.476 | 0.895 | 0.651 | -0.067 |
| diabetes | on | triangular | 0.494 | 0.880 | 0.608 | +0.029 |
| diabetes | on | gauss c=2 | 0.494 | 0.880 | 0.612 | +0.014 |
| diabetes | on | gauss c=3 | 0.493 | 0.880 | 0.605 | +0.071 |

Classification (predicted-class firing-interval width):

| dataset | family | accuracy | width correct | width incorrect |
|---|---|--:|--:|--:|
| wine | IT2 / triangular / c=2 / c=3 | 0.951 (all four) | 0.259 | 0.192 |
| breast_cancer | IT2 | 0.965 | 0.351 | 0.314 |
| breast_cancer | triangular | 0.959 | 0.351 | 0.313 |
| breast_cancer | gauss c=2 | 0.961 | 0.351 | 0.319 |
| breast_cancer | gauss c=3 | 0.957 | 0.351 | 0.299 |

## Findings

1. **Accuracy / R²: no effect.** Every family is within 0.003 R² and 0.004
   accuracy of every other -- noise.
2. **Raw regression width is monotone in the schedule, nothing more.** Wider
   schedule (c=2) means wider interval and more coverage, sitting between
   triangular and IT2. Same thing `uncertainty_width` already does.
3. **With conformal calibration on, the family stops mattering.** All
   families hit target coverage (0.88-0.92) at widths within 2% of each
   other. The split-conformal margin is what makes the interval honest, and
   it swamps the base width.
4. **No family makes the interval adaptive.** ρ(width, |residual|) stays
   in [-0.11, +0.07] for every family, IT2 included. A shaped secondary
   grade was supposed to put width where the error is; it doesn't, because
   the interval's spread comes from rule disagreement, not from the
   antecedent footprint's shape (see `IT2_GUIDE.md` "Interval Calibration").
5. **Classification intervals are identical by construction.**
   `GT2TribbleClassifier.predict_intervals` uses the alpha=0 plane, which
   every family maps to the full FOU. Only `predict`'s argmax sees the other
   planes, hence the tiny breast_cancer accuracy wobble.

## Correction: the classification sign problem is not a secondary-family problem

`IT2_GUIDE.md` said fixing #149's backwards classification width "would
require a different secondary-membership representation (see #145)". It
would not. The width is zero at `x = mu` because upper and lower share `mu`
-- a property of the **primary** footprint (uncertain sigma, certain mean).
Every alpha-plane of every secondary family inherits that shared `mu`, so
every plane's width is also zero at `mu`. The levers that could change the
sign are:

| lever | what changes | cost |
|---|---|---|
| uncertain-mean FOU (`mu ± delta`, flat-topped upper MF) | width > 0 at the centre | new IT2 membership type + `widen_membership` variant |
| define the classification interval as the *margin* interval (predicted class lower bound vs. runner-up upper bound) | measures decision ambiguity, not boundary ambiguity | output-side only; no model change |
| conformal prediction *sets* for classification | calibrated set coverage, model-agnostic | mirrors the regressor's `conformal_calibration` |

The guide has been corrected to point here.

## Recommendation

- **Close #145 as not planned.** Keep triangular as the only shipped family.
- If a caller wants calibrated or adaptive regression intervals, the
  supported path is `conformal_calibration=True` (shipped in #149), not a
  secondary-family knob.
- If classification uncertainty ever needs a correctly-signed signal, open a
  new issue for one of the three levers above; the cheapest is conformal
  prediction sets, which needs no new membership math.
- Revisit only if a workload shows ρ(width, |resid|) materially > 0 for IT2
  and GT2 losing it -- the only evidence that the secondary *shape* matters.

## Reproducing

```bash
python -m benchmarks.gt2_secondary_family_probe
python -m benchmarks.gt2_secondary_family_probe -o benchmarks/results/gt2_secondary_family.json
```
