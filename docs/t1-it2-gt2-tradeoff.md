# T1 vs IT2 vs GT2: is the extra machinery worth its construction cost?

> **Status: one directional read, not a powered study.** Single seed, single
> train/test split, single dataset per task. Treat relative orderings as
> signal and absolute numbers as approximate. This answers the question
> `docs/gt2-evaluation.md` explicitly left open ("whether GT2's added
> expressiveness actually improves accuracy... is an empirical question [the
> GT2 implementation] does not answer") -- as a first pass, not the last word.

**Answer, on these two workloads: no, not for point-estimate accuracy.**
Sweeping GT2's `n_alpha_planes` from 1 to 10 moved classification accuracy by
exactly 0.0000 and regression R² by 0.004. The type-2 footprint alone (IT2 or
GT2, unrefined) tracks T1 at matched cost rather than beating it. The only
knob that moved either metric by a meaningful amount was post-fit refinement
-- and it costs 80-1500x the construction time.

## Setup

- **Classification**: sklearn `wine` (178 rows, 13 features, 3 classes),
  70/30 stratified split.
- **Regression**: sklearn `make_friedman1` (500 rows, 10 features, 5
  informative, nonlinear), 70/30 split.
- **Sweep**: `n_gaussians` in 1-5 for all three families; GT2 additionally
  sweeps `n_alpha_planes` in {1,2,3,5,8,10} at `n_gaussians=3`; one
  `refine_it2=True` point per task at `n_gaussians=2`.
- **Metric**: accuracy / R² on the held-out split vs. wall-clock `.fit()`
  time.

## Results

Full sweep (42 fits) in `results/t1-it2-gt2-tradeoff.json`. Best point per
family per task (✓ = Pareto-optimal, i.e. not dominated by any faster *and*
more accurate point across all four families):

| task | family | best config | fit time | performance | pareto |
|---|---|---|---|---|---|
| classification | T1 | n_gaussians=2 | 44 ms | 0.926 acc | |
| classification | IT2 | n_gaussians=2 | 41 ms | 0.926 acc | ✓ |
| classification | IT2+refine | n_gaussians=2 | 3.38 s | 0.981 acc | ✓ |
| classification | GT2 | n_gaussians=2, K=5 | 43 ms | 0.926 acc | |
| regression | T1 | n_gaussians=1 | 15 ms | 0.787 R² | ✓ |
| regression | IT2 | n_gaussians=5 | 40 ms | 0.712 R² | |
| regression | IT2+refine | n_gaussians=2 | 35.3 s | 0.880 R² | ✓ |
| regression | GT2 | n_gaussians=2, K=5 | 22 ms | 0.768 R² | |

T1 lands on the classification frontier's plateau but not the frontier
itself: IT2 matches its accuracy 3 ms faster. On regression T1's
`n_gaussians=1` point is Pareto-optimal outright -- the fastest fit in the
entire sweep, and nothing untrained beats its R² for less time.

Three findings:

**1. Alpha-plane count is nearly free and nearly inert.** Holding
`n_gaussians=3` fixed and sweeping `n_alpha_planes` 1→10: classification
accuracy stayed at exactly 0.889 across every K; regression R² drifted
0.723→0.727, a 0.004 spread. Construction time stayed flat too. The
alpha-plane machinery is computing something real (this isn't a no-op --
see `GT2_GUIDE.md`), but on these two workloads it never changes which class
wins or how far off the point estimate lands.

**2. The type-2 footprint alone doesn't pay for itself.** Without
refinement, IT2 and GT2 sit on roughly T1's own accuracy/R² band at matched
`n_gaussians`. One case is worse than "roughly the same": IT2 regression at
`n_gaussians=1` scores R² = -0.27 (worse than predicting the mean) where the
T1 model it was converted from scores 0.79. A wide footprint of uncertainty
on a model with too little structure can actively hurt, not just widen the
interval.

**3. Refinement is the whole story, and it is expensive.**
`refine_it2` moved classification accuracy +0.056 at ~82x the construction
time, and regression R² +0.24 at ~1460x the construction time. If type-2
modeling is worth deploying on a given workload, this data says it's because
of refinement, not the interval or alpha-plane structure by itself.

## What this does not settle

- One dataset per task, one seed, no interval-calibration metric (coverage
  of `predict_intervals()` at its nominal level) -- accuracy/R² is the wrong
  lens for a method whose premise is better-calibrated uncertainty, not
  better point estimates. See "Next steps" below.
- `refine_it2`'s cost multiplier is sensitive to `n_gaussians` and dataset
  size; the 82x/1460x figures are single data points, not a fitted curve.

## Next steps toward a fuller T2 implementation

In priority order -- each gates the next, the way GT2 itself was built in
phases (data model → forward inference → type reduction → refinement):

1. **Extend this benchmark** across several real/UCI datasets and seeds, and
   add an interval-calibration metric alongside accuracy/R² before trusting
   the "not worth it" read broadly.
2. **Close the membership-type gap between IT2 and GT2.** IT2 supports
   Gaussian, trapezoidal, and triangular antecedents; GT2 is Gaussian-only
   (KISS, matching IT2's own v1 scope). Worth doing once step 1 finds a
   workload where GT2 earns its keep.
3. **Resolve the secondary-membership-family question** (triangular vs.
   Gaussian-on-Gaussian, flagged undecided in `docs/gt2-evaluation.md`) --
   only if step 1's calibration metric shows a real workload needs a
   non-uniform confidence profile within the footprint.
4. **GPU acceleration.** Already on the IT2/T1 roadmap (`IT2_GUIDE.md`).
   GT2 inherits it for free once IT2 has it: every GT2 call is an
   unmodified IT2 kernel call run K times, which is exactly the kind of
   embarrassingly parallel work a batched GPU path suits.
5. **Hierarchical GT2** via `tribble-tree`, mirroring `IT2_GUIDE.md`'s own
   "Hierarchical IT2" future-extension entry.
6. **Documentation cleanup.** `IT2_GUIDE.md`'s "Future Extensions" list still
   says norm families are probability-only and that the scipy.optimize swap
   is outstanding -- both are done (IT2/GT2 already reuse the shared
   5-family Type-1 kernel path; #119/#134/#136 removed every direct
   `scipy.optimize` call from this project).

## Reproducing

```bash
python -m benchmarks.t1_it2_gt2_tradeoff
python -m benchmarks.t1_it2_gt2_tradeoff -o benchmarks/results/mine.json
```
