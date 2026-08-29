# Does a correctly-built smooth trapezoid EM close the gap with `fast`?

**Answer: no.** The differentiable relaxation shipped in #195
(`trapz_math_smooth.py`) had a real shape/normalization bug -- fixing it
still leaves smooth EM comparable to plain (crisp) EM on raw fit quality,
10-70x slower, and it does not touch either structural reason trapezoid/
triangular antecedents lose to Gaussian and the fast histogram method. This
reconfirms `ISSUE_163_RESOLUTION_PLAN.md`'s Phase 1 conclusion, this time
against a correctly-built implementation rather than a buggy one.

## Why this was revisited

`docs/antecedent-membership-function-evaluation.md` and
`ISSUE_163_RESOLUTION_PLAN.md` already recommend against EM-fit trapezoids
(mode-hugging, 10-25x slower than `fast`, and Phase 1 of that plan reports a
smooth-sigmoid prototype that was "10-30% worse quality, not better"). A
request to simplify the antecedent-fitting surface down to EM asked to
double-check that prototype before accepting its conclusion -- particularly
since it was never merged into the main antecedent path, only into
`tribble-tree`'s gate refinement.

## The bug

The #195 shape function built its rising/falling "ramps" as the product of
two full-height sigmoids:

```python
left_ramp = sigmoid(k*(x-a)) * (1 - sigmoid(k*(x-b)))
```

This is a smoothed **indicator of the whole `[a, b]` interval** (value ≈ 1
throughout, ≈ 0.5 at both endpoints), not a ramp rising from 0 to 1. Combined
with the `plateau` term (a hard boolean mask, not smooth at all, despite the
module's "infinitely differentiable" claim), the true unnormalized shape is
close to a boxcar spanning the whole `[a, d]` support, regardless of where
`b`/`c` sit inside it. Direct quadrature confirms it: the module's declared
normalization (`(b-a)/2 + (c-b) + (d-c)/2 + 2*log(2)/steepness`) is off by
6-50% depending on the knots -- not the few-percent sigmoid-tail correction
the formula implies.

This means the #195 prototype's "10-30% worse" finding was measured against
a mixture of box densities, not smooth trapezoids -- it didn't actually test
the idea it set out to test.

## The fix

`trapz_math_smooth.py` was rewritten:

- **Shape**: softplus-based smoothed ramps (`_rising_ramp`/`_falling_ramp`) --
  the actual smoothed `clip((x-a)/(b-a), 0, 1)`, linear in the middle with
  only the two corners rounded (width ~`1/steepness`), combined via `min`
  exactly as a crisp trapezoid combines its two ramps.
- **Normalization**: computed by quadrature (`_smooth_trapz_area`) over the
  shape's actual support, so it is exact regardless of steepness or how
  narrow a component is (no closed-form guess to get wrong).
- **Ordering**: the M-step now reuses `trapz_math._solve_ordered_params`'s
  gap-reparametrization, so `a ≤ b ≤ c ≤ d` is enforced. The original M-step
  optimized each parameter over an independent box, so nothing stopped e.g.
  `b > c`; `smooth_trapz_pdf`'s guard would then just zero out an invalid
  configuration, silently discarding that step's work.
- **Annealing**: steepness anneals geometrically from a soft, wide-basin
  start to a near-crisp end over the EM iterations, rather than a single
  fixed value -- intended to give the optimizer room to move knots past
  whatever the piecewise-linear objective's kinks were stalling on.
- **`width_reg`**: carried over from `trapz_math.py`, since annealing the
  optimization surface does not address the actual reason bounded-support
  MLE collapses support onto the mode (see below).

`tests/test_trapz_math_smooth.py` (new -- this module had no test coverage
before) pins the corrected shape, the ordering constraint, and the
crisp-comparable log-likelihood the fit reports.

## Re-benchmark (corrected implementation)

Raw fit, five synthetic distributions, `n=1000`, `n_bins=50`:

| distribution | plain EM (s) | smooth EM (s) | plain EM LL | smooth EM LL |
|---|--:|--:|--:|--:|
| unimodal | 0.82 | 1.22 | -1485.7 | -1493.8 |
| bimodal | 0.09 | 5.13 | -1489.5 | -1506.3 |
| trimodal | 0.22 | 8.64 | -1645.2 | -1669.2 |
| exponential | 0.03 | 1.36 | -2100.3 | -2099.9 |
| heavy_tail | 0.04 | 2.17 | -2195.0 | -2006.4 |

No consistent direction: smooth EM wins outright on `heavy_tail`, is a wash
on `exponential`, and loses on the rest -- while being 10-70x slower than
plain EM in every case (which is itself ~200-1000x slower than `fast`; see
`tests/benchmark_trapz_performance.py`). The slowdown is structural: correct
normalization requires a quadrature evaluation inside the optimizer's inner
loop, on every objective call.

## Why smoothing the landscape was never going to be enough

Two separate problems get conflated as "EM trapezoids don't work well as
antecedents":

1. **Optimization difficulty** -- the piecewise-linear objective has kinks a
   local optimizer can stall on. Smoothing (annealed or not) addresses this
   one, and only this one.
2. **Objective shape** -- `trapz_pdf` normalizes by area, so for a mixture
   concentrated near one mode, *shrinking the support* increases the fitted
   density's peak and therefore the likelihood -- a genuine MLE property of
   normalized bounded-support densities, not an optimizer artifact. This is
   what `width_reg` (issue #167) was added to counter, and per that issue's
   own accounting it only "recovers a large part of the gap," not all of it.

Smoothing the M-step's landscape touches (1) only. It cannot touch (2)
because (2) is a property of what the objective *is*, evaluated exactly, not
of how hard it is to optimize. Separately, the antecedent-role-specific
failure mode from `docs/antecedent-membership-function-evaluation.md` --
firing coverage collapsing to 0% in higher dimensions because a
product-t-norm rule fires only where *every* bounded-support feature lands
inside its support -- is a property of the membership family (bounded vs.
unbounded support), not of how any of these fitting methods construct it.
Neither the buggy nor the corrected smooth relaxation was ever going to move
that number.

## Decision

Unchanged from `ISSUE_163_RESOLUTION_PLAN.md`: EM refinement of trapezoid
gates in `tribble-tree` is deprecated (`FutureWarning`, `gate_style="gaussian"`
recommended for `refine_em`), and `TribbleClassifier`/`TribbleRegressor`'s
`member_function`/`trapz_method` surface is unchanged --
`trapz_method="fast"` remains the recommended trapezoid option, plain EM and
the (now-corrected) smooth EM remain available but not defaults. The
`trapz_math_smooth.py` fix ships regardless, as a correctness fix
independent of this decision.
