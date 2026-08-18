# Which antecedent membership function should you use?

**Answer: Gaussian by default. The fast-histogram trapezoid is a competitive,
much cheaper alternative *when its membership functions cover the input space*,
but every bounded-support family (trapezoid, triangular) can collapse in higher
dimensions and should be reached for deliberately. EM-fit trapezoids/triangles
are the weakest and slowest, for a structural reason (below).**

`TribbleClassifier` and `TribbleRegressor` both accept `member_function`
(`"gaussian" | "trap" | "triangular"`) and, for the trapezoid, `trapz_method`
(`"fast"` histogram | `"em"`). This note measures what that choice costs and
buys for the **antecedents** of a Type-1 regressor, and explains the one failure
mode you must know about.

## Benchmark

`benchmarks/membership_function_antecedent.py` (Type-1 `TribbleRegressor`,
`tsk_order="1st"`, `top_p=0.99`, 70/30 split, standardized features):

| dataset | membership | test RMSE | test R² | firing coverage | fit s |
|---|---|--:|--:|--:|--:|
| friedman1 (8 feats) | gaussian | 2.12 | **0.83** | 100% | 0.06 |
| | trap-fast | 2.82 | 0.70 | 99% | **0.01** |
| | trap-em | 3.87 | 0.43 | 95% | 17.8 |
| | triangular (em) | 3.24 | 0.60 | 98% | 11.7 |
| diabetes (10 feats) | gaussian | **53.5** | **0.44** | 100% | 0.05 |
| | trap-fast | 168.1 | −4.54 | **0%** | 0.01 |
| | trap-em | 168.1 | −4.54 | 0% | 24.3 |
| | triangular (em) | 168.1 | −4.54 | 0% | 12.7 |

(A separate run on N-CMAPSS DS02 RUL, memory features, quadratic consequents,
had `trap-fast` at **6.42** per-sample RMSE vs Gaussian **6.48** — i.e. fast
trapezoid can *win* when its MFs tile well. See issue #165.)

## Two things to take away

### 1. Bounded-support MFs can zero out the firing (the diabetes row)

A firing strength is a t-norm **product** over the antecedent features. Gaussian
membership is never exactly zero, so a rule always fires a little and coverage is
100% regardless of dimensionality. Trapezoid/triangular membership is **exactly
zero outside `[a, d]`**, so a rule fires only where *every* feature lands inside
its support. As the feature count grows, the probability that some feature falls
outside its trapezoid approaches 1, firing collapses to ~0, and every row takes
the "no rule fired" fallback (a constant) — which is exactly the diabetes result:
0% coverage, identical degenerate R² across all three bounded families.

Practical guidance: with trapezoid/triangular antecedents, keep the antecedent
set small (tune `top_p`/`top_n`), verify firing coverage, and prefer wide MFs.
When in doubt, Gaussian is the safe default.

### 2. EM antecedents are structurally mis-aimed (issue #163)

The EM M-step maximizes the likelihood of the **area-normalized** trapezoid PDF
(`trapz_pdf` divides by area). For a bounded-support density that rewards
*shrinking* the support and collapsing the plateau to concentrate mass on the
data mode — the classic MLE pathology. The result is narrow, mode-hugging MFs: a
good density estimate but a poor antecedent partition, worse on train and test
alike, and 10–25× slower than the histogram method here. The `width_reg`
parameter added in #167 counters this (rewarding wider support) and recovers a
large part of the gap, but does not fully close it, because the likelihood
objective still *centers* components on modes rather than tiling the range.

## Recommendation

- **Default: `member_function="gaussian"`** — robust across dimensionality.
- **Lean/interpretable alternative: `member_function="trap", trapz_method="fast"`**
  — competitive and the cheapest to fit, *provided* firing coverage stays high
  (few antecedents, wide MFs).
- **`trapz_method="em"` / `"triangular"`**: density-estimation-oriented; use only
  with `width_reg > 0` and expect it to trail `fast`. Not recommended as a
  default antecedent.

Reproduce: `python -m benchmarks.membership_function_antecedent -o benchmarks/results/membership-function-antecedent.json`
