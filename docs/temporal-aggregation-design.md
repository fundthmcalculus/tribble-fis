# Learned fuzzy aggregation over time (`TribbleTimeSeries`, issue #127)

**Status: experimental, opt-in.** New module `tribblefis.temporal_aggregation`;
no existing estimator changes behaviour. Measured results are at the bottom.

## 1. Problem

`TribbleRegressor` / `MimoGaussianPredictor` are flat tabular estimators. Every
sequence workload so far (the N-CMAPSS RUL DOE in #127) got the time dimension
removed by hand first:

| #127 arm | Window | Statistic | Who chose it |
|---|---|---|---|
| whole-cycle | entire cycle | mean/std/min/max/last | analyst |
| terciles | crisp thirds | mean/std/min/max/last per third | analyst |
| memory window | fixed rolling widths | mean | analyst |

Same move each time: pick *where*, pick *what*, flatten. Both choices should be
learned.

## 2. Operator choice

Aggregation theory is the time-axis counterpart of the t-norm/t-conorm machinery
this package already uses across features.

| Candidate | Params over L samples | Aggregation axioms | Verdict |
|---|---|---|---|
| General Choquet integral | 2^L | yes | infeasible at L~100 |
| 2-additive Choquet | O(L²) | yes | still length-dependent, no variable L |
| Sugeno integral | 2^L, min/max only | yes | non-smooth; no gradient |
| OWA | L (order weights) | yes | ignores *when* a sample occurred |
| Weighted mean | L | yes | ignores *order* (no max/min) |
| **WOWA, Torra (1997)** | **2K + K·F, independent of L** | **yes** | **chosen** |

WOWA combines both of OWA's and the weighted mean's degrees of freedom: an
*importance* weight per sample (where it is in time) and an *order* profile
(which ranks count). It is exactly a Choquet integral with respect to the
distorted-probability fuzzy measure `μ(A) = Q(Σ_{i∈A} p_i)`, so it keeps every
aggregation-function property (bounded by min/max, idempotent, monotone) while
having a parameter count that does not grow with sequence length.

### 2.1 Where: fuzzy temporal partition

Normalised time `τ_i = (i + 0.5)/L` (midpoint rule: crisp K-ths split evenly
when K divides L; variable-length cycles share one axis).

`μ_k(τ) = g_k(τ) / Σ_j g_j(τ)`, `g_k` Gaussian with centre `c_k`, width `h_k`
→ a Ruspini partition (sums to 1 everywhere). Computed in log space, so very
narrow widths degrade to a crisp nearest-centre partition rather than 0/0.

Inside segment k, sample importances are `p_i ∝ μ_k(τ_i)`.

### 2.2 What: exponential RIM quantifier

`Q_λ(x) = (1 − e^{−λx}) / (1 − e^{−λ})`, one λ per (segment, feature).

| λ | Behaviour | Orness `∫Q` |
|---|---|---|
| 0 | importance-weighted mean | 0.5 |
| → +∞ | max over segment | → 1 |
| → −∞ | min over segment | → 0 |

Orness = `1/(1 − e^{−λ}) − 1/λ`; reported per segment/feature as `orness_`.

**Why not the textbook power quantifier `Q(x)=x^α`?** Found during testing:
`Q'(0)=∞` for `α<1`, so a sample with importance 1e-14 (any sample *outside* a
soft segment) still gets weight `(1e-14)^α ≈ 0.5`, and every max-like segment
returns the *global* max. With a fuzzy partition every sample has non-zero
importance everywhere, so this is the common case. The exponential family has
`Q'(0) ≈ λ`, bounding an outside sample's weight by ≈ `λ·p`.
`test_max_like_segment_does_not_leak_the_global_max` pins both halves.

### 2.3 Nesting property (design anchor)

Centres `(k+0.5)/K`, widths → 0, λ = 0 reproduce the crisp "mean of each K-th"
exactly; λ → ±∞ at the same centres gives per-K-th max/min. The hand-engineered
baseline is *inside* the family, and learning starts from (a soft version of)
it, so the model begins where an analyst would and moves only if the data pays.

## 3. Learning

| Item | Choice | Why |
|---|---|---|
| Objective | K-fold CV MSE of a standardised ridge head on the aggregated features | closed-form, cheap; decoupled from the FIS so the representation is inspectable and reusable |
| Leakage | standardisation stats from each fold's train rows only | — |
| Gradient | exact, hand-derived: ridge solve → standardisation → WOWA → softmax partition | finite differences took 102 s/fit; analytic 12 s; checked to 1e-4 rel. |
| Solver | new `optimizer_utils.projected_lbfgs_solve` | repo rule: no direct `scipy.optimize` (`test_no_module_imports_scipy_optimize_directly`); existing `projected_gradient_solve` stalled ~17% above |
| Parameter scaling | all boxes width ~1 (centres on [0,1], log-width/5.3, λ/400) | one step size suits all parameters |
| Starts | even partition first, then `n_restarts` random centre sets; best CV loss wins | centre placement is multimodal (§6) |
| Guarantee | `objective_ ≤ objective_init_` | solver never returns worse than its start |

## 4. API

```python
from tribblefis.temporal_aggregation import FuzzyTemporalAggregator, TribbleTimeSeriesRegressor

agg = FuzzyTemporalAggregator(n_segments=3).fit(cycles, y)   # cycles: list of (T_i, F)
Z = agg.transform(cycles)          # DataFrame, columns "{sensor}__seg{k}"
agg.segment_report()               # centre, width, lambda, orness per segment x sensor

model = TribbleTimeSeriesRegressor(n_segments=3).fit(cycles, y)   # MimoGaussianPredictor head
model.predict(test_cycles)         # ndarray for 1-D y, DataFrame for multi-output
```

| Parameter | Default | Meaning |
|---|---|---|
| `n_segments` | 3 | K |
| `fit_segments` / `fit_quantifier` | True / True | ablation switches |
| `init_width` | 0.25 | start width as fraction of `1/K` |
| `ridge`, `cv` | 1.0, 5 | surrogate head |
| `max_evals`, `n_restarts` | 300, 2 | budget per start, extra starts |
| `base_estimator` | `MimoGaussianPredictor` | any sklearn regressor, cloned |

Input: 3-D array, 2-D array (one feature), or list of 2-D arrays / DataFrames
(lengths may differ). NaN/inf, empty sequences, mismatched widths raise.

## 5. Test protocol

`tests/test_temporal_aggregation.py` (≈8 s) and additions to
`tests/test_optimizer_utils.py`. Accuracy claims are *not* asserted in CI —
they live in the benchmark (§7), per the repo's benchmark convention.

| Group | Property | Tests |
|---|---|---|
| A. Axioms | min ≤ WOWA ≤ max; idempotent; monotone in every argument; symmetric under uniform importance; importance scale-invariant; zero mass rejected | 6 (×λ grid) |
| B. Special cases | Q is a RIM quantifier for λ∈[−200,200]; dQ/dx, dQ/dλ vs finite differences; λ=0 → weighted mean; λ→±∞ → max/min within the exact bound `span·e^{−|λ|·p_extreme}` (200 draws); uniform importance → Yager OWA weights; orness formula; **leak regression vs power quantifier**; crisp limit = hand tercile means; crisp limit + λ=±200 = per-third max/min | 9 |
| C. Partition | Ruspini (sums to 1, non-negative); crisp limit = nearest centre; underflowing segment stays finite | 3 |
| D. Inputs | padding cannot leak across lengths; array/list/DataFrame agree, names propagate; length-1 sequence; validation (6 cases); transform feature mismatch; misaligned/non-finite y | 6 |
| E. Learning | full-objective analytic gradient vs finite differences; surrogate = sklearn `Ridge` on same folds; objective never worsens; learned segment lands within 0.08 of a planted phase with loss halved; spike target → orness > 0.8; ablation switches freeze params; unsupervised fit = soft even partition; segments sorted, report consistent | 8 |
| F. Contract | deterministic under seed; end-to-end single- and multi-output with default MIMO head; custom base estimator cloned not mutated; `get_params`/`clone` round-trip | 5 |
| Solver | quadratic minimum; Rosenbrock (beats steepest descent at equal budget); active box constraints; never worse than start; exact budget | 4 |

Benchmark protocol (`benchmarks/temporal_aggregation.py`, not in CI):

1. Synthetic cycles, length 60–140, 4 sensors, per-cycle offsets and drift.
   Target = height of a bump on `s0` at phase τ\* + ½ × height of a 2–3 sample
   spike on `s1` at a random time. `s2`, `s3` nuisance.
2. Arms: `whole-cycle` (#127 arm 1), `crisp-thirds` (#127 arm 2),
   `soft-thirds` (aggregator start point, nothing learned — isolates
   *learning* from *fuzziness*), `learned`.
3. Two heads per arm: `TribbleRegressor(n_gaussians=3)` and standardised
   `RidgeCV` (the latter reads the representation alone; it is also the
   learned arm's surrogate, i.e. home turf — hence both).
4. 75/25 split, paired seeds; report mean ± SEM, paired delta vs
   `crisp-thirds`, win count.
5. **Mechanism check (`--dial`)**: sweep τ\* from 0.5 (inside the middle third)
   to 2/3 (on the boundary). If learning helps *because* it relocates segments,
   the gap must be small at 0.5 and open toward 2/3.

## 6. Limitations / non-goals

- **Multimodal segment placement.** With one start the optimiser can settle on
  a worse basin; on the test problem, three starts found CV losses 0.095–0.110
  where a fourth basin reached 0.078 (scipy L-BFGS-B found the same spread from
  the same starts, so this is the landscape, not the solver). `n_restarts`
  trades fit time for this.
- **Surrogate ≠ downstream model.** Segments are chosen for a linear head, then
  frozen for the FIS. End-to-end learning through `TribbleRegressor` would
  need its gradient and is out of scope.
- **Fit cost.** Seconds to tens of seconds for hundreds of cycles; O(N·K·F·L)
  memory per evaluation. Not tuned for full-rate N-CMAPSS (subsample first).
- **Shared segments across sensors.** Centres/widths are shared; only the
  quantifier is per sensor. Per-sensor segments multiply parameters by F.
- **No real-data result.** N-CMAPSS lives in the downstream repo; this PR
  provides the reproducible synthetic evidence only. The #127 owner comment
  also suggests judging against a uniformity-style diagnostic; not done here.
- **Scope.** The #127 discussion suggests this might belong outside
  `tribble-fis`. It is kept self-contained (one module, one solver helper, no
  changes to existing estimators) so it can be lifted out cleanly.

## 7. Measured results

`uv run python -m benchmarks.temporal_aggregation --seeds 8` (raw numbers:
`benchmarks/results/temporal-aggregation.json`). 240 cycles, phase τ\*=0.68,
8 paired seeds, mean ± SEM.

| arm | feats | ridge R² | Δ vs crisp-thirds | wins | FIS R² |
|---|---|---|---|---|---|
| whole-cycle | 20 | 0.825 ± 0.015 | −0.038 ± 0.010 | 0/8 | 0.112 ± 0.122 |
| crisp-thirds | 60 | 0.863 ± 0.011 | — | — | −5.465 ± 0.291 |
| soft-thirds | 12 | 0.218 ± 0.073 | −0.645 ± 0.068 | 0/8 | −0.018 ± 0.088 |
| **learned** | 12 | **0.905 ± 0.010** | **+0.042 ± 0.009** | **7/8** | **0.687 ± 0.069** |

Reading it:

- **Learning, not fuzziness, does the work.** The un-learned soft partition is
  the worst arm; the learned one the best. Fuzziness alone blurs the signal.
- **It finds the mechanism.** Every seed placed a segment centre within ~0.01
  of the planted phase 0.68, and the spike sensor `s1` got orness ≈ 1.0
  (max-like) in at least one segment — both found from data, neither given.
- **Five times fewer features** than crisp thirds for a better fit.
- **FIS head**: crisp thirds' −5.5 is `TribbleRegressor` failing on 60
  features / 180 rows, not a statement about the representation; that is why
  the ridge head is reported. With 12 learned features the FIS is usable
  (0.69), but still well below ridge on this linear-in-features target.
- **Cost**: ~72 s per learned feature-map fit (180 cycles, 3 starts × 300
  evaluations) vs ~0.1 s for hand features.

### 7.1 Mechanism check: the dial came back flat

`--dial --seeds 5` (raw: `benchmarks/results/temporal-aggregation-dial.json`):

| phase τ\* | crisp-thirds (ridge) | learned (ridge) | Δ |
|---|---|---|---|
| 0.500 | 0.851 | 0.900 | +0.049 ± 0.014 |
| 0.580 | 0.842 | 0.890 | +0.048 ± 0.012 |
| 0.620 | 0.846 | 0.884 | +0.038 ± 0.010 |
| 0.667 | 0.856 | 0.902 | +0.046 ± 0.007 |
| 0.720 | 0.863 | 0.910 | +0.048 ± 0.011 |

(FIS head: crisp −5.6 throughout, learned 0.63–0.71.)

**The prediction in §5 failed.** The gap does not open toward the tercile
boundary; it is flat at ≈ +0.045. By the criterion stated before running it,
segment *relocation* is not what separates the learned arm from the
five-statistic hand arm.

### 7.2 Why: redundant statistics cover the boundary

`--stat-ablation --seeds 8` (raw: `benchmarks/results/temporal-aggregation-stat-ablation.json`),
crisp thirds, ridge head:

| statistics per third | feats | phase 0.5 | phase 0.68 |
|---|---|---|---|
| mean | 12 | 0.687 ± 0.027 | **0.198 ± 0.073** |
| max | 12 | 0.632 ± 0.031 | 0.564 ± 0.018 |
| mean + max | 24 | 0.847 ± 0.015 | 0.770 ± 0.019 |
| all 5 | 60 | 0.853 ± 0.017 | 0.863 ± 0.011 |

The boundary effect is real and large for #127's literal "mean of a third"
(0.69 → 0.20 when the signal straddles a boundary). The five-statistic arm hides
it: std/min/max/last per third recover what the mean loses, at 5× the features.

### 7.3 What the evidence supports

| Claim | Status |
|---|---|
| A crisp boundary through the signal breaks the mean-per-third summary | **supported** (0.69 → 0.20) |
| Learning relocates segments onto the signal | **supported** (centre within ~0.01 of τ\*=0.68 on 8/8 seeds) |
| Learning finds max-like aggregation for spike signals | **supported** (orness ≈ 1.0 on `s1`) |
| Learned beats the *best* hand arm *because of* relocation | **not supported**: lead is flat across phase |
| Learned matches/exceeds the best hand arm with 5× fewer features | **supported** (+0.04 ± 0.01, 7/8 seeds) |

Net: on this generator the method's demonstrated value is *compactness with a
readable result* — 12 features whose centres and orness say where and what —
against a brute-force hand set that needs 60 features to be robust. Whether
that translates to the FIS head on real data (where 60 hand features broke
`TribbleRegressor` here) needs the N-CMAPSS run downstream.

## References

- V. Torra, "The weighted OWA operator," *Int. J. Intelligent Systems* 12 (1997).
- R. R. Yager, "Quantifier guided aggregation using OWA operators," *IJIS* 11 (1996).
- M. Grabisch, J.-L. Marichal, R. Mesiar, E. Pap, *Aggregation Functions*, CUP (2009).
