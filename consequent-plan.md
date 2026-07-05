# Implementation Plan: TSK Fit-Quality Improvement (Consequent-First)

> Handoff document. Target: improve **R² and MSE** of the Gaussian-mixture TSK
> regressor, primarily measured on `gaussian_mixture/concrete.py`. Two systems,
> sequenced so the consequent work (Phase 1) is standalone and shippable, and the
> antecedent-refinement work (Phase 2) builds on it. **Save the final approved
> version as `consequent-plan.md` in the repo root.**

---

## Context

The Gaussian-mixture TSK pipeline has two independent parameter sets, but only
one is ever optimized against the regression objective:

| Stage | Set by | Optimized against MSE? |
|---|---|---|
| **Antecedents** — `(μ, σ)` of every membership function → firing strengths | Heuristic: KMeans clusters + `stats.norm.fit` per output bucket (`fit_gaussians` in `src/tribblefis/gauss_math.py`) | ❌ Never |
| **Consequents** — per-rule polynomial coefficients | Per-bucket least-squares init (`compute_*_order_corrections`) then L-BFGS-B on weighted MSE (`optimize_tsk_coefficients`) | ✅ Yes, but sub-optimally |

Two structural facts drive this plan:

1. **For fixed firing strengths, the TSK output is _linear_ in the consequent
   coefficients.** ⟹ the optimal consequents have a **closed-form** regularized
   least-squares solution. The current L-BFGS optimizer is iterative,
   approximate, slower, and strictly dominated. Replacing it is a free accuracy +
   speed win, and it becomes the fast inner solver that Phase 2 needs.
2. **The antecedent→MSE surface is non-convex, multi-modal, and (with the default
   `min/max` t-norm) non-differentiable.** ⟹ this is the part that needs global
   search (GA / DE / CMA-ES) or a smoothed gradient method. This is the
   never-optimized lever and the largest expected accI also have a sample showing them overlaiduracy gain.

The intended outcome: a `ConsequentSolver` (Phase 1) that replaces the current
correction/optimize functions with an exact regularized solve over a pluggable
basis, then an antecedent-refinement module (Phase 2) that wraps the whole
forward pass in a global optimizer whose per-candidate fitness calls the Phase 1
solver.

---

## Current architecture (orientation for the implementer)

Pipeline in `gaussian_mixture/concrete.py` → `main()`:

1. Load/transform data; `partition_output(n_output_buckets, y)` (`regression.py`)
   buckets `y` via `pd.qcut`; each bucket is one **rule / output label**.
   `y_bucket_mean` is the per-bucket mean (endpoints forced to min/max).
2. Feature ranking (`calculate_gaussian_correlation`, `take_top_features`) →
   `top_n_todo`.
3. `create_gaussian_membership_dict` → `GaussianMixtureModel`: for each
   `(feature, bucket)` a `LabelModel` holding one or more `GaussianMembership`
   (`mu`, `sigma`) fit by `fit_gaussians`.
4. `tsk_firing_strengths(X, model)` (`gauss_math.py`) → `(firing_strengths
   [n_samples, n_rules], labels)`. Per rule: `t_norm` (AND) over features of
   `t_conorm` (OR) over that feature-bucket's MFs. Row-normalized downstream.
5. Consequents: `compute_{first,second,third,full_second}_order_corrections`
   (per-bucket `pinv`/`lstsq`), then `optimize_tsk_coefficients` (L-BFGS-B).
6. Prediction: `ŷ = Σ_r norm_fs[:, r] · (bucket_mean[r] + basis(X) · corr[r])`.

**Key files:**
- `src/tribblefis/regression.py` — all consequent math (§ to replace in Phase 1).
- `src/tribblefis/gauss_math.py` — `tsk_firing_strengths`, `membership`,
  `t_norm`/`t_conorm`, `fit_gaussians` (Phase 2 target).
- `src/tribblefis/gauss_data.py` — `GaussianMembership`, `TrapezoidMembership`,
  `GaussianMixtureModel` (has `all_membership_fcns`, `rule_ids`,
  `n_membership_functions`). Defaults: `DefaultNormCornorm = "min/max"`,
  `DefaultMemberFunction = "gaussian"`.
- `src/tribblefis/gaussian_regressor.py` — `MixtureOfGaussiansFuzzyRegressor`
  (sklearn `BaseEstimator`/`RegressorMixin`, `fit`/`predict`). **The natural home
  for both phases.** `MimoGaussianPredictor` wraps it for multi-output.
- Callers duplicating the predict loop: `gaussian_mixture/{concrete,
  concrete_trapz, turbine, wec, wec-p1, powerconsumption}.py`.
- Tests: `tests/test_regression.py` exercises `MixtureOfGaussiansFuzzyRegressor`.

**Reuse, don't reinvent:** `partition_output`, `tsk_firing_strengths`,
`report_regression_performance`, `_rsquared`, `_mse`, the `combinations`-based
cross-term construction, and the `l2_reg` ridge hook already in
`optimize_tsk_coefficients` (currently used only by `concrete_trapz.py`:
`l2_reg = 1e-4` for trapz, `0.0` for gaussian).

---

## Phase 1 — Consequent refinement (System 2). Standalone & shippable.

Goal: replace the per-bucket LS + L-BFGS consequent stage with a single exact
regularized weighted least-squares solve over a pluggable basis, add cross-
validated selection of `(order, λ)`, and add sparse interaction selection.

### 1a. Closed-form regularized weighted-LS backbone `ConsequentSolver`

Because `ŷ` is linear in the coefficients for fixed firing strengths, stack a
design matrix across all rules and solve once.

- For rule `r`, let `w_r = norm_fs[:, r]` (n_samples) and `Φ_r = [1 | basis(X)]`
  (n_samples × (1 + n_terms)). The per-rule block is `w_r[:, None] * Φ_r`.
- Horizontally stack all rule blocks → `Φ` (n_samples × n_rules·(1+n_terms)).
- Solve ridge normal equations `(ΦᵀΦ + λ D) β = Φᵀ y`, where `D` is diagonal with
  **0 on the intercept/bucket-mean columns and 1 on correction columns** (mirror
  the existing "constants unpenalized" rule in `optimize_tsk_coefficients`).
- Reshape `β` back to `y_bucket_mean_opt` (n_rules) and `corr_terms_opt`
  (n_rules × n_terms). Return the same tuple shape the callers already expect.

Properties: exact global optimum for the consequent stage (current per-bucket LS
is per-bucket and unweighted → suboptimal), one linear solve vs. 1000-iter
L-BFGS, free ridge regularization. Handle rank deficiency with
`np.linalg.lstsq`/`scipy.linalg.solve(..., assume_a="pos")` on the regularized
system. **Preserve the zero-firing-row convention** from
`optimize_tsk_coefficients` (rows with `row_sum ≤ 1e-6` get uniform `1/n_labels`)
so behavior matches evaluation.

Suggested location: new class/functions in `src/tribblefis/regression.py`
(keeps the existing import surface). Signature parallel to the current optimizer:

```python
def solve_tsk_consequents(
    X_train, gaussian_memberships, top_n_todo, y_bucket_mean, y_train,
    n_output_buckets, order="2nd", l2_reg=0.0, basis="raw",
) -> tuple[corr_terms_opt, y_bucket_mean_opt]:
    ...
```

Keep `optimize_tsk_coefficients` in place initially (don't break the 6 scripts);
have the new solver be a drop-in alternative selected by a flag.

### 1b. Pluggable basis abstraction

Factor feature expansion out of the order-`if/elif` chains (currently duplicated
in `compute_*_order_corrections`, `optimize_tsk_coefficients`, and every predict
loop). One function maps `(X_rule, order/basis) → design columns`:

- `raw` — current monomials `[x]`, `[x, x²]`, `[x, x², x³]`, full-2nd with cross
  terms (reuse the existing `itertools.combinations` cross-term code).
- `orthogonal` — **Legendre/Chebyshev polynomials on standardized features**
  (`numpy.polynomial.legendre`/`chebyshev`). Same expressive power as raw
  monomials, far better conditioning → smaller coefficients, less overfit at
  2nd/3rd order. This directly attacks the "~1e4 coefficient" overfit noted for
  cubic/full-2nd consequents. **Highest-ROI single change in Phase 1.**

Centralizing the basis also lets the predict loops in the caller scripts call one
shared helper instead of re-deriving `X_rule2`, `X_rule3`, `X_rule2f` inline.

### 1c. Cross-validated selection of `(order, λ)`

Add a helper that carves a validation fold from `X_train` (see Phase 2's shared
validation split) and picks the `(order, basis, λ)` combination maximizing
validation R² (equivalently minimizing validation MSE). Report chosen params.
This replaces choosing order/regularization by eye and prevents the higher-order
models from overfitting the test set.

### 1d. Sparse interaction selection (full-2nd)

`full-2nd` cross-terms grow as `O(n_features²)`. Add an option to select
interactions via Lasso / elastic-net (`sklearn.linear_model.LassoCV` /
`ElasticNetCV`) on the stacked design matrix, keeping only nonzero-coefficient
interactions before the final ridge solve. Improves test R² vs. dense full-2nd
when many cross-terms are noise.

### Phase 1 integration

- Wire `MixtureOfGaussiansFuzzyRegressor.fit` (`gaussian_regressor.py`) to use
  `solve_tsk_consequents` instead of `compute_*` + `optimize_tsk_coefficients`,
  behind a constructor flag (default to the new solver once validated).
- Update `gaussian_mixture/concrete.py` `main()` to call the new solver for
  each order; keep the existing 5-panel `plot_tsk_order_comparison` output.
- Leave `concrete_trapz.py` behavior intact (it relies on `l2_reg`); the new
  solver must accept `l2_reg` identically so it can adopt it later.

### Phase 1 expected outcome

Match-or-beat the current optimized baseline (Gaussian test R² ≈
0.44 / 0.77 / 0.87 / 0.86 / 0.88 for orders 0/1/2/2f/3) with **less overfit at
orders 2f/3** and **faster** fitting (no 1000-iter L-BFGS). Orthogonal basis +
CV'd ridge are the movers.

---

## Phase 2 — Post-model antecedent refinement (System 1)

Goal: optimize the never-tuned `(μ, σ)` of every membership function against the
regression objective, using the Phase 1 closed-form solver as the fast inner
consequent step. **Do not start until Phase 1's solver exists** — it is the
per-candidate fitness primitive.

### Nested architecture

- **Decision vector:** the `(μ, σ)` of every MF in
  `model.all_membership_fcns` (a few dozen reals). Box-constrain `μ` to the
  feature's observed range and `σ ∈ [σ_min, k·range]`.
- **Fitness(candidate):** write the vector back into a copy of the
  `GaussianMixtureModel` (rebuild `GaussianMembership` tuples — they are
  immutable `NamedTuple`s), recompute `tsk_firing_strengths`, call the Phase 1
  `ConsequentSolver` (exact consequents), return **validation-fold MSE / R²**.
- This collapses the search dimension from `n_MFs·2 + n_rules·n_terms` to just
  `n_MFs·2`, because consequents are solved exactly, not searched.

### Overfitting guard (shared with Phase 1c)

Carve a validation fold from `X_train` (e.g. an inner `train_test_split`, fixed
seed). Select candidates on **validation** R², never training R². Touch the test
set only for the final report. This protects the headline R²/MSE metric.

### Methods (implement in this order)

1. **Differential Evolution — baseline, lowest effort.**
   `scipy.optimize.differential_evolution` over the box-constrained `(μ, σ)`
   vector. Zero new infrastructure. Use it to *prove the nested loop lifts R²*
   before investing in a bespoke GA. Seed via `x0`/`init` from the current
   heuristic model so it can only improve on today's baseline.
2. **Genetic Algorithm — the `feat/ga-refinement` deliverable.**
   Real-valued GA: tournament selection, SBX or BLX-α crossover, Gaussian
   (polynomial) mutation, elitism. Same fitness. Seed the initial population from
   the heuristic model + jittered copies. t-norm/MF-agnostic — works unchanged
   for `min/max`, trapezoids (bounded support), any basis. New module, e.g.
   `src/tribblefis/refine.py`, kept dependency-light (pure numpy) or optionally
   backed by an existing library if the team prefers.
3. **(Optional) GD / ANFIS polish.** Alternate (a) exact LS consequents (Phase 1)
   with (b) an ADAM step on `(μ, σ)` via autograd/analytic gradients — the
   textbook ANFIS hybrid rule, run as a local polish after GA/DE.
   **Prerequisite:** the default t-norm is `min/max`, which is **not
   differentiable**. GD requires switching the forward pass to the product
   ("probability") t-norm (already supported in `t_norm`/`t_conorm`) or a smooth
   surrogate (softmin/log-sum-exp). GA/DE do **not** need this — hence GD is last.

CMA-ES is a strong drop-in alternative to the GA at this dimensionality; note it
as an option but not required.

### Phase 2 integration

- Add an opt-in `refine=` flag / method to `MixtureOfGaussiansFuzzyRegressor`
  that, after the initial heuristic fit, runs the chosen optimizer and stores the
  refined `model_` before the final consequent solve.
- Expose the toggle in `concrete.py` (e.g. a `b_refine_antecedents` flag next to
  the existing `b_optimize_coeff`) so the comparison plot can show
  refined-vs-unrefined side by side.

---

## Verification

Run everything on the **same fixed split** (`random_state=42`, as today) so gains
are attributable, and compare against the recorded baseline.

1. **Unit / regression tests:** extend `tests/test_regression.py`.
   - `ConsequentSolver` on a synthetic problem with known linear consequents
     recovers coefficients (λ=0) to tolerance.
   - New solver matches or beats `optimize_tsk_coefficients` training MSE on a
     fixed model+data (it should, being the exact optimum).
   - `MixtureOfGaussiansFuzzyRegressor` still passes
     `test_gaussian_mixture_regression_2d` with the new solver wired in.
2. **End-to-end Phase 1:** `MPLBACKEND=Agg python gaussian_mixture/concrete.py`
   (the `Agg` backend makes `plt.show()` a no-op so it doesn't hang — see project
   notes). Confirm per-order test R²/RMSE ≥ current baseline
   (0.44/0.77/0.87/0.86/0.88), especially reduced overfit at 2f/3, and shorter
   runtime.
3. **End-to-end Phase 2:** run `concrete.py` with antecedent refinement enabled;
   confirm validation-selected R² improves over the heuristic-antecedent baseline
   and that test R²/MSE improves (not just training). Log wall-clock for DE vs GA.
4. **Regression across scripts:** smoke-run `concrete_trapz.py` (must keep
   `l2_reg` behavior) and one of `wec.py`/`turbine.py` to confirm the shared
   basis/predict refactor didn't change their outputs.

---

## Risks & notes

- **Numerical conditioning:** even with ridge, raw high-order monomials are
  ill-conditioned; prefer the orthogonal basis for orders ≥ 2. Solve the
  regularized system, not the raw normal equations, when `λ = 0`.
- **Zero-firing rows / uniform blend:** the `1/n_labels` convention for rows with
  no firing must be identical in the solver, the fitness function, and every
  predict path, or train/eval will silently disagree (this was a real trap noted
  for the trapz consequents).
- **Immutability:** `GaussianMembership`/`GaussianMixtureModel` are `NamedTuple`s;
  Phase 2 must construct new instances per candidate, not mutate in place.
- **Duplication debt:** six scripts inline the predict loop; the shared basis
  helper (1b) is the opportunity to de-duplicate, but keep changes behavior-
  preserving and verify each script.
- **Scope discipline:** Phase 1 must be independently shippable and
  regression-safe before Phase 2 begins, since Phase 2's fitness depends on it.
