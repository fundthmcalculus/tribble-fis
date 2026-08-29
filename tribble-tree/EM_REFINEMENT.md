# EM Refinement of the Hierarchical Fuzzy Experts (design note)

**Status:** implemented in `fuzzytree/em.py` (call `.refine_em(X, y)` on a
fitted `HierarchicalFuzzyExpertsRegressor`/`Classifier`). This document is the
design spec it follows; the sections below describe the intended algorithm,
and the implementation notes at the bottom of this status box record where
the shipped code took a pragmatic option among those the spec lists.

Implementation notes:
- **Regression M-step** uses Option A (§5): expert antecedents (the sub-FIS
  Gaussian memberships) stay frozen; only the closed-form TSK consequents and
  per-leaf `sigma2_` are refit, weighted by the posterior. This is an exact
  weighted-MLE step, so the incomplete-data log-likelihood is monotone in
  practice.
- **Classification M-step** uses Option B (§5): the classifier sub-FIS has no
  separable consequent (it's zeroth-order; the antecedents *are* the whole
  model), so its experts are refit via posterior-weighted importance
  resampling followed by an unmodified `.fit()`. This is stochastic and only
  approximates the weighted MLE, so it can occasionally decrease the
  log-likelihood for one step.
- **Gate M-step**: Gaussian gate terms (`gate_style="gaussian"`) get the exact
  closed-form weighted mean/variance update in §4.2 and can genuinely
  *sharpen* (shrink sigma) around a sharp regime boundary -- this is the
  headline validation case in §10 and is covered by
  `tests/test_em_refinement.py`. **The default trapezoid gates
  (`gate_style="trapezoid"`) are deprecated for `refine_em`**: their M-step
  resamples rows by branch responsibility and refits a smooth-trapezoid EM
  (`tribblefis.trapz_math_smooth`), which is stochastic and only
  approximates a weighted MLE, not the closed-form update Gaussian gates get
  -- see `ISSUE_163_RESOLUTION_PLAN.md` for why trapezoid+EM is a poor fit
  in general (a bounded-support MLE's incentive to shrink onto the data
  mode is a property of the objective, not an optimizer artifact, so no
  amount of smoothing the M-step's landscape fixes it). Calling
  `_rebuild_gate_tree` on trapezoid gates now issues a `FutureWarning`.
  **Use `gate_style="gaussian"` whenever `refine_em` will be called**; if
  crisp trapezoid interpretability is required without EM refinement, build
  the gates with trapezoids and skip `refine_em`, or use
  `TribbleRegressor`/`TribbleClassifier`'s `trapz_method="fast"` at the leaf
  level instead.
- **No-worse guarantee** (§10): rather than hoping every M-step improves,
  both drivers snapshot the model's parameters at whichever iteration had the
  best observed log-likelihood and restore that snapshot before returning --
  a real safeguard, not just documented intent, needed in particular for the
  stochastic classification M-step.
- **Structural EM** (§9), **multi-input gates**, and **annealed E-steps**
  remain out of scope, as originally specified.

---

---

## 1. Why EM, and what it changes

The current builder infers the gate tree once (via `build_tree`), then fits each
leaf's expert sub-FIS on the samples whose **gate** responsibility exceeds a
threshold (`_expert_training_index`). Two approximations are baked in:

1. **Responsibilities ignore the experts.** A sample is routed to experts purely by
   the gates `g_leaf(x)` — how well an expert actually *predicts* that sample never
   feeds back into who trains on it. Two experts straddling a soft boundary cannot
   specialize against each other.
2. **Training assignment is hard-ish.** Soft-inclusion (threshold + argmax) is a
   crude stand-in for a real posterior weight per (sample, expert).

EM fixes both by alternating:

- **E-step** — compute the *posterior* responsibility `h_{nℓ}` that leaf `ℓ`
  generated sample `n`, using **both** the gate probability *and* the expert's
  likelihood of `y_n`.
- **M-step** — refit every expert and every gate to maximize the expected complete
  data log-likelihood, each weighted by those posteriors.

This is the Jordan & Jacobs (1994) HME trained by EM, specialized to fuzzy gates and
TRIBBLE TSK experts. Structure (the tree shape and which variable gates where) stays
**fixed**; EM only refines the *parameters* of the gates and experts. (Structural
search remains the job of `build_tree` / `VariablePlan`; a later "structural EM"
extension is noted in §9.)

---

## 2. The probabilistic model

Treat the fitted HME as a generative mixture. For an input `x`:

- Each internal node `i` has a **gating distribution** over its children:
  `g_{i,j}(x) ≥ 0`, `Σ_j g_{i,j}(x) = 1`. In the fuzzy model these are the
  partition-of-unity term memberships already computed inside
  `compute_responsibilities`.
- The probability of reaching leaf `ℓ` is the product of gates on its root→leaf path:

  ```
  π_ℓ(x) = Π_{(i,j) ∈ path(ℓ)} g_{i,j}(x)          (this is today's R[:, ℓ])
  ```

- Each leaf `ℓ` is an **expert** defining a conditional density `p(y | x, ℓ)`:
  - **Regression:** Gaussian around the expert sub-FIS output `f_ℓ(x)`:
    ```
    p(y | x, ℓ) = N(y ; f_ℓ(x), σ_ℓ²)
    ```
    `f_ℓ` is `TribbleRegressor.predict`; `σ_ℓ²` is a new per-expert
    noise variance the current model does not store.
  - **Classification:** categorical from the expert classifier:
    ```
    p(y = c | x, ℓ) = expert_ℓ.predict_proba(x)[c]
    ```

- The marginal (what we actually score) is the mixture:
  ```
  p(y | x) = Σ_ℓ π_ℓ(x) · p(y | x, ℓ)
  ```

The **incomplete-data log-likelihood** to be maximized is
`L = Σ_n log Σ_ℓ π_ℓ(x_n) p(y_n | x_n, ℓ)`. EM increases `L` monotonically.

---

## 3. E-step — posterior responsibilities

For every training pair `(x_n, y_n)` and leaf `ℓ`:

```
h_{nℓ} = π_ℓ(x_n) · p(y_n | x_n, ℓ)  /  Σ_{ℓ'} π_ℓ'(x_n) · p(y_n | x_n, ℓ')
```

`Σ_ℓ h_{nℓ} = 1`. Contrast with today: the greedy code uses `π_ℓ(x_n)` alone; the
new factor `p(y_n | x_n, ℓ)` is what lets a well-fitting expert "claim" a sample and
a poorly-fitting one release it.

The gate factor `π_ℓ(x_n)` is exactly `compute_responsibilities(...)`; the E-step
multiplies it elementwise by the expert-likelihood matrix and row-normalizes.

### Nested responsibilities (needed for the gate M-step)

Because the gates live at internal nodes, the M-step for gates needs
per-node/per-branch responsibilities, obtained by summing leaf posteriors within
subtrees:

```
node responsibility   γ_{n,i}      = Σ_{ℓ ∈ subtree(i)}        h_{nℓ}
branch responsibility τ_{n,i,j}    = Σ_{ℓ ∈ subtree(child_j)}  h_{nℓ}
conditional branch    τ̃_{n,i,j}   = τ_{n,i,j} / γ_{n,i}     (0 if γ_{n,i}=0)
```

`τ̃_{n,i,·}` is a proper distribution over node `i`'s children for sample `n`, and
is the fitting target for that node's gate.

---

## 4. M-step

Maximize the expected complete-data log-likelihood; it separates into independent
expert and gate subproblems.

### 4.1 Expert updates (weighted fits)

For each leaf `ℓ`, maximize `Σ_n h_{nℓ} log p(y_n | x_n, ℓ)`:

- **Regression consequents** — weighted least squares of `f_ℓ` to `y` with weights
  `h_{nℓ}`. The current closed-form solver `solve_leaf_consequents` already does a
  *firing*-weighted ridge solve; the EM version multiplies the per-sample design row
  and target by `√h_{nℓ}` (see §5, option A). Then update the noise variance:
  ```
  σ_ℓ² = Σ_n h_{nℓ} (y_n − f_ℓ(x_n))²  /  Σ_n h_{nℓ}
  ```
  (Floor `σ_ℓ²` at a small ε to prevent an expert collapsing onto a few points.)

- **Regression antecedents (MF centers/spreads of the expert sub-FIS)** — ideally
  re-estimated with the same weights `h_{nℓ}`. This is the harder part because the
  sub-FIS fits its Gaussians via KMeans + `norm.fit`; the weighted analogue is a
  weighted GMM / weighted moments (§5). A cheaper acceptable variant freezes the
  antecedents after the greedy build and lets EM move only the consequents + `σ_ℓ²`
  (the consequents carry most of the expressive power for TSK).

- **Classification experts** — weighted MLE of the leaf classifier with weights
  `h_{nℓ}`: weighted class priors and weighted MF moments. The categorical
  likelihood `p(y_n=c | x_n, ℓ)` comes from `predict_proba`.

### 4.2 Gate updates (weighted MF re-estimation)

For each internal node `i`, fit its gating parameters to the branch responsibilities
`τ_{n,i,j}`. With **Gaussian gate terms** this is a closed-form, responsibility-
weighted 1-D mixture fit over the node's routing variable `v_i`:

```
μ_{i,j} = Σ_n τ_{n,i,j} · v_i(x_n)               / Σ_n τ_{n,i,j}
σ_{i,j}² = Σ_n τ_{n,i,j} · (v_i(x_n) − μ_{i,j})² / Σ_n τ_{n,i,j}
```

i.e. each branch's gate MF becomes the responsibility-weighted mean/variance of the
routing variable for the samples that (softly) took that branch. This reuses
TRIBBLE's Gaussian MF representation directly and is the gate analogue of the expert
`σ_ℓ²` update.

> **Representation note (deprecated path).** Today's gates use open-shouldered
> **trapezoids** from `build_split_terms`, which have no closed-form weighted
> MLE. Two options were explored for EM:
> (a) switch the gate terms to **Gaussian** (closed-form updates above), or
> (b) keep trapezoids and refit them via a smooth-trapezoid EM on a
> responsibility-weighted resample (`tribblefis.trapz_math_smooth`).
> **Option (a) is recommended and option (b) is deprecated** --
> `ISSUE_163_RESOLUTION_PLAN.md` found the trapezoid+EM combination
> structurally weak (the area-normalized MLE objective rewards shrinking
> support onto the data mode regardless of how smooth the M-step's
> optimization landscape is) and calling it now issues a `FutureWarning`.
> Use Gaussian gates for `refine_em`; trapezoid gates remain fine for the
> initial greedy build, just skip `refine_em` on them or use
> `trapz_method="fast"` (no EM) at the leaf level. The routing *variable* at
> each node stays fixed either way (structure is frozen); only its term
> shapes move.

Multi-input gates (a gate that routes on a small variable *group*, mentioned as a
future option in the HME docstring) generalize this to a weighted multivariate GMM
over the group — again a standard weighted-moment update.

---

## 5. The weighting obstacle and three ways around it

Every M-step fit is **weighted by responsibilities**, but the stock sub-FIS
(`TribbleRegressor` / `...Classifier`) and `build_split_terms` do not
accept per-sample weights. Options, in order of fidelity:

- **Option A — weighted solve (recommended, principled).** Add an optional
  `sample_weight` to the consequent solve: scale each design row and target by
  `√h_{nℓ}` before forming the normal equations (the exact weighted-LS optimum; keeps
  the closed form). For antecedent/gate MFs, replace `norm.fit`/KMeans with weighted
  moments (formulas in §4). This is the "correct" EM M-step. It touches the fitting
  internals (can live entirely in `fuzzytree`, mirroring what `solve_leaf_consequents`
  already reimplements, so `src/tribblefis` need not change).

- **Option B — importance resampling (pragmatic drop-in).** For each leaf `ℓ`, draw a
  bootstrap sample of the training rows with probabilities `∝ h_{nℓ}` and fit the
  **unmodified** sub-FIS on that resample. Requires zero changes to the fitting code,
  approximates the weighted MLE, and is easy to validate. Downsides: stochastic
  (seed it), and needs enough effective samples per leaf.

- **Option C — soft-inclusion (today's method), reinterpreted.** The current
  threshold assignment is the crude limit where `h ≈ π` (gate only) and weights are
  hard {0,1}. Keeping it but recomputing the inclusion mask from the *EM* `h` each
  iteration is a middle ground with no weighted-fit machinery, but it forfeits most of
  EM's benefit and is not recommended beyond a baseline.

Recommendation: **Option A for the consequents + `σ_ℓ²` + Gaussian gates** (all
closed-form and cheap), with expert antecedents either weighted (A) or frozen; fall
back to **Option B** if a sub-FIS component has no convenient weighted form.

---

## 6. Initialization, convergence, stopping

- **Init:** warm-start from the current greedy build — reuse its inferred structure,
  gate terms, and soft-inclusion experts as EM iteration 0. Initialize each `σ_ℓ²`
  from that expert's weighted residual variance. Good initialization is important;
  EM only finds a local optimum.
- **Monitor:** the incomplete-data log-likelihood `L` (§2) — Gaussian LL for
  regression, categorical log-loss for classification. `L` must be non-decreasing;
  a decrease signals a bug in a weighted fit.
- **Stop** when `ΔL < tol` (e.g. `1e-4` relative) or `iter ≥ max_iter` (e.g. 10–30).
  Also expose early-stopping on a held-out split to avoid over-specialization.
- **Cost:** each iteration is `O(N · n_leaves)` for the E-step plus the sum of the
  per-leaf/per-node weighted fits — comparable to one greedy build per iteration.

---

## 7. Numerical safeguards

- **Variance floor:** clamp `σ_ℓ²` (and gate `σ_{i,j}²`) at a small ε so no expert or
  gate collapses to a delta and hijacks its neighborhood.
- **Starved components:** if `Σ_n h_{nℓ} < min_mass`, either prune the leaf (and
  renormalize its parent's gates) or reinitialize it from the largest-residual
  region. Log any prune so coverage changes are visible.
- **Underflow:** compute `π_ℓ` and likelihoods in log-space and use a log-sum-exp
  normalization in the E-step; path products over deep trees underflow otherwise.
- **Ridge:** retain `l2_reg` in the weighted consequent solve; with small effective
  per-leaf mass the design can be ill-conditioned.
- **Determinism:** if using Option B, thread a `random_state` so refinement is
  reproducible.

---

## 8. Algorithm (pseudocode)

```
EM_refine(HME model, X, y, max_iter, tol):
    # model = greedy-built tree with gates {g_ij}, experts {f_ℓ or clf_ℓ}, σ_ℓ²
    L_prev = -inf
    for it in range(max_iter):

        # ---------- E-step ----------
        Π   = compute_responsibilities(tree, X)          # (N, n_leaves) gate part
        Lik = expert_likelihood_matrix(model, X, y)      # (N, n_leaves)
                                                         #  regression: N(y; f_ℓ, σ_ℓ²)
                                                         #  classification: predict_proba(y)
        num = Π * Lik
        H   = num / num.sum(axis=1, keepdims=True)       # posterior responsibilities
        L   = sum(log(num.sum(axis=1)))                  # incomplete-data log-lik

        if L - L_prev < tol * |L_prev|: break
        L_prev = L

        # nested responsibilities for gates
        for node i: γ_i   = sum_{ℓ in subtree(i)} H[:, ℓ]
        for branch (i,j):  τ_ij = sum_{ℓ in subtree(child_j)} H[:, ℓ]

        # ---------- M-step: experts ----------
        for leaf ℓ:
            w = H[:, ℓ]
            f_ℓ  = weighted_fit_expert(X, y, w)          # Option A or B (§5)
            σ_ℓ² = max(Σ w (y - f_ℓ(X))² / Σ w, ε)       # regression only

        # ---------- M-step: gates ----------
        for internal node i, each branch j:
            μ_ij  = Σ τ_ij · v_i(X) / Σ τ_ij             # Gaussian gate update (§4.2)
            σ_ij² = max(Σ τ_ij (v_i(X) - μ_ij)² / Σ τ_ij, ε)
            # (or τ-weighted smooth-trapezoid EM if keeping trapezoid gates -- deprecated, §4.2)

    return model
```

Prediction after refinement is unchanged: the gated blend already in
`HierarchicalFuzzyExperts*.predict`, now using the refined gate MFs and experts.

---

## 9. Extensions (out of scope, noted for completeness)

- **Structural EM:** interleave parameter EM with occasional structure moves
  (grow/prune a gate, swap a routing variable) accepted only if held-out `L`
  improves. Turns the fixed-structure refinement into joint structure+parameter
  learning.
- **Multi-input gates:** replace single-variable gates with small gating sub-FIS over
  variable groups; the M-step becomes a weighted multivariate GMM per node.
- **Annealed / deterministic-annealing EM:** temper the E-step
  (`h ∝ (π·Lik)^{1/T}`, `T→1`) to reduce sensitivity to the greedy initialization.
- **Shared integration point:** a generic weighted-fit / EM utility could be factored
  into a common module (e.g. alongside `src/tribblefis/refine.py`) if other TRIBBLE
  estimators want responsibility-weighted fitting; the HME loop would then just supply
  the E-step and the tree bookkeeping.

---

## 10. Validation plan

- **Likelihood monotonicity:** assert `L` is non-decreasing across iterations on a
  synthetic set (a decrease means a weighted fit is wrong).
- **Sharp-boundary synthetic:** the piecewise-linear target in `TestHME`
  (`y = 2b` if `a<5` else `-3b+40`) has a hard regime boundary that soft gates blur;
  EM should *sharpen* the gate around `a=5` (smaller gate `σ`) and raise held-out R²
  above the greedy build. Use this as the headline regression check.
- **No-worse guarantee:** on concrete and phishing, refined held-out accuracy should
  be ≥ the greedy HME (allowing a small tolerance), or the run should early-stop back
  to the greedy parameters.
- **Degeneracy tests:** feed a dataset that starves one leaf and confirm the
  prune/reinit safeguard fires and coverage is logged.

---

## References

- M. I. Jordan, R. A. Jacobs. *Hierarchical mixtures of experts and the EM
  algorithm.* Neural Computation, 6(2):181–214, 1994. (The HME + EM derivation this
  note specializes; nested responsibilities, gate and expert M-steps.)
- A. P. Dempster, N. M. Laird, D. B. Rubin. *Maximum likelihood from incomplete data
  via the EM algorithm.* J. Royal Statistical Society B, 39(1):1–38, 1977.
- R. A. Jacobs, M. I. Jordan, S. J. Nowlan, G. E. Hinton. *Adaptive mixtures of local
  experts.* Neural Computation, 3(1):79–87, 1991. (Single-level mixture of experts.)
- N. Ueda, R. Nakano. *Deterministic annealing EM algorithm.* Neural Networks,
  11(2):271–282, 1998. (Annealed E-step for better optima — §9.)
