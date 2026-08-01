# Notes: TSK Fuzzy Systems in High Dimension

The mathematical prerequisites for making a fuzzy inference system work at
`D = 64`–`256` inputs. Ignoring §2 will produce a model that trains to a
degenerate single-rule solution; this is documented, not hypothetical.

---

## 1. Baseline TSK, and the identity that matters

A first-order Takagi-Sugeno-Kang system with `R` rules over input `x ∈ ℝ^D`:

**Rule r:** IF `x₁` is `X_{r,1}` AND … AND `x_D` is `X_{r,D}` THEN `y = a_rᵀx + b_r`

With Gaussian membership functions `μ_{r,d}(x_d) = exp(−(x_d − m_{r,d})² / (2σ_{r,d}²))`
and the **product** t-norm for AND, the firing level is

```
f_r(x) = ∏_d μ_{r,d}(x_d) = exp( −Σ_d (x_d − m_{r,d})² / (2σ_{r,d}²) )
```

and defuzzification (centre-of-gravity) is

```
ŷ = Σ_r f̄_r(x) · (a_rᵀx + b_r),      f̄_r(x) = f_r(x) / Σ_i f_i(x)
```

**The identity (Cui, Wu & Xu 2021):** define `Z_r = −Σ_d (x_d−m_{r,d})²/(2σ_{r,d}²)`.
Then

```
f̄_r(x) = exp(Z_r) / Σ_i exp(Z_i) = softmax(Z)_r
```

Normalised Gaussian-product TSK defuzzification **is** a softmax over negative
weighted squared distances to rule centres. Equivalently: it is the posterior
responsibility of a diagonal-covariance Gaussian mixture with uniform priors.

Three things follow, and all three are useful:

1. A TSK antecedent layer is a **soft nearest-prototype router**. `R` rules,
   `2RD` antecedent parameters.
2. The consequent layer is a **mixture of linear experts**, gated by that router.
3. Therefore a TSK system is trainable by exactly the machinery a neural net is —
   but its parameters are *readable*: `m_r` is a prototype point in feature space
   and `σ_r` is its per-dimension tolerance.

## 2. The curse of dimensionality — the failure mode, precisely

`Z_r` is a sum of `D` non-negative terms, negated: `Z_r ≤ 0` always, and
`|Z_r| = O(D)`. As `D` grows, the **spread** of `Z` across rules grows linearly
with `D` too. Softmax on logits of magnitude `O(D)` saturates: for `D = 256` with
typical standardised features, `Z` values sit in the hundreds and

```
f̄ → one-hot
```

Consequences:
- **Winner-takes-all.** One rule handles each input; the "fuzzy" blending that
  gives the system its expressive power and smoothness disappears.
- **Dead gradients.** `∂softmax/∂Z ≈ 0` at saturation, so the antecedent
  parameters `(m, σ)` stop learning almost immediately.
- **Underflow.** Computing `∏_d μ` directly (rather than in log space) underflows
  to 0 for every rule; `0/0` in the normaliser.

Cui et al. also note the geometric side of it: in high dimension pairwise
distances concentrate, so all `Z_r` become similar in *relative* terms even as
they diverge in absolute terms — fuzzy partitions collapse.

> **Experimental correction (2026-07-31, ablation A3).** Everything in §2 below is
> correct *at initialisation with fixed σ*, and we reproduce it. But we measured
> that it stops mattering once `σ` is trained: the plain product t-norm matched
> HTSK end-to-end (MTEB-14 50.86 vs 50.96; NanoBEIR 0.4181 vs 0.4166).
>
> The reason is that **the temperature and σ are the same parameter**. Dividing the
> exponent by τ is exactly scaling every `σ_{r,d}` by `√τ`, so a trainable
> per-rule, per-dimension `σ` strictly subsumes any global τ — including HTSK's
> `1/D`. Our calibrated τ came out at **1.195 for both D = 64 and D = 128**,
> i.e. independent of `D`, which says `1/D` is normalising against the wrong
> quantity: what sets the required temperature is the feature-to-σ scale ratio,
> not the dimensionality.
>
> So HTSK is best understood as an **initialisation-conditioning device**. It is
> genuinely load-bearing in the regime Cui/Wu/Xu study (σ fixed or set by heuristic,
> as in classic TSK fitting — and as in this repository's own NumPy regression
> code, where `fit_gaussians` sets σ once and never optimises it). It is not
> load-bearing when the antecedents are learned end-to-end by SGD. See `LOG.md`
> Findings 7 and 9.

### Fix 1 — HTSK (what we use)

Change the sum to a **mean**:

```
Z'_r = −(1/D) Σ_d (x_d − m_{r,d})² / (2σ_{r,d}²)

f̄'_r(x) = softmax(Z')_r  =  f_r(x)^(1/D) / Σ_i f_i(x)^(1/D)
```

Interpretations, all equivalent and all worth keeping in mind:
- The **geometric mean** t-norm replaces the product t-norm. Still a valid t-norm
  aggregation, still "AND"-like, but scale-free in `D`.
- Equivalent to adaptively scaling every `σ` by `√D` in vanilla TSK.
- A **temperature** of `D` on the router softmax.

Reported result: across 14 datasets with `D` from 10 to 4,955, HTSK ranked 2.1
average vs 3.5–7.3 for vanilla TSK variants, and was insensitive to the σ-init
scale `h` for `h ≥ 0.5`. On the `D = 4955` dataset vanilla TSK-1 collapsed
outright while HTSK held up.

### Fix 2 — LogTSK (alternative)

ℓ1-normalise reciprocals instead of exponentiating:

```
f̄^log_r = (−1/Z_r) / Σ_i (−1/Z_i)
```

Comparable performance (rank 2.4). Heavier-tailed than HTSK, so more rules stay
active. Worth an ablation but HTSK is the simpler default.

### Fix 3 — learned temperature (recent)

The 2025 "adaptive double-parameter softmin TSK" line replaces the fixed `1/D`
with learnable scalars. Cheap (one or two parameters) and strictly generalises
HTSK. **We implement HTSK with a learnable log-temperature initialised at
`log(1/D)`** — this is free, reduces to HTSK at init, and cannot be worse.

## 3. Rule-base structure: grid vs scatter

- **Grid partition** — `K` MFs per input, all combinations ⇒ `K^D` rules.
  At `D = 64` this is absurd. Never applicable here.
- **Scatter partition** — `R` rules whose centres are placed where the data is,
  each rule owning one full `D`-dimensional MF vector. `R` is a free
  hyperparameter, independent of `D`. **This is the only viable option**, and it
  is what the tribble-fis repo's own Gaussian-mixture models already do
  (`fit_gaussians` → per-bucket Gaussians, KMeans-seeded).

Standard initialisation (PyTSK): **KMeans** (or fuzzy c-means) on the input
features to place `m_r`; `σ_{r,d}` init to `h ×` the per-dimension std with
`h ≈ 1`. We do exactly this, on token feature vectors.

## 4. Regularisation tricks worth importing

| Trick | Formula / mechanism | Why we want it |
|---|---|---|
| **Uniform Regularisation (UR)** | `ℓ_UR = Σ_r ( (1/N)Σ_n f̄_{n,r} − τ )²`, `τ ∈ (0,1)`, typically `1/R` | Directly penalises rule-usage imbalance. This is the *load-balancing loss* of the MoE literature arriving independently in the fuzzy literature. Essential: without it a subset of rules goes unused and the effective parameter count silently drops. |
| **DropRule** | Dropout on `f̄` | Prevents co-adaptation of rules; cheap. |
| **BatchNorm on consequent inputs** | `precons = BatchNorm1d(D)` | Conditions the linear experts. |
| **LayerNorm + ReLU after antecedent** | on the `R`-vector | Alternative anti-vanishing-gradient measure. |
| **Group Lasso on inputs and rules** | KBS 2024 | Route to *pruning* rules and input dims after training — the "scale down / expand later" lever. |

## 5. Why a rule base is a good fit for token features specifically

Token feature space is not homogeneous — it has obvious structure that a scatter
partition can capture: function words vs content words, morphological
continuations (`##ing`), numerals, punctuation, named-entity-ish subwords, and
topical clusters. A rule base assigns each of these regions its own linear
projection. Concretely, the hypothesis is:

> `IF token-feature is near prototype r THEN contribute A_r·v + b_r`

where prototype `r` might be "subword continuation fragments" and its consequent
learns to contribute almost nothing, while a "domain noun" prototype learns a
high-magnitude topical projection. This is a **learned, interpretable
generalisation of SIF/IDF weighting** — SIF is the special case where every
`A_r = α_r · I`.

That statement is the testable claim of the project, and it is what we will
inspect in the interpretability pass (`DESIGN.md` §7).

## 6. Reuse from this repository

`tribble-fis` already contains the pieces in NumPy for regression:

- `src/tribblefis/gauss_math.py` — `tsk_firing_strengths`, `membership`,
  `t_norm`/`t_conorm`, `fit_gaussians`
- `src/tribblefis/gauss_data.py` — `GaussianMembership`, `GaussianMixtureModel`
- `src/tribblefis/regression.py` — consequent solvers
- `consequent-plan.md` — the ANFIS-style closed-form-consequent + searched-antecedent plan

**We do not reuse the code directly** (it is NumPy, batch-oriented, and
regression-specific; we need autograd on GPU over ~100M token pairs), but we do
mirror its vocabulary and semantics so results are transferable, and we note two
divergences forced by the domain:

1. The repo defaults to the `min/max` t-norm, which is non-differentiable. We use
   the product/geometric-mean t-norm throughout (the repo supports "probability"
   t-norm too), for the reason `consequent-plan.md` §Phase-2.3 already flags.
2. The repo solves consequents in closed form because its objective is
   least-squares. InfoNCE is not, so consequents are learned by SGD here.
