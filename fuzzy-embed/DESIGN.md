# FES: Fuzzy Embedding System

A text-embedding model whose forward pass is a first-order Takagi–Sugeno–Kang
fuzzy inference system.

Read `docs/00-literature-index.md` first for the references cited by tag (A1, B1, …).

---

## 1. The observation the design rests on

A static embedding model (Model2Vec / POTION / `static-retrieval-mrl-en-v1`)
computes, for token ids `t₁…t_L`:

```
e = (1/L) Σ_t  E[t]                E ∈ ℝ^{|V| × d_out}
```

Write it as a fuzzy system. One rule, zero-order consequent, per-token:

> **R1:** IF (anything) THEN `y = E[t]`

That is a degenerate TSK system: `R = 1`, so `f̄₁ ≡ 1`, no antecedent, and the
consequent is a lookup rather than a function of features. **Every static
embedding model in the literature is a one-rule fuzzy system**, and the entire
`R > 1` region — where the antecedent actually gates between local linear
experts — is unexplored for this task (B8, B9, C-gap statement in the index).

So the design is: keep the static model's economics (a gather and a small matmul
per token — no attention, no depth), and spend the parameter budget on a **rule
base** instead of on a wide vocabulary table.

## 2. Architecture

### 2.1 Forward pass

Input: token ids `t₁…t_L` from `bert-base-uncased` (|V| = 30,522).

**Step 1 — token feature vectors (the fuzzified input).**

```
v_t = F[t_i] ∈ ℝ^{d_in}                F ∈ ℝ^{|V| × d_in},  d_in ≪ d_out
```

`F` is the antecedent *input space*: a narrow learned table. Optionally
LayerNorm'd, which matters because HTSK's scale-freeness assumes roughly
standardised inputs (B1 §σ-init sensitivity).

**Step 2 — antecedent / rule firing (HTSK).** For each rule `r ∈ 1…R`:

```
Z_r(v) = − (1/2) Σ_{d=1}^{D} ( (v_d − m_{r,d}) / σ_{r,d} )²          [log firing, product t-norm]
f̄_r(v) = softmax_r( Z_r(v) · exp(−λ) )                              [HTSK, learned temperature]
```

with `λ` a single learnable scalar initialised to `log D`, so at initialisation
`Z_r/D` — exactly HTSK (B1) — and the model may anneal it (B6). `σ` is stored as
`log σ` and exponentiated, guaranteeing positivity. Computation is entirely in
log space: no `∏ μ` is ever formed, so no underflow.

**Correction (E003, after running ablation A3).** I originally called this "the
load-bearing piece". That was an overstatement, and the experiment refuted it.
A3 — the plain product t-norm with no `1/D` — scored 50.86 / 0.4181 against A2's
50.96 / 0.4166, i.e. indistinguishable, with a healthy rule-usage entropy of 0.997.

The reason is that **τ and σ are mathematically redundant**: dividing the exponent
by τ is *exactly* scaling every `σ_{r,d}` by `√τ`. Since `log_sigma` is a trainable
per-rule, per-dimension parameter here, it simply absorbs the missing `1/D` during
training. Independent confirmation: calibrating τ against a target firing entropy
gives **τ = 1.195 at both D = 64 and D = 128** — the required temperature does not
vary with `D` at all, so `1/D` is normalising against the wrong quantity.

The accurate claim is narrower:

- At random initialisation with fixed σ, the saturation is real and severe
  (per-token firing entropy 0.14 vs 1.00 at `D = 256`; `test_htsk_prevents_softmax_saturation`).
- End-to-end **with learned antecedent widths, it stops mattering.** HTSK is an
  initialisation-conditioning device, not a fundamental requirement. It matters in
  the regime Cui/Wu/Xu study (fixed or heuristically-set σ) and not in ours.

Keeping HTSK as the default is still reasonable — it costs nothing and starts the
model in a sane regime — but it is not what makes FES work. See
`docs/02-fuzzy-systems-high-dim.md` §2 and `LOG.md` Findings 7 and 9.

**Step 3 — consequent (first-order TSK, mixture of linear experts).**

```
u_t = Σ_{r=1}^{R} f̄_r(v_t) · ( A_r v_t + b_r )          A_r ∈ ℝ^{d_out × d_in}, b_r ∈ ℝ^{d_out}
```

Vectorised without materialising per-token matrices: compute all `R` expert
outputs `(A_r v_t + b_r)` as one `einsum`, then contract with `f̄`. Cost per token
is `R·d_in·d_out` MACs.

**Step 4 — pooling.** Learned per-token log-weight `ω[t]` (the trainable
generalisation of SIF/IDF, A2), softmax-normalised over the sequence under the
attention mask:

```
a_i = softmax_i( ω[t_i] )            e = Σ_i a_i · u_{t_i}            ê = e / ‖e‖₂
```

`ω` is initialised from corpus token frequency using the POTION/SIF form
`ω = log(1e-3 / (1e-3 + p(t)))` (A2), so the model *starts* at a strong classical
baseline and learns from there.

### 2.2 Rule semantics — why this is a fuzzy system and not just an MLP

Each rule reads, literally:

> **Rule r:** IF `v` is near prototype `m_r` (within per-dimension tolerance `σ_r`)
> THEN this token contributes `A_r v + b_r` to the document embedding.

`m_r` is a point in token-feature space, so it can be *decoded*: find the
vocabulary items with the highest `f̄_r`. `b_r` is the rule's unconditional
contribution and `‖A_r‖` its sensitivity. This gives a per-rule readout that a
lookup table cannot provide (§7).

### 2.3 Variant FES-C: context-conditioned antecedents

Steps 1–4 above are still a bag-of-words model: `u_t` depends only on `t`, so the
whole thing collapses to an equivalent (if implicit) lookup table. It can only
win on **parameter efficiency**, never on expressivity.

To exceed the bag-of-words ceiling, condition the *antecedent* on document
context. Let `c = Σ_i a_i v_{t_i}` (mean token feature vector, same pooling
weights, one extra cheap pass) and fire rules on the concatenation:

```
Z_r([v_t ; c])        D = 2·d_in
```

Now the rule reads:

> IF token is near `m_r^tok` AND context is near `m_r^ctx` THEN contribute `A_r v_t + b_r`

This is genuinely more expressive than any static table — the same token routes
to different experts in different documents — while adding only `R·d_in·2`
parameters and one extra `D`-dim distance computation. It is a *fuzzy* gating
mechanism: cheap, second-order, order-insensitive, and interpretable. It is not
attention (no token-token interaction, still O(L) not O(L²)).

**Both variants get trained.** FES-S tests the compression claim; FES-C tests the
expressivity claim.

> ### REFUTED (E004/E006). FES-C is worse than FES-S in both routing regimes.
>
> | config | MTEB-14 | NanoBEIR | H_fire |
> |---|---|---|---|
> | A2 FES-S (no context) | 50.96 | 0.4166 | 0.208 |
> | A5c FES-C, hard routing | 43.34 | 0.2676 | 0.000 |
> | A5c FES-C, anchored soft | 49.19 | 0.3596 | 0.980 |
>
> Two distinct failure modes, and the mechanism is identified in each:
>
> 1. **Hard routing breaks query–document comparability.** Left alone, the
>    context-conditioned router saturates to one-hot (H_fire = 0.000). A query and
>    its answer document have very different context vectors `c`, so they route to
>    *different* experts — their embeddings are no longer produced by the same map
>    and cosine similarity between them stops meaning anything. Retrieval took the
>    largest hit of any task type (20.9 vs 35.8). Note this is specific to gating on
>    *context*: A4 also routes near-hard (H_fire = 0.010) and is the best rung.
> 2. **Soft routing gets neutralised on purpose.** With the fuzziness anchor
>    targeting H_fire = 0.5, the model instead settled at **0.980** and *accepted a
>    ~2.3 loss penalty to do so*. Near-uniform firing means `u_t = Ā v_t + b̄`, a
>    single linear map — i.e. the optimiser spent real objective value to make
>    itself context-free again, wasting the rule base on an averaged expert. It
>    lands below even the 2.00M A1 control.
>
> So conditioning the antecedent on a pooled context vector does not break the
> bag-of-words ceiling; it is actively harmful, and the model will pay to escape it.
> The likely reason the idea was wrong: a mean-pooled `c` is nearly identical for
> every token in a document, so it acts as a per-document *bias* on routing rather
> than as token-level disambiguation — all the cost of gating, none of the benefit.
> Any future attempt at this should make the context term token-specific (which is
> approaching attention, and leaves the cost tier that motivates the whole design).

## 3. Parameter budget

`|V| = 30522`, `d_out = 256` (matching potion-base-8M's output dim so retrieval
comparisons are dimension-fair).

| Component | Count | `d_in=64, R=32` |
|---|---|---|
| Token feature table `F` | `|V| · d_in` | 1,953,408 |
| Token pool weights `ω` | `|V|` | 30,522 |
| Antecedents `m, log σ` | `2 · R · D` | 4,096 (FES-S) / 8,192 (FES-C) |
| HTSK temperature `λ` | 1 | 1 |
| Consequents `A_r, b_r` | `R · d_out · (d_in + 1)` | 532,480 |
| **Total** | | **≈ 2.52 M** |

Comparison at equal output quality ambition:

| Model | Params | Output dim |
|---|---|---|
| potion-base-8M | 7.56 M | 256 |
| potion-base-4M | ~3.7 M | 128 |
| potion-base-2M | ~1.9 M | 64 |
| **FES-S (d_in=64, R=32)** | **2.52 M** | **256** |
| **FES-tiny (d_in=32, R=16)** | **1.32 M** | **256** |

The point: a plain static table at `d_out = 256` costs 7.8M parameters for the
table *alone*. FES gets a 256-d output for 2.5M because the `d_in → d_out`
expansion is shared across the vocabulary and only the narrow `d_in` part is
per-token. **`R` and `d_in` are the two knobs; the vocabulary table stops being
the whole model.**

Consequent cost grows as `R·d_in·d_out`, which caps `R`. A `consequent_rank = k`
option factorises `A_r = U_r V` with `V ∈ ℝ^{k×d_out}` shared, dropping the cost
to `R·d_in·k + k·d_out` — the lever for scaling `R` later (deliberately out of
scope for now per the brief).

## 4. Training

Following A5, because it is the only published from-scratch static-embedder recipe
and because an identical objective across all ablations is what makes the fuzzy
layer's contribution measurable (see `docs/01-small-embedding-models.md` §3).

- **Objective:** `MultipleNegativesRankingLoss` (in-batch-negative InfoNCE,
  scale 20 ⇒ temperature 0.05) wrapped in `MatryoshkaLoss` over
  `[32, 64, 128, 256]`, so the model is usable at 32-d and the comparison against
  potion-base-2M (64-d) is fair at 64-d.
- **Auxiliary loss — Uniform Regularisation** (B2): `ℓ_UR = Σ_r (mean_batch f̄_r − 1/R)²`,
  weight `γ`. Without it rules go unused and the effective parameter count drops
  silently. This is the fuzzy literature's name for the MoE load-balancing loss.
- **Two parameter groups — mandatory.** The sparse tables `F, ω` want LR ~0.2
  (A5); the dense antecedent/consequent parameters want ~1e-3. One LR for both
  diverges or crawls.
- **Antecedent init:** KMeans (`R` clusters) on the token feature vectors of the
  top-20k most frequent tokens after a short warm-up, `log σ` init to
  `log(h · per-dim std)` with `h = 1` (B2, and B1's finding that HTSK is
  insensitive to `h ≥ 0.5`).
- **Data:** a subset of A5's list — AllNLI triplets, GooAQ, MS MARCO, S2ORC
  title–abstract, SQuAD, Quora/StackExchange duplicates. Round-robin sampled.
- **Precision:** bf16 autocast, `batch_size` as large as 12 GB allows (the model
  is ~2.5M params, so batch size is bounded by activations and by the InfoNCE
  similarity matrix, not by weights → expect ≥ 2048).

## 5. Ablation ladder (this is the experiment, not the model)

Every rung shares data, steps, loss, and seed. Run in order.

| ID | Configuration | Question answered |
|---|---|---|
| **A0** | `R = 1`, order-0 consequent, mean pool | Reproduces a plain static embedding model. Sanity floor. |
| **A1** | `R = 1`, order-1 consequent, learned pool | Low-rank static table + SIF. Isolates the value of the *pooling* and the `d_in→d_out` factorisation, with **no fuzzy gating at all**. **This is the control FES must beat.** |
| **A2** | FES-S, `R = 32`, HTSK | Does a rule base beat one rule at matched parameters? The compression claim. |
| **A3** | A2 but **product t-norm** (no `1/D`) | Confirms the curse-of-dimensionality failure mode empirically (B1). Expected to collapse toward A1. |
| **A4** | A2 without UR loss | Measures rule collapse; report the rule-usage entropy. |
| **A5c** | **FES-C**, context-conditioned antecedents | Does breaking the bag-of-words ceiling help? The expressivity claim. |
| **A6** | Sweep `R ∈ {1,4,16,32,64}` at fixed total params | Where is the rule/table parameter trade-off optimum? |
| **A7** | LogTSK instead of HTSK | Alternative defuzzification (B1). |

Report for each: NanoBEIR nDCG@10, the MTEB subset average, param count, rule
usage entropy `H(mean f̄) / log R`, and CPU embed throughput.

## 6. Evaluation

See `docs/03-benchmarks.md`. Summary: a fixed 14-task MTEB subset covering all 7
task types, plus NanoBEIR, run through the **same `mteb` code path** for FES and
for every baseline (`potion-base-{2,8}M`, `all-MiniLM-L6-v2`,
`static-retrieval-mrl-en-v1`, and the labelled `EmbeddingGemma-300M` /
`LFM2.5-Encoder-230M` references). Self-measured baseline numbers, not
leaderboard-quoted ones, are what get compared.

## 7. Interpretability pass (the thing only a fuzzy model gets)

After training, for each rule `r`:

1. **Decode the prototype** — the 20 vocabulary tokens with highest `f̄_r`.
2. **Report `‖b_r‖` and `‖A_r‖_F`** — is this a high- or low-contribution rule?
3. **Rule usage histogram** over a corpus; entropy as a collapse metric.
4. For FES-C: which *contexts* activate the rule, holding the token fixed.

The hypothesis to falsify: rules specialise into linguistically recognisable
regions (subword continuations, function words, numerals, topical content words),
and low-contribution rules coincide with low-information tokens — i.e. the model
**rediscovers IDF/SIF weighting as an emergent rule structure** rather than
having it imposed.

> ### REFUTED (E008). It was falsified, cleanly.
>
> - Prototypes decode to no recognisable theme (`RULE 25`: *##mate, melbourne,
>   freezing, indigenous, pakistani, glen, protesters, clutching, broadcaster*).
> - `‖A_r‖_F` spans 15.142–15.617 — a **3.1% spread**. Every rule contributes
>   equally; there are no loud or quiet rules.
> - corr(rule-token rarity, rule contribution) = **+0.080**, i.e. zero.
>
> Mechanism: the model converges to a near-hard router (vocabulary-wide firing
> entropy **0.018**), so it is vector-quantising the token-feature space into ~32
> equal-mass cells — but that space is learned from scratch with no semantic
> anchoring, and nothing in a contrastive objective rewards linguistically
> meaningful cell boundaries. A learned VQ of a learned space is arbitrary by
> default.
>
> To pursue this: initialise `F` from a distilled static embedding (Model2Vec)
> instead of randomly, so KMeans prototypes land on pre-existing semantic
> clusters. Untested.

## 8. Risks, and what we do about each

| Risk | Mitigation |
|---|---|
| Softmax saturation kills antecedent learning | HTSK + learned temperature; A3 measures it directly |
| Rule collapse to 1 effective rule | UR loss; entropy metric reported every eval; A4 quantifies |
| Bag-of-words ceiling means A2 ≈ A1 no matter the rule count | Anticipated — that is *why* FES-C exists. A2 is a compression result; A5c is the expressivity result. |
| KMeans init on a table that is itself random at step 0 | Warm-up phase: train A1 first, init `m_r` from *its* converged table |
| 12 GB VRAM limits batch size, and InfoNCE quality depends on batch size | Model is 2.5M params; use `CachedMultipleNegativesRankingLoss` (GradCache) to decouple effective batch size from memory if needed |
| Comparing to leaderboard numbers instead of self-run ones | All baselines re-run locally through one code path |
| The literature review looks thorough but the claim is wrong | The gap statement is falsifiable and narrow: "no published work uses a TSK FIS as the embedding function itself". Recorded as a claim to keep challenging, not a settled fact. |
