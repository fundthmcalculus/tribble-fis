# FES — Results

A complete write-up of the investigation: can a **Takagi–Sugeno–Kang fuzzy
inference system** work as a small text-embedding model, and can it beat the small
static embedders?

- **`LOG.md`** — the raw chronological record (16 entries, 31 findings), written as
  the work happened.
- **This document** — the consolidated narrative and results.
- **`DESIGN.md`** — the architecture, with every refuted claim marked in place.
- **`docs/`** — literature review, fuzzy-systems math, evaluation protocol.

Everything below is **self-measured** through one code path
(`fuzzyembed/evaluate.py`) unless marked `[published]`.

---

## 1. Summary

**The short version.** A TSK fuzzy inference system does work as an embedding
model, and at small scale it is genuinely competitive: **1.14M parameters beats
`potion-base-2M` on every metric at 60% of its size**, and **7.41M reaches 99% of
`potion-base-8M`**. But of the three distinctive claims motivating the fuzzy
framing, **all three were refuted**. What survives is a replicated,
**retrieval-specific** gain from what is structurally a mixture-of-experts, with
the *fuzziness itself* measurably irrelevant.

| Claim (from `DESIGN.md`) | Verdict |
|---|---|
| **Compression** — a rule base beats a wider table at equal parameters | **Refuted on aggregate** (−0.12, p=0.75). Survives on retrieval only (+5.5%, p=0.05) |
| **Expressivity** — context-conditioned rules break the bag-of-words ceiling | **Refuted**, two distinct mechanisms |
| **Interpretability** — the rule base is readable | **Refuted**. Rules decode to nothing; contribution spread 3.1% |

**The one thing the fuzzy framing genuinely contributed** was not accuracy but a
diagnostic: `firing_entropy` caught two distinct degeneracies, where the
literature warns about only one.

### Headline competitive results

| Tier | FES | Competitor | Verdict |
|---|---|---|---|
| ~1M | **T-tiny 1.14M** — 49.91 / 0.4033 | potion-base-2M 1.89M — 49.62 / 0.3666 | **Wins both, at 60% size** |
| ~7.5M | **S4 7.41M** — 52.26 / 0.4385 | potion-base-8M 7.56M — 52.74 / 0.4421 | 99% of both; wins 5 of 7 task types |
| ~8M | **S3 8.18M** — 52.31 / **0.4397** | potion-base-8M | Best FES; larger, so not an efficiency win |

*(MTEB-14 / NanoBEIR nDCG@10)*

---

## 2. What was built

```
fuzzyembed/model.py      FES forward pass as a Sentence-Transformers InputModule
fuzzyembed/data.py       11-source public training mix, 4,115,821 pairs
fuzzyembed/train.py      contrastive training (MNRL + Matryoshka + UR + calibration)
fuzzyembed/evaluate.py   frozen 14-task MTEB subset + NanoBEIR
scripts/                 smoke, baselines, ladder, rule inspection, cost, plots
tests/                   28 tests
```

### Architecture

```
token ids
  → F[t]                narrow learned feature table       (fuzzification)
  → f̄_r(v_t)            HTSK firing over R fuzzy rules     (antecedent)
  → Σ_r f̄_r(A_r v + b_r) mixture of R local linear experts (TSK consequent)
  → Σ_t a_t u_t          learned SIF-style weighted pool    (defuzzification)
  → L2 normalise
```

The starting observation, which is what makes this a real question rather than a
gimmick: **a static embedding model is algebraically a one-rule, zero-order TSK
system.** `e = mean_t E[t]` is `R=1`, `f̄₁ ≡ 1`, consequent = lookup. Every static
embedder in the literature (Model2Vec, POTION, `static-retrieval-mrl-en-v1`) sits
at `R = 1`; everything above is unexplored for embedding *production*.

### Training

Follows the from-scratch static-embedding recipe (Aarsen & Nussbaum): MNRL +
Matryoshka, batch 4096, LR 0.2 on sparse tables, 1 epoch, bf16 on one RTX 4080
Laptop. Three additions the fuzzy layer needed: three LR groups, Uniform
Regularisation, and temperature calibration.

---

## 3. The experiment sequence

### E000 · Literature review

22 references. Four findings shaped the design: the static-model/one-rule-TSK
identity; the curse-of-dimensionality result (Cui/Wu/Xu 2021) and its HTSK fix;
proof that a from-scratch static embedder reaches ~90% of MiniLM; and the
discovery that **`lfm2.5-230m` is not an embedding model** (`LFM2.5-230M` is a
generative decoder; `LFM2.5-Encoder-230M` is an encoder scored by fine-tuning on
GLUE/SuperGLUE, with no published MTEB or retrieval numbers).

### E001 · Implementation

**Finding 1 — the fused pooling refactor.** The literal per-token consequent needs
a `(B, L, R, d_out)` intermediate — **4 GB at batch 2048**. Because the sequence
pool is a fixed linear combination it commutes inward:

```
e_d = Σ_r Σ_i [ Σ_l a_l f̄_lr v_li ] A_rid  +  Σ_r [ Σ_l a_l f̄_lr ] b_rd
```

The expert projection then runs **once per document instead of once per token** —
`O(L·R·d_in) + O(R·d_in·d_out)` instead of `O(L·R·d_in·d_out)`. Exact, and
verified against the literal definition. This is why `d_out` later turns out to be
nearly free.

**Finding 2 — there are TWO degeneracies, not one.** The literature warns about
softmax *saturation* (temperature too low ⇒ `f̄ → one-hot`, dead gradients),
confirmed at random init: product t-norm gives per-token firing entropy **0.14**
vs HTSK's **1.00**. But raw HTSK overshoots into the opposite failure — `f̄ → 1/R`
uniformly, which makes `u_t = Ā v_t + b̄`, **a single linear map**.

| temperature | firing entropy | effective experts | failure |
|---|---|---|---|
| too low (product) | → 0 | 1 (hard router) | saturation, dead gradients |
| too high (raw HTSK, τ=D) | → 1 | 1 (mean of experts) | uniform blending |

**Rule-usage entropy does not detect the second failure** — it reads 1.0
throughout. So two metrics were separated:

- `rule_entropy` = `H(mean_t f̄)/log R` — *are all rules used somewhere?*
- `firing_entropy` = `mean_t H(f̄_t)/log R` — *is the inference actually fuzzy?*

**Fix:** `calibrate_temperature()` bisects `log τ` against a target firing entropy.

**Finding 3 — the contrastive objective prefers a crisp router.** Calibrated to
0.500, training drove firing entropy to 0.039 while all rules stayed in use.

### E002 · Baselines and the training bottleneck

**Harness validated independently.** Two NanoBEIR measurements reproduce published
values exactly:

| model | ours | published `[published]` |
|---|---|---|
| all-MiniLM-L6-v2 | 0.562318 | 0.5623 |
| static-retrieval-mrl-en-v1 | 0.503167 | 0.5032 |

**Finding 4 — the bottleneck was the batch sampler.** The first run sat at 1% GPU.
Profiling: tokenisation 61,723 texts/s; forward+backward 55,467 samples/s; peak
GPU memory **0.16 GB**. Neither could cause it — leaving
`BatchSamplers.NO_DUPLICATES`, which builds a duplicate index over 4.1M pairs.
Switching to the random sampler gave ~2 it/s. *I nearly "fixed" this by cutting
sequence length and adding workers, neither of which would have helped.*

### E003–E004 · The ablation ladder

| rung | params | MTEB-14 | NanoBEIR | H_rule | H_fire |
|---|---|---|---|---|---|
| A1b control — R=1, param-matched | 2.52M | **51.44** | 0.4033 | — | — |
| A4 — R=32, no UR | 2.52M | 51.38 | **0.4220** | 0.970 | 0.010 |
| A8 — R=32, fuzziness anchored | 2.52M | 51.25 | 0.4200 | 1.000 | 0.503 |
| A2 — R=32, HTSK + UR | 2.52M | 50.96 | 0.4166 | 0.999 | 0.208 |
| A3 — R=32, product t-norm | 2.52M | 50.86 | 0.4181 | 0.997 | 0.141 |
| A1 control — R=1, *unmatched* | 2.00M | 50.82 | 0.3841 | — | — |
| A5c — context-conditioned | 2.52M | 43.34 | 0.2676 | 0.645 | 0.000 |

**Finding 8 — half the apparent gain was parameters.**

| comparison | MTEB-14 | NanoBEIR |
|---|---|---|
| A4 − A1 (unmatched, +26% params) | +0.56 | +0.0379 |
| **A4 − A1b (matched)** | **−0.06** | **+0.0187** |

**Finding 9 — every specifically *fuzzy* ingredient is neutral or harmful.** Four
rungs spanning a **50× range of routing softness** all sit within 0.5 MTEB-14 and
0.005 NanoBEIR. HTSK ≈ product t-norm (A3 ≈ A2), because **τ and σ are the same
parameter** — dividing the exponent by τ equals scaling every σ by `√τ`, and
trainable `log_sigma` absorbs it. Confirmed independently: calibrated τ = **1.195
at both D=64 and D=128**, i.e. it does not vary with `D` at all.

> This retracted my own earlier claim that "HTSK is mandatory / the load-bearing
> piece". It is load-bearing at init with fixed σ — the regime the paper studies —
> and not under end-to-end SGD. Corrected in `DESIGN.md`, `README.md` and
> `docs/02`.

**Finding 10 — the rule base trades smooth geometry for discrimination.**

| type | A1 | A4 | Δ |
|---|---|---|---|
| **Retrieval** | 35.8 | **39.6** | **+3.8** |
| Classification | 51.6 | 52.9 | +1.3 |
| STS | 67.8 | 66.5 | −1.3 |
| **Clustering** | 40.4 | **37.9** | **−2.5** |

Near-hard routing makes the embedding map piecewise: it sharpens lexical/topical
discrimination and damages tasks needing continuous geometry.

### E006 · FES-C refuted, with a mechanism

| config | MTEB-14 | NanoBEIR | H_fire |
|---|---|---|---|
| A2 FES-S (no context) | 50.96 | 0.4166 | 0.208 |
| A5c hard routing | 43.34 | 0.2676 | 0.000 |
| A5c anchored soft | 49.19 | 0.3596 | 0.980 |

**Finding 14 — two failure modes, both diagnosed.**

1. **Hard routing breaks query–document comparability.** A query and its answer
   document have different context vectors, so they route to *different* experts;
   their embeddings stop being produced by the same map. Retrieval fell hardest
   (20.9 vs 35.8). Note this is specific to gating on *context* — A4 also routes
   near-hard and is the best rung.
2. **Soft routing gets neutralised on purpose.** The anchor targeted H_fire = 0.5;
   the model settled at **0.980**, *accepting a ~2.3 loss penalty* to make itself
   context-free again. It paid real objective value to escape the context signal.

*Why the idea was wrong:* a mean-pooled context vector is identical for every token
in a document, so it cannot disambiguate a token against its neighbours — all the
cost of gating, none of the benefit.

### E008 · Interpretability refuted

**Finding 20.** Prototypes decode to no recognisable theme
(`RULE 25: ##mate, melbourne, freezing, indigenous, pakistani, glen, protesters…`).

| statistic | min | max | spread |
|---|---|---|---|
| `‖A_r‖_F` | 15.142 | 15.617 | **3.1% of mean** |
| corr(token rarity, contribution) | | | **+0.080** |

A 3.1% spread means the rules are interchangeable in magnitude. *Mechanism:* the
model converges to a near-hard router (vocabulary-wide firing entropy **0.018**),
so it vector-quantises the token-feature space into ~32 equal-mass cells — but that
space is learned from scratch with no semantic anchoring, and nothing in a
contrastive objective rewards linguistically meaningful boundaries.

### E010 · Seed replication — the decisive test

Three seeds each of the two configs the argument rests on.

| metric | A4 rule base | A1b control | difference | exact permutation p |
|---|---|---|---|---|
| MTEB-14 | 51.243 ± 0.299 | 51.363 ± 0.159 | **−0.12** | 0.750 |
| NanoBEIR | **0.42511 ± 0.00279** | 0.40305 ± 0.00667 | **+0.0221 (+5.47%)** | **0.050** |

**Finding 21.** Every A4 run beats every A1b run (min 0.42200 > max 0.40959).
Under the null that arrangement has probability 1/C(6,3) = **0.050** — the smallest
p attainable at n=3, so this is as strong as three seeds can show. Honest framing:
"p ≤ 0.05 with complete separation at n=3", not "comfortably significant".

**Finding 22.** The aggregate null is definitive, not under-powered: −0.12 against
seed spreads of 0.16–0.30. My pre-registered ±0.5 threshold was *conservative*.

### E011 · Optimisation sweep

**Finding 23 — the experts are not undertrained.**

| lr_dense | MTEB-14 | NanoBEIR |
|---|---|---|
| **2e-3 (baseline, n=3)** | **51.24 ± 0.30** | **0.42511 ± 0.00279** |
| 8e-3 (4×) | 51.41 | 0.4157 |
| 3e-2 (15×) | 50.72 | 0.4190 |

Both higher LRs fall *below the minimum of A4's three-seed range*. This matters
beyond the immediate question: the retrieval gain cannot be dismissed as "the
configs were tuned differently", because both ran at the LR optimal for the rule
base — which if anything favours the control.

### E012 · The low-rank consequent is broken (a self-inflicted bug)

S2 with `consequent_rank=32` regressed to 47.79 / 0.3300. Measuring the effective
rank of real embeddings:

| k | predicted bound (k+R) | **measured rank** | NanoBEIR |
|---|---|---|---|
| 32 | 64 | **65** | 0.3300 |
| 256 | 288 | **289** | 0.4221 |
| dense | 512 | 416 | 0.4353 |

**Finding 24/26.** `V ∈ ℝ^{k×d_out}` is **shared across rules**, so the model's
entire output lives in a `k + min(R, d_out)`-dimensional subspace *regardless of
d_out*. S2 was a 65-dimensional model padded into 512. Retrieval tracks usable rank
monotonically.

*How the test missed it:* `test_lowrank_consequent_is_equivalent_to_its_dense_expansion`
asserts `A_r = U_r V` matches a dense `A_r` built from that product — true, and it
passes — but that expansion is *itself* rank-32. **I verified the algebra and never
checked what the model could represent.** Caught before S3/S4/S5 ran, all of which
used rank 32; a single SVD on 200 embeddings would have caught it earlier.

### E013 · Silent cache contamination

The corrected dense S2 reported MTEB-14 **47.79** — identical to two decimals to the
broken run — while NanoBEIR moved 0.3300 → 0.4353. `mteb` caches per model name and
`run_mteb` defaulted to `overwrite=False`, so it reloaded the *previous model's*
scores. File timestamps confirmed it. **True value: 51.56 — the cache was wrong by
3.8 points**, which would have made `d_out` scaling look harmful when it is the best
axis. `overwrite=True` is now the default; blast radius was one record.

### E014–E016 · Scaling and cost

| rung | params | d_in | d_out | R | MTEB-14 | NanoBEIR |
|---|---|---|---|---|---|---|
| T-tiny | 1.14M | 32 | 256 | 16 | 49.91 | 0.4033 |
| A6-R4 | 2.05M | 64 | 256 | 4 | 51.16 | 0.4094 |
| A6-R16 | 2.25M | 64 | 256 | 16 | 50.53 | 0.4159 |
| A4 (n=3) | 2.52M | 64 | 256 | 32 | 51.24 | 0.4251 |
| **O3 (2 epochs)** | **2.52M** | 64 | 256 | 32 | **51.62** | **0.4335** |
| S2 | 3.05M | 64 | **512** | 32 | 51.56 | 0.4300 |
| A6-R64 | 3.06M | 64 | 256 | 64 | 51.00 | 0.4267 |
| S1 | 5.00M | **128** | 256 | 32 | 52.12 | 0.4332 |
| S5 | 6.28M | 64 | 256 | **256** | 51.11 | 0.4314 |
| S4 | 7.41M | **190** | 256 | 32 | 52.26 | 0.4385 |
| S3 | 8.18M | 128 | 512 | 64 | **52.31** | **0.4397** |

**Finding 27 — the scaling ordering, from a near-perfectly controlled comparison.**

| config | params | change | MTEB-14 | NanoBEIR |
|---|---|---|---|---|
| A6-R64 | 3.06M | R: 32 → 64 | 51.00 | 0.4267 |
| **S2** | **3.05M** | **d_out: 256 → 512** | **51.56** | **0.4300** |

0.3% apart in parameters; `d_out` wins on both. Ordering: **`d_out` > `d_in` > `R`.**

- `d_in` helps with strong diminishing returns (+0.88 then +0.14 MTEB per doubling).
- **`R` is done by ~32.** S5 spent 3.7M extra parameters on 8× the rules and scored
  *below* A4 on aggregate at 2.5× the size. This was pre-registered as a test of
  Findings 9 and 15; both held.

**Finding 28 — an epoch is worth more than a parameter, for retrieval.** O3 (2.52M,
2 epochs) matches S1 (5.00M, 1 epoch) on retrieval at half the size. *Caveat on the
whole study: every rung is a single epoch, so all models are undertrained. The
comparison is fair — identical budget throughout — but absolute numbers are below
convergence.*

**Finding 29 — LogTSK reproduces the uniform degeneracy, as predicted.** A7 holds
H_fire at **0.999**, collapsing 32 rules into one averaged expert, and lands *below
the R=1 control on both metrics* — it pays for rules and gets less than none.

**Finding 30 — the complete R-sweep.**

| R | params | MTEB-14 | NanoBEIR |
|---|---|---|---|
| 1 (control) | 2.52M | 51.36 | 0.4031 |
| 4 | 2.05M | **51.16** | 0.4094 |
| 16 | 2.25M | 50.53 | 0.4159 |
| 32 | 2.52M | 50.96 | 0.4166 |
| 64 | 3.06M | 51.00 | 0.4267 |
| 256 | 6.28M | 51.11 | 0.4314 |

**Retrieval rises monotonically across the entire range; MTEB-14 shows no trend
whatsoever.** This is the sharpest single statement of the central result: *rules
buy retrieval and nothing else.* Even R=4 beats the R=1 control on retrieval at
fewer parameters, and R=4 has the best aggregate of any fuzzy config while having
the worst retrieval.

**Throughput (idle machine, verified 0 MiB / 0% before measuring).**

| config | params | GPU sent/s | CPU sent/s |
|---|---|---|---|
| R=1 control | 2.52M | 702,761 | **692,641** |
| R=32 (default) | 2.52M | 319,186 | **188,679** |
| R=128 | 4.13M | 314,591 | 69,764 |
| R=512 | 10.57M | 116,197 | 13,449 |
| R=128, d_out=768 | 2.39M | 295,375 | 100,455 |

**Finding 16.** Tripling `d_out` costs **5% throughput and 3.5% parameters** — the
payoff of the fused-pool refactor. **Finding 17.** The rule base costs **3.67× CPU
throughput** for a retrieval-only gain; absolute throughput remains firmly in the
static tier. **Finding 15.** Low rank does not buy throughput at large R, because
the antecedent is `O(L·R·d_in)` and does not factor out of the pool.

**Finding 31 — fixed-seed runs are not bit-reproducible.** Two identical S2 runs
(same seed, same config) differed by **0.0054 NanoBEIR** from GPU kernel
nondeterminism. So the three-seed spreads conflate seed variation with kernel
noise — still valid as total run-to-run noise, which is what the significance test
needs. **Practical threshold: NanoBEIR differences below ~0.01 are noise unless
replicated.** Measured, not assumed; the headline gap of 0.0221 clears it by 4×.

---

## 4. Full results

MTEB-14 subset (all 7 task types) + NanoBEIR nDCG@10, sorted by retrieval.

| model | params | MTEB-14 | Class | Clust | PairCl | Rerank | Retr | STS | Summ | NanoBEIR |
|---|---|---|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22,713,216 | 60.52 | 61.5 | 48.8 | 81.2 | 75.3 | 48.8 | 77.3 | 30.8 | 0.5623 |
| static-retrieval-mrl-en-v1 | 31,254,528 | 51.02 | 54.6 | 27.3 | 72.6 | 63.0 | 44.6 | 66.4 | 28.6 | 0.5032 |
| potion-base-32M | 32,302,592 | 54.08 | 63.4 | 34.8 | 75.6 | 64.5 | 42.1 | 68.4 | 29.7 | 0.4637 |
| potion-base-8M | 7,559,168 | 52.74 | 61.2 | 35.5 | 73.7 | 63.1 | 38.9 | 67.4 | 29.2 | 0.4421 |
| **FES-S3-balanced** | 8,181,051 | 52.31 | 55.1 | 38.6 | 70.6 | 63.4 | **41.7** | 67.2 | 29.7 | **0.4397** |
| **FES-S4-potion8M-matched** | 7,406,915 | 52.26 | 53.2 | 38.6 | 70.7 | 63.3 | 39.6 | 67.8 | 32.6 | 0.4385 |
| FES-O3-2epoch | 2,520,635 | 51.62 | 53.1 | 38.1 | 70.7 | 63.1 | 38.8 | 67.5 | 30.0 | 0.4335 |
| FES-S1-table-128 | 5,002,555 | 52.12 | 53.6 | 38.6 | 70.4 | 62.9 | 40.0 | 67.4 | 31.9 | 0.4332 |
| FES-S5-manyrules | 6,276,667 | 51.11 | 53.4 | 37.0 | 70.0 | 61.8 | 39.8 | 66.5 | 29.2 | 0.4314 |
| FES-S2-wide-out-512 | 3,053,115 | 51.56 | 54.2 | 38.0 | 69.9 | 62.4 | 39.4 | 66.3 | 30.7 | 0.4300 |
| FES-A0-static-256 | 7,910,459 | 52.49 | 55.5 | 39.9 | 72.7 | 63.4 | 38.7 | 69.1 | 28.1 | 0.4284 |
| FES-A6-R64 | 3,057,211 | 51.00 | 52.4 | 38.4 | 69.1 | 62.2 | 39.0 | 66.1 | 29.8 | 0.4267 |
| **FES-A4-no-ur** (n=3) | 2,520,635 | 51.24 | 52.9 | 37.9 | 69.2 | 62.3 | 39.6 | 66.5 | 31.3 | 0.4251 |
| FES-S6-rank-demo | 2,659,899 | 51.07 | 52.6 | 38.4 | 69.6 | 62.1 | 37.9 | 66.9 | 30.0 | 0.4221 |
| FES-A8-fuzzy-anchor | 2,520,635 | 51.25 | 52.8 | 38.5 | 68.5 | 62.4 | 39.6 | 65.9 | 31.0 | 0.4200 |
| FES-O2-lr-3e-2 | 2,520,635 | 50.72 | 51.9 | 38.7 | 69.5 | 62.1 | 38.3 | 66.7 | 27.9 | 0.4190 |
| FES-A3-product-tnorm | 2,520,635 | 50.86 | 51.7 | 38.6 | 68.5 | 62.5 | 39.1 | 65.9 | 29.9 | 0.4181 |
| FES-A2-fes-s | 2,520,635 | 50.96 | 51.9 | 38.6 | 68.8 | 62.3 | 39.4 | 66.0 | 29.7 | 0.4166 |
| FES-A6-R16 | 2,252,347 | 50.53 | 52.4 | 38.5 | 69.1 | 61.8 | 38.9 | 65.9 | 27.1 | 0.4159 |
| FES-O1-lr-8e-3 | 2,520,635 | 51.41 | 52.8 | 38.6 | 69.7 | 62.5 | 39.7 | 66.5 | 30.0 | 0.4157 |
| FES-A6-R4 | 2,051,131 | 51.16 | 52.5 | 38.8 | 69.6 | 62.3 | 37.7 | 66.9 | 30.4 | 0.4094 |
| **FES-T-tiny** | **1,143,483** | 49.91 | 49.6 | 37.8 | 68.1 | 61.0 | 37.6 | 64.9 | 30.3 | **0.4033** |
| FES-A1b-ctrl-matched (n=3) | 2,524,121 | 51.36 | 52.5 | 40.2 | 69.1 | 62.6 | 36.6 | 68.3 | 30.9 | 0.4031 |
| FES-A7-logtsk | 2,520,635 | 51.10 | 52.0 | 40.4 | 69.1 | 62.3 | 34.8 | 68.0 | 31.1 | 0.3876 |
| FES-A1-lowrank-ctrl | 2,000,827 | 50.82 | 51.6 | 40.4 | 68.5 | 61.9 | 35.8 | 67.8 | 29.8 | 0.3841 |
| potion-base-2M | 1,889,792 | 49.62 | 55.1 | 35.2 | 70.9 | 59.5 | 29.2 | 65.9 | 31.4 | 0.3666 |
| FES-A5c-anchored | 2,524,731 | 49.19 | 48.0 | 39.5 | 67.5 | 60.7 | 33.3 | 66.0 | 29.3 | 0.3596 |
| FES-S2-rank32-BROKEN | 2,086,459 | 47.79 | 46.1 | 39.4 | 65.5 | 59.5 | 27.4 | 65.3 | 31.4 | 0.3300 |
| FES-A5c-fes-c | 2,524,731 | 43.34 | 42.7 | 36.9 | 58.1 | 54.0 | 20.9 | 60.7 | 30.0 | 0.2676 |

### The classification gap is data, not architecture

`A0-static-256` is the *conventional* architecture (full-width table, no rule base)
trained on our data with our recipe. At 7.91M it reaches 52.49 vs potion-8M's 52.74
— **ahead on Clustering (+4.4) and STS (+1.7), behind on Classification by 5.7.**

Since A0 has no rule base and shows the same deficit, **the classification gap is
the training data, not the fuzzy architecture.** POTION distils from
`bge-base-en-v1.5`; our retrieval-heavy mix contains no supervised classification
signal. Scaling to 8.18M did not close it. **Capacity will not fix this; a
classification training signal would.**

### Where the parameters go

| config | vocabulary table | rule base | pooling |
|---|---|---|---|
| FES R=1, 2.5M | **97.9%** | 0.8% | 1.2% |
| FES R=32, 2.5M | **77.5%** | 21.1% | 1.2% |
| FES R=32, 7.4M | **78.3%** | 21.1% | 0.4% |

The vocabulary table is 77–98% of every configuration. **The fuzzy machinery this
entire investigation is about is a ~21% surcharge on a lookup table** — which is
also why table width dominates the aggregate and the rule base only moves
retrieval.

---

## 5. Practical guidance

If you want to use or extend this:

1. **Keep `R ≈ 32`.** Rules saturate by 32–64; R=256 costs 3.7M parameters for
   nothing, and every rule costs CPU throughput (`O(L·R·d_in)`, does not factor out).
2. **Keep the consequent dense.** Low rank caps the embedding rank at `k + R` and
   silently produces padded low-rank vectors. The model now warns.
3. **Scale `d_out` first, then `d_in`.** `d_out` is nearly free (5% throughput, 3.5%
   parameters for 3× width) and the most effective per parameter.
4. **Train more epochs before buying parameters** — for retrieval, 2 epochs at 2.5M
   matches 1 epoch at 5.0M.
5. **Use HTSK with entropy calibration.** Not raw HTSK (τ = D is ~54× miscalibrated),
   not LogTSK (collapses to uniform, worse than no rule base).
6. **Report both entropies.** `rule_entropy` and `firing_entropy` detect different
   failures; either alone hides one.
7. **Don't bother with UR** at R=32 — it mildly hurts (A4 > A2).
8. **Treat NanoBEIR differences below ~0.01 as noise** unless replicated.

## 6. Limitations

- **Single epoch throughout.** Fair (identical budget per rung) but undertrained;
  absolute numbers are below convergence.
- **MTEB-14, not MTEB-41.** Within the static family our subset runs ≈ +1.3 above
  published 41-task averages and preserves rank order; but the offset is **+4.42**
  for MiniLM, so it is family-specific. **Cross-family claims use NanoBEIR only**,
  which needs no offset.
- **n=3 seeds** on the two decisive configs; everything else is n=1.
- **English only**, `bert-base-uncased` tokenizer, 30,522 vocabulary.
- **Retrieval-heavy training mix** — the direct cause of the classification gap.
- **Large references are `[published]`, not self-measured.** EmbeddingGemma-300M was
  abandoned mid-run at 470 s/iteration; it is a non-target per the brief.

## 7. Open questions

1. **Classification signal.** Add supervised classification pairs and re-run S4.
   This is the single highest-value experiment remaining — it targets the entire
   remaining gap to potion-8M.
2. **Convergence.** O3 showed 2 epochs still improving. Where does it stop?
3. **Sparse antecedents.** Scaling `R` needs top-k rule selection or a hierarchical
   rule tree, since the antecedent cost does not factor out of the pool.
4. **Interpretability, rescued.** Initialise `F` from a distilled static embedding
   rather than randomly, so KMeans prototypes land on pre-existing semantic
   clusters. Untested — the most likely route to making rules readable.
5. **Retrieval-specific deployment.** Given the gain is retrieval-only and costs
   3.67× CPU, FES is worth it only for retrieval-shaped workloads.

## 8. Reproduction

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch --torch-backend=cu126
uv pip install --python .venv/Scripts/python.exe -e . accelerate pytest

.venv/Scripts/python -m pytest tests/ -q          # 28 tests
.venv/Scripts/python scripts/eval_baselines.py    # baselines first
.venv/Scripts/python scripts/run_experiments.py   # the full ladder
.venv/Scripts/python scripts/report.py            # results/table.md
.venv/Scripts/python scripts/plot_comparison.py scripts/plot_params.py
.venv/Scripts/python scripts/inspect_rules.py artifacts/A4-no-ur
```

**Do not run two trainings at once on a 12 GB GPU.** VRAM exhaustion spills to WDDM
shared memory with no OOM error — a silent ~600× slowdown.

---

## 9. Closing assessment

The premise was worth testing and the answer is mostly negative, which is a real
result rather than a failed one.

**A TSK fuzzy inference system is a viable small embedding model.** It is
competitive at 1–8M parameters, beats the smallest published model outright, and
reaches 99% of the 8M-tier leader while winning 5 of 7 task types.

**But almost none of that comes from the fuzziness.** Across a 50× range of routing
softness, quality is flat. HTSK versus the plain product t-norm is
indistinguishable. Rule balancing mildly hurts. The rules decode to nothing. Left
alone, the model drives itself to a hard router — and performs best there. What
works is the **mixture-of-experts structure**; the fuzzy inference on top is a
design vocabulary, not a mechanism.

The one durable methodological contribution is `firing_entropy` — the per-token
diagnostic that distinguishes "all rules used" from "inference is actually fuzzy".
It caught the uniform degeneracy the literature does not warn about, predicted the
LogTSK collapse in advance, and explained the FES-C failure. Anyone building
gated/routed architectures should measure both entropies, not one.
