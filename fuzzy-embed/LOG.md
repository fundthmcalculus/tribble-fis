# FES Experiment Log

Append-only, newest entries at the bottom. Every entry: date, what was planned,
what was run, what came out, what it means, what's next. Numbers in this file are
**self-measured** unless marked `[quoted]`.

---

## 2026-07-31 — E000 · Project opened, literature review

**Plan.** Deep-dive local small embedding models; find a defensible way to build
an embedding model out of a fuzzy inference system; pick a public dataset and
standard benchmarks; target the small static embedders, not the 300M-class models.

**Done.**
- Reviewed 22 references → `docs/00-literature-index.md`, with detailed notes in
  `docs/01-small-embedding-models.md`, `docs/02-fuzzy-systems-high-dim.md`,
  `docs/03-benchmarks.md`.
- Environment: dedicated `fuzzy-embed/.venv` (py3.12), torch 2.13.0+cu126 on an
  RTX 4080 Laptop (12 GB), sentence-transformers 5.6.1, mteb 2.18.11,
  datasets 5.0.1, transformers 5.14.1, model2vec. CUDA verified.

**Findings — the four that shaped the design.**

1. **A static embedding model is algebraically a one-rule, zero-order TSK fuzzy
   system.** `e = mean_t E[t]` is `R=1`, `f̄₁ ≡ 1`, consequent = lookup. So the
   whole `R > 1` region is unexplored for embedding *production*. Fuzzy methods
   have been applied to text classification and to *evaluating* embeddings
   (CogniFNN, B7; multi-task TSK sentiment, B9) but not as the embedding function.
   This is the project's actual claim, and it is narrow enough to be falsifiable.

2. **The curse of dimensionality is the whole ballgame, and it has a known fix.**
   Cui/Wu/Xu 2021 (B1) prove TSK defuzzification *is* a softmax over
   `Z_r = −Σ_d (x_d−m_rd)²/(2σ_rd²)`. `|Z_r| = O(D)`, so at `D = 64` the softmax
   one-hots, one rule wins, and antecedent gradients die. **HTSK** — divide the
   exponent by `D`, i.e. use the geometric-mean t-norm — fixes it. Without this
   the model provably degenerates to `R=1` and the project is dead on arrival.
   This is now the single most-cited reference in the design.

3. **A from-scratch static embedder can reach ~90% of MiniLM.** Aarsen &
   Nussbaum (A5): `EmbeddingBag` + MNRL + Matryoshka, bs 2048, **lr 0.2**, 1
   epoch, 17.8 h on one RTX 3090 → NanoBEIR 0.5032 vs MiniLM's 0.5623 `[quoted]`.
   So no distillation teacher is needed — which matters, because an identical
   objective across fuzzy and non-fuzzy ablations is the only way to attribute a
   gain to the fuzzy layer. Recipe adopted wholesale.

4. **`lfm2.5-230m` is not an embedding model.** `LFM2.5-230M` is a generative
   decoder; `LFM2.5-Encoder-230M` is a bidirectional encoder scored by
   *fine-tuning* on GLUE/SuperGLUE (17-task mean 79.29 `[quoted]`). No MTEB or
   retrieval numbers published for either. Recorded in `docs/03-benchmarks.md` §4:
   we will run the encoder mean-pooled and label it as such, but it is not a fair
   target, and `EmbeddingGemma-300M` (MTEB eng v2 = 69.67 `[quoted]`) is the
   honest large-model reference. Both are ~90–120× our parameter budget and out
   of scope as targets per the brief.

**Targets fixed** (all `[quoted]`, to be re-measured locally):
`potion-base-8M` 7.56M params → MTEB 51.32; `potion-base-2M` ~1.9M → ~45–48;
`static-retrieval-mrl-en-v1` → NanoBEIR 0.5032; `all-MiniLM-L6-v2` 22.7M → ~56.1.

**Design frozen** in `DESIGN.md`: token feature table `F ∈ ℝ^{|V|×d_in}` →
HTSK antecedent over `R` rules with learned temperature → first-order TSK
consequent (mixture of `R` linear experts, `d_in → d_out`) → learned SIF-style
pooling → L2 norm. `d_in=64, R=32, d_out=256` ⇒ **2.52M params** for a 256-d
output, vs 7.8M for the lookup table alone in a plain static model. Parameter
efficiency comes from the vocabulary table no longer being the entire model.

**Key risk acknowledged up front:** FES-S is still bag-of-words, so it can only
win on *compression*, never expressivity — an implicit lookup table matches it.
Hence variant **FES-C**, which conditions rule firing on a document context
vector `[v_t ; c]`, making the same token route to different experts in different
documents. A2 tests compression; A5c tests expressivity. Stating this before
running anything so a null result on A2-vs-A1 isn't rationalised later.

**Next.** Implement the model + ablation ladder A0–A7; build the training data
mix; measure all baselines locally first so there is a fixed yardstick before any
FES number exists.

---

## 2026-07-31 — E001 · Implementation + pipeline smoke test

**Plan.** Implement `FuzzyEmbedding` as a Sentence-Transformers `InputModule`,
wire up contrastive training, and smoke-test the whole path end to end on a tiny
data slice before spending GPU hours.

**Built.**
- `fuzzyembed/model.py` — the FES forward pass, plus `parameter_counts`,
  `param_groups`, KMeans/SIF init, and the diagnostic metrics.
- `fuzzyembed/data.py` — the 11-source public mix, ~4.77M pairs, all verified
  loadable today (row counts recorded in the file).
- `fuzzyembed/train.py`, `fuzzyembed/evaluate.py`, `scripts/{smoke,eval_baselines,run_experiments,inspect_rules}.py`
- `tests/test_model.py` — 19 tests, all passing.

**Finding 1 — the fused pooling refactor (an efficiency result, not just an
optimisation).** The literal definition computes `u_t = Σ_r f̄_r(v_t)(A_r v_t + b_r)`
per token, needing a `(B, L, R, d_out)` intermediate — **4 GB at B=2048**. But the
sequence pool is a *fixed linear* combination, so it commutes inward:

```
e_d = Σ_r Σ_i [ Σ_l a_l f̄_lr v_li ] A_rid  +  Σ_r [ Σ_l a_l f̄_lr ] b_rd
```

Define the per-rule pooled feature `G_ri` and per-rule mass `h_r` and the expert
projection runs **once per document instead of once per token**. Cost drops from
`O(L·R·d_in·d_out)` to `O(L·R·d_in) + O(R·d_in·d_out)`. Exact, not approximate —
`test_fused_pool_matches_naive` checks it against the literal definition for both
variants and both consequent orders. Consequence: FES's *marginal* cost per token
is only the gating term, so it stays in the static-embedding cost tier no matter
how wide `d_out` is. Same expansion also removes the `(B,L,R,D)` distance tensor
(~1 GB) via the standard three-matmul quadratic expansion.

**Finding 2 — there are TWO degeneracies, not one, and the literature only warns
about one of them.** This is the most important result so far.

Cui/Wu/Xu warn about softmax **saturation** (temperature too low ⇒ `f̄ → one-hot`).
Confirmed at random init, `d_in=256, R=16`: product t-norm gives per-token firing
entropy **0.14** (~1.5 effective rules of 16) vs HTSK's **1.00**.

But raw HTSK then lands in the *opposite* degeneracy. In the smoke run, `τ = D`
gave firing entropy **0.9987** — essentially uniform. And uniform is equally
fatal, because if `f̄ ≡ 1/R` then

```
u_t = (1/R) Σ_r (A_r v_t + b_r) = Ā v_t + b̄
```

— a **single linear map**. The rule base has collapsed again. Both ends of the
temperature axis give one effective expert:

| temperature | firing entropy | effective experts | failure |
|---|---|---|---|
| too low (product t-norm) | → 0 | 1 (hard routing) | saturation, dead gradients |
| too high (raw HTSK, τ=D) | → 1 | 1 (mean of experts) | uniform blending |

Critically, **rule-usage entropy does not detect the second failure** — it reads a
perfect 1.0 throughout. So I split the diagnostics into two metrics that were
being conflated:
- `rule_entropy` = `H(mean_t f̄)/log R` — "are all rules used somewhere?" (what UR targets)
- `firing_entropy` = `mean_t H(f̄_t)/log R` — "is the inference actually fuzzy?"

**Fix:** `calibrate_temperature()` bisects `log τ` to hit a target `firing_entropy`
(default 0.5 ⇒ `√R` ≈ 5.7 effective rules of 32). Verified to land within 0.02 of
targets 0.2/0.5/0.8 for both variants. This replaces HTSK's arbitrary `D` with a
choice made against an observable. Also gave `log τ` its own parameter group at
50× the dense LR — at the dense LR it moved 0.07 across a whole epoch, far too
slow to escape a bad init.

**Finding 3 — the contrastive objective *prefers* a crisp router.** After
calibrating to 0.500, training drove firing entropy down to **0.039** while
`rule_entropy` stayed at 0.95 (all rules in use), and train loss improved (3.86 vs
4.21 uncalibrated). So SGD converts the fuzzy system into a hard mixture of
experts. That is still a valid TSK system (narrow σ), and it is *not* the
saturation failure — gradients kept flowing and every rule stayed alive — but it
abandons the smooth interpolation that motivates fuzzy inference.

This is a real open question rather than a bug, so I added ablation **A8**: an
optional penalty holding `firing_entropy` near target, so crisp-vs-fuzzy can be
*measured* instead of assumed. Caveat: observed on a 101-step run with a
deliberately tiny config (d_in=32, R=8, 25k pairs); it must be rechecked at scale
before any conclusion is drawn.

**Verified working.** Pipeline runs end to end; SIF init, KMeans centre init and
temperature calibration all fire; save/reload reproduces embeddings exactly
(max |Δ| = 0.00e+00); 7,367 pairs/sec at bs=256 on the smoke config.

**Next.** Baselines through our own eval path, then the ladder.

---

## 2026-07-31 — E002 · Baselines measured; training throughput fixed

**Plan.** Measure every small-model baseline through our own `mteb` path *before*
any FES number exists, so the yardstick cannot be retrofitted. Then run the ladder.

**Data acquired.** Full mix loaded: **4,115,821 pairs** over 11 public sources
(gooaq 1.5M, all-nli 558k, msmarco-tas-b 503k, s2orc 400k, agnews 400k,
stackexchange 251k, altlex 113k, simple-wiki 102k, quora 102k, natural-questions
100k, squad 88k). Token counts cached: 28,759 / 30,522 vocabulary items observed.

s2orc needed a streaming loader — its `title-abstract-pair` config has 41.7M rows
and `load_dataset` downloads every shard before you can subset. Streaming reads
only the leading shards. Trade-off recorded in `data.py`: a streamed subset is the
*head* of the file, not a random sample.

### Baselines (self-measured, MTEB-14 + NanoBEIR)

| model | params | MTEB-14 | NanoBEIR | published 41-task `[quoted]` |
|---|---|---|---|---|
| potion-base-32M | 32.30M | **54.08** | 0.4637 | 52.83 |
| potion-base-8M | 7.56M | **52.74** | 0.4421 | 51.32 |
| potion-base-2M | 1.89M | **49.62** | 0.3666 | ~45–48 |

**Harness sanity check passed.** Our 14-task subset ranks these three in the same
order as the published 41-task averages and sits within ~1.5 points of them. The
subset is measuring the same thing, at a consistent offset. It is still not
*equal* to a 41-task average and is never reported as one.

potion-base-8M's task profile reproduces the known static-model weakness:
Retrieval 38.9 and Clustering 35.5 against Classification 61.2 and STS 67.4. That
is the axis where a per-token nonlinearity has the most room to help.

**Targets fixed.** FES-S at ~2.5M params must beat **potion-base-2M (49.62)** to be
interesting, and beating **potion-base-8M (52.74)** at a third of the parameters
would be the strong result.

**Finding 4 — the training bottleneck was the batch sampler, not the model.** The
first ladder launch sat at 1% GPU utilisation and produced no steps in ~15 min.
Rather than guess, I profiled the two candidates:

| component | throughput |
|---|---|
| tokenisation (Rust, batched) | 61,723 texts/s |
| forward+backward, batch 2048, bf16 | 36.9 ms → 55,467 samples/s |
| peak GPU memory, batch 2048 | **0.16 GB** |

Neither is capable of causing the stall, which left `BatchSamplers.NO_DUPLICATES`
— it builds a duplicate index over the whole corpus, and at 4.1M pairs that
dominates everything. The static-embedding blog uses it, but at our corpus size it
is not affordable. Switched to the plain random sampler (configurable): at batch
4096 over 4.1M pairs an accidental in-batch duplicate is rare enough not to
distort InfoNCE. Result: **~2 it/s, ~9 min per rung** instead of unbounded.

The 0.16 GB figure also means batch size was being left on the table, so batches
went from 2048 to 4096 — larger batches give MNRL more in-batch negatives, which
matters most for exactly the retrieval tasks static models are weak at.

Lesson worth keeping: I nearly "fixed" this by cutting sequence length and adding
dataloader workers, neither of which would have helped at all. Measuring first
took two minutes.

**Next.** Ladder rungs A1 (control), A2 (FES-S), A5c (FES-C), A3 (product t-norm),
A4 (no UR), A8 (fuzziness anchor).

**Harness independently validated.** Two of our NanoBEIR measurements reproduce
published values *exactly*:

| model | our NanoBEIR | published `[quoted]` |
|---|---|---|
| all-MiniLM-L6-v2 | 0.562318 | 0.5623 |
| static-retrieval-mrl-en-v1 | 0.503167 | 0.5032 |

Both to 4 decimal places. So NanoBEIR figures in this log are directly comparable
to the static-embedding blog's numbers, and any FES retrieval result can be placed
against the published landscape without an offset caveat. (MTEB-14 still cannot —
it is a 14-task subset, not the 41-task average.)

Full baseline set: MiniLM-L6 60.52 / 0.5623 (22.71M) · potion-32M 54.08 / 0.4637 ·
potion-8M 52.74 / 0.4421 · static-retrieval-mrl 51.02 / 0.5032 (31.25M) ·
potion-2M 49.62 / 0.3666.

---

## 2026-07-31 — E003 · Ladder: control vs. rule base

**Results so far** (1 epoch, 4.12M pairs, batch 4096, seed 42, identical for every rung):

| rung | params | MTEB-14 | NanoBEIR | H_rule | H_fire | train loss |
|---|---|---|---|---|---|---|
| A1 control (R=1, no fuzzy layer) | 2.00M | 50.82 | 0.3841 | — | — | 9.22 |
| A2 FES-S (R=32, HTSK) | 2.52M | 50.96 | **0.4166** | 0.999 | 0.208 | 10.08 |

**Finding 5 — the rule base buys retrieval, not aggregate score.**
NanoBEIR **+0.0325 (+8.5% relative)**; MTEB-14 **+0.14**, which is *below* the
0.5-point threshold this project pre-registered as meaningful on a single run, so
the aggregate difference is **not** claimed. The gain is specifically in retrieval
— which is where static models are weakest (potion-8M: Retrieval 38.9 vs
Classification 61.2) and precisely where a per-token nonlinearity was predicted to
help. Prediction and result line up, but on one axis only.

Placed against the baselines, FES-S at 2.52M params beats potion-base-2M
(50.96 vs 49.62; 0.417 vs 0.367), matches static-retrieval-mrl-en-v1 at **12× the
parameters** on MTEB-14 (50.96 vs 51.02), and reaches **94% of potion-base-8M's
retrieval at ⅓ the parameters** (0.417 vs 0.442). It does not beat potion-8M overall.

**Caveat that must be resolved before claiming anything: A2 has 26% more
parameters than A1.** Added rung **A1b-ctrl-matched** (`d_in=81, R=1` →
2,524,121 params, within 0.14% of A2's 2,520,635). If A2 does not beat A1b, the
rule base is not earning its parameters and the compression claim fails. Queued.

**Finding 6 — A2's *training* loss is worse than A1's (10.08 vs 9.22) while its
retrieval is better.** Same objective, same Matryoshka dims, same steps; the UR
term is ~0.01 and cannot explain it. So the rule base is *harder to fit* yet
generalises better on retrieval — consistent with the rule structure acting as a
regulariser rather than as added capacity. Alternative explanation not yet
excluded: with 32 experts each receives a fraction of the gradient signal, so
`lr_dense=2e-3` may simply be too low for A2. That is a confound between
"regularisation" and "undertrained experts", and it needs an LR sweep on the dense
group before either story is told. Recorded as open.

**Finding 7 — the calibrated temperature is independent of D, which undercuts the
stated motivation for HTSK's `1/D`.** A2 (D=64) calibrated to **τ = 1.195**. A5c
(D=128) calibrated to **τ = 1.195** — the same value. Raw HTSK would have used 64
and 128 respectively, a 2× difference, to normalise a quantity that turns out not
to vary. Once features are LayerNorm'd and σ is fitted from KMeans cluster spread,
the temperature required for a given firing entropy is set by the *feature-to-σ
scale ratio*, not by dimensionality. `1/D` is normalising against the wrong
quantity here; it is merely a crude proxy that happens to break saturation.
Calibrating against firing entropy directly is both better-founded and, per E001,
54× different from `D` in practice.

**Next.** A5c (expressivity), A3 (product t-norm), A4 (no UR), A8 (fuzziness
anchor), then A1b (param-matched control) and the dense-LR sweep for Finding 6.

---

## 2026-07-31 — E004 · Full ladder, and the claims that survived it

All rungs: 1 epoch, 4.12M pairs, batch 4096, seed 42, identical data and step count.

| rung | params | MTEB-14 | NanoBEIR | H_rule | H_fire |
|---|---|---|---|---|---|
| **A1b control — R=1, param-matched** | **2.52M** | **51.44** | 0.4033 | — | — |
| A4 — R=32, no UR | 2.52M | 51.38 | **0.4220** | 0.970 | 0.010 |
| A8 — R=32, fuzziness anchored | 2.52M | 51.25 | 0.4200 | 1.000 | 0.503 |
| A2 — R=32, HTSK + UR | 2.52M | 50.96 | 0.4166 | 0.999 | 0.208 |
| A3 — R=32, product t-norm | 2.52M | 50.86 | 0.4181 | 0.997 | 0.141 |
| A1 control — R=1, *unmatched* | 2.00M | 50.82 | 0.3841 | — | — |
| A5c — context-conditioned | 2.52M | 43.34 | 0.2676 | 0.645 | 0.000 |

### Finding 8 — at matched parameters the rule base buys retrieval only, and half the apparent gain was parameters

This is the result the whole project turns on, and it is much weaker than the
unmatched comparison suggested.

| comparison | MTEB-14 | NanoBEIR |
|---|---|---|
| A4 − A1 (unmatched control, +26% params for A4) | +0.56 | +0.0379 |
| **A4 − A1b (matched control)** | **−0.06** | **+0.0187** |

- **Aggregate: a tie.** −0.06 is far inside the ±0.5 noise band this project
  pre-registered. The entire +0.56 against A1 was the extra 26% of parameters.
- **Retrieval: +0.0187 (+4.6%) survives**, but that is *half* the +0.0379 measured
  against the unmatched control. So roughly half the retrieval gain was capacity
  and half is architecture.

Stated plainly: **a TSK rule base gives no aggregate benefit over a plain wider
static table at equal cost, and a modest retrieval-only benefit.** Running the
matched control before writing anything up was the difference between that
sentence and a much more flattering, wrong one.

I never set a significance threshold for NanoBEIR the way I did for MTEB-14, so
+0.0187 is currently unqualified. Seed replication of exactly A1b and A4 (seeds 7,
1234) is running to settle it — this took priority over the rule-count sweep,
because a sweep over an effect that might be noise is wasted compute.

### Finding 9 — every specifically *fuzzy* ingredient is neutral or mildly harmful

Four rungs span a **50× range in routing softness** and are all within 0.5 MTEB-14
and 0.005 NanoBEIR:

| H_fire | rung | MTEB-14 | NanoBEIR |
|---|---|---|---|
| 0.010 | A4 (no UR) | 51.38 | 0.4220 |
| 0.141 | A3 (product t-norm) | 50.86 | 0.4181 |
| 0.208 | A2 (HTSK + UR) | 50.96 | 0.4166 |
| 0.503 | A8 (anchored) | 51.25 | 0.4200 |

- **The fuzziness does not matter.** A8 proves the anchor works mechanically
  (entropy held at exactly its 0.503 target) and that holding the system genuinely
  fuzzy changes nothing measurable.
- **HTSK does not matter** (A3 ≈ A2) — because τ and σ are the same parameter; see
  Finding 7 and the corrections now in `DESIGN.md` §2.1, `README.md`, and
  `docs/02-fuzzy-systems-high-dim.md` §2. My earlier "HTSK is mandatory / the
  load-bearing piece" framing was an overstatement and has been retracted in all
  three documents. It is load-bearing at init with fixed σ — which is the regime
  Cui/Wu/Xu study, and the regime this repo's own NumPy `fit_gaussians` sits in —
  and not load-bearing under end-to-end SGD.
- **Rule balancing mildly hurts**: A4 (no UR) > A2 (UR) on both metrics. The
  load-balancing term buys entropy 0.970 → 0.999 and pays for it in quality.
- Left to itself the model drives H_fire to 0.010, i.e. it *wants* to be a hard
  router.

So what works is the **mixture-of-experts structure** — R local linear maps gated
by a prototype router — and not the fuzzy inference on top of it. That is a
negative result about fuzzy systems for this task, and it should be reported as
one.

### Finding 10 — the rule base trades smooth geometry for discrimination

Per-task-type, A4 against the A1 control:

| type | A1 | A4 | Δ |
|---|---|---|---|
| **Retrieval** | 35.8 | **39.6** | **+3.8** |
| Summarization | 29.8 | 31.3 | +1.5 |
| Classification | 51.6 | 52.9 | +1.3 |
| Pair classification | 68.5 | 69.2 | +0.7 |
| Reranking | 61.9 | 62.3 | +0.4 |
| STS | 67.8 | 66.5 | −1.3 |
| **Clustering** | 40.4 | **37.9** | **−2.5** |

Coherent mechanism: near-hard routing makes the embedding map piecewise, so it
sharpens lexical/topical discrimination (retrieval, classification) and damages
tasks that need continuous geometry (k-means clustering, correlation-based STS).
Supporting detail — A8, the genuinely fuzzy rung, has the best clustering of any
fuzzy config (38.5 vs A4's 37.9), i.e. softness helps precisely where continuity
matters. That is the one place fuzziness showed a directional benefit, and it is
too small to claim on one seed.

### Finding 11 — FES-A4 beats potion-base-8M on 3 of 7 task types at ⅓ the parameters

| type | potion-8M (7.56M) | FES-A4 (2.52M) |
|---|---|---|
| Retrieval | 38.9 | **39.6** |
| Clustering | 35.5 | **37.9** |
| Summarization | 29.2 | **31.3** |
| Reranking | **63.1** | 62.3 |
| STS | **67.4** | 66.5 |
| Pair classification | **73.7** | 69.2 |
| Classification | **61.2** | 52.9 |
| **average** | **52.74** | 51.38 |

The whole 1.36-point deficit is Classification (−8.3) and Pair classification
(−4.5). Hypothesis, **not** a conclusion: this is a *data* effect, not an
architectural one. POTION is distilled from `bge-base-en-v1.5`, a strong
general-purpose teacher, whereas our mix is retrieval-heavy (gooaq 1.5M + msmarco
503k + s2orc 400k of 4.12M) and contains no supervised classification signal at
all. Testable by adding classification-style pair data and re-running; not yet done.

### Finding 12 — FES-C failed, with a specific and falsifiable mechanism

A5c is the one large regression: MTEB-14 43.34 (−8.1), NanoBEIR 0.2676 (−0.155),
H_fire **0.000**, H_rule 0.645. Despite being calibrated to 0.500 at init, the
context-conditioned router collapsed to hard one-hot routing.

Retrieval fell hardest (20.9 vs 35.8), which points at the mechanism:
**hard gating on document context breaks query–document comparability.** A query
and its answer document have very different context vectors `c`, so they route to
*different* experts, and their embeddings are no longer produced by the same map.
Cosine similarity between them stops being meaningful.

Note the contrast that makes this specific rather than generic: A4 also has
essentially hard routing (H_fire 0.010) and is the *best* rung. So hard routing per
se is fine — hard routing **on context** is what breaks. Rung `A5c-anchored`
(context conditioning + fuzziness anchor, forcing soft routing) is queued as the
direct test. If the mechanism is right, most of the loss should come back.

**Status of the two original claims.**
- *Compression* (A2/A4 vs A1b): **not supported** on aggregate; partially supported
  on retrieval, pending seed replication.
- *Expressivity* (A5c): **refuted as designed**, with a mechanism and a queued fix.

---

## 2026-07-31 — E005 · Protocol change: accept published figures for comparison models

**Directed.** Stop spending compute re-measuring competitors; use their published
benchmarks. All remaining GPU time goes to FES ablations. This retroactively
endorses killing the EmbeddingGemma-300M run (E004 note: 470 s/iteration, ~5 h for
one task, and it was slowing the ladder 2.5×).

The five small baselines already measured are kept — they are strictly more
informative than the published numbers, and they turn out to enable something the
published numbers alone cannot.

### Finding 13 — the MTEB-14-to-published offset is family-specific, so cross-family aggregate comparison is invalid

Having both numbers for four models lets the relationship be measured rather than
assumed:

| model | ours (MTEB-14) | published (41-task) | offset |
|---|---|---|---|
| potion-base-32M | 54.08 | 52.83 | **+1.25** |
| potion-base-8M | 52.74 | 51.32 | **+1.42** |
| all-MiniLM-L6-v2 | 60.52 | ~56.1 | **+4.42** |

A tight +1.25…+1.42 across the static family, but **+4.42** for a transformer.
MTEB-14 is therefore *not* a simple shift of the 41-task average — our 14-task
subset is relatively kinder to transformers (equivalently, harsher on static
models). I had been treating "consistent offset" as established in E002 on the
strength of the three potion models alone; adding MiniLM shows that was too
confident, and the conclusion needed narrowing.

Rules adopted, and now enforced in `scripts/report.py` and `docs/03-benchmarks.md` §1a:

1. **Within the static family** (potion-*, static-retrieval-mrl, FES — i.e. our
   actual competitive set) the offset is stable at ≈ **+1.3** and rank order holds.
   Published static figures are comparable to ours through that offset, stated
   each time.
2. **Across families**, MTEB-14 is not used. No FES-vs-MiniLM or FES-vs-Gemma
   aggregate claims.
3. **NanoBEIR needs no offset** — ours reproduces published values to four decimal
   places. Every cross-family claim is made on NanoBEIR. This is a second reason
   the retrieval result, not the aggregate, is what carries the argument.

### FES placed on the published static scale

Applying rule 1 (subtract ~1.3):

| model | params | published-scale MTEB avg | NanoBEIR |
|---|---|---|---|
| potion-base-8M | 7.56M | 51.32 `[published]` | 0.4421 |
| **FES-A4 (R=32)** | **2.52M** | **~50.1** | **0.4220** |
| **FES-A1b (R=1 control)** | **2.52M** | **~50.1** | 0.4033 |
| potion-base-2M | 1.89M | ~45–48 `[published]` | 0.3666 |

So the 2.5M-parameter model lands **clearly above the published 2M tier** and
**~1.2 below potion-base-8M at one third the parameters**. That holds for the
control as much as for the fuzzy model, which is exactly Finding 8's point: the
win over the 2M tier comes from the narrow-table + learned-SIF-pool + contrastive
recipe, and the rule base adds retrieval specifically.

---

## 2026-07-31 — E006 · FES-C: mechanism confirmed, variant refuted

**Plan.** A5c collapsed with H_fire = 0.000. Hypothesis (E004 Finding 12): hard
gating on document context routes a query and its answer document to *different*
experts, so their embeddings stop being comparable. Test: rerun with the fuzziness
anchor forcing soft routing. If the mechanism is right, most of the loss returns.

| config | MTEB-14 | NanoBEIR | H_rule | H_fire |
|---|---|---|---|---|
| A2 FES-S (no context) | 50.96 | 0.4166 | 0.999 | 0.208 |
| A5c FES-C, hard | 43.34 | 0.2676 | 0.645 | 0.000 |
| **A5c-anchored, soft** | **49.19** | **0.3596** | 1.000 | 0.980 |

### Finding 14 — the mechanism is confirmed, and the variant still fails

Soft routing recovered **+5.85 MTEB-14 and +0.0920 NanoBEIR** over hard routing.
The prediction was quantitative and directional and it held, so the
query–document comparability explanation stands.

But A5c-anchored (49.19 / 0.3596) is still well below plain FES-S (50.96 / 0.4166)
and below even the 2.00M A1 control (50.82 / 0.3841). **Context conditioning is net
harmful in both regimes**, for two different reasons:

1. **Hard** → broken query/document comparability (above).
2. **Soft** → the model *neutralises the context signal on purpose*. The anchor
   targeted H_fire = 0.5; the model settled at **0.980**, accepting a penalty of
   ≈2.3 in the objective to get there. Near-uniform firing means
   `u_t = Ā v_t + b̄`, one linear map. The optimiser spent real objective value to
   make itself context-free, then wasted 0.5M parameters on 32 experts it averages
   together. That is stronger evidence than a simple score drop: the model is not
   merely failing to exploit context, it is actively paying to escape it.

**Why the idea was wrong, in hindsight.** A mean-pooled `c` is *the same vector for
every token in a document*. So it cannot disambiguate a token against its
neighbours; it only shifts the whole document's routing. That is all of the cost of
gating with none of the benefit — and worse, it makes the routing a function of
document statistics, which is precisely what must *not* differ between a query and
its answer. To be useful the context term would have to be token-specific, which is
approaching attention and leaves the O(L) cost tier that motivates the design.

**Marked REFUTED in `DESIGN.md` §2.3** rather than left as an open direction.

**Consequence for the project's framing.** Both original claims are now settled:
*compression* is a tie on aggregate with a retrieval-only gain (Finding 8, pending
seeds), and *expressivity* is refuted. What remains defensible is narrower and
should be stated as such: FES is a **competitive sub-3M embedding model** whose
rule structure gives a **retrieval-specific** edge, and whose fuzzy-inference
framing turns out to be an interpretability and design lens rather than a source of
accuracy.

---

## 2026-07-31 — E007 · Scaling and optimisation: cost curves first

**Plan.** The brief deferred "scaling expansion". Before spending GPU hours on
quality, measure the *cost* side, so the expensive question is asked of a short
list instead of a grid. Also implement the one scaling lever specified in
`DESIGN.md` §3 but never built: low-rank consequents.

### Implemented: low-rank consequents

`A_r = U_r @ V` with `V ∈ ℝ^{k×d_out}` shared across rules. Each rule keeps its own
`d_in → k` map (so rules stay individually interpretable); what they share is the
`k → d_out` output basis. Cost goes from `R·d_in·d_out` to `R·d_in·k + k·d_out`.

Fused into the existing pooling refactor — contract to the shared `k`-dim basis
first, so the `(k, d_out)` matrix is touched once per batch rather than once per
rule. `test_lowrank_consequent_is_equivalent_to_its_dense_expansion` checks it
against a dense `A_r` built from the same product. 26 tests pass.

### Parameter cost (exact, `d_in=64, d_out=256`)

| R | dense | rank 32 | rank 16 |
|---|---|---|---|
| 32 | 2,520,635 | 2,070,075 | 2,033,211 |
| 128 | 4,130,363 | 2,303,547 | 2,168,379 |
| 512 | 10,569,275 | **3,237,435** | 2,709,051 |

And at `R=64`, scaling `d_out` 256 → 768 costs 5,187,131 dense but only
**2,197,051** at rank 32.

### Measured throughput (batch 512, seq ≤ 256)

⚠️ **Caveat: two training jobs were running on the same GPU/CPU during these
measurements, so absolute numbers are contaminated and one pair is
non-monotonic (R=64 dense measured *faster* than R=32 on CPU, which cannot be
real). The 3–7× effects below are far larger than that noise, but the table needs
re-running on an idle machine before any absolute figure is quoted.**

| config | params | GPU sent/s | CPU sent/s |
|---|---|---|---|
| R=1 control (d_in=81) | 2,524,121 | 842,175 | **426,217** |
| R=32 dense (current) | 2,520,635 | 271,310 | 94,859 |
| R=128 dense | 4,130,363 | 249,061 | 58,179 |
| R=512 dense | 10,569,275 | 33,588 | 13,924 |
| R=64 rank32 | 2,147,899 | 351,766 | 177,453 |
| R=512 rank32 | 3,237,435 | 44,014 | 18,770 |
| R=128 rank32 d_out=512 | 2,344,507 | 114,382 | 85,702 |
| R=128 rank32 d_out=768 | 2,385,467 | 122,014 | 79,536 |

### Finding 15 — low rank decouples R from *parameters* but not from *compute*

R=512 costs 3.3× fewer parameters at rank 32 (3.24M vs 10.57M) but throughput is
essentially unchanged (18,770 vs 13,924 CPU sent/s — same order, both ~5–7× worse
than R=32). The reason is structural and worth stating precisely:

- The **consequent** factors out of the sequence pool (E001 Finding 1), so it runs
  once per *document* — that is why `d_out` is nearly free.
- The **antecedent** does not. Firing strengths are `O(L·R·d_in)` and must be
  computed for every token against every rule. Low rank touches only the
  consequent, so it cannot help here.

So the bottleneck moves from consequent to antecedent as R grows, and rank is the
wrong tool for that half. Scaling R needs a *sparse* antecedent (top-k rule
selection, or a hierarchical rule tree) — which is exactly the "hierarchical /
stacked TSK" direction in the fusion survey (B4) and the group-Lasso rule pruning
in B5. Recorded as the concrete next step for R.

### Finding 16 — `d_out` is nearly free, so that is where scaling should go first

256 → 768 at rank 32: **+3.5% parameters, −8% throughput**. This is the direct
payoff of the fused-pool refactor and the most useful scaling result here. Output
width is the cheapest axis FES has, and it is the axis that most directly buys
retrieval quality in static models (potion-32M uses 512-d, static-retrieval-mrl
1024-d).

### Finding 17 — the rule base costs 4.5× CPU throughput for a retrieval-only gain

R=1 control: 426,217 CPU sent/s. R=32: 94,859. Same parameter count.

Set against Finding 8 (aggregate tie, +0.019 NanoBEIR), the honest cost/benefit is
**poor**: 4.5× the compute for no aggregate gain and ~4.6% relative retrieval.
Absolute throughput is still firmly in the static tier — ~95k sentences/s on CPU is
orders of magnitude above any transformer — so FES remains *usable*, but "spend the
compute on more rules" is not supported by these numbers. Spend it on `d_in`/`d_out`
instead, which is to say: on the axes a plain static model would use.

This is the sharpest version of the project's central negative result, and it is a
cost measurement rather than an accuracy one, so it is independent of the seed
question hanging over Finding 8.

### Queued

**Optimisation first** (the right LR is a prerequisite for scaling — Finding 6's
confound between "the rule base regularises" and "32 experts are undertrained at
`lr_dense=2e-3`"):
- `O1-lr-dense-8e-3`, `O2-lr-dense-3e-2` — 4× and 15× the dense LR
- `O3-2epoch` — is one epoch simply not enough for a rule base?

**Then the scaling ladder**, ordered by what the cost curves recommend:
- `S2-wide-out-512` (2.09M) — `d_out` doubled for ~nothing; the cheapest bet
- `S1-table-128` (5.00M) — spend on the vocabulary table instead
- `S3-balanced` (4.27M) — table + output + R=64 together
- `S4-potion8M-matched` (8.14M) — parameter-matched to potion-base-8M, the
  head-to-head the whole project has been building toward
- `S5-manyrules-rank` (2.61M) — R=256 now that it is affordable in parameters;
  Finding 15 predicts it will be slow and Finding 9 predicts it will not help,
  so this is a genuine test of both


### Finding 18 — A0 settles the classification gap: it is the training data, not the architecture

`A0-static-256` is the *conventional* static architecture (full-width 256-d table,
mean pool, no rule base) at 7.91M, trained on our data with our recipe. It was
queued as a sanity reference and turned out to answer Finding 11's open question.

| model | params | MTEB-14 | Class | Clust | PairCl | Rerank | Retr | STS | Summ | NanoBEIR |
|---|---|---|---|---|---|---|---|---|---|---|
| potion-base-8M (distilled from BGE) | 7.56M | 52.74 | **61.2** | 35.5 | **73.7** | 63.1 | 38.9 | 67.4 | 29.2 | **0.4421** |
| **A0 static-256 (our recipe)** | 7.91M | 52.49 | 55.5 | **39.9** | 72.7 | **63.4** | 38.7 | **69.1** | 28.1 | 0.4284 |
| A4 fuzzy (R=32) | 2.52M | 51.38 | 52.9 | 37.9 | 69.2 | 62.3 | **39.6** | 66.5 | **31.3** | 0.4220 |
| A1b control (R=1) | 2.52M | 51.44 | 52.5 | 40.2 | 69.1 | 62.6 | 36.6 | 68.3 | 30.9 | 0.4033 |

**(a) The from-scratch recipe matches distillation at equal size.** A0 52.49 vs
potion-8M 52.74 — ahead on Clustering (+4.4) and STS (+1.7), behind on
Classification. Independent validation of the training setup, separate from
anything fuzzy.

**(b) Finding 11's hypothesis is confirmed.** A0 has *no rule base* and still
trails potion by **5.7** on Classification. So the classification deficit is not an
artifact of the fuzzy architecture — it is the data/teacher difference (POTION
distils from `bge-base-en-v1.5`; our mix is retrieval-heavy with no supervised
classification signal). Free test: A0 was queued for another purpose entirely.

**(c) Finding 19 — the two scaling levers are complementary, and target different
task types.** Comparing A1b → A0 (widen the table, 2.5M → 7.9M) against
A1b → A4 (add a rule base at fixed parameters):

| lever | Class | PairCl | STS | Retrieval | NanoBEIR |
|---|---|---|---|---|---|
| table width (A1b → A0, +5.4M params) | **+3.0** | **+3.6** | +0.8 | +2.1 | +0.025 |
| rule base (A1b → A4, +0 params) | +0.4 | +0.1 | −1.8 | **+3.0** | **+0.019** |

Table width buys the classification family; the rule base buys retrieval. And
notably **A4 at 2.52M beats A0 at 7.91M on Retrieval** (39.6 vs 38.7) — on that
task type the rule base is a better use of parameters than 3.1× the table.

So the scaling recipe is *both levers together*, not either alone, and the queued
`S1-table-128` / `S3-balanced` / `S4-potion8M-matched` are exactly that
combination. This is the most directly actionable result in the log.


## 2026-07-31 — E008 · Interpretability pass: the rules are not interpretable

**Plan.** `DESIGN.md` §7 claimed the distinctive payoff of a fuzzy formulation is a
*readable* rule base — "the thing only a fuzzy model gets". Decode the trained A4
rule base and test the stated hypothesis: rules specialise into linguistically
recognisable regions, and low-contribution rules coincide with low-information
tokens (i.e. the model rediscovers IDF as emergent rule structure).

**Run.** `scripts/inspect_rules.py artifacts/A4-no-ur --counts …` → `results/A4_rules.json`.

### Finding 20 — REFUTED. The rule base is an arbitrary partition, not a semantic one.

**(a) Prototypes decode to nothing recognisable.** Top-firing tokens for the
most-used rules:

```
RULE 25 (8.82% usage): ##mate melbourne freezing indigenous pakistani glen
                        protesters clutching broadcaster abbey
RULE 18 (8.02% usage): days investigation voiced brad 1840 …
RULE 11 (5.53% usage): married rebels 1975 2013 reasonable downstairs telugu
```

No theme — not subword continuations, not function words, not numerals, not
topical clusters. The predicted categories do not appear in any rule.

**(b) Every rule contributes equally, so there are no "important" rules.**

| statistic | min | max | spread |
|---|---|---|---|
| `‖A_r‖_F` | 15.142 | 15.617 | **3.1% of mean** |
| `‖b_r‖` | 0.886 | 1.086 | 20.1% of mean |
| corpus usage | 1.56% | 8.82% | (uniform = 3.12%) |

A 3.1% spread in the expert weight norms means the rules are essentially
interchangeable in magnitude. The hoped-for structure — some rules loud, some
quiet, tracking information content — is absent.

**(c) The emergent-IDF correlation is +0.080.** Between rule-token rarity and rule
contribution magnitude, across 32 rules. That is zero. The §7 hypothesis is refuted
outright, not merely weak.

**Why, mechanistically.** The model trains to a near-hard router (per-token firing
entropy over the vocabulary: **0.018**), so it is performing vector quantisation of
the token-feature space into 32 roughly equal-mass cells. But that feature space is
itself *learned from scratch with no semantic anchoring* — nothing in the objective
rewards cells that align with linguistic categories, and a contrastive loss is
perfectly happy with an arbitrary partition. So we get a learned VQ of a learned
space: functional, and semantically meaningless.

**Caveat on the one thing that does look IDF-like.** The learned pooling weights
put the highest mass on rare proper nouns (`temeraire`, `netball`, `maccabi`,
`bethany`) and low mass on connectives (`additionally`, `-`), which reads exactly
like IDF. But those weights were **SIF-initialised from corpus frequency**
(`init_pool_weights_from_frequency`), so this is largely inherited from
initialisation, not learned structure. It cannot be claimed as emergent without an
ablation against zero-init pooling, which has not been run.

**What would be needed to get interpretable rules.** Anchor the antecedent space to
something with pre-existing structure — e.g. initialise `F` from a distilled static
embedding (Model2Vec) rather than randomly, so the KMeans prototypes land on
genuine semantic clusters, or regularise the partition toward a known taxonomy.
Untested; recorded as the concrete next step for anyone pursuing the
interpretability angle.

**Status of the project's claims after E008.** All three distinctive claims have
now been tested and none survived in the form originally stated:

| claim | origin | status |
|---|---|---|
| compression (rule base beats a wider table at equal params) | DESIGN §2.2 | **aggregate: refuted** (tie vs A1b); retrieval-only gain survives |
| expressivity (context conditioning breaks the bag-of-words ceiling) | DESIGN §2.3 | **refuted**, two mechanisms (E006) |
| interpretability (readable rule base) | DESIGN §7 | **refuted** (this entry) |

What remains, and it is worth stating without decoration: FES is a **competitive
sub-3M embedding model** — it beats potion-base-2M substantially and matches
potion-base-8M on retrieval at a third the parameters — whose rule base delivers a
**real, replicated, retrieval-specific gain** (E004 Finding 8; seeds non-overlapping
as of E009). The *fuzzy* framing supplied a useful design vocabulary and a genuinely
useful diagnostic (`firing_entropy`), but it did not deliver accuracy, expressivity,
or interpretability beyond what the underlying mixture-of-experts structure gives.


## 2026-07-31 — E009 · Operational: never run two trainings on this GPU

**Symptom.** Both active runs collapsed to **~320 s/it** — ladder3 reached step
16/1010 in 54 minutes (89 hours projected), and the seed run, which had been fine,
dropped from ~2 it/s to 317 s/it at step 651.

**Diagnosis.** GPU memory read **11,722 / 12,282 MiB**. On Windows/WDDM, once VRAM
is exhausted the driver silently spills allocations to shared system memory rather
than failing, so there is no OOM error — just a ~600× slowdown that looks like a
hang. System RAM was fine (69 GB free of 100 GB), so it was purely VRAM.

Note it was *not* the models: FES weights are ~10 MB and a training step peaks at
0.16 GB. The pressure came from two full training processes plus their dataloader
and evaluation buffers coexisting, each having sized its caching allocator against
a GPU it assumed it had to itself.

**Fix.** Killed ladder3 (16 steps in, nothing lost). GPU dropped to 6,179 MiB and
the seed run immediately recovered to **1.98 it/s** — confirming the diagnosis
rather than assuming it.

**The first fix was itself broken, and reproduced the failure.** I added
`scripts/run_after.sh` to block until the prior run exited — using `pgrep -f`.
In Git Bash `pgrep` does not see **native Windows** processes, so the wait loop
matched nothing, exited immediately, and launched the second run anyway. Within
minutes both runs were back at 413 → 675 s/it. The guard caused exactly the
failure it was written to prevent, and I only caught it because the monitor
surfaced the step timings.

Replaced with `scripts/run_after.ps1`, which polls `Get-CimInstance
Win32_Process` — that *does* see native processes. Verified after launch that
only the prior run was active and the guard was genuinely waiting, rather than
assuming it worked this time.

**Lesson worth keeping.** I had been treating "the model is tiny, so two runs fit"
as obviously true, and it was true of the *weights* and false of everything else.
Earlier concurrency was already costing ~2.5× (rung A5c took 49 min instead of
~19, A0 ran at 3.0 s/it instead of ~1) — I noticed and explained that at the time
but did not act on it, and it then degraded non-linearly into the 600× cliff. The
signal was there an hour before the failure.


## 2026-07-31 — E010 · Seed replication: the retrieval gain is real, the aggregate null is definitive

Three seeds (42, 7, 1234) of the two configs that the whole argument rests on,
identical in everything but the rule base.

| metric | A4 rule base (R=32) | A1b control (R=1) | difference | exact permutation p |
|---|---|---|---|---|
| MTEB-14 | 51.243 ± 0.299 | 51.363 ± 0.159 | **−0.12** | 0.750 |
| NanoBEIR | **0.42511 ± 0.00279** | 0.40305 ± 0.00667 | **+0.0221 (+5.47%)** | **0.050** |

Raw values —
A4 NanoBEIR {0.42200, 0.42738, 0.42595}; A1b {0.40332, 0.40959, 0.39625}.

### Finding 21 — the retrieval gain replicates with complete separation

**Every** A4 run beats **every** A1b run (min A4 0.42200 > max A1b 0.40959). Under
the null, the probability of that arrangement is 1/C(6,3) = **0.050**, which is the
smallest p attainable at n=3 per group — i.e. this is as strong as three seeds can
possibly show, and a fourth seed each would be needed to go below it. Effect size
+5.47% relative, roughly 3× the pooled seed spread.

This is the project's one surviving quantitative claim, and it is now properly
supported rather than asserted from a single run. Note the honest limit: the exact
test is *at* its floor, so "p = 0.05" here means "maximally separated at this sample
size", not "comfortably significant". Reporting it as `p ≤ 0.05` with n=3 and
complete separation stated explicitly is the accurate form.

### Finding 22 — the aggregate null is definitive, not merely under-powered

MTEB-14 difference is **−0.12 in the control's favour**, p = 0.750, with both
groups' seed spreads (0.16–0.30) larger than the gap. Combined with E004's
parameter-matched design, this closes the compression claim: **a TSK rule base does
not improve aggregate embedding quality over a plain wider static table at equal
parameter count.** Three seeds, one code path, matched to 0.14% in parameters.

My original pre-registered threshold (±0.5 points) turned out to be conservative —
actual MTEB-14 seed noise is 0.16–0.30 — so the null holds under a tighter bar than
the one I set in advance.

**Where the project lands, quantitatively.** The rule base is a **retrieval-specific
mechanism**, not a general one. That is consistent with every other result in this
log: Finding 10 (retrieval +3.8, clustering −2.5), Finding 19 (rule base buys
retrieval, table width buys the classification family), and Finding 17 (it costs
4.5× CPU throughput to get it). Whether that trade is worth making depends entirely
on whether the deployment is retrieval-shaped.


### E009 addendum — the serialisation guard failed twice more; simplest option was correct

Full sequence, recorded because the pattern (over-engineering a guard, then
mis-verifying it) cost ~35 minutes across three attempts:

1. **bash `pgrep -f` guard** — Git Bash `pgrep` cannot see native Windows
   processes, so it never waited. Launched a second concurrent run.
2. **My kill of that guard silently did nothing.** I used
   `Get-Process bash | Where-Object { $_.CommandLine -like … }`. `Get-Process`
   has **no `CommandLine` property**, so the filter matched nothing and killed
   nothing. The guard survived and fired later. (`Get-CimInstance Win32_Process`
   is the one that exposes `CommandLine`.)
3. **PowerShell guard** — the wait logic was correct, but
   `Start-Process -ArgumentList` split the `-Cmd` string on spaces, so
   `run_after.ps1` received only `.venv\Scripts\python.exe` and ran
   `cmd /c "python.exe >> log"` → **an interactive Python REPL**, whose `>>>`
   prompt went into the experiment log. No training ran at all.

Net effect: for ~30 minutes I believed the ladder was running when the GPU was at
0 MiB. Nothing was lost — all 18 result records intact — but the *appearance* of
progress was the dangerous part, and it existed because I checked "did I launch
it?" rather than "is it computing?".

**Resolution: deleted the guard entirely and launched directly with `nohup`**, the
approach that had worked for every earlier ladder. Serialisation was only needed
while the seed run was in flight; that run had already finished, so the guard was
solving a problem that no longer existed.

**Two checks now used before trusting any launch**, both cheap:
- `nvidia-smi --query-gpu=memory.used` — is the GPU actually allocated?
- count of `run_experiments` processes via `Get-CimInstance` — exactly one logical
  run (which appears as 2 PIDs: a 4 MB `.venv` launcher shim plus the real worker).


## 2026-07-31 — E011 · Optimisation sweep resolves the Finding 6 confound

**The question.** E003 Finding 6: A2's *training* loss was worse than the R=1
control's (10.08 vs 9.22) while its retrieval was better. Two explanations were
open — (a) the rule base regularises, or (b) with 32 experts each seeing a
fraction of the gradient, `lr_dense=2e-3` is simply too low and the experts are
undertrained. Explanation (b) is testable directly.

**Sweep** (identical architecture, `R=32`, `ur_weight=0`, only `lr_dense` varies):

| rung | lr_dense | MTEB-14 | NanoBEIR | H_fire |
|---|---|---|---|---|
| **A4 (baseline)** | **2e-3** | **51.243 ± 0.299** (n=3) | **0.42511 ± 0.00279** (n=3) | 0.010 |
| O1 | 8e-3 (4×) | 51.41 | 0.41572 | 0.006 |
| O2 | 3e-2 (15×) | 50.72 | 0.41900 | 0.002 |

### Finding 23 — the experts are not undertrained; explanation (b) is eliminated

Both higher learning rates are **worse on retrieval**, and both fall *below the
minimum of A4's three-seed range* (0.4220) — 0.41572 and 0.41900 against
[0.42200, 0.42738]. So the effect is outside seed noise, not a coin flip. On
aggregate, O1 is within noise and O2 is clearly worse.

If the experts were starved of gradient at 2e-3, raising the LR 4× and 15× would
have helped. It did the opposite. **`lr_dense = 2e-3` is at or near optimal, and
Finding 6's higher training loss is not an optimisation artefact** — it is the rule
base being genuinely harder to fit while generalising better on retrieval, i.e. the
regularisation reading.

That is a satisfying result for a specific reason: it means the retrieval gain in
Finding 21 cannot be dismissed as "the two configs were just tuned differently".
The control and the rule base were both run at the LR that is best for the rule
base, which if anything favours the control.

**Side observation.** `H_fire` falls monotonically with LR (0.010 → 0.006 → 0.002):
a larger dense LR drives the router to crisp decisions faster. Consistent with
Finding 9 — the model always heads toward hard routing when nothing holds it back —
and with the fact that it does not help.

**Still open on this axis:** `O3-2epoch` asks the complementary question (is one
epoch simply not enough?), which the LR sweep does not answer.


## 2026-07-31 — E012 · The low-rank consequent is broken as a scaling lever

**S2 result that triggered this:** `d_out=512, consequent_rank=32` →
MTEB-14 **47.79**, NanoBEIR **0.3300**, against A4's 51.24 / 0.4251 at *half* the
output width. A 3.5-point / 0.095 regression from a change advertised as free.

S2 varied two things at once (d_out and rank), so rather than guess I measured the
effective rank of the embeddings each model actually produces:

| model | d_out | effective embedding rank |
|---|---|---|
| S2 (`consequent_rank=32`) | 512 | **65** (13% of d_out) |
| A4 (dense) | 256 | 256 (100%) |

### Finding 24 — a shared output basis caps the whole model's embedding rank

The factorisation is `A_r = U_r V` with **`V ∈ ℝ^{k×d_out}` shared across rules**.
The fused forward pass is therefore

```
e = (Σ_r Σ_i G_ri U_rik) V  +  h B
```

The first term lands in the row space of `V` — dimension `k`. The second has rank at
most `min(R, d_out)`. So the model's entire output lives in a subspace of dimension
`k + min(R, d_out)`, **regardless of `d_out`**. At `k=32, R=32, d_out=512` that is
64, and 65 is what was measured.

S2 was therefore a 65-dimensional model padded out to 512 — strictly worse than
A4's genuine 256. The regression is fully explained.

**How my test missed it.** `test_lowrank_consequent_is_equivalent_to_its_dense_expansion`
asserts `A_r = U_r V` behaves identically to a dense `A_r` built from that product.
That assertion is *true* and it passes — but the dense expansion is **itself
rank-32**. I verified the algebra and never checked the modelling consequence. The
parameter table in E007 was correct and the cost curves were correct; the thing I
never measured was whether the resulting model could use the width it was paying for.

**Correction to E007 Finding 16.** "d_out is nearly free" was a *cost* measurement
and remains true as such. But combined with low rank it is misleading: you get the
cheap parameter count and a model that cannot fill the output space. To scale
`d_out` honestly, `k` must grow with it — which removes essentially all of the
saving. **Low-rank consequents do not enable scaling `d_out`.**

They may still help scale `R` at *fixed* `d_out` where `k ≈ d_out`, but then the
saving is only on the `R·d_in·d_out` term, which is smaller than advertised.

**Actions taken.**
- Killed the ladder before S3/S4/S5 ran — **all three used `consequent_rank=32`**
  and would have produced three more hours of rank-crippled results that looked
  like genuine scaling failures.
- Rewrote S2/S3/S4/S5 to use **dense** consequents. S4-potion8M-matched is now
  `d_in=190, d_out=256, R=32` → **7,406,915 params vs potion-base-8M's 7,560,000**
  (within 2%), which is the head-to-head the project has been building toward.
- Added `test_lowrank_caps_the_embedding_rank`, which measures the SVD rank of real
  embeddings and asserts `rank ≤ k + R` and `rank < d_out` — the check that was
  missing.
- The model now emits a `UserWarning` at construction when
  `consequent_rank + n_rules < d_out`, quoting this finding.
- Kept the failed run as `FES-S2-rank32-BROKEN` in the records rather than deleting
  it, and added `S6-rank-bottleneck-demo` (`k=256, d_out=512`) so the bottleneck can
  be shown to disappear when `k` is adequate.

**Lesson.** I validated the new lever with a mathematical identity and a parameter
count — both of which it satisfied — and shipped it into five queued experiments
without once checking what the model could actually represent. A single SVD on 200
embeddings, which takes seconds, would have caught it before any of them ran.


## 2026-07-31 — E013 · Silent cache contamination in the MTEB harness

**Symptom.** The corrected dense S2 reported MTEB-14 **47.79** — *identical to two
decimals* to the broken rank-32 S2 — while its NanoBEIR moved 0.3300 → 0.4353. A
0.105 jump on one metric and a 0.00 change on the other is not a plausible
measurement.

**Cause.** `mteb` caches per output folder, which `run_mteb` keys on `model_name`.
Both S2 runs used the tag `FES-S2-wide-out-512`, and `run_mteb` defaulted to
`overwrite=False`, so the second run **reloaded the first model's task results**.
File timestamps in `results/mteb/FES-S2-wide-out-512/` confirm it: all 16 task
JSONs written 20:42–20:44 (the broken run), only `model_meta.json` rewritten at
21:06 (the corrected run). NanoBEIR is computed live by the sentence-transformers
evaluator and never cached, which is the only reason the discrepancy was visible
at all.

**Blast radius: one record.** Checked every tag for duplicates — S2 is the only
model ever re-run under an existing name; all other rungs had unique tags and were
computed fresh. The contaminated record was deleted along with its results folder,
and S2 is queued to re-run properly.

**Fix.** `run_mteb(..., overwrite=True)` is now the **default**. Recomputation costs
~8 minutes per model and eliminates a class of silent wrong answers. Caching was
never worth that trade in a project whose entire method is "re-measure everything
through one code path".

### Finding 25 — the rank-bottleneck diagnosis is confirmed by the honest half of the data

Even with the aggregate contaminated, NanoBEIR is trustworthy, and it recovered
**0.3300 → 0.4353 (+0.105)** purely from removing `consequent_rank=32` at fixed
`d_out=512`. That independently confirms E012 Finding 24: the shared output basis,
not the width, was what destroyed S2. Dense S2 (3.05M) now sits at 0.4353 —
between A4 (2.52M, 0.4251) and S4 (7.41M, 0.4385), i.e. exactly where a 3M model
scaling `d_out` should sit.

**Pattern worth naming.** This is the third time in this session that a *plausible
number* was wrong for a mechanical reason (WDDM spill masquerading as a hang; a
guard that never guarded; now a cache serving stale scores). In each case the
tell was an internal inconsistency — 320 s/it on a model that profiles at 37 ms;
"launched" with a GPU at 0 MiB; two metrics disagreeing about whether anything
changed. Cross-checking two independent signals caught all three; trusting the
headline number would have caught none.


## 2026-07-31 — E014 · Scaling results

All 1 epoch, batch 4096, seed 42, dense consequents unless noted.

| rung | params | d_in | d_out | R | MTEB-14 | NanoBEIR |
|---|---|---|---|---|---|---|
| A6-R16 | 2.25M | 64 | 256 | 16 | 50.53 | 0.4159 |
| A4 (n=3) | 2.52M | 64 | 256 | 32 | 51.24 ± 0.30 | 0.4251 ± 0.0028 |
| **O3 (2 epochs)** | **2.52M** | 64 | 256 | 32 | **51.62** | **0.4335** |
| S6 (rank 256) | 2.66M | 64 | 512 | 32 | 51.07 | 0.4221 |
| S2 | 3.05M | 64 | **512** | 32 | *(re-run pending)* | **0.4353** |
| A6-R64 | 3.06M | 64 | 256 | 64 | 51.00 | 0.4267 |
| S1 | 5.00M | **128** | 256 | 32 | 52.12 | 0.4332 |
| S5 | 6.28M | 64 | 256 | **256** | 51.11 | 0.4314 |
| S4 | 7.41M | **190** | 256 | 32 | 52.26 | 0.4385 |
| S3 | 8.18M | 128 | 512 | 64 | **52.31** | **0.4397** |
| *potion-base-8M* | *7.56M* | — | 256 | — | *52.74* | *0.4421* |

### Finding 26 — retrieval quality tracks *usable embedding rank*, monotonically

Direct confirmation of E012 Finding 24, with the predicted bound verified by SVD on
600 real embeddings:

| k | predicted bound (k+R) | **measured rank** | NanoBEIR |
|---|---|---|---|
| 32 | 64 | **65** | 0.3300 |
| 256 | 288 | **289** | 0.4221 |
| dense | 512 | 416 | 0.4353 |

The bound is accurate to +1 in both low-rank cases (the extra dimension comes from
the pooled-mass term). Dense reaches 416 rather than 512 — that residual is a
property of the trained model, not an architectural cap, since dense has no rank
limit at these dimensions.

So the constructor warning added in E012 predicts the failure correctly, and the
relationship it warns about is smooth rather than a cliff: every dimension of
usable rank buys retrieval.

### Finding 27 — the scaling ordering, with one near-perfectly controlled comparison

At **effectively identical parameter counts**, output width beats rule count:

| config | params | change | NanoBEIR |
|---|---|---|---|
| A6-R64 | 3.06M | R: 32 → 64 | 0.4267 |
| S2 | 3.05M | d_out: 256 → 512 | **0.4353** |

0.3% apart in parameters, 2% apart in retrieval. Combined with the rest of the
table the ordering is: **`d_out` > `d_in` > `R`**, with `R` saturating by ~32–64.

- **`d_in` (table width)** helps with strong diminishing returns: 2.52M → 5.00M
  buys +0.88 MTEB-14; 5.00M → 7.41M buys only +0.14.
- **`R` (rule count)** is done by ~32. S5 spent **3.7M extra parameters** on 8× the
  rules and scored *below* A4 on aggregate (51.11 vs 51.24) at 2.5× the size, with
  rule-usage entropy degrading to 0.898. This was pre-registered as a test of
  Findings 9 and 15; both held.

### Finding 28 — an epoch is worth more than a parameter, for retrieval

`O3` is A4's architecture trained for 2 epochs instead of 1:

| config | params | epochs | NanoBEIR |
|---|---|---|---|
| A4 | 2.52M | 1 | 0.4251 ± 0.0028 |
| **O3** | **2.52M** | **2** | **0.4335** |
| S1 | 5.00M | 1 | 0.4332 |

+0.0083 is **3× A4's seed standard deviation**, so it is real. And O3 at 2.52M
**matches S1 at 5.00M** on retrieval — doubling compute is as effective as doubling
parameters, at half the model size.

**Caveat this places on the whole study:** every rung is a *single* epoch, so all
models here are undertrained. The comparison remains fair — identical budget for
every rung is what makes the ablations attributable — but the absolute numbers sit
below what these architectures reach at convergence, and should not be read as
converged. This is a separate axis from Finding 23: the learning *rate* was already
optimal (raising it hurt), while the number of *passes* was not.

### Where scaling leaves the headline comparison

Best FES config (`S3`, 8.18M) reaches 52.31 / 0.4397 against potion-base-8M's
52.74 / 0.4421 at 7.56M — close on both but now *larger*, so not an efficiency win.
The remaining gap is still Classification, which E010 Finding 18 established is a
data/teacher artefact (A0, with no rule base at all, showed the same 5.7-point
deficit). **Capacity does not close it; a classification training signal would.**
The efficient point remains `S1` at 5.00M — 98.8% of potion-8M's aggregate at 66%
of its parameters.


### Finding 29 — LogTSK reproduces the uniform degeneracy, as predicted

`A7-logtsk` (ℓ1-normalised `-1/Z` instead of softmax, per B1's alternative fix):

| config | H_fire | MTEB-14 | NanoBEIR |
|---|---|---|---|
| A4 (HTSK, calibrated) | 0.010 | 51.24 | **0.4251** |
| A1b (R=1 control) | — | 51.36 | 0.4031 |
| **A7 (LogTSK)** | **0.999** | 51.10 | **0.3877** |

LogTSK's heavier tail holds firing at **H_fire = 0.999** — essentially uniform.
E001 Finding 2 predicted exactly what that implies: `f̄ ≈ 1/R` makes
`u_t = Ā v_t + b̄`, a single averaged expert. The scores confirm it — A7 lands
*below the R=1 control on both metrics*, i.e. it pays for 32 rules, collapses them
into one, and ends up worse than never having had them, because the parameters are
wasted and the averaged expert is a worse single map than a directly-trained one.

Recommendation: **do not use LogTSK here.** HTSK-with-calibration is the only
defuzzification of the three that produces a functioning rule base at this scale.

### Finding 30 — T-tiny: the "as small as possible" point beats the smallest published model

| model | params | MTEB-14 | NanoBEIR |
|---|---|---|---|
| **FES T-tiny** (d_in=32, d_out=256, R=16) | **1,143,483** | **49.91** | **0.4033** |
| potion-base-2M (smallest model on MTEB) | 1,889,792 | 49.62 | 0.3666 |

Beats it on both metrics at **60% of the parameters**: +0.29 aggregate, **+10%
relative retrieval**.

And T-tiny's retrieval (0.4033) *equals* the R=1 control at 2.52M (0.4031) — for
retrieval the rule base reaches the same quality at **2.2× fewer parameters** than
widening the table. That is the `DESIGN.md` §2.2 compression claim holding, but
only on retrieval, which is the consistent shape of every result in this log.

87% of T-tiny is still the vocabulary table.

### Complete R-sweep (ur_weight=1.0, d_in=64, d_out=256)

| R | params | MTEB-14 | NanoBEIR |
|---|---|---|---|
| 1 (control) | 2.52M | 51.36 | 0.4031 |
| 4 | 2.05M | **51.16** | 0.4094 |
| 16 | 2.25M | 50.53 | 0.4159 |
| 32 | 2.52M | 50.96 | 0.4166 |
| 64 | 3.06M | 51.00 | 0.4267 |
| 256 | 6.28M | 51.11 | 0.4314 |

**Retrieval rises monotonically across the whole range; MTEB-14 shows no trend
whatsoever** (51.16 → 50.53 → 51.11 is noise). This is the sharpest single statement
of the project's central result: *rules buy retrieval and nothing else.*

Two details: even **R=4 beats the R=1 control on retrieval at fewer parameters**
(0.4094 at 2.05M vs 0.4031 at 2.52M), so the benefit appears immediately rather
than needing a large base. And **R=4 has the best aggregate of any fuzzy config**
while having the worst retrieval — the cost to non-retrieval task types appears as
soon as rules are added and grows with them.


## 2026-07-31 — E015 · S2 re-run: the cached value was wrong by 3.8 points

Fresh evaluation of dense S2 (`d_in=64, d_out=512, R=32`, 3.05M):

| run | MTEB-14 | NanoBEIR |
|---|---|---|
| dense, **stale cache** (E013) | 47.79 | 0.4353 |
| **dense, fresh** | **51.56** | 0.4300 |

The cached aggregate was **3.77 points low** — it was the broken rank-32 model's
score. Materially wrong, not a rounding issue, and it would have made `d_out`
scaling look actively harmful when it is in fact the best axis.

**Finding 27 survives and strengthens.** The like-for-like comparison at matched
parameters now holds on *both* metrics:

| config | params | change | MTEB-14 | NanoBEIR |
|---|---|---|---|---|
| A6-R64 | 3.06M | R: 32 → 64 | 51.00 | 0.4267 |
| **S2** | **3.05M** | **d_out: 256 → 512** | **51.56** | **0.4300** |

0.3% apart in parameters; `d_out` wins on aggregate (+0.56) and retrieval (+0.008).

### Finding 31 — fixed-seed runs are not bit-reproducible; ~0.005 NanoBEIR of kernel noise

The two dense S2 runs used **the same seed (42) and the same config**, yet returned
NanoBEIR 0.43533 and 0.42998 — a 0.0054 difference. GPU training is not
deterministic by default (non-deterministic reduction kernels, atomics), so
fixed-seed repeats still vary.

Consequences, and they are worth being precise about:

- The three-seed spreads reported in E010 (A4 sd 0.0028, A1b sd 0.0067) **conflate
  seed variation with kernel nondeterminism**. They are still valid as an estimate
  of total run-to-run noise — which is the quantity the significance test needs —
  but they are not purely "seed" effects.
- The headline gap this noise has to clear is **0.0221** (A4 vs A1b), roughly 4×
  this 0.0054 and 8× A4's sd. Finding 21 is unaffected.
- Any future claim resting on a NanoBEIR difference **below ~0.01 should be treated
  as noise** unless replicated. That threshold is now measured rather than assumed,
  and it retroactively justifies not claiming several sub-0.01 differences earlier
  in this log (A2 vs A3, A4 vs A8).

To make runs reproducible would require `torch.use_deterministic_algorithms(True)`
plus `CUBLAS_WORKSPACE_CONFIG`, at a throughput cost. Not done — the noise is
characterised and small relative to the effects claimed.


## 2026-08-01 — E016 · Throughput re-measured on an idle machine

E007 flagged its throughput table as contaminated (two training jobs were sharing
the GPU/CPU, and one pair was non-monotonic). Re-run with the machine genuinely
idle — verified `0 MiB, 0%` before starting.

| config | params | GPU sent/s | CPU sent/s | peak MB |
|---|---|---|---|---|
| R=1 control (d_in=81) | 2,524,121 | 702,761 | **692,641** | 30.1 |
| **R=32 dense (default)** | 2,520,635 | 319,186 | **188,679** | 39.5 |
| R=64 dense | 3,057,211 | 361,015 | 141,466 | 56.6 |
| R=128 dense | 4,130,363 | 314,591 | 69,764 | 89.4 |
| R=512 dense | 10,569,275 | 116,197 | 13,449 | 292.2 |
| R=128 rank32 d_out=512 | 2,344,507 | 338,982 | 103,359 | 74.6 |
| R=128 rank32 d_out=768 | 2,385,467 | 295,375 | 100,455 | 75.3 |

**The suspect non-monotonicity is gone.** E007 measured R=64 as *faster* than R=32
on CPU, which I flagged as impossible. Idle, CPU throughput now falls monotonically
with R (188,679 → 141,466 → 69,764 → 13,449 for R = 32/64/128/512), as it must. That
retroactively confirms flagging the earlier table rather than quoting it.

**Corrections to E007's numbers** (conclusions unchanged, magnitudes revised):

| claim | contended | idle |
|---|---|---|
| R=1 control vs R=32 (Finding 17) | 4.5× faster | **3.67× faster** |
| d_out 256 → 768 throughput cost (Finding 16) | −8% | **−5%** |

**Finding 16 stands, more strongly.** Tripling output width costs 5% throughput and
3.5% parameters. It remains the cheapest axis, and E015 confirmed it is also the
most *effective* axis per parameter.

**Finding 17 stands, slightly softened.** The rule base costs **3.67×** CPU
throughput, not 4.5×, for a retrieval-only gain. Absolute throughput is still firmly
in the static tier — **188,679 sentences/s on CPU** for the default 2.52M config,
orders of magnitude above any transformer — so FES remains cheap in absolute terms
even while the rule base is expensive in relative ones.

**Finding 15 stands.** Low rank still does not buy throughput at large R: R=512
runs at 13,449 (dense) vs 22,514 (rank 32) sent/s — same order, both ~10× worse
than R=32. The antecedent is `O(L·R·d_in)` and does not factor out of the pool, so
rank cannot help it. Combined with E014 Finding 27 (R saturates by ~32–64) and E012
Finding 24 (rank caps embedding rank), the practical guidance is unambiguous:
**keep R ≈ 32, keep the consequent dense, and scale d_out and d_in.**

