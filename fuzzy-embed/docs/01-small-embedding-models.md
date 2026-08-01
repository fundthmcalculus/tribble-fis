# Notes: The Small Embedding Model Landscape

Purpose: fix the numbers we are measuring ourselves against, and record *how*
those numbers were obtained, because the comparison is only meaningful if the
evaluation path is the same.

---

## 1. The three architecture tiers

| Tier | Mechanism | Cost per token | Examples |
|---|---|---|---|
| **Contextual transformer** | Self-attention over the sequence | O(L²·d) | EmbeddingGemma-300M, MiniLM-L6, LFM2.5-Encoder |
| **Static lookup + pooling** | Table lookup, mean-pool | O(d) | Model2Vec/POTION, `static-retrieval-mrl-en-v1`, SwiftEmbed |
| **Sparse lexical** | Term statistics | — | BM25, SPLADE (learned) |

Our work sits in the **static lookup** tier but adds a per-token nonlinearity, so
it is best described as *static-with-local-experts*. The key economic property of
this tier is preserved: **no attention, no layer stack, embedding a document is a
gather plus a small fixed-size matmul per token.**

## 2. Reference scores

### MTEB (English) averages

| Model | Params | Dim | MTEB avg | Notes |
|---|---|---|---|---|
| EmbeddingGemma-300M | 308M | 768 (MRL→128) | **69.67** (Eng v2) | Ceiling reference. ~100M backbone + ~200M embedding table. |
| all-MiniLM-L6-v2 | 22.7M | 384 | ~56.1 | The canonical "100%" denominator. |
| potion-base-32M | 32M | 512 | **52.83** | Best static model; 94.66% of MiniLM. |
| potion-base-8M | 7.56M | 256 | **51.32** | 91.96% of MiniLM. **Primary target.** |
| potion-base-4M | ~3.7M | 128 | ~50.0 | |
| potion-base-2M | ~1.9M | 64 | ~45–48 | Smallest model on MTEB (~8 MB). **Secondary target — the "as small as possible" tier.** |

`potion-base-8M` per-task-type breakdown (from its model card) — this is the shape
of the profile we need to match, not just the average:

| Task type | Score |
|---|---|
| Classification | 70.34 |
| Clustering | 39.74 |
| Pair classification | 76.62 |
| Reranking | 41.79 |
| **Retrieval** | **31.11** |
| STS | 72.91 |
| Summarization | 25.06 |

**Reading:** static models are strong on classification/STS/pair-classification
(where a bag-of-words topical signal suffices) and weak on **retrieval** and
**clustering** (where word order, negation, and query/document asymmetry matter).
That weak retrieval number is where a per-token nonlinearity has the most room
to help, and it is the axis we should watch.

### NanoBEIR retrieval (nDCG@10)

| Model | Score | Relative |
|---|---|---|
| all-MiniLM-L6-v2 | 0.5623 | 100% |
| static-retrieval-mrl-en-v1 | 0.5032 | 89.5% |

Trained from scratch, no distillation, 17.8 h on one RTX 3090. This is the single
most useful data point in the whole review: it proves a *from-scratch* static
model can reach ~90% of MiniLM on retrieval, so we do not need a distillation
teacher to be competitive.

## 3. Two training recipes, and which we use

### 3a. Distillation route (Model2Vec / POTION)

1. Forward the entire tokenizer vocabulary through a teacher sentence-transformer
   → one static vector per token.
2. Reduce with PCA; the resulting table *is* the model.
3. **Tokenlearn** (POTION only): generate ~1M mean-pooled teacher embeddings over
   C4, then train the static table to minimise cosine distance to them.
4. **Post-training re-regularisation** — the "bag of tricks":
   - frequency weighting with the SIF formula `w = 1e-3 / (1e-3 + p(token))`
     (this beat the earlier Zipf form `w = log(1/rank)`),
   - PCA,
   - final SIF weighting.

Requires a teacher. Cheap, but the ceiling is the teacher.

### 3b. From-scratch contrastive route (`static-retrieval-mrl-en-v1`)

- **Module**: `StaticEmbedding` = `nn.EmbeddingBag` (gather + mean-pool).
- **Tokenizer**: `google-bert/bert-base-uncased` (30,522 wordpieces).
- **Dim**: 1024.
- **Loss**: `MultipleNegativesRankingLoss` (in-batch-negative InfoNCE) wrapped in
  `MatryoshkaLoss` over dims `[32, 64, 128, 256, 512, 1024]`.
- **Data**: 13 English retrieval datasets, ~130M pairs (GooAQ, MS MARCO, SQuAD,
  S2ORC, AllNLI, PAQ, TriviaQA, SWIM-IR, PubMedQA, MIRACL-en, MLDR-en, MR-TyDi-en).
- **Hyperparameters**: batch size **2048**, learning rate **2e-1** (yes, 0.2 —
  sparse embedding tables tolerate and want very large LRs), warmup ratio 0.1,
  **1 epoch**, bf16, `NO_DUPLICATES` batch sampler, `PROPORTIONAL` multi-dataset
  sampler.
- Result: 0.5032 NanoBEIR, 397× faster than MiniLM on CPU.

### Our choice

**3b, with the frequency-weighting and PCA tricks from 3a.** Reasons:

1. It is self-contained and reproducible on the available hardware (one RTX 4080
   Laptop, 12 GB — comparable to the 3090 used for A5, so ~1 epoch of a *subset*
   is a few hours, not days).
2. A distillation target would confound the experiment: we want to know whether
   the **fuzzy layer** helps, which requires an identical objective for the fuzzy
   model and its non-fuzzy ablation. Contrastive-from-scratch gives that cleanly.
3. The huge LR (0.2) applies to the sparse table only. The fuzzy layer
   (antecedents + consequents) is dense and needs a normal LR (~1e-3). **Two
   parameter groups is a hard requirement** — this is the kind of detail that
   silently ruins a run.

## 4. Correction to the brief: LFM2.5-Encoder-230M is not an embedding model

The user asked for comparison against `lfm2.5-230m`. What exists:

- **`LiquidAI/LFM2.5-230M`** — a 230M *generative* decoder LM (GPQA, MMLU-Pro,
  IFEval, BFCL). Not an embedder.
- **`LiquidAI/LFM2.5-Encoder-230M`** — 229.7M-param bidirectional encoder,
  1024 hidden, 8192 context, gated short-conv + grouped-query attention.
  Evaluated by **fine-tuning** on 17 GLUE/SuperGLUE/multilingual classification
  tasks (mean 79.29 ± 1.02, 6th of 14; beats ModernBERT-base 78.19, below
  mDeBERTa-v3 80.37). **No MTEB or retrieval numbers are published.**

So there is no published MTEB score to compare to. Options, in order of honesty:

1. Run `LFM2.5-Encoder-230M` through the same `mteb` path with mean pooling and
   report it as an *un-tuned encoder baseline*, clearly labelled as not what the
   model was built for. (Mean-pooled raw encoders typically score poorly on MTEB
   — expect it to look worse than potion-8M, which is a fact about the evaluation
   protocol, not the model.)
2. Report EmbeddingGemma-300M as the named large reference instead, since it *is*
   an embedding model at a similar size.

We will do **both**, labelled, and state the caveat. Neither is a target — the
user was explicit that the goal is to beat the small embedding models, and both of
these are 30–80× our parameter budget.

## 5. What "as small as possible" costs

Static models spend nearly all parameters on the vocabulary table: `|V| × d`.
At `|V| = 30522`, a 256-d table alone is 7.8M params — already the whole
potion-base-8M budget. So the *only* way to be meaningfully smaller at a given
output dimension is to **stop storing the output dimension per token**:

> keep a narrow per-token table (`|V| × d_in`, `d_in ≪ d_out`) and expand
> `d_in → d_out` with a small shared nonlinear map.

That is precisely what a TSK rule base is: `R` local linear maps plus a soft
gate. This is the parameter-efficiency argument for the whole project, and it is
developed formally in `../DESIGN.md` §3.
