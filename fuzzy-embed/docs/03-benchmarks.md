# Notes: Benchmarks and Evaluation Protocol

**Protocol, as directed 2026-07-31: for comparison models we accept the published
benchmark figures.** No further GPU time goes to re-measuring competitors; it goes
to FES ablations instead. The five small baselines that *were* already measured
locally are kept, because they are strictly more informative than the published
numbers and they let us calibrate the two scales against each other (§1a).

Everything FES-side is still measured through one code path, so rung-to-rung
differences remain attributable.

---

## 1a. Calibration: how our MTEB-14 relates to published 41-task averages

Because we happen to have both numbers for four models, the relationship can be
measured rather than assumed. This is what makes "accept the published figures"
workable.

| model | ours (MTEB-14) | published (41-task) | offset |
|---|---|---|---|
| potion-base-32M | 54.08 | 52.83 | **+1.25** |
| potion-base-8M | 52.74 | 51.32 | **+1.42** |
| all-MiniLM-L6-v2 | 60.52 | ~56.1 | **+4.42** |
| static-retrieval-mrl-en-v1 | 51.02 | (none published) | — |

**The offset is not uniform, and that is the important finding.** It is a tight
+1.25…+1.42 across the static-embedding family but +4.42 for a transformer. So
MTEB-14 is *not* a simple shift of the 41-task average — our subset is relatively
kinder to transformer models (or, equivalently, harsher on static ones). Three
consequences, and we hold to all three:

1. **Within the static family** (our actual competitive set: potion-*,
   static-retrieval-mrl, FES) the offset is stable at ≈ **+1.3** and rank order is
   preserved. Published static numbers can be compared to ours through that offset,
   stated explicitly each time.
2. **Across architecture families**, MTEB-14 comparisons are unreliable. We do not
   use it to compare FES against MiniLM or EmbeddingGemma.
3. **NanoBEIR needs no offset at all** — our figures reproduce published values to
   four decimals (all-MiniLM-L6-v2 0.562318 vs 0.5623; static-retrieval-mrl-en-v1
   0.503167 vs 0.5032). **So any cross-family claim is made on NanoBEIR, not
   MTEB-14.** This is why the retrieval result carries the argument in `LOG.md`.

## 1. MTEB

**MTEB(eng, v2)** (C1) — 41 datasets across 7 task types: Retrieval,
Classification, Clustering, Pair Classification, Reranking, STS, Summarization.
The v2 revision de-duplicates and downsamples relative to v1; MMTEB also ships
downsampled suites that retain **>90% rank-order fidelity at ~2% of original
document counts**, which is the licence we need to evaluate cheaply without
producing misleading rankings.

Running all 41 tasks for ~7 models is not affordable here, so we fix a subset up
front (before seeing any results) and never change it.

### The fixed subset — 14 tasks, all 7 types

| Type | Tasks | Main metric |
|---|---|---|
| Classification | `AmazonCounterfactualClassification` (en), `Banking77Classification`, `EmotionClassification` | accuracy |
| Clustering | `TwentyNewsgroupsClustering.v2`, `StackExchangeClustering.v2` | V-measure |
| Pair classification | `SprintDuplicateQuestions`, `TwitterSemEval2015` | AP |
| Reranking | `AskUbuntuDupQuestions`, `SciDocsRR` | MAP |
| Retrieval | `SciFact`, `NFCorpus`, `ArguAna` | nDCG@10 |
| STS | `STS12`, `STSBenchmark`, `SICK-R` | Spearman |
| Summarization | `SummEval` | Spearman |

Chosen because: all seven types are represented (so the task-type profile in
`01-small-embedding-models.md` §2 is reproducible), the retrieval sets are the
small BEIR ones (SciFact 5k docs, NFCorpus 3.6k, ArguAna 8.7k), and every task
appears in MTEB(eng, v2) so our averages are interpretable against published ones
even though they are not directly equal to them.

**Naming discipline:** our subset average is reported as **`MTEB-14`**, never as
"MTEB average". Any comparison against a published 41-task average is labelled as
approximate.

## 2. NanoBEIR

13 BEIR subsets at 50 queries × ≤5k documents (C2), available as
`NanoBEIREvaluator` in sentence-transformers. Datasets: climatefever, dbpedia,
fever, fiqa2018, hotpotqa, msmarco, nfcorpus, nq, quoraretrieval, scidocs,
arguana, scifact, touche2020.

Used as the **in-training dev metric** (fast enough to run every N steps) and as
a headline retrieval number, because it is the metric A5 reports — giving us one
externally anchored comparison point:

| Model | NanoBEIR nDCG@10 |
|---|---|
| all-MiniLM-L6-v2 | 0.5623 |
| static-retrieval-mrl-en-v1 | 0.5032 |

## 3. Baselines and how each is run

| Model | Params | How embedded | Status |
|---|---|---|---|
| `minishlab/potion-base-2M` | ~1.9M | model2vec via sentence-transformers | **target** |
| `minishlab/potion-base-8M` | 7.56M | same | **target** |
| `sentence-transformers/static-retrieval-mrl-en-v1` | ~32M | StaticEmbedding | **target** (from-scratch static SOTA) |
| `sentence-transformers/all-MiniLM-L6-v2` | 22.7M | transformer, mean pool | reference denominator |
| `google/embeddinggemma-300m` | 308M | native ST, task prompts | ceiling reference, **not a target** |
| `LiquidAI/LFM2.5-Encoder-230M` | 230M | mean pool over last hidden | **caveated** — see §4 |

## 4. The LFM2.5 caveat (important)

The brief named `lfm2.5-230m`. Two distinct models exist and **neither is an MTEB
embedding model**:

- `LiquidAI/LFM2.5-230M` — a generative decoder LM. Benchmarked on GPQA Diamond,
  MMLU-Pro, IFEval, IFBench, Multi-IF, BFCLv3/v4, τ²-Bench, CaseReportBench.
- `LiquidAI/LFM2.5-Encoder-230M` — 229.7M-param bidirectional encoder, 1024
  hidden, 8192 context, gated short-convolution + grouped-query attention.
  Evaluated by **fine-tuning** on 17 GLUE/SuperGLUE/multilingual classification
  tasks: mean **79.29 ± 1.02**, 6th of 14 models (above ModernBERT-base 78.19,
  below mDeBERTa-v3 80.37). No MTEB, no retrieval numbers published.

An untuned bidirectional encoder with mean pooling generally scores *poorly* on
MTEB — that is a fact about the protocol (it has no contrastive objective in its
training), not a statement about the model's quality. So:

- We run it, through the same path, and report it as **"LFM2.5-Encoder-230M
  (mean-pooled, no contrastive fine-tune)"**.
- We do **not** claim to "beat LFM2.5" on the basis of that number. If FES scores
  higher, the honest statement is that FES was trained for this task and the
  encoder was not.
- `EmbeddingGemma-300M` is the meaningful large-model reference, since it *is* an
  embedding model. Published MTEB(eng, v2) = 69.67.

Both remain out of scope as targets: the brief was explicit that the goal is the
small embedding models, and these are 90–120× the FES parameter budget.

### Update 2026-07-31: the large references are quoted, not self-measured

`EmbeddingGemma-300M` was started through our path and abandoned. At batch 96 on
the shared RTX 4080 it ran at **470 s/iteration** — a projected 5+ hours for a
*single* retrieval task, while also slowing the FES ablation ladder by 2.5×
(rung A5c took 49 min instead of ~19). Spending more GPU time on one non-target
reference than on the entire experiment was not a defensible trade, so it was
killed.

Consequence, stated plainly: **EmbeddingGemma's 69.67 in this repo is a `[quoted]`
MTEB(eng, v2) figure, not our measurement, and is therefore not strictly
comparable to our MTEB-14 numbers.** It is present for orientation about the
300M-class ceiling and nothing else. The same applies to `LFM2.5-Encoder-230M`,
which additionally has no published embedding number at all (§4 above).

Every model that FES is actually *compared* to — potion-base-2M/8M/32M,
static-retrieval-mrl-en-v1, all-MiniLM-L6-v2 — **was** measured locally through
the identical code path, and two of those measurements reproduce published
NanoBEIR values to four decimal places (see `LOG.md` E002). The comparisons that
carry the argument are self-measured; only the out-of-scope ceiling references are
quoted.

## 5. Efficiency metrics

Recorded alongside quality, since the whole point of this tier is cost:

- **Params** (excluding nothing — the vocabulary table counts).
- **On-disk size** at fp32 and int8.
- **CPU throughput**: sentences/sec, single thread and all-core, on a fixed 10k
  sentence sample of the MS MARCO corpus, seq-len-capped at 256.
- **GPU throughput**: same, batch 1024.

## 6. Statistical honesty

- Fixed seed (42) for training; `mteb` tasks are deterministic given the model.
- For the ablation ladder, differences under ~0.5 MTEB-14 points are **not**
  treated as meaningful on a single run. Where a conclusion depends on a
  difference that small, re-run with 3 seeds or do not claim it.
- Rule-usage entropy `H(mean f̄)/log R` is reported for every fuzzy config,
  because a good score with 2 active rules out of 32 means the parameter count is
  a fiction.
