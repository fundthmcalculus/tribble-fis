# Literature Index — Fuzzy Inference Systems for Text Embeddings

Curated during the initial deep dive (2026-07-31). Each entry is tagged with
**why it matters here**. Detailed notes live in the sibling documents:

- `01-small-embedding-models.md` — the competitive landscape and its recipes
- `02-fuzzy-systems-high-dim.md` — the fuzzy-systems math we depend on
- `03-benchmarks.md` — how these models are actually scored

---

## A. Small / static embedding models (our competitive set)

| # | Reference | Why it matters |
|---|---|---|
| A1 | **Model2Vec** — MinishLab. <https://github.com/MinishLab/model2vec> | Distils a sentence-transformer into a *static* token lookup table. Establishes that a bag-of-embeddings model with **zero** contextual computation reaches ~92% of MiniLM. This is the architecture class we are competing in. |
| A2 | **POTION / Tokenlearn blog post** — <https://minishlab.github.io/tokenlearn_blogpost/> | The full recipe behind `potion-base-{2,4,8,32}M`: distil → generate ~1M mean-pooled C4 target embeddings → regress on them → post-hoc re-regularise with frequency weighting, PCA, and SIF. Our token-weighting and PCA steps are lifted from here. |
| A3 | **`minishlab/potion-base-8M`** — <https://huggingface.co/minishlab/potion-base-8M> | 7.56M params, MTEB avg **51.32**. Our primary target to beat on a parameter-matched basis. |
| A4 | **`minishlab/potion-base-4M` / `-2M`** — HF model cards | The genuinely tiny tier. `potion-base-2M` is the smallest model on MTEB (~8 MB). Sets the floor for "as small as possible". |
| A5 | **"Train 400x faster Static Embedding Models with Sentence Transformers"** — Aarsen & Nussbaum, HF blog. <https://huggingface.co/blog/static-embeddings> | *The* practical training recipe for a static embedder trained **from scratch** (no distillation): `EmbeddingBag` + `MultipleNegativesRankingLoss` + `MatryoshkaLoss`, bs=2048, lr=0.2, 1 epoch, RTX 3090, 17.8 h → 89.5% of MiniLM on NanoBEIR. Directly reusable; we adopt the loss, the LR regime, and the dataset list. |
| A6 | **`sentence-transformers/all-MiniLM-L6-v2`** | 22.7M params, 6-layer transformer. The universal "small model" reference point; everything above is quoted as a fraction of it. |
| A7 | **EmbeddingGemma-300M** — Google. <https://arxiv.org/abs/2509.20354> | 308M params (≈100M backbone + 200M embedding table), 768-d with Matryoshka to 128. MTEB English v2 **69.67**. Explicitly *out of our weight class* — recorded as the ceiling reference the user named, not a target. |
| A8 | **LFM2.5-Encoder-230M** — Liquid AI. <https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M> | 229.7M params, 1024-d, 8k context, gated short-conv + GQA hybrid. **Important finding: it is not an MTEB embedding model.** It is a bidirectional *encoder backbone* scored by fine-tuning on GLUE/SuperGLUE (17-task mean 79.29). Not directly comparable to an embedding model without fine-tuning — see `03-benchmarks.md`. |
| A9 | **SwiftEmbed** — <https://arxiv.org/pdf/2510.24793> | Ultra-fast embeddings via static token lookup; corroborates the static-lookup design point and its latency profile. |

## B. Fuzzy inference systems — the parts we build on

| # | Reference | Why it matters |
|---|---|---|
| B1 | **Cui, Wu & Xu, "Curse of Dimensionality for TSK Fuzzy Neural Networks: Explanation and Solutions"**, IJCNN 2021. <https://arxiv.org/abs/2102.04271> | **The single most load-bearing reference.** Shows TSK defuzzification *is* a softmax, and that its logit `Z_r = -Σ_d (x_d-m_rd)²/(2σ_rd²)` grows in magnitude with `D`, saturating the softmax so one rule wins and gradients die. Fixes: **HTSK** (`Z'_r = Z_r / D`) and **LogTSK** (ℓ1-normalise `-1/Z_r`). We operate at D=64–256, so this is mandatory, not optional. |
| B2 | **PyTSK** — Cui & Wu. <https://pytskdocs.readthedocs.io/en/latest/models.html> | Reference implementation and the practitioner's bag of tricks: `AntecedentGMF(high_dim=True)`, KMeans centre init, **Uniform Regularisation** `ℓ_UR = Σ_r (mean_n f_{n,r} − τ)²` to stop "winner-takes-all", DropRule, BatchNorm on consequent inputs, LayerNorm+ReLU after the antecedent. We adopt HTSK, KMeans init, and UR. |
| B3 | **ANFIS** — Jang 1993; <https://en.wikipedia.org/wiki/Adaptive_neuro_fuzzy_inference_system> | The hybrid learning rule (exact least-squares on consequents, gradient on antecedents) that this repo's `consequent-plan.md` already implements for regression. We reuse the *concept* but train end-to-end, because the embedding objective (InfoNCE) is not least-squares. |
| B4 | **Takagi-Sugeno-Kang fusion survey (hierarchical / wide / stacked)** — Inf. Fusion 2024. <https://sciencedirect.com/science/article/abs/pii/S1566253523002932> | Survey of how TSK systems are composed to escape rule explosion. Justifies our choice of a **single wide scatter-partition rule base** over a grid partition. |
| B5 | **Embedded feature selection with sparse TSK rule base** — KBS 2024. <https://sciencedirect.com/science/article/abs/pii/S095070512400443X> | Group-Lasso on inputs *and* rules for sparsity. Our route to pruning the rule base later ("scaling expansion"). |
| B6 | **Adaptive double-parameter softmin TSK for high-dimensional data** — 2025. <https://sciencedirect.com/science/article/abs/pii/S0165011425003215> | Post-HTSK follow-up; a learnable temperature on the t-norm. Directly motivates making the HTSK denominator a **learned scalar** rather than fixed `D`. Queued as an ablation. |
| B7 | **CogniFNN** — <https://arxiv.org/pdf/2009.11485> | Prior art for fuzzy neural nets consuming *word embeddings* (ellipsoidal basis functions + TS consequents) for cognitive-plausibility evaluation. Closest existing "fuzzy system over embeddings" work; notably it *evaluates* embeddings rather than *producing* them. |
| B8 | **"Fusion of fuzzy theories and NLP: a state-of-the-art survey"** — ASOC 2024. <https://sciencedirect.com/science/article/abs/pii/S1568494624005921> | Survey establishing the gap: fuzzy methods in NLP are used for classification, sentiment, and matching — **not** as general-purpose sentence-embedding producers. Supports the novelty claim. |
| B9 | **Multi-task fuzzy clustering TSK for text sentiment** — TALLIP 2021. <https://dl.acm.org/doi/10.1145/3476103> | Existing TSK-over-text work; consumes pre-made features and emits a class label, not a vector. |
| B10 | **Ensemble deep RVFL based on FIS** — <https://arxiv.org/pdf/2406.00801> | Rule-base ensembling as a dimensionality-curse mitigation; alternative to our single wide base. |

## C. Benchmarks & evaluation

| # | Reference | Why it matters |
|---|---|---|
| C1 | **MMTEB / MTEB(eng, v2)** — Enevoldsen et al., ICLR 2025. <https://arxiv.org/abs/2502.13595> | 41 datasets / 7 task types. Also introduces the **downsampled** suites that keep >90% rank-order fidelity at ~2% of the documents — how we make evaluation affordable. |
| C2 | **NanoBEIR** — Zeta Alpha, unified as `NanoBEIREvaluator` in sentence-transformers | 13 BEIR subsets, 50 queries × ≤5k docs each. The fast retrieval signal used by A5. Our per-epoch dev metric. |
| C3 | **`mteb` python package** — <https://sbert.net/docs/sentence_transformer/usage/mteb_evaluation.html> | The runner. Lets us score our model and every baseline through one identical code path, which matters more than absolute numbers. |

---

## Gap statement (the thing we are actually claiming)

Combining B7–B9 with A1–A5: fuzzy inference systems have been applied *on top of*
pretrained embeddings to classify, cluster, or score text, and static-lookup models
have been shown to be viable embedders. **No published work uses a TSK fuzzy
inference system as the embedding function itself.**

The connection that makes it worth trying is in `DESIGN.md`, but stated briefly:
a mean-pooled static embedding model is *algebraically a zero-order, single-rule
TSK system*. Everything from `R = 1` upward is unexplored territory, and the
rule-based structure buys interpretability that a lookup table cannot offer.
