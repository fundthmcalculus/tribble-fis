# Experiment B — A TSK FIS Head on a Frozen Neural Embedding Model

**Goal.** Train TRIBBLE's fuzzy inference systems directly on the output of a small,
locally-runnable neural embedding model, and measure the accuracy and interpretability
cost against standard non-fuzzy heads on textual sentiment analysis.

**Why this is the experiment to run first.** It is cheap, it reuses code that already
exists, and — most importantly — it is the *control* that determines whether
[Experiment A](FUZZY_EMBEDDING_PLAN.md) is worth building. See §5: the expected finding is
that a FIS over anonymous embedding dimensions is *transparent but not interpretable*, and
that finding is the argument for A.

Code: [`exp_b/`](exp_b/). Prior art: **this architecture is already published** — see §6.

---

## 1. Status of this directory

The harness in [`exp_b/`](exp_b/) is complete and self-contained, but **it has not been
executed against a real encoder or real SST data.** This session's network egress was
restricted to an allowlist that excluded Hugging Face, so neither model weights nor the SST
dataset could be downloaded. What *was* run here is the synthetic smoke test
(`--synthetic`), which exercises the full plumbing — head fitting, MRL truncation sweep,
perturbation, metrics, reporting — on random embeddings with a planted signal. Treat the
numbers from a real run as unmeasured until you run it. See [`exp_b/README.md`](exp_b/README.md).

## 2. Encoder

Primary: **EmbeddingGemma-300M** — 768-d, ~622MB, 2K context, MTEB English v2 ≈ 69.67, and
crucially **MRL-truncatable to 512 / 256 / 128**. The Matryoshka property makes the
dimension sweep free: no retraining, just prefix slicing, which directly answers "how many
embedding dimensions does a FIS actually need?"

Fallbacks, in order: `thenlper/gte-small` (already used by
`tests/test_textclassifier.py`, so zero setup), then `lfm2`-family or
`granite-embedding-r2` if a second architecture is wanted for a robustness check.

Encoder stays **frozen**. Embeddings are computed once and cached to `.npz`; every head then
trains in seconds. This is what makes the head sweep cheap enough to be thorough.

## 3. Data

| Dataset | Target | Why |
|---|---|---|
| SST-2 | binary | standard, comparable to published numbers |
| SST-5 | 5-class | fine-grained; where fuzzy heads should show an edge |
| **SST continuous** | **regression on [0,1]** | **the interesting one — see below** |
| IMDB | binary, long text | tests behaviour past the 2K context and on multi-sentence input |
| any of the above + character noise | as above | robustness axis |

**Regress the continuous score, not just the bucket.** SST was annotated with a slider
admitting up to 25 sentiment levels, over 215,154 uniquely labeled phrases, and SST-2/SST-5
are *discretizations* of those graded values. Collapsing to 5 classes throws away the one
property of the dataset that plays to a fuzzy system's strength. `MixtureOfGaussiansFuzzyRegressor`
takes a continuous target directly; report MAE and Spearman ρ against the graded score. This
is both a fairer and a more distinctive evaluation than SST-5 accuracy, and it is the
headline result to aim for.

Character noise for the robustness axis comes from `exp_b/perturb.py` (keyboard-adjacent
substitution, transposition, deletion, insertion, doubling), swept over noise rates. Note
the caveat from the literature review: perturbation generators taken from AdvGLUE /
TextBugger are preferable to home-grown ones for anything published, because a home-grown
generator makes it too easy to beat your own noise model. `perturb.py` is for internal
sweeps; swap it before publishing.

## 4. Heads to compare

All on identical splits and identical cached embeddings.

| # | Head | Role |
|---|---|---|
| 1 | Logistic regression / ridge | **the number to beat.** The standard linear probe |
| 2 | MLP (1 hidden layer) | non-linear non-fuzzy reference |
| 3 | `MixtureOfGaussiansFuzzyClassifier` | flat TSK, the repo's workhorse |
| 4 | `MixtureOfGaussiansFuzzyRegressor` | on the continuous SST target |
| 5 | `FuzzyClassificationTree` | short readable rules |
| 6 | `HierarchicalFuzzyExpertsClassifier` | best accuracy in the repo's own concrete/phishing results |
| 7 | `RuspiniFuzzyClassifier` | explicit shared linguistic partitions |
| 8 | Fuzzy Fingerprints | **the published prior art (§6) — implement or cite explicitly** |

Metrics per head: accuracy / macro-F1 (or MAE / Spearman ρ for regression), **rule count**,
**mean antecedents per rule**, fit wall-clock, predict wall-clock. The interpretability
columns matter as much as accuracy — a head that wins by 1 point with 400 rules has lost.

## 5. The expected finding, and why it is the point

A rule like

```
IF dim_417 is High AND dim_22 is Low THEN sentiment = positive
```

is **transparent** — you can read every parameter — but it is **not interpretable**, because
`dim_417` of EmbeddingGemma has no name and no meaning available to a human. Transparency
without semantics buys nothing.

This is the crux. Experiment B is not primarily a bid to beat a linear probe; it is the
demonstration that *the interpretability bottleneck is the representation, not the
inference engine*. TRIBBLE's rules are already readable. What they lack is readable
variables. That is exactly what Experiment A supplies, and framing B this way is both more
honest and a stronger argument than claiming B is interpretable.

**Partial mitigation, and the bridge to A:** `exp_b/atlas.py` names the FIS-selected
dimensions post hoc, by finding the training texts that most strongly activate each one and
extracting their distinctive n-grams. Rules then read `IF <dim_417: "refund, broken,
disappointed"> is High …`. This is the SAE auto-interp move (§1 and §C of the review) with
all of its known weaknesses — explanations that are too broad, and polysemantic dimensions
that resist a single label. Running it here is worthwhile precisely because experiencing
those weaknesses first-hand is the sharpest possible motivation for a hierarchy whose axes
are named *a priori*.

## 6. Prior art — this architecture is not new

**Fuzzy Fingerprints** (arXiv:2309.04292; Springer EUSFLAT/AGOP 2023; arXiv:2605.02665) is
the same architecture: pre-trained encoder embeddings → fuzzified, rank-based class
prototypes → fuzzy-similarity matching at inference, with the authors explicitly claiming to
"bridge the gap between Fuzzy Systems and LLMs." There is also an established fuzzy-BERT
sentiment line (three-step fuzzy BERT; FDiBD Fuzzification–DistilBERT–Defuzzification;
scalable fuzzy-inference ensembles). See §7 of the
[literature review](literature/FLM_LITERATURE_REVIEW.md).

**Experiment B is therefore a replication-plus-comparison, not a contribution.** Its value is
(a) establishing the accuracy/interpretability trade-off on TRIBBLE's specific estimator
family, including the HME, which the prior work does not use; (b) the continuous-target
regression framing of §3, which the prior work does not do; and (c) the negative result of §5
that motivates Experiment A. Claiming novelty for "a FIS on embeddings" would be wrong and
would be caught in review.

## 7. Milestones

- **B0 (≈½ day)** — cache embeddings for SST-2/SST-5/continuous at 768/512/256/128; verify
  MRL truncation degrades gracefully under a linear probe (sanity check on the encoder).
- **B1 (≈1 day)** — run all heads at every width; produce the accuracy × interpretability ×
  cost table.
- **B2 (≈½ day)** — noise sweep; measure per-head degradation. Expect *all* heads to
  degrade together, since they share a frozen encoder — which is itself the evidence that
  robustness must be fixed at the representation layer, not the head.
- **B3 (≈1 day)** — continuous-target regression; MAE and Spearman ρ; render the best
  tree's rules with `render_tree_text`.
- **B4 (≈1 day)** — dimension atlas (§5), then re-render the rules with named dimensions and
  judge honestly whether they became interpretable.
- **B5 (≈1 day)** — implement or faithfully cite Fuzzy Fingerprints as a baseline.

## 8. How to run

```bash
uv sync --extra dev

# plumbing check, no network needed
uv run python flm/exp_b/run_sentiment.py --synthetic

# real run
uv run python flm/exp_b/embed.py --dataset sst2 --model google/embeddinggemma-300m \
    --dims 768 512 256 128 --out flm/exp_b/cache
uv run python flm/exp_b/run_sentiment.py --cache flm/exp_b/cache --dataset sst2
```

See [`exp_b/README.md`](exp_b/README.md) for flags.
