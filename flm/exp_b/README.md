# Experiment B — FIS heads on a frozen embedding model

Design rationale and the honest framing of what this experiment is for:
[`../FIS_ON_EMBEDDINGS_PLAN.md`](../FIS_ON_EMBEDDINGS_PLAN.md).

## Execution status

| Path | Status |
|---|---|
| `run_sentiment.py --synthetic` (classification) | **run, passes** — 7 heads, complexity introspection verified |
| `run_sentiment.py --synthetic --synthetic-task regression` | **run, passes** — 6 heads, MAE/R²/Spearman verified |
| `run_sentiment.py --synthetic --atlas` | **run, passes** — profiling path exercised |
| `embed.py` against a real encoder | **not run** — Hugging Face was unreachable from the authoring session |
| `run_sentiment.py` against real SST | **not run** — same reason |

So: the plumbing is verified, the science is not. Every number below the synthetic
line is unmeasured until you run it on a real corpus.

Note the synthetic mode's placeholder texts are literally `"synthetic document N"`,
so `--atlas` under `--synthetic` produces meaningless terms. That is the path
being exercised, not a result.

## Files

| File | Role |
|---|---|
| `data.py` | SST-2 / SST-5 / **SST continuous** / IMDB loaders, plus an offline planted-signal generator |
| `perturb.py` | keyboard-adjacent character noise (substitute / transpose / delete / insert / double) |
| `embed.py` | encode once with a frozen model, cache to `.npz`, MRL prefix widths |
| `heads.py` | the head registry + rule/antecedent-count introspection + metrics |
| `atlas.py` | post-hoc naming of FIS-selected dimensions — the bridge to Experiment A |
| `run_sentiment.py` | the sweep: heads × widths × noise → markdown table |

## Quick start

```bash
uv sync --extra dev

# 1. Plumbing check. No network, no encoder, ~40s.
uv run python flm/exp_b/run_sentiment.py --synthetic

# 2. Cache real embeddings. gte-small needs no auth and is what
#    tests/test_textclassifier.py already uses.
uv run python flm/exp_b/embed.py --dataset sst2 --model gte-small \
    --dims 384 256 128 --noise 0.0 0.1 0.25 --out flm/exp_b/cache

# 3. Sweep.
uv run python flm/exp_b/run_sentiment.py --cache flm/exp_b/cache \
    --dataset sst2 --noise 0.0 0.1 0.25 --top-n 20 --atlas \
    --out flm/exp_b/results_sst2.json

# 4. The graded target -- the framing that actually suits a fuzzy system.
uv run python flm/exp_b/embed.py --dataset sst_cont --model gte-small --dims 384
uv run python flm/exp_b/run_sentiment.py --dataset sst_cont --top-n 20
```

For EmbeddingGemma (768-d, MRL → 512/256/128) pass `--model gemma`; it needs a
Hugging Face token and licence acceptance.

## Heads

| Name | Estimator | Kind |
|---|---|---|
| `linear_probe` | `LogisticRegression` / `RidgeCV` | **the number to beat** |
| `mlp` | `MLPClassifier` / `MLPRegressor` | non-fuzzy non-linear reference |
| `tribble_flat` | `MixtureOfGaussiansFuzzy{Classifier,Regressor}` | flat TSK |
| `tribble_flat_refined` | same, `refine=True` | antecedent refinement (classification only) |
| `tribble_flat_0th` | regressor, `tsk_order="0th"` | constant consequents (regression only) |
| `fuzzy_tree` | `Fuzzy{Classification,Regression}Tree` | short readable rules |
| `hme` | `HierarchicalFuzzyExperts{Classifier,Regressor}` | gated sub-FIS |
| `ruspini` | `RuspiniFuzzyClassifier` | explicit shared linguistic partitions |

Not yet implemented: **Fuzzy Fingerprints** (arXiv:2309.04292), which is the
published prior art for this exact architecture and belongs in the table as a
baseline. See §6 of the plan.

## The knob that matters

`--top-n` caps how many embedding dimensions the fuzzy heads may select. It is the
central variable of the experiment, not a tuning detail: a flat FIS over 768 raw
dimensions is not interpretable at any rule count, and TRIBBLE's
differentiation-based `take_top_features` is the only thing keeping the antecedent
count readable. Sweep it (5, 10, 20, 40) and report accuracy against antecedent
count — that curve *is* the accuracy/interpretability trade-off this experiment
exists to measure.

## Reading the output

The table reports `rules` and `antec` next to accuracy. A head that wins on
accuracy with 400 rules has lost. From the synthetic run, note how the families
differ structurally: `tribble_flat` produces 3 rules of 20 antecedents each (one
per class, ANDed over every selected dimension), while `hme` produces 4 rules of 2
antecedents. Same accuracy, very different readability — and that distinction
survives onto real data.

## What to expect, stated in advance

1. `linear_probe` will be hard to beat. A linear probe on a good frozen encoder is
   a strong baseline, and saying so now is what makes any win credible later.
2. **All** heads will degrade together under noise, because they share a frozen
   encoder. That common-mode degradation is the evidence that robustness has to be
   fixed at the representation layer — i.e. Experiment A — not in the head.
3. The rules will be transparent but **not interpretable**, because `dim_417` has
   no name. This is the expected finding and the whole argument for Experiment A;
   `atlas.py` shows how far post-hoc naming gets you, which is not far.
