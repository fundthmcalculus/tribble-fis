# FES — Fuzzy Embedding System

A small text-embedding model whose forward pass **is** a first-order
Takagi–Sugeno–Kang fuzzy inference system.

The starting observation: a static embedding model (Model2Vec / POTION /
`static-retrieval-mrl-en-v1`) computes `e = mean_t E[t]`, which is algebraically a
*one-rule, zero-order* TSK system — one rule, firing strength identically 1, a
lookup for a consequent. Every static embedder in the literature sits at `R = 1`.
This project asks what happens above that.

```
token ids
  → F[t]              narrow learned feature table      (fuzzification)
  → f̄_r(v_t)          HTSK firing over R fuzzy rules    (antecedent)
  → Σ_r f̄_r(A_r v+b_r) mixture of R local linear experts (TSK consequent)
  → Σ_t a_t u_t        learned SIF-style weighted pool    (defuzzification)
  → L2 normalise
```

Each rule *should* be readable: *IF the token feature is near prototype `m_r`
(within per-dimension tolerance `σ_r`) THEN contribute `A_r v + b_r`.* In practice
it is not — see the refutation below.

## Why this could be worth it

At `|V| = 30522`, a static table at 256 dimensions costs **7.8M parameters for the
table alone** — the whole `potion-base-8M` budget. FES keeps a narrow `d_in = 64`
per-token table and shares the `d_in → d_out` expansion across the vocabulary as a
rule base, so a 256-d output costs **~2.5M parameters**. The vocabulary table
stops being the entire model.

That was the motivating argument. **It was tested and it did not hold**, at least
not on aggregate:

| claim | verdict |
|---|---|
| **Compression** — a rule base beats a wider table at equal parameters | **refuted on aggregate** (-0.12, p=0.75); survives on retrieval (+5.5%, p=0.05) |
| **Expressivity** — context-conditioned rules break the bag-of-words ceiling | **refuted**, two mechanisms |
| **Interpretability** — the rule base is readable | **refuted**; rules decode to nothing, contribution spread 3.1% |

What the model *does* deliver is competitive small-scale performance and a real
retrieval-specific gain. What it does not deliver is any benefit attributable to
the inference being *fuzzy*.

## Results in one table

| Tier | FES | Competitor | Verdict |
|---|---|---|---|
| ~1M | **T-tiny 1.14M** — 49.91 / 0.4033 | potion-base-2M 1.89M — 49.62 / 0.3666 | **Wins both, at 60% size** |
| ~7.5M | **S4 7.41M** — 52.26 / 0.4385 | potion-base-8M 7.56M — 52.74 / 0.4421 | 99% of both; wins 5 of 7 task types |

*(MTEB-14 / NanoBEIR nDCG@10, all self-measured through one harness.)*

**And the honest caveat**: of the three distinctive claims motivating the fuzzy
framing, all three were refuted. What survives is a replicated, retrieval-specific
gain (+5.5%, complete seed separation, p ≤ 0.05) from what is structurally a
mixture-of-experts — with the *fuzziness itself* measurably irrelevant across a 50x
range of routing softness. Full accounting in `RESULTS.md`.

## Layout

```
RESULTS.md                 START HERE - the consolidated write-up
FINDINGS.md                all 31 findings indexed, plus the mistakes and what caught them
LOG.md                     raw chronological record, 16 entries, written as it happened
DESIGN.md                  architecture, with refuted claims marked in place
docs/00-literature-index.md  22 references, tagged by why they matter
docs/01-small-embedding-models.md  the competitive set and its training recipes
docs/02-fuzzy-systems-high-dim.md  the fuzzy math this depends on
docs/03-benchmarks.md      evaluation protocol, offset calibration, the LFM2.5 caveat
fuzzyembed/model.py        the FES forward pass
fuzzyembed/data.py         the public training mix (4,115,821 pairs, 11 sources)
fuzzyembed/train.py        contrastive training (MNRL + Matryoshka + UR + calibration)
fuzzyembed/evaluate.py     the frozen MTEB-14 subset + NanoBEIR
scripts/                   smoke, baselines, ladder, rule inspection, cost, plots
results/                   records.jsonl, table.md, figures
tests/                     28 tests
```

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch --torch-backend=cu126
uv pip install --python .venv/Scripts/python.exe -e . accelerate pytest
```

## Use

```bash
.venv/Scripts/python -m pytest tests/ -q         # 28 correctness tests
.venv/Scripts/python scripts/smoke.py            # end-to-end on a tiny slice
.venv/Scripts/python scripts/eval_baselines.py   # measure the competition first
.venv/Scripts/python scripts/run_experiments.py --only A1b-ctrl-matched,A4-no-ur
.venv/Scripts/python scripts/report.py           # regenerate results/table.md
.venv/Scripts/python scripts/inspect_rules.py artifacts/A4-no-ur

# Do NOT run two trainings at once on a 12 GB GPU: VRAM exhaustion spills to
# WDDM shared memory with no OOM error, a silent ~600x slowdown.
```

## Two things worth knowing before you touch the model

**1. The temperature has two failure modes at init, and neither survives contact
with learned σ.** TSK defuzzification with Gaussian MFs and a product t-norm *is*
`softmax(Z)` with `Z_r = -Σ_d (v_d-m_rd)²/(2σ_rd²)`, and `|Z| = O(D)`. At random
init with fixed σ this saturates (`f̄ → one-hot`, dead antecedent gradients — the
Cui/Wu/Xu 2021 result, reproduced in `test_htsk_prevents_softmax_saturation`).
Raw HTSK (`τ = D`) overshoots the other way to `f̄ → 1/R`, which makes
`u_t = Ā v_t + b̄` — a single linear map. **Both ends of the axis give one
effective expert**, hence `calibrate_temperature()`.

**But ablation A3 showed none of this is load-bearing end-to-end**: the plain
product t-norm matched HTSK (50.86/0.4181 vs 50.96/0.4166). τ and σ are redundant —
dividing the exponent by τ equals scaling every σ by `√τ` — and trainable
`log_sigma` absorbs the difference. Confirmation: calibrated τ is **1.195 at both
D=64 and D=128**, so it does not vary with `D`. Treat HTSK as init conditioning,
not as the mechanism. `LOG.md` Findings 7 and 9.

**2. `rule_entropy` and `firing_entropy` are different metrics and you need both.**
`rule_entropy` (usage) answers "are all rules used somewhere?"; `firing_entropy`
(per-token) answers "is the inference actually fuzzy?". A saturated model is a
perfectly balanced *hard router*: usage entropy 1.0, firing entropy ~0. Reporting
only the first hides the failure.

## Status: complete

33 evaluated configurations, 5 baselines, 28 tests, 31 findings. All comparisons
self-measured through one code path; two NanoBEIR figures reproduce published
values to four decimals, which validates the harness independently.

Read `RESULTS.md` for the full account, `FINDINGS.md` for the indexed findings
(including every mistake and what caught it), `LOG.md` for the chronology.

**Note on the brief:** `lfm2.5-230m` is not an embedding model. `LFM2.5-230M` is a
generative decoder; `LFM2.5-Encoder-230M` is an encoder backbone scored by
fine-tuning on GLUE/SuperGLUE, with no published MTEB or retrieval numbers.
`EmbeddingGemma-300M` is the honest large-model reference. Both are 30-100x the FES
budget and are references, not targets. See `docs/03-benchmarks.md` 4.
