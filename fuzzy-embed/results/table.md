# Results

All figures below are **self-measured** through `fuzzyembed/evaluate.py` on the
frozen 14-task MTEB subset (`MTEB-14`, all 7 task types) plus NanoBEIR. They are
*not* comparable to published 41-task MTEB averages; see `docs/03-benchmarks.md`.

`H_rule` = rule-usage entropy (are all rules used?). `H_fire` = per-token firing
entropy (is the inference actually fuzzy?). Both normalised to [0,1]. A strong
score at low `H_rule` means the parameter count is a fiction.

## Baselines (the yardstick)

| model                      | params | MTEB-14 | Classi | Cluste | PairCl | Rerank | Retrie | STS  | Summar | NanoBEIR | H_rule | H_fire |
|----------------------------|--------|---------|--------|--------|--------|--------|--------|------|--------|----------|--------|--------|
| all-MiniLM-L6-v2           | 22.71M | 60.52   | 61.5   | 48.8   | 81.2   | 75.3   | 48.8   | 77.3 | 30.8   | 0.5623   | -      | -      |
| potion-base-32M            | 32.30M | 54.08   | 63.4   | 34.8   | 75.6   | 64.5   | 42.1   | 68.4 | 29.7   | 0.4637   | -      | -      |
| potion-base-8M             | 7.56M  | 52.74   | 61.2   | 35.5   | 73.7   | 63.1   | 38.9   | 67.4 | 29.2   | 0.4421   | -      | -      |
| static-retrieval-mrl-en-v1 | 31.25M | 51.02   | 54.6   | 27.3   | 72.6   | 63.0   | 44.6   | 66.4 | 28.6   | 0.5032   | -      | -      |
| potion-base-2M             | 1.89M  | 49.62   | 55.1   | 35.2   | 70.9   | 59.5   | 29.2   | 65.9 | 31.4   | 0.3666   | -      | -      |

## FES ablation ladder

| model                       | params | MTEB-14 | Classi | Cluste | PairCl | Rerank | Retrie | STS  | Summar | NanoBEIR | H_rule | H_fire |
|-----------------------------|--------|---------|--------|--------|--------|--------|--------|------|--------|----------|--------|--------|
| FES-A0-static-256           | 7.91M  | 52.49   | 55.5   | 39.9   | 72.7   | 63.4   | 38.7   | 69.1 | 28.1   | 0.4284   | -      | -      |
| FES-S3-balanced             | 8.18M  | 52.31   | 55.1   | 38.6   | 70.6   | 63.4   | 41.7   | 67.2 | 29.7   | 0.4397   | 0.944  | 0.003  |
| FES-S4-potion8M-matched     | 7.41M  | 52.26   | 53.2   | 38.6   | 70.7   | 63.3   | 39.6   | 67.8 | 32.6   | 0.4385   | 0.967  | 0.002  |
| FES-S1-table-128            | 5.00M  | 52.12   | 53.6   | 38.6   | 70.4   | 62.9   | 40.0   | 67.4 | 31.9   | 0.4332   | 0.970  | 0.004  |
| FES-O3-2epoch               | 2.52M  | 51.62   | 53.1   | 38.1   | 70.7   | 63.1   | 38.8   | 67.5 | 30.0   | 0.4335   | 0.981  | 0.008  |
| FES-S2-wide-out-512         | 3.05M  | 51.56   | 54.2   | 38.0   | 69.9   | 62.4   | 39.4   | 66.3 | 30.7   | 0.4300   | 0.945  | 0.012  |
| FES-A1b-ctrl-matched        | 2.52M  | 51.44   | 52.5   | 40.2   | 69.1   | 62.6   | 36.6   | 68.3 | 30.9   | 0.4033   | -      | -      |
| FES-O1-lr-dense-8e-3        | 2.52M  | 51.41   | 52.8   | 38.6   | 69.7   | 62.5   | 39.7   | 66.5 | 30.0   | 0.4157   | 0.978  | 0.006  |
| FES-A4-no-ur                | 2.52M  | 51.38   | 52.9   | 37.9   | 69.2   | 62.3   | 39.6   | 66.5 | 31.3   | 0.4220   | 0.970  | 0.010  |
| FES-A8-fuzzy-anchor         | 2.52M  | 51.25   | 52.8   | 38.5   | 68.5   | 62.4   | 39.6   | 65.9 | 31.0   | 0.4200   | 1.000  | 0.503  |
| FES-A6-R4                   | 2.05M  | 51.16   | 52.5   | 38.8   | 69.6   | 62.3   | 37.7   | 66.9 | 30.4   | 0.4094   | 0.999  | 0.008  |
| FES-S5-manyrules-rank       | 6.28M  | 51.11   | 53.4   | 37.0   | 70.0   | 61.8   | 39.8   | 66.5 | 29.2   | 0.4314   | 0.898  | 0.003  |
| FES-A7-logtsk               | 2.52M  | 51.10   | 52.0   | 40.4   | 69.1   | 62.3   | 34.8   | 68.0 | 31.1   | 0.3876   | 1.000  | 0.999  |
| FES-S6-rank-bottleneck-demo | 2.66M  | 51.07   | 52.6   | 38.4   | 69.6   | 62.1   | 37.9   | 66.9 | 30.0   | 0.4221   | 0.969  | 0.006  |
| FES-A6-R64                  | 3.06M  | 51.00   | 52.4   | 38.4   | 69.1   | 62.2   | 39.0   | 66.1 | 29.8   | 0.4267   | 1.000  | 0.377  |
| FES-A2-fes-s                | 2.52M  | 50.96   | 51.9   | 38.6   | 68.8   | 62.3   | 39.4   | 66.0 | 29.7   | 0.4166   | 0.999  | 0.208  |
| FES-A3-product-tnorm        | 2.52M  | 50.86   | 51.7   | 38.6   | 68.5   | 62.5   | 39.1   | 65.9 | 29.9   | 0.4181   | 0.997  | 0.141  |
| FES-A1-lowrank-ctrl         | 2.00M  | 50.82   | 51.6   | 40.4   | 68.5   | 61.9   | 35.8   | 67.8 | 29.8   | 0.3841   | -      | -      |
| FES-O2-lr-dense-3e-2        | 2.52M  | 50.72   | 51.9   | 38.7   | 69.5   | 62.1   | 38.3   | 66.7 | 27.9   | 0.4190   | 0.968  | 0.002  |
| FES-A6-R16                  | 2.25M  | 50.53   | 52.4   | 38.5   | 69.1   | 61.8   | 38.9   | 65.9 | 27.1   | 0.4159   | 0.999  | 0.081  |
| FES-T-tiny                  | 1.14M  | 49.91   | 49.6   | 37.8   | 68.1   | 61.0   | 37.6   | 64.9 | 30.3   | 0.4033   | 1.000  | 0.136  |
| FES-A5c-anchored            | 2.52M  | 49.19   | 48.0   | 39.5   | 67.5   | 60.7   | 33.3   | 66.0 | 29.3   | 0.3596   | 1.000  | 0.980  |
| FES-S2-rank32-BROKEN        | 2.09M  | 47.79   | 46.1   | 39.4   | 65.5   | 59.5   | 27.4   | 65.3 | 31.4   | 0.3300   | 0.972  | 0.011  |
| FES-A5c-fes-c               | 2.52M  | 43.34   | 42.7   | 36.9   | 58.1   | 54.0   | 20.9   | 60.7 | 30.0   | 0.2676   | 0.645  | 0.000  |

## Published figures for the comparison models

Accepted as published rather than re-measured (directed 2026-07-31).

| model | params | pub. MTEB avg | pub. NanoBEIR | source | scope |
|---|---|---|---|---|---|
| potion-base-2M | ~1.9M | ~45-48 | - | MTEB 41-task | target |
| potion-base-8M | 7.56M | 51.32 | - | MTEB 41-task | target |
| potion-base-32M | 32M | 52.83 | - | MTEB 41-task | target |
| static-retrieval-mrl-en-v1 | ~32M | - | 0.5032 | HF blog | target |
| all-MiniLM-L6-v2 | 22.7M | ~56.1 | 0.5623 | MTEB 41-task / HF blog | reference |
| embeddinggemma-300m | 308M | 69.67 | - | MTEB(eng, v2) | NOT a target |
| LFM2.5-Encoder-230M | 230M | 79.29 | - | 17-task GLUE/SuperGLUE, fine-tuned | NOT an embedding model |

### Reading these against our numbers

Our MTEB-14 runs about **+1.3** higher than the published
41-task average *within the static-embedding family* (measured: +1.25 on
potion-32M, +1.42 on potion-8M). It is **+4.42** on all-MiniLM-L6-v2, so the
offset is family-specific and MTEB-14 must not be used for cross-family
comparison. **NanoBEIR needs no offset** — ours reproduces published values to
four decimals — so every cross-family claim is made on NanoBEIR.
See `docs/03-benchmarks.md` §1a.

Equivalently, to place an FES MTEB-14 score on the published static scale,
subtract ~1.3.

Best FES rung `FES-A0-static-256` (7.91M): MTEB-14 52.49 → **~51.2 on the published static scale**, NanoBEIR 0.4284.
