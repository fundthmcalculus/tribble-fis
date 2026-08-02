# tribble-fis performance benchmarks

A small, deterministic suite for the fuzzy-inference hot paths, used as the
receipt for every performance change in this repository. The rule is simple:
an optimization may move the **time** column and must not move the
**checksum** column.

## Running

```bash
# time everything and store the numbers
python -m benchmarks.bench -o benchmarks/results/mine.json --label "my change"

# compare against a stored run (prints speedups, fails on a changed checksum)
python -m benchmarks.bench --compare benchmarks/results/baseline.json

# just the forward-pass workloads
python -m benchmarks.bench -k forward

# find out where the time actually goes
python -m benchmarks.bench -n refine-classifier --profile
```

`--compare` exits non-zero if any checksum moved, so it can be wired into CI.

The compiled kernel, when built, is picked up automatically. To measure without
it, or to measure it serially:

```bash
TRIBBLEFIS_NUM_THREADS=1 python -m benchmarks.bench -k forward   # compiled, one thread
```

## Workloads

| workload | what it measures |
|---|---|
| `forward-small` | `tsk_firing_strengths`, 1k x 8 x 3 x 3 — per-call Python overhead dominates |
| `forward-wide` | 2k x 40 features x 6 labels x 4 MF — many small array ops |
| `forward-large` | 50k x 20 x 8 x 4 — per-element arithmetic dominates |
| `forward-prob` | 20k x 20 x 8 x 4 under `probability` norms (the smooth family used by the analytic-gradient refinement) |
| `predict-large` | `MixtureOfGaussiansFuzzyClassifier.predict_proba`, i.e. the kernel plus the deployed estimator's overhead |
| `refine-classifier` | end-to-end `refine_classifier_antecedents` — the training cost, ~1.3k fitness evaluations |
| `refine-classifier-wide` | the same, at a size anyone would deploy: 4k x 20 features x 6 labels x 3 MF, 720 free parameters |
| `forward-huge-cpu` / `-gpu64` / `-gpu32` | 1M-sample forward pass, data resident, CPU vs Torch |
| `batch-candidates-cpu` / `-gpu` / `-gpu-seq` | 64 candidate parameter vectors, the shape a population search evaluates |

`refine-classifier` is small so it catches regressions in under a second, but at
that size the forward pass is only about a fifth of the work — the rest is
SciPy's L-BFGS-B machinery over 48 tiny sub-problems. `refine-classifier-wide`
inverts that ratio, which is why both exist: an optimization that helps only one
regime should be visibly an optimization for only one regime.

The forward-pass workloads pass a pre-extracted `feature_arrays` mapping,
matching what the refinement path already does, so they measure the kernel and
not pandas column lookup.

## Baseline (`results/baseline.json`)

Measured at `origin/main` (`b7d25c5`), Python 3.12.3 / NumPy 2.4.6, Windows 11,
13th-gen mobile i9 + RTX 4080 Laptop.

| workload | min | median |
|---|---|---|
| forward-small | 472 us | 485 us |
| forward-wide | 9.91 ms | 10.50 ms |
| forward-large | 164.98 ms | 177.81 ms |
| forward-prob | 71.32 ms | 73.28 ms |
| predict-large | 99.91 ms | 101.15 ms |
| refine-classifier | 625.08 ms | 634.40 ms |

## Progress against that baseline

Each row is `min` time; every checksum is unchanged from the baseline, so these
are like-for-like.

| workload | baseline | + compiled model | + Cython kernel | + incremental fitness | total |
|---|---|---|---|---|---|
| forward-small | 472 us | 474 us | 123 us | 119 us | **3.97x** |
| forward-wide | 9.91 ms | 9.72 ms | 1.87 ms | 1.84 ms | **5.39x** |
| forward-large | 164.98 ms | 164.42 ms | 24.69 ms | 24.20 ms | **6.82x** |
| forward-prob | 71.32 ms | 76.59 ms | 8.36 ms | 8.37 ms | **8.52x** |
| predict-large | 99.91 ms | 99.21 ms | 31.09 ms | 30.47 ms | **3.28x** |
| refine-classifier | 625.08 ms | 537.49 ms | 247.12 ms | 169.49 ms | **3.69x** |
| refine-classifier-wide | — | — | 3.701 s | 670.90 ms | **5.52x** |

`refine-classifier-wide` was added with the incremental-fitness work, so its
"before" is measured at the Cython commit rather than at the original baseline.
The forward workloads are untouched by that change; their last two columns differ
only by measurement noise.

Measured with the default (non-fast-math) OpenMP build on 24 cores. A
`TRIBBLEFIS_FAST_MATH=1` build is roughly a further 1.5x on the forward
workloads; see `setup_cython.py` for why it is not the default.

## GPU

Skipped with a recorded reason when PyTorch or a CUDA device is absent, so a
missing row never looks like an unrun one. Each `-cpu` row is the same shape,
seed and timing boundary as the `-gpu*` rows below it.

| workload | min | vs CPU |
|---|---|---|
| forward-huge-cpu (1M x 20 x 8 x 4) | 351.24 ms | — |
| forward-huge-gpu64 | 188.72 ms | 1.86x |
| forward-huge-gpu32 | 90.40 ms | 3.88x |
| batch-candidates-cpu (64 candidates) | 67.95 ms | — |
| batch-candidates-gpu (batched) | 13.85 ms | 4.91x |
| batch-candidates-gpu-seq (one at a time) | 15.83 ms | 4.29x |

The last two rows are the point of keeping both: **batching is worth ~1.15x, not
a multiplier.** The device is already saturated by a single candidate at this
size, and what batching saves is 64 parameter uploads.

Note also that the CPU rows are the thermally variable ones — the 24-thread
kernel ranged 351–435 ms across repeats of the same input while the GPU rows
held within 1%. The ratios above use the CPU's best run.

The GPU backend is **never** selected by `backend="auto"`: CUDA's `exp` differs
from libm's by about an ULP, so a silent substitution would move every checksum
in this table.

## Where the time goes at baseline

`--profile` on `forward-large` (0.169 s total):

```
ncalls  tottime  cumtime  function
   640    0.157    0.157  gauss_data.py:121(GaussianMembership.evaluate)
   640    0.006    0.006  gauss_math.py:346(t_conorm)
   160    0.002    0.002  gauss_math.py:313(t_norm)
```

93% of a large forward pass is inside `GaussianMembership.evaluate`. Each call
allocates roughly five 50k-element temporaries (`x - mu`, `/sigma`, `**2`,
`* -0.5`, `exp`) for one membership function, so the kernel is bandwidth-bound
on temporaries rather than on useful arithmetic.

`--profile` on `refine-classifier` (0.954 s total under the profiler):

```
cumtime  function
  0.936  scipy minimize (L-BFGS-B, 96 sub-problems)
  0.861    refine.fitness
  0.654      refine._classifier_proba
  0.598        gauss_math.tsk_firing_strengths
  0.356          gauss_data.GaussianMembership.evaluate   (63 600 calls)
  0.166    refine.apply_gaussian_params                   (64 778 namedtuple _replace)
  0.045    pandas DataFrame.__getitem__                   (10 648 calls)
```

Three separable costs, which is what the later PRs in this stack attack:

1. **the kernel itself** — `evaluate` plus the norm folds, ~63% of training
   time, and structurally the same work as `forward-large`;
2. **the per-evaluation model rebuild** — `apply_gaussian_params` reconstructs
   the whole immutable `NamedTuple` tree on every one of the ~1.3k fitness
   calls, ~17%;
3. **residual pandas lookups** — `_classifier_proba` calls
   `tsk_firing_strengths` without the `feature_arrays` mapping, so it
   re-extracts unchanging columns on every call.
