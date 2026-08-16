# Tribble FIS

Building FIS using a consequent first approach.

## Installation

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

To install with optional dev dependencies (testing and benchmarking):

```bash
uv sync --extra dev
```

## ANFIS

`tribblefis.anfis` implements the canonical Jang (1993) ANFIS network:
grid-Cartesian rules, product t-norm, closed-form consequent least squares
alternated with a vectorized batch gradient step on every premise parameter.
It is a different rule structure from the rest of the package (see
`docs/anfis-design.md` for why, and for a measured comparison against
`gaussian_regressor.MixtureOfGaussiansFuzzyRegressor`), so it lives
alongside it rather than inside it.

```python
from tribblefis.anfis import ANFISRegressor

reg = ANFISRegressor(n_terms=3, n_epochs=200, learning_rate=0.05)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(reg.describe_rules()[0])  # "IF x0 is term_0 AND x1 is term_1 THEN y = ..."
```

Rules are the Cartesian product of per-feature terms (`n_terms ** n_features`),
so this is the right tool for a handful of features with a few terms each --
past that, `fit` raises `RuleExplosionError` and points at the mixture
regressor, whose implicit rule base doesn't grow that way.

## Optional compiled kernel

The forward pass (`tsk_firing_strengths`, and therefore every prediction and
every antecedent-refinement fitness evaluation) has an optional Cython
implementation. It is not built by default — `uv sync` needs no C compiler,
and everything works on the NumPy path without it.

To build it in place:

```bash
uv run --with cython --with setuptools python setup_cython.py build_ext --inplace
```

Nothing else changes: `tribblefis.kernel` picks the compiled kernel up
automatically for models it can represent exactly (all-Gaussian, with every
feature carrying every label) and falls back silently otherwise. Output is
bit-identical either way.

Measured on 24 cores, against the same code without the extension: 3.8x on a
small forward pass, 6.7x on a large one, 2.5x on an end-to-end classifier
refinement. See `benchmarks/README.md` for the full table and how to reproduce
it.

`TRIBBLEFIS_NO_OPENMP=1` builds without threading, `TRIBBLEFIS_FAST_MATH=1`
trades the bit-exactness guarantee for roughly a further 1.5x, and
`TRIBBLEFIS_NUM_THREADS` overrides the thread count at runtime.

## Optional GPU backend

`tribblefis.gpu` runs the same forward pass on a Torch device. PyTorch is not a
dependency; install it to enable the backend.

```python
from tribblefis import gpu, kernel

compiled = kernel.compile_model(model, list(X.columns))
handle = gpu.TorchFIS(compiled, compiled.feature_matrix(arrays), norms)
strengths = handle.firing_strengths()             # (n_samples, n_labels)
```

It is **never** chosen automatically. CUDA's `exp` differs from libm's by about
an ULP, so a silent substitution would change results everywhere; ask for it with
`kernel.firing_strengths(..., backend="torch")` and you are opting into that.

Measured against the 24-thread CPU kernel on an RTX 4080 Laptop: 1.86x on a
million-sample forward pass in float64, 3.88x in float32, 4.91x for repeated
candidate evaluation. Batching candidates is worth about 1.15x on top — the
device is already saturated by one candidate — so it is a convenience, not the
speedup.
