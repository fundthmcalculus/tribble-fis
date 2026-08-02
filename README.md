# Tribble FIS

Building FIS using a consequent first approach.

## Optional compiled kernel

The forward pass (`tsk_firing_strengths`, and therefore every prediction and
every antecedent-refinement fitness evaluation) has an optional Cython
implementation. It is not built by default — `pip install` needs no C compiler,
and everything works on the NumPy path without it.

To build it in place:

```bash
pip install cython setuptools
python setup_cython.py build_ext --inplace
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
