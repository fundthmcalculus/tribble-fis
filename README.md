# Tribble FIS

Building FIS using a consequent first approach.

## Cross-term exclusion (second-stage admissibility reduction)

Each label gets one rule, and that rule is a conjunction *of disjunctions*:

```
IF (x is X1 OR X2 OR X4) AND (y is Y1 OR Y3) THEN A
```

The per-feature conorm loses which term fired, so the rule admits the entire
outer product — `X1&Y1, X2&Y1, … X4&Y3`. If the data says `X1&Y3` is really
class `B`, rule `A` claims it anyway, and no re-fit of `X1` or `Y3` helps: both
terms are individually right for `A`. The defect is in the combination.

`exclude_cross_terms=True` mines those cells from the training data and appends
their negation to the rule that over-claims:

```python
clf = MixtureOfGaussiansFuzzyClassifier(n_gaussians=2, exclude_cross_terms=True)
clf.fit(X, y)

from tribblefis.exclusion import describe_exclusions
print(describe_exclusions(clf.model_))
# RULE A:
#   AND NOT (x is mf1 AND y is mf0)   -- mostly B; n=51, 0% really A
#   AND NOT (x is mf0 AND y is mf1)   -- mostly B; n=28, 0% really A
```

Each clause names one cell and attaches to one parent rule, so only that rule's
firing drops and only inside that cell — `X1` keeps firing for `A` alongside
`Y1`. The blamed class is not boosted; it wins the argmax by default.

Off by default. Measured over 36 held-out cases it is +1.9 points at
`n_gaussians=2` and +3.4 at `n_gaussians=3`, and was never worse in any case —
but on half the datasets tested it correctly mines nothing, and with one
membership function per feature-label there is no outer product to reduce and it
is an exact no-op. See `docs/cross-term-exclusion-evaluation.md`.

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
