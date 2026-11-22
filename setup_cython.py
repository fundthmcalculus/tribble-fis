"""Build the optional Cython forward-pass kernel in place.

    python setup_cython.py build_ext --inplace

The package builds and runs perfectly well without this: :mod:`tribblefis.kernel`
falls back to its NumPy implementation whenever ``tribblefis._fis_kernel`` cannot
be imported. This is a separate script rather than a `pyproject.toml` build
backend precisely so that an ordinary ``pip install`` never needs a C compiler --
the accelerator is opt-in, and a developer who wants it runs one command.

Environment switches:

``TRIBBLEFIS_NO_OPENMP=1``
    Build without OpenMP. The kernel then runs serially, and
    :mod:`tribblefis.kernel` routes large inputs back to NumPy, whose ``exp`` is
    SIMD-vectorized and beats a serial scalar loop.

``TRIBBLEFIS_FAST_MATH=1``
    Add ``/fp:fast`` (MSVC) or ``-ffast-math`` (GCC/Clang). Worth roughly a
    further 1.5x on the reference machine, because it lets the compiler use a
    vectorized ``exp``. It is **not** the default: fast-math also permits
    reassociating the norm folds and contracting multiply-adds, and this
    kernel's contract is that it reproduces the reference arithmetic term for
    term. Measured on the reference machine a fast-math build happened to stay
    bit-identical across every parity case, but "happened to" is not a
    guarantee, so opting in means accepting last-bit drift. (The default build
    passes ``-ffp-contract=off`` on GCC/Clang so FMA-capable default targets
    cannot fuse the folds either; see issue #240.)
"""

import os
import sys

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover - a build-time dependency, not a runtime one
    sys.exit(
        "Cython is required to build the accelerator:\n"
        "    pip install cython setuptools\n"
        "(tribble-fis runs without it, on the NumPy kernel.)"
    )

USE_OPENMP = not os.environ.get("TRIBBLEFIS_NO_OPENMP")
FAST_MATH = bool(os.environ.get("TRIBBLEFIS_FAST_MATH"))

if sys.platform == "win32":
    compile_args = ["/O2"] + (["/openmp"] if USE_OPENMP else []) \
        + (["/fp:fast"] if FAST_MATH else [])
    link_args = []
else:
    # `-ffp-contract=off` is load-bearing, not hygiene. GCC defaults to
    # `-ffp-contract=fast`, so on any toolchain whose default target has FMA
    # (a runner image defaulting to x86-64-v3 -- issue #240) it fuses `a + b - a*b`
    # and `(a + b) / (1 + a*b)` into single-rounded fused multiply-adds. The
    # kernel's contract is that it reproduces the NumPy reference arithmetic term
    # for term, which one rounding fewer breaks in the last bit. Reproduced with
    # `CFLAGS=-march=x86-64-v3`: exactly the probability and einstein folds fail
    # `test_folds_are_bit_identical_given_the_same_exp`, and adding this flag
    # restores them. Skipped under FAST_MATH, which opts in to contraction anyway.
    compile_args = ["-O3"] + (["-fopenmp"] if USE_OPENMP else []) \
        + (["-ffast-math"] if FAST_MATH else ["-ffp-contract=off"])
    link_args = ["-fopenmp"] if USE_OPENMP else []

extension = Extension(
    "tribblefis._fis_kernel",
    sources=["src/tribblefis/_fis_kernel.pyx"],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="tribblefis-kernel",
    ext_modules=cythonize(
        [extension],
        compiler_directives={"language_level": "3"},
        annotate=False,
    ),
    zip_safe=False,
)
