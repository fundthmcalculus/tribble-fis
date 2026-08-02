# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Fused C kernel for the zeroth-order TSK forward pass.

The NumPy implementation in :mod:`tribblefis.kernel` is already at NumPy's
floor: profiling showed a large forward pass is ~93% ``np.exp``, and ``np.exp``
on float64 runs at ~280 M elem/s single-threaded on the reference machine, which
is almost exactly the measured runtime. Rewriting the same expression in NumPy a
different way cannot beat that. Two things can:

* **fusion** -- the NumPy path makes six separate passes over an
  ``(rows, K*L)`` block (subtract, divide, square, scale, exp, mask) plus one
  pass per conorm step. This does all of it in registers, one sample at a time,
  so nothing but the ``(n, L)`` output is ever written to memory;
* **threads** -- every sample is independent, so the sample loop is a
  ``prange``. NumPy's ufuncs are single-threaded, and the reference machine has
  24 cores.

Layout matches :class:`tribblefis.kernel.CompiledFIS` exactly: ``mu``, ``sigma``
and ``active`` are ``(F, K, L)``, the sample matrix is ``(n, F)``, and the output
is ``(n, L)``.

Operator codes (kept in sync with ``kernel._NORM_CODES``):
``0`` min/max, ``1`` probability, ``2`` luk, ``3`` hamacher, ``4`` einstein.

Every expression below is written in the same order as its NumPy counterpart in
:mod:`tribblefis.kernel`, so the two agree to the last bit *except* through
``exp`` itself, where the C library and NumPy's vectorized implementation may
differ by an ULP. Callers should compare with a tight tolerance, not exactly;
see ``tests/test_cython_kernel.py``.
"""

from cython.parallel cimport prange
from libc.math cimport exp, fabs


cdef extern from *:
    """
    #ifdef _OPENMP
    #define TRIBBLEFIS_HAVE_OPENMP 1
    #else
    #define TRIBBLEFIS_HAVE_OPENMP 0
    #endif
    """
    int TRIBBLEFIS_HAVE_OPENMP

# Whether this build can actually thread. `kernel.py` needs to know: a *serial*
# compiled loop is about half the speed of NumPy on large inputs, because NumPy's
# exp is SIMD-vectorized and libm's is not. Threading is what turns that around,
# so without it large inputs should go back to NumPy.
HAVE_OPENMP = bool(TRIBBLEFIS_HAVE_OPENMP)


cdef inline double _conorm(double a, double b, int code) noexcept nogil:
    """``t_conorm(a, b)``; mirrors ``gauss_math.t_conorm`` term for term."""
    cdef double ab, den
    if code == 0:            # min/max
        return a if a > b else b
    elif code == 1:          # probability: a + b - ab
        return a + b - a * b
    elif code == 2:          # luk: min(1, a + b)
        ab = a + b
        return ab if ab < 1.0 else 1.0
    elif code == 3:          # hamacher: (a + b - 2ab) / (1 - ab)
        ab = a * b
        den = 1.0 - ab
        if fabs(den) > 1e-12:
            return (a + b - 2.0 * ab) / den
        return 1.0           # the reference's `out=np.ones(...)` fallback
    else:                    # einstein: (a + b) / (1 + ab)
        return (a + b) / (1.0 + a * b)


cdef inline double _tnorm(double a, double b, int code) noexcept nogil:
    """``t_norm(a, b)``; mirrors ``gauss_math.t_norm`` term for term."""
    cdef double ab, den, s
    if code == 0:            # min/max
        return a if a < b else b
    elif code == 1:          # probability
        return a * b
    elif code == 2:          # luk: max(0, a + b - 1)
        s = a + b - 1.0
        return s if s > 0.0 else 0.0
    elif code == 3:          # hamacher: ab / (a + b - ab)
        ab = a * b
        den = a + b - ab
        if fabs(den) > 1e-12:
            return ab / den
        return 0.0           # the reference's `out=np.zeros(...)` fallback
    else:                    # einstein: ab / (2 - (a + b - ab))
        ab = a * b
        return ab / (2.0 - (a + b - ab))


def firing_strengths(
    const double[:, ::1] x,
    const double[:, :, ::1] mu,
    const double[:, :, ::1] sigma,
    const double[:, :, ::1] active,
    double[:, ::1] out,
    int t_norm_code,
    int t_conorm_code,
    int num_threads,
):
    """Fill `out` (n, L) with per-label firing strengths for `x` (n, F).

    `num_threads` <= 1 runs serially, which is what short inputs want: the
    OpenMP fork/join costs more than the work when a call only has a few hundred
    samples, and the refinement inner loop makes tens of thousands of exactly
    those calls.
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t n_f = mu.shape[0]
    cdef Py_ssize_t n_k = mu.shape[1]
    cdef Py_ssize_t n_l = mu.shape[2]
    cdef Py_ssize_t i, f, k, l
    cdef double xv, d, g, cell

    if n == 0:
        return

    if num_threads > 1:
        for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
            for l in range(n_l):
                out[i, l] = 1.0                       # t-norm identity
            for f in range(n_f):
                xv = x[i, f]
                for l in range(n_l):
                    cell = 0.0                        # t-conorm identity
                    for k in range(n_k):
                        d = (xv - mu[f, k, l]) / sigma[f, k, l]
                        g = exp(-0.5 * d * d) * active[f, k, l]
                        cell = _conorm(cell, g, t_conorm_code)
                    out[i, l] = _tnorm(out[i, l], cell, t_norm_code)
    else:
        with nogil:
            for i in range(n):
                for l in range(n_l):
                    out[i, l] = 1.0
                for f in range(n_f):
                    xv = x[i, f]
                    for l in range(n_l):
                        cell = 0.0
                        for k in range(n_k):
                            d = (xv - mu[f, k, l]) / sigma[f, k, l]
                            g = exp(-0.5 * d * d) * active[f, k, l]
                            cell = _conorm(cell, g, t_conorm_code)
                        out[i, l] = _tnorm(out[i, l], cell, t_norm_code)
