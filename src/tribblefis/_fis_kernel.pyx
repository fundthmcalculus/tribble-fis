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


cdef inline void _sample(
    Py_ssize_t i,
    const double[:, ::1] x,
    const double[:, :, ::1] mu,
    const double[:, :, ::1] sigma,
    const double[:, :, ::1] active,
    double[:, ::1] out,
    int t_norm_code,
    int t_conorm_code,
) noexcept nogil:
    """One sample's full forward pass, written into ``out[i, :]``."""
    cdef Py_ssize_t n_f = mu.shape[0], n_k = mu.shape[1], n_l = mu.shape[2]
    cdef Py_ssize_t f, k, l
    cdef double xv, d, g, cell

    for l in range(n_l):
        out[i, l] = 1.0                                   # t-norm identity
    for f in range(n_f):
        xv = x[i, f]
        for l in range(n_l):
            cell = 0.0                                    # t-conorm identity
            for k in range(n_k):
                d = (xv - mu[f, k, l]) / sigma[f, k, l]
                g = exp(-0.5 * d * d) * active[f, k, l]
                cell = _conorm(cell, g, t_conorm_code)
            out[i, l] = _tnorm(out[i, l], cell, t_norm_code)


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
    cdef Py_ssize_t i

    if n == 0:
        return

    if num_threads > 1:
        for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
            _sample(i, x, mu, sigma, active, out, t_norm_code, t_conorm_code)
    else:
        with nogil:
            for i in range(n):
                _sample(i, x, mu, sigma, active, out, t_norm_code, t_conorm_code)


# ---------------------------------------------------------------------------
# Incremental evaluation.
#
# Block coordinate descent perturbs exactly one membership function -- one
# (feature, membership, label) slot -- and then, with the reference forward pass,
# recomputes every membership of every feature for every label. Almost all of
# that is unchanged from the previous evaluation.
#
# Only two things actually depend on the perturbed slot:
#
#   cell[f*, l*]  the conorm fold over that (feature, label) cell's memberships
#   out[:, l*]    the t-norm fold over features, for that one label
#
# Every other cell, and every other label's column, is exactly what it was. So
# caching the per-(label, sample, feature) cell values turns an O(F*K*L)
# evaluation into O(K + F): recompute one cell, refold one column. The two
# functions below are that -- one to fill the cache, one to use it.
# ---------------------------------------------------------------------------

def firing_strengths_cells(
    const double[:, ::1] x,
    const double[:, :, ::1] mu,
    const double[:, :, ::1] sigma,
    const double[:, :, ::1] active,
    double[:, :, ::1] cells,
    double[:, ::1] out,
    int t_norm_code,
    int t_conorm_code,
    int num_threads,
):
    """Full forward pass that also records the per-cell conorm folds.

    `cells` is ``(L, n, F)`` -- label outermost so one label's plane is
    contiguous, and feature innermost so the t-norm fold over features reads
    consecutive memory. `out` is the usual ``(n, L)``.
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i

    if n == 0:
        return

    if num_threads > 1:
        for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
            _sample_cells(i, x, mu, sigma, active, cells, out,
                          t_norm_code, t_conorm_code)
    else:
        with nogil:
            for i in range(n):
                _sample_cells(i, x, mu, sigma, active, cells, out,
                              t_norm_code, t_conorm_code)


cdef inline void _sample_cells(
    Py_ssize_t i,
    const double[:, ::1] x,
    const double[:, :, ::1] mu,
    const double[:, :, ::1] sigma,
    const double[:, :, ::1] active,
    double[:, :, ::1] cells,
    double[:, ::1] out,
    int t_norm_code,
    int t_conorm_code,
) noexcept nogil:
    cdef Py_ssize_t n_f = mu.shape[0], n_k = mu.shape[1], n_l = mu.shape[2]
    cdef Py_ssize_t f, k, l
    cdef double xv, d, g, cell

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
            cells[l, i, f] = cell
            out[i, l] = _tnorm(out[i, l], cell, t_norm_code)


def refold_label(
    const double[::1] xcol,
    const double[::1] mu_k,
    const double[::1] sigma_k,
    const double[::1] active_k,
    const double[:, ::1] cells_l,
    Py_ssize_t fi,
    double[::1] new_cell,
    double[::1] out_col,
    int t_norm_code,
    int t_conorm_code,
    int num_threads,
):
    """Recompute one ``(feature, label)`` cell and refold that label's column.

    `xcol` is the perturbed feature's column, `mu_k`/`sigma_k`/`active_k` are that
    cell's ``K`` memberships with the candidate substituted, and `cells_l` is the
    cached ``(n, F)`` plane for the label. Writes the recomputed cell to
    `new_cell` and the label's new firing strength to `out_col`; the caller
    commits `new_cell` into `cells_l` only if the candidate is accepted.

    The t-norm fold is over features in the same order, and starts from the same
    1.0, as the full pass -- so the column this produces is bit-identical to
    recomputing everything.
    """
    cdef Py_ssize_t n = xcol.shape[0]
    cdef Py_ssize_t i

    if n == 0:
        return

    if num_threads > 1:
        for i in prange(n, nogil=True, schedule='static', num_threads=num_threads):
            _sample_refold(i, xcol, mu_k, sigma_k, active_k, cells_l, fi,
                           new_cell, out_col, t_norm_code, t_conorm_code)
    else:
        with nogil:
            for i in range(n):
                _sample_refold(i, xcol, mu_k, sigma_k, active_k, cells_l, fi,
                               new_cell, out_col, t_norm_code, t_conorm_code)


cdef inline void _sample_refold(
    Py_ssize_t i,
    const double[::1] xcol,
    const double[::1] mu_k,
    const double[::1] sigma_k,
    const double[::1] active_k,
    const double[:, ::1] cells_l,
    Py_ssize_t fi,
    double[::1] new_cell,
    double[::1] out_col,
    int t_norm_code,
    int t_conorm_code,
) noexcept nogil:
    cdef Py_ssize_t n_f = cells_l.shape[1], n_k = mu_k.shape[0]
    cdef Py_ssize_t f, k
    cdef double xv = xcol[i]
    cdef double d, g, cell, acc

    cell = 0.0
    for k in range(n_k):
        d = (xv - mu_k[k]) / sigma_k[k]
        g = exp(-0.5 * d * d) * active_k[k]
        cell = _conorm(cell, g, t_conorm_code)
    new_cell[i] = cell

    acc = 1.0
    for f in range(n_f):
        acc = _tnorm(acc, cell if f == fi else cells_l[i, f], t_norm_code)
    out_col[i] = acc
