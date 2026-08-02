"""A flat, array-shaped view of an all-Gaussian :class:`GaussianMixtureModel`.

The model that :mod:`tribblefis.gauss_math` evaluates is a dict of features, each
holding a dict of labels, each holding a list of immutable ``NamedTuple``
membership functions. That shape is right for building and inspecting a model
and wrong for evaluating one: the forward pass walks it with three nested Python
loops and calls ``np.exp`` once per membership function, and antecedent
refinement rebuilds the entire tree on every fitness evaluation just to move two
floats.

:class:`CompiledFIS` is the same model laid out as a few contiguous arrays, so
that

* moving parameters during refinement is an in-place write into a
  ``(F, K, L)`` array rather than a reconstruction of ~65 000 ``NamedTuple``\\ s;
* the feature columns are marshalled once instead of per evaluation;
* later backends (Cython, GPU) have a representation they can consume directly.

**What this does not buy.** The NumPy forward pass here is *not* meaningfully
faster than the reference loop, and it is not meant to be. Profiling says a
large forward pass is ~93% ``np.exp``, and ``np.exp`` on float64 runs at
~280 M elem/s -- which is essentially the reference's whole runtime. Batching the
same arithmetic differently in NumPy cannot beat that floor; measured, the
rewrite is within a couple of percent either way. Beating it needs either fusion
plus threads (a compiled kernel) or different hardware, both of which consume
this representation rather than replace it. The win banked here is on the
*training* path, where the model rebuild and the pandas lookups were real.

**Bit-exactness is a hard requirement.** Everything here performs the same
floating-point operations, in the same order, as the reference implementation in
:func:`tribblefis.gauss_math.tsk_firing_strengths`; the norm folds in particular
are still sequential in the reference's order, because reassociating them (a
``max``-reduce, or ``1 - prod(1 - g)`` for the probabilistic sum) would perturb
the last bits and make every downstream benchmark checksum unfalsifiable.

Models this module declines to compile (see :func:`can_compile`) simply take the
reference path, so nothing needs a fallback flag.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .gauss_data import (
    GaussianMembership,
    GaussianMixtureModel,
    NormConorm,
    NormPair,
)

# The optional compiled accelerator (see setup_cython.py). Absent by default:
# `pip install tribble-fis` never needs a C compiler, and everything below works
# without it, just slower.
try:
    from . import _fis_kernel  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - depends on whether the build was run
    _fis_kernel = None

HAVE_CYTHON_KERNEL = _fis_kernel is not None
HAVE_OPENMP = bool(getattr(_fis_kernel, "HAVE_OPENMP", False))

# Operator -> integer code, kept in sync with the switch in _fis_kernel.pyx.
_NORM_CODES: dict[str, int] = {
    "min/max": 0, "probability": 1, "luk": 2, "hamacher": 3, "einstein": 4,
}

# Below this much work (n_samples * F * K * L membership evaluations) the
# compiled kernel runs serially, because an OpenMP parallel region has a fixed
# entry cost. Measured on the reference machine that cost is small -- threading
# already wins ~9x at 4 800 evaluations, the smallest size swept -- so the
# threshold only exists to keep genuinely trivial calls out of the thread pool.
_THREAD_WORK_THRESHOLD = 2048

# Row block size, in elements, for the per-feature membership buffer. Each block
# holds ``rows x (L * K)`` doubles; keeping that near 1 MiB means the buffer and
# the running per-label accumulator stay resident in L2 while a feature's
# memberships are folded together, instead of streaming a 50 000-row temporary
# through memory once per feature.
_BLOCK_BYTES = 1 << 20


class NotCompilable(ValueError):
    """Raised by :func:`compile_model` for a model the flat layout cannot hold."""


@dataclass
class CompiledFIS:
    """An all-Gaussian model as contiguous ``(F, L, K)`` parameter arrays.

    Attributes:
        feature_names: Features in the model's own iteration order, which is the
            order the t-norm folds them in. ``F`` of them.
        labels: Output labels in ``ordered_keys`` (sorted) order -- the column
            order of the firing-strength matrix. ``L`` of them.
        mu, sigma: ``(F, K, L)`` parameter arrays, where ``K`` is the largest
            number of membership functions on any one (feature, label) cell.
            Cells with fewer than ``K`` memberships are padded. Membership index
            comes *before* label so that the k-th slice of the evaluated block is
            contiguous over labels -- the conorm fold reads one such slice per
            step, and a strided read there would cost a second full pass over the
            block.
        active: ``(F, K, L)`` float mask, 1.0 for a real membership function and
            0.0 for padding. It multiplies the evaluated memberships, which
            drives padded slots to exactly 0.0 -- the identity of every t-conorm
            -- so padding cannot perturb a fold.
        slot_index: For each Gaussian in ``refine``'s flat parameter order (see
            ``refine._iter_gaussian_slots``), its flat index into ``mu``/``sigma``.
            This is what lets a refinement write parameters straight into the
            arrays.
    """

    feature_names: tuple[str, ...]
    labels: tuple[Any, ...]
    mu: np.ndarray
    sigma: np.ndarray
    active: np.ndarray
    slot_index: np.ndarray

    @property
    def is_dense(self) -> bool:
        """True when no slot is padding, so the ``active`` mask can be skipped.

        This is the overwhelmingly common case (every (feature, label) cell has
        the same number of memberships) and skipping the mask removes one full
        pass over the evaluated block per feature.
        """
        return bool(self.active.all())

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_labels(self) -> int:
        return len(self.labels)

    @property
    def n_slots(self) -> int:
        """Number of real (non-padding) membership functions."""
        return int(self.slot_index.size)

    # -- parameter round-trip -------------------------------------------------

    def extract_params(self) -> np.ndarray:
        """The ``[mu_0, sigma_0, mu_1, sigma_1, ...]`` vector, in refine order."""
        flat_mu = self.mu.reshape(-1)[self.slot_index]
        flat_sigma = self.sigma.reshape(-1)[self.slot_index]
        out = np.empty(2 * self.n_slots, dtype=float)
        out[0::2] = flat_mu
        out[1::2] = flat_sigma
        return out

    def set_params(self, vec: np.ndarray) -> None:
        """Write a flat refine-order parameter vector into the arrays, in place.

        This is the whole point of the compiled layout for *training*: the
        reference path rebuilds every ``NamedTuple`` in the model for each of the
        tens of thousands of fitness evaluations, which profiled at ~17% of a
        refinement. Here it is two scatter writes.

        ``sigma`` is floored at 1e-6 exactly as
        :meth:`GaussianMembership.evaluate` and ``refine.apply_gaussian_params``
        do, so a candidate vector with a degenerate sigma behaves identically.
        """
        vec = np.asarray(vec, dtype=float)
        expected = 2 * self.n_slots
        if vec.size != expected:
            raise ValueError(f"expected {expected} parameters, got {vec.size}")
        self.mu.reshape(-1)[self.slot_index] = vec[0::2]
        self.sigma.reshape(-1)[self.slot_index] = np.maximum(vec[1::2], 1e-6)

    def copy(self) -> "CompiledFIS":
        """A deep copy, so a candidate can be mutated without touching the original."""
        return CompiledFIS(
            feature_names=self.feature_names,
            labels=self.labels,
            mu=self.mu.copy(),
            sigma=self.sigma.copy(),
            active=self.active.copy(),
            slot_index=self.slot_index,
        )

    # -- data marshalling -----------------------------------------------------

    def feature_matrix(self, feature_arrays: dict[str, np.ndarray]) -> np.ndarray:
        """Stack the caller's columns into a C-contiguous ``(n, F)`` matrix in
        this model's feature order."""
        cols = [np.asarray(feature_arrays[name], dtype=float) for name in self.feature_names]
        return np.ascontiguousarray(np.column_stack(cols)) if cols else np.empty((0, 0))


def can_compile(
    model: GaussianMixtureModel, available_features: Sequence[str] | None = None
) -> bool:
    """Whether :func:`compile_model` will accept this model.

    Three things disqualify a model, all of them because the flat layout would
    otherwise have to reproduce a *structural* branch of the reference loop and
    would stop being bit-exact while doing so:

    1. a non-Gaussian membership function anywhere (trapezoid, triangular);
    2. a ragged label set -- some feature model missing a label that another
       has. The reference ``continue``\\ s past that cell, which is the t-norm's
       identity element mathematically but not always in floating point (under
       Lukasiewicz, ``max(0, z + 1 - 1)`` is not ``z`` for tiny ``z``);
    3. a feature the caller has no column for, which the reference also skips.

    All three fall back to the reference path, which is correct and merely
    slower, so callers never need to handle a failure.
    """
    try:
        compile_model(model, available_features)
    except NotCompilable:
        return False
    return True


def compile_model(
    model: GaussianMixtureModel, available_features: Sequence[str] | None = None
) -> CompiledFIS:
    """Flatten `model` into a :class:`CompiledFIS`, or raise :class:`NotCompilable`."""
    feature_models = model.feature_models
    if not feature_models:
        raise NotCompilable("model has no features")

    feature_names = tuple(feature_models)
    if available_features is not None:
        have = set(available_features)
        missing = [f for f in feature_names if f not in have]
        if missing:
            raise NotCompilable(f"no data column for feature(s) {missing}")

    labels = tuple(next(iter(feature_models.values())).ordered_keys)
    label_set = set(labels)
    for fname, fmodel in feature_models.items():
        if set(fmodel.label_models) != label_set:
            raise NotCompilable(
                f"feature {fname!r} has labels {sorted(map(str, fmodel.label_models))}, "
                f"expected {sorted(map(str, labels))}"
            )

    max_k = 0
    for fmodel in feature_models.values():
        for lmodel in fmodel.label_models.values():
            for mf in lmodel.memberships:
                if not isinstance(mf, GaussianMembership):
                    raise NotCompilable(f"non-Gaussian membership {type(mf).__name__}")
            max_k = max(max_k, len(lmodel.memberships))
    if max_k == 0:
        raise NotCompilable("model has no membership functions")

    n_f, n_l = len(feature_names), len(labels)
    # Padding defaults are chosen so a padded slot evaluates to a finite number
    # (sigma=1 avoids a division by zero producing a NaN that 0.0 * NaN could
    # not clear) which `active` then multiplies to exactly 0.0.
    mu = np.zeros((n_f, max_k, n_l), dtype=float)
    sigma = np.ones((n_f, max_k, n_l), dtype=float)
    active = np.zeros((n_f, max_k, n_l), dtype=float)

    label_pos = {lab: i for i, lab in enumerate(labels)}
    slot_index: list[int] = []
    for fi, (_fname, fmodel) in enumerate(feature_models.items()):
        # `refine`'s parameter order walks `label_models` in *insertion* order
        # while the firing-strength columns are in *sorted* order; the two can
        # differ, so `slot_index` is built by walking insertion order and
        # writing to the sorted position.
        for label, lmodel in fmodel.label_models.items():
            li = label_pos[label]
            for k, mf in enumerate(lmodel.memberships):
                mu[fi, k, li] = mf.mu
                sigma[fi, k, li] = max(mf.sigma, 1e-6)
                active[fi, k, li] = 1.0
                slot_index.append((fi * max_k + k) * n_l + li)

    return CompiledFIS(
        feature_names=feature_names,
        labels=labels,
        mu=mu,
        sigma=sigma,
        active=active,
        slot_index=np.asarray(slot_index, dtype=np.intp),
    )


def to_model(compiled: CompiledFIS, template: GaussianMixtureModel) -> GaussianMixtureModel:
    """Rebuild a ``GaussianMixtureModel`` from `compiled`, reusing `template`'s
    structure and membership ids.

    Only needed at the *end* of a refinement, to hand a normal model back to the
    caller -- never inside the fitness loop, which is the entire saving.
    """
    from .gauss_data import FeatureModel, LabelModel

    label_pos = {lab: i for i, lab in enumerate(compiled.labels)}
    new_features = {}
    for fi, (fname, fmodel) in enumerate(template.feature_models.items()):
        new_labels = {}
        for label, lmodel in fmodel.label_models.items():
            li = label_pos[label]
            new_labels[label] = LabelModel(memberships=[
                mf._replace(mu=float(compiled.mu[fi, k, li]),
                            sigma=float(compiled.sigma[fi, k, li]))
                for k, mf in enumerate(lmodel.memberships)
            ])
        new_features[fname] = FeatureModel(label_models=new_labels)
    return template._replace(feature_models=new_features)


# ---------------------------------------------------------------------------
# In-place norm folds.
#
# These mirror `gauss_math.t_norm` / `t_conorm` exactly, but write through `out=`
# so a fold over K memberships allocates one buffer instead of K. The arithmetic
# and its order are unchanged, which is what keeps the result bit-identical.
# ---------------------------------------------------------------------------

def _conorm_into(acc: np.ndarray, g: np.ndarray, norm: NormConorm, scratch: np.ndarray) -> None:
    """``acc = t_conorm(acc, g)``, in place."""
    if norm == "min/max":
        np.maximum(acc, g, out=acc)
    elif norm == "probability":
        # x + y - x*y, in the reference's order.
        np.multiply(acc, g, out=scratch)
        np.add(acc, g, out=acc)
        np.subtract(acc, scratch, out=acc)
    elif norm == "luk":
        np.add(acc, g, out=acc)
        np.minimum(acc, 1.0, out=acc)
    elif norm == "einstein":
        np.multiply(acc, g, out=scratch)   # x*y
        np.add(scratch, 1.0, out=scratch)  # 1 + x*y
        np.add(acc, g, out=acc)            # x + y
        np.divide(acc, scratch, out=acc)
    elif norm == "hamacher":
        # (x + y - 2xy) / (1 - xy), with the reference's 1e-12 guard leaving the
        # accumulator at the reference's fallback value of 1.0.
        xy = acc * g
        num = acc + g - 2.0 * xy
        den = 1.0 - xy
        ok = np.abs(den) > 1e-12
        acc.fill(1.0)
        np.divide(num, den, out=acc, where=ok)
    else:
        raise ValueError(f"Invalid NORM_CORNOM value: {norm}")


def _norm_into(acc: np.ndarray, g: np.ndarray, norm: NormConorm, scratch: np.ndarray) -> None:
    """``acc = t_norm(acc, g)``, in place."""
    if norm == "min/max":
        np.minimum(acc, g, out=acc)
    elif norm == "probability":
        np.multiply(acc, g, out=acc)
    elif norm == "luk":
        np.add(acc, g, out=acc)
        np.subtract(acc, 1.0, out=acc)
        np.maximum(acc, 0.0, out=acc)
    elif norm == "einstein":
        np.multiply(acc, g, out=scratch)      # x*y
        np.add(acc, g, out=acc)               # x + y
        np.subtract(acc, scratch, out=acc)    # x + y - x*y
        np.subtract(2.0, acc, out=acc)        # 2 - (x + y - x*y)
        np.divide(scratch, acc, out=acc)
    elif norm == "hamacher":
        xy = acc * g
        den = acc + g - xy
        ok = np.abs(den) > 1e-12
        acc.fill(0.0)
        np.divide(xy, den, out=acc, where=ok)
    else:
        raise ValueError(f"Invalid NORM_CORNOM value: {norm}")


def _block_rows(n_samples: int, n_cells: int) -> int:
    """Rows per block, sized so the per-feature membership buffer stays in cache."""
    per_row = max(n_cells, 1) * 8
    return max(256, min(n_samples, _BLOCK_BYTES // per_row))


def _thread_count(work: int) -> int:
    """Threads for a compiled call of the given size.

    ``TRIBBLEFIS_NUM_THREADS`` overrides the choice outright, which is what a
    benchmark comparing serial against parallel needs.
    """
    override = os.environ.get("TRIBBLEFIS_NUM_THREADS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    if work < _THREAD_WORK_THRESHOLD:
        return 1
    return max(1, os.cpu_count() or 1)


def firing_strengths(
    compiled: CompiledFIS,
    feature_matrix: np.ndarray,
    norms: NormPair,
    backend: str = "auto",
) -> np.ndarray:
    """Raw per-label firing strengths, ``(n_samples, n_labels)``.

    `feature_matrix` is ``(n_samples, n_features)`` in ``compiled.feature_names``
    order -- see :meth:`CompiledFIS.feature_matrix`.

    `backend` is one of:

    ``"auto"``
        The compiled kernel when it was built, else NumPy. Bit-identical to the
        reference forward pass either way, which is why it is the default.
    ``"cython"`` / ``"numpy"``
        Force one of those two.
    ``"torch"``
        The Torch/CUDA backend in :mod:`tribblefis.gpu`. Never selected by
        ``auto``: CUDA's ``exp`` differs from libm's by about an ULP, so
        substituting it silently would move every benchmark checksum in the
        repo. Asking for it explicitly is the opt-in to that drift.
    """
    if backend not in ("auto", "cython", "numpy", "torch"):
        raise ValueError(
            f"backend must be 'auto', 'cython', 'numpy' or 'torch', got {backend!r}"
        )
    if backend == "torch":
        from . import gpu

        return gpu.firing_strengths(compiled, feature_matrix, norms)
    if backend == "cython" and not HAVE_CYTHON_KERNEL:
        raise RuntimeError(
            "the compiled kernel is not built; run "
            "`python setup_cython.py build_ext --inplace`"
        )
    if backend == "numpy" or not HAVE_CYTHON_KERNEL:
        return firing_strengths_numpy(compiled, feature_matrix, norms)
    if backend == "auto" and not _cython_is_faster(compiled, feature_matrix):
        return firing_strengths_numpy(compiled, feature_matrix, norms)
    return _firing_strengths_cython(compiled, feature_matrix, norms)


def _cython_is_faster(compiled: CompiledFIS, feature_matrix: np.ndarray) -> bool:
    """Whether ``auto`` should prefer the compiled kernel for this call.

    It is not unconditional, because a *serial* compiled loop loses to NumPy
    once the input is more than a few thousand evaluations -- measured 0.50x on
    the 50k-sample workload. The reason is ``exp``: NumPy's is SIMD-vectorized
    and libm's is one scalar call per element, and at that size the pass is
    nothing but ``exp``. Threading reverses it decisively (9-14x across the whole
    swept range), so with OpenMP the compiled kernel always wins, and without it
    only the small end -- where removing per-call NumPy dispatch is the point --
    is worth taking.
    """
    if HAVE_OPENMP:
        return True
    work = int(np.shape(feature_matrix)[0]) * compiled.mu.size
    return work < _THREAD_WORK_THRESHOLD


def _firing_strengths_cython(
    compiled: CompiledFIS, feature_matrix: np.ndarray, norms: NormPair
) -> np.ndarray:
    """Dispatch to the compiled kernel, which fuses the whole forward pass.

    The NumPy path makes six passes over an ``(rows, K*L)`` block per feature and
    one more per conorm step; this makes none -- every intermediate lives in a
    register and only the ``(n, L)`` output is written. It also threads the
    sample loop, which NumPy's ufuncs cannot do.
    """
    x = np.ascontiguousarray(feature_matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] != compiled.n_features:
        raise ValueError(
            f"feature_matrix must be (n, {compiled.n_features}), got {x.shape}"
        )
    try:
        t_norm_code = _NORM_CODES[norms.t_norm]
        t_conorm_code = _NORM_CODES[norms.t_conorm]
    except KeyError as exc:  # pragma: no cover - resolve_norm_pair validates first
        raise ValueError(f"Invalid NORM_CORNOM value: {exc.args[0]}") from None

    out = np.empty((x.shape[0], compiled.n_labels), dtype=float)
    work = x.shape[0] * compiled.mu.size
    _fis_kernel.firing_strengths(
        x,
        np.ascontiguousarray(compiled.mu),
        np.ascontiguousarray(compiled.sigma),
        np.ascontiguousarray(compiled.active),
        out,
        t_norm_code,
        t_conorm_code,
        _thread_count(work),
    )
    return out


def firing_strengths_numpy(
    compiled: CompiledFIS,
    feature_matrix: np.ndarray,
    norms: NormPair,
    cells: np.ndarray | None = None,
) -> np.ndarray:
    """The portable NumPy implementation, bit-identical to the reference.

    The loop structure is blocks of rows on the outside, then features, then the
    K-fold conorm and the F-fold t-norm. Within one feature every ``(label,
    membership)`` Gaussian is evaluated in a single broadcast ``exp``, so the
    reference's ``L * K`` NumPy calls per feature become a handful. That removes
    dispatch overhead but not arithmetic, and the arithmetic is the binding
    constraint (see the module docstring) -- so this is the always-available
    implementation of the compiled representation, and the definition of
    correctness the compiled kernel is checked against, rather than the fast one.

    When `cells` is given (an ``(L, n, F)`` array) each ``(feature, label)``
    cell's conorm fold is recorded into it, which is what
    :class:`IncrementalFIS` caches.
    """
    x_all = np.asanyarray(feature_matrix, dtype=float)
    if x_all.ndim != 2 or x_all.shape[1] != compiled.n_features:
        raise ValueError(
            f"feature_matrix must be (n, {compiled.n_features}), got {x_all.shape}"
        )
    n_samples = x_all.shape[0]
    n_l, n_k = compiled.n_labels, compiled.mu.shape[1]
    out = np.empty((n_samples, n_l), dtype=float)
    if n_samples == 0:
        return out

    # (F, K*L) views: one row of parameters per feature, flat over membership
    # index and label, which is exactly the shape a broadcast against a column of
    # samples needs.
    mu_f = compiled.mu.reshape(compiled.n_features, n_k * n_l)
    sigma_f = compiled.sigma.reshape(compiled.n_features, n_k * n_l)
    active_f = compiled.active.reshape(compiled.n_features, n_k * n_l)

    masked = not compiled.is_dense
    rows = _block_rows(n_samples, n_k * n_l)
    # Every buffer the inner loops need is allocated once here, for the largest
    # block; short final blocks take a view. Allocating inside the fold would put
    # an O(K * F) malloc/free stream in the hot path, which is the specific cost
    # this rewrite exists to remove.
    buf = np.empty((rows, n_k * n_l), dtype=float)
    cell_buf = np.empty((rows, n_l), dtype=float)
    cell_scratch_buf = np.empty((rows, n_l), dtype=float)
    acc_scratch_buf = np.empty((rows, n_l), dtype=float)

    for start in range(0, n_samples, rows):
        stop = min(start + rows, n_samples)
        m = stop - start
        g = buf[:m]
        cell = cell_buf[:m]
        cell_scratch = cell_scratch_buf[:m]
        acc_scratch = acc_scratch_buf[:m]
        acc = out[start:stop]
        acc.fill(1.0)  # t-norm identity; the fold below starts from ones

        for fi in range(compiled.n_features):
            x = x_all[start:stop, fi, None]
            # g = exp(-0.5 * ((x - mu) / sigma) ** 2) * active, elementwise
            # identical to GaussianMembership.evaluate followed by the padding
            # mask.
            np.subtract(x, mu_f[fi], out=g)
            np.divide(g, sigma_f[fi], out=g)
            np.multiply(g, g, out=g)
            np.multiply(g, -0.5, out=g)
            np.exp(g, out=g)
            if masked:
                np.multiply(g, active_f[fi], out=g)

            g3 = g.reshape(m, n_k, n_l)
            cell.fill(0.0)  # t-conorm identity
            for k in range(n_k):
                _conorm_into(cell, g3[:, k, :], norms.t_conorm, cell_scratch)
            if cells is not None:
                cells[:, start:stop, fi] = cell.T
            _norm_into(acc, cell, norms.t_norm, acc_scratch)

    return out


# ---------------------------------------------------------------------------
# Incremental evaluation for block coordinate descent.
# ---------------------------------------------------------------------------

class IncrementalFIS:
    """Firing strengths under a *one-membership-at-a-time* perturbation.

    Block coordinate descent (``refine.refine_classifier_antecedents`` with
    ``method="coordinate"``, ``block=2``) moves exactly one Gaussian's
    ``(mu, sigma)`` per sub-problem and then asks for the whole forward pass
    again -- of which almost everything is bit-for-bit what it was on the
    previous call. Only two quantities depend on the moved membership function:

    * the conorm fold of its own ``(feature, label)`` cell, and
    * the t-norm fold over features for its own label.

    Caching the per-cell folds therefore reduces an evaluation from
    ``O(n * F * K * L)`` to ``O(n * (K + F))``. On the wide training benchmark
    (20 features, 6 labels, 3 memberships) that is 360 membership evaluations per
    sample replaced by 23.

    The result is *bit-identical* to a full pass, not an approximation: the
    recomputed cell uses the same fold in the same order, the refold over
    features starts from the same 1.0 and visits features in the same order, and
    every untouched column is a value the same code produced earlier.

    Memory is ``n * F * L`` doubles for the cache. That is the trade -- 3.8 MB on
    the wide benchmark, but it scales with the training set, so
    :meth:`from_compiled` is the place to add a cap if one is ever needed.

    Usage is propose-then-commit::

        fs = inc.evaluate_slot(slot, mu, sigma)   # candidate, cache untouched
        ...                                        # score it
        inc.commit()                               # only if the caller accepts

    Nothing mutates until :meth:`commit`, so a rejected candidate needs no undo.
    """

    def __init__(self, compiled: CompiledFIS, feature_matrix: np.ndarray, norms: NormPair):
        self.compiled = compiled
        self.norms = norms
        self.x = np.ascontiguousarray(feature_matrix, dtype=float)
        if self.x.ndim != 2 or self.x.shape[1] != compiled.n_features:
            raise ValueError(
                f"feature_matrix must be (n, {compiled.n_features}), got {self.x.shape}"
            )
        n = self.x.shape[0]
        n_f, n_k, n_l = compiled.n_features, compiled.mu.shape[1], compiled.n_labels
        self.n_k = n_k
        self.cells = np.empty((n_l, n, n_f), dtype=float)
        self.base = np.empty((n, n_l), dtype=float)
        # Scratch reused across evaluations: a candidate must not allocate.
        self._new_cell = np.empty(n, dtype=float)
        self._new_col = np.empty(n, dtype=float)
        self._candidate = np.empty((n, n_l), dtype=float)
        self._mu_k = np.empty(n_k, dtype=float)
        self._sigma_k = np.empty(n_k, dtype=float)
        self._pending: tuple[int, int, int, float, float] | None = None
        self.refresh()

    @property
    def n_samples(self) -> int:
        return self.x.shape[0]

    def refresh(self) -> None:
        """Recompute the cache from the compiled model's current parameters."""
        n_f, n_k, n_l = (
            self.compiled.n_features, self.compiled.mu.shape[1], self.compiled.n_labels
        )
        if HAVE_CYTHON_KERNEL:
            _fis_kernel.firing_strengths_cells(
                self.x,
                np.ascontiguousarray(self.compiled.mu),
                np.ascontiguousarray(self.compiled.sigma),
                np.ascontiguousarray(self.compiled.active),
                self.cells,
                self.base,
                _NORM_CODES[self.norms.t_norm],
                _NORM_CODES[self.norms.t_conorm],
                _thread_count(self.n_samples * self.compiled.mu.size),
            )
        else:
            self.base[...] = firing_strengths_numpy(
                self.compiled, self.x, self.norms, cells=self.cells
            )
        self._pending = None

    def _decode(self, slot: int) -> tuple[int, int, int]:
        """Slot index in refine's parameter order -> ``(feature, membership, label)``."""
        flat = int(self.compiled.slot_index[slot])
        n_k, n_l = self.compiled.mu.shape[1], self.compiled.n_labels
        li = flat % n_l
        ki = (flat // n_l) % n_k
        fi = flat // (n_l * n_k)
        return fi, ki, li

    def evaluate_slot(self, slot: int, mu: float, sigma: float) -> np.ndarray:
        """Firing strengths with membership `slot` set to ``(mu, sigma)``.

        The returned array is a reused buffer -- valid until the next call, which
        is all the fitness closure needs and saves an ``(n, L)`` allocation per
        candidate. ``sigma`` is floored at 1e-6 exactly as everywhere else.
        """
        fi, ki, li = self._decode(slot)
        sigma = max(float(sigma), 1e-6)

        self._mu_k[:] = self.compiled.mu[fi, :, li]
        self._sigma_k[:] = self.compiled.sigma[fi, :, li]
        self._mu_k[ki] = mu
        self._sigma_k[ki] = sigma
        active_k = np.ascontiguousarray(self.compiled.active[fi, :, li])

        cells_l = self.cells[li]
        if HAVE_CYTHON_KERNEL:
            _fis_kernel.refold_label(
                np.ascontiguousarray(self.x[:, fi]),
                self._mu_k, self._sigma_k, active_k,
                cells_l, fi, self._new_cell, self._new_col,
                _NORM_CODES[self.norms.t_norm],
                _NORM_CODES[self.norms.t_conorm],
                _thread_count(self.n_samples * (self.n_k + self.compiled.n_features)),
            )
        else:
            _refold_label_numpy(
                self.x[:, fi], self._mu_k, self._sigma_k, active_k,
                cells_l, fi, self._new_cell, self._new_col, self.norms,
            )

        self._candidate[...] = self.base
        self._candidate[:, li] = self._new_col
        self._pending = (fi, ki, li, float(mu), sigma)
        return self._candidate

    def commit(self) -> None:
        """Accept the last :meth:`evaluate_slot` candidate into the cache."""
        if self._pending is None:
            raise RuntimeError("commit() without a preceding evaluate_slot()")
        fi, ki, li, mu, sigma = self._pending
        self.cells[li, :, fi] = self._new_cell
        self.base[:, li] = self._new_col
        self.compiled.mu[fi, ki, li] = mu
        self.compiled.sigma[fi, ki, li] = sigma
        self._pending = None


def _refold_label_numpy(
    xcol, mu_k, sigma_k, active_k, cells_l, fi, new_cell, new_col, norms: NormPair
) -> None:
    """NumPy twin of ``_fis_kernel.refold_label``; see :class:`IncrementalFIS`."""
    g = np.exp(-0.5 * ((xcol[:, None] - mu_k) / sigma_k) ** 2) * active_k
    cell = np.zeros(len(xcol), dtype=float)
    scratch = np.empty_like(cell)
    for k in range(g.shape[1]):
        _conorm_into(cell, g[:, k], norms.t_conorm, scratch)
    new_cell[...] = cell

    acc = np.ones(len(xcol), dtype=float)
    for f in range(cells_l.shape[1]):
        _norm_into(acc, cell if f == fi else cells_l[:, f], norms.t_norm, scratch)
    new_col[...] = acc
