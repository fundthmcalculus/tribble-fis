"""Optional Torch/CUDA backend for the TSK forward pass.

Consumes the same :class:`~tribblefis.kernel.CompiledFIS` layout as the NumPy and
Cython backends. PyTorch is not a dependency of this package; every entry point
here degrades to :func:`is_available` returning ``False``.

**This backend is never chosen automatically, and that is deliberate.** CUDA's
``exp`` is not bit-identical to libm's -- measured, the two agree to about one
ULP -- so a silent substitution would move every benchmark checksum in the
repository and destroy the property the rest of the optimization work is
verified against. Ask for it explicitly (``backend="torch"``) and you are opting
into last-bit drift in exchange for the speed.

Measured on an RTX 4080 Laptop against the 24-thread Cython kernel, same shapes,
same seeds, same timing boundary (data already resident on both sides):

    1M samples x 20 features x 8 labels x 4 MF   351.2 ms -> 188.7 ms  1.86x  (float64)
                                                          ->  90.4 ms  3.88x  (float32)
    64 candidates x 4k x 20 x 6 x 3               68.0 ms ->  13.9 ms  4.91x  (float32)

Four things those numbers say that are worth knowing before reaching for a GPU:

* **The float64 margin is modest.** Consumer NVIDIA parts run double precision
  at a fraction of single-precision rate, so a 24-core CPU with a vectorized
  ``exp`` is genuinely competitive -- under 2x on a million samples.
* **float32 roughly doubles that again**, and is a much larger numerical change
  than the ULP-level ``exp`` difference. It is for inference throughput, not for
  training.
* **Batching candidates is worth about 1.15x, not a multiplier.** The same 64
  candidates run one at a time take 15.8 ms against 13.9 ms batched; the device
  is already saturated by a single candidate at this size, and what batching
  actually saves is 64 parameter uploads. So the 4.91x above is
  GPU-versus-CPU, essentially none of it batching-versus-not.
  :meth:`TorchFIS.firing_strengths_batch` is therefore mostly a convenience for
  population searches, and the benchmark suite keeps a sequential GPU row next
  to the batched one so that stays checkable.
* **The CPU side of these comparisons is the thermally variable one.** The
  24-thread kernel ranged 351-435 ms across repeats of the same input while the
  GPU rows held within 1%. The ratios above use the CPU's *best* run, which is
  the reading least flattering to the GPU.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .gauss_data import NormConorm, NormPair
from .kernel import CompiledFIS

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional extra
    torch = None  # type: ignore[assignment]

# Cap on the transient ``(..., n, F, K, L)`` membership tensor. Chunking the
# sample axis to stay under this keeps a million-row input from asking for
# multiple gigabytes of device memory to hold an intermediate that is reduced
# away immediately.
_CHUNK_BYTES = 256 << 20


def is_available(require_cuda: bool = False) -> bool:
    """Whether this backend can run; with `require_cuda`, whether it can run on a GPU."""
    if torch is None:
        return False
    return torch.cuda.is_available() if require_cuda else True


def default_device() -> str:
    return "cuda" if is_available(require_cuda=True) else "cpu"


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "the torch backend needs PyTorch installed: pip install torch"
        )


def _conorm(a, b, norm: NormConorm):
    """``t_conorm`` on tensors; term for term the same as ``gauss_math.t_conorm``."""
    if norm == "min/max":
        return torch.maximum(a, b)
    if norm == "probability":
        return a + b - a * b
    if norm == "luk":
        return torch.clamp(a + b, max=1.0)
    if norm == "einstein":
        return (a + b) / (1.0 + a * b)
    if norm == "hamacher":
        ab = a * b
        den = 1.0 - ab
        return torch.where(den.abs() > 1e-12, (a + b - 2.0 * ab) / den,
                           torch.ones_like(a))
    raise ValueError(f"Invalid NORM_CORNOM value: {norm}")


def _tnorm(a, b, norm: NormConorm):
    """``t_norm`` on tensors; term for term the same as ``gauss_math.t_norm``."""
    if norm == "min/max":
        return torch.minimum(a, b)
    if norm == "probability":
        return a * b
    if norm == "luk":
        return torch.clamp(a + b - 1.0, min=0.0)
    if norm == "einstein":
        ab = a * b
        return ab / (2.0 - (a + b - ab))
    if norm == "hamacher":
        ab = a * b
        den = a + b - ab
        return torch.where(den.abs() > 1e-12, ab / den, torch.zeros_like(a))
    raise ValueError(f"Invalid NORM_CORNOM value: {norm}")


class TorchFIS:
    """A compiled model with its data resident on a Torch device.

    The sample matrix is uploaded once and reused, which is what makes repeated
    calls -- a population search, or a classifier predicting over the same frame
    many times -- worth doing on a GPU at all. A single upload-evaluate-download
    is dominated by the transfer.
    """

    def __init__(
        self,
        compiled: CompiledFIS,
        feature_matrix: np.ndarray,
        norms: NormPair,
        device: str | None = None,
        dtype: str = "float64",
    ):
        _require_torch()
        if dtype not in ("float64", "float32"):
            raise ValueError(f"dtype must be 'float64' or 'float32', got {dtype!r}")
        self.compiled = compiled
        self.norms = norms
        self.device = torch.device(device or default_device())
        self.torch_dtype = torch.float64 if dtype == "float64" else torch.float32
        self.dtype = dtype

        x = np.ascontiguousarray(feature_matrix, dtype=float)
        if x.ndim != 2 or x.shape[1] != compiled.n_features:
            raise ValueError(
                f"feature_matrix must be (n, {compiled.n_features}), got {x.shape}"
            )
        self.x = torch.as_tensor(x, device=self.device, dtype=self.torch_dtype)
        self.active = torch.as_tensor(
            compiled.active, device=self.device, dtype=self.torch_dtype
        )
        self._upload_params()

    # -- parameters -----------------------------------------------------------

    def _upload_params(self) -> None:
        self.mu = torch.as_tensor(
            self.compiled.mu, device=self.device, dtype=self.torch_dtype)
        self.sigma = torch.as_tensor(
            self.compiled.sigma, device=self.device, dtype=self.torch_dtype)

    def set_params(self, vec: np.ndarray) -> None:
        """Write a flat refine-order parameter vector and re-upload."""
        self.compiled.set_params(vec)
        self._upload_params()

    # -- evaluation -----------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])

    def _chunk_rows(self, n_candidates: int) -> int:
        """Rows per chunk so the transient membership tensor stays under the cap."""
        per_row = self.compiled.mu.size * n_candidates * self.torch_dtype.itemsize
        return max(1, min(self.n_samples, _CHUNK_BYTES // max(per_row, 1)))

    def _forward(self, mu, sigma, active, x):
        """Core reduction shared by the single and batched paths.

        `mu`/`sigma`/`active` broadcast against ``x[..., :, :, None, None]``, so
        this serves both a plain ``(F, K, L)`` model and a ``(P, F, K, L)`` stack
        of candidates without a second implementation.
        """
        g = torch.exp(-0.5 * ((x[..., :, :, None, None] - mu) / sigma) ** 2) * active
        # g is (..., n, F, K, L). Fold memberships, then features, in the same
        # order and from the same identity elements as every other backend --
        # `amax`/`amin` only where the operator is exactly associative, so the
        # shortcut cannot change the answer.
        if self.norms.t_conorm == "min/max":
            cell = g.amax(dim=-2)
        else:
            cell = torch.zeros_like(g[..., 0, :])       # t-conorm identity
            for k in range(g.shape[-2]):
                cell = _conorm(cell, g[..., k, :], self.norms.t_conorm)
        if self.norms.t_norm == "min/max":
            return cell.amin(dim=-2)
        out = torch.ones_like(cell[..., 0, :])          # t-norm identity
        for f in range(cell.shape[-2]):
            out = _tnorm(out, cell[..., f, :], self.norms.t_norm)
        return out

    def firing_strengths(self, as_numpy: bool = True):
        """Per-label firing strengths, ``(n, L)``."""
        chunks = []
        rows = self._chunk_rows(1)
        for start in range(0, self.n_samples, rows):
            x = self.x[start:start + rows]
            chunks.append(self._forward(self.mu, self.sigma, self.active, x))
        out = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        return out.double().cpu().numpy() if as_numpy else out

    def firing_strengths_batch(self, param_matrix: np.ndarray, as_numpy: bool = True):
        """Firing strengths for ``P`` candidate parameter vectors at once.

        `param_matrix` is ``(P, 2 * n_slots)`` in refine's flat order. Returns
        ``(P, n, L)``.

        Measured, this is only about 1.15x faster than calling
        :meth:`firing_strengths` in a loop (13.9 ms against 15.8 ms for 64
        candidates) -- the device is already saturated by a single candidate at
        these sizes, and what batching saves is the repeated parameter uploads,
        not arithmetic. It is here mainly because a generation-at-a-time
        population search is easier to express against it.
        """
        params = np.asarray(param_matrix, dtype=float)
        if params.ndim != 2 or params.shape[1] != 2 * self.compiled.n_slots:
            raise ValueError(
                f"param_matrix must be (P, {2 * self.compiled.n_slots}), "
                f"got {params.shape}"
            )
        n_p = params.shape[0]
        scratch = self.compiled.copy()
        mu_stack = np.empty((n_p,) + self.compiled.mu.shape, dtype=float)
        sigma_stack = np.empty_like(mu_stack)
        for p in range(n_p):
            scratch.set_params(params[p])
            mu_stack[p] = scratch.mu
            sigma_stack[p] = scratch.sigma

        mu = torch.as_tensor(mu_stack, device=self.device, dtype=self.torch_dtype)
        sigma = torch.as_tensor(sigma_stack, device=self.device, dtype=self.torch_dtype)
        active = self.active.unsqueeze(0)

        chunks = []
        rows = self._chunk_rows(n_p)
        for start in range(0, self.n_samples, rows):
            x = self.x[start:start + rows].unsqueeze(0)
            chunks.append(self._forward(mu.unsqueeze(1), sigma.unsqueeze(1),
                                        active.unsqueeze(1), x))
        out = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
        return out.double().cpu().numpy() if as_numpy else out


def firing_strengths(
    compiled: CompiledFIS,
    feature_matrix: np.ndarray,
    norms: NormPair,
    device: str | None = None,
    dtype: str = "float64",
) -> np.ndarray:
    """One-shot ``(n, L)`` forward pass on a Torch device.

    Convenience wrapper. It uploads the sample matrix per call, so for anything
    repeated hold a :class:`TorchFIS` instead.
    """
    return TorchFIS(compiled, feature_matrix, norms, device, dtype).firing_strengths()
