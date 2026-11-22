"""Learned fuzzy aggregation over time (issue #127).

`MimoGaussianPredictor` and `TribbleRegressor` are flat tabular estimators, so
a sequence has to be collapsed to a row before they see it. The usual way to do
that is by hand -- "mean of the first third", "max of the last third" -- which
fixes two things the data never got a say in: *where* the windows sit and
*which statistic* summarises each one.

This module learns both, with an operator from aggregation theory rather than
an invented one:

* **Where** -- a fuzzy partition of normalised time ``tau in (0, 1)`` into
  ``K`` segments. Each segment is a normalised Gaussian (a softmax over
  Gaussian log-memberships), so the memberships sum to one at every instant
  -- a Ruspini partition, the same structure ``tribblefis.ruspini`` uses across
  features, here used across time. Centres and widths are learned.

* **Which statistic** -- Torra's weighted OWA (WOWA). Inside segment ``k`` the
  samples carry importance weights ``p_i`` (their membership, normalised) and
  are combined by an OWA whose order weights come from a RIM quantifier
  ``Q``::

      sort values descending: x_(1) >= x_(2) >= ...
      S_j     = p_(1) + ... + p_(j)
      WOWA    = sum_j [Q(S_j) - Q(S_{j-1})] * x_(j)

  The quantifier is the exponential RIM family
  ``Q(x) = (1 - exp(-lam * x)) / (1 - exp(-lam))``: ``lam = 0`` is the
  importance-weighted mean, ``lam -> +inf`` the maximum over the segment,
  ``lam -> -inf`` the minimum; orness is ``1/(1 - exp(-lam)) - 1/lam``. One
  ``lam`` is learned per (segment, feature).

  Why not the textbook power family ``Q(x) = x**alpha``: its slope at 0 is
  infinite for ``alpha < 1``, so a sample with importance ``1e-200`` -- any
  sample outside a soft segment -- still receives weight
  ``(1e-200)**alpha``, and a max-like segment returns the *global* maximum.
  With a fuzzy partition every sample has non-zero importance in every
  segment, so that leak is the common case, not an edge case. The
  exponential family has ``Q'(0) ~ lam``, bounding an outside sample's
  weight by about ``lam * p``.

WOWA is a Choquet integral with respect to the distorted-probability fuzzy
measure ``mu(A) = Q(sum_{i in A} p_i)``. A general Choquet integral over ``L``
time samples needs ``2**L`` measure values; this family needs ``2K + K*F``
parameters regardless of sequence length, keeps every axiom of an aggregation
function (bounded by min/max, idempotent, monotone), and -- the property the
design leans on -- **nests the hand-engineered baseline**: centres at
``(k + 0.5) / K``, widths -> 0 and ``lam = 0`` reproduce the crisp
"mean of each K-th" exactly. Learning starts from (a soft version of) that
point, so it begins where an analyst would and moves only if the data pays
for it.

Parameters are learned by minimising K-fold cross-validated ridge-regression
error of the target on the aggregated features -- a cheap, closed-form
surrogate for the downstream FIS, deliberately decoupled from it so the
representation can be inspected and reused. See
``docs/temporal-aggregation-design.md`` for the design, the test protocol and
measured results.

Nothing here imports or changes the existing estimators' behaviour; it is an
opt-in front end.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

from .gaussian_regressor import MimoGaussianPredictor
from .optimizer_utils import projected_lbfgs_solve

__all__ = [
    "FuzzyTemporalAggregator",
    "TribbleTimeSeriesRegressor",
    "PaddedSequences",
    "pad_sequences",
    "temporal_partition",
    "wowa",
    "exponential_quantifier",
    "quantifier_orness",
]

# Parameter bounds. Widths are in units of normalised time. |lam| <= 200 keeps
# exp(|lam|) ~ 1e87 finite (the quantifier never squares it) while reaching
# orness 0.995 -- max-/min-like over segments of ~100 samples.
_LAMBDA_BOUND = 200.0
_WIDTH_BOUNDS = (0.005, 1.0)
# Every parameter is scaled to a box of width ~1 so one step size and one
# curvature guess suit all of them: centres are optimised directly on [0, 1],
# log-widths divided by log(1/0.005), lam divided by 2 * _LAMBDA_BOUND.
_LOG_WIDTH_SCALE = float(np.log(_WIDTH_BOUNDS[1] / _WIDTH_BOUNDS[0]))
_LAMBDA_SCALE = 2.0 * _LAMBDA_BOUND


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------


class PaddedSequences:
    """Variable-length sequences, padded and pre-sorted for WOWA.

    Sorting does not depend on any learned parameter, so it is done once here
    and reused by every objective evaluation during fitting.

    Attributes:
        values: ``(N, F, L)`` sample values sorted descending per (sequence,
            feature); padding positions hold 0 and sort last.
        tau: ``(N, F, L)`` normalised time of each sorted sample.
        mask: ``(N, F, L)`` True where the sorted position is a real sample.
        lengths: ``(N,)`` true sequence lengths.
        feature_names: column names, ``x0..`` when none were given.
    """

    def __init__(self, values, tau, mask, lengths, feature_names):
        self.values = values
        self.tau = tau
        self.mask = mask
        self.lengths = lengths
        self.feature_names = feature_names

    @property
    def n_sequences(self) -> int:
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        return self.values.shape[1]


def pad_sequences(X) -> PaddedSequences:
    """Normalise sequence input to a :class:`PaddedSequences`.

    Accepts a 3-D array ``(n_sequences, n_timesteps, n_features)``, a 2-D
    array (one feature), or a list of 2-D arrays / DataFrames of possibly
    different lengths -- flight cycles are not all the same length.

    Sample ``i`` of a length-``L`` sequence sits at ``tau = (i + 0.5) / L``,
    the midpoint of its share of the interval. With that convention a crisp
    partition at ``1/K, 2/K, ...`` assigns exactly ``L/K`` samples to each
    segment whenever ``K`` divides ``L``, and a length-1 sequence sits at 0.5.
    """
    if isinstance(X, PaddedSequences):
        return X

    feature_names = None
    if isinstance(X, np.ndarray):
        if X.ndim == 2:
            X = X[:, :, None]
        if X.ndim != 3:
            raise ValueError(
                "array input must be (n_sequences, n_timesteps, n_features); "
                f"got shape {X.shape}"
            )
        seqs = [np.asarray(s, dtype=float) for s in X]
    elif isinstance(X, Sequence) and not isinstance(X, (str, bytes)):
        seqs = []
        for s in X:
            if isinstance(s, pd.DataFrame):
                names = [str(c) for c in s.columns]
                if feature_names is None:
                    feature_names = names
                elif names != feature_names:
                    raise ValueError("all DataFrame sequences must share the same columns")
                s = s.to_numpy(dtype=float)
            s = np.asarray(s, dtype=float)
            if s.ndim == 1:
                s = s[:, None]
            if s.ndim != 2:
                raise ValueError(f"each sequence must be 2-D (timesteps, features); got {s.shape}")
            seqs.append(s)
    else:
        raise TypeError(
            "sequences must be a 3-D array or a list of 2-D arrays/DataFrames; "
            f"got {type(X).__name__}"
        )

    if not seqs:
        raise ValueError("no sequences given")
    n_feat = seqs[0].shape[1]
    for s in seqs:
        if s.shape[1] != n_feat:
            raise ValueError("all sequences must have the same number of features")
        if s.shape[0] == 0:
            raise ValueError("empty sequence: every sequence needs at least one sample")
        if not np.all(np.isfinite(s)):
            raise ValueError("sequences contain NaN or inf")
    if feature_names is None:
        feature_names = [f"x{j}" for j in range(n_feat)]

    n = len(seqs)
    lengths = np.array([s.shape[0] for s in seqs])
    max_len = int(lengths.max())

    raw = np.zeros((n, n_feat, max_len))
    tau_raw = np.zeros((n, max_len))
    valid = np.zeros((n, max_len), dtype=bool)
    for i, s in enumerate(seqs):
        length = s.shape[0]
        raw[i, :, :length] = s.T
        tau_raw[i, :length] = (np.arange(length) + 0.5) / length
        valid[i, :length] = True

    # Descending sort, padding last. Stable so ties keep temporal order, which
    # makes the result deterministic (WOWA's value does not depend on how ties
    # are broken -- tied values contribute the same amount either way).
    keyed = np.where(valid[:, None, :], -raw, np.inf)
    order = np.argsort(keyed, axis=2, kind="stable")
    values = np.take_along_axis(raw, order, axis=2)
    tau = np.take_along_axis(np.broadcast_to(tau_raw[:, None, :], raw.shape), order, axis=2)
    mask = np.take_along_axis(np.broadcast_to(valid[:, None, :], raw.shape), order, axis=2)
    values = np.where(mask, values, 0.0)
    return PaddedSequences(values, tau, mask, lengths, feature_names)


# --------------------------------------------------------------------------
# The operator
# --------------------------------------------------------------------------


def temporal_partition(tau: np.ndarray, centers: np.ndarray, widths: np.ndarray,
                       log: bool = False) -> np.ndarray:
    """Ruspini partition of normalised time into ``K`` fuzzy segments.

    ``mu_k(tau) = g_k(tau) / sum_j g_j(tau)`` with Gaussian ``g_k``. Returned
    with a trailing ``K`` axis. Computed in log space so very narrow widths
    degrade to a crisp nearest-centre partition instead of to 0/0.
    """
    tau = np.asarray(tau, dtype=float)[..., None]
    logits = -0.5 * ((tau - centers) / widths) ** 2
    log_mu = logits - _logsumexp(logits, axis=-1)
    return log_mu if log else np.exp(log_mu)


def _safe_lambda(lam):
    lam = np.asarray(lam, dtype=float)
    # lam = 0 is the identity quantifier, a removable singularity of the
    # closed form; |lam| < 1e-6 is replaced by +/-1e-6, which moves Q by <1e-7.
    return np.where(np.abs(lam) < 1e-6, np.where(lam < 0, -1e-6, 1e-6), lam)


def exponential_quantifier(x, lam, with_grad: bool = False):
    """RIM quantifier ``Q(x) = (1 - exp(-lam x)) / (1 - exp(-lam))`` on [0, 1].

    ``Q(0) = 0``, ``Q(1) = 1``, increasing for every ``lam``; ``lam = 0`` is
    the identity. With ``with_grad`` also returns ``dQ/dx`` and ``dQ/dlam``.
    """
    x = np.asarray(x, dtype=float)
    lam = _safe_lambda(lam)
    denom = np.expm1(-lam)
    q = np.expm1(-lam * x) / denom
    if not with_grad:
        return q
    e = np.exp(-lam * x)
    dq_dx = -lam * e / denom
    dq_dlam = (-x * e + q * np.exp(-lam)) / denom
    return q, dq_dx, dq_dlam


def quantifier_orness(lam) -> np.ndarray:
    """Orness ``integral_0^1 Q`` of the exponential quantifier.

    ``1/(1 - exp(-lam)) - 1/lam``: 0.5 at ``lam = 0`` (mean), -> 1 as
    ``lam -> +inf`` (max), -> 0 as ``lam -> -inf`` (min).
    """
    lam = np.asarray(lam, dtype=float)
    safe = _safe_lambda(lam)
    out = -1.0 / np.expm1(-safe) - 1.0 / safe
    return np.where(np.abs(lam) < 1e-4, 0.5 + lam / 12.0, out)


def wowa(values_desc: np.ndarray, importance: np.ndarray, lam) -> np.ndarray:
    """Weighted OWA of values already sorted descending along the last axis.

    Args:
        values_desc: ``(..., L)`` values, sorted descending.
        importance: ``(..., L)`` non-negative importance weights aligned with
            ``values_desc``; normalised here. Zero-weight entries (padding,
            out-of-segment samples) contribute nothing.
        lam: exponential-quantifier parameter, broadcastable to
            ``values_desc[..., 0]``. 0 = weighted mean, >0 max-like, <0 min-like.

    Returns:
        ``(...)`` aggregated values.
    """
    p = np.asarray(importance, dtype=float)
    total = p.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("every aggregation needs positive total importance")
    p = p / total
    lam = np.asarray(lam, dtype=float)[..., None]
    s = np.clip(np.cumsum(p, axis=-1), 0.0, 1.0)
    s_prev = np.clip(s - p, 0.0, 1.0)
    weights = exponential_quantifier(s, lam) - exponential_quantifier(s_prev, lam)
    return np.sum(weights * values_desc, axis=-1)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))


def _aggregate(seqs: PaddedSequences, centers, widths, lambdas, with_grad: bool = False):
    """``(N, K, F)`` aggregated features. ``lambdas`` has shape ``(K, F)``.

    With ``with_grad=True`` also returns a function mapping ``dL/dZ`` (shape
    ``(N, K, F)``) to ``(dL/dcenters, dL/dwidths, dL/dlambdas)``. The gradient is
    exact (checked against finite differences in the test suite); it is what
    makes fitting take seconds rather than minutes.
    """
    centers = np.asarray(centers, dtype=float)
    widths = np.asarray(widths, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    mask = seqs.mask                                          # (N, F, L)
    diff = seqs.tau[..., None] - centers                      # (N, F, L, K)
    logits = -0.5 * (diff / widths) ** 2
    mu = np.exp(logits - _logsumexp(logits, axis=-1))         # Ruspini partition
    log_mu = np.where(mask[..., None], np.log(np.maximum(mu, 1e-300)), -np.inf)
    log_p = log_mu - _logsumexp(log_mu, axis=2)
    p = np.moveaxis(np.exp(log_p), -1, 1)                     # (N, K, F, L)

    s = np.clip(np.cumsum(p, axis=-1), 0.0, 1.0)
    s_prev = np.clip(s - p, 0.0, 1.0)
    lam = lambdas[None, :, :, None]
    v = seqs.values[:, None, :, :]                            # (N, 1, F, L)
    if not with_grad:
        w = exponential_quantifier(s, lam) - exponential_quantifier(s_prev, lam)
        return np.sum(w * v, axis=-1)
    q_s, dq_s, dlam_s = exponential_quantifier(s, lam, with_grad=True)
    q_sp, _, dlam_sp = exponential_quantifier(s_prev, lam, with_grad=True)
    Z = np.sum((q_s - q_sp) * v, axis=-1)

    def backward(gZ):
        g_lam = np.einsum("nkf,nkf->kf", gZ, np.sum((dlam_s - dlam_sp) * v, axis=-1))

        # p: telescope Z = sum_l Q(S_l) (v_l - v_{l+1}), so dZ/dS_l = Q'(S_l) dv_l
        # and dZ/dp_j = sum_{l >= j} dZ/dS_l.
        dv = v - np.concatenate([v[..., 1:], np.zeros_like(v[..., :1])], axis=-1)
        d_s = dq_s * dv
        d_p = np.flip(np.cumsum(np.flip(d_s, -1), axis=-1), -1)
        g_p = gZ[..., None] * d_p                             # (N, K, F, L)

        # p = softmax over time of log_mu (masked).
        g_logmu = p * (g_p - np.sum(p * g_p, axis=-1, keepdims=True))
        g_logmu = np.moveaxis(g_logmu, 1, -1)                 # (N, F, L, K)
        g_logmu = np.where(mask[..., None], g_logmu, 0.0)
        # log_mu = logits - logsumexp_k(logits).
        g_logits = g_logmu - mu * np.sum(g_logmu, axis=-1, keepdims=True)
        g_c = np.sum(g_logits * diff / widths ** 2, axis=(0, 1, 2))
        g_w = np.sum(g_logits * diff ** 2 / widths ** 3, axis=(0, 1, 2))
        return g_c, g_w, g_lam

    return Z, backward


# --------------------------------------------------------------------------
# Learned transformer
# --------------------------------------------------------------------------


class FuzzyTemporalAggregator(TransformerMixin, BaseEstimator):
    """Sequence -> row transformer with a learned fuzzy aggregation over time.

    Produces ``n_segments * n_features`` columns, ``{feature}__seg{k}``, where
    segment ``k`` is the ``k``-th learned fuzzy region of normalised time
    (ordered by centre) and the value is the WOWA of that feature's samples
    weighted by their membership in the region.

    Args:
        n_segments: Number of fuzzy temporal segments ``K``.
        fit_segments: Learn segment centres and widths. False keeps the even
            partition ``(k + 0.5) / K``.
        fit_quantifier: Learn the per-(segment, feature) quantifier ``lam``.
            False keeps ``lam = 0`` (importance-weighted means).
        init_width: Initial segment width as a fraction of segment spacing
            ``1/K``. Small values start near the crisp K-th split.
        ridge: L2 penalty of the surrogate ridge head (on standardised
            features and targets).
        cv: Folds for the cross-validated surrogate objective.
        max_evals: Objective+gradient evaluation budget per start, spent by
            ``optimizer_utils.projected_lbfgs_solve`` (the project's
            exact-gradient sub-solver; no direct ``scipy.optimize``).
        n_restarts: Extra starts from random segment centres, on top of the
            always-first even-partition start. The best CV objective wins.
        random_state: Seed for folds and restarts.

    Attributes:
        centers_, widths_: ``(K,)`` learned segment parameters, sorted by centre.
        lambdas_: ``(K, F)`` learned quantifier parameters.
        orness_: ``(K, F)`` quantifier orness; >0.5 max-like, <0.5 min-like.
        objective_init_: surrogate CV loss at the even-partition, lam=0 start.
        objective_: surrogate CV loss at the learned parameters (never larger).
        feature_names_in_: input feature names.
    """

    def __init__(
        self,
        n_segments: int = 3,
        fit_segments: bool = True,
        fit_quantifier: bool = True,
        init_width: float = 0.25,
        ridge: float = 1.0,
        cv: int = 5,
        max_evals: int = 300,
        n_restarts: int = 2,
        random_state: int | None = 42,
    ):
        self.n_segments = n_segments
        self.fit_segments = fit_segments
        self.fit_quantifier = fit_quantifier
        self.init_width = init_width
        self.ridge = ridge
        self.cv = cv
        self.max_evals = max_evals
        self.n_restarts = n_restarts
        self.random_state = random_state

    # -- parameter vector <-> arrays ---------------------------------------

    def _unpack(self, theta, n_feat):
        k = self.n_segments
        centers = self._c0.copy()
        widths = self._w0.copy()
        lambdas = np.zeros((k, n_feat))
        i = 0
        if self.fit_segments:
            centers = theta[i:i + k].copy()
            i += k
            widths = np.exp(_LOG_WIDTH_SCALE * theta[i:i + k])
            i += k
        if self.fit_quantifier:
            lambdas = _LAMBDA_SCALE * theta[i:i + k * n_feat].reshape(k, n_feat)
        return centers, widths, lambdas

    def _pack(self, centers, widths, n_feat):
        parts, bounds = [], []
        k = self.n_segments
        if self.fit_segments:
            parts += [np.asarray(centers, dtype=float), np.log(widths) / _LOG_WIDTH_SCALE]
            bounds += [(0.0, 1.0)] * k
            bounds += [tuple(np.log(_WIDTH_BOUNDS) / _LOG_WIDTH_SCALE)] * k
        if self.fit_quantifier:
            parts.append(np.zeros(k * n_feat))
            bound = _LAMBDA_BOUND / _LAMBDA_SCALE
            bounds += [(-bound, bound)] * (k * n_feat)
        theta = np.concatenate(parts) if parts else np.zeros(0)
        return theta, bounds

    # -- surrogate objective -----------------------------------------------

    def _cv_loss(self, Z: np.ndarray, Y: np.ndarray, folds, with_grad: bool = False):
        """Mean held-out MSE of a ridge head, standardised per fold.

        Features are standardised with each fold's *training* statistics, so
        no held-out information reaches the head. With ``with_grad`` also
        returns ``dL/dZ`` -- through the ridge solve and through the
        standardisation statistics, which both depend on ``Z``.
        """
        losses = []
        gZ = np.zeros_like(Z) if with_grad else None
        n_folds = len(folds)
        for tr, va in folds:
            zt, zv = Z[tr], Z[va]
            yt, yv = Y[tr], Y[va]
            zm, zs = zt.mean(0), zt.std(0)
            live = zs > 1e-12
            zs = np.where(live, zs, 1.0)
            ym, ys = yt.mean(0), yt.std(0)
            ys = np.where(ys > 1e-12, ys, 1.0)
            at = (zt - zm) / zs
            av = (zv - zm) / zs
            bt = (yt - ym) / ys
            bv = (yv - ym) / ys
            gram = at.T @ at + self.ridge * np.eye(at.shape[1])
            coef = np.linalg.solve(gram, at.T @ bt)
            resid = av @ coef - bv
            losses.append(np.mean(resid ** 2))
            if not with_grad:
                continue
            g_resid = 2.0 * resid / (resid.size * n_folds)
            g_av = g_resid @ coef.T
            pmat = np.linalg.solve(gram, av.T @ g_resid)
            g_at = (bt - at @ coef) @ pmat.T - at @ pmat @ coef.T
            # Back through a = (z - m) / s, m and s from the training rows.
            g_m = -(g_at.sum(0) + g_av.sum(0)) / zs
            g_s = -(np.sum(g_at * at, 0) + np.sum(g_av * av, 0)) / zs
            g_s = np.where(live, g_s, 0.0)
            n_t = len(tr)
            gZ[tr] += g_at / zs + g_m / n_t + g_s * at / n_t
            gZ[va] += g_av / zs
        loss = float(np.mean(losses))
        return (loss, gZ) if with_grad else loss

    def _objective(self, theta, seqs, Y, folds, with_grad: bool = True):
        n_feat = seqs.n_features
        centers, widths, lambdas = self._unpack(theta, n_feat)
        Z3, backward = _aggregate(seqs, centers, widths, lambdas, with_grad=True)
        Z = Z3.reshape(seqs.n_sequences, -1)
        if not with_grad:
            return self._cv_loss(Z, Y, folds)
        loss, gZ = self._cv_loss(Z, Y, folds, with_grad=True)
        g_c, g_w, g_a = backward(gZ.reshape(Z3.shape))
        parts = []
        if self.fit_segments:
            parts += [g_c, g_w * widths * _LOG_WIDTH_SCALE]
        if self.fit_quantifier:
            parts.append((g_a * _LAMBDA_SCALE).ravel())
        grad = np.concatenate(parts) if parts else np.zeros(0)
        return loss, grad

    # -- sklearn API -------------------------------------------------------

    def fit(self, X, y=None):
        """Learn the temporal partition and quantifiers.

        With ``y=None`` nothing is learned: the transformer keeps the even
        partition and ``lam = 0``, i.e. soft "mean of each K-th".
        """
        if int(self.n_segments) < 1:
            raise ValueError("n_segments must be >= 1")
        seqs = pad_sequences(X)
        k = int(self.n_segments)
        f = seqs.n_features
        self.feature_names_in_ = np.asarray(seqs.feature_names, dtype=object)
        self.n_features_in_ = f

        self._c0 = (np.arange(k) + 0.5) / k
        self._w0 = np.full(k, float(self.init_width) / k)
        centers, widths, lambdas = self._c0.copy(), self._w0.copy(), np.zeros((k, f))
        self.objective_init_ = None
        self.objective_ = None

        if y is not None:
            Y = np.asarray(y, dtype=float)
            if Y.ndim == 1:
                Y = Y[:, None]
            if Y.shape[0] != seqs.n_sequences:
                raise ValueError(
                    f"y has {Y.shape[0]} rows but there are {seqs.n_sequences} sequences"
                )
            if not np.all(np.isfinite(Y)):
                raise ValueError("y contains NaN or inf")
            n_splits = min(int(self.cv), seqs.n_sequences)
            if n_splits < 2:
                raise ValueError("need at least 2 sequences to cross-validate")
            folds = list(KFold(n_splits, shuffle=True,
                               random_state=self.random_state).split(Y))

            theta0, bounds = self._pack(self._c0, self._w0, f)
            self.objective_init_ = self._objective(theta0, seqs, Y, folds, with_grad=False)
            best_theta, best_loss = theta0, self.objective_init_

            if theta0.size:
                rng = np.random.default_rng(self.random_state)
                starts = [theta0]
                for _ in range(int(self.n_restarts) if self.fit_segments else 0):
                    c = np.sort(rng.uniform(0.02, 0.98, size=k))
                    starts.append(self._pack(c, self._w0, f)[0])
                for start in starts:
                    res = projected_lbfgs_solve(
                        lambda t: self._objective(t, seqs, Y, folds),
                        start, bounds, max_evals=int(self.max_evals),
                    )
                    if res.fun < best_loss:
                        best_theta, best_loss = res.x, float(res.fun)
            centers, widths, lambdas = self._unpack(best_theta, f)
            self.objective_ = best_loss

        order = np.argsort(centers, kind="stable")
        self.centers_ = centers[order]
        self.widths_ = widths[order]
        self.lambdas_ = lambdas[order]
        self.orness_ = quantifier_orness(self.lambdas_)
        return self

    def transform(self, X) -> pd.DataFrame:
        check_is_fitted(self, "centers_")
        seqs = pad_sequences(X)
        if seqs.n_features != self.n_features_in_:
            raise ValueError(
                f"sequences have {seqs.n_features} features; fitted on {self.n_features_in_}"
            )
        Z = _aggregate(seqs, self.centers_, self.widths_, self.lambdas_)
        return pd.DataFrame(Z.reshape(seqs.n_sequences, -1),
                            columns=self.get_feature_names_out())

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "centers_")
        names = self.feature_names_in_ if input_features is None else input_features
        return np.asarray(
            [f"{name}__seg{k}" for k in range(len(self.centers_)) for name in names],
            dtype=object,
        )

    def segment_report(self) -> pd.DataFrame:
        """One row per (segment, feature): where the segment sits and what it computes."""
        check_is_fitted(self, "centers_")
        rows = []
        for k in range(len(self.centers_)):
            for j, name in enumerate(self.feature_names_in_):
                rows.append({
                    "segment": k,
                    "feature": name,
                    "center": self.centers_[k],
                    "width": self.widths_[k],
                    "lambda": self.lambdas_[k, j],
                    "orness": self.orness_[k, j],
                })
        return pd.DataFrame(rows)

    def placement_report(self, X) -> dict:
        """Uniformity diagnostic for where the learned segments sit on the data (#127).

        Each sampled time point ``tau`` (pooled over sequences and features) is
        split across segments by the Ruspini partition. ``mass_k`` is the
        fraction of sampled time that segment ``k`` sees. Even coverage is
        ``1/K`` per segment; a segment with near-zero mass is starved and a
        cluster of centres in one dense region is bunching.

        This is a diagnostic, not a penalty: a learned placement is meant to
        concentrate on the signal, so low uniformity is not an error by itself.
        Compare it with the even-partition baseline and with downstream error.

        Args:
            X: sequences, same format as :meth:`fit`.

        Returns:
            dict with ``segments`` (DataFrame: segment, center, width, mass,
            even_mass, deviation), ``normalized_entropy`` (1 = even coverage,
            0 = all mass in one segment), and ``max_deviation`` (largest
            ``|mass_k - 1/K|``).
        """
        check_is_fitted(self, "centers_")
        seqs = pad_sequences(X)
        tau = seqs.tau[seqs.mask]
        if tau.size == 0:
            raise ValueError("no sampled time points to place segments against")
        mu = temporal_partition(tau, self.centers_, self.widths_)
        mass = mu.mean(axis=0)
        mass = mass / mass.sum()
        k = len(mass)
        even = np.full(k, 1.0 / k)
        if k == 1:
            entropy = 1.0
        else:
            nz = mass[mass > 0]
            entropy = float(-np.sum(nz * np.log(nz)) / np.log(k))
        segments = pd.DataFrame({
            "segment": np.arange(k),
            "center": self.centers_,
            "width": self.widths_,
            "mass": mass,
            "even_mass": even,
            "deviation": mass - even,
        })
        return {
            "segments": segments,
            "normalized_entropy": entropy,
            "max_deviation": float(np.max(np.abs(mass - even))),
        }


# --------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------


class TribbleTimeSeriesRegressor(RegressorMixin, BaseEstimator):
    """`MimoGaussianPredictor` over a learned fuzzy aggregation of each sequence.

    ``fit(X, y)`` takes one sequence per target row (3-D array or list of
    2-D arrays/DataFrames, lengths may differ) -- e.g. one flight cycle per
    remaining-useful-life label. A :class:`FuzzyTemporalAggregator` learns how
    to collapse each sequence to a row; the base estimator is then fitted on
    those rows exactly as it would be on hand-engineered features.

    Args:
        n_segments, fit_segments, fit_quantifier, init_width, ridge, cv, max_evals,
        n_restarts: forwarded to :class:`FuzzyTemporalAggregator`.
        base_estimator: Tabular regressor fitted on the aggregated rows; cloned
            before fitting. Defaults to ``MimoGaussianPredictor`` with its
            defaults and this ``random_state``.
        random_state: Seed for the aggregator and the default base estimator.
    """

    def __init__(
        self,
        n_segments: int = 3,
        fit_segments: bool = True,
        fit_quantifier: bool = True,
        init_width: float = 0.25,
        ridge: float = 1.0,
        cv: int = 5,
        max_evals: int = 300,
        n_restarts: int = 2,
        base_estimator=None,
        random_state: int | None = 42,
    ):
        self.n_segments = n_segments
        self.fit_segments = fit_segments
        self.fit_quantifier = fit_quantifier
        self.init_width = init_width
        self.ridge = ridge
        self.cv = cv
        self.max_evals = max_evals
        self.n_restarts = n_restarts
        self.base_estimator = base_estimator
        self.random_state = random_state

    def fit(self, X, y):
        self.aggregator_ = FuzzyTemporalAggregator(
            n_segments=self.n_segments,
            fit_segments=self.fit_segments,
            fit_quantifier=self.fit_quantifier,
            init_width=self.init_width,
            ridge=self.ridge,
            cv=self.cv,
            max_evals=self.max_evals,
            n_restarts=self.n_restarts,
            random_state=self.random_state,
        ).fit(X, y)
        Z = self.aggregator_.transform(X)
        if self.base_estimator is None:
            base = MimoGaussianPredictor(random_state=self.random_state)
        else:
            base = clone(self.base_estimator)
        self._y_1d = np.ndim(y) == 1
        self.estimator_ = base.fit(Z, y)
        return self

    def transform(self, X) -> pd.DataFrame:
        """The aggregated rows the base estimator sees."""
        check_is_fitted(self, "estimator_")
        return self.aggregator_.transform(X)

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        pred = self.estimator_.predict(self.aggregator_.transform(X))
        if self._y_1d and isinstance(pred, pd.DataFrame):
            return pred.iloc[:, 0].to_numpy()
        return pred
