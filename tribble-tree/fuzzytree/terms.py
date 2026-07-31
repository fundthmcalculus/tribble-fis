"""Per-variable linguistic term construction for tree splits.

At each internal node we partition one input variable into ``n_terms`` fuzzy
linguistic terms (e.g. Low / Med / High). We use overlapping trapezoids placed
at *weighted* quantile knots of the samples reaching the node, with **open
shoulders** on the first and last term so the whole axis is always covered.

Open shoulders matter: if a test point falls beyond the training range and every
term evaluated to 0 there, the product of memberships down every path would be 0,
the leaf firing row would normalise to all-zeros, and the prediction would
silently collapse to 0 (the zero-firing fallback in
``tribblefis.regression._normalize_firing_strengths``). Extending the outer
plateaus to +/- infinity keeps membership at 1.0 in the tails, giving graceful
extrapolation.

Trapezoids are preferred over data-driven Gaussian fits here because the goal is
*interpretability*: a fixed count of nameable, ordered terms ("Low" is always the
leftmost band) reads far better than per-node mu/sigma. A Gaussian variant is
provided for callers who want smooth, unbounded memberships.
"""

from __future__ import annotations

import numpy as np

from tribblefis.gauss_data import (
    AnyMembership,
    GaussianMembership,
    TrapezoidMembership,
)

# A large finite offset used to emulate an open (infinite) shoulder. Trapezoid
# evaluation uses finite slopes, so a plateau that starts this far out reads 1.0
# for every realistic value while keeping a valid a<=b<=c<=d ordering.
_OPEN_SHOULDER = 1e12

DEFAULT_TERM_LABELS: tuple[str, ...] = ("Low", "Med", "High")


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, qs: np.ndarray
) -> np.ndarray:
    """Weighted quantiles of ``values`` at probabilities ``qs``.

    Uses the standard "weighted midpoint CDF" estimator so that repeated calls on
    the same soft subset are stable. Falls back to unweighted quantiles when the
    total weight is degenerate.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 1e-12 or not np.isfinite(total):
        return np.quantile(values, qs)

    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w) - 0.5 * w
    cdf /= w.sum()
    return np.interp(qs, cdf, v)


def default_labels(n_terms: int) -> tuple[str, ...]:
    """Sensible default labels for ``n_terms`` bands."""
    if n_terms == 2:
        return ("Low", "High")
    if n_terms == 3:
        return ("Low", "Med", "High")
    return tuple(f"L{i}" for i in range(n_terms))


def build_split_terms(
    values: np.ndarray,
    weights: np.ndarray,
    n_terms: int,
    labels: tuple[str, ...] | None = None,
    style: str = "trapezoid",
) -> list[tuple[str, AnyMembership]]:
    """Build ``n_terms`` ordered fuzzy linguistic terms for one variable.

    Args:
        values: The variable's values for samples reaching this node.
        weights: Fuzzy path weights of those samples (used for weighted quantiles).
        n_terms: Number of terms (bands). Typically 2 or 3.
        labels: Term labels; defaults to Low/Med/High style. Truncated/extended to
            ``n_terms``.
        style: ``"trapezoid"`` (default, open-shouldered) or ``"gaussian"``.

    Returns:
        A list of ``(label, membership)`` pairs sorted left-to-right by centre.
        Returns ``[]`` if the samples are degenerate (no spread), signalling the
        builder to make this node a leaf.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if labels is None:
        labels = default_labels(n_terms)
    if len(labels) < n_terms:
        labels = tuple(labels) + tuple(
            f"L{i}" for i in range(len(labels), n_terms)
        )

    knots = _weighted_quantiles(values, weights, np.linspace(0.0, 1.0, n_terms + 1))

    # Support of the samples that actually reach this node (non-negligible weight).
    if weights.max() > 0:
        present = values[weights > weights.max() * 1e-6]
    else:
        present = values
    lo, hi = float(present.min()), float(present.max())

    # Degenerate variable (all weighted mass at one value) -> cannot split.
    span = hi - lo
    if not np.all(np.isfinite(knots)) or span <= 1e-12:
        return []

    # A feature with a heavy point mass (e.g. a bounded score that is exactly its
    # maximum for most samples) makes the weighted quantiles collapse: interior
    # knots duplicate, and the open-shouldered extreme term then covers the whole
    # range at membership 1.0, so the split fails to partition. Detect that and
    # fall back to evenly-spaced knots across the node's support.
    if np.any(np.diff(knots) < span * 1e-3):
        knots = np.linspace(lo, hi, n_terms + 1)

    if style == "gaussian":
        terms = _gaussian_terms(knots, n_terms)
    else:
        terms = _trapezoid_terms(knots, n_terms)

    labelled = list(zip(labels[:n_terms], terms))
    labelled.sort(key=lambda lt: _center(lt[1]))
    # Reassign labels by sorted position so "Low" is always leftmost.
    return [(labels[i], mf) for i, (_, mf) in enumerate(labelled)]


def _trapezoid_terms(knots: np.ndarray, n_terms: int) -> list[TrapezoidMembership]:
    terms: list[TrapezoidMembership] = []
    for c in range(n_terms):
        lo, hi = float(knots[c]), float(knots[c + 1])
        ramp = 0.5 * (hi - lo)
        # Left shoulder open on the first term, right shoulder open on the last.
        a = -_OPEN_SHOULDER if c == 0 else lo - ramp
        b = -_OPEN_SHOULDER if c == 0 else lo
        cc = _OPEN_SHOULDER if c == n_terms - 1 else hi
        d = _OPEN_SHOULDER if c == n_terms - 1 else hi + ramp
        terms.append(TrapezoidMembership.create(a=a, b=b, c=cc, d=d))
    return terms


def _gaussian_terms(knots: np.ndarray, n_terms: int) -> list[GaussianMembership]:
    terms: list[GaussianMembership] = []
    for c in range(n_terms):
        lo, hi = float(knots[c]), float(knots[c + 1])
        mu = 0.5 * (lo + hi)
        sigma = max((hi - lo) / 2.0, 1e-6)
        terms.append(GaussianMembership.create(mu=mu, sigma=sigma))
    return terms


def _center(mf: AnyMembership) -> float:
    """Representative centre of a membership function, for ordering."""
    if isinstance(mf, TrapezoidMembership):
        # Use the (possibly open) plateau midpoint, clamped so open shoulders do
        # not dominate the ordering.
        b = max(mf.b, -_OPEN_SHOULDER / 2)
        c = min(mf.c, _OPEN_SHOULDER / 2)
        return 0.5 * (b + c)
    return float(mf.mu)
