"""Tests for the GT2 (general type-2) alpha-plane data model (`gauss_data.py`).

These validate exactly the containment properties issue #122's research spike
(`docs/gt2-evaluation.md`, phase 1) called for before any inference code is
trusted: alpha=0 must recover today's IT2 footprint exactly, alpha=1 must
collapse to the principal membership function, and the interval must narrow
monotonically in between.
"""

import numpy as np
import pytest

from tribblefis.gauss_data import GT2GaussianMembership, IT2GaussianMembership, GaussianMembership


@pytest.fixture
def gt2_mf():
    return GT2GaussianMembership.create(
        upper_mu=1.0, upper_sigma=2.0, lower_mu=1.0, lower_sigma=0.5, principal_sigma=1.2,
    )


def test_alpha_zero_recovers_it2_footprint_exactly(gt2_mf):
    a0 = gt2_mf.alpha_cut(0.0)
    assert isinstance(a0, IT2GaussianMembership)
    assert a0.upper_mf.sigma == gt2_mf.upper_mf.sigma
    assert a0.lower_mf.sigma == gt2_mf.lower_mf.sigma
    assert a0.upper_mf.mu == a0.lower_mf.mu == gt2_mf.principal_mf.mu


def test_alpha_one_collapses_to_principal(gt2_mf):
    a1 = gt2_mf.alpha_cut(1.0)
    assert a1.upper_mf.sigma == pytest.approx(gt2_mf.principal_mf.sigma)
    assert a1.lower_mf.sigma == pytest.approx(gt2_mf.principal_mf.sigma)
    # A collapsed IT2 footprint (upper == lower) is a degenerate, but valid,
    # crisp Type-1 membership -- firing_upper == firing_lower everywhere.
    assert a1.upper_mf.sigma == a1.lower_mf.sigma


def test_alpha_cut_narrows_monotonically(gt2_mf):
    alphas = np.linspace(0.0, 1.0, 11)
    widths = [gt2_mf.alpha_cut(a).upper_mf.sigma - gt2_mf.alpha_cut(a).lower_mf.sigma for a in alphas]
    assert all(w1 <= w0 + 1e-12 for w0, w1 in zip(widths, widths[1:]))
    assert all(w >= -1e-12 for w in widths)


def test_alpha_cut_shares_mu_across_the_whole_footprint(gt2_mf):
    """Mirrors the invariant `it2_refine._iter_it2_gaussian_slots` documents for
    IT2: a shared peak is what keeps the alpha-cut interval well-ordered."""
    for alpha in (0.0, 0.3, 0.7, 1.0):
        ac = gt2_mf.alpha_cut(alpha)
        assert ac.upper_mf.mu == ac.lower_mf.mu


def test_create_defaults_principal_sigma_to_midpoint():
    """Omitting `principal_sigma` should behave like a uniform secondary grade
    -- i.e. every alpha-plane's centroid concept collapses to IT2's own
    midpoint reduction, since the "most likely" sigma is just the interval
    center with no additional information."""
    mf = GT2GaussianMembership.create(upper_mu=0.0, upper_sigma=2.0, lower_mu=0.0, lower_sigma=0.0)
    assert mf.principal_mf.sigma == pytest.approx(1.0)


def test_principal_sigma_out_of_order_is_not_silently_fixed_by_alpha_cut():
    """`alpha_cut` is a pure interpolation -- it does not clamp `principal_mf`
    into `[lower_mf.sigma, upper_mf.sigma]` itself. Callers that construct a
    `GT2GaussianMembership` directly (as opposed to via `gt2_refine`'s
    slot-apply function, which does enforce ordering) are responsible for the
    invariant; this test documents that `alpha_cut` does not paper over a
    violation, so a bug upstream is visible rather than silently absorbed."""
    mf = GT2GaussianMembership(
        upper_mf=GaussianMembership(mu=0.0, sigma=1.0),
        lower_mf=GaussianMembership(mu=0.0, sigma=2.0),  # deliberately inverted
        principal_mf=GaussianMembership(mu=0.0, sigma=1.5),
    )
    a0 = mf.alpha_cut(0.0)
    assert a0.upper_mf.sigma == 1.0
    assert a0.lower_mf.sigma == 2.0
