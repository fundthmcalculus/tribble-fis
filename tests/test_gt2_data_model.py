"""Tests for the GT2 (general type-2) alpha-plane data model (`gauss_data.py`).

These validate exactly the containment properties issue #122's research spike
(`docs/gt2-evaluation.md`, phase 1) called for before any inference code is
trusted: alpha=0 must recover today's IT2 footprint exactly, alpha=1 must
collapse to the principal membership function, and the interval must narrow
monotonically in between.
"""

import numpy as np
import pytest

from tribblefis.gauss_data import (
    GT2GaussianMembership, IT2GaussianMembership, GaussianMembership,
    GT2TrapezoidMembership, IT2TrapezoidMembership, TrapezoidMembership,
    GT2TriangularMembership, IT2TriangularMembership, TriangularMembership,
    widen_membership, to_it2_membership, to_gt2_membership,
)


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


# ---------------------------------------------------------------------------
# GT2TrapezoidMembership -- same invariants, mirroring the Gaussian suite above.
# ---------------------------------------------------------------------------

@pytest.fixture
def gt2_trap():
    return GT2TrapezoidMembership.create(
        b=5.0, c=6.0, upper_a=1.0, upper_d=10.0, lower_a=4.0, lower_d=7.0, principal_a=3.0, principal_d=8.0,
    )


def test_trap_alpha_zero_recovers_it2_footprint_exactly(gt2_trap):
    a0 = gt2_trap.alpha_cut(0.0)
    assert isinstance(a0, IT2TrapezoidMembership)
    assert a0.upper_mf == gt2_trap.upper_mf
    assert a0.lower_mf == gt2_trap.lower_mf


def test_trap_alpha_one_collapses_to_principal(gt2_trap):
    a1 = gt2_trap.alpha_cut(1.0)
    assert a1.upper_mf.a == pytest.approx(gt2_trap.principal_mf.a)
    assert a1.upper_mf.d == pytest.approx(gt2_trap.principal_mf.d)
    assert a1.upper_mf == a1.lower_mf


def test_trap_alpha_cut_narrows_monotonically(gt2_trap):
    alphas = np.linspace(0.0, 1.0, 11)
    widths = [gt2_trap.alpha_cut(a).upper_mf.d - gt2_trap.alpha_cut(a).upper_mf.a for a in alphas]
    assert all(w1 <= w0 + 1e-12 for w0, w1 in zip(widths, widths[1:]))


def test_trap_alpha_cut_shares_plateau_across_the_whole_footprint(gt2_trap):
    for alpha in (0.0, 0.3, 0.7, 1.0):
        ac = gt2_trap.alpha_cut(alpha)
        assert ac.upper_mf.b == ac.lower_mf.b == gt2_trap.principal_mf.b
        assert ac.upper_mf.c == ac.lower_mf.c == gt2_trap.principal_mf.c


def test_trap_create_defaults_principal_to_midpoint():
    mf = GT2TrapezoidMembership.create(b=5.0, c=6.0, upper_a=0.0, upper_d=10.0, lower_a=4.0, lower_d=7.0)
    assert mf.principal_mf.a == pytest.approx(2.0)
    assert mf.principal_mf.d == pytest.approx(8.5)


# ---------------------------------------------------------------------------
# GT2TriangularMembership -- same invariants, plus shoulder passthrough.
# ---------------------------------------------------------------------------

@pytest.fixture
def gt2_tri():
    return GT2TriangularMembership.create(
        b=5.0, upper_a=1.0, upper_c=10.0, lower_a=4.0, lower_c=7.0, principal_a=3.0, principal_c=8.0,
    )


def test_tri_alpha_zero_recovers_it2_footprint_exactly(gt2_tri):
    a0 = gt2_tri.alpha_cut(0.0)
    assert isinstance(a0, IT2TriangularMembership)
    assert a0.upper_mf == gt2_tri.upper_mf
    assert a0.lower_mf == gt2_tri.lower_mf


def test_tri_alpha_one_collapses_to_principal(gt2_tri):
    a1 = gt2_tri.alpha_cut(1.0)
    assert a1.upper_mf.a == pytest.approx(gt2_tri.principal_mf.a)
    assert a1.upper_mf == a1.lower_mf


def test_tri_alpha_cut_narrows_monotonically(gt2_tri):
    alphas = np.linspace(0.0, 1.0, 11)
    widths = [gt2_tri.alpha_cut(a).upper_mf.c - gt2_tri.alpha_cut(a).upper_mf.a for a in alphas]
    assert all(w1 <= w0 + 1e-12 for w0, w1 in zip(widths, widths[1:]))


def test_tri_alpha_cut_shares_apex_across_the_whole_footprint(gt2_tri):
    for alpha in (0.0, 0.3, 0.7, 1.0):
        ac = gt2_tri.alpha_cut(alpha)
        assert ac.upper_mf.b == ac.lower_mf.b == gt2_tri.principal_mf.b


def test_tri_alpha_cut_passes_shoulder_through_unchanged():
    """A left-shoulder leg (`a=-inf`) has no finite spread to cut -- every
    alpha level must leave it exactly `-inf`, not NaN from `inf - inf` arithmetic."""
    mf = GT2TriangularMembership.create(
        b=5.0, upper_a=-np.inf, upper_c=10.0, lower_a=-np.inf, lower_c=7.0,
        principal_a=-np.inf, principal_c=8.0,
    )
    for alpha in (0.0, 0.5, 1.0):
        ac = mf.alpha_cut(alpha)
        assert ac.upper_mf.a == -np.inf
        assert ac.lower_mf.a == -np.inf


# ---------------------------------------------------------------------------
# widen_membership / to_it2_membership / to_gt2_membership
# ---------------------------------------------------------------------------

def test_widen_gaussian_keeps_mu_scales_sigma():
    mf = GaussianMembership(mu=3.0, sigma=1.0)
    upper, lower = widen_membership(mf, uncertainty_width=0.5)
    assert upper.mu == lower.mu == 3.0
    assert lower.sigma < mf.sigma < upper.sigma


def test_widen_trapezoid_keeps_plateau_scales_slopes():
    mf = TrapezoidMembership(a=1.0, b=4.0, c=6.0, d=9.0)
    upper, lower = widen_membership(mf, uncertainty_width=0.5)
    assert upper.b == lower.b == mf.b
    assert upper.c == lower.c == mf.c
    assert lower.a > mf.a > upper.a
    assert lower.d < mf.d < upper.d


def test_widen_triangular_keeps_apex_scales_legs():
    mf = TriangularMembership(a=1.0, b=4.0, c=9.0)
    upper, lower = widen_membership(mf, uncertainty_width=0.5)
    assert upper.b == lower.b == mf.b
    assert lower.a > mf.a > upper.a
    assert lower.c < mf.c < upper.c


def test_widen_rejects_trapezoid_shoulders():
    mf = TrapezoidMembership(a=-np.inf, b=4.0, c=6.0, d=9.0)
    with pytest.raises(ValueError):
        widen_membership(mf, uncertainty_width=0.5)


def test_widen_rejects_triangular_shoulders():
    mf = TriangularMembership(a=-np.inf, b=4.0, c=9.0)
    with pytest.raises(ValueError):
        widen_membership(mf, uncertainty_width=0.5)


@pytest.mark.parametrize("mf,it2_cls", [
    (GaussianMembership(mu=0.0, sigma=1.0), IT2GaussianMembership),
    (TrapezoidMembership(a=1.0, b=4.0, c=6.0, d=9.0), IT2TrapezoidMembership),
    (TriangularMembership(a=1.0, b=4.0, c=9.0), IT2TriangularMembership),
])
def test_to_it2_membership_dispatches_by_type(mf, it2_cls):
    upper, lower = widen_membership(mf, uncertainty_width=0.5)
    it2_mf = to_it2_membership(upper, lower)
    assert isinstance(it2_mf, it2_cls)
    assert it2_mf.upper_mf is upper
    assert it2_mf.lower_mf is lower


@pytest.mark.parametrize("mf,gt2_cls", [
    (GaussianMembership(mu=0.0, sigma=1.0), GT2GaussianMembership),
    (TrapezoidMembership(a=1.0, b=4.0, c=6.0, d=9.0), GT2TrapezoidMembership),
    (TriangularMembership(a=1.0, b=4.0, c=9.0), GT2TriangularMembership),
])
def test_to_gt2_membership_dispatches_by_type(mf, gt2_cls):
    upper, lower = widen_membership(mf, uncertainty_width=0.5)
    gt2_mf = to_gt2_membership(upper, lower, principal_mf=mf)
    assert isinstance(gt2_mf, gt2_cls)
    assert gt2_mf.principal_mf is mf
