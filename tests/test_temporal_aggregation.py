"""Test protocol for `tribblefis.temporal_aggregation` (issue #127).

Organised to match the protocol table in docs/temporal-aggregation-design.md:

  A. Aggregation-function axioms (what makes WOWA an aggregation operator)
  B. Special cases / nesting (the hand-engineered baseline is inside the family)
  C. Temporal partition (Ruspini property, crisp limit)
  D. Input handling (variable length, padding cannot leak, validation)
  E. Learning (exact gradient, monotone objective, recovers planted structure)
  F. Estimator contract (sklearn API, determinism, MIMO)

Every test uses small data and a capped optimiser so the file runs in seconds;
accuracy claims live in benchmarks/temporal_aggregation.py, not here.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from tribblefis.temporal_aggregation import (
    FuzzyTemporalAggregator,
    TribbleTimeSeriesRegressor,
    exponential_quantifier,
    pad_sequences,
    quantifier_orness,
    temporal_partition,
    wowa,
)
from tribblefis.temporal_aggregation import _aggregate

RNG = np.random.default_rng(127)


@pytest.fixture(autouse=True)
def _fresh_rng(request):
    """Give every test its own RNG, seeded from its name.

    A single module-level generator made each test's random inputs depend on
    which tests ran before it in the same xdist worker, so CI and local runs
    drew different cases (and CI found ones local never saw).
    """
    global RNG
    RNG = np.random.default_rng(zlib.crc32(request.node.name.encode()))


def _desc(x):
    order = np.argsort(-x, axis=-1, kind="stable")
    return order, np.take_along_axis(x, order, axis=-1)


def _random_wowa_case(n=7):
    x = RNG.normal(size=n)
    p = RNG.uniform(0.01, 1.0, size=n)
    order, xs = _desc(x)
    return x, p, xs, p[order]


def _planted(n_seq=60, phase=0.7, seed=0):
    """Target = height of a bump at `phase` on sensor 0; sensor 1 is noise."""
    rng = np.random.default_rng(seed)
    seqs, y = [], []
    for _ in range(n_seq):
        length = int(rng.integers(30, 60))
        tau = (np.arange(length) + 0.5) / length
        h = rng.uniform(0, 1)
        s0 = rng.normal(0, 0.2, length) + rng.normal(0, 0.5) + \
            1.5 * h * np.exp(-0.5 * ((tau - phase) / 0.05) ** 2)
        s1 = rng.normal(0, 0.2, length)
        seqs.append(np.c_[s0, s1])
        y.append(h)
    return seqs, np.array(y)


FAST = dict(max_evals=80, n_restarts=0, cv=3)


# ---------------------------------------------------------------- A. axioms

@pytest.mark.parametrize("lam", [-150.0, -5.0, 0.0, 2.0, 150.0])
def test_wowa_bounded_by_min_and_max(lam):
    for _ in range(50):
        x, p, xs, ps = _random_wowa_case()
        out = wowa(xs, ps, lam)
        assert x.min() - 1e-12 <= out <= x.max() + 1e-12


@pytest.mark.parametrize("lam", [-150.0, 0.0, 150.0])
def test_wowa_idempotent(lam):
    _, p, _, _ = _random_wowa_case()
    assert wowa(np.full(7, 3.25), p, lam) == pytest.approx(3.25)


@pytest.mark.parametrize("lam", [-20.0, 0.0, 20.0])
def test_wowa_monotone_in_every_argument(lam):
    for _ in range(30):
        x, p, xs, ps = _random_wowa_case()
        base = wowa(xs, ps, lam)
        j = RNG.integers(len(x))
        bumped = x.copy()
        bumped[j] += RNG.uniform(0.0, 2.0)
        order, bs = _desc(bumped)
        assert wowa(bs, p[order], lam) >= base - 1e-12


def test_wowa_symmetric_when_importance_uniform():
    """With equal importances WOWA is an OWA: order of inputs is irrelevant."""
    x = RNG.normal(size=9)
    a = wowa(np.sort(x)[::-1], np.ones(9), 3.0)
    b = wowa(np.sort(RNG.permutation(x))[::-1], np.ones(9), 3.0)
    assert a == pytest.approx(b)


def test_wowa_importance_scale_invariant():
    _, p, xs, ps = _random_wowa_case()
    assert wowa(xs, ps, -4.0) == pytest.approx(wowa(xs, 13.0 * ps, -4.0))


def test_wowa_rejects_zero_total_importance():
    with pytest.raises(ValueError):
        wowa(np.array([2.0, 1.0]), np.zeros(2), 1.0)


# ----------------------------------------------------- B. special cases

def test_quantifier_is_a_rim_quantifier():
    x = np.linspace(0.0, 1.0, 201)
    for lam in (-200.0, -3.0, 0.0, 1e-8, 3.0, 200.0):
        q = exponential_quantifier(x, lam)
        assert np.all(np.isfinite(q))
        assert q[0] == pytest.approx(0.0, abs=1e-12) and q[-1] == pytest.approx(1.0)
        assert np.all(np.diff(q) >= -1e-12)
    np.testing.assert_allclose(exponential_quantifier(x, 0.0), x, atol=1e-6)


def test_quantifier_gradients_match_finite_differences():
    x = np.linspace(0.05, 0.95, 7)
    for lam in (-60.0, -2.0, 0.5, 60.0):
        _, dx, dlam = exponential_quantifier(x, lam, with_grad=True)
        h = 1e-6
        num_dx = (exponential_quantifier(x + h, lam) - exponential_quantifier(x - h, lam)) / (2 * h)
        num_dl = (exponential_quantifier(x, lam + h) - exponential_quantifier(x, lam - h)) / (2 * h)
        np.testing.assert_allclose(dx, num_dx, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dlam, num_dl, rtol=1e-5, atol=1e-8)


def test_lambda_zero_is_importance_weighted_mean():
    x, p, xs, ps = _random_wowa_case()
    assert wowa(xs, ps, 0.0) == pytest.approx(np.sum(x * p) / p.sum(), abs=1e-6)


def test_extreme_lambda_approaches_max_and_min_at_the_bounded_rate():
    """lam -> +inf gives the max, lam -> -inf the min -- but at a rate set by
    the importance of the extreme sample, because Q'(0) is finite (that is the
    point of this quantifier, see the leak test below):

        |WOWA - max| <= span * exp(-lam * p_top)      (lam > 0)
        |WOWA - min| <= span * exp(-|lam| * p_bottom) (lam < 0)

    An earlier version asserted a fixed tolerance on one random draw; it held
    or failed depending on which draw xdist handed it. This checks the bound
    on many draws, including tiny extreme-sample importances."""
    rng = np.random.default_rng(1270)
    for _ in range(200):
        x = rng.normal(size=7)
        p = rng.uniform(0.001, 1.0, size=7)
        order, xs = _desc(x)
        ps = p[order] / p.sum()
        span = x.max() - x.min()
        for lam in (20.0, 200.0):
            assert x.max() - wowa(xs, ps, lam) <= span * np.exp(-lam * ps[0]) / -np.expm1(-lam) + 1e-12
            assert wowa(xs, ps, -lam) - x.min() <= span * np.exp(-lam * ps[-1]) / -np.expm1(-lam) + 1e-12
    # And with comparable importances the limit is effectively reached.
    xs = np.sort(rng.normal(size=7))[::-1]
    assert wowa(xs, np.ones(7), 200.0) == pytest.approx(xs[0], abs=1e-9)
    assert wowa(xs, np.ones(7), -200.0) == pytest.approx(xs[-1], abs=1e-9)


def test_uniform_importance_matches_yager_owa_weights():
    """OWA weights from a RIM quantifier: w_j = Q(j/n) - Q((j-1)/n)."""
    n, lam = 6, 2.5
    xs = np.sort(RNG.normal(size=n))[::-1]
    j = np.arange(1, n + 1)
    w = exponential_quantifier(j / n, lam) - exponential_quantifier((j - 1) / n, lam)
    assert wowa(xs, np.ones(n), lam) == pytest.approx(np.dot(w, xs))


@pytest.mark.parametrize("lam", [-8.0, -0.5, 0.0, 1e-5, 3.0, 40.0])
def test_orness_formula(lam):
    """orness = sum_j w_j (n - j)/(n - 1) -> integral of Q as n grows."""
    n = 4000
    j = np.arange(1, n + 1)
    w = exponential_quantifier(j / n, lam) - exponential_quantifier((j - 1) / n, lam)
    measured = np.sum(w * (n - j) / (n - 1))
    assert measured == pytest.approx(quantifier_orness(lam), abs=2e-3)


def test_max_like_segment_does_not_leak_the_global_max():
    """Regression for the quantifier choice. With a near-crisp partition the
    first half's max is 0. The power quantifier x**alpha returns ~the global
    max anyway -- Q'(0) = inf turns the spike's ~1e-14 importance into weight
    ~0.5 -- while the exponential quantifier bounds it by ~lam * p."""
    s = np.r_[np.zeros(10), np.full(10, 5.0)]             # spike lives in 2nd half
    Z = _aggregate(pad_sequences([s[:, None]]), np.array([0.25, 0.75]), np.full(2, 0.02),
                   np.full((2, 1), 200.0))
    assert Z[0, 0, 0] < 1e-6
    assert Z[0, 1, 0] == pytest.approx(5.0)

    tau = (np.arange(20) + 0.5) / 20
    p = temporal_partition(tau, np.array([0.25, 0.75]), np.full(2, 0.02))[:, 0]
    order = np.argsort(-s, kind="stable")
    ps = p[order] / p.sum()
    cum = np.cumsum(ps)
    power = np.sum(((cum ** 0.02) - np.clip(cum - ps, 0, 1) ** 0.02) * s[order])
    assert power > 2.0                                     # the leak being guarded


def test_crisp_limit_reproduces_hand_engineered_tercile_means():
    """Nesting: centres (k+.5)/3, width -> 0, lam=0 is "mean of each third"."""
    seqs = [RNG.normal(size=(length, 2)) for length in (9, 12, 30)]
    padded = pad_sequences(seqs)
    Z = _aggregate(padded, np.array([1, 3, 5]) / 6, np.full(3, 1e-3), np.zeros((3, 2)))
    for i, s in enumerate(seqs):
        thirds = np.split(s, 3)
        for k in range(3):
            np.testing.assert_allclose(Z[i, k], thirds[k].mean(0), atol=1e-6)


@pytest.mark.parametrize("lam, stat", [(200.0, np.max), (-200.0, np.min)])
def test_crisp_limit_with_extreme_lambda_is_per_third_max_or_min(lam, stat):
    s = RNG.normal(size=(12, 1))
    Z = _aggregate(pad_sequences([s]), np.array([1, 3, 5]) / 6, np.full(3, 1e-3),
                   np.full((3, 1), lam))
    np.testing.assert_allclose(Z[0, :, 0], [stat(t) for t in np.split(s[:, 0], 3)], atol=1e-6)


# ------------------------------------------------ C. temporal partition

def test_partition_is_ruspini():
    tau = np.linspace(0.0, 1.0, 101)
    mu = temporal_partition(tau, np.array([0.1, 0.4, 0.9]), np.array([0.05, 0.2, 0.01]))
    np.testing.assert_allclose(mu.sum(-1), 1.0)
    assert np.all(mu >= 0)


def test_partition_crisp_limit_is_nearest_centre():
    tau = np.array([0.1, 0.32, 0.34, 0.6, 0.7, 0.95])
    c = np.array([1, 3, 5]) / 6
    mu = temporal_partition(tau, c, np.full(3, 1e-3))
    np.testing.assert_array_equal(mu.argmax(-1), np.abs(tau[:, None] - c).argmin(-1))
    np.testing.assert_allclose(mu.max(-1), 1.0)


def test_narrow_segment_far_from_samples_stays_finite():
    """Underflowing memberships must fall back to the nearest samples, not NaN."""
    seqs = [RNG.normal(size=(20, 1))]
    Z = _aggregate(pad_sequences(seqs), np.array([0.5, 0.999]), np.array([0.3, 0.005]),
                   np.zeros((2, 1)))
    assert np.all(np.isfinite(Z))


# ----------------------------------------------------- D. input handling

def test_padding_does_not_leak_between_lengths():
    seqs = [RNG.normal(size=(length, 3)) for length in (5, 17, 40)]
    c, w, a = np.array([0.2, 0.6]), np.array([0.1, 0.2]), RNG.uniform(-30, 30, (2, 3))
    together = _aggregate(pad_sequences(seqs), c, w, a)
    for i, s in enumerate(seqs):
        np.testing.assert_allclose(together[i], _aggregate(pad_sequences([s]), c, w, a)[0])


def test_array_list_and_dataframe_inputs_agree():
    arr = RNG.normal(size=(4, 10, 2))
    dfs = [pd.DataFrame(a, columns=["u", "v"]) for a in arr]
    agg = FuzzyTemporalAggregator(n_segments=2).fit(arr)
    np.testing.assert_allclose(agg.transform(arr).to_numpy(), agg.transform(dfs).to_numpy())
    named = FuzzyTemporalAggregator(n_segments=2).fit(dfs)
    assert list(named.get_feature_names_out()) == ["u__seg0", "v__seg0", "u__seg1", "v__seg1"]


def test_length_one_sequence_returns_its_value():
    agg = FuzzyTemporalAggregator(n_segments=3).fit([np.array([[2.0, -1.0]])])
    np.testing.assert_allclose(agg.transform([np.array([[2.0, -1.0]])]).to_numpy(),
                               [[2.0, -1.0] * 3])


@pytest.mark.parametrize("bad, err", [
    ([np.zeros((0, 2))], ValueError),
    ([np.array([[1.0, np.nan]])], ValueError),
    ([np.zeros((3, 2)), np.zeros((3, 3))], ValueError),
    ([], ValueError),
    ("abc", TypeError),
    (np.zeros((2, 2, 2, 2)), ValueError),
])
def test_input_validation(bad, err):
    with pytest.raises(err):
        pad_sequences(bad)


def test_transform_rejects_feature_count_mismatch():
    agg = FuzzyTemporalAggregator().fit([np.zeros((4, 2))])
    with pytest.raises(ValueError):
        agg.transform([np.zeros((4, 3))])


def test_fit_rejects_misaligned_or_nonfinite_targets():
    seqs = [np.zeros((4, 1))] * 5
    with pytest.raises(ValueError):
        FuzzyTemporalAggregator(**FAST).fit(seqs, np.zeros(4))
    with pytest.raises(ValueError):
        FuzzyTemporalAggregator(**FAST).fit(seqs, np.array([0, 1, 2, np.nan, 4.0]))


# ---------------------------------------------------------- E. learning

def test_analytic_gradient_matches_finite_differences():
    seqs = [RNG.normal(size=(int(RNG.integers(4, 15)), 3)) for _ in range(25)]
    y = RNG.normal(size=(25, 2))
    agg = FuzzyTemporalAggregator(n_segments=3)
    agg._c0, agg._w0 = np.array([0.2, 0.5, 0.8]), np.full(3, 0.1)
    padded = pad_sequences(seqs)
    folds = list(KFold(5, shuffle=True, random_state=0).split(y))
    # Inside the optimiser's box: lam = 400 * theta, so an unbounded draw puts
    # |lam| past 200 where exp(|lam|) overflows -- a point fitting never visits.
    _, bounds = agg._pack(agg._c0, agg._w0, 3)
    bounds = np.array(bounds)
    theta = RNG.uniform(bounds[:, 0], bounds[:, 1])
    _, analytic = agg._objective(theta, padded, y, folds)
    # Central differences: forward differences carry O(h) truncation error,
    # which the x400 scaling of the lambda coordinates turns into ~1e-4.
    f = lambda t: agg._objective(t, padded, y, folds, with_grad=False)  # noqa: E731
    h = 1e-6
    numeric = np.array([(f(theta + h * e) - f(theta - h * e)) / (2 * h)
                        for e in np.eye(theta.size)])
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


def test_cv_loss_matches_sklearn_ridge():
    """The surrogate is plain standardised ridge; pin it to sklearn's."""
    Z = RNG.normal(size=(40, 4))
    y = Z @ RNG.normal(size=4) + RNG.normal(0, 0.1, 40)
    agg = FuzzyTemporalAggregator(ridge=2.0)
    folds = list(KFold(4, shuffle=True, random_state=0).split(Z))
    ours = agg._cv_loss(Z, y[:, None], folds)
    ref = []
    for tr, va in folds:
        zm, zs, ym, ys = Z[tr].mean(0), Z[tr].std(0), y[tr].mean(), y[tr].std()
        m = Ridge(alpha=2.0, fit_intercept=False).fit((Z[tr] - zm) / zs, (y[tr] - ym) / ys)
        ref.append(np.mean((m.predict((Z[va] - zm) / zs) - (y[va] - ym) / ys) ** 2))
    assert ours == pytest.approx(np.mean(ref))


def test_learning_never_worsens_the_surrogate():
    seqs, y = _planted(seed=1)
    agg = FuzzyTemporalAggregator(**FAST).fit(seqs, y)
    assert agg.objective_ <= agg.objective_init_ + 1e-12


def test_learning_moves_a_segment_onto_the_planted_phase():
    seqs, y = _planted(n_seq=80, phase=0.72, seed=2)
    # Default restarts: centre placement is multimodal, and a single start
    # from the even partition can settle on a worse local optimum.
    agg = FuzzyTemporalAggregator(n_segments=3, max_evals=160, cv=3).fit(seqs, y)
    assert agg.objective_ < 0.5 * agg.objective_init_
    assert np.min(np.abs(agg.centers_ - 0.72)) < 0.08


def test_peak_target_learns_max_like_quantifier():
    """Target = spike height at a random time -> sensor 0 should go max-like."""
    rng = np.random.default_rng(3)
    seqs, y = [], []
    for _ in range(80):
        s = rng.normal(0, 0.2, 40)
        h = rng.uniform(0, 1)
        t = int(rng.integers(0, 39))
        s[t:t + 2] += 2 * h
        seqs.append(s[:, None])
        y.append(h)
    agg = FuzzyTemporalAggregator(n_segments=1, fit_segments=False, max_evals=120, cv=3)
    agg.fit(seqs, np.array(y))
    assert agg.orness_[0, 0] > 0.8


def test_ablation_switches_freeze_their_parameters():
    seqs, y = _planted(seed=4)
    seg_only = FuzzyTemporalAggregator(fit_quantifier=False, **FAST).fit(seqs, y)
    np.testing.assert_array_equal(seg_only.lambdas_, 0.0)
    alpha_only = FuzzyTemporalAggregator(fit_segments=False, **FAST).fit(seqs, y)
    np.testing.assert_allclose(alpha_only.centers_, [1 / 6, 1 / 2, 5 / 6])


def test_unsupervised_fit_is_the_soft_even_partition():
    agg = FuzzyTemporalAggregator(n_segments=4).fit([np.zeros((5, 1))])
    np.testing.assert_allclose(agg.centers_, (np.arange(4) + 0.5) / 4)
    np.testing.assert_array_equal(agg.lambdas_, 0.0)
    assert agg.objective_ is None


def test_segments_reported_in_centre_order():
    seqs, y = _planted(seed=5)
    agg = FuzzyTemporalAggregator(max_evals=80, n_restarts=2, cv=3).fit(seqs, y)
    assert np.all(np.diff(agg.centers_) >= 0)
    report = agg.segment_report()
    assert len(report) == 3 * 2
    np.testing.assert_allclose(report["orness"], quantifier_orness(report["lambda"]))


# ------------------------------------------------- F. estimator contract

def test_fit_is_deterministic():
    seqs, y = _planted(seed=6)
    a = FuzzyTemporalAggregator(max_evals=60, n_restarts=1, cv=3, random_state=0).fit(seqs, y)
    b = FuzzyTemporalAggregator(max_evals=60, n_restarts=1, cv=3, random_state=0).fit(seqs, y)
    np.testing.assert_array_equal(a.centers_, b.centers_)
    np.testing.assert_array_equal(a.lambdas_, b.lambdas_)


def test_regressor_end_to_end_single_output(capsys):
    seqs, y = _planted(n_seq=50, seed=7)
    model = TribbleTimeSeriesRegressor(**FAST).fit(seqs[:40], y[:40])
    pred = model.predict(seqs[40:])
    assert isinstance(pred, np.ndarray) and pred.shape == (10,)
    assert np.all(np.isfinite(pred))
    assert model.transform(seqs[40:]).shape == (10, 6)


def test_regressor_end_to_end_multi_output(capsys):
    seqs, y = _planted(n_seq=50, seed=8)
    Y = pd.DataFrame({"a": y, "b": -2 * y})
    model = TribbleTimeSeriesRegressor(**FAST).fit(seqs[:40], Y.iloc[:40])
    pred = model.predict(seqs[40:])
    assert isinstance(pred, pd.DataFrame) and list(pred.columns) == ["a", "b"]
    assert pred.shape == (10, 2)


def test_regressor_clones_custom_base_estimator():
    base = Ridge(alpha=1.0)
    seqs, y = _planted(n_seq=40, seed=9)
    model = TribbleTimeSeriesRegressor(base_estimator=base, **FAST).fit(seqs, y)
    assert model.estimator_ is not base
    assert not hasattr(base, "coef_")
    assert model.score(seqs, y) > 0.5


def test_sklearn_params_roundtrip():
    model = TribbleTimeSeriesRegressor(n_segments=5, ridge=0.3)
    cloned = clone(model)
    assert cloned.get_params()["n_segments"] == 5
    assert cloned.get_params()["ridge"] == 0.3
