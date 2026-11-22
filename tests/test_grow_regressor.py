"""`TribbleRegressor.fit_growing`: grow one ranked feature at a time until a held-out R2 / RMSE (#248).

Expectations registered before the first run (the speedup deliberately had no prediction: #248 said to
measure it, and the guess there -- that the consequent solve would dominate -- turned out wrong for 1st order):

  R1  the held-out score after the k-th feature equals a model refit from scratch with ``top_n=k`` on the
      same rows, for EVERY k, exactly: growth calls the same `_fit_memberships`, `_solve` and `predict`;
  R2  it stops at the oracle's first size meeting the target; for ``rmse`` the target is a CEILING;
  R3  the features are ranked once;
  R4  the output partition, ranking, memberships and consequents never see the held-out targets.
"""

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import tribblefis.gauss_math as gm
from tribblefis.gaussian_regressor import TribbleRegressor

KW = dict(correlation_threshold=0.0, tsk_order="1st", n_output_buckets=3, random_state=42)


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


@pytest.fixture(scope="module")
def data():
    X, y = make_regression(n_samples=1200, n_features=10, n_informative=5, noise=10.0, random_state=0)
    X, y = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), pd.Series(y)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=0)
    return tuple(v.reset_index(drop=True) for v in (Xtr, Xva, ytr, yva))


@pytest.fixture(scope="module")
def oracle(data):
    """Held-out (R2, RMSE) of a model REFIT from scratch with top_n=k, for every k."""
    Xtr, Xva, ytr, yva = data
    r2s, rmses = [], []
    for k in range(1, Xtr.shape[1] + 1):
        pred = TribbleRegressor(top_n=k, **KW).fit(Xtr, ytr).predict(Xva)
        r2s.append(r2_score(yva, pred))
        rmses.append(rmse(yva, pred))
    return np.array(r2s), np.array(rmses)


def grown(data, **kw):
    Xtr, Xva, ytr, yva = data
    kw.setdefault("validation", (Xva, yva))
    return TribbleRegressor(**KW).fit_growing(Xtr, ytr, **kw)


def mid_largest_gap(values):
    v = np.sort(values)
    i = int(np.argmax(np.diff(v)))
    return float((v[i] + v[i + 1]) / 2), float(np.diff(v)[i])


def test_history_equals_the_refit_oracle_at_every_size(data, oracle):  # R1
    with pytest.warns(UserWarning, match="not reached"):
        g = grown(data, target=1.0)
    scores = g.growth_history_["score"].to_numpy()
    assert len(scores) == len(oracle[0])
    assert np.max(np.abs(scores - oracle[0])) <= 1e-12


def test_stops_at_the_oracles_first_size_meeting_an_r2_floor(data, oracle):  # R2
    target, gap = mid_largest_gap(oracle[0])
    first = next(i for i, s in enumerate(oracle[0], 1) if s >= target)
    g = grown(data, target=target)
    assert g.target_reached_ and g.n_features_selected_ == first
    h = g.growth_history_
    assert len(h) == first and h["score"].iloc[-1] >= target and (h["score"].iloc[:-1] < target).all()
    assert g.top_features_ == g.growth_order_[:first]


def test_rmse_target_is_a_ceiling(data, oracle):  # R2
    target, _ = mid_largest_gap(oracle[1])
    first = next(i for i, s in enumerate(oracle[1], 1) if s <= target)
    g = grown(data, target=target, metric="rmse")
    assert g.target_reached_ and g.n_features_selected_ == first
    h = g.growth_history_
    assert h["score"].iloc[-1] <= target and (h["score"].iloc[:-1] > target).all()
    assert np.allclose(h["score"].to_numpy(), oracle[1][:first], atol=1e-9)
    assert (h["best_so_far"].diff().dropna() <= 0).all()  # "best" means lowest


def test_unreachable_target_returns_the_best_size_not_the_last(data, oracle):
    with pytest.warns(UserWarning, match="best .* at n="):
        g = grown(data, target=1.0)
    assert not g.target_reached_
    assert g.n_features_selected_ == int(np.argmax(oracle[0])) + 1
    assert g.predict(data[1]).shape == (len(data[1]),)
    with pytest.warns(UserWarning, match="not reached"):
        low = grown(data, target=1e-9, metric="rmse")  # no RMSE is that small
    assert not low.target_reached_ and low.n_features_selected_ == int(np.argmin(oracle[1])) + 1


def test_patience_ends_growth_early(data):
    with pytest.warns(UserWarning, match="not reached"):
        g = grown(data, target=1.0, patience=1)
    assert len(g.growth_history_) < data[0].shape[1] and not g.target_reached_


def test_features_are_ranked_once(data, monkeypatch):  # R3
    calls = []
    real = gm.calculate_gaussian_correlation
    monkeypatch.setattr(gm, "calculate_gaussian_correlation", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    with pytest.warns(UserWarning):
        g = grown(data, target=1.0)
    assert len(g.growth_history_) == data[0].shape[1] and len(calls) == 1


def test_selection_never_sees_the_held_out_targets(data):  # R4
    Xtr, Xva, ytr, yva = data
    shuffled = pd.Series(np.random.default_rng(0).permutation(yva.to_numpy()))
    with pytest.warns(UserWarning):
        a = grown(data, target=1.0)
        b = TribbleRegressor(**KW).fit_growing(Xtr, ytr, target=1.0, validation=(Xva, shuffled))
    assert a.growth_order_ == b.growth_order_
    assert not np.allclose(a.growth_history_["score"], b.growth_history_["score"])


def test_refit_false_keeps_the_scored_model(data):
    g = grown(data, target=0.5, refit=False)
    n = g.n_features_selected_
    assert g.top_features_ == g.growth_order_[:n] and list(g.model_.feature_models) == g.top_features_
    assert r2_score(data[3], g.predict(data[1])) == pytest.approx(g.growth_history_["score"].iloc[n - 1], abs=1e-12)


def test_refit_true_builds_a_full_model_on_the_chosen_features(data):
    g = grown(data, target=0.5)
    assert g.top_features_ == g.growth_order_[: g.n_features_selected_]
    assert (g.predict(data[1].to_numpy()) == g.predict(data[1])).all()
    assert g.n_rules_ > 0 and r2_score(data[3], g.predict(data[1])) > 0.3


def test_history_table_and_reproducibility(data):
    Xtr, _, ytr, _ = data
    runs = []
    for _ in range(2):
        with pytest.warns(UserWarning):
            runs.append(TribbleRegressor(**KW).fit_growing(Xtr, ytr, target=1.0, validation=0.25))
    h = runs[0].growth_history_
    assert list(h.columns) == ["n", "feature", "score", "best_so_far", "step_seconds", "cumulative_seconds"]
    assert h["n"].tolist() == list(range(1, len(h) + 1)) and h["cumulative_seconds"].is_monotonic_increasing
    assert runs[0].growth_seconds_ == pytest.approx(h["cumulative_seconds"].iloc[-1])
    assert runs[0].growth_order_ == runs[1].growth_order_
    np.testing.assert_array_equal(h["score"], runs[1].growth_history_["score"])


def test_user_weight_sets_the_order(data):
    Xtr, Xva, ytr, yva = data
    with pytest.warns(UserWarning):
        base = grown(data, target=1.0)
    user = list(reversed(base.growth_order_))
    with pytest.warns(UserWarning):
        w0 = TribbleRegressor(user_ranking=user, **KW).fit_growing(Xtr, ytr, target=1.0, validation=(Xva, yva), user_weight=0.0)
        w1 = TribbleRegressor(user_ranking=user, **KW).fit_growing(Xtr, ytr, target=1.0, validation=(Xva, yva), user_weight=1.0)
    assert w0.growth_order_ == base.growth_order_ and w1.growth_order_[0] == user[0]


def test_redundant_feature_is_skipped(data):
    Xtr, Xva, ytr, yva = data
    noise = np.random.default_rng(1)
    Xtr = Xtr.assign(dup=Xtr["f0"] + noise.normal(0, 1e-3, len(Xtr)))
    Xva = Xva.assign(dup=Xva["f0"] + noise.normal(0, 1e-3, len(Xva)))
    kw = {**KW, "correlation_threshold": 0.85}
    with pytest.warns(UserWarning):
        g = TribbleRegressor(**kw).fit_growing(Xtr, ytr, target=1.0, validation=(Xva, yva))
    assert {"dup", "f0"} & set(g.growth_skipped_) and not {"dup", "f0"} <= set(g.growth_order_)


@pytest.mark.parametrize("member_function,trapz_method", [("gaussian", "fast"), ("trap", "fast")])
def test_growth_equals_the_refit_oracle_for_the_other_builders(data, member_function, trapz_method):
    Xtr, Xva, ytr, yva = data
    kw = {**KW, "member_function": member_function, "trapz_method": trapz_method}
    with pytest.warns(UserWarning):
        g = TribbleRegressor(**kw).fit_growing(Xtr, ytr, target=1.0, validation=(Xva, yva), max_n=5)
    ref = [r2_score(yva, TribbleRegressor(top_n=k, **kw).fit(Xtr, ytr).predict(Xva)) for k in range(1, 6)]
    assert np.max(np.abs(g.growth_history_["score"].to_numpy() - ref)) <= 1e-12


def test_estimator_params_still_clone(data):
    est = TribbleRegressor(**KW)
    assert clone(est).get_params() == est.get_params()  # fit_growing adds no constructor parameter
    assert not hasattr(est, "n_features_selected_")


def test_argument_validation(data):
    Xtr, Xva, ytr, yva = data
    base = TribbleRegressor(**KW)
    with pytest.raises(ValueError, match="tsk_order"):
        TribbleRegressor(**{**KW, "tsk_order": "auto"}).fit_growing(Xtr, ytr, target=0.5)
    with pytest.raises(ValueError, match="detect_interactions"):
        TribbleRegressor(**KW, detect_interactions=True).fit_growing(Xtr, ytr, target=0.5)
    with pytest.raises(ValueError, match="higher_is_better"):
        base.fit_growing(Xtr, ytr, target=0.5, metric=lambda a, b: 0.0)  # a callable must say which way is good
    with pytest.raises(ValueError, match="higher-is-better"):
        base.fit_growing(Xtr, ytr, target=0.5, metric="r2", higher_is_better=False)
    with pytest.raises(ValueError, match="metric"):
        base.fit_growing(Xtr, ytr, target=0.5, metric="nope")
    for target, metric in ((1.5, "r2"), (0.0, "rmse"), (-1.0, "mae"), (float("nan"), "r2")):
        with pytest.raises(ValueError, match="target"):
            base.fit_growing(Xtr, ytr, target=target, metric=metric)
    with pytest.raises(ValueError, match="validation"):
        base.fit_growing(Xtr, ytr, target=0.5, validation=2.0)
    with pytest.raises(ValueError, match="user_weight"):
        base.fit_growing(Xtr, ytr, target=0.5, user_weight=-0.1)
    with pytest.raises(ValueError, match="member_function"):
        TribbleRegressor(**{**KW, "member_function": "nope"}).fit_growing(Xtr, ytr, target=0.5)


def test_callable_metrics_in_both_directions(data, oracle):
    Xtr, Xva, ytr, yva = data
    # higher-is-better callable: negative RMSE; its floor is minus the ceiling
    target, _ = mid_largest_gap(oracle[1])
    first = next(i for i, s in enumerate(oracle[1], 1) if s <= target)
    up = TribbleRegressor(**KW).fit_growing(
        Xtr, ytr, target=-target, metric=lambda a, b: -rmse(a, b), higher_is_better=True, validation=(Xva, yva)
    )
    down = TribbleRegressor(**KW).fit_growing(
        Xtr, ytr, target=target, metric=lambda a, b: rmse(a, b), higher_is_better=False, validation=(Xva, yva)
    )
    assert up.target_reached_ and down.target_reached_
    assert up.n_features_selected_ == down.n_features_selected_ == first



@pytest.mark.parametrize("pin_extremes", [True, False])
def test_growth_equals_the_refit_oracle_with_and_without_pinned_extremes(data, pin_extremes):
    """`pin_extremes` is an option that touches the bucket means, so check the oracle under both settings.

    (Resetting those means between steps is defensive: carrying a previous step's result forward changes
    no score today, so no test can tell the reset from its absence -- see `_solve`.)
    """
    Xtr, Xva, ytr, yva = data
    kw = {**KW, "pin_extremes": pin_extremes}
    with pytest.warns(UserWarning):
        g = TribbleRegressor(**kw).fit_growing(Xtr, ytr, target=1.0, validation=(Xva, yva), max_n=6)
    ref = [r2_score(yva, TribbleRegressor(top_n=k, **kw).fit(Xtr, ytr).predict(Xva)) for k in range(1, 7)]
    assert np.max(np.abs(g.growth_history_["score"].to_numpy() - ref)) <= 1e-12
