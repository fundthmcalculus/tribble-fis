"""higher_order metric and Pareto selection against a user feature ranking."""

import numpy as np
import pandas as pd
import pytest

from tribblefis.feature_selection import normalize_user_ranking, pareto_front, pareto_select_features
from tribblefis.gauss_math import _higher_order_distance, calculate_gaussian_correlation


def test_higher_order_sees_shape_when_mean_and_variance_match():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.exponential(1, 5000)
    b = (b - b.mean()) / b.std()  # same mean/var as a, different skew/kurtosis
    assert _higher_order_distance(a, rng.normal(0, 1, 5000)) < 0.2
    assert _higher_order_distance(a, b) > 1.0


def test_higher_order_zero_for_identical_and_constant():
    x = np.arange(10.0)
    assert _higher_order_distance(x, x) == 0.0
    assert _higher_order_distance(np.ones(5), np.ones(5)) == 0.0


def test_higher_order_ranks_shape_feature():
    rng = np.random.default_rng(1)
    n = 2000
    y = pd.Series(np.repeat([0, 1], n))
    shape = np.concatenate([rng.normal(0, 1, n), (rng.exponential(1, n) - 1)])
    noise = rng.normal(0, 1, 2 * n)
    X = pd.DataFrame({"shape": shape, "noise": noise})
    ranked = calculate_gaussian_correlation(X, y, method="higher_order")
    assert ranked[0][0] == "shape"


def test_invalid_method_still_rejected():
    with pytest.raises(ValueError):
        calculate_gaussian_correlation(pd.DataFrame({"a": [1.0, 2]}), pd.Series([0, 1]), method="nope")


SCORES = [("a", 1.0), ("b", 0.8), ("c", 0.5), ("d", 0.2)]


def test_no_user_ranking_is_plain_top_n():
    (sel,) = pareto_select_features(SCORES, None, top_n=2)
    assert sel.features == ("a", "b")


def test_user_ranking_forms_agree():
    ordered = normalize_user_ranking(["d", "c"], list("abcd"))
    assert ordered["d"] > ordered["c"] > ordered["a"] == ordered["b"] == 0.0
    assert normalize_user_ranking({"d": 5, "c": 1}, list("abcd"))["d"] == 1.0
    with pytest.raises(ValueError):
        normalize_user_ranking(["zzz"], list("abcd"))


def test_conflicting_rankings_give_multiple_optimal_subsets():
    sels = pareto_select_features(SCORES, ["d", "c", "b", "a"], top_n=2)
    assert len(sels) > 1
    assert sels[0].features == ("a", "b") and set(sels[-1].features) == {"c", "d"}
    for s in sels:  # mutually non-dominated
        assert not any(
            o.data_score >= s.data_score and o.user_score >= s.user_score and o is not s
            and (o.data_score > s.data_score or o.user_score > s.user_score)
            for o in sels
        )


def test_agreeing_rankings_collapse_to_one_subset():
    assert len(pareto_select_features(SCORES, ["a", "b", "c", "d"], top_n=2)) == 1


def test_pareto_front_ranks():
    fr = {f: r for f, _, _, r in pareto_front(SCORES, ["d", "c", "b", "a"])}
    assert set(fr.values()) == {0}  # perfectly anti-correlated: all non-dominated
    fr = {f: r for f, _, _, r in pareto_front(SCORES, ["a", "b", "c", "d"])}
    assert fr == {"a": 0, "b": 1, "c": 2, "d": 3}


def test_top_n_clipped():
    assert len(pareto_select_features(SCORES, None, top_n=99)[0].features) == 4


# ---- estimator wiring -------------------------------------------------------
from tribblefis.gaussian_classifier import TribbleClassifier  # noqa: E402
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402
from sklearn.base import clone  # noqa: E402


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = pd.DataFrame(
        {
            "strong": y * 3.0 + rng.normal(0, 1, n),
            "mid": y * 1.0 + rng.normal(0, 1, n),
            "weak": y * 0.2 + rng.normal(0, 1, n),
            "noise": rng.normal(0, 1, n),
        }
    )
    return X, y


def test_classifier_default_unchanged_by_new_params():
    X, y = _toy()
    base = TribbleClassifier(top_n=2).fit(X, y)
    again = TribbleClassifier(top_n=2, scoring_method="wasserstein", user_ranking=None).fit(X, y)
    assert base.top_features_ == again.top_features_
    assert base.pareto_candidates_ is None
    assert clone(again).get_params()["pareto_choice"] == 0


@pytest.mark.parametrize("method", ["bhattacharyya", "higher_order"])
def test_classifier_scoring_method_reaches_ranking(method):
    X, y = _toy()
    m = TribbleClassifier(top_n=2, scoring_method=method).fit(X, y)
    assert m.top_features_[0] == "strong"


def test_classifier_user_ranking_changes_selection_and_choice_picks_subset():
    X, y = _toy()
    user = ["noise", "weak", "mid", "strong"]
    data_led = TribbleClassifier(top_n=2, user_ranking=user, pareto_choice=0).fit(X, y)
    user_led = TribbleClassifier(top_n=2, user_ranking=user, pareto_choice=-1).fit(X, y)
    assert len(data_led.pareto_candidates_) > 1
    assert set(data_led.top_features_) == {"strong", "mid"}
    assert "noise" in user_led.top_features_
    assert len(user_led.predict(X)) == len(X)
    with pytest.raises(ValueError):
        TribbleClassifier(top_n=2, user_ranking=user, pareto_choice=99).fit(X, y)
    with pytest.raises(ValueError):
        TribbleClassifier(user_ranking=user).fit(X, y)  # top_n unset


def test_regressor_user_ranking_and_method():
    X, y = _toy()
    yr = X["strong"] * 2 + X["mid"] + np.random.default_rng(1).normal(0, 0.1, len(X))
    m = TribbleRegressor(top_n=2, scoring_method="higher_order", user_ranking={"noise": 1.0}, pareto_choice=-1)
    m.fit(X, yr)
    assert "noise" in m.top_features_ and len(m.pareto_candidates_) >= 1
    assert len(m.predict(X)) == len(X)
    with pytest.raises(ValueError):
        TribbleRegressor(top_n=2, user_ranking=["noise"], detect_interactions=True).fit(X, yr)


def test_user_ranking_needs_top_n_and_ignores_top_p():
    X, y = _toy()
    user = ["noise", "weak", "mid", "strong"]
    with pytest.raises(ValueError, match="top_n"):
        TribbleClassifier(user_ranking=user).fit(X, y)
    a = TribbleClassifier(top_n=2, user_ranking=user, top_p=0.1).fit(X, y)
    b = TribbleClassifier(top_n=2, user_ranking=user, top_p=0.99).fit(X, y)
    assert a.top_features_ == b.top_features_ and len(a.top_features_) == 2


# ---- front ranks against an independent oracle --------------------------------
def _oracle_ranks(objs):
    """Brute-force non-dominated sorting, maximisation: rank = peel count."""
    objs = np.asarray(objs, float)
    left, ranks, r = list(range(len(objs))), {}, 0
    while left:
        front = [
            i
            for i in left
            if not any(
                (objs[j] >= objs[i]).all() and (objs[j] > objs[i]).any() for j in left if j != i
            )
        ]
        for i in front:
            ranks[i] = r
        left = [i for i in left if i not in front]
        r += 1
    return np.array([ranks[i] for i in range(len(objs))])


@pytest.mark.parametrize("seed", range(5))
def test_front_ranks_match_oracle_including_ties(seed):
    from tribblefis.feature_selection import _nondominated_ranks

    rng = np.random.default_rng(seed)
    objs = rng.integers(0, 6, size=(40, 2)).astype(float)  # small grid -> many duplicates/ties
    assert (_nondominated_ranks(objs) == _oracle_ranks(objs)).all()
