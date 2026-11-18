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
