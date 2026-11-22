"""The pairwise `interaction` scorer, on the XOR and product samples.

Why it exists: `wasserstein`, `bhattacharyya`, `composite` and `higher_order` each score a
feature on its OWN class-conditional distribution. A feature that matters only jointly with
another -- XOR, ``y = [x1 * x2 > 0]`` -- has identical marginals in both classes and looks
like noise to all four. `interaction` scores a feature as its own separation plus its best
partner's *excess* joint separation (closed-form 2-D Gaussian Bhattacharyya, which sees the
class-conditional correlation), so it finds such a pair.

Expectations registered before the first run (a contradicting result needs an explanation,
not a quiet edit):

  E1  XOR: `interaction` ranks (x1, x2) first and second; the four marginal scorers do not.
  E2  Product threshold ``[x1 * x2 > 1]``: `interaction` finds the pair. (The marginals find it
      too -- class 1 forces both |x| large -- so this case does NOT discriminate; it is here
      so nobody mistakes it for an interaction-only sample.)
  E3  Additive control: x3 first, and noise columns are not inflated by pairing with it.
  E4  Selecting the XOR pair does NOT let TribbleClassifier learn XOR (it combines per-feature
      memberships and XOR classes share every marginal) -- accuracy stays near 0.5. An explicit
      cross column does (~0.99). This is the groundwork for variable crossover features, not
      yet the feature itself.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tribblefis.feature_selection import rank_features
from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    calculate_interaction_scores,
)
from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.gaussian_regressor import TribbleRegressor

MARGINAL = ("wasserstein", "bhattacharyya", "composite", "higher_order")


def make(kind, seed, n=3000, noise=8):
    r = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": r.normal(size=n) for i in range(1, 4 + noise)})
    y = {
        "xor": X.x1 * X.x2 > 0,
        "product": X.x1 * X.x2 > 1,
        "additive": X.x3 > 0,
    }[kind]
    return X, pd.Series(y.astype(int).to_numpy(), index=X.index)


def ranked(X, y, method):
    return [f for f, _ in calculate_gaussian_correlation(X, y, method=method)]


@pytest.mark.parametrize("seed", range(4))
def test_interaction_finds_the_xor_pair_and_marginal_scorers_do_not(seed):
    X, y = make("xor", seed)
    assert set(ranked(X, y, "interaction")[:2]) == {"x1", "x2"}
    for method in MARGINAL:
        assert set(ranked(X, y, method)[:2]) != {"x1", "x2"}, method


@pytest.mark.parametrize("seed", range(3))
def test_interaction_finds_the_product_threshold_pair(seed):
    X, y = make("product", seed)
    assert set(ranked(X, y, "interaction")[:2]) == {"x1", "x2"}


def test_additive_control_is_not_inflated_by_pairing():
    X, y = make("additive", 0)
    scores = dict(calculate_gaussian_correlation(X, y, method="interaction"))
    assert max(scores, key=scores.get) == "x3" and scores["x3"] == pytest.approx(1.0)
    assert max(v for k, v in scores.items() if k != "x3") < 0.05


def test_scores_are_affine_invariant():
    X, y = make("xor", 1)
    base = dict(calculate_gaussian_correlation(X, y, method="interaction"))
    Z = X.copy()
    Z["x1"] = Z["x1"] * 1000 + 5
    Z["x4"] = Z["x4"] * 1e-3 - 40
    moved = dict(calculate_gaussian_correlation(Z, y, method="interaction"))
    for k in base:
        assert moved[k] == pytest.approx(base[k], abs=1e-6), k


def test_gaussian_engine_of_calculate_interaction_scores():
    X, y = make("xor", 0)
    fd = calculate_gaussian_correlation(X, y, method="wasserstein")
    pairs = calculate_interaction_scores(X, y, fd, engine="gaussian")
    assert len(pairs) == X.shape[1] * (X.shape[1] - 1) // 2
    assert set(pairs[0][:2]) == {"x1", "x2"} and pairs[0][2] == pytest.approx(1.0)
    lifts = [p[2] for p in pairs]
    assert lifts == sorted(lifts, reverse=True) and min(lifts) >= 0
    assert len(calculate_interaction_scores(X, y, fd, engine="gaussian", max_pairs=3)) == 3  # a cap, not an error
    # the default engine is the original, untouched
    assert calculate_interaction_scores(X, y, fd) == calculate_interaction_scores(X, y, fd, engine="product")
    with pytest.raises(ValueError, match="engine='gaussian'"):
        calculate_interaction_scores(X, y, fd, method="interaction")
    with pytest.raises(ValueError, match="engine must be"):
        calculate_interaction_scores(X, y, fd, engine="nope")


def test_gaussian_engine_has_no_pair_cap():
    X, y = make("xor", 0, n=1500, noise=77)  # 80 features -> 3,160 pairs, over the product engine's 2,000
    fd = calculate_gaussian_correlation(X, y, method="wasserstein")
    with pytest.raises(ValueError, match="max_pairs"):
        calculate_interaction_scores(X, y, fd)
    top = calculate_interaction_scores(X, y, fd, engine="gaussian")
    assert set(top[0][:2]) == {"x1", "x2"}


def test_degenerate_inputs_do_not_crash():
    X, y = make("xor", 0, n=600, noise=2)
    X["const"] = 3.0
    X["with_nan"] = X["x4"].where(np.arange(len(X)) % 7 != 0)
    X["cat"] = np.where(X["x5"] > 0, "a", "b")
    scores = dict(calculate_gaussian_correlation(X, y, method="interaction"))
    assert scores["const"] == 0.0
    assert set(sorted(scores, key=scores.get, reverse=True)[:2]) == {"x1", "x2"}
    # integer column names print without error; string labels work
    Xi = X[["x1", "x2", "x3"]].copy()
    Xi.columns = [0, 1, 2]
    ys = y.map({0: "no", 1: "yes"})
    assert [f for f, _ in calculate_gaussian_correlation(Xi, ys, method="interaction")][:2] in ([0, 1], [1, 0])
    # one feature, and one class, return something sensible
    assert len(calculate_gaussian_correlation(X[["x1"]], y, method="interaction")) == 1
    one_class = calculate_gaussian_correlation(X[["x1", "x2"]], pd.Series(0, index=X.index), method="interaction")
    assert all(score == 0.0 for _, score in one_class)


def test_unknown_method_still_rejected_and_message_lists_interaction():
    X, y = make("xor", 0, n=200, noise=1)
    with pytest.raises(ValueError, match="interaction"):
        calculate_gaussian_correlation(X, y, method="nope")


def test_estimator_wiring_and_pareto_path():
    X, y = make("xor", 0, n=1500)
    m = TribbleClassifier(top_n=2, scoring_method="interaction", correlation_threshold=0.0).fit(X, y)
    assert set(m.top_features_) == {"x1", "x2"}
    # the Pareto path takes the scorer too
    user = ["x1", "x2"]
    diff, cands = rank_features(X, y, method="interaction", top_n=2, user_ranking=user)
    assert set(f for f, _ in diff) == {"x1", "x2"} and len(cands) >= 1


def _acc(X, y, seed, **kw):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)
    m = TribbleClassifier(top_n=2, correlation_threshold=0.0, random_state=seed, **kw).fit(Xtr, ytr)
    return accuracy_score(yte, m.predict(Xte))


def test_selecting_the_pair_is_not_enough_but_a_cross_column_is():
    """E4: the honest limit of selection alone, pinned so it is not mistaken for a solved problem."""
    X, y = make("xor", 0, n=3000)
    # Selection finds the pair, but the classifier still cannot use it (~chance).
    assert _acc(X, y, 0, scoring_method="interaction") < 0.6
    # An explicit product column fixes it, for every scorer.
    Xc = X.assign(cross=X["x1"] * X["x2"])
    assert _acc(Xc, y, 0, scoring_method="interaction") > 0.95
    assert _acc(Xc, y, 0, scoring_method="wasserstein") > 0.95


def test_classifier_detect_interactions_builds_the_crossover_column():
    X, y = make("xor", 0, n=3000)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    base = TribbleClassifier(top_n=2, correlation_threshold=0.0).fit(Xtr, ytr)
    m = TribbleClassifier(top_n=2, correlation_threshold=0.0, detect_interactions=True).fit(Xtr, ytr)
    assert accuracy_score(yte, base.predict(Xte)) < 0.6  # without it: chance
    assert m.cross_pairs_ and set(m.cross_pairs_[0]) == {"x1", "x2"}
    assert "x1*x2" in m.top_features_
    assert accuracy_score(yte, m.predict(Xte)) > 0.95
    # every post-fit entry point builds the same column: DataFrame, ndarray, proba, firing strengths
    assert (m.predict(Xte.to_numpy()) == m.predict(Xte)).all()
    assert m.predict_proba(Xte).shape == (len(Xte), 2)
    fs, _ = m.firing_strengths(Xte)
    assert len(fs) == len(Xte)
    assert m.get_params()["max_cross_features"] == 3 and m.get_params()["interaction_n_permutations"] == 20


def test_classifier_without_interactions_is_unchanged():
    """No positive-lift pair -> no cross columns, and the model is the detect_interactions=False model."""
    X, y = make("additive", 0, n=2000)
    off = TribbleClassifier(top_n=3, correlation_threshold=0.0).fit(X, y)
    on = TribbleClassifier(top_n=3, correlation_threshold=0.0, detect_interactions=True).fit(X, y)
    assert on.cross_pairs_ == []
    assert on.top_features_ == off.top_features_
    assert (on.predict(X) == off.predict(X)).all()
    # and the off path never builds anything
    assert off.cross_pairs_ == []


@pytest.mark.parametrize("engine", ["product", "gaussian"])
def test_regressor_engine_option_on_the_product_problem(engine):
    """y = x0 * x1: at top_n=1 univariate ranking keeps one half and R2 collapses; detection rescues it."""
    r = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame({"x0": r.uniform(-1, 1, n), "x1": r.uniform(-1, 1, n)})
    for i in range(3):
        X[f"noise{i}"] = r.uniform(-1, 1, n)
    y = X.x0 * X.x1 + r.normal(0, 0.02, n)
    kw = dict(tsk_order="full-2nd", top_n=1, n_output_buckets=6, random_state=0)
    without = TribbleRegressor(**kw).fit(X, y)
    with_ = TribbleRegressor(**kw, detect_interactions=True, select_interactions=True, interaction_engine=engine).fit(X, y)
    r2 = lambda m: 1 - ((y - m.predict(X)) ** 2).sum() / ((y - y.mean()) ** 2).sum()  # noqa: E731
    assert r2(without) < 0.3
    assert r2(with_) > 0.95
