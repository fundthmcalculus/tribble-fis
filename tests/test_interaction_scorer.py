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
    calculate_pairwise_lift,
)
from tribblefis.gaussian_classifier import TribbleClassifier

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


def test_pairwise_lift_view():
    X, y = make("xor", 0)
    pairs = calculate_pairwise_lift(X, y)
    assert len(pairs) == X.shape[1] * (X.shape[1] - 1) // 2
    assert set(pairs[0][:2]) == {"x1", "x2"}
    assert all(lift >= 0 for *_, lift in pairs)
    assert [p[2] for p in pairs] == sorted((p[2] for p in pairs), reverse=True)
    assert len(calculate_pairwise_lift(X, y, top=3)) == 3
    # the 1-D-product screen is a different tool and must refuse the pairwise method
    fd = calculate_gaussian_correlation(X, y, method="wasserstein")
    with pytest.raises(ValueError, match="calculate_pairwise_lift"):
        calculate_interaction_scores(X, y, fd, method="interaction")


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
    """E4: the honest limit of this change, pinned so it is not mistaken for a solved problem."""
    X, y = make("xor", 0, n=3000)
    # Selection finds the pair, but the classifier still cannot use it (~chance).
    assert _acc(X, y, 0, scoring_method="interaction") < 0.6
    # Build the cross column from the strongest pair, as the pairwise view allows ...
    a, b, _ = calculate_pairwise_lift(X, y, top=1)[0]
    Xc = X.assign(cross=X[a] * X[b])
    # ... and every scorer, including the original, then succeeds.
    assert _acc(Xc, y, 0, scoring_method="interaction") > 0.95
    assert _acc(Xc, y, 0, scoring_method="wasserstein") > 0.95
