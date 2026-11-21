"""`TribbleClassifier.fit_growing`: grow one ranked feature at a time until a held-out target (#242).

Expectations registered before the first run:

  G1  the held-out score after adding the k-th feature equals the score of a model refit from scratch
      with ``top_n=k`` on the same rows, for EVERY k (the incremental fold is the full fold, unrolled);
  G2  growth stops at the smallest size meeting the target -- the oracle's first size;
  G3  XOR + ``detect_interactions=True`` reaches the target at n = 1 (the crossover column); without
      it the target is never reached, so it warns and returns the best size seen;
  G4  the features are ranked once, not once per size;
  G5  the ranking/crossover search/memberships never see the held-out labels.

On the compiled-kernel CI leg the oracle's ``predict`` goes through libm ``exp`` while growth folds with
NumPy, so a row can flip on a 1-ULP difference: equality is asserted exactly on the NumPy path and
within a few rows on the compiled one.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import tribblefis.gauss_math as gm
from tribblefis import kernel
from tribblefis.feature_selection import cross_name
from tribblefis.gauss_data import AnomalyParameters
from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.growth import RunningFiringStrengths, metric_function

TOL = 5e-3 if kernel.HAVE_CYTHON_KERNEL else 1e-12


def f1m(a, b):
    return f1_score(a, b, average="macro", zero_division=0)


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=1600, n_features=12, n_informative=6, n_redundant=0, n_classes=3,
        n_clusters_per_class=1, class_sep=1.2, random_state=0,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    y = pd.Series(y)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    return tuple(a.reset_index(drop=True) for a in (Xtr, Xva, ytr, yva))


@pytest.fixture(scope="module")
def oracle(data):
    """Held-out macro F1 of a model REFIT from scratch with top_n=k, for every k (the brute-force sweep)."""
    Xtr, Xva, ytr, yva = data
    out = []
    for k in range(1, Xtr.shape[1] + 1):
        m = TribbleClassifier(top_n=k, correlation_threshold=0.0, random_state=42).fit(Xtr, ytr)
        out.append(f1m(yva, m.predict(Xva)))
    return np.array(out)


def grown(data, **kw):
    Xtr, Xva, ytr, yva = data
    kw.setdefault("validation", (Xva, yva))
    return TribbleClassifier(correlation_threshold=0.0, random_state=42).fit_growing(Xtr, ytr, **kw)


def test_history_equals_the_refit_oracle_at_every_size(data, oracle):  # G1
    with pytest.warns(UserWarning, match="not reached"):
        g = grown(data, target=1.0)
    assert len(g.growth_history_) == len(oracle)
    assert np.max(np.abs(g.growth_history_["score"].to_numpy() - oracle)) <= TOL


def test_stops_at_the_oracles_first_size_meeting_the_target(data, oracle):  # G2
    # A target in the middle of the LARGEST gap between the oracle's scores: no size's score is within
    # half that gap of it, so kernel noise (compiled leg) cannot change which size is first.
    vals = np.sort(oracle)
    gap = int(np.argmax(np.diff(vals)))
    target = float((vals[gap] + vals[gap + 1]) / 2)
    assert np.diff(vals)[gap] / 2 > 2 * TOL
    first = next(i for i, s in enumerate(oracle, 1) if s >= target)
    g = grown(data, target=target)
    assert g.target_reached_ and g.n_features_selected_ == first
    h = g.growth_history_
    assert len(h) == first and h["score"].iloc[-1] >= target
    assert (h["score"].iloc[:-1] < target).all()
    assert g.top_features_ == g.growth_order_[:first]


def test_unreachable_target_returns_the_best_size_not_the_last(data, oracle):
    with pytest.warns(UserWarning, match="best .* at n="):
        g = grown(data, target=1.0)
    assert not g.target_reached_
    best = int(np.argmax(oracle)) + 1
    assert best < len(oracle)  # this data is non-monotone: more features eventually hurt
    assert g.n_features_selected_ == best
    assert g.predict(data[1]).shape == (len(data[1]),)  # still a usable, fitted model


def test_patience_ends_growth_early(data):
    with pytest.warns(UserWarning, match="not reached"):
        g = grown(data, target=1.0, patience=2)
    assert len(g.growth_history_) < data[0].shape[1]
    assert not g.target_reached_


def test_features_are_ranked_once(data, monkeypatch):  # G4
    calls = []
    real = gm.calculate_gaussian_correlation
    monkeypatch.setattr(gm, "calculate_gaussian_correlation", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    with pytest.warns(UserWarning):
        g = grown(data, target=1.0)
    assert len(g.growth_history_) == data[0].shape[1] and len(calls) == 1


def test_grown_model_is_the_model_a_fresh_build_on_those_features_would_give(data):
    """refit=False keeps the training-part model; its held-out score is the history's at n_selected."""
    g = grown(data, target=0.6, refit=False)
    n = g.n_features_selected_
    assert list(g.model_.feature_models) == g.top_features_ == g.growth_order_[:n]
    score = f1m(data[3], g.predict(data[1]))
    assert abs(score - g.growth_history_["score"].iloc[n - 1]) <= TOL


def test_running_fold_is_the_full_fold_unrolled(data):
    Xtr, Xva, ytr, _ = data
    names = list(Xtr.columns[:5])
    norms = AnomalyParameters().norms()
    full = gm.create_gaussian_membership_dict(Xtr, ytr, top_n_var_names=names, random_state=42)
    run = None
    for i, name in enumerate(names, 1):
        fm = full.feature_models[name]
        run = run or RunningFiringStrengths(Xva, fm.ordered_keys, norms)
        run.add(name, fm)
        sliced = type(full)(feature_models={n: full.feature_models[n] for n in names[:i]})
        ref, labels = gm.tsk_firing_strengths(Xva, sliced, norms=norms)
        np.testing.assert_allclose(run.strengths, ref, rtol=1e-9, atol=1e-12)
        assert run.labels == list(labels)


def test_history_table(data):
    with pytest.warns(UserWarning):
        g = grown(data, target=1.0)
    h = g.growth_history_
    assert list(h.columns) == ["n", "feature", "score", "best_so_far", "step_seconds", "cumulative_seconds"]
    assert h["n"].tolist() == list(range(1, len(h) + 1))
    assert h["feature"].tolist() == g.growth_order_[: len(h)]
    assert (h["step_seconds"] >= 0).all() and h["cumulative_seconds"].is_monotonic_increasing
    assert g.growth_seconds_ == pytest.approx(h["cumulative_seconds"].iloc[-1])
    assert (h["best_so_far"].diff().dropna() >= 0).all()


def test_selection_never_sees_the_held_out_labels(data):  # G5
    """Flip every held-out label: the chosen order and the memberships must not move."""
    Xtr, Xva, ytr, yva = data
    flipped = (yva + 1) % 3
    with pytest.warns(UserWarning):
        a = grown(data, target=1.0)
    with pytest.warns(UserWarning):
        b = TribbleClassifier(correlation_threshold=0.0, random_state=42).fit_growing(
            Xtr, ytr, target=1.0, validation=(Xva, flipped)
        )
    assert a.growth_order_ == b.growth_order_
    assert not np.allclose(a.growth_history_["score"], b.growth_history_["score"])  # scores DO use them


def test_fraction_validation_is_seeded_and_reproducible(data):
    Xtr, _, ytr, _ = data
    runs = []
    for _ in range(2):
        with pytest.warns(UserWarning):
            runs.append(TribbleClassifier(random_state=7).fit_growing(Xtr, ytr, target=1.0, validation=0.25))
    assert runs[0].growth_order_ == runs[1].growth_order_
    np.testing.assert_array_equal(runs[0].growth_history_["score"], runs[1].growth_history_["score"])


def test_redundant_features_are_skipped_and_recorded(data):
    Xtr, Xva, ytr, yva = data
    noisy = np.random.default_rng(0)
    Xtr = Xtr.assign(dup=Xtr["f0"] + noisy.normal(0, 1e-3, len(Xtr)))
    Xva = Xva.assign(dup=Xva["f0"] + noisy.normal(0, 1e-3, len(Xva)))
    with pytest.warns(UserWarning):
        kept = TribbleClassifier(random_state=42).fit_growing(Xtr, ytr, target=1.0, validation=(Xva, yva))
    assert "dup" in kept.growth_skipped_ or "f0" in kept.growth_skipped_
    assert not {"dup", "f0"} <= set(kept.growth_order_)
    with pytest.warns(UserWarning):
        off = TribbleClassifier(correlation_threshold=0.0, random_state=42).fit_growing(
            Xtr, ytr, target=1.0, validation=(Xva, yva)
        )
    assert off.growth_skipped_ == [] and {"dup", "f0"} <= set(off.growth_order_)


def test_user_ranking_weight_sets_the_growth_order(data):
    Xtr, Xva, ytr, yva = data
    with pytest.warns(UserWarning):
        base = grown(data, target=1.0)
    reversed_order = list(reversed(base.growth_order_))
    with pytest.warns(UserWarning):
        w0 = TribbleClassifier(correlation_threshold=0.0, random_state=42, user_ranking=reversed_order).fit_growing(
            Xtr, ytr, target=1.0, validation=(Xva, yva), user_weight=0.0
        )
        w1 = TribbleClassifier(correlation_threshold=0.0, random_state=42, user_ranking=reversed_order).fit_growing(
            Xtr, ytr, target=1.0, validation=(Xva, yva), user_weight=1.0
        )
    assert w0.growth_order_ == base.growth_order_  # weight 0 ignores the user
    assert w1.growth_order_[0] == reversed_order[0]  # weight 1 follows the user's best first


def _xor(seed, n=2500):
    r = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": r.normal(size=n) for i in range(8)})
    return X, pd.Series((X["x0"] * X["x1"] > 0).astype(int))


def test_crossover_column_reaches_the_target_at_one_feature():  # G3
    X, y = _xor(0)
    m = TribbleClassifier(detect_interactions=True, random_state=0).fit_growing(X, y, target=0.95, metric="f1_macro")
    assert m.target_reached_ and m.n_features_selected_ == 1
    assert m.top_features_ == [cross_name("x0", "x1")]
    Xt, yt = _xor(99)
    assert f1m(yt, m.predict(Xt)) > 0.95  # built from the originals at predict time
    off = TribbleClassifier(random_state=0)
    with pytest.warns(UserWarning, match="not reached"):
        off.fit_growing(X, y, target=0.95)
    assert not off.target_reached_ and f1m(yt, off.predict(Xt)) < 0.6


def test_other_entry_points_work_after_growing(data):
    with pytest.warns(UserWarning):
        g = grown(data, target=1.0)
    proba = g.predict_proba(data[1])
    assert proba.shape == (len(data[1]), 3) and np.allclose(proba.sum(axis=1), 1.0)
    assert (g.predict(data[1].to_numpy()) == g.predict(data[1])).all()


@pytest.mark.parametrize("metric", ["f1_macro", "f1_weighted", "f1_micro", "accuracy", "balanced_accuracy"])
def test_named_metrics(data, metric):
    g = grown(data, target=0.3, metric=metric, refit=False)  # refit=False: predict() is the model that was scored
    assert g.target_reached_
    expected = metric_function(metric)(data[3], g.predict(data[1]))
    assert g.growth_history_["score"].iloc[-1] == pytest.approx(expected, abs=TOL)


def test_callable_metric_and_binary_f1():
    X, y = _xor(1, n=1500)
    m = TribbleClassifier(detect_interactions=True, random_state=0).fit_growing(
        X, y, target=0.9, metric=lambda yt, yp: float((yt == yp).mean())
    )
    assert m.target_reached_
    b = TribbleClassifier(detect_interactions=True, random_state=0).fit_growing(X, y, target=0.9, metric="f1")
    assert b.target_reached_


def test_argument_validation(data):
    Xtr, Xva, ytr, yva = data
    base = TribbleClassifier(random_state=0)
    with pytest.raises(NotImplementedError, match="gaussian"):
        TribbleClassifier(member_function="trap").fit_growing(Xtr, ytr, target=0.5)
    for bad_target in (0.0, 1.5, float("nan")):
        with pytest.raises(ValueError, match="target"):
            base.fit_growing(Xtr, ytr, target=bad_target)
    with pytest.raises(ValueError, match="pos_label"):
        base.fit_growing(Xtr, ytr, target=0.5, metric="f1")  # 3 classes, no pos_label
    with pytest.raises(ValueError, match="validation"):
        base.fit_growing(Xtr, ytr, target=0.5, validation=1.5)
    with pytest.raises(ValueError, match="user_weight"):
        base.fit_growing(Xtr, ytr, target=0.5, user_weight=2.0)
    with pytest.raises(ValueError, match="metric"):
        base.fit_growing(Xtr, ytr, target=0.5, metric="nope")
    assert not hasattr(base, "n_features_selected_")  # a failed call leaves nothing half-set that matters
