"""Tests for TribbleOneClassDetector -- fuzzy one-class novelty detection."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from tribblefis.one_class import TribbleOneClassDetector


@pytest.fixture
def normal_and_outliers():
    """A tight normal cluster and a far-away outlier cluster."""
    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(6)]
    X_train = pd.DataFrame(rng.normal(0, 1, size=(300, 6)), columns=cols)
    X_normal = pd.DataFrame(rng.normal(0, 1, size=(100, 6)), columns=cols)
    X_out = pd.DataFrame(rng.normal(5, 1, size=(100, 6)), columns=cols)
    return X_train, X_normal, X_out


def test_fits_on_a_single_class(normal_and_outliers):
    """The whole point: fit with no second class, no labels."""
    X_train, _, _ = normal_and_outliers
    det = TribbleOneClassDetector(n_gaussians=2).fit(X_train)  # no y
    assert det.is_fitted_
    assert det.top_features_ == list(X_train.columns)


def test_outliers_score_higher_than_normal(normal_and_outliers):
    X_train, X_normal, X_out = normal_and_outliers
    det = TribbleOneClassDetector(n_gaussians=2).fit(X_train)
    assert det.anomaly_score(X_out).mean() > det.anomaly_score(X_normal).mean()
    # score_samples is the sklearn-oriented negative: higher == more normal
    assert det.score_samples(X_normal).mean() > det.score_samples(X_out).mean()


def test_separates_well_separated_clusters(normal_and_outliers):
    X_train, X_normal, X_out = normal_and_outliers
    det = TribbleOneClassDetector(n_gaussians=2).fit(X_train)
    y = np.r_[np.zeros(len(X_normal)), np.ones(len(X_out))]
    s = np.r_[det.anomaly_score(X_normal), det.anomaly_score(X_out)]
    assert roc_auc_score(y, s) > 0.99


def test_predict_and_decision_function_conventions(normal_and_outliers):
    X_train, X_normal, X_out = normal_and_outliers
    det = TribbleOneClassDetector(n_gaussians=2, contamination=0.05).fit(X_train)
    # sklearn OutlierMixin: +1 inlier, -1 outlier
    assert set(np.unique(det.predict(X_normal))) <= {-1, 1}
    assert (det.predict(X_out) == -1).mean() > 0.95
    assert (det.predict(X_normal) == 1).mean() > 0.85
    # decision_function sign agrees with predict
    assert np.all((det.decision_function(X_normal) >= 0) == (det.predict(X_normal) == 1))


def test_contamination_sets_training_flag_rate(normal_and_outliers):
    """Roughly `contamination` of the training data should be flagged outlier."""
    X_train, _, _ = normal_and_outliers
    det = TribbleOneClassDetector(n_gaussians=2, contamination=0.1).fit(X_train)
    flagged = (det.predict(X_train) == -1).mean()
    assert 0.03 < flagged < 0.20  # near 0.1, with quantile discreteness slack


def test_feature_selection_variance(normal_and_outliers):
    X_train, _, _ = normal_and_outliers
    det = TribbleOneClassDetector(feature_selection="variance", top_n=3).fit(X_train)
    assert len(det.top_features_) == 3


def test_feature_selection_explicit_list(normal_and_outliers):
    X_train, _, _ = normal_and_outliers
    det = TribbleOneClassDetector(feature_selection=["f0", "f2"]).fit(X_train)
    assert det.top_features_ == ["f0", "f2"]


def test_feature_selection_rejects_unknown_column(normal_and_outliers):
    X_train, _, _ = normal_and_outliers
    with pytest.raises(ValueError):
        TribbleOneClassDetector(feature_selection=["nope"]).fit(X_train)


def test_accepts_numpy_input():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(200, 4))
    det = TribbleOneClassDetector(n_gaussians=1).fit(X)
    assert det.anomaly_score(rng.normal(5, 1, size=(10, 4))).mean() > 0.5


def test_fit_predict_from_outlier_mixin(normal_and_outliers):
    """OutlierMixin gives fit_predict for free; it should run and return labels."""
    X_train, _, _ = normal_and_outliers
    labels = TribbleOneClassDetector(n_gaussians=2).fit_predict(X_train)
    assert set(np.unique(labels)) <= {-1, 1}
    assert len(labels) == len(X_train)


def test_get_params_roundtrip():
    """sklearn clone / get_params-set_params must work (BaseEstimator contract)."""
    from sklearn.base import clone

    det = TribbleOneClassDetector(n_gaussians=3, contamination=0.02,
                                  feature_selection="variance", top_n=5)
    cloned = clone(det)
    assert cloned.get_params() == det.get_params()


def test_whiten_recovers_covariance_anomaly_invisible_per_feature():
    """The decisive case: an anomaly whose per-feature marginals are all normal
    but whose *correlation structure* is violated. Per-feature Gaussians (the
    raw detector) cannot see it; whitening decorrelates and catches it.
    """
    rng = np.random.default_rng(3)
    d = 6
    # normal: all features nearly equal (a single latent factor) -> tight
    # positive correlation, each marginal ~ N(0, 1).
    def positively_correlated(n):
        base = rng.normal(0, 1, size=(n, 1))
        return base + rng.normal(0, 0.05, size=(n, d))

    # anomaly: same per-feature marginals (~N(0,1)) but the correlation is
    # broken -- features are independent draws, so each column is in-range yet
    # the joint pattern never occurs in normal data.
    def decorrelated(n):
        return rng.normal(0, 1, size=(n, d))

    cols = [f"f{i}" for i in range(d)]
    Xtr = pd.DataFrame(positively_correlated(400), columns=cols)
    Xn = pd.DataFrame(positively_correlated(100), columns=cols)
    Xa = pd.DataFrame(decorrelated(100), columns=cols)
    y = np.r_[np.zeros(100), np.ones(100)]

    raw = TribbleOneClassDetector(whiten=False, n_gaussians=1).fit(Xtr)
    wht = TribbleOneClassDetector(whiten=True, n_gaussians=1).fit(Xtr)
    auc_raw = roc_auc_score(y, np.r_[raw.anomaly_score(Xn), raw.anomaly_score(Xa)])
    auc_wht = roc_auc_score(y, np.r_[wht.anomaly_score(Xn), wht.anomaly_score(Xa)])
    # per-feature marginals match, so the raw detector is near chance...
    assert auc_raw < 0.65
    # ...and whitening recovers the covariance-borne signal.
    assert auc_wht > 0.85
    assert auc_wht > auc_raw + 0.2


def test_whiten_components_limits_rank():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"f{i}" for i in range(10)])
    det = TribbleOneClassDetector(whiten=True, whiten_components=4).fit(X)
    assert det._pca_.n_components_ == 4
    assert len(det.top_features_) == 4
    # scoring still runs end to end through the stored transform
    assert det.anomaly_score(X).shape == (200,)


def test_whiten_get_params_roundtrip():
    from sklearn.base import clone

    det = TribbleOneClassDetector(whiten=True, whiten_components=0.95)
    assert clone(det).get_params() == det.get_params()


def test_scoring_is_invariant_to_column_order(normal_and_outliers):
    """Scoring must key off column *names*, not the caller's column order.

    The raw path already did -- `tsk_firing_strengths` looks columns up by name.
    The whitening path did not: `PCA.transform` is positional, so a permuted
    frame was decorrelated against the wrong axes and scored nonsense without
    raising.
    """
    X_train, X_normal, _ = normal_and_outliers
    permuted = X_normal[list(X_normal.columns)[::-1]]

    for kwargs in ({"whiten": False}, {"whiten": True}):
        det = TribbleOneClassDetector(n_gaussians=2, **kwargs).fit(X_train)
        np.testing.assert_allclose(
            det.anomaly_score(X_normal), det.anomaly_score(permuted)
        )


def test_scoring_rejects_a_frame_missing_fitted_features(normal_and_outliers):
    """A missing column must raise, not score everything as normal.

    `tsk_firing_strengths` skips features it cannot find in `X`, so a frame
    whose columns do not match the fitted ones leaves the rule firing at 1.0 --
    anomaly 0.0 for every row. Silently reporting "all normal" is the worst
    possible failure mode for a detector, so it is an error instead.
    """
    X_train, X_normal, _ = normal_and_outliers
    det = TribbleOneClassDetector(n_gaussians=2).fit(X_train)

    with pytest.raises(ValueError, match="missing features"):
        det.anomaly_score(X_normal.drop(columns=["f0"]))

    # The specific case that produced the silent zeros: fit on numpy (synthetic
    # `feature_0..` names), then score a real frame with its own column names.
    numpy_fit = TribbleOneClassDetector(n_gaussians=2).fit(X_train.to_numpy())
    with pytest.raises(ValueError, match="missing features"):
        numpy_fit.anomaly_score(X_normal)


def test_contamination_above_half_warns_but_is_honoured(normal_and_outliers):
    """A contamination past 0.5 must reach the threshold, and must warn.

    It used to be silently clipped to 0.5, so `contamination=0.9` flagged 50%
    of the training data and said nothing -- the threshold ended up somewhere
    the caller did not ask for. Now the value is used as given, and the warning
    carries the "this is probably a mistake" signal that the clip was standing
    in for.
    """
    X_train, _, _ = normal_and_outliers

    with pytest.warns(UserWarning, match="above 0.5"):
        det = TribbleOneClassDetector(n_gaussians=2, contamination=0.9).fit(X_train)
    assert 0.85 < (det.predict(X_train) == -1).mean() < 0.95


def test_contamination_within_range_does_not_warn(normal_and_outliers):
    X_train, _, _ = normal_and_outliers
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        TribbleOneClassDetector(n_gaussians=2, contamination=0.5).fit(X_train)


def test_contamination_zero_flags_nothing(normal_and_outliers):
    """`<= 0` puts the threshold at the training minimum: nothing flagged."""
    X_train, _, X_out = normal_and_outliers
    det = TribbleOneClassDetector(n_gaussians=2, contamination=0.0).fit(X_train)
    assert (det.predict(X_train) == -1).sum() == 0
    # still separates -- contamination only moves the threshold, not the score
    assert (det.predict(X_out) == -1).mean() > 0.95


# -- score aggregation (issue #108) ---------------------------------------


@pytest.fixture
def high_dimensional_one_class():
    """Correlated normal data wide enough for the complement to saturate.

    128 whitened components: a *typical normal* point carries
    sum_j z_j^2 ~ 128, so firing = exp(-64) is far below float64's resolution
    next to 1.0 and `1 - firing` rounds to exactly 1.0 for normal and anomalous
    points alike.
    """
    d = 128
    rng = np.random.default_rng(0)
    A = rng.normal(size=(d, d))
    L = np.linalg.cholesky(A @ A.T / d + 0.1 * np.eye(d))
    cols = [f"f{i}" for i in range(d)]

    def normal(n):
        return pd.DataFrame(rng.normal(size=(n, d)) @ L.T, columns=cols)

    def anomalous(n):
        z = rng.normal(size=(n, d))
        z[:, :4] += 2.5          # shifted along a few low-variance directions
        return pd.DataFrame(z @ L.T, columns=cols)

    return normal(1500), normal(500), anomalous(500)


def test_surprisal_is_the_log_of_the_complement_firing(normal_and_outliers):
    """`surprisal` must be the same quantity as the complement, in the log
    domain -- not a different detector that happens to also work.

    Under the product t-norm, firing = prod_j m_j, so
    sum_j -log(m_j) == -log(firing) exactly. Pinning the identity is what makes
    "same ordering, more resolution" a claim rather than a hope.
    """
    X_train, X_normal, _ = normal_and_outliers
    kw = dict(n_gaussians=2, norm_conorm="probability")

    complement = TribbleOneClassDetector(score="complement", **kw).fit(X_train)
    surprisal = TribbleOneClassDetector(score="surprisal", **kw).fit(X_train)

    firing = 1.0 - complement.anomaly_score(X_normal)
    np.testing.assert_allclose(
        surprisal.anomaly_score(X_normal), -np.log(firing), rtol=1e-9
    )


def test_complement_saturates_where_surprisal_does_not(high_dimensional_one_class):
    """The defect in #108, and the fix, in one measurement.

    At 128 features the complement ties every point at exactly 1.0, so it is
    literally chance (AUROC 0.5) and catches nothing at a strict threshold. The
    same fitted memberships, summed in the log domain, still separate.
    """
    X_train, X_normal, X_anom = high_dimensional_one_class
    y = np.r_[np.zeros(len(X_normal)), np.ones(len(X_anom))]

    def evaluate(score):
        det = TribbleOneClassDetector(whiten=True, n_gaussians=1, score=score)
        det.fit(X_train)
        s_n, s_a = det.anomaly_score(X_normal), det.anomaly_score(X_anom)
        assert np.all(np.isfinite(s_n)) and np.all(np.isfinite(s_a)), (
            f"{score} produced non-finite scores"
        )
        # detection rate at a 1% false-positive threshold
        det_at_1pct = float((s_a > np.quantile(s_n, 0.99)).mean())
        return roc_auc_score(y, np.r_[s_n, s_a]), det_at_1pct, (s_n == 1.0).mean()

    auc_c, det_c, saturated = evaluate("complement")
    auc_s, det_s, _ = evaluate("surprisal")

    assert saturated > 0.95, "fixture no longer saturates the complement"
    assert auc_c < 0.6 and det_c < 0.05, "complement unexpectedly survives here"
    assert auc_s > 0.8, f"surprisal AUROC collapsed to {auc_s}"
    assert det_s > 0.15, f"surprisal det@1%FPR collapsed to {det_s}"


def test_surprisal_terms_are_finite_under_membership_underflow():
    """A Gaussian membership underflows to exactly 0.0 far from the mean, and
    -log(0) is inf. Infinities would re-tie every distant point at the top --
    the same failure the log domain exists to avoid, at the other end.
    """
    rng = np.random.default_rng(5)
    cols = [f"f{i}" for i in range(4)]
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=cols)
    det = TribbleOneClassDetector(n_gaussians=1, score="surprisal").fit(X)

    absurd = pd.DataFrame(np.full((3, 4), 1e6), columns=cols)
    scores = det.anomaly_score(absurd)
    assert np.all(np.isfinite(scores))
    assert np.all(scores > det.anomaly_score(X).max())


def test_trimmed_drops_the_largest_terms(normal_and_outliers):
    X_train, X_normal, _ = normal_and_outliers
    kw = dict(n_gaussians=1)
    trimmed = TribbleOneClassDetector(score="trimmed", trim=2, **kw).fit(X_train)
    full = TribbleOneClassDetector(score="surprisal", **kw).fit(X_train)

    terms = trimmed._surprisal_terms(trimmed._transform(trimmed._to_frame(X_normal)))
    np.testing.assert_allclose(
        trimmed.anomaly_score(X_normal), np.sort(terms, axis=1)[:, :-2].sum(axis=1)
    )
    # dropping non-negative terms can only lower the score
    assert np.all(trimmed.anomaly_score(X_normal) <= full.anomaly_score(X_normal) + 1e-9)


def test_trim_zero_equals_surprisal(normal_and_outliers):
    X_train, X_normal, _ = normal_and_outliers
    a = TribbleOneClassDetector(n_gaussians=1, score="trimmed", trim=0).fit(X_train)
    b = TribbleOneClassDetector(n_gaussians=1, score="surprisal").fit(X_train)
    np.testing.assert_allclose(a.anomaly_score(X_normal), b.anomaly_score(X_normal))


def test_default_score_is_unchanged(normal_and_outliers):
    """`complement` stays the default: #105's callers must be unaffected."""
    X_train, X_normal, _ = normal_and_outliers
    default = TribbleOneClassDetector(n_gaussians=2).fit(X_train)
    explicit = TribbleOneClassDetector(n_gaussians=2, score="complement").fit(X_train)
    assert default.score == "complement"
    np.testing.assert_allclose(
        default.anomaly_score(X_normal), explicit.anomaly_score(X_normal)
    )


def test_sklearn_conventions_hold_for_every_score(normal_and_outliers):
    """`score` changes the score's scale, not the estimator's contract."""
    X_train, X_normal, X_out = normal_and_outliers
    for score in ("complement", "surprisal", "trimmed"):
        det = TribbleOneClassDetector(n_gaussians=2, score=score).fit(X_train)
        assert det.anomaly_score(X_out).mean() > det.anomaly_score(X_normal).mean()
        np.testing.assert_allclose(det.score_samples(X_normal),
                                   -det.anomaly_score(X_normal))
        assert np.all(
            (det.decision_function(X_normal) >= 0) == (det.predict(X_normal) == 1)
        )
        assert (det.predict(X_out) == -1).mean() > 0.95


def test_invalid_score_and_trim_are_rejected(normal_and_outliers):
    X_train, _, _ = normal_and_outliers
    with pytest.raises(ValueError, match="score must be one of"):
        TribbleOneClassDetector(score="mahalanobis").fit(X_train)
    # 6 features in the fixture, so trimming 6 leaves an empty sum
    with pytest.raises(ValueError, match="trim=6"):
        TribbleOneClassDetector(score="trimmed", trim=6).fit(X_train)
    with pytest.raises(ValueError, match="trim=-1"):
        TribbleOneClassDetector(score="trimmed", trim=-1).fit(X_train)


def test_score_params_survive_clone():
    from sklearn.base import clone

    det = TribbleOneClassDetector(score="trimmed", trim=3)
    assert clone(det).get_params() == det.get_params()
