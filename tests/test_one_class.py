"""Tests for TribbleOneClassDetector -- fuzzy one-class novelty detection."""

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
