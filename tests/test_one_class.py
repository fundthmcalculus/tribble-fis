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
