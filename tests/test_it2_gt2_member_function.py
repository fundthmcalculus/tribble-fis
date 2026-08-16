"""End-to-end IT2/GT2 fit/predict coverage for member_function="trap"/"triangular".

Closes the rest of #144's gap: `_convert_to_it2`/`_convert_to_gt2` now widen
whichever Type-1 membership shape the base model was fit with
(`gauss_data.widen_membership`), and IT2/GT2TribbleClassifier/Regressor now
expose `member_function` to actually reach that path.
"""

import numpy as np
import pytest
from sklearn.datasets import load_wine, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from tribblefis.it2_classifier import IT2TribbleClassifier
from tribblefis.it2_regressor import IT2TribbleRegressor
from tribblefis.gt2_classifier import GT2TribbleClassifier
from tribblefis.gt2_regressor import GT2TribbleRegressor

MEMBER_FUNCTIONS = ["gaussian", "trap", "triangular"]


@pytest.fixture
def wine_split():
    X, y = load_wine(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


@pytest.fixture
def regression_split():
    X, y = make_regression(n_samples=300, n_features=5, n_informative=3, noise=5.0, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.mark.parametrize("member_function", MEMBER_FUNCTIONS)
@pytest.mark.parametrize("cls", [IT2TribbleClassifier, GT2TribbleClassifier])
def test_classifier_fits_predicts_and_intervals_are_ordered(cls, member_function, wine_split):
    X_train, X_test, y_train, y_test = wine_split
    clf = cls(top_n=4, member_function=member_function, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert accuracy_score(y_test, y_pred) > 0.5

    upper, lower = clf.predict_intervals(X_test)
    assert np.all(upper >= lower - 1e-9)


@pytest.mark.parametrize("member_function", MEMBER_FUNCTIONS)
@pytest.mark.parametrize("cls", [IT2TribbleRegressor, GT2TribbleRegressor])
def test_regressor_fits_predicts_and_intervals_contain_point_estimate(cls, member_function, regression_split):
    X_train, X_test, y_train, y_test = regression_split
    reg = cls(top_n=3, n_gaussians=2, member_function=member_function, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    assert np.all(np.isfinite(y_pred))
    assert r2_score(y_test, y_pred) > -1.0  # not a total blowup; trap/triangular fits are noisier

    y_lower, y_upper = reg.predict_intervals(X_test)
    assert np.all(y_upper >= y_lower - 1e-6)
    assert np.all(y_pred >= y_lower - 1e-6)
    assert np.all(y_pred <= y_upper + 1e-6)


@pytest.mark.parametrize("member_function", ["trap", "triangular"])
def test_widen_membership_used_for_conversion_not_gaussian_fallback(member_function):
    """A trap/triangular IT2 model's antecedents should actually be the
    matching non-Gaussian type -- guards against a silent fallback to
    Gaussian widening somewhere in the conversion path."""
    from tribblefis.gauss_data import TrapezoidMembership, TriangularMembership

    X, y = load_wine(return_X_y=True)
    clf = IT2TribbleClassifier(top_n=3, member_function=member_function, random_state=42)
    clf.fit(X, y)

    expected_type = TrapezoidMembership if member_function == "trap" else TriangularMembership
    any_feature_model = next(iter(clf.model_.feature_models.values()))
    any_label_model = next(iter(any_feature_model.label_models.values()))
    for mf in any_label_model.memberships:
        assert isinstance(mf.upper_mf, expected_type)
        assert isinstance(mf.lower_mf, expected_type)
