"""End-to-end fit/predict coverage for member_function="trap"/"triangular".

Only EM-level math (`test_triangle_math.py`) was covered before this; no test
anywhere fit a `TribbleClassifier`/`TribbleRegressor` end-to-end with a
non-Gaussian member_function. `TribbleRegressor` didn't even accept the
parameter (see #144's investigation) -- these are the baseline smoke tests
the rest of the trapezoid/triangular IT2/GT2 stack builds on.
"""

import numpy as np
import pytest
from sklearn.datasets import load_wine, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.gaussian_regressor import TribbleRegressor

MEMBER_FUNCTIONS = ["gaussian", "trap", "triangular"]


@pytest.mark.parametrize("member_function", MEMBER_FUNCTIONS)
def test_classifier_fits_and_predicts(member_function):
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = TribbleClassifier(top_n=4, member_function=member_function, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert y_pred.shape == y_test.shape
    assert np.all(np.isin(y_pred, clf.classes_))
    assert accuracy_score(y_test, y_pred) > 0.5


@pytest.mark.parametrize("member_function", MEMBER_FUNCTIONS)
def test_regressor_fits_and_predicts(member_function):
    X, y = make_regression(n_samples=300, n_features=5, n_informative=3, noise=5.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    reg = TribbleRegressor(top_n=3, n_gaussians=2, member_function=member_function, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    assert y_pred.shape == y_test.shape
    assert np.all(np.isfinite(y_pred))
    assert r2_score(y_test, y_pred) > 0.0


def test_regressor_unknown_member_function_raises():
    X, y = make_regression(n_samples=50, n_features=2, random_state=42)
    reg = TribbleRegressor(member_function="not-a-real-shape")
    with pytest.raises(ValueError, match="Unknown member_function"):
        reg.fit(X, y)


def test_regressor_unknown_trapz_method_raises():
    X, y = make_regression(n_samples=50, n_features=2, random_state=42)
    reg = TribbleRegressor(member_function="trap", trapz_method="not-a-real-method")
    with pytest.raises(ValueError, match="Unknown trapz_method"):
        reg.fit(X, y)
