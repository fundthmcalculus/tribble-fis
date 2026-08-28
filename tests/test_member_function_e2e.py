"""End-to-end fit/predict coverage for member_function="trap"/"triangular".

Only EM-level math (`test_triangle_math.py`) was covered before this; no test
anywhere fit a `TribbleClassifier`/`TribbleRegressor` end-to-end with a
non-Gaussian member_function. `TribbleRegressor` didn't even accept the
parameter (see #144's investigation) -- these are the baseline smoke tests
the rest of the trapezoid/triangular IT2/GT2 stack builds on.
"""

import inspect

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_wine, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.gauss_data import (
    GaussianMembership,
    TrapezoidMembership,
    TriangularMembership,
)

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


# --------------------------------------------------------------------------
# #164 re-verification: the tests above prove `member_function` is *accepted*
# and that a fit/predict round-trip survives, which is not the same as proving
# the value reaches the antecedent builder. A wrapper that stores the parameter
# and then constructs its inner estimator without it passes every test above
# while silently fitting Gaussians -- that exact failure mode is live today in
# `TribbleSequenceClassifier._make_layer`, which drops `member_function` on the
# floor. The assertions below are the ones that would catch it:
#
#   * the membership objects actually built carry the requested shape,
#   * the shipped default really is "gaussian"/"fast",
#   * different families produce genuinely different models (not a no-op knob),
#   * get_params/set_params/clone round-trip and a cloned estimator still fits
#     the requested family, which is what sklearn's search/CV wrappers rely on.
#
# `trapz_method="em"` also had no end-to-end coverage on either estimator; it
# is parametrized here alongside "fast".
# --------------------------------------------------------------------------

# (member_function, trapz_method, expected membership type)
MEMBERSHIP_CASES = [
    ("gaussian", "fast", GaussianMembership),
    ("trap", "fast", TrapezoidMembership),
    ("trap", "em", TrapezoidMembership),
    ("triangular", "fast", TriangularMembership),
]


def _regression_frame():
    X, y = make_regression(
        n_samples=300, n_features=5, n_informative=3, noise=5.0, random_state=42
    )
    return X, y


def _built_types(estimator):
    return {type(m).__name__ for m in estimator.model_.all_membership_fcns}


@pytest.mark.parametrize("member_function,trapz_method,expected", MEMBERSHIP_CASES)
def test_regressor_member_function_reaches_membership_construction(
    member_function, trapz_method, expected
):
    """The parameter must select the antecedent family, not merely be stored."""
    X, y = _regression_frame()
    reg = TribbleRegressor(
        top_n=3, n_gaussians=2, member_function=member_function,
        trapz_method=trapz_method, random_state=42,
    )
    reg.fit(X, y)

    built = reg.model_.all_membership_fcns
    assert built, "fit produced no membership functions"
    assert all(isinstance(m, expected) for m in built), _built_types(reg)


@pytest.mark.parametrize("member_function,trapz_method,expected", MEMBERSHIP_CASES)
def test_classifier_member_function_reaches_membership_construction(
    member_function, trapz_method, expected
):
    """Same assertion on the classifier the regressor was mirrored from."""
    X, y = load_wine(return_X_y=True)
    clf = TribbleClassifier(
        top_n=4, member_function=member_function,
        trapz_method=trapz_method, random_state=42,
    )
    clf.fit(X, y)

    built = clf.model_.all_membership_fcns
    assert built, "fit produced no membership functions"
    assert all(isinstance(m, expected) for m in built), _built_types(clf)


def test_regressor_defaults_are_gaussian_and_fast():
    """#164's hard constraint: adding the knob must not move the default.

    Guards both halves -- the declared signature default and the family the
    default actually builds -- because either drifting alone would silently
    change every existing TribbleRegressor result.
    """
    params = inspect.signature(TribbleRegressor.__init__).parameters
    assert params["member_function"].default == "gaussian"
    assert params["trapz_method"].default == "fast"

    X, y = _regression_frame()
    reg = TribbleRegressor(top_n=3, n_gaussians=2, random_state=42).fit(X, y)
    assert _built_types(reg) == {"GaussianMembership"}


def test_regressor_member_function_actually_changes_the_model():
    """A knob that is read but has no effect on predictions is still broken."""
    X, y = _regression_frame()
    preds = {}
    for member_function, trapz_method, _ in MEMBERSHIP_CASES:
        reg = TribbleRegressor(
            top_n=3, n_gaussians=2, member_function=member_function,
            trapz_method=trapz_method, random_state=42,
        ).fit(X, y)
        preds[(member_function, trapz_method)] = np.asarray(reg.predict(X), dtype=float)

    keys = list(preds)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not np.array_equal(preds[a], preds[b]), f"{a} and {b} predict identically"


@pytest.mark.parametrize("member_function,trapz_method,expected", MEMBERSHIP_CASES)
def test_regressor_member_function_survives_clone(member_function, trapz_method, expected):
    """sklearn's search/CV wrappers refit clones, not the object you handed them."""
    original = TribbleRegressor(
        top_n=3, n_gaussians=2, member_function=member_function,
        trapz_method=trapz_method, random_state=42,
    )
    params = original.get_params()
    assert params["member_function"] == member_function
    assert params["trapz_method"] == trapz_method

    copy = clone(original)
    assert copy.get_params() == params

    X, y = _regression_frame()
    copy.fit(X, y)
    assert all(isinstance(m, expected) for m in copy.model_.all_membership_fcns)


def test_regressor_set_params_reaches_the_next_fit():
    """set_params must retarget the antecedent family, not just the attribute."""
    X, y = _regression_frame()
    reg = TribbleRegressor(top_n=3, n_gaussians=2, random_state=42)
    reg.fit(X, y)
    assert _built_types(reg) == {"GaussianMembership"}

    reg.set_params(member_function="triangular")
    assert reg.get_params()["member_function"] == "triangular"
    reg.fit(X, y)
    assert _built_types(reg) == {"TriangularMembership"}
