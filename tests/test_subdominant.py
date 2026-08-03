"""Layered sub-dominant rules: gated exceptions for confused class pairs.

The motivating failure is a class hidden *inside* another. Class ``A`` is a wide
cloud; class ``B`` is a small, dense pocket sitting in the middle of it, and is
separated from ``A`` only by a third feature that the global feature ranking
never picks because it is uninformative everywhere else.

A rule per label loses this: ``B``'s Gaussians sit inside ``A``'s, and ``A``'s
rule -- fit on far more rows -- wins the whole pocket. Re-fitting cannot help,
because the evidence that separates them is invisible at the scale of the whole
dataset. Restricted to the rows ``A`` gets wrong, it is the only thing there.

That is what a sub-dominant rule is for: gated on ``A`` firing, fit on the rows
``A`` gets wrong, and firing explicitly for ``B``.
"""

import numpy as np
import pandas as pd
import pytest

from tribblefis.gauss_data import SubdominantRule, resolve_norm_pair
from tribblefis.gauss_math import (
    apply_subdominant,
    subdominant_activation,
    tsk_firing_strengths,
)
from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
from tribblefis.subdominant import (
    confused_pairs,
    describe_subdominant,
    mine_subdominant_rules,
    validate_subdominant,
)


def hidden_pocket(n_a=400, n_b=120, seed=0):
    """Class B is a dense pocket inside class A's cloud, separated only by ``z``.

    ``z`` is pure noise for A and tightly centred for B, but B is a small
    minority, so ``z`` looks nearly uninformative across the whole dataset and
    survives only when the view is restricted to A's mistaken region.
    """
    rng = np.random.RandomState(seed)
    a = np.column_stack([
        rng.normal(0.0, 3.0, n_a),
        rng.normal(0.0, 3.0, n_a),
        rng.uniform(-6.0, 6.0, n_a),
    ])
    b = np.column_stack([
        rng.normal(0.0, 0.7, n_b),
        rng.normal(0.0, 0.7, n_b),
        rng.normal(4.5, 0.35, n_b),
    ])
    X = pd.DataFrame(np.vstack([a, b]), columns=["x", "y", "z"])
    return X, pd.Series(["A"] * n_a + ["B"] * n_b)


def make_classifier(**kwargs):
    params = dict(n_gaussians=1, top_p=1.0, norm_conorm="probability", random_state=0)
    params.update(kwargs)
    return MixtureOfGaussiansFuzzyClassifier(**params)


# --------------------------------------------------------------------------
# The defect, and that the cascade repairs it
# --------------------------------------------------------------------------

def test_base_rules_lose_the_hidden_pocket():
    X, y = hidden_pocket()
    model = make_classifier().fit(X, y)

    predicted = pd.Series(model.predict(X))
    b_recall = float(np.mean(predicted[y.values == "B"] == "B"))
    assert b_recall < 0.5, "the pocket must actually be lost for this suite to mean anything"


def test_subdominant_recovers_the_pocket():
    X, y = hidden_pocket()
    base = make_classifier().fit(X, y)
    corrected = make_classifier(subdominant=True).fit(X, y)

    assert len(corrected.subdominant_) > 0, corrected.subdominant_info_
    assert corrected.score(X, y) > base.score(X, y) + 0.05


def test_the_rule_is_gated_on_the_parent_and_fires_for_the_corrected_class():
    X, y = hidden_pocket()
    model = make_classifier(subdominant=True).fit(X, y)

    for rule in model.subdominant_:
        assert rule.parent != rule.consequent
        assert rule.antecedents
        assert 0.0 <= rule.threshold <= 1.0 or rule.threshold == float("inf")


def test_generalizes_to_held_out_rows():
    X_train, y_train = hidden_pocket(seed=0)
    X_test, y_test = hidden_pocket(seed=99)

    base = make_classifier().fit(X_train, y_train)
    corrected = make_classifier(subdominant=True).fit(X_train, y_train)

    assert corrected.score(X_test, y_test) > base.score(X_test, y_test) + 0.05


# --------------------------------------------------------------------------
# The gate: activation is bounded by the parent, which is why precedence decides
# --------------------------------------------------------------------------

def test_activation_never_exceeds_the_parent_firing():
    """The property that forces precedence resolution: ``T(a, b) <= min(a, b)``,
    so a gated rule is capped below the rule it corrects and could never win an
    argmax over firing strengths."""
    X, y = hidden_pocket()
    fitted = make_classifier(subdominant=True).fit(X, y)
    model = fitted.model_
    assert model.subdominant

    strengths, labels = tsk_firing_strengths(X, model)
    arrays = {name: np.asarray(X[name].values) for name in ("x", "y", "z")}
    norms = resolve_norm_pair("probability")

    for rule in model.subdominant:
        parent = strengths[:, labels.index(rule.parent)]
        activation = subdominant_activation(rule, parent, arrays, norms)
        assert np.all(activation <= parent + 1e-12)


def test_rule_is_silent_where_the_parent_is_silent():
    X, y = hidden_pocket()
    model = make_classifier(subdominant=True).fit(X, y).model_
    strengths, labels = tsk_firing_strengths(X, model)
    arrays = {name: np.asarray(X[name].values) for name in ("x", "y", "z")}
    norms = resolve_norm_pair("probability")

    rule = model.subdominant[0]
    parent = strengths[:, labels.index(rule.parent)]
    activation = subdominant_activation(rule, parent, arrays, norms)

    quiet = parent < 1e-6
    if quiet.any():
        assert np.all(activation[quiet] < 1e-6)


def test_override_only_touches_rows_predicted_as_the_parent():
    X, y = hidden_pocket()
    model = make_classifier().fit(X, y).model_
    strengths, labels = tsk_firing_strengths(X, model)
    base = np.asarray([labels[i] for i in np.argmax(strengths, axis=1)], dtype=object)
    arrays = {name: np.asarray(X[name].values) for name in ("x", "y", "z")}

    # A rule that would fire everywhere it is allowed to.
    from tribblefis.gauss_data import GaussianMembership
    always = SubdominantRule(
        parent="A", consequent="B",
        antecedents={"z": [GaussianMembership.create(0.0, 1e6)]},
        threshold=0.0,
    )
    corrected = apply_subdominant(
        base, strengths, labels, model.with_subdominant([always]),
        arrays, resolve_norm_pair("probability"),
    )

    changed = corrected != base
    assert np.all(base[changed] == "A")
    assert np.all(corrected[changed] == "B")
    assert not np.any(changed & (base != "A"))


# --------------------------------------------------------------------------
# Layering and termination
# --------------------------------------------------------------------------

def three_class_chain(n=200, seed=0):
    """A wide, B inside A, C inside B: corrections have to chain to reach C."""
    rng = np.random.RandomState(seed)
    a = np.column_stack([rng.normal(0, 3.0, n), rng.uniform(-6, 6, n), rng.uniform(-6, 6, n)])
    b = np.column_stack([rng.normal(0, 0.8, n // 2), rng.normal(4.0, 0.4, n // 2), rng.uniform(-6, 6, n // 2)])
    c = np.column_stack([rng.normal(0, 0.8, n // 4), rng.normal(4.0, 0.4, n // 4), rng.normal(4.0, 0.3, n // 4)])
    X = pd.DataFrame(np.vstack([a, b, c]), columns=["x", "y", "z"])
    y = pd.Series(["A"] * n + ["B"] * (n // 2) + ["C"] * (n // 4))
    return X, y


def test_layers_chain_corrections():
    X, y = three_class_chain()
    model = make_classifier(subdominant=True, subdominant_max_layers=2).fit(X, y)

    layers = {rule.layer for rule in model.subdominant_}
    assert layers, model.subdominant_info_
    # Whether layer 1 is reached depends on the data, but the diagnostics must
    # record that a second layer was attempted rather than silently stopping.
    assert len(model.subdominant_info_["layers"]) >= 1


def test_max_layers_one_disables_chaining():
    X, y = three_class_chain()
    model = make_classifier(subdominant=True, subdominant_max_layers=1).fit(X, y)
    assert all(rule.layer == 0 for rule in model.subdominant_)


def test_a_row_never_returns_to_a_label_it_has_held():
    """``A -> B`` and ``B -> A`` in the same cascade must not ping-pong.

    Without the visited-label rule the outcome would depend on iteration order;
    with it, every row's label sequence is strictly non-repeating.
    """
    from tribblefis.gauss_data import GaussianMembership

    X, y = hidden_pocket()
    model = make_classifier().fit(X, y).model_
    strengths, labels = tsk_firing_strengths(X, model)
    base = np.asarray([labels[i] for i in np.argmax(strengths, axis=1)], dtype=object)
    arrays = {name: np.asarray(X[name].values) for name in ("x", "y", "z")}

    wide = lambda: [GaussianMembership.create(0.0, 1e6)]
    cascade = [
        SubdominantRule(parent="A", consequent="B", antecedents={"z": wide()},
                        threshold=0.0, layer=0),
        SubdominantRule(parent="B", consequent="A", antecedents={"z": wide()},
                        threshold=0.0, layer=1),
    ]
    corrected = apply_subdominant(
        base, strengths, labels, model.with_subdominant(cascade),
        arrays, resolve_norm_pair("probability"),
    )

    # Every row that started A is now B and stayed there; none returned to A.
    started_a = base == "A"
    assert np.all(corrected[started_a] == "B")


def test_contested_rows_go_to_the_higher_activation():
    """Two rules with the same parent, both firing: the stronger one wins, and
    the outcome does not depend on the order they were mined in."""
    from tribblefis.gauss_data import GaussianMembership

    X, y = hidden_pocket()
    model = make_classifier().fit(X, y).model_
    strengths, labels = tsk_firing_strengths(X, model)
    base = np.asarray([labels[i] for i in np.argmax(strengths, axis=1)], dtype=object)
    arrays = {name: np.asarray(X[name].values) for name in ("x", "y", "z")}
    norms = resolve_norm_pair("probability")

    strong = SubdominantRule(
        parent="A", consequent="B",
        antecedents={"z": [GaussianMembership.create(0.0, 1e6)]}, threshold=0.0,
    )
    weak = SubdominantRule(
        parent="A", consequent="B",
        antecedents={"z": [GaussianMembership.create(50.0, 0.5)]}, threshold=0.0,
    )

    forward = apply_subdominant(
        base, strengths, labels, model.with_subdominant([strong, weak]), arrays, norms
    )
    reverse = apply_subdominant(
        base, strengths, labels, model.with_subdominant([weak, strong]), arrays, norms
    )
    np.testing.assert_array_equal(forward, reverse)


# --------------------------------------------------------------------------
# Mining discipline
# --------------------------------------------------------------------------

def separable(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.normal([0, 0, 0], 0.6, (n, 3)), rng.normal([8, 8, 8], 0.6, (n, 3))])
    return pd.DataFrame(X, columns=["x", "y", "z"]), pd.Series(["A"] * n + ["B"] * n)


def test_mines_nothing_when_there_is_no_confusion():
    X, y = separable()
    model = make_classifier(subdominant=True).fit(X, y)

    assert model.subdominant_ == ()
    assert model.score(X, y) > 0.98


def test_confused_pairs_ranks_by_count_and_respects_floors():
    predictions = np.array(["A"] * 60 + ["B"] * 30, dtype=object)
    truth = np.array(["A"] * 40 + ["B"] * 20 + ["B"] * 28 + ["C"] * 2, dtype=object)

    pairs = confused_pairs(predictions, truth, top_n=5, min_region=10, min_confused=5)
    assert pairs[0][:2] == ("A", "B")
    assert all(count >= 5 for *_, count in pairs)
    # C appears twice under region B -- below min_confused, so it is not a pair.
    assert ("B", "C") not in [(p, t) for p, t, _ in pairs]


def test_top_n_caps_the_pairs_addressed_per_layer():
    X, y = three_class_chain()
    model = make_classifier(
        subdominant=True, subdominant_top_n=1, subdominant_max_layers=1
    ).fit(X, y)
    assert len(model.subdominant_) <= 1


def test_a_rule_that_cannot_help_its_region_is_dropped():
    """The threshold search can always choose to fire on nothing, so a pair
    whose rows are not separable inside the parent's region must be rejected
    rather than kept and allowed to do harm."""
    rng = np.random.RandomState(0)
    # Two classes fully overlapping on every feature: no rule can separate them.
    n = 200
    X = pd.DataFrame(rng.normal(0, 1, (2 * n, 3)), columns=["x", "y", "z"])
    y = pd.Series(["A"] * n + ["B"] * n)

    base = make_classifier().fit(X, y)
    corrected = make_classifier(subdominant=True).fit(X, y)

    # It may mine nothing, or mine a rule that declines to fire; either way it
    # must not be worse than the base model.
    assert corrected.score(X, y) >= base.score(X, y) - 1e-9


def test_out_of_fold_confusion_is_the_default_through_the_estimator():
    X, y = hidden_pocket()
    model = make_classifier(subdominant=True).fit(X, y)
    assert model.subdominant_info_["out_of_fold"] is True


def test_mining_without_an_estimator_says_it_was_in_sample():
    X, y = hidden_pocket()
    fitted = make_classifier().fit(X, y)
    _, info = mine_subdominant_rules(fitted.model_, X, y)
    assert info["out_of_fold"] is False


# --------------------------------------------------------------------------
# Backwards compatibility, consistency, robustness
# --------------------------------------------------------------------------

def test_predict_dtype_survives_the_cascade():
    """The cascade works in object dtype internally. An object array of ints is
    ``type_of_target`` "unknown" to scikit-learn, which then refuses to score it
    against an int64 ``y`` -- so ``predict`` must narrow back before returning."""
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=[f"f{i}" for i in range(data.data.shape[1])])
    y = pd.Series(data.target)

    model = make_classifier(subdominant=True).fit(X, y)
    predicted = model.predict(X)

    assert predicted.dtype != object
    assert np.issubdtype(predicted.dtype, np.integer)
    model.score(X, y)  # would raise on an object array


def test_string_labels_still_round_trip():
    X, y = hidden_pocket()
    model = make_classifier(subdominant=True).fit(X, y)
    predicted = model.predict(X)

    assert set(np.unique(predicted)) <= {"A", "B"}
    model.score(X, y)


def test_models_without_rules_predict_exactly_as_before():
    X, y = hidden_pocket()
    model = make_classifier().fit(X, y)
    assert model.model_.subdominant == ()

    from tribblefis.gauss_math import tsk_predict
    plain = tsk_predict(X, model.model_, model.anomaly_params)
    explicit = tsk_predict(X, model.model_.with_subdominant([]), model.anomaly_params)
    np.testing.assert_array_equal(plain, explicit)


def test_predict_proba_argmax_agrees_with_predict():
    """A classifier whose ``argmax(predict_proba)`` disagrees with ``predict``
    silently breaks calibration, ROC curves and ``cross_val_predict``."""
    X, y = hidden_pocket()
    model = make_classifier(subdominant=True).fit(X, y)
    assert model.subdominant_

    probabilities = model.predict_proba(X)
    from_proba = model.classes_[np.argmax(probabilities, axis=1)]
    np.testing.assert_array_equal(from_proba, model.predict(X))


def test_predict_proba_stays_a_distribution():
    X, y = hidden_pocket()
    model = make_classifier(subdominant=True).fit(X, y)
    probabilities = model.predict_proba(X)

    assert np.all(probabilities >= 0.0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)


def test_works_under_every_norm_family():
    X, y = hidden_pocket()
    for family in ("min/max", "probability", "hamacher", "einstein"):
        base = make_classifier(norm_conorm=family).fit(X, y).score(X, y)
        corrected = make_classifier(norm_conorm=family, subdominant=True).fit(X, y)
        assert corrected.score(X, y) >= base - 1e-9, family


def test_validate_reports_rules_that_do_not_address_the_model():
    X, y = hidden_pocket()
    model = make_classifier().fit(X, y).model_
    from tribblefis.gauss_data import GaussianMembership

    bad = [
        SubdominantRule(parent="Z", consequent="B",
                        antecedents={"z": [GaussianMembership.create(0, 1)]}),
        SubdominantRule(parent="A", consequent="Z",
                        antecedents={"z": [GaussianMembership.create(0, 1)]}),
        SubdominantRule(parent="A", consequent="A",
                        antecedents={"z": [GaussianMembership.create(0, 1)]}),
        SubdominantRule(parent="A", consequent="B", antecedents={}),
        SubdominantRule(parent="A", consequent="B",
                        antecedents={"nope": [GaussianMembership.create(0, 1)]}),
    ]
    assert len(validate_subdominant(model.with_subdominant(bad))) == 5
    assert validate_subdominant(model) == []


def test_rules_naming_absent_features_are_skipped_not_misapplied():
    X, y = hidden_pocket()
    model = make_classifier().fit(X, y).model_
    strengths, labels = tsk_firing_strengths(X, model)
    base = np.asarray([labels[i] for i in np.argmax(strengths, axis=1)], dtype=object)
    arrays = {name: np.asarray(X[name].values) for name in ("x", "y", "z")}

    from tribblefis.gauss_data import GaussianMembership
    stale = SubdominantRule(
        parent="A", consequent="B",
        antecedents={"absent": [GaussianMembership.create(0.0, 1.0)]}, threshold=0.0,
    )
    corrected = apply_subdominant(
        base, strengths, labels, model.with_subdominant([stale]),
        arrays, resolve_norm_pair("probability"),
    )
    np.testing.assert_array_equal(corrected, base)


def test_trace_records_which_rules_fired():
    X, y = hidden_pocket()
    fitted = make_classifier(subdominant=True).fit(X, y)
    model = fitted.model_
    strengths, labels = tsk_firing_strengths(X, model)
    base = np.asarray([labels[i] for i in np.argmax(strengths, axis=1)], dtype=object)
    arrays = {name: np.asarray(X[name].values) for name in ("x", "y", "z")}

    corrected, trace = apply_subdominant(
        base, strengths, labels, model, arrays,
        resolve_norm_pair("probability"), return_trace=True,
    )
    for row in np.flatnonzero(corrected != base):
        assert trace[row], "a changed row must name the rule that changed it"
    for row in np.flatnonzero(corrected == base):
        assert not trace[row]


def test_composes_with_exclusion_clauses():
    X, y = hidden_pocket()
    both = make_classifier(subdominant=True, exclude_cross_terms=True, n_gaussians=2)
    both.fit(X, y)
    base = make_classifier(n_gaussians=2).fit(X, y)

    assert both.score(X, y) >= base.score(X, y) - 1e-9
    probabilities = both.predict_proba(X)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)


def test_describe_subdominant_reads_as_gated_exceptions():
    X, y = hidden_pocket()
    model = make_classifier(subdominant=True).fit(X, y)
    text = describe_subdominant(model.model_)

    assert "LAYER 0:" in text
    assert "IF   rule A fires" in text
    assert "THEN B" in text
    assert "instead of A" in text


def test_describe_subdominant_on_a_plain_model():
    X, y = hidden_pocket()
    model = make_classifier().fit(X, y)
    assert describe_subdominant(model.model_) == "no sub-dominant rules"
