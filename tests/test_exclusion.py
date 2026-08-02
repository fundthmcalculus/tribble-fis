"""Second-stage admissibility reduction: negated cross-terms.

The motivating failure is the checkerboard. Two features, two blobs each, and a
class that owns the diagonal cells while the other class owns the anti-diagonal:

        y
     Y2 | B    A
     Y1 | A    B
        +---------- x
          X1   X2

Both classes have *identical marginals* -- class A is at X1 and X2, and at Y1 and
Y2; so is class B. A rule per label therefore cannot separate them at all, no
matter where its Gaussians sit, because the rule folds each feature's terms
through a conorm before the t-norm ever sees the pairing. Its firing strength is
a function of the marginals, and the marginals are the same.

An exclusion clause names the cell, which is the one thing the rule cannot, so
this is the sharpest available test that the correction does what it claims.
"""

import numpy as np
import pandas as pd
import pytest

from tribblefis.exclusion import (
    describe_exclusions,
    mine_exclusions,
    validate_exclusions,
)
from tribblefis.gauss_data import ExclusionClause, resolve_norm_pair
from tribblefis.gauss_math import cell_strength, tsk_firing_strengths
from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier


def checkerboard(n_per_cell=90, seed=0, spread=0.55):
    """Four blobs at (+/-3, +/-3); class A on the diagonal, B on the anti-diagonal."""
    rng = np.random.RandomState(seed)
    rows, labels = [], []
    for cx, cy, label in [
        (-3.0, -3.0, "A"), (3.0, 3.0, "A"),
        (-3.0, 3.0, "B"), (3.0, -3.0, "B"),
    ]:
        rows.append(rng.normal([cx, cy], spread, size=(n_per_cell, 2)))
        labels += [label] * n_per_cell
    X = pd.DataFrame(np.vstack(rows), columns=["x", "y"])
    return X, pd.Series(labels)


def make_classifier(**kwargs):
    params = dict(n_gaussians=2, top_p=1.0, norm_conorm="probability", random_state=0)
    params.update(kwargs)
    return MixtureOfGaussiansFuzzyClassifier(**params)


# --------------------------------------------------------------------------
# The defect, and that the clause repairs it
# --------------------------------------------------------------------------

def test_outer_product_cannot_separate_the_checkerboard():
    """Baseline: without exclusions the rule base is at chance on this data.

    Not a property of a bad fit -- a consequence of the representation. Both
    classes present the same per-feature evidence, so both rules fire the same.
    """
    X, y = checkerboard()
    model = make_classifier().fit(X, y)
    assert model.score(X, y) < 0.65


def test_exclusions_recover_the_checkerboard():
    X, y = checkerboard()
    model = make_classifier(exclude_cross_terms=True).fit(X, y)

    assert len(model.exclusions_) > 0, model.exclusion_info_
    assert model.score(X, y) > 0.95


def test_clauses_name_cross_cells_and_blame_the_right_class():
    """Each clause must constrain *both* features and blame the other class.

    A clause naming one feature would be a marginal repair, and this data has no
    marginal defect to repair -- which is exactly why the cell is the only thing
    worth naming.
    """
    X, y = checkerboard()
    model = make_classifier(exclude_cross_terms=True).fit(X, y)

    for clause in model.exclusions_:
        assert set(clause.features) == {"x", "y"}
        assert clause.blamed is not None and clause.blamed != clause.label
        assert clause.purity <= 0.5
        assert clause.support >= 10


def test_generalizes_to_held_out_rows():
    """The clauses are cells of the input space, not memorised rows."""
    X_train, y_train = checkerboard(seed=0)
    X_test, y_test = checkerboard(seed=99)

    baseline = make_classifier().fit(X_train, y_train)
    corrected = make_classifier(exclude_cross_terms=True).fit(X_train, y_train)

    assert corrected.score(X_test, y_test) > baseline.score(X_test, y_test) + 0.25


# --------------------------------------------------------------------------
# Narrowness: the reduction touches only what it names
# --------------------------------------------------------------------------

def test_clause_lowers_only_its_parent_rule():
    X, y = checkerboard()
    fitted = make_classifier().fit(X, y)
    model = fitted.model_
    labels_before = None

    before, labels_before = tsk_firing_strengths(X, model)
    clause = ExclusionClause(label="A", terms=(("x", 0), ("y", 0)))
    after, labels_after = tsk_firing_strengths(X, model.with_exclusions([clause]))

    assert labels_before == labels_after
    parent = labels_after.index("A")
    other = labels_after.index("B")

    assert np.all(after[:, parent] <= before[:, parent] + 1e-12)
    np.testing.assert_allclose(after[:, other], before[:, other])
    # And it really did remove something, rather than being a silent no-op.
    assert after[:, parent].sum() < before[:, parent].sum() - 1e-6


def test_clause_spares_the_cells_its_terms_belong_to():
    """``NOT (x is X1 AND y is Y1)`` must leave ``X1 AND Y2`` alone.

    This is the whole point of excluding a cell rather than a term: the
    membership functions named by the clause keep firing for the parent
    everywhere else in the outer product.
    """
    X, y = checkerboard()
    fitted = make_classifier().fit(X, y)
    model = fitted.model_
    norms = resolve_norm_pair("probability")

    before, labels = tsk_firing_strengths(X, model)
    clause = ExclusionClause(label="A", terms=(("x", 0), ("y", 0)))
    after, _ = tsk_firing_strengths(X, model.with_exclusions([clause]))
    parent = labels.index("A")

    arrays = {name: np.asarray(X[name].values) for name in ("x", "y")}
    cell = cell_strength(clause, model, arrays, norms)
    outside = cell < 0.01
    assert outside.sum() > 100, "need rows outside the cell for this to mean anything"
    np.testing.assert_allclose(after[outside, parent], before[outside, parent], atol=1e-12)


def test_strength_zero_is_a_no_op_and_scales_monotonically():
    X, y = checkerboard()
    model = make_classifier().fit(X, y).model_
    before, labels = tsk_firing_strengths(X, model)
    parent = labels.index("A")

    totals = []
    for strength in (0.0, 0.25, 0.5, 1.0):
        clause = ExclusionClause(label="A", terms=(("x", 0), ("y", 0)), strength=strength)
        fired, _ = tsk_firing_strengths(X, model.with_exclusions([clause]))
        totals.append(fired[:, parent].sum())

    np.testing.assert_allclose(totals[0], before[:, parent].sum())
    assert totals[0] > totals[1] > totals[2] > totals[3]


# --------------------------------------------------------------------------
# Mining discipline: it must decline the cases it cannot justify
# --------------------------------------------------------------------------

def separable_blobs(n=150, seed=0):
    """Two well-separated classes: nothing for a cross-term to explain."""
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.normal([0, 0], 0.6, (n, 2)), rng.normal([6, 6], 0.6, (n, 2))])
    return pd.DataFrame(X, columns=["x", "y"]), pd.Series(["A"] * n + ["B"] * n)


def test_mines_nothing_when_the_rules_are_already_right():
    X, y = separable_blobs()
    model = make_classifier(exclude_cross_terms=True).fit(X, y)

    assert model.exclusions_ == ()
    assert model.score(X, y) > 0.98


def test_single_membership_models_have_no_outer_product_to_reduce():
    """One Gaussian per feature-label means the rule *is* one cell.

    There is no combination to condition on, so mining must find nothing and say
    why rather than leaving the caller to guess.
    """
    X, y = checkerboard()
    fitted = make_classifier(n_gaussians=1).fit(X, y)

    clauses, info = mine_exclusions(fitted.model_, X, y)
    assert clauses == []
    assert info["no_multi_mf_features"] is True


def test_marginal_confusion_is_not_treated_as_cross_confusion():
    """A cell that is impure only because one of its terms is impure is refused.

    Class A occupies all four cells; class B sits across A's whole left-hand
    ``x`` blob, in *both* ``y`` blobs. So ``X1&Y1`` is impure for A -- but no
    more impure than ``X1`` is on its own. That is a badly placed membership
    function, and the honest repair is to the membership fit; a cross-term would
    blame the pairing for something the single feature already does, and would
    withdraw only half of the region actually affected.
    """
    rng = np.random.RandomState(0)
    a = np.vstack([rng.normal([cx, cy], 0.55, (60, 2)) for cx in (-3, 3) for cy in (-3, 3)])
    b = np.vstack([rng.normal([-3, cy], 0.55, (70, 2)) for cy in (-3, 3)])
    X = pd.DataFrame(np.vstack([a, b]), columns=["x", "y"])
    y = pd.Series(["A"] * len(a) + ["B"] * len(b))

    fitted = make_classifier().fit(X, y)
    strict, info = mine_exclusions(fitted.model_, X, y, cross_margin=0.05)
    permissive, _ = mine_exclusions(fitted.model_, X, y, cross_margin=-1.0)

    # The cells exist and are impure -- they are refused on the cross test
    # specifically, not for want of evidence.
    assert info["cells_rejected"].get("not_cross_confusion", 0) > 0
    assert len(permissive) > len(strict)


def test_the_cross_test_does_not_fire_on_genuine_cross_confusion():
    """The converse of the previous test: on the checkerboard every impure cell
    is worse than both its terms, so nothing is refused as merely marginal."""
    X, y = checkerboard()
    fitted = make_classifier().fit(X, y)
    clauses, info = mine_exclusions(fitted.model_, X, y)

    assert len(clauses) > 0
    assert info["cells_rejected"].get("not_cross_confusion", 0) == 0


def test_min_support_refuses_thin_cells():
    X, y = checkerboard(n_per_cell=90)
    fitted = make_classifier().fit(X, y)

    found, _ = mine_exclusions(fitted.model_, X, y, min_support=10)
    starved, info = mine_exclusions(fitted.model_, X, y, min_support=10_000)
    assert len(found) > 0
    assert starved == []
    assert info["cells_examined"] == 0


def test_max_clauses_per_label_is_honoured():
    X, y = checkerboard()
    fitted = make_classifier().fit(X, y)
    clauses, _ = mine_exclusions(fitted.model_, X, y, max_clauses_per_label=1)

    counts = {}
    for clause in clauses:
        counts[clause.label] = counts.get(clause.label, 0) + 1
    assert all(count <= 1 for count in counts.values())


def test_order_must_span_at_least_two_features():
    X, y = checkerboard()
    fitted = make_classifier().fit(X, y)
    with pytest.raises(ValueError, match="order must be >= 2"):
        mine_exclusions(fitted.model_, X, y, order=1)


# --------------------------------------------------------------------------
# Backwards compatibility and robustness
# --------------------------------------------------------------------------

def test_models_without_clauses_fire_exactly_as_before():
    X, y = checkerboard()
    model = make_classifier().fit(X, y).model_

    assert model.exclusions == ()
    plain, labels = tsk_firing_strengths(X, model)
    explicit, _ = tsk_firing_strengths(X, model.with_exclusions([]))
    np.testing.assert_array_equal(plain, explicit)


def test_stale_clauses_are_reported_and_skipped_not_misapplied():
    """A clause indexing past the end of a membership list must not silently
    land on the wrong function."""
    X, y = checkerboard()
    model = make_classifier().fit(X, y).model_

    stale = ExclusionClause(label="A", terms=(("x", 99), ("y", 0)))
    unknown_feature = ExclusionClause(label="A", terms=(("nope", 0), ("y", 0)))
    with_stale = model.with_exclusions([stale, unknown_feature])

    problems = validate_exclusions(with_stale)
    assert len(problems) == 2

    before, _ = tsk_firing_strengths(X, model)
    after, _ = tsk_firing_strengths(X, with_stale)
    np.testing.assert_array_equal(before, after)


def test_exclusions_apply_under_every_norm_family():
    """Scored against each family's *own* baseline. The families differ in how
    much of a cell ``T(w, 1 - c)`` actually withdraws -- min/max only clamps to
    ``1 - c`` where that is the smaller of the two -- so an absolute accuracy bar
    would be testing the family, not the clause."""
    X, y = checkerboard()
    for family in ("min/max", "probability", "hamacher", "einstein"):
        baseline = make_classifier(norm_conorm=family).fit(X, y).score(X, y)
        corrected = make_classifier(
            norm_conorm=family, exclude_cross_terms=True
        ).fit(X, y)
        assert corrected.exclusions_, family
        assert corrected.score(X, y) > baseline + 0.2, family


def test_predict_proba_stays_a_distribution_with_clauses():
    X, y = checkerboard()
    model = make_classifier(exclude_cross_terms=True).fit(X, y)
    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.all(probabilities >= 0.0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)


def test_anomaly_column_reflects_the_narrowed_rules():
    """The anomaly strength is derived from the class columns, so it has to be
    computed after the clauses narrow them -- a sample in an excluded cell is no
    longer well explained by any rule."""
    from tribblefis.gauss_data import AnomalyParameters

    X, y = checkerboard()
    model = make_classifier().fit(X, y).model_
    details = AnomalyParameters(include_anomaly=True, threshold=0.0, norm_conorm="probability")

    clause = ExclusionClause(label="A", terms=(("x", 0), ("y", 0)))
    arrays = {name: np.asarray(X[name].values) for name in ("x", "y")}
    cell = cell_strength(clause, model, arrays, resolve_norm_pair("probability"))
    inside = cell > 0.7
    assert inside.sum() > 10

    before, labels = tsk_firing_strengths(X, model, anomaly_details=details)
    after, _ = tsk_firing_strengths(
        X, model.with_exclusions([clause]), anomaly_details=details
    )
    anomaly = labels.index("anomaly")
    assert after[inside, anomaly].mean() > before[inside, anomaly].mean()


def test_describe_exclusions_reads_as_rule_exceptions():
    X, y = checkerboard()
    model = make_classifier(exclude_cross_terms=True).fit(X, y)
    text = describe_exclusions(model.model_)

    assert "AND NOT" in text
    assert "RULE" in text
    for clause in model.exclusions_:
        assert f"mostly {clause.blamed}" in text


def test_describe_exclusions_on_a_plain_model():
    X, y = checkerboard()
    model = make_classifier().fit(X, y)
    assert describe_exclusions(model.model_) == "no exclusion clauses"
