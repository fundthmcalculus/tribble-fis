"""Second-stage admissibility reduction: mining negated cross-terms.

The problem
-----------
A zeroth-order TSK classifier gives each label one rule, and that rule is a
conjunction *of disjunctions* -- one disjunction per feature, folded by the
t-conorm over that feature-label's membership functions::

    IF (x is X1 OR X2 OR X4) AND (y is Y1 OR Y3) THEN A

The conorm is where the information is lost. By the time the t-norm ANDs the
features together, the rule no longer knows *which* ``x`` term fired, so it
cannot condition on the pairing. It therefore admits the entire outer product
of its disjunctions -- ``X1&Y1, X2&Y1, X4&Y1, ... X4&Y3`` -- with no way to
prefer one cell over another.

That is correct when the label really does occupy every cell. It is wrong when
it does not. If ``X1&Y3`` is in truth class ``B``, rule ``A`` still claims it at
full strength, and no amount of re-fitting the ``(mu, sigma)`` of ``X1`` or
``Y3`` fixes it: both terms are individually right for ``A``, and moving either
one to dodge the cell costs ``A`` the cells it legitimately owns. The defect is
in the *combination*, so the repair has to name the combination.

The correction
--------------
Mine the cells from the training data and append the negation of the bad ones to
the parent rule::

    IF (x is X1 OR X2 OR X4) AND (y is Y1 OR Y3)
       AND NOT (x is X1 AND y is Y3)
    THEN A

Each clause is attached to one parent label and names one cell, so the reduction
is as narrow as the evidence: rule ``B`` is untouched, ``X1`` still fires for
``A`` alongside ``Y1``, and ``Y3`` still fires for ``A`` alongside ``X2`` and
``X4``. Only the confused block is withdrawn. See
:class:`~tribblefis.gauss_data.ExclusionClause` for the representation and
:func:`~tribblefis.gauss_math.apply_exclusions` for the inference half.

Why this is not just more membership functions
----------------------------------------------
The obvious alternative is to give each label a *rule per cell* rather than a
rule per label, which restores the lost pairing by never folding the conorm. It
also multiplies the rule base by the size of the outer product, which is what
the conorm was there to avoid, and it re-fits everything -- including the cells
that were already right. Mining exclusions keeps one rule per label, adds a
clause only where the data shows a specific cell is wrong, and leaves the
interpretation of the rule base intact: the clauses read as exceptions, and
:func:`describe_exclusions` prints them as such.

What mining will and will not find
----------------------------------
Only features carrying **two or more** membership functions for a label can
participate in a cell. A feature with a single term contributes no choice to the
outer product, so a cell naming it would not be a cross-term at all -- excluding
``X1 AND y_only`` is just excluding ``X1``, a marginal repair that belongs in
the membership fit rather than here. Models where every feature-label has one
Gaussian therefore mine nothing, and :func:`mine_exclusions` reports that in its
diagnostics rather than silently returning an empty list.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .gauss_data import (
    AnomalyParameters,
    ExclusionClause,
    GaussianMixtureModel,
    NormPair,
    resolve_norm_pair,
)
from .gauss_math import apply_exclusions, cell_strength, tsk_firing_strengths

__all__ = [
    "mine_exclusions",
    "validate_exclusions",
    "describe_exclusions",
    "apply_exclusions",
    "cell_strength",
]


def _feature_arrays(X: pd.DataFrame, model: GaussianMixtureModel) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(X[name].values)
        for name in model.feature_models
        if name in X.columns
    }


def _dominant_membership(
    model: GaussianMixtureModel,
    label: Any,
    feature_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per feature, the index of the membership function that fires hardest.

    This is the cell coordinate of each row. The t-conorm inside the rule
    reports only the folded value, but the argmax of the same terms says *which*
    one carried it -- which is precisely the information the rule discards and a
    clause needs in order to name a cell.

    Only features with two or more memberships for ``label`` are returned; the
    rest offer no choice and so cannot be part of a cross-term.
    """
    coordinates: dict[str, np.ndarray] = {}
    for feature_name, feature_model in model.feature_models.items():
        label_model = feature_model.label_models.get(label)
        if label_model is None or len(label_model.memberships) < 2:
            continue
        feature_data = feature_arrays.get(feature_name)
        if feature_data is None:
            continue
        values = np.stack([mf.evaluate(feature_data) for mf in label_model.memberships])
        coordinates[feature_name] = np.argmax(values, axis=0)
    return coordinates


def _marginal_purity(
    coordinate: np.ndarray, mf_index: int, is_label: np.ndarray
) -> float:
    """Fraction of rows that really are the parent label, among candidate rows
    whose dominant membership on one feature is ``mf_index``.

    The single-feature counterpart of a cell's purity, and the reference a cell
    has to beat to count as *cross*-confusion.
    """
    rows = coordinate == mf_index
    total = int(rows.sum())
    return float(is_label[rows].mean()) if total else 1.0


def mine_exclusions(
    model: GaussianMixtureModel,
    X: pd.DataFrame,
    y,
    *,
    norms: NormPair | None = None,
    anomaly_details: AnomalyParameters | None = None,
    order: int | Sequence[int] = 2,
    min_support: int = 10,
    max_purity: float = 0.5,
    cross_margin: float = 0.05,
    max_clauses_per_label: int = 4,
    strength: float = 1.0,
    max_features: int = 24,
) -> tuple[list[ExclusionClause], dict[str, Any]]:
    """Mine negated cross-terms for the rules of a fitted model.

    Walks each label's *own* prediction region, groups those rows into cells of
    the label's outer product, and emits a clause for every cell the data says
    belongs to a different class.

    The region restriction is what makes the reduction targeted. A clause can
    only ever lower one label's firing strength, so it can only change a
    prediction on rows where that label currently *wins*. Mining anywhere else
    would collect cells whose exclusion is unobservable, and rank them against
    cells that matter.

    Args:
        model: A fitted model. Mine after any antecedent refinement -- refining
            rewrites the memberships a clause indexes into, and the refinement
            objective does not know about clauses.
        X: Training features. Must carry the model's feature columns.
        y: Training labels, aligned with ``X``.
        norms: Resolved operator pair. Defaults to ``anomaly_details``' pair, or
            the library default. Use the pair the model will predict with:
            cells are scored through the same t-norm inference will apply.
        anomaly_details: Passed to the baseline firing-strength call. The
            anomaly column, if any, is ignored when locating each label's
            region -- an exclusion competes with other *classes*.
        order: Size of the cells to consider, i.e. how many features a clause
            names. ``2`` (pairs) is the default and covers the confusion the
            outer product actually produces most of; pass a sequence such as
            ``(2, 3)`` to also consider larger cells. Orders are tried smallest
            first and a larger cell is skipped when a clause already accepted
            for that label covers a subset of its terms -- the smaller clause
            already withdraws the region, and the larger one would only make the
            rule harder to read.
        min_support: Minimum rows in a cell before it may be excluded, and also
            the minimum number of those rows belonging to the blamed class. The
            main defence against carving a hole out of a rule on noise.
        max_purity: A cell is a candidate only when at most this fraction of its
            rows really are the parent label. The default ``0.5`` means "the
            parent is a minority in its own cell".
        cross_margin: A cell must be worse than *each* of its constituent
            single-feature terms by this margin in purity. This is the test that
            separates genuine cross-confusion from a membership function that is
            simply badly placed: if ``X1`` is impure on its own, the cell
            ``X1&Y3`` inherits that impurity and excluding the pair would be
            treating a marginal defect with a cross-term. Such a feature wants
            re-fitting (or refinement), not a clause.
        max_clauses_per_label: Cap on clauses per parent rule, highest net
            confusion first. Keeps the rule base readable and bounds how much of
            a rule the second stage may withdraw.
        strength: :attr:`ExclusionClause.strength` for every mined clause.
            ``1.0`` is the hard "AND NOT cell".
        max_features: Only the features with the most membership functions for a
            label are paired, capped here. Cell enumeration is combinatorial in
            the feature count, and features with few terms contribute the least
            outer-product ambiguity.

    Returns:
        ``(clauses, diagnostics)``. ``diagnostics`` records what mining saw --
        ``eligible_features`` per label, ``cells_examined``, ``cells_rejected``
        by reason, and ``regions`` sizes -- so an empty result can be explained
        rather than guessed at. In particular ``no_multi_mf_features`` being set
        means the model has at most one membership function per feature-label,
        in which case it has no outer product to reduce.
    """
    if norms is None:
        norms = anomaly_details.norms() if anomaly_details else resolve_norm_pair()

    orders = sorted({int(order)} if isinstance(order, int) else {int(o) for o in order})
    if any(o < 2 for o in orders):
        raise ValueError(f"order must be >= 2 (a cell spans two or more features); got {order}")

    X = X.reset_index(drop=True)
    y_values = np.asarray(y.values if isinstance(y, pd.Series) else y, dtype=object)

    feature_arrays = _feature_arrays(X, model)
    baseline, labels = tsk_firing_strengths(
        X, model, anomaly_details=anomaly_details, norms=norms, feature_arrays=feature_arrays
    )

    # Class columns only: a clause redistributes probability among classes, so
    # the region that matters is "which class rule currently wins here".
    anomaly_label = (
        anomaly_details.label if anomaly_details and anomaly_details.include_anomaly else None
    )
    class_labels = [label for label in labels if label != anomaly_label]
    class_block = baseline[:, : len(class_labels)]
    winner = np.asarray([class_labels[i] for i in np.argmax(class_block, axis=1)], dtype=object)

    diagnostics: dict[str, Any] = {
        "regions": {},
        "eligible_features": {},
        "cells_examined": 0,
        "cells_rejected": Counter(),
        "no_multi_mf_features": True,
    }
    clauses: list[ExclusionClause] = []

    for label in class_labels:
        region = winner == label
        n_region = int(region.sum())
        diagnostics["regions"][label] = n_region
        if n_region < min_support:
            continue

        coordinates = _dominant_membership(model, label, feature_arrays)
        if coordinates:
            diagnostics["no_multi_mf_features"] = False
        if len(coordinates) < min(orders):
            diagnostics["eligible_features"][label] = list(coordinates)
            continue

        # Most-split features first: they carry the most outer-product ambiguity
        # and so the most opportunity for a cell to disagree with its marginals.
        ranked = sorted(
            coordinates,
            key=lambda name: (
                -len(model.feature_models[name].label_models[label].memberships),
                name,
            ),
        )[:max_features]
        diagnostics["eligible_features"][label] = ranked

        # Restrict to the label's own region once, and score every cell there.
        region_coordinates = {name: coordinates[name][region] for name in ranked}
        is_label = (y_values[region] == label).astype(float)
        y_region = y_values[region]

        accepted: list[tuple[float, ExclusionClause]] = []
        accepted_terms: list[frozenset] = []

        for size in orders:
            for subset in combinations(ranked, size):
                stacked = np.stack([region_coordinates[name] for name in subset], axis=1)
                cells, inverse = np.unique(stacked, axis=0, return_inverse=True)
                # NumPy 2 returns an inverse shaped like the axis it reduced;
                # NumPy 1 returns it flat. Ravel so row indexing matches either.
                inverse = np.asarray(inverse).ravel()

                for cell_id, cell in enumerate(cells):
                    rows = inverse == cell_id
                    support = int(rows.sum())
                    if support < min_support:
                        continue
                    diagnostics["cells_examined"] += 1

                    terms = tuple(
                        (name, int(index)) for name, index in zip(subset, cell)
                    )
                    term_set = frozenset(terms)
                    if any(previous <= term_set for previous in accepted_terms):
                        # A smaller accepted cell already withdraws this region.
                        diagnostics["cells_rejected"]["covered_by_smaller_cell"] += 1
                        continue

                    purity = float(is_label[rows].mean())
                    if purity > max_purity:
                        diagnostics["cells_rejected"]["parent_still_majority"] += 1
                        continue

                    others = Counter(
                        value for value in y_region[rows] if value != label
                    )
                    if not others:
                        diagnostics["cells_rejected"]["no_blamed_class"] += 1
                        continue
                    blamed, blamed_count = others.most_common(1)[0]
                    own_count = int((y_region[rows] == label).sum())
                    if blamed_count < min_support or blamed_count <= own_count:
                        diagnostics["cells_rejected"]["blamed_class_too_weak"] += 1
                        continue

                    # The cross test: the conjunction must be worse than every
                    # one of its parts. Otherwise the impurity is marginal and
                    # belongs to the membership fit, not to a cross-term.
                    marginals = [
                        _marginal_purity(region_coordinates[name], index, is_label)
                        for name, index in terms
                    ]
                    if min(marginals) < purity + cross_margin:
                        diagnostics["cells_rejected"]["not_cross_confusion"] += 1
                        continue

                    accepted.append((
                        float(blamed_count - own_count),
                        ExclusionClause(
                            label=label,
                            terms=terms,
                            strength=float(strength),
                            blamed=blamed,
                            support=support,
                            purity=purity,
                        ),
                    ))
                    accepted_terms.append(term_set)

        accepted.sort(key=lambda item: item[0], reverse=True)
        clauses.extend(clause for _, clause in accepted[:max_clauses_per_label])

    diagnostics["cells_rejected"] = dict(diagnostics["cells_rejected"])
    diagnostics["n_clauses"] = len(clauses)
    return clauses, diagnostics


def validate_exclusions(
    model: GaussianMixtureModel,
    clauses: Iterable[ExclusionClause] | None = None,
) -> list[str]:
    """Reasons any clause fails to address ``model``, one message per bad clause.

    Clauses index memberships positionally, so a model whose feature-label
    membership lists have changed length (after ``augment``, or a differently
    configured re-fit) can silently carry clauses that name the wrong function.
    Inference skips such clauses rather than mis-firing; this reports them.
    An empty list means every clause resolves.
    """
    clauses = model.exclusions if clauses is None else clauses
    problems: list[str] = []
    for clause in clauses:
        for feature_name, mf_index in clause.terms:
            feature_model = model.feature_models.get(feature_name)
            if feature_model is None:
                problems.append(
                    f"{clause.label}: no feature {feature_name!r} in the model"
                )
                break
            label_model = feature_model.label_models.get(clause.label)
            if label_model is None:
                problems.append(
                    f"{clause.label}: feature {feature_name!r} carries no label {clause.label!r}"
                )
                break
            if not (0 <= mf_index < len(label_model.memberships)):
                problems.append(
                    f"{clause.label}: {feature_name!r} membership index {mf_index} is out of "
                    f"range (feature-label has {len(label_model.memberships)})"
                )
                break
    return problems


def describe_exclusions(
    model: GaussianMixtureModel,
    clauses: Iterable[ExclusionClause] | None = None,
) -> str:
    """The mined clauses as readable rule exceptions, grouped by parent rule.

    Each line names the cell that was withdrawn, the class the rows in it really
    belonged to, and the support behind the decision, so a reader can judge the
    exception on the same evidence that produced it.
    """
    clauses = list(model.exclusions if clauses is None else clauses)
    if not clauses:
        return "no exclusion clauses"

    by_label: dict[Any, list[ExclusionClause]] = {}
    for clause in clauses:
        by_label.setdefault(clause.label, []).append(clause)

    lines: list[str] = []
    for label, label_clauses in by_label.items():
        lines.append(f"RULE {label}:")
        for clause in label_clauses:
            cell = " AND ".join(
                f"{feature} is mf{index}" for feature, index in clause.terms
            )
            detail = f"n={clause.support}, {clause.purity:.0%} really {label}"
            if clause.blamed is not None:
                detail = f"mostly {clause.blamed}; {detail}"
            suffix = "" if clause.strength == 1.0 else f" [strength {clause.strength:g}]"
            lines.append(f"  AND NOT ({cell})   -- {detail}{suffix}")
    return "\n".join(lines)
