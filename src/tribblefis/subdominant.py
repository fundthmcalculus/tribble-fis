"""Layered correction: mining sub-dominant rules for confused class pairs.

The problem
-----------
A rule per label is a single account of that class, fit from that class's own
marginals. Where two classes overlap, one rule wins the whole overlap and the
other loses it, and nothing in the rule base records that the region is
contested. The confusion matrix knows -- it is exactly the off-diagonal mass --
but the rules do not.

The correction
--------------
Run the training data through the fitted model, read the top-n confusions off
the resulting matrix, and for each ``(predicted P, true T)`` pair add a *more
specific rule underneath the one that gets it wrong*::

    IF   rule P fires
    AND  x is [...] AND y is [...]        <- fit on the rows P gets wrong
    THEN T   (instead of P)

Three things make this different from re-fitting ``T``'s own rule:

- **It is gated.** ``w_sub = T(w_P, a_sub)`` is silent everywhere rule ``P`` is
  silent, so the sub-rule is an exception to ``P`` rather than a competing
  account of ``T``.
- **It is fit on the confusion.** The antecedent is fit on the rows ``P``
  claims but that are truly ``T`` -- so it describes *where P is wrong*, not
  where ``T`` generally lives. Restricted to that subset it can key on evidence
  far too weak to survive the global feature ranking, which is the whole reason
  a sub-rule can separate what the parent could not.
- **It is layered.** A sub-rule's consequent may itself be some other sub-rule's
  parent, so corrections chain: ``P -> T`` at layer 0 can feed ``T -> U`` at
  layer 1. Each layer is mined against the labels the previous layer produced.

Why the decision is by precedence
---------------------------------
Every t-norm obeys ``T(a, b) <= min(a, b)``, so ``w_sub <= w_P`` identically:
the gate that makes a sub-rule subordinate also caps its activation below the
rule it exists to correct. A flat argmax over firing strengths would render
every sub-rule inert. Resolution is therefore by *specificity* -- where the
gated activation clears the rule's threshold, the more specific rule takes the
label, and the parent's own firing strength is left untouched. This is the
ordinary reading of an exception in a rule base, and it is why
:func:`~tribblefis.gauss_math.apply_subdominant` acts on labels rather than on
strengths.

Honesty of the fit
------------------
Both the confusion matrix that selects the pairs and the scores that set each
rule's threshold come from *out-of-fold* predictions. An in-sample confusion
matrix understates the confusions a model will actually make, and a threshold
bisected against scores the sub-rule memorised is tuned to its own noise.

Relationship to the other two mechanisms
----------------------------------------
- :mod:`tribblefis.exclusion` repairs a rule that *over-claims* a region, by
  withdrawing it. This module repairs a region where the right answer needs
  evidence the parent rule never had, by adding one. They compose.
- :class:`~tribblefis.gaussian_classifier.MixtureOfGaussiansFuzzySequenceClassifier`
  reaches the same confusions with whole binary *models* consulted in
  :meth:`predict`. A sub-dominant rule is a rule in the rule base instead: it
  is inspectable alongside every other rule, it carries its own antecedents, and
  ``describe_subdominant`` prints it as the exception it is.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .gauss_data import (
    AnomalyParameters,
    GaussianMixtureModel,
    NormPair,
    SubdominantRule,
    resolve_norm_pair,
)
from .gauss_math import (
    apply_subdominant,
    create_gaussian_membership_dict,
    subdominant_activation,
    tsk_firing_strengths,
)

__all__ = [
    "mine_subdominant_rules",
    "confused_pairs",
    "validate_subdominant",
    "describe_subdominant",
    "apply_subdominant",
    "subdominant_activation",
]


def confused_pairs(
    predictions,
    y_true,
    top_n: int = 3,
    min_region: int = 20,
    min_confused: int = 8,
) -> list[tuple[Any, Any, int]]:
    """The ``(predicted P, true T, count)`` confusions worth a rule, worst first.

    Every off-diagonal cell of the confusion matrix is a candidate, not just the
    largest per row: a class can be confused with two others for different
    reasons, and each deserves its own sub-rule. Cells are ranked by raw count,
    since that is the number of rows a correct sub-rule would fix.

    Both directions of a confusion are kept. Unlike the sequence classifier's
    experts -- which are whole models re-deciding a pair and so would undo each
    other -- a sub-rule only ever moves rows *out of* its parent's region, and
    :func:`~tribblefis.gauss_math.apply_subdominant` forbids a row from
    returning to a label it has held. ``P -> T`` and ``T -> P`` therefore
    describe two genuinely different regions and cannot cycle.
    """
    predictions = np.asarray(predictions, dtype=object)
    y_true = np.asarray(y_true, dtype=object)

    pairs: list[tuple[Any, Any, int]] = []
    for parent in np.unique(predictions):
        region = predictions == parent
        if int(region.sum()) < min_region:
            continue
        truth_here = y_true[region]
        for corrected in np.unique(truth_here):
            if corrected == parent:
                continue
            count = int(np.sum(truth_here == corrected))
            if count >= min_confused:
                pairs.append((parent, corrected, count))

    pairs.sort(key=lambda row: (-row[2], str(row[0]), str(row[1])))
    return pairs[:top_n] if top_n and top_n > 0 else pairs


def _bisect_threshold(activation, is_corrected, n_steps: int = 60) -> tuple[float, float]:
    """The activation threshold maximising accuracy over the parent's region.

    Only rows the sub-rule could move respond to the threshold, and ordering
    them by activation makes region accuracy unimodal in it: admitting the most
    confident overrides first helps, until the overrides start being wrong.
    Golden-section search narrows ``[0, 1]`` onto that peak.

    Returns ``(threshold, accuracy)``. A threshold above every activation --
    firing on nothing -- is always available and is returned whenever the tuned
    one does not beat it, so a sub-rule can decline to act rather than do net
    harm on the region it owns.
    """
    activation = np.asarray(activation, dtype=float)
    is_corrected = np.asarray(is_corrected, dtype=bool)
    if activation.size == 0:
        return float("inf"), 0.0

    def region_accuracy(threshold):
        # Above the threshold the row becomes the corrected class, and is right
        # exactly when it really was; below, it stays the parent, and is right
        # exactly when it was not the corrected class.
        overridden = activation >= threshold
        return float(np.mean(np.where(overridden, is_corrected, ~is_corrected)))

    inert = float(np.nextafter(activation.max(), np.inf))
    baseline = region_accuracy(inert)

    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    lo, hi = 0.0, 1.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc, fd = region_accuracy(c), region_accuracy(d)
    for _ in range(n_steps):
        if hi - lo < 1e-6:
            break
        if fc < fd:
            lo, c, fc = c, d, fd
            d = lo + inv_phi * (hi - lo)
            fd = region_accuracy(d)
        else:
            hi, d, fd = d, c, fc
            c = hi - inv_phi * (hi - lo)
            fc = region_accuracy(c)

    best = 0.5 * (lo + hi)
    if region_accuracy(best) <= baseline:
        return inert, baseline
    return float(best), region_accuracy(best)


def _fit_antecedents(X_rows: pd.DataFrame, corrected, features, n_gaussians: int):
    """Fit a sub-rule's own terms on the rows the parent gets wrong."""
    if len(X_rows) < 2:
        return None
    sub_model = create_gaussian_membership_dict(
        X_rows.reset_index(drop=True),
        pd.Series([corrected] * len(X_rows)),
        top_n_var_names=list(features),
        n_gaussians=n_gaussians,
    )
    antecedents = {
        feature_name: list(feature_model.label_models[corrected].memberships)
        for feature_name, feature_model in sub_model.feature_models.items()
        if corrected in feature_model.label_models
        and feature_model.label_models[corrected].memberships
    }
    return antecedents or None


def _region_activation_out_of_fold(
    X, y_values, region, corrected, parent_strength, feature_arrays,
    norms, n_gaussians, cv, random_state,
):
    """Activation over the parent's region, from sub-rules that never saw the row.

    The threshold is the one number that decides how much of a region the
    override takes, and a sub-rule fit on ~dozens of confused rows will fit them
    closely. Bisecting against its in-sample activation therefore tunes the
    threshold to memorised rows and picks one far too permissive for rows the
    rule has not seen. Splitting the region into folds -- fitting the antecedent
    on the confused rows of each training part and scoring the held-out part --
    gives the bisection scores the deployed rule will actually produce.

    Returns ``None`` when the region cannot be split, so the caller can fall back
    to the in-sample scores and say so.
    """
    region_index = np.flatnonzero(region)
    is_corrected = y_values[region_index] == corrected
    n_splits = min(int(cv), int(is_corrected.sum()), int((~is_corrected).sum()))
    if n_splits < 2:
        return None

    activation = np.zeros(len(region_index), dtype=float)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_pos, test_pos in folds.split(region_index, is_corrected):
        confused_rows = region_index[train_pos][is_corrected[train_pos]]
        antecedents = _fit_antecedents(
            X.iloc[confused_rows], corrected, feature_arrays, n_gaussians
        )
        if antecedents is None:
            return None
        test_rows = region_index[test_pos]
        fold_arrays = {name: values[test_rows] for name, values in feature_arrays.items()}
        scored = subdominant_activation(
            SubdominantRule(parent=None, consequent=corrected, antecedents=antecedents),
            parent_strength[test_rows], fold_arrays, norms,
        )
        if scored is None:
            return None
        activation[test_pos] = scored
    return activation


def _out_of_fold_predictions(estimator, X, y, cv: int, random_state: int) -> np.ndarray:
    """Out-of-fold labels, for an honest confusion matrix.

    In-sample predictions understate the confusions the model will actually
    make -- it has already seen the rows it is being scored on -- so the pairs
    they suggest are the pairs that survived memorisation, not the ones a
    sub-rule needs to fix. Falls back to in-sample when the data cannot be
    stratified into ``cv`` folds.

    ``estimator`` is expected to be unfitted and to have the sub-dominant stage
    disabled -- it stands in for the rule base being corrected, not for the
    corrected result.
    """
    counts = y.value_counts()
    n_splits = min(int(cv), int(counts.min())) if len(counts) else 0
    if n_splits >= 2:
        try:
            folds = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            return np.asarray(
                cross_val_predict(clone(estimator), X, y, cv=folds), dtype=object
            )
        except Exception:
            pass
    in_sample = clone(estimator).fit(X, y)
    return np.asarray(in_sample.predict(X), dtype=object)


def mine_subdominant_rules(
    model: GaussianMixtureModel,
    X: pd.DataFrame,
    y,
    *,
    estimator=None,
    norms: NormPair | None = None,
    anomaly_details: AnomalyParameters | None = None,
    top_n: int = 3,
    max_layers: int = 2,
    min_region: int = 20,
    min_confused: int = 8,
    n_gaussians: int = 1,
    cv: int = 3,
    random_state: int = 42,
    min_region_gain: float = 0.03,
    honest_threshold: bool = True,
) -> tuple[list[SubdominantRule], dict[str, Any]]:
    """Mine a layered cascade of sub-dominant rules from the model's confusions.

    One layer at a time: read the confusion matrix of the labels as they
    currently stand, take the top ``top_n`` confusions, fit a rule on each
    confusion's rows, tune its threshold, and keep it if it improves accuracy
    over the region it governs. Then re-predict with the layer applied and
    repeat, so layer 1 sees the confusions layer 0 left behind rather than the
    ones it already fixed.

    Args:
        model: A fitted model. Mine after refinement and after any exclusion
            clauses -- both change which rows the confusion matrix contains.
        X: Training features.
        y: Training labels, aligned with ``X``.
        estimator: A fitted, cloneable estimator wrapping ``model``, used for the
            out-of-fold confusion matrix. When ``None``, in-sample predictions
            are used and ``diagnostics["out_of_fold"]`` records that -- the
            result is still valid, just optimistic about which pairs matter.
        norms: Resolved operator pair; defaults to ``anomaly_details``' pair or
            the library default. Use the pair the model predicts with.
        anomaly_details: Passed through to the firing-strength call.
        top_n: Confusions to address per layer, largest off-diagonal cell first.
        max_layers: Depth of the cascade. ``1`` disables chaining.
        min_region: A parent's region must have at least this many rows before
            it is eligible -- a rule mined on a handful of rows describes noise.
        min_confused: Minimum rows in a ``(P, T)`` cell before it earns a rule.
        n_gaussians: Memberships per feature for a sub-rule's own antecedent.
            ``1`` by default: the rule is fit on one confusion region, which is
            usually unimodal, and more terms mostly buys variance.
        cv: Folds for the out-of-fold confusion matrix and threshold scores.
        min_region_gain: A tuned sub-rule is kept only when it improves accuracy
            over its parent's region by more than this. The threshold search can
            always choose to fire on nothing, so ``0.0`` would already reject a
            rule that cannot help at all -- but measured over 36 held-out cases,
            every rule kept on a *marginal* gain was a coin flip (7 better, 5
            worse). The default of 0.03 asks for a region gain large enough not
            to be noise: it removes every one of those losses, and unlike a
            stricter floor it still admits the deeper layers of a chained
            correction, whose later rules necessarily govern smaller gains than
            the first. See docs/subdominant-rule-evaluation.md.
        honest_threshold: Bisect each threshold against activations from folds
            that never saw the row. A sub-rule fit on dozens of confused rows
            fits them closely, so its in-sample activation is far higher on those
            rows than on new ones, and a threshold tuned against it is too
            permissive on everything else. Falls back to in-sample scores when a
            region cannot be split, recording which pairs in
            ``diagnostics["layers"][i]["in_sample_thresholds"]``.

    Returns:
        ``(rules, diagnostics)``. ``diagnostics`` records the pairs considered
        and rejected per layer, whether the confusion matrix was out-of-fold,
        and the region accuracy each kept rule bought, so an empty cascade can
        be explained rather than guessed at.
    """
    if norms is None:
        norms = anomaly_details.norms() if anomaly_details else resolve_norm_pair()

    X = X.reset_index(drop=True)
    y_series = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y))
    y_series = y_series.reset_index(drop=True)
    y_values = np.asarray(y_series.values, dtype=object)

    feature_arrays = {
        name: np.asarray(X[name].values)
        for name in model.feature_models
        if name in X.columns
    }
    firing_strengths, labels = tsk_firing_strengths(
        X, model, anomaly_details=anomaly_details, norms=norms,
        feature_arrays=feature_arrays,
    )
    anomaly_label = (
        anomaly_details.label if anomaly_details and anomaly_details.include_anomaly else None
    )
    class_labels = [label for label in labels if label != anomaly_label]
    class_block = firing_strengths[:, : len(class_labels)]
    column_of = {label: i for i, label in enumerate(class_labels)}

    if estimator is not None:
        current = _out_of_fold_predictions(estimator, X, y_series, cv, random_state)
    else:
        current = np.asarray(
            [class_labels[i] for i in np.argmax(class_block, axis=1)], dtype=object
        )

    diagnostics: dict[str, Any] = {
        "out_of_fold": estimator is not None,
        "layers": [],
        "n_rules": 0,
    }
    rules: list[SubdominantRule] = []

    for layer in range(max(1, int(max_layers))):
        pairs = confused_pairs(
            current, y_values, top_n=top_n,
            min_region=min_region, min_confused=min_confused,
        )
        layer_info: dict[str, Any] = {
            "layer": layer,
            "pairs": [(parent, corrected, count) for parent, corrected, count in pairs],
            "kept": [],
            "rejected": {},
        }
        if not pairs:
            diagnostics["layers"].append(layer_info)
            break

        layer_rules: list[SubdominantRule] = []
        for parent, corrected, _ in pairs:
            region = current == parent
            confused = region & (y_values == corrected)
            if int(confused.sum()) < min_confused:
                layer_info["rejected"][f"{parent}->{corrected}"] = "too few confused rows"
                continue

            # Fit the sub-rule's antecedent on the confused rows *only*: the rows
            # the parent claims and gets wrong. That is what makes it a
            # description of the parent's mistake rather than of the corrected
            # class in general.
            antecedents = _fit_antecedents(
                X[confused], corrected, feature_arrays, n_gaussians
            )
            if antecedents is None:
                layer_info["rejected"][f"{parent}->{corrected}"] = "no antecedent fit"
                continue

            candidate = SubdominantRule(
                parent=parent, consequent=corrected, antecedents=antecedents,
                threshold=0.0, layer=layer,
            )
            parent_strength = class_block[:, column_of[parent]]
            activation = subdominant_activation(
                candidate, parent_strength, feature_arrays, norms
            )
            if activation is None:
                layer_info["rejected"][f"{parent}->{corrected}"] = "rule does not evaluate"
                continue

            # Tune and score over the parent's region only -- the rows this rule
            # will actually be consulted for -- against activations from folds
            # that never saw the row, so the threshold is not set on rows the
            # antecedent has memorised.
            is_corrected = y_values[region] == corrected
            scores = None
            if honest_threshold:
                scores = _region_activation_out_of_fold(
                    X, y_values, region, corrected, parent_strength,
                    feature_arrays, norms, n_gaussians, cv, random_state,
                )
            if scores is None:
                scores = activation[region]
                layer_info.setdefault("in_sample_thresholds", []).append(
                    f"{parent}->{corrected}"
                )
            threshold, accuracy = _bisect_threshold(scores, is_corrected)
            baseline = float(np.mean(~is_corrected))

            if accuracy <= baseline + min_region_gain:
                layer_info["rejected"][f"{parent}->{corrected}"] = (
                    f"no region gain ({accuracy:.4f} vs {baseline:.4f})"
                )
                continue

            kept = candidate._replace(
                threshold=threshold,
                support=int(confused.sum()),
                purity=float(np.mean(is_corrected)),
            )
            layer_rules.append(kept)
            layer_info["kept"].append(
                {
                    "pair": (parent, corrected),
                    "threshold": threshold,
                    "region_accuracy": accuracy,
                    "region_baseline": baseline,
                    "support": kept.support,
                }
            )

        diagnostics["layers"].append(layer_info)
        if not layer_rules:
            break

        rules.extend(layer_rules)
        # Re-predict with everything mined so far, so the next layer reads the
        # confusions this one *left*, not the ones it fixed.
        current = apply_subdominant(
            current, class_block, class_labels, model, feature_arrays, norms, rules
        )

    diagnostics["n_rules"] = len(rules)
    return rules, diagnostics


def validate_subdominant(
    model: GaussianMixtureModel,
    rules: Iterable[SubdominantRule] | None = None,
) -> list[str]:
    """Reasons any sub-rule fails to address ``model``, one message per bad rule.

    A sub-rule carries its own memberships, so it cannot go stale the way an
    exclusion clause can -- but it still names a parent label and a consequent
    that must exist in the rule base, and features the model actually carries.
    An empty list means every rule resolves.
    """
    rules = model.subdominant if rules is None else rules
    known_labels = {
        label
        for feature_model in model.feature_models.values()
        for label in feature_model.label_models
    }

    problems: list[str] = []
    for rule in rules:
        if rule.parent not in known_labels:
            problems.append(f"{rule.parent}->{rule.consequent}: no rule for parent {rule.parent!r}")
            continue
        if rule.consequent not in known_labels:
            problems.append(
                f"{rule.parent}->{rule.consequent}: consequent {rule.consequent!r} is not a class"
            )
            continue
        if rule.parent == rule.consequent:
            problems.append(f"{rule.parent}->{rule.consequent}: parent and consequent are the same")
            continue
        if not rule.antecedents:
            problems.append(f"{rule.parent}->{rule.consequent}: rule has no antecedent")
            continue
        missing = [
            feature_name
            for feature_name in rule.antecedents
            if feature_name not in model.feature_models
        ]
        if missing:
            problems.append(
                f"{rule.parent}->{rule.consequent}: unknown feature(s) "
                f"{', '.join(map(repr, missing))}"
            )
    return problems


def describe_subdominant(
    model: GaussianMixtureModel,
    rules: Iterable[SubdominantRule] | None = None,
) -> str:
    """The cascade as readable gated exceptions, grouped by layer.

    Each rule prints as what it is -- a condition on the parent firing, its own
    antecedent, and the class it corrects to -- with the confusion count that
    justified it and the threshold its activation must clear.
    """
    rules = list(model.subdominant if rules is None else rules)
    if not rules:
        return "no sub-dominant rules"

    lines: list[str] = []
    for layer in sorted({rule.layer for rule in rules}):
        lines.append(f"LAYER {layer}:")
        for rule in [r for r in rules if r.layer == layer]:
            lines.append(f"  IF   rule {rule.parent} fires")
            for feature_name, memberships in rule.antecedents.items():
                terms = ", ".join(_membership_text(mf) for mf in memberships)
                lines.append(f"  AND  {feature_name} is [{terms}]")
            lines.append(
                f"  THEN {rule.consequent}   -- instead of {rule.parent}; "
                f"activation >= {rule.threshold:.3f}, n={rule.support}, "
                f"{rule.purity:.0%} of region {rule.parent} really {rule.consequent}"
            )
    return "\n".join(lines)


def _membership_text(mf) -> str:
    mu = getattr(mf, "mu", None)
    if mu is not None:
        return f"~{mu:.3g}+/-{getattr(mf, 'sigma', 0.0):.3g}"
    return type(mf).__name__
