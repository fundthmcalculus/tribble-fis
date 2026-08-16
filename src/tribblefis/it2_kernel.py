"""Interval Type-2 FIS kernel with Karnik-Mendel type reduction.

Two distinct type-reduction problems live here, and they are not the same
computation:

1. **Per-rule interval reduction** (`karnik_mendel_type_reduction`): each rule's
   own upper/lower firing strength, ``[f_lower, f_upper]``, collapsed to one
   crisp number *independently* of every other rule. This is what a classifier
   needs -- each class's own score, for `argmax`. For a single interval with a
   uniform secondary membership function (the standard "interval type-2" set
   this package builds), the centroid of that interval is provably its
   midpoint -- there is no switch-point search to run, because there is only
   one point mass to place, not several to sort and re-weight. See the
   function's docstring for the short proof.

2. **Cross-rule Karnik-Mendel** (`karnik_mendel_tsk`): a TSK model's crisp
   output is a weighted average of *every* rule's own consequent value,
   weighted by that rule's firing strength. Type-2 uncertainty puts each
   weight in an interval ``[f_lower_i, f_upper_i]`` rather than pinning it to
   one number, so the output is a *range* of possible weighted averages --
   and finding its endpoints is the actual textbook Karnik-Mendel problem
   (sort the rules by consequent value, then search for the "switch point"
   separating which rules get their lower vs. upper weight). This is the
   computation `it2_regressor.py` needs, and the one for which "type
   reduction" traditionally means something nontrivial.

Both were previously conflated: `it2_regressor.py` ran (1) on each rule's
firing strength alone, then fed the result into the same weighted-consequent
evaluation Type-1 uses -- never giving the search in (2) the one thing it
needs to do anything (each rule's own consequent value). See `it2_regressor.py`
for how it now calls `karnik_mendel_tsk` directly instead.
"""

import numpy as np
import pandas as pd
from numba import njit, prange

from .gauss_data import (
    IT2GaussianMixtureModel,
    IT2GaussianMembership,
    IT2TrapezoidMembership,
    IT2TriangularMembership,
    NormPair,
    ZERO_FIRING_THRESHOLD,
)
from .gauss_math import tsk_firing_strengths, GaussianMixtureModel, FeatureModel, LabelModel


def it2_firing_strengths(
    X: pd.DataFrame,
    model: IT2GaussianMixtureModel,
    norms: NormPair,
    km_iterations: int | None = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Compute IT2 firing strengths with *per-rule* type reduction.

    Computes both upper and lower membership function evaluations, then
    collapses each rule's own interval independently (see module docstring,
    case 1). This is what the classifier uses directly (`argmax` over
    `firing_crisp`'s columns) and what the regressor uses for its firing
    *bounds* before running the cross-rule search in `karnik_mendel_tsk`.

    Args:
        X: Input feature matrix (n_samples, n_features)
        model: IT2GaussianMixtureModel
        norms: (t_norm, t_conorm) pair for fuzzy operations
        km_iterations: accepted for backward compatibility; any truthy value
            selects `karnik_mendel_type_reduction` (mathematically identical to
            the `None`/averaging branch for this per-rule problem -- see that
            function's docstring), so this only affects which code path runs,
            never the result.

    Returns:
        firing_upper: (n_samples, n_labels) upper bound firing strengths
        firing_lower: (n_samples, n_labels) lower bound firing strengths
        firing_crisp: (n_samples, n_labels) type-reduced crisp outputs
        labels: list of output labels
    """
    labels = sorted(model.all_output_labels)

    # Extract upper and lower Type-1 models from IT2 model
    upper_model = _extract_upper_model(model)
    lower_model = _extract_lower_model(model)

    # Compute firing strengths for both upper and lower
    firing_upper = tsk_firing_strengths(X, upper_model, norms=norms)[0]
    firing_lower = tsk_firing_strengths(X, lower_model, norms=norms)[0]

    if km_iterations is None or km_iterations == 0:
        firing_crisp = 0.5 * (firing_upper + firing_lower)
    else:
        firing_crisp = karnik_mendel_type_reduction(
            firing_upper, firing_lower, max_iterations=km_iterations
        )

    return firing_upper, firing_lower, firing_crisp, labels


def _extract_upper_model(model: IT2GaussianMixtureModel) -> GaussianMixtureModel:
    """Extract the upper bound Type-1 model from an IT2 model."""
    feature_models = {}
    for feature_name, it2_feature_model in model.feature_models.items():
        label_models = {}
        for label, it2_label_model in it2_feature_model.label_models.items():
            # Extract upper MFs from all IT2 memberships (works for any IT2 type)
            upper_mfs = [mf.upper_mf for mf in it2_label_model.memberships]
            label_models[label] = LabelModel(upper_mfs)
        feature_models[feature_name] = FeatureModel(label_models)

    # Use the same anomaly_params as the original model
    anomaly_params = getattr(model, 'anomaly_params', None)
    return GaussianMixtureModel(feature_models, anomaly_params=anomaly_params)


def _extract_lower_model(model: IT2GaussianMixtureModel) -> GaussianMixtureModel:
    """Extract the lower bound Type-1 model from an IT2 model."""
    feature_models = {}
    for feature_name, it2_feature_model in model.feature_models.items():
        label_models = {}
        for label, it2_label_model in it2_feature_model.label_models.items():
            # Extract lower MFs from all IT2 memberships (works for any IT2 type)
            lower_mfs = [mf.lower_mf for mf in it2_label_model.memberships]
            label_models[label] = LabelModel(lower_mfs)
        feature_models[feature_name] = FeatureModel(label_models)

    # Use the same anomaly_params as the original model
    anomaly_params = getattr(model, 'anomaly_params', None)
    return GaussianMixtureModel(feature_models, anomaly_params=anomaly_params)


def karnik_mendel_type_reduction(
    firing_upper: np.ndarray,
    firing_lower: np.ndarray,
    max_iterations: int = 10,
) -> np.ndarray:
    """Collapse each rule's own ``[firing_lower, firing_upper]`` to a crisp value.

    ``max_iterations`` is accepted only for backward compatibility with the
    previous (iterative, and never actually convergent -- see below) signature;
    it has no effect on the result.

    **Why this is exactly the midpoint, not a search.** This package's interval
    type-2 sets use a *uniform* secondary membership function: every point in
    ``[firing_lower, firing_upper]`` is equally possible, nothing in between is
    preferred. The centroid (type reduction) of a uniform distribution over a
    single interval is that interval's midpoint by definition -- there is only
    one interval, so there is nothing to sort or search a switch point over
    (contrast `karnik_mendel_tsk`, which reduces *several* rules' worth of
    interval-weighted consequents at once and genuinely needs the search).

    The previous implementation ran a fixed-point iteration
    (``y_left, y_right <- avg(bound, midpoint)``) for `max_iterations` steps in
    a pure Python `for i in range(n_samples): for j in range(n_outputs)` loop.
    Algebraically that iteration's `(y_left + y_right) / 2` invariant is the
    midpoint from its very first step onward regardless of how many more steps
    run (substitute the fixed point to check), so it always returned this same
    number -- just after up to 10x redundant work, in Python, cell by cell.
    """
    return 0.5 * (firing_upper + firing_lower)


@njit(cache=True)
def _km_direction(y_sorted, fl_sorted, fu_sorted, want_max, max_iterations, tol):
    """One direction (min or max) of the Karnik-Mendel switch-point search for a
    single sample's already-sorted rule values.

    Standard KM iteration (Karnik & Mendel, 2001 / Mendel, 2001): start from
    the firing-strength midpoint weights, find the index where the running
    weighted average crosses the sorted consequent values (the "switch
    point"), re-weight rules on either side of it with the extreme weight that
    pushes the average further toward the direction being searched, and
    repeat until the switch point stops moving. `want_max=False` searches the
    left (minimizing) endpoint; `want_max=True` searches the right
    (maximizing) endpoint.
    """
    n = y_sorted.shape[0]
    if n == 1:
        return y_sorted[0]

    num = 0.0
    den = 0.0
    for i in range(n):
        f = 0.5 * (fl_sorted[i] + fu_sorted[i])
        num += f * y_sorted[i]
        den += f
    y_est = num / den if den > 1e-12 else y_sorted[n // 2]

    for _ in range(max_iterations):
        # Largest k with y_sorted[k] <= y_est (k in [0, n-1]).
        k = 0
        for i in range(n):
            if y_sorted[i] <= y_est:
                k = i
            else:
                break

        num = 0.0
        den = 0.0
        if want_max:
            for i in range(n):
                f = fl_sorted[i] if i <= k else fu_sorted[i]
                num += f * y_sorted[i]
                den += f
        else:
            for i in range(n):
                f = fu_sorted[i] if i <= k else fl_sorted[i]
                num += f * y_sorted[i]
                den += f
        y_new = num / den if den > 1e-12 else y_est

        if abs(y_new - y_est) < tol:
            y_est = y_new
            break
        y_est = y_new

    return y_est


@njit(parallel=True, cache=True)
def _karnik_mendel_batch(y_rule, f_lower, f_upper, max_iterations, tol):
    n_samples, n_rules = y_rule.shape
    y_l = np.empty(n_samples)
    y_r = np.empty(n_samples)
    for s in prange(n_samples):
        row_y = y_rule[s]
        row_fl = f_lower[s]
        row_fu = f_upper[s]

        total_fu = 0.0
        for i in range(n_rules):
            total_fu += row_fu[i]
        if total_fu <= ZERO_FIRING_THRESHOLD:
            y_l[s] = 0.0
            y_r[s] = 0.0
            continue

        order = np.argsort(row_y)
        y_sorted = row_y[order]
        fl_sorted = row_fl[order]
        fu_sorted = row_fu[order]

        y_l[s] = _km_direction(y_sorted, fl_sorted, fu_sorted, False, max_iterations, tol)
        y_r[s] = _km_direction(y_sorted, fl_sorted, fu_sorted, True, max_iterations, tol)

    return y_l, y_r


def karnik_mendel_tsk(
    rule_values: np.ndarray,
    firing_lower: np.ndarray,
    firing_upper: np.ndarray,
    max_iterations: int = 50,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Karnik-Mendel type reduction of a type-2 TSK output.

    A type-2 TSK model's crisp output would (if every weight were pinned to a
    single number) be ``sum(f_i * y_i) / sum(f_i)`` across rules ``i``, exactly
    what `regression.apply_tsk_consequents` computes for Type-1. Interval
    type-2 makes each weight an interval, ``f_i in [firing_lower_i,
    firing_upper_i]``, so the output is the *range* of that weighted average
    over every admissible combination of weights -- and the range's endpoints
    are exactly `y_l` (minimum) and `y_r` (maximum) returned here. The
    type-reduced crisp estimate is their midpoint, ``0.5 * (y_l + y_r)``,
    which by construction always lies inside ``[y_l, y_r]``.

    Args:
        rule_values: (n_samples, n_rules) each rule's own crisp consequent
            output per sample (`regression.rule_consequent_values`).
        firing_lower, firing_upper: (n_samples, n_rules) interval firing-
            strength bounds per rule (raw, i.e. *not* row-normalized -- KM's
            weighted average normalizes internally).
        max_iterations: cap on switch-point refinements per direction. The
            search provably terminates in at most `n_rules` steps (each step
            either moves the switch point or converges); with the handful of
            rules typical of this package's output-bucket models this default
            is far more headroom than ever needed.
        tol: convergence tolerance on the weighted-average estimate between
            successive switch-point refinements.

    Returns:
        (y_l, y_r): (n_samples,) arrays, the type-reduced output interval.
        Rows where no rule fires at all (``firing_upper`` row-sum
        ``<= ZERO_FIRING_THRESHOLD``) return ``(0, 0)`` -- the exact same
        gate `regression._normalize_firing_strengths` uses, via the shared
        `gauss_data.ZERO_FIRING_THRESHOLD` constant. This must stay a shared
        constant, not two independently-chosen thresholds: a row whose firing
        sum falls between two different gates gets a real computed answer
        from one code path and a hard zero from the other, for the same
        input, which is exactly the failure this once had (see that
        constant's own docstring).

    The per-sample switch-point search is inherently sequential (each
    iteration depends on the previous one's switch point) and was previously
    absent entirely -- `it2_kernel` had no cross-rule reduction at all (see
    module docstring). `_karnik_mendel_batch` compiles that search with numba
    and parallelizes it across samples (`prange`), rather than the naive
    `for i in range(n_samples): for j in ...` Python loop this package's only
    prior (per-rule, not cross-rule) type reduction used.
    """
    rule_values = np.ascontiguousarray(rule_values, dtype=np.float64)
    firing_lower = np.ascontiguousarray(firing_lower, dtype=np.float64)
    firing_upper = np.ascontiguousarray(firing_upper, dtype=np.float64)
    return _karnik_mendel_batch(rule_values, firing_lower, firing_upper, max_iterations, tol)
