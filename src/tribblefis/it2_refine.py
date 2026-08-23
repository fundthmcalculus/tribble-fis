"""Post-fit antecedent refinement for Interval Type-2 FIS models.

This is the IT2 counterpart of `refine.py`'s block coordinate descent for
Type-1: cycle through one IT2 membership at a time (any of Gaussian,
trapezoid, triangular -- see `_slot_x0_and_bounds`/`_apply_slot_params`) and
run a small bounded local solve on just its shared peak plus two independent
spread gaps, with every other parameter in the model held fixed, repeating
for a few sweeps. `refine.py`'s classifier section explains why this applies
unchanged here: *"A zeroth-order TSK classifier has no consequents... so
refining the antecedents is the whole model"* -- exactly the IT2 classifier's
situation, since `it2_classifier.predict` reads a class score directly off
`firing_crisp` with no consequent evaluation downstream. So each sub-problem's
fitness is the cross-entropy of the type-reduced, row-normalized firing
strengths against the training labels, and there is no closed-form inner solve
to run before it (contrast Type-1's coordinate descent, whose sub-problem
re-solves TSK consequents in closed form -- IT2 regression has no analogue of
that here; see the module-level TODO below).

Each sub-problem is optimized with `refine.py`'s `_optimizers_sub_solve` --
bounded local descent via the in-house `optimizers` package (#119), which
estimates its own gradient by finite differences under the hood (the
switch-point search inside `karnik_mendel_tsk` is not differentiable in
closed form -- it is a discrete arg-sort-and-partition, not a smooth function
of `(mu, sigma)` -- so an analytic gradient the way `refine.py`'s
"probability"-norm block coordinate descent has one is not available here).
An L2 penalty pulls each sub-problem back toward its value at the start of
the current sweep, and -- mirroring `refine.py`'s "never return a model worse
than the heuristic start" guarantee -- a candidate is only kept if it
strictly improves the best training loss seen so far.

**Regression refinement** (`refine_it2_regressor_antecedents`) is the same
coordinate descent, but unlike the classifier, `it2_regressor`'s consequents
are not something the fitness can ignore: they were solved once, at
conversion time, from the base Type-1 fit's antecedents, so evaluating a
*candidate* set of antecedents against those same fixed consequents scores a
mismatched pair -- consequents fit for a different footprint of uncertainty
than the one being tried. Every fitness evaluation therefore re-solves the
consequents in closed form for the candidate antecedents first (the
"consequent solving" this module now does for regression), the same
`Phi^T Phi` ridge normal-equations solve `regression.solve_tsk_consequents`
uses, before scoring held-out MSE through the full Karnik-Mendel prediction
path (`it2_kernel.karnik_mendel_tsk`) -- mirroring the bilevel structure of
`refine.py`'s Type-1 regressor coordinate descent (antecedents outer, LSE-fit
consequents inner), just with a finite-difference-driven search on the outer
loop rather than the Type-1 path's analytic gradient (KM's switch-point
search isn't differentiable in closed form -- see above).
The consequent re-solve uses each rule's *midpoint* firing strength,
``0.5 * (firing_upper + firing_lower)``, as its ridge design weight: this is
the natural "Type-1-equivalent" firing-strength matrix for a candidate whose
uncertainty is otherwise expressed only as an interval, and it collapses to
the base Type-1 model's own firing strengths exactly when the footprint of
uncertainty vanishes.
"""

import numpy as np
import pandas as pd

from .gauss_data import (
    IT2GaussianMixtureModel, IT2FeatureModel, IT2LabelModel,
    IT2GaussianMembership, IT2TrapezoidMembership, IT2TriangularMembership,
    GaussianMembership, TrapezoidMembership, TriangularMembership, NormPair,
)
from .gauss_math import tsk_firing_strengths
from .it2_kernel import it2_firing_strengths, karnik_mendel_tsk, _extract_upper_model, _extract_lower_model
from .optimizer_utils import optimizers_sub_solve as _optimizers_sub_solve
from .refine import _make_folds, _prepare_folds
from .regression import solve_tsk_consequents_from_firing, rule_consequent_values, _mse


def _iter_it2_slots(model: IT2GaussianMixtureModel):
    """Yield ``(feature_name, label, mf_index, IT2AnyMembership)`` for every
    IT2 membership (both the upper and lower half together, any of
    Gaussian/trapezoid/triangular) in a deterministic order.

    **Why one slot per membership, not one per half.** An earlier version of
    this module optimized `upper_mf` and `lower_mf` as fully independent
    parameter slots. Nothing then stopped the search from moving them past
    each other -- e.g. widening `lower_mf` past `upper_mf` -- which breaks
    the one invariant every caller of this module's output relies on:
    `firing_lower <= firing_upper` pointwise (asserted by
    `it2_kernel.it2_firing_strengths`'s callers and, concretely, required by
    `karnik_mendel_tsk`, which can return `y_l > y_r` -- observed on a real
    fit -- if fed a firing interval that's inverted). The fix keeps each
    membership's "peak" (Gaussian's `mu`, trapezoid's flat top `[b, c]`,
    triangular's apex `b`) *shared* between the two halves and searches the
    peak plus two independent non-negative "extra spread" gaps per slot (see
    `_apply_slot_params`) -- for two memberships sharing a peak, the wider
    one dominates the narrower one at every point by construction, which is
    exactly the "footprint of uncertainty" shape `it2_classifier`/
    `it2_regressor`'s own conversion already builds new memberships in
    (`gauss_data.widen_membership`) at `fit()` time.
    """
    for fname, fmodel in model.feature_models.items():
        for label, lmodel in fmodel.label_models.items():
            for idx, it2_mf in enumerate(lmodel.memberships):
                yield fname, label, idx, it2_mf


def _slot_x0_and_bounds(it2_mf, lo: float, hi: float, rng: float, sigma_min_frac: float):
    """Initial parameter vector and box bounds for one slot, dispatched by
    `it2_mf`'s membership type. Every type's vector starts with its shared
    peak parameter(s), followed by non-negative "extra spread" gaps on each
    side so the optimizer's box bounds alone enforce
    `firing_lower <= firing_upper` (see `_apply_slot_params`)."""
    if isinstance(it2_mf, IT2GaussianMembership):
        return _gaussian_slot_x0_and_bounds(it2_mf, lo, hi, rng, sigma_min_frac)
    if isinstance(it2_mf, IT2TrapezoidMembership):
        return _trapezoid_slot_x0_and_bounds(it2_mf, lo, hi, rng, sigma_min_frac)
    if isinstance(it2_mf, IT2TriangularMembership):
        return _triangular_slot_x0_and_bounds(it2_mf, lo, hi, rng, sigma_min_frac)
    raise TypeError(f"Unsupported IT2 membership type for refinement: {type(it2_mf)!r}")


def _gaussian_slot_x0_and_bounds(it2_mf, lo, hi, rng, sigma_min_frac):
    """`(mu, sigma_lower, sigma_upper)`. `mu` starts at the upper half's
    center (the two halves share one `mu` at conversion time; if a caller
    ever hands in a model where they've diverged, the upper half's is kept
    as the anchor since it is the one that determines the footprint's outer
    edge)."""
    mu0 = it2_mf.upper_mf.mu
    sigma_lower0 = min(it2_mf.lower_mf.sigma, it2_mf.upper_mf.sigma)
    sigma_upper0 = max(it2_mf.lower_mf.sigma, it2_mf.upper_mf.sigma)
    x0 = np.array([mu0, sigma_lower0, sigma_upper0])
    sigma_lo = sigma_min_frac * rng
    bounds = [(lo, hi), (sigma_lo, rng), (sigma_lo, 2.0 * rng)]
    return x0, bounds


def _side_widths(peak: float, lower_edge: float, upper_edge: float, min_width: float):
    """``(lower_width, extra_width)`` for one side of one slot: the lower
    half's own half-width from `peak`, and how much wider the upper half's
    own half-width is beyond it (clamped to >= 0 -- if the input model
    happens to have them inverted, the extra collapses to 0 rather than
    going negative, mirroring `_gaussian_slot_x0_and_bounds`'s min/max
    anchoring)."""
    lower_width = max(abs(peak - lower_edge), min_width)
    upper_width = max(abs(peak - upper_edge), min_width)
    return lower_width, max(upper_width - lower_width, 0.0)


def _trapezoid_slot_x0_and_bounds(it2_mf, lo, hi, rng, sigma_min_frac):
    """`(b, gap_bc, left_lower, extra_left, right_lower, extra_right)`.

    `b`/`c` (the flat top) start at the upper half's own values, mirroring
    `_gaussian_slot_x0_and_bounds`'s `mu` anchoring. The four remaining
    parameters are non-negative gaps (see `_apply_trapezoid_slot_params`):
    reconstructing `a`/`d` from them makes `a_upper <= a_lower <= b <= c <=
    d_lower <= d_upper` -- and therefore `firing_lower <= firing_upper` --
    true for *any* point in these box bounds, not just the optimum found.
    """
    min_width = sigma_min_frac * rng
    b0, c0 = it2_mf.upper_mf.b, it2_mf.upper_mf.c
    left_lower0, extra_left0 = _side_widths(b0, it2_mf.lower_mf.a, it2_mf.upper_mf.a, min_width)
    right_lower0, extra_right0 = _side_widths(c0, it2_mf.lower_mf.d, it2_mf.upper_mf.d, min_width)
    x0 = np.array([b0, max(c0 - b0, 0.0), left_lower0, extra_left0, right_lower0, extra_right0])
    bounds = [(lo, hi), (0.0, rng), (min_width, rng), (0.0, rng), (min_width, rng), (0.0, rng)]
    return x0, bounds


def _triangular_slot_x0_and_bounds(it2_mf, lo, hi, rng, sigma_min_frac):
    """`(b, left_lower, extra_left, right_lower, extra_right)` -- the
    triangular analogue of `_trapezoid_slot_x0_and_bounds`, one parameter
    fewer since the apex `b` is a single point rather than an interval."""
    min_width = sigma_min_frac * rng
    b0 = it2_mf.upper_mf.b
    left_lower0, extra_left0 = _side_widths(b0, it2_mf.lower_mf.a, it2_mf.upper_mf.a, min_width)
    right_lower0, extra_right0 = _side_widths(b0, it2_mf.lower_mf.c, it2_mf.upper_mf.c, min_width)
    x0 = np.array([b0, left_lower0, extra_left0, right_lower0, extra_right0])
    bounds = [(lo, hi), (min_width, rng), (0.0, rng), (min_width, rng), (0.0, rng)]
    return x0, bounds


def _apply_slot_params(it2_mf, v: np.ndarray):
    """Build a fresh IT2 membership of `it2_mf`'s own type from `v`, dispatched
    by type. Every reconstruction enforces `firing_lower <= firing_upper` by
    construction (see each type's own docstring), not just by clamping after
    the fact -- the invariant `_iter_it2_slots` documents cannot be violated
    regardless of what the optimizer proposes, since the box bounds passed to
    it (`_slot_x0_and_bounds`) only ever contain valid points.
    """
    if isinstance(it2_mf, IT2GaussianMembership):
        return _apply_gaussian_slot_params(v, it2_mf.upper_mf.id, it2_mf.lower_mf.id)
    if isinstance(it2_mf, IT2TrapezoidMembership):
        return _apply_trapezoid_slot_params(v, it2_mf.upper_mf.id, it2_mf.lower_mf.id)
    if isinstance(it2_mf, IT2TriangularMembership):
        return _apply_triangular_slot_params(v, it2_mf.upper_mf.id, it2_mf.lower_mf.id)
    raise TypeError(f"Unsupported IT2 membership type for refinement: {type(it2_mf)!r}")


def _apply_gaussian_slot_params(v: np.ndarray, upper_id, lower_id) -> IT2GaussianMembership:
    """Build a fresh `IT2GaussianMembership` from `(mu, sigma_lower, sigma_upper)`,
    clamping `sigma_upper >= sigma_lower` so the invariant `_iter_it2_slots`
    documents cannot be violated regardless of what the optimizer proposes.
    """
    mu = float(v[0])
    sigma_lower = max(float(v[1]), 1e-6)
    sigma_upper = max(float(v[2]), sigma_lower)
    return IT2GaussianMembership(
        upper_mf=GaussianMembership(mu=mu, sigma=sigma_upper, id=upper_id),
        lower_mf=GaussianMembership(mu=mu, sigma=sigma_lower, id=lower_id),
    )


def _apply_trapezoid_slot_params(v: np.ndarray, upper_id, lower_id) -> IT2TrapezoidMembership:
    """Build a fresh `IT2TrapezoidMembership` from `(b, gap_bc, left_lower,
    extra_left, right_lower, extra_right)`. `gap_bc`/`extra_left`/
    `extra_right` are clamped non-negative so `a_upper <= a_lower <= b <= c
    <= d_lower <= d_upper` holds regardless of what the optimizer proposes.
    """
    b = float(v[0])
    c = b + max(float(v[1]), 0.0)
    left_lower = max(float(v[2]), 1e-6)
    left_upper = left_lower + max(float(v[3]), 0.0)
    right_lower = max(float(v[4]), 1e-6)
    right_upper = right_lower + max(float(v[5]), 0.0)
    return IT2TrapezoidMembership(
        upper_mf=TrapezoidMembership(a=b - left_upper, b=b, c=c, d=c + right_upper, id=upper_id),
        lower_mf=TrapezoidMembership(a=b - left_lower, b=b, c=c, d=c + right_lower, id=lower_id),
    )


def _apply_triangular_slot_params(v: np.ndarray, upper_id, lower_id) -> IT2TriangularMembership:
    """Build a fresh `IT2TriangularMembership` from `(b, left_lower,
    extra_left, right_lower, extra_right)` -- the triangular analogue of
    `_apply_trapezoid_slot_params`, apex `b` shared instead of a flat top."""
    b = float(v[0])
    left_lower = max(float(v[1]), 1e-6)
    left_upper = left_lower + max(float(v[2]), 0.0)
    right_lower = max(float(v[3]), 1e-6)
    right_upper = right_lower + max(float(v[4]), 0.0)
    return IT2TriangularMembership(
        upper_mf=TriangularMembership(a=b - left_upper, b=b, c=b + right_upper, id=upper_id),
        lower_mf=TriangularMembership(a=b - left_lower, b=b, c=b + right_lower, id=lower_id),
    )


def _replace_slot(
    model: IT2GaussianMixtureModel,
    fname: str,
    label: int,
    idx: int,
    new_it2_mf,
) -> IT2GaussianMixtureModel:
    """Return a copy of `model` with one IT2 membership replaced.

    Every container here (`IT2GaussianMixtureModel`, `IT2FeatureModel`,
    `IT2LabelModel`, and any `IT2AnyMembership`) is an immutable `NamedTuple`, so
    a one-slot update rebuilds every level on the path to it -- this is the
    same reconstruction cost `refine.py`'s Type-1 `apply_gaussian_params` pays
    per candidate, just for a single slot instead of the whole flattened
    vector, because sub-problems here are evaluated one slot at a time.
    """
    fmodel = model.feature_models[fname]
    lmodel = fmodel.label_models[label]
    memberships = list(lmodel.memberships)
    memberships[idx] = new_it2_mf

    new_label_models = dict(fmodel.label_models)
    new_label_models[label] = IT2LabelModel(memberships)
    new_feature_models = dict(model.feature_models)
    new_feature_models[fname] = IT2FeatureModel(new_label_models)
    return model._replace(feature_models=new_feature_models)


def _normalize_proba(fs: np.ndarray, n_labels: int) -> np.ndarray:
    """Row-normalize (non-negative) firing strengths into probabilities, with
    zero-firing rows falling back to uniform -- the same convention
    `it2_classifier.predict`'s `argmax` implicitly relies on, made explicit
    here because cross-entropy needs an actual probability, not a raw score.
    """
    fs = np.clip(fs, 0.0, None)
    row = fs.sum(axis=1, keepdims=True)
    proba = np.full_like(fs, 1.0 / max(n_labels, 1))
    np.divide(fs, row, out=proba, where=row > 0)
    return proba


def _cross_entropy_loss(
    it2_model: IT2GaussianMixtureModel,
    X: pd.DataFrame,
    y_idx: np.ndarray,
    norms: NormPair,
    km_iterations: int | None,
) -> float:
    """Mean cross-entropy of the type-reduced, row-normalized firing strengths
    against integer class indices `y_idx` (positions into the model's sorted
    label list -- callers map original class labels to this before calling).
    """
    _, _, firing_crisp, _ = it2_firing_strengths(X, it2_model, norms, km_iterations=km_iterations)
    proba = _normalize_proba(firing_crisp, it2_model.n_classes)
    p = np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)
    return float(np.mean(-np.log(p)))


def refine_it2_antecedents(
    X: pd.DataFrame,
    y_labels: np.ndarray,
    it2_model: IT2GaussianMixtureModel,
    norms: NormPair,
    method: str = "coordinate",
    max_iterations: int = 100,
    km_iterations: int | None = 10,
    l2_shrink: float = 0.05,
    n_sweeps: int = 3,
    sub_maxfun: int = 25,
    sigma_min_frac: float = 0.02,
    tol: float = 1e-5,
    verbose: bool = True,
) -> IT2GaussianMixtureModel:
    """Refine IT2 Gaussian antecedents to minimize training cross-entropy.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature data.
    y_labels : np.ndarray
        Integer class indices, one per row of `X`, positioned into
        `it2_model`'s sorted label list (i.e. column ``j`` of
        `it2_firing_strengths`'s `firing_crisp` is class `j`). Callers holding
        original class labels should map them through that ordering first
        (`it2_classifier.py` does this via `np.searchsorted` on its sorted
        `classes_`).
    it2_model : IT2GaussianMixtureModel
        The IT2 model to refine (already converted from a fitted Type-1 base).
    norms : NormPair
        (t-norm, t-conorm) pair for inference.
    method : str, default="coordinate"
        "coordinate" (block coordinate descent, the only implemented method)
        or "none"/`None` (return `it2_model` unchanged).
    max_iterations : int, default=100
        Accepted for backward compatibility; superseded by `n_sweeps` (each
        sweep already visits every slot, so it plays the role this bounded).
    km_iterations : int | None, default=10
        Karnik-Mendel iterations used for the *loss* evaluated during
        refinement. Independent of whatever `km_iterations` the caller later
        predicts with -- refinement only needs a stable objective to descend,
        not the exact inference-time setting.
    l2_shrink : float, default=0.05
        Per-sweep L2 penalty weight pulling each slot back toward its value at
        the start of the current sweep (a ridge-style anchor, not a hard
        trust region), preventing any single slot from taking an unbounded
        step off of a nearly-flat direction in the loss.
    n_sweeps : int, default=3
        Number of full passes over every IT2 Gaussian membership.
    sub_maxfun : int, default=25
        Currently unused (accepted for backward compatibility): each slot's
        sub-problem now runs through `refine._optimizers_sub_solve` (#119),
        which has no evaluation-budget knob to forward this to.
    sigma_min_frac : float, default=0.02
        Lower bound on sigma, as a fraction of the owning feature's observed
        range (mirrors `refine.py`'s `build_param_bounds`).
    tol : float, default=1e-5
        Stop sweeping early once a sweep improves the loss by less than this.
    verbose : bool, default=True
        Print per-sweep progress, matching `refine.py`'s convention.

    Returns
    -------
    refined_model : IT2GaussianMixtureModel
        The best model found, which is never worse (on training cross-entropy)
        than `it2_model` itself -- a sweep's candidate is adopted only on a
        strict improvement, so an unlucky search simply returns the input
        model unchanged. `firing_lower <= firing_upper` is preserved by
        construction (see `_iter_it2_slots`).
    """
    if method in (None, "none"):
        return it2_model
    if method != "coordinate":
        raise ValueError(f"Unknown refinement method: {method!r}")

    y_idx = np.asarray(y_labels, dtype=np.intp)
    n_classes = it2_model.n_classes

    slots = list(_iter_it2_slots(it2_model))
    if not slots:
        return it2_model

    # Per-feature (peak, spread) box bounds from the observed data range,
    # shared by every IT2 membership on that feature -- mirrors `refine.py`'s
    # `build_param_bounds`.
    feature_bounds: dict[str, tuple[float, float, float]] = {}
    for fname in {s[0] for s in slots}:
        col = X[fname].to_numpy(dtype=float)
        lo, hi = float(np.min(col)), float(np.max(col))
        rng = hi - lo if hi > lo else 1.0
        feature_bounds[fname] = (lo, hi, rng)

    current = it2_model
    best_model = it2_model
    best_loss = _cross_entropy_loss(it2_model, X, y_idx, norms, km_iterations)
    init_loss = best_loss

    if verbose:
        print(f"\nIT2 classifier coordinate-descent antecedent refinement: {len(slots)} "
              f"memberships, init loss={init_loss:.4f}")

    for sweep in range(n_sweeps):
        sweep_start_loss = best_loss
        for fname, label, idx, it2_mf in _iter_it2_slots(current):
            lo, hi, rng = feature_bounds[fname]
            x0, bounds = _slot_x0_and_bounds(it2_mf, lo, hi, rng, sigma_min_frac)

            def fitness(v, fname=fname, label=label, idx=idx, it2_mf=it2_mf, x0=x0):
                new_it2_mf = _apply_slot_params(it2_mf, v)
                trial = _replace_slot(current, fname, label, idx, new_it2_mf)
                loss = _cross_entropy_loss(trial, X, y_idx, norms, km_iterations)
                penalty = l2_shrink * float(np.sum((v - x0) ** 2))
                return loss + penalty

            res = _optimizers_sub_solve(fitness, x0, bounds)

            candidate = _replace_slot(current, fname, label, idx, _apply_slot_params(it2_mf, res.x))
            candidate_loss = _cross_entropy_loss(candidate, X, y_idx, norms, km_iterations)
            if candidate_loss < best_loss - 1e-12:
                current = candidate
                best_model = candidate
                best_loss = candidate_loss

        if verbose:
            print(f"  sweep {sweep + 1}/{n_sweeps}: loss={best_loss:.4f}")
        if sweep_start_loss - best_loss < tol:
            break

    if verbose:
        print(f"  IT2 refinement done: loss {init_loss:.4f} -> {best_loss:.4f} "
              f"({100 * (init_loss - best_loss) / max(init_loss, 1e-12):.1f}% lower)")

    return best_model


# ---------------------------------------------------------------------------
# Regression: antecedent refinement with a per-candidate consequent re-solve.
# ---------------------------------------------------------------------------

def _it2_rule_firing(
    it2_model: IT2GaussianMixtureModel,
    X: pd.DataFrame,
    top_n_todo: list,
    norms: NormPair,
    feature_arrays: dict[str, np.ndarray] | None = None,
):
    """(firing_upper, firing_lower, labels) for every rule, from the IT2
    antecedents directly -- the regressor analogue of `it2_firing_strengths`,
    kept separate because the regressor refiner never wants that function's
    per-rule *crisp* reduction (see `it2_kernel`'s module docstring: that
    reduction is the wrong one for combining rules, which is exactly the
    problem being solved here)."""
    upper_model = _extract_upper_model(it2_model)
    lower_model = _extract_lower_model(it2_model)
    firing_upper, labels = tsk_firing_strengths(
        X[top_n_todo], upper_model, norms=norms, feature_arrays=feature_arrays
    )
    firing_lower, _ = tsk_firing_strengths(
        X[top_n_todo], lower_model, norms=norms, feature_arrays=feature_arrays
    )
    return firing_upper, firing_lower, labels


def _solve_it2_consequents(
    it2_model: IT2GaussianMixtureModel,
    X: pd.DataFrame,
    y_df: pd.DataFrame,
    top_n_todo: list,
    norms: NormPair,
    order: str,
    l2_reg: float,
    basis: str,
    cross_pairs: list[tuple[int, int]] | None,
    feature_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Re-solve TSK consequents in closed form for `it2_model`'s *current*
    antecedents, weighting each rule's ridge design block by its midpoint
    firing strength (see module docstring). Returns
    ``(corr_terms, y_bucket_mean, labels)``.
    """
    firing_upper, firing_lower, labels = _it2_rule_firing(
        it2_model, X, top_n_todo, norms, feature_arrays=feature_arrays
    )
    midpoint = 0.5 * (firing_upper + firing_lower)
    corr_terms, y_bucket_mean = solve_tsk_consequents_from_firing(
        midpoint, labels, X, top_n_todo, None, y_df,
        order=order, l2_reg=l2_reg, basis=basis, cross_pairs=cross_pairs,
        pin_extremes=False, verbose=False, feature_arrays=feature_arrays,
    )
    return corr_terms, y_bucket_mean, labels


def _it2_regressor_fold_mse(
    it2_model: IT2GaussianMixtureModel,
    X_tr: pd.DataFrame, y_tr_df: pd.DataFrame, fa_tr: dict,
    X_val: pd.DataFrame, y_val_true: np.ndarray, fa_val: dict,
    top_n_todo: list, norms: NormPair,
    order: str, l2_reg: float, basis: str, cross_pairs: list[tuple[int, int]] | None,
    km_iterations: int,
) -> float:
    """Held-out MSE for one fold: re-solve consequents on the training split,
    then predict the validation split through the full Karnik-Mendel path.
    """
    corr_terms, y_bucket_mean, labels = _solve_it2_consequents(
        it2_model, X_tr, y_tr_df, top_n_todo, norms, order, l2_reg, basis, cross_pairs,
        feature_arrays=fa_tr,
    )
    firing_upper_val, firing_lower_val, _ = _it2_rule_firing(
        it2_model, X_val, top_n_todo, norms, feature_arrays=fa_val
    )
    rule_values_val = rule_consequent_values(
        X_val, top_n_todo, labels, y_bucket_mean, corr_terms,
        order=order, basis=basis, cross_pairs=cross_pairs, feature_arrays=fa_val,
    )
    y_l, y_r = karnik_mendel_tsk(
        rule_values_val, firing_lower_val, firing_upper_val, max_iterations=km_iterations
    )
    return _mse(y_val_true, 0.5 * (y_l + y_r))


def _it2_regressor_cv_fitness(
    it2_model: IT2GaussianMixtureModel,
    prepared: list,
    top_n_todo: list, norms: NormPair,
    order: str, l2_reg: float, basis: str, cross_pairs: list[tuple[int, int]] | None,
    km_iterations: int,
) -> float:
    """Mean held-out MSE over `prepared`'s folds (see `refine._prepare_folds`) --
    the regressor's coordinate-descent fitness, analogous to
    `refine._make_kfold_fitness` for Type-1."""
    total, n = 0.0, 0
    for X_tr, y_tr_df, fa_tr, X_val, y_val_true, fa_val in prepared:
        try:
            mse = _it2_regressor_fold_mse(
                it2_model, X_tr, y_tr_df, fa_tr, X_val, y_val_true, fa_val,
                top_n_todo, norms, order, l2_reg, basis, cross_pairs, km_iterations,
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, ZeroDivisionError):
            # Narrowed from bare Exception: swallow only the numerical failures a bad
            # candidate can legitimately cause; let real bugs surface instead of
            # silently rejecting the step and "stopping improvement".
            return 1e6
        total += mse
        n += 1
    return total / max(n, 1)


def refine_it2_regressor_antecedents(
    X: pd.DataFrame,
    y: np.ndarray,
    it2_model: IT2GaussianMixtureModel,
    norms: NormPair,
    top_n_todo: list,
    order: str = "1st",
    l2_reg: float = 1e-6,
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
    km_iterations: int = 15,
    method: str = "coordinate",
    l2_shrink: float = 0.0,
    n_sweeps: int = 3,
    sub_maxfun: int = 20,
    sigma_min_frac: float = 0.02,
    tol: float = 1e-6,
    val_fraction: float = 0.2,
    n_folds: int = 3,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[IT2GaussianMixtureModel, np.ndarray, np.ndarray, dict]:
    """Refine IT2 Gaussian antecedents against held-out regression MSE,
    re-solving TSK consequents in closed form for every candidate.

    Parameters
    ----------
    X : pd.DataFrame
        Training feature data (only the `top_n_todo` columns are used).
    y : np.ndarray
        Training target values, one per row of `X`.
    it2_model : IT2GaussianMixtureModel
        The IT2 model to refine (already converted from a fitted Type-1 base).
    norms : NormPair
        (t-norm, t-conorm) pair for inference.
    top_n_todo : list
        The base regressor's selected feature names/order (`top_features_`).
    order, l2_reg, basis, cross_pairs : as in `regression.solve_tsk_consequents`
        -- must match the base regressor's own settings (`tsk_order`, `l2_reg`,
        `consequent_basis`, `cross_pairs_`) for the re-solved consequents to be
        directly comparable to the ones already in use.
    km_iterations : int, default=15
        Karnik-Mendel iterations for the loss evaluated during refinement
        (independent of whatever the caller predicts with afterward).
    method : str, default="coordinate"
        "coordinate" (block coordinate descent, the only implemented method)
        or "none"/`None` (skip the antecedent search but still re-solve
        consequents once against the *unchanged* antecedents, so the returned
        consequents are never stale relative to `it2_model`).
    n_sweeps, sub_maxfun, sigma_min_frac, tol : as in
        `refine_it2_antecedents` (the classifier's coordinate descent).
    val_fraction, n_folds, seed : cross-validation split, reusing
        `refine._make_folds`/`refine._prepare_folds` (same convention as
        `refine.py`'s Type-1 regressor coordinate descent).
    verbose : bool, default=True
        Print per-sweep progress.

    Returns
    -------
    (refined_model, corr_terms, y_bucket_mean, info) : the best antecedents
        found (never worse, on cross-validated MSE, than `it2_model`'s own),
        together with consequents re-solved on the *full* training set for
        those antecedents (the per-fold re-solves inside the search are for
        scoring candidates only -- the final consequents handed back are
        fit once, on all of `X`, the same way the base Type-1 regressor's are).
        `info` holds `init_val_mse`/`val_mse` for the CV objective actually
        searched.
    """
    y_df = pd.DataFrame({"y_value": np.asarray(y, dtype=float)})

    if method not in (None, "none", "coordinate"):
        raise ValueError(f"Unknown refinement method: {method!r}")

    if method in (None, "none"):
        corr_terms, y_bucket_mean, _ = _solve_it2_consequents(
            it2_model, X, y_df, top_n_todo, norms, order, l2_reg, basis, cross_pairs,
        )
        return it2_model, corr_terms, y_bucket_mean, {"init_val_mse": None, "val_mse": None}

    slots = list(_iter_it2_slots(it2_model))
    if not slots:
        corr_terms, y_bucket_mean, _ = _solve_it2_consequents(
            it2_model, X, y_df, top_n_todo, norms, order, l2_reg, basis, cross_pairs,
        )
        return it2_model, corr_terms, y_bucket_mean, {"init_val_mse": None, "val_mse": None}

    feature_bounds: dict[str, tuple[float, float, float]] = {}
    for fname in {s[0] for s in slots}:
        col = X[fname].to_numpy(dtype=float)
        lo, hi = float(np.min(col)), float(np.max(col))
        rng = hi - lo if hi > lo else 1.0
        feature_bounds[fname] = (lo, hi, rng)

    folds = _make_folds(len(X), n_folds, val_fraction, seed)
    prepared = _prepare_folds(X, y_df, folds)

    def cv_fitness(model):
        return _it2_regressor_cv_fitness(
            model, prepared, top_n_todo, norms, order, l2_reg, basis, cross_pairs, km_iterations
        )

    current = it2_model
    best_model = it2_model
    best_loss = cv_fitness(it2_model)
    init_loss = best_loss

    if verbose:
        print(f"\nIT2 regressor coordinate-descent antecedent refinement: {len(slots)} "
              f"memberships, {n_folds}-fold init val MSE={init_loss:.5f}")

    for sweep in range(n_sweeps):
        sweep_start_loss = best_loss
        for fname, label, idx, it2_mf in _iter_it2_slots(current):
            lo, hi, rng = feature_bounds[fname]
            x0, bounds = _slot_x0_and_bounds(it2_mf, lo, hi, rng, sigma_min_frac)

            def fitness(v, fname=fname, label=label, idx=idx, it2_mf=it2_mf, x0=x0):
                new_it2_mf = _apply_slot_params(it2_mf, v)
                trial = _replace_slot(current, fname, label, idx, new_it2_mf)
                # L2 pull-back toward the heuristic start, mirroring the classifier
                # path so a flat CV direction cannot take a large, barely-helping
                # step. Defaults to 0.0 (no change) and is opt-in for the regressor.
                penalty = l2_shrink * float(np.sum((v - x0) ** 2))
                return cv_fitness(trial) + penalty

            res = _optimizers_sub_solve(fitness, x0, bounds)

            candidate = _replace_slot(current, fname, label, idx, _apply_slot_params(it2_mf, res.x))
            candidate_loss = cv_fitness(candidate)
            if candidate_loss < best_loss - 1e-12:
                current = candidate
                best_model = candidate
                best_loss = candidate_loss

        if verbose:
            print(f"  sweep {sweep + 1}/{n_sweeps}: val MSE={best_loss:.5f}")
        if sweep_start_loss - best_loss < tol:
            break

    if verbose:
        print(f"  IT2 regressor refinement done: val MSE {init_loss:.5f} -> {best_loss:.5f} "
              f"({100 * (init_loss - best_loss) / max(init_loss, 1e-12):.1f}% lower)")

    corr_terms, y_bucket_mean, _ = _solve_it2_consequents(
        best_model, X, y_df, top_n_todo, norms, order, l2_reg, basis, cross_pairs,
    )
    return best_model, corr_terms, y_bucket_mean, {"init_val_mse": init_loss, "val_mse": best_loss}
