"""Post-fit antecedent refinement for General Type-2 (GT2) FIS models.

This is the GT2 counterpart of `it2_refine.py`'s block coordinate descent,
extended by exactly one dimension per side per slot: an IT2 Gaussian slot
searches ``(mu, sigma_lower, sigma_upper)``; a GT2 Gaussian slot searches
``(mu, sigma_lower, sigma_principal, sigma_upper)``, with
``sigma_lower <= sigma_principal <= sigma_upper`` enforced by construction
in `_apply_slot_params` -- the direct GT2 analogue of `it2_refine`'s own
``sigma_upper >= sigma_lower`` fix, for the same reason: an unordered triple
would let some alpha-plane's footprint invert
(`GT2GaussianMembership.alpha_cut`'s narrowing property depends on the
ordering holding). Trapezoid/triangular slots follow the same pattern one
level further: `it2_refine.py`'s two independent per-side spread gaps each
grow one more non-negative "extra principal" gap (see
`_trapezoid_slot_x0_and_bounds`/`_triangular_slot_x0_and_bounds`).

**Classifier** (`refine_gt2_antecedents`): unchanged rationale from
`it2_refine.refine_it2_antecedents` -- a zeroth-order TSK classifier's
antecedents *are* the whole model, so the fitness is the cross-entropy of the
alpha-combined, row-normalized firing strengths
(`gt2_kernel.gt2_firing_strengths`) against the training labels, with no
closed-form inner solve.

**Regressor** (`refine_gt2_regressor_antecedents`): unchanged rationale from
`it2_refine.refine_it2_regressor_antecedents` -- a candidate antecedent set's
consequents must be re-solved in closed form before scoring it (a fixed
consequent set fit for a different footprint of uncertainty scores a
mismatched pair). The re-solve here uses the *alpha-weighted average* of each
plane's own midpoint firing strength (`0.5 * (firing_upper + firing_lower)`)
as the ridge design weight -- the natural GT2 generalisation of IT2's own
midpoint weight, and it collapses to IT2's exactly when the GT2 model's
principal sigma sits at the interval midpoint (a uniform secondary grade).

Both reuse `refine.py`'s CV-fold plumbing (`_make_folds`/`_prepare_folds`)
and `it2_refine.py`'s classifier probability helpers (`_normalize_proba`)
directly -- neither depends on whether the underlying model is IT2 or GT2.
"""

import numpy as np
import pandas as pd

from .gauss_data import (
    GT2GaussianMixtureModel, GT2FeatureModel, GT2LabelModel,
    GT2GaussianMembership, GT2TrapezoidMembership, GT2TriangularMembership,
    GaussianMembership, TrapezoidMembership, TriangularMembership, NormPair,
)
from .gt2_kernel import (
    gt2_firing_strengths, gt2_rule_firing, gt2_karnik_mendel_tsk, alpha_weighted_average,
)
from .it2_refine import _normalize_proba, _side_widths
from .optimizer_utils import optimizers_sub_solve as _optimizers_sub_solve
from .refine import _make_folds, _prepare_folds
from .regression import solve_tsk_consequents_from_firing, rule_consequent_values, _mse


def _iter_gt2_slots(model: GT2GaussianMixtureModel):
    """Yield ``(feature_name, label, mf_index, GT2AnyMembership)`` for every
    GT2 membership (any of Gaussian/trapezoid/triangular) in a deterministic
    order.

    Mirrors `it2_refine._iter_it2_slots` exactly, one level up: a slot here
    is a whole ``(upper_mf, lower_mf, principal_mf)`` triple, never a single
    half, for the same reason IT2's own slot is the whole membership -- see
    this module's docstring.
    """
    for fname, fmodel in model.feature_models.items():
        for label, lmodel in fmodel.label_models.items():
            for idx, gt2_mf in enumerate(lmodel.memberships):
                yield fname, label, idx, gt2_mf


def _side_widths3(peak: float, lower_edge: float, principal_edge: float, upper_edge: float, min_width: float):
    """``(lower_width, extra_principal, extra_upper)`` for one side of one
    GT2 slot -- the three-level extension of `it2_refine._side_widths`
    (lower/upper only) that a GT2 slot's extra ``principal`` component needs.
    """
    lower_width = max(abs(peak - lower_edge), min_width)
    principal_width = max(abs(peak - principal_edge), min_width)
    upper_width = max(abs(peak - upper_edge), min_width)
    extra_principal = max(principal_width - lower_width, 0.0)
    extra_upper = max(upper_width - (lower_width + extra_principal), 0.0)
    return lower_width, extra_principal, extra_upper


def _slot_x0_and_bounds(gt2_mf, lo: float, hi: float, rng: float, sigma_min_frac: float):
    """Initial parameter vector and box bounds for one slot, dispatched by
    `gt2_mf`'s membership type -- the GT2 analogue of
    `it2_refine._slot_x0_and_bounds`, one dimension wider per side (an extra
    non-negative gap for the principal component)."""
    if isinstance(gt2_mf, GT2GaussianMembership):
        return _gaussian_slot_x0_and_bounds(gt2_mf, lo, hi, rng, sigma_min_frac)
    if isinstance(gt2_mf, GT2TrapezoidMembership):
        return _trapezoid_slot_x0_and_bounds(gt2_mf, lo, hi, rng, sigma_min_frac)
    if isinstance(gt2_mf, GT2TriangularMembership):
        return _triangular_slot_x0_and_bounds(gt2_mf, lo, hi, rng, sigma_min_frac)
    raise TypeError(f"Unsupported GT2 membership type for refinement: {type(gt2_mf)!r}")


def _gaussian_slot_x0_and_bounds(gt2_mf, lo, hi, rng, sigma_min_frac):
    """``(mu, sigma_lower, sigma_principal, sigma_upper)``. ``mu`` starts at
    the principal half's center, mirroring `it2_refine`'s choice of the
    upper half's own ``mu`` as anchor -- for a freshly converted model all
    three halves share one ``mu`` anyway (see `GT2GaussianMembership`'s
    docstring)."""
    mu0 = gt2_mf.principal_mf.mu
    sigma_lower0 = gt2_mf.lower_mf.sigma
    sigma_principal0 = gt2_mf.principal_mf.sigma
    sigma_upper0 = gt2_mf.upper_mf.sigma
    x0 = np.array([mu0, sigma_lower0, sigma_principal0, sigma_upper0])
    sigma_lo = sigma_min_frac * rng
    bounds = [(lo, hi), (sigma_lo, rng), (sigma_lo, rng), (sigma_lo, 2.0 * rng)]
    return x0, bounds


def _trapezoid_slot_x0_and_bounds(gt2_mf, lo, hi, rng, sigma_min_frac):
    """``(b, gap_bc, left_lower, extra_left_principal, extra_left_upper,
    right_lower, extra_right_principal, extra_right_upper)`` -- the GT2
    analogue of `it2_refine._trapezoid_slot_x0_and_bounds`, one extra
    non-negative gap per side for the principal component. Reconstructing
    `a`/`d` from these makes ``a_upper <= a_principal <= a_lower <= b <= c
    <= d_lower <= d_principal <= d_upper`` true for *any* point in these box
    bounds (see `_apply_trapezoid_slot_params`)."""
    min_width = sigma_min_frac * rng
    b0, c0 = gt2_mf.principal_mf.b, gt2_mf.principal_mf.c
    left_lower0, extra_left_p0, extra_left_u0 = _side_widths3(
        b0, gt2_mf.lower_mf.a, gt2_mf.principal_mf.a, gt2_mf.upper_mf.a, min_width,
    )
    right_lower0, extra_right_p0, extra_right_u0 = _side_widths3(
        c0, gt2_mf.lower_mf.d, gt2_mf.principal_mf.d, gt2_mf.upper_mf.d, min_width,
    )
    x0 = np.array([
        b0, max(c0 - b0, 0.0),
        left_lower0, extra_left_p0, extra_left_u0,
        right_lower0, extra_right_p0, extra_right_u0,
    ])
    bounds = [
        (lo, hi), (0.0, rng),
        (min_width, rng), (0.0, rng), (0.0, rng),
        (min_width, rng), (0.0, rng), (0.0, rng),
    ]
    return x0, bounds


def _triangular_slot_x0_and_bounds(gt2_mf, lo, hi, rng, sigma_min_frac):
    """``(b, left_lower, extra_left_principal, extra_left_upper, right_lower,
    extra_right_principal, extra_right_upper)`` -- the triangular analogue of
    `_trapezoid_slot_x0_and_bounds`, one parameter fewer since the apex ``b``
    is a single point rather than an interval."""
    min_width = sigma_min_frac * rng
    b0 = gt2_mf.principal_mf.b
    left_lower0, extra_left_p0, extra_left_u0 = _side_widths3(
        b0, gt2_mf.lower_mf.a, gt2_mf.principal_mf.a, gt2_mf.upper_mf.a, min_width,
    )
    right_lower0, extra_right_p0, extra_right_u0 = _side_widths3(
        b0, gt2_mf.lower_mf.c, gt2_mf.principal_mf.c, gt2_mf.upper_mf.c, min_width,
    )
    x0 = np.array([
        b0,
        left_lower0, extra_left_p0, extra_left_u0,
        right_lower0, extra_right_p0, extra_right_u0,
    ])
    bounds = [
        (lo, hi),
        (min_width, rng), (0.0, rng), (0.0, rng),
        (min_width, rng), (0.0, rng), (0.0, rng),
    ]
    return x0, bounds


def _apply_slot_params(gt2_mf, v: np.ndarray):
    """Build a fresh GT2 membership of `gt2_mf`'s own type from `v`,
    dispatched by type -- the GT2 analogue of `it2_refine._apply_slot_params`.
    """
    if isinstance(gt2_mf, GT2GaussianMembership):
        ids = (gt2_mf.upper_mf.id, gt2_mf.lower_mf.id, gt2_mf.principal_mf.id)
        return _apply_gaussian_slot_params(v, *ids)
    if isinstance(gt2_mf, GT2TrapezoidMembership):
        ids = (gt2_mf.upper_mf.id, gt2_mf.lower_mf.id, gt2_mf.principal_mf.id)
        return _apply_trapezoid_slot_params(v, *ids)
    if isinstance(gt2_mf, GT2TriangularMembership):
        ids = (gt2_mf.upper_mf.id, gt2_mf.lower_mf.id, gt2_mf.principal_mf.id)
        return _apply_triangular_slot_params(v, *ids)
    raise TypeError(f"Unsupported GT2 membership type for refinement: {type(gt2_mf)!r}")


def _apply_gaussian_slot_params(v: np.ndarray, upper_id, lower_id, principal_id) -> GT2GaussianMembership:
    """Build a fresh `GT2GaussianMembership` from
    ``(mu, sigma_lower, sigma_principal, sigma_upper)``, clamping
    ``sigma_lower <= sigma_principal <= sigma_upper`` so the ordering
    `_iter_gt2_slots` documents cannot be violated regardless of what
    the optimizer proposes.
    """
    mu = float(v[0])
    sigma_lower = max(float(v[1]), 1e-6)
    sigma_principal = max(float(v[2]), sigma_lower)
    sigma_upper = max(float(v[3]), sigma_principal)
    return GT2GaussianMembership(
        upper_mf=GaussianMembership(mu=mu, sigma=sigma_upper, id=upper_id),
        lower_mf=GaussianMembership(mu=mu, sigma=sigma_lower, id=lower_id),
        principal_mf=GaussianMembership(mu=mu, sigma=sigma_principal, id=principal_id),
    )


def _apply_trapezoid_slot_params(v: np.ndarray, upper_id, lower_id, principal_id) -> GT2TrapezoidMembership:
    """Build a fresh `GT2TrapezoidMembership` from
    ``(b, gap_bc, left_lower, extra_left_principal, extra_left_upper,
    right_lower, extra_right_principal, extra_right_upper)``, all gaps
    clamped non-negative so ``a_upper <= a_principal <= a_lower <= b <= c <=
    d_lower <= d_principal <= d_upper`` holds regardless of what the
    optimizer proposes."""
    b = float(v[0])
    c = b + max(float(v[1]), 0.0)
    left_lower = max(float(v[2]), 1e-6)
    left_principal = left_lower + max(float(v[3]), 0.0)
    left_upper = left_principal + max(float(v[4]), 0.0)
    right_lower = max(float(v[5]), 1e-6)
    right_principal = right_lower + max(float(v[6]), 0.0)
    right_upper = right_principal + max(float(v[7]), 0.0)
    return GT2TrapezoidMembership(
        upper_mf=TrapezoidMembership(a=b - left_upper, b=b, c=c, d=c + right_upper, id=upper_id),
        lower_mf=TrapezoidMembership(a=b - left_lower, b=b, c=c, d=c + right_lower, id=lower_id),
        principal_mf=TrapezoidMembership(a=b - left_principal, b=b, c=c, d=c + right_principal, id=principal_id),
    )


def _apply_triangular_slot_params(v: np.ndarray, upper_id, lower_id, principal_id) -> GT2TriangularMembership:
    """Build a fresh `GT2TriangularMembership` from ``(b, left_lower,
    extra_left_principal, extra_left_upper, right_lower,
    extra_right_principal, extra_right_upper)`` -- the triangular analogue
    of `_apply_trapezoid_slot_params`, apex ``b`` shared instead of a flat top."""
    b = float(v[0])
    left_lower = max(float(v[1]), 1e-6)
    left_principal = left_lower + max(float(v[2]), 0.0)
    left_upper = left_principal + max(float(v[3]), 0.0)
    right_lower = max(float(v[4]), 1e-6)
    right_principal = right_lower + max(float(v[5]), 0.0)
    right_upper = right_principal + max(float(v[6]), 0.0)
    return GT2TriangularMembership(
        upper_mf=TriangularMembership(a=b - left_upper, b=b, c=b + right_upper, id=upper_id),
        lower_mf=TriangularMembership(a=b - left_lower, b=b, c=b + right_lower, id=lower_id),
        principal_mf=TriangularMembership(a=b - left_principal, b=b, c=b + right_principal, id=principal_id),
    )


def _replace_slot(
    model: GT2GaussianMixtureModel,
    fname: str,
    label: int,
    idx: int,
    new_gt2_mf,
) -> GT2GaussianMixtureModel:
    """Return a copy of `model` with one GT2 membership replaced.

    Same NamedTuple-rebuild cost `it2_refine._replace_slot` pays, one level
    up the container hierarchy.
    """
    fmodel = model.feature_models[fname]
    lmodel = fmodel.label_models[label]
    memberships = list(lmodel.memberships)
    memberships[idx] = new_gt2_mf

    new_label_models = dict(fmodel.label_models)
    new_label_models[label] = GT2LabelModel(memberships)
    new_feature_models = dict(model.feature_models)
    new_feature_models[fname] = GT2FeatureModel(new_label_models)
    return model._replace(feature_models=new_feature_models)


def _cross_entropy_loss(
    gt2_model: GT2GaussianMixtureModel,
    X: pd.DataFrame,
    y_idx: np.ndarray,
    norms: NormPair,
    n_alpha_planes: int,
    km_iterations: int | None,
) -> float:
    """Mean cross-entropy of the alpha-combined, row-normalized firing
    strengths against integer class indices ``y_idx`` -- the GT2 analogue of
    `it2_refine._cross_entropy_loss`."""
    firing_crisp, _ = gt2_firing_strengths(
        X, gt2_model, norms, n_alpha_planes=n_alpha_planes, km_iterations=km_iterations
    )
    proba = _normalize_proba(firing_crisp, gt2_model.n_classes)
    p = np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)
    return float(np.mean(-np.log(p)))


def refine_gt2_antecedents(
    X: pd.DataFrame,
    y_labels: np.ndarray,
    gt2_model: GT2GaussianMixtureModel,
    norms: NormPair,
    method: str = "coordinate",
    max_iterations: int = 100,
    n_alpha_planes: int = 5,
    km_iterations: int | None = None,
    l2_shrink: float = 0.05,
    n_sweeps: int = 3,
    sub_maxfun: int = 25,
    sigma_min_frac: float = 0.02,
    tol: float = 1e-5,
    verbose: bool = True,
) -> GT2GaussianMixtureModel:
    """Refine GT2 Gaussian antecedents to minimize training cross-entropy.

    Direct GT2 analogue of `it2_refine.refine_it2_antecedents` -- see that
    function's docstring for the parameters shared verbatim
    (`method`, `l2_shrink`, `n_sweeps`, `sub_maxfun`, `sigma_min_frac`,
    `tol`, `verbose`). ``n_alpha_planes``/``km_iterations`` are passed
    through to every `gt2_firing_strengths` call the search makes.

    Returns
    -------
    refined_model : GT2GaussianMixtureModel
        The best model found, never worse (on training cross-entropy) than
        `gt2_model` itself. ``sigma_lower <= sigma_principal <= sigma_upper``
        is preserved by construction (see `_iter_gt2_slots`).
    """
    if method in (None, "none"):
        return gt2_model
    if method != "coordinate":
        raise ValueError(f"Unknown refinement method: {method!r}")

    y_idx = np.asarray(y_labels, dtype=np.intp)

    slots = list(_iter_gt2_slots(gt2_model))
    if not slots:
        return gt2_model

    feature_bounds: dict[str, tuple[float, float, float]] = {}
    for fname in {s[0] for s in slots}:
        col = X[fname].to_numpy(dtype=float)
        lo, hi = float(np.min(col)), float(np.max(col))
        rng = hi - lo if hi > lo else 1.0
        feature_bounds[fname] = (lo, hi, rng)

    current = gt2_model
    best_model = gt2_model
    best_loss = _cross_entropy_loss(gt2_model, X, y_idx, norms, n_alpha_planes, km_iterations)
    init_loss = best_loss

    if verbose:
        print(f"\nGT2 classifier coordinate-descent antecedent refinement: {len(slots)} "
              f"memberships, {n_alpha_planes} alpha-planes, init loss={init_loss:.4f}")

    for sweep in range(n_sweeps):
        sweep_start_loss = best_loss
        for fname, label, idx, gt2_mf in _iter_gt2_slots(current):
            lo, hi, rng = feature_bounds[fname]
            x0, bounds = _slot_x0_and_bounds(gt2_mf, lo, hi, rng, sigma_min_frac)

            def fitness(v, fname=fname, label=label, idx=idx, gt2_mf=gt2_mf, x0=x0):
                new_gt2_mf = _apply_slot_params(gt2_mf, v)
                trial = _replace_slot(current, fname, label, idx, new_gt2_mf)
                loss = _cross_entropy_loss(trial, X, y_idx, norms, n_alpha_planes, km_iterations)
                penalty = l2_shrink * float(np.sum((v - x0) ** 2))
                return loss + penalty

            res = _optimizers_sub_solve(fitness, x0, bounds)

            candidate = _replace_slot(current, fname, label, idx, _apply_slot_params(gt2_mf, res.x))
            candidate_loss = _cross_entropy_loss(candidate, X, y_idx, norms, n_alpha_planes, km_iterations)
            if candidate_loss < best_loss - 1e-12:
                current = candidate
                best_model = candidate
                best_loss = candidate_loss

        if verbose:
            print(f"  sweep {sweep + 1}/{n_sweeps}: loss={best_loss:.4f}")
        if sweep_start_loss - best_loss < tol:
            break

    if verbose:
        print(f"  GT2 refinement done: loss {init_loss:.4f} -> {best_loss:.4f} "
              f"({100 * (init_loss - best_loss) / max(init_loss, 1e-12):.1f}% lower)")

    return best_model


# ---------------------------------------------------------------------------
# Regression: antecedent refinement with a per-candidate consequent re-solve.
# ---------------------------------------------------------------------------

def _solve_gt2_consequents(
    gt2_model: GT2GaussianMixtureModel,
    X: pd.DataFrame,
    y_df: pd.DataFrame,
    top_n_todo: list,
    norms: NormPair,
    order: str,
    l2_reg: float,
    basis: str,
    cross_pairs: list[tuple[int, int]] | None,
    n_alpha_planes: int,
    feature_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Re-solve TSK consequents in closed form for `gt2_model`'s *current*
    antecedents, weighting each rule's ridge design block by the
    alpha-weighted average of each plane's own midpoint firing strength (see
    module docstring). Returns ``(corr_terms, y_bucket_mean, labels)``.
    """
    firing_uppers, firing_lowers, alphas, labels = gt2_rule_firing(
        gt2_model, X, top_n_todo, norms, n_alpha_planes=n_alpha_planes, feature_arrays=feature_arrays
    )
    midpoints = [0.5 * (fu + fl) for fu, fl in zip(firing_uppers, firing_lowers)]
    combined_midpoint = alpha_weighted_average(alphas, midpoints)
    corr_terms, y_bucket_mean = solve_tsk_consequents_from_firing(
        combined_midpoint, labels, X, top_n_todo, None, y_df,
        order=order, l2_reg=l2_reg, basis=basis, cross_pairs=cross_pairs,
        pin_extremes=False, verbose=False, feature_arrays=feature_arrays,
    )
    return corr_terms, y_bucket_mean, labels


def _gt2_regressor_fold_mse(
    gt2_model: GT2GaussianMixtureModel,
    X_tr: pd.DataFrame, y_tr_df: pd.DataFrame, fa_tr: dict,
    X_val: pd.DataFrame, y_val_true: np.ndarray, fa_val: dict,
    top_n_todo: list, norms: NormPair,
    order: str, l2_reg: float, basis: str, cross_pairs: list[tuple[int, int]] | None,
    n_alpha_planes: int, km_iterations: int,
) -> float:
    """Held-out MSE for one fold: re-solve consequents on the training split,
    then predict the validation split through the full alpha-combined
    Karnik-Mendel path."""
    corr_terms, y_bucket_mean, labels = _solve_gt2_consequents(
        gt2_model, X_tr, y_tr_df, top_n_todo, norms, order, l2_reg, basis, cross_pairs,
        n_alpha_planes, feature_arrays=fa_tr,
    )
    firing_uppers_val, firing_lowers_val, alphas, _ = gt2_rule_firing(
        gt2_model, X_val, top_n_todo, norms, n_alpha_planes=n_alpha_planes, feature_arrays=fa_val
    )
    rule_values_val = rule_consequent_values(
        X_val, top_n_todo, labels, y_bucket_mean, corr_terms,
        order=order, basis=basis, cross_pairs=cross_pairs, feature_arrays=fa_val,
    )
    y_l, y_r = gt2_karnik_mendel_tsk(
        rule_values_val, firing_uppers_val, firing_lowers_val, alphas, max_iterations=km_iterations
    )
    return _mse(y_val_true, 0.5 * (y_l + y_r))


def _gt2_regressor_cv_fitness(
    gt2_model: GT2GaussianMixtureModel,
    prepared: list,
    top_n_todo: list, norms: NormPair,
    order: str, l2_reg: float, basis: str, cross_pairs: list[tuple[int, int]] | None,
    n_alpha_planes: int, km_iterations: int,
) -> float:
    """Mean held-out MSE over `prepared`'s folds -- the GT2 regressor's
    coordinate-descent fitness, analogous to
    `it2_refine._it2_regressor_cv_fitness`."""
    total, n = 0.0, 0
    for X_tr, y_tr_df, fa_tr, X_val, y_val_true, fa_val in prepared:
        try:
            mse = _gt2_regressor_fold_mse(
                gt2_model, X_tr, y_tr_df, fa_tr, X_val, y_val_true, fa_val,
                top_n_todo, norms, order, l2_reg, basis, cross_pairs,
                n_alpha_planes, km_iterations,
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, ZeroDivisionError):
            # Narrowed from bare Exception: swallow only the numerical failures a bad
            # candidate can legitimately cause; let real bugs surface instead of
            # silently rejecting the step and "stopping improvement".
            return 1e6
        total += mse
        n += 1
    return total / max(n, 1)


def refine_gt2_regressor_antecedents(
    X: pd.DataFrame,
    y: np.ndarray,
    gt2_model: GT2GaussianMixtureModel,
    norms: NormPair,
    top_n_todo: list,
    order: str = "1st",
    l2_reg: float = 1e-6,
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
    n_alpha_planes: int = 5,
    km_iterations: int = 15,
    method: str = "coordinate",
    n_sweeps: int = 3,
    sub_maxfun: int = 20,
    sigma_min_frac: float = 0.02,
    tol: float = 1e-6,
    val_fraction: float = 0.2,
    n_folds: int = 3,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[GT2GaussianMixtureModel, np.ndarray, np.ndarray, dict]:
    """Refine GT2 Gaussian antecedents against held-out regression MSE,
    re-solving TSK consequents in closed form for every candidate.

    Direct GT2 analogue of `it2_refine.refine_it2_regressor_antecedents` --
    see that function's docstring for the parameters shared verbatim.
    ``n_alpha_planes`` is passed through to every `gt2_rule_firing`/
    `gt2_karnik_mendel_tsk` call the search makes.

    Returns
    -------
    (refined_model, corr_terms, y_bucket_mean, info) : the best antecedents
        found (never worse, on cross-validated MSE, than `gt2_model`'s own),
        with consequents re-solved on the *full* training set for those
        antecedents, exactly as `refine_it2_regressor_antecedents` does.
    """
    y_df = pd.DataFrame({"y_value": np.asarray(y, dtype=float)})

    if method not in (None, "none", "coordinate"):
        raise ValueError(f"Unknown refinement method: {method!r}")

    if method in (None, "none"):
        corr_terms, y_bucket_mean, _ = _solve_gt2_consequents(
            gt2_model, X, y_df, top_n_todo, norms, order, l2_reg, basis, cross_pairs, n_alpha_planes,
        )
        return gt2_model, corr_terms, y_bucket_mean, {"init_val_mse": None, "val_mse": None}

    slots = list(_iter_gt2_slots(gt2_model))
    if not slots:
        corr_terms, y_bucket_mean, _ = _solve_gt2_consequents(
            gt2_model, X, y_df, top_n_todo, norms, order, l2_reg, basis, cross_pairs, n_alpha_planes,
        )
        return gt2_model, corr_terms, y_bucket_mean, {"init_val_mse": None, "val_mse": None}

    feature_bounds: dict[str, tuple[float, float, float]] = {}
    for fname in {s[0] for s in slots}:
        col = X[fname].to_numpy(dtype=float)
        lo, hi = float(np.min(col)), float(np.max(col))
        rng = hi - lo if hi > lo else 1.0
        feature_bounds[fname] = (lo, hi, rng)

    folds = _make_folds(len(X), n_folds, val_fraction, seed)
    prepared = _prepare_folds(X, y_df, folds)

    def cv_fitness(model):
        return _gt2_regressor_cv_fitness(
            model, prepared, top_n_todo, norms, order, l2_reg, basis, cross_pairs,
            n_alpha_planes, km_iterations,
        )

    current = gt2_model
    best_model = gt2_model
    best_loss = cv_fitness(gt2_model)
    init_loss = best_loss

    if verbose:
        print(f"\nGT2 regressor coordinate-descent antecedent refinement: {len(slots)} "
              f"memberships, {n_alpha_planes} alpha-planes, "
              f"{n_folds}-fold init val MSE={init_loss:.5f}")

    for sweep in range(n_sweeps):
        sweep_start_loss = best_loss
        for fname, label, idx, gt2_mf in _iter_gt2_slots(current):
            lo, hi, rng = feature_bounds[fname]
            x0, bounds = _slot_x0_and_bounds(gt2_mf, lo, hi, rng, sigma_min_frac)

            def fitness(v, fname=fname, label=label, idx=idx, gt2_mf=gt2_mf):
                new_gt2_mf = _apply_slot_params(gt2_mf, v)
                trial = _replace_slot(current, fname, label, idx, new_gt2_mf)
                return cv_fitness(trial)

            res = _optimizers_sub_solve(fitness, x0, bounds)

            candidate = _replace_slot(current, fname, label, idx, _apply_slot_params(gt2_mf, res.x))
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
        print(f"  GT2 regressor refinement done: val MSE {init_loss:.5f} -> {best_loss:.5f} "
              f"({100 * (init_loss - best_loss) / max(init_loss, 1e-12):.1f}% lower)")

    corr_terms, y_bucket_mean, _ = _solve_gt2_consequents(
        best_model, X, y_df, top_n_todo, norms, order, l2_reg, basis, cross_pairs, n_alpha_planes,
    )
    return best_model, corr_terms, y_bucket_mean, {"init_val_mse": init_loss, "val_mse": best_loss}
