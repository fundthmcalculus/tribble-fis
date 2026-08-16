"""Post-fit antecedent refinement for Interval Type-2 FIS models.

This is the IT2 counterpart of `refine.py`'s block coordinate descent for
Type-1: cycle through one Gaussian sub-membership at a time -- upper or lower
half of one IT2 membership -- and run a small bounded local solve on just its
``(mu, sigma)`` with every other parameter in the model held fixed, repeating
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

Each sub-problem is optimized with bounded L-BFGS-B, which estimates its own
gradient by finite differences (the switch-point search inside `karnik_mendel_tsk`
is not differentiable in closed form -- it is a discrete arg-sort-and-partition,
not a smooth function of `(mu, sigma)` -- so an analytic gradient the way
`refine.py`'s "probability"-norm block coordinate descent has one is not
available here). An L2 penalty pulls each sub-problem back toward its value at
the start of the current sweep, and -- mirroring `refine.py`'s "never return a
model worse than the heuristic start" guarantee -- a candidate is only kept if
it strictly improves the best training loss seen so far.

Regression refinement (optimizing IT2 antecedents against held-out MSE, the way
`refine.py`'s regressor path does) is not implemented: unlike the classifier,
`it2_regressor`'s consequents are fixed at conversion time from the base
Type-1 fit, so refining antecedents alone changes the *inputs* to
`karnik_mendel_tsk` without ever re-solving those consequents for the
candidate antecedents -- a materially bigger undertaking (each fitness
evaluation would need its own closed-form consequent re-solve, as in
`refine.py`'s Type-1 regressor path) left for future work.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .gauss_data import (
    IT2GaussianMixtureModel, IT2FeatureModel, IT2LabelModel, IT2GaussianMembership,
    GaussianMembership, NormPair,
)
from .it2_kernel import it2_firing_strengths


def _iter_it2_gaussian_slots(model: IT2GaussianMixtureModel):
    """Yield ``(feature_name, label, mf_index, IT2GaussianMembership)`` for
    every IT2 Gaussian membership (both the upper and lower half together) in
    a deterministic order.

    Non-Gaussian IT2 memberships (trapezoid, triangular) are skipped: this
    module refines only Gaussian antecedents, matching `IT2_GUIDE.md`'s
    documented scope ("Gaussian memberships only" in v1).

    **Why one slot per membership, not one per half.** An earlier version of
    this module optimized `upper_mf` and `lower_mf` as fully independent
    two-parameter slots. Nothing then stopped the search from moving them
    past each other -- e.g. widening `lower_mf` past `upper_mf` -- which
    breaks the one invariant every caller of this module's output relies on:
    `firing_lower <= firing_upper` pointwise (asserted by
    `it2_kernel.it2_firing_strengths`'s callers and, concretely, required by
    `karnik_mendel_tsk`, which can return `y_l > y_r` -- observed on a real
    fit -- if fed a firing interval that's inverted). The fix keeps `mu`
    *shared* between the two halves and searches
    `(mu, sigma_lower, sigma_upper)` together per membership with
    `sigma_upper >= sigma_lower` enforced by construction (see
    `_apply_slot_params`) -- for two Gaussians sharing a peak, the wider one
    dominates the narrower one at every point, which is exactly the
    "footprint of uncertainty" shape `it2_classifier`/`it2_regressor`'s own
    conversion already builds new memberships in (same `mu`, upper sigma
    wider) at `fit()` time.
    """
    for fname, fmodel in model.feature_models.items():
        for label, lmodel in fmodel.label_models.items():
            for idx, it2_mf in enumerate(lmodel.memberships):
                if isinstance(it2_mf.upper_mf, GaussianMembership) and isinstance(it2_mf.lower_mf, GaussianMembership):
                    yield fname, label, idx, it2_mf


def _slot_x0_and_bounds(it2_mf, lo: float, hi: float, rng: float, sigma_min_frac: float):
    """Initial `(mu, sigma_lower, sigma_upper)` and box bounds for one slot.

    `mu` starts at the upper half's center (the two halves share one `mu` at
    conversion time; if a caller ever hands in a model where they've
    diverged, the upper half's is kept as the anchor since it is the one that
    determines the footprint's outer edge).
    """
    mu0 = it2_mf.upper_mf.mu
    sigma_lower0 = min(it2_mf.lower_mf.sigma, it2_mf.upper_mf.sigma)
    sigma_upper0 = max(it2_mf.lower_mf.sigma, it2_mf.upper_mf.sigma)
    x0 = np.array([mu0, sigma_lower0, sigma_upper0])
    sigma_lo = sigma_min_frac * rng
    bounds = [(lo, hi), (sigma_lo, rng), (sigma_lo, 2.0 * rng)]
    return x0, bounds


def _apply_slot_params(v: np.ndarray, upper_id, lower_id) -> IT2GaussianMembership:
    """Build a fresh `IT2GaussianMembership` from `(mu, sigma_lower, sigma_upper)`,
    clamping `sigma_upper >= sigma_lower` so the invariant `_iter_it2_gaussian_slots`
    documents cannot be violated regardless of what the optimizer proposes.
    """
    mu = float(v[0])
    sigma_lower = max(float(v[1]), 1e-6)
    sigma_upper = max(float(v[2]), sigma_lower)
    return IT2GaussianMembership(
        upper_mf=GaussianMembership(mu=mu, sigma=sigma_upper, id=upper_id),
        lower_mf=GaussianMembership(mu=mu, sigma=sigma_lower, id=lower_id),
    )


def _replace_slot(
    model: IT2GaussianMixtureModel,
    fname: str,
    label: int,
    idx: int,
    new_it2_mf: IT2GaussianMembership,
) -> IT2GaussianMixtureModel:
    """Return a copy of `model` with one IT2 Gaussian membership replaced.

    Every container here (`IT2GaussianMixtureModel`, `IT2FeatureModel`,
    `IT2LabelModel`, `IT2GaussianMembership`) is an immutable `NamedTuple`, so
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
        Number of full passes over every Gaussian sub-membership.
    sub_maxfun : int, default=25
        Function-evaluation budget for each slot's L-BFGS-B sub-problem.
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
        construction (see `_iter_it2_gaussian_slots`).
    """
    if method in (None, "none"):
        return it2_model
    if method != "coordinate":
        raise ValueError(f"Unknown refinement method: {method!r}")

    y_idx = np.asarray(y_labels, dtype=np.intp)
    n_classes = it2_model.n_classes

    slots = list(_iter_it2_gaussian_slots(it2_model))
    if not slots:
        return it2_model

    # Per-feature (mu, sigma) box bounds from the observed data range, shared by
    # every IT2 membership on that feature -- mirrors `refine.py`'s
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
        print(f"\nIT2 coordinate-descent antecedent refinement: {len(slots)} Gaussian "
              f"memberships, init loss={init_loss:.4f}")

    for sweep in range(n_sweeps):
        sweep_start_loss = best_loss
        for fname, label, idx, it2_mf in _iter_it2_gaussian_slots(current):
            lo, hi, rng = feature_bounds[fname]
            x0, bounds = _slot_x0_and_bounds(it2_mf, lo, hi, rng, sigma_min_frac)
            upper_id, lower_id = it2_mf.upper_mf.id, it2_mf.lower_mf.id

            def fitness(v, fname=fname, label=label, idx=idx, upper_id=upper_id, lower_id=lower_id, x0=x0):
                new_it2_mf = _apply_slot_params(v, upper_id, lower_id)
                trial = _replace_slot(current, fname, label, idx, new_it2_mf)
                loss = _cross_entropy_loss(trial, X, y_idx, norms, km_iterations)
                penalty = l2_shrink * float(np.sum((v - x0) ** 2))
                return loss + penalty

            res = minimize(
                fitness, x0, method="L-BFGS-B", bounds=bounds,
                options={"maxfun": sub_maxfun, "maxiter": sub_maxfun},
            )

            candidate = _replace_slot(current, fname, label, idx, _apply_slot_params(res.x, upper_id, lower_id))
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
