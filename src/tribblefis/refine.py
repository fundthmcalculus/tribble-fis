"""Phase 2: post-model antecedent (mu, sigma) refinement via held-out validation.

Refines Gaussian membership parameters to minimize validation MSE. Consequents
solved in closed form per candidate (2*n_params search dimension). Provides:
- `refine_antecedents_de`: SciPy differential evolution (global)
- `refine_antecedents_ga`: genetic algorithm (tournament + BLX-alpha + Gaussian mutation)
- `refine_antecedents_local`: L-BFGS-B local descent

Two optimizers are provided:
- `refine_antecedents_de`  -- global population search via the in-house
                              `optimizers` package (see `_run_optimizer_search`).
- `refine_antecedents_ga`  -- a dependency-light real-coded genetic algorithm
                              (tournament + BLX-alpha crossover + Gaussian
                              mutation + elitism), seeded from the heuristic model.

Both hold out an inner validation fold and guarantee they never return a model
worse (on that fold) than the heuristic starting point.

No `scipy.optimize` is imported directly by this module: every sub-solve routes
through `optimizer_utils.optimizers_sub_solve` / `optimizer_utils.projected_gradient_solve`
(both backed by the in-house `optimizers` package, which itself still imports
scipy internally) or `_run_optimizer_search` below.
"""

import typing

import numpy as np
import pandas as pd

from .gauss_data import (
    GaussianMembership, LabelModel, FeatureModel, GaussianMixtureModel, NormPair, resolve_norm_pair,
    ZERO_FIRING_THRESHOLD,
)
from .gauss_math import tsk_firing_strengths, firing_strengths_and_mf_grad
from .kernel import (
    IncrementalFIS, NotCompilable, compile_model,
    firing_strengths as kernel_firing_strengths,
)
from .optimizer_utils import (
    optimizers_sub_solve as _optimizers_sub_solve,
    projected_gradient_solve as _projected_gradient_solve,
)
from .regression import (
    solve_tsk_consequents, predict_tsk, _mse, _rsquared,
    build_consequent_features, _normalize_firing_strengths,
)

# Thousands of tiny linear solves: single-thread BLAS to avoid spawn overhead.
try:
    from threadpoolctl import threadpool_limits

    def _single_threaded():
        return threadpool_limits(limits=1)
except ImportError:  # threadpoolctl not installed -> no-op (set OMP/OPENBLAS threads=1 manually)
    from contextlib import nullcontext

    def _single_threaded():
        return nullcontext()


# ---------------------------------------------------------------------------
# Model <-> flat parameter vector.
# ---------------------------------------------------------------------------

def _iter_gaussian_slots(model: GaussianMixtureModel):
    """Yield (feature_name, label, membership_index, membership) for every
    GaussianMembership, in a deterministic order shared by extract/apply/bounds."""
    for fname, fmodel in model.feature_models.items():
        for label, lmodel in fmodel.label_models.items():
            for i, mf in enumerate(lmodel.memberships):
                if isinstance(mf, GaussianMembership):
                    yield fname, label, i, mf


def extract_gaussian_params(model: GaussianMixtureModel) -> np.ndarray:
    """Flatten every Gaussian membership to ``[mu_0, sigma_0, mu_1, sigma_1, ...]``."""
    vec = []
    for _, _, _, mf in _iter_gaussian_slots(model):
        vec.extend([mf.mu, mf.sigma])
    return np.asarray(vec, dtype=float)


def apply_gaussian_params(model: GaussianMixtureModel, vec: np.ndarray) -> GaussianMixtureModel:
    """Return a new model with Gaussian (mu, sigma) taken from `vec`.

    Non-Gaussian memberships (e.g. trapezoids) and membership ids are preserved.
    NamedTuples are immutable, so this constructs fresh instances throughout.
    """
    k = 0
    new_feature_models: dict[str, FeatureModel] = {}
    for fname, fmodel in model.feature_models.items():
        new_label_models: dict[int, LabelModel] = {}
        for label, lmodel in fmodel.label_models.items():
            new_mfs = []
            for mf in lmodel.memberships:
                if isinstance(mf, GaussianMembership):
                    mu, sigma = float(vec[k]), float(vec[k + 1])
                    k += 2
                    new_mfs.append(mf._replace(mu=mu, sigma=max(sigma, 1e-6)))
                else:
                    new_mfs.append(mf)
            new_label_models[label] = LabelModel(memberships=new_mfs)
        new_feature_models[fname] = FeatureModel(label_models=new_label_models)
    return model._replace(feature_models=new_feature_models)


def feature_span(col) -> tuple[float, float, float]:
    """``(lo, hi, rng)`` for one feature, widened when the feature is constant.

    A constant column gives ``lo == hi``, and a zero-width interval is not a
    usable box bound. `optimizers` rejects it outright --
    ``InputContinuousVariable`` raises *"lower_bound must be less than
    upper_bound"* -- and even where a solver accepts it, pinning mu to a single
    point makes the sub-solve a no-op that still costs a full evaluation budget.

    Constant columns are ordinary in real data rather than a pathology to assume
    away: RT-IOT2022 ships one (``bwd_URG_flag_count``) among 82 numeric
    features, and a train split of a low-cardinality column can produce one from
    data that is not globally constant. Nothing upstream of here drops them.

    The convention for the degenerate case is the one this function's callers
    already used for sigma -- ``rng = 1.0`` -- now applied to mu as well, giving
    a unit-width interval centred on the constant. Keeping mu and sigma on a
    single convention is the point: the old code guarded ``rng`` on one line and
    left mu with a zero-width bound on the next, which is how the two drifted
    apart in the first place.
    """
    lo, hi = float(np.min(col)), float(np.max(col))
    if hi > lo:
        return lo, hi, hi - lo
    rng = 1.0
    return lo - 0.5 * rng, hi + 0.5 * rng, rng


def build_param_bounds(
    model: GaussianMixtureModel,
    X_train: pd.DataFrame,
    sigma_min_frac: float = 0.02,
    sigma_max_frac: float = 1.0,
) -> list[tuple[float, float]]:
    """Box bounds per parameter: mu within the feature's observed range, sigma in
    ``[sigma_min_frac, sigma_max_frac] * feature_range``."""
    bounds: list[tuple[float, float]] = []
    for fname, _, _, _ in _iter_gaussian_slots(model):
        lo, hi, rng = feature_span(X_train[fname].to_numpy())
        bounds.append((lo, hi))                                  # mu
        bounds.append((sigma_min_frac * rng, sigma_max_frac * rng))  # sigma
    return bounds


# ---------------------------------------------------------------------------
# Fitness: apply candidate antecedents -> closed-form consequents -> val MSE.
# ---------------------------------------------------------------------------

def _prepare_folds(X_train: pd.DataFrame, y_train: pd.DataFrame, folds):
    """Per-fold train/val splits, each with its feature columns pre-extracted once.

    `refine_antecedents_coordinate` (and the other refiners) evaluate the fitness
    tens of thousands of times against these same frames -- only the candidate
    Gaussian parameters vary between calls, `X_train` never does. Building each
    fold's ``{feature_name: ndarray}`` mapping here, instead of inside the fitness
    closure, turns ~84k x n_features pandas column lookups into a handful of
    conversions for the whole refinement run. See issue #42.
    """
    prepared = []
    for tr_idx, val_idx in folds:
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        prepared.append((
            X_tr, y_train.iloc[tr_idx], {c: np.asarray(X_tr[c].to_numpy()) for c in X_tr.columns},
            X_val, y_train.iloc[val_idx]["y_value"].to_numpy(),
            {c: np.asarray(X_val[c].to_numpy()) for c in X_val.columns},
        ))
    return prepared


def _make_kfold_fitness(
    model, X_train, y_train, folds, top_n_todo, n_output_buckets, order, l2_reg, basis, cross_pairs,
    pin_extremes=False, norms: NormPair | None = None, prepared=None,
):
    """Cross-validated fitness: mean held-out MSE over k-folds to avoid overfitting."""
    if prepared is None:
        prepared = _prepare_folds(X_train, y_train, folds)
    y_bucket_mean_dummy = np.zeros(n_output_buckets)  # solver ignores this arg when pin_extremes=False

    def fitness(vec: np.ndarray) -> float:
        candidate = apply_gaussian_params(model, vec)
        total, n = 0.0, 0
        for X_tr, y_tr, fa_tr, X_val, y_val_true, fa_val in prepared:
            try:
                corr, means = solve_tsk_consequents(
                    X_tr, candidate, top_n_todo, y_bucket_mean_dummy, y_tr,
                    n_output_buckets=n_output_buckets, order=order,
                    l2_reg=l2_reg, basis=basis, cross_pairs=cross_pairs, pin_extremes=pin_extremes,
                    norms=norms, feature_arrays=fa_tr, verbose=False,
                )
                y_hat = predict_tsk(
                    X_val, candidate, top_n_todo, means, corr,
                    order=order, basis=basis, cross_pairs=cross_pairs, norms=norms, feature_arrays=fa_val,
                )
            except Exception:
                return 1e6
            keep = ~np.isnan(y_hat)
            if not np.any(keep):
                return 1e6
            total += _mse(y_val_true[keep], y_hat[keep])
            n += 1
        return total / max(n, 1)

    return fitness


def _make_folds(n_samples, n_folds, val_fraction, random_state):
    """k-fold indices when n_folds > 1, else a single holdout split."""
    idx = np.arange(n_samples)
    if n_folds <= 1:
        from sklearn.model_selection import train_test_split
        tr_idx, val_idx = train_test_split(idx, test_size=val_fraction, random_state=random_state)
        return [(tr_idx, val_idx)]
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(kf.split(idx))


# ---------------------------------------------------------------------------
# Differential evolution.
# ---------------------------------------------------------------------------

def refine_antecedents_de(
    model: GaussianMixtureModel,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    top_n_todo: list[typing.Any],
    n_output_buckets: int,
    order: str = "full-2nd",
    l2_reg: float = 1e-2,
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
    val_fraction: float = 0.2,
    n_folds: int = 3,
    maxiter: int = 40,
    popsize: int = 8,
    seed: int = 42,
) -> tuple[GaussianMixtureModel, dict]:
    """Refine antecedents with a global population search via the in-house
    `optimizers` package (`_run_optimizer_search`; this used to run SciPy
    differential evolution).

    `maxiter`/`popsize` are kept for backward compatibility, mapped onto
    `_run_optimizer_search`'s `num_generations`/`population_size`, with a
    genuinely global (not localized) search box -- `local_scale=None` --
    matching DE's original unlocalized behavior.

    Returns (refined_model, info) where info has the initial/final validation MSE.
    Never returns a model worse than the heuristic start on the CV fitness.
    """
    folds = _make_folds(len(X_train), n_folds, val_fraction, seed)
    fitness = _make_kfold_fitness(model, X_train, y_train, folds, top_n_todo,
                                  n_output_buckets, order, l2_reg, basis, cross_pairs)
    bounds = build_param_bounds(model, X_train)
    x0 = np.clip(extract_gaussian_params(model),
                 [b[0] for b in bounds], [b[1] for b in bounds])

    print(f"\nDE-replacement (GA) antecedent refinement: {len(bounds)} params, order={order}")
    with _single_threaded():
        best_x, best_fit, info = _run_optimizer_search(
            fitness, bounds, x0, method="ga", local_grad_optim="single-var-grad",
            population_size=popsize, num_generations=maxiter,
            local_scale=None, seed=seed, label="antecedents-de",
        )
    return apply_gaussian_params(model, best_x), {
        "init_val_mse": info["init_fit"], "val_mse": best_fit,
    }


# ---------------------------------------------------------------------------
# Local gradient refinement (ANFIS-style) from the heuristic start.
# ---------------------------------------------------------------------------

def refine_antecedents_local(
    model: GaussianMixtureModel,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    top_n_todo: list[typing.Any],
    n_output_buckets: int,
    order: str = "full-2nd",
    l2_reg: float = 1e-2,
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
    val_fraction: float = 0.2,
    n_folds: int = 3,
    maxiter: int = 80,
    maxfun: int = 15000,
    seed: int = 42,
) -> tuple[GaussianMixtureModel, dict]:
    """Local descent from the heuristic start via `optimizer_utils.optimizers_sub_solve`
    (previously SciPy's L-BFGS-B). Kept for comparison; `refine_antecedents_coordinate`
    is the recommended default at this scale. `maxiter`/`maxfun` are kept for backward
    compatibility but unused. Never returns a model worse than the heuristic start on
    the CV fitness.
    """
    folds = _make_folds(len(X_train), n_folds, val_fraction, seed)
    fitness = _make_kfold_fitness(model, X_train, y_train, folds, top_n_todo,
                                  n_output_buckets, order, l2_reg, basis, cross_pairs)
    bounds = build_param_bounds(model, X_train)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0 = np.clip(extract_gaussian_params(model), lo, hi)
    init_fit = fitness(x0)

    print(f"\nLocal antecedent refinement: {len(bounds)} params, "
          f"order={order}, {n_folds}-fold init val MSE={init_fit:.5f}")
    with _single_threaded():
        result = _optimizers_sub_solve(fitness, x0, bounds)

    best_x, best_fit = (result.x, result.fun) if result.fun <= init_fit else (x0, init_fit)
    if result.fun > init_fit:
        print("  Local refine did not beat the heuristic start; keeping heuristic.")
    print(f"  Local refine done: val MSE {init_fit:.5f} -> {best_fit:.5f} "
          f"({100 * (init_fit - best_fit) / max(init_fit, 1e-12):.1f}% lower)")
    return apply_gaussian_params(model, best_x), {"init_val_mse": init_fit, "val_mse": best_fit}


# ---------------------------------------------------------------------------
# Analytic gradient for one Gaussian (mu, sigma) block: bilevel derivative
# (consequents re-solved per candidate, envelope theorem applies).
# Only supported with "probability" t-norms (smooth everywhere).
# ---------------------------------------------------------------------------

def _analytic_block_supported(norms: NormPair, pin_extremes: bool, block: int) -> bool:
    return (
        block == 2
        and not pin_extremes
        and norms.t_norm == "probability"
        and norms.t_conorm == "probability"
    )


def _design_matrix(feature_arrays, top_n_todo, F, order, basis, cross_pairs):
    """Stacked per-rule design ``[norm_fs_r * [1 | basis(X)]]_r``, plus the
    ingredients (`phi`, `norm_fs`) the gradient needs to reuse."""
    X_rule = np.column_stack([feature_arrays[c] for c in top_n_todo]) if top_n_todo \
        else np.empty((F.shape[0], 0))
    feats = build_consequent_features(X_rule, order, basis=basis, cross_pairs=cross_pairs)
    phi = np.hstack([np.ones((X_rule.shape[0], 1)), feats])
    norm_fs = _normalize_firing_strengths(F)
    n_rules = F.shape[1]
    design = (norm_fs[:, :, np.newaxis] * phi[:, np.newaxis, :]).reshape(X_rule.shape[0], n_rules * phi.shape[1])
    return design, phi, norm_fs


def _norm_fs_grad(F: np.ndarray, r0: int, dF_r0: np.ndarray) -> np.ndarray:
    """d(row-normalized firing strengths)/dtheta given the derivative of only the
    target rule's raw firing-strength column.

    Every other raw column is a constant in theta (each output label owns
    independent Gaussian antecedents -- see `firing_strengths_and_mf_grad`); only
    the shared row-sum denominator couples ``norm_fs[:, r0]``'s change into every
    other column. Zero-firing rows keep the same all-zero convention as
    `_normalize_firing_strengths` (their derivative is likewise zero).
    """
    row_sums = F.sum(axis=1)
    valid = row_sums > ZERO_FIRING_THRESHOLD
    safe_s = np.where(row_sums > 0, row_sums, 1.0)
    d_norm = -F * dF_r0[:, np.newaxis] / safe_s[:, np.newaxis] ** 2
    d_norm[:, r0] += dF_r0 / safe_s
    d_norm[~valid, :] = 0.0
    return d_norm


def _fold_mse_and_grad(
    fa_tr, y_tr, fa_val, y_val_true, candidate, top_n_todo, order, l2_reg, basis, cross_pairs,
    target_feature, target_label, target_mf_index,
):
    """(val MSE, [d(val MSE)/d(mu), d(val MSE)/d(sigma)]) for one fold, at
    `candidate`'s current parameters for the targeted Gaussian.

    ``beta* = argmin_beta ||Phi_tr @ beta - y_tr||^2 + l2_reg * ||D^(1/2) beta||^2``
    is solved once here (mirroring `solve_tsk_consequents`'s unconstrained ridge
    branch -- this path only ever runs with `pin_extremes=False`), then
    differentiated via the normal-equation identity
    ``dbeta*/dtheta = A^-1 [(dPhi_tr/dtheta)^T r - Phi_tr^T (dPhi_tr/dtheta) beta*]``
    with ``A = Phi_tr^T Phi_tr + l2_reg * D`` and ``r = y_tr - Phi_tr @ beta*``.
    """
    F_tr, labels, dF_tr_mu, dF_tr_sigma = firing_strengths_and_mf_grad(
        fa_tr, candidate, target_feature, target_label, target_mf_index
    )
    r0 = labels.index(target_label)
    design_tr, phi_tr, norm_fs_tr = _design_matrix(fa_tr, top_n_todo, F_tr, order, basis, cross_pairs)
    n_rules, n_coeffs = F_tr.shape[1], phi_tr.shape[1]

    y = np.asarray(y_tr["y_value"].values, dtype=float)
    penalty = np.ones(n_rules * n_coeffs)
    penalty[::n_coeffs] = 0.0  # never penalize each rule's intercept/bucket-mean column

    # beta*, dbeta*/dmu and dbeta*/dsigma are three ridge solves against the SAME
    # normal-equation matrix A = Phi^T Phi + l2_reg * D. Solve them against A with
    # np.linalg.solve (LU) rather than a fresh SVD-lstsq each; the beta solve also
    # drops the (N+p) x p augmented matrix the old ridge path assembled. On a
    # singular fold (e.g. rank-deficient unpenalized intercept columns) LU raises
    # and we fall back to the exact previous lstsq behaviour, so degenerate folds
    # are unchanged. numpy-only by design (no scipy); see issue #177.
    A = design_tr.T @ design_tr + l2_reg * np.diag(penalty)
    try:
        beta = np.linalg.solve(A, design_tr.T @ y)
    except np.linalg.LinAlgError:
        if l2_reg > 0:
            sqrt_pen = np.sqrt(l2_reg * penalty)
            beta = np.linalg.lstsq(
                np.vstack([design_tr, np.diag(sqrt_pen)]), np.hstack([y, np.zeros_like(sqrt_pen)]), rcond=None
            )[0]
        else:
            beta = np.linalg.lstsq(design_tr, y, rcond=None)[0]
    resid = y - design_tr @ beta

    F_val, _, dF_val_mu, dF_val_sigma = firing_strengths_and_mf_grad(
        fa_val, candidate, target_feature, target_label, target_mf_index
    )
    design_val, phi_val, norm_fs_val = _design_matrix(fa_val, top_n_todo, F_val, order, basis, cross_pairs)
    y_hat_val = design_val @ beta
    if not np.all(np.isfinite(y_hat_val)):
        raise FloatingPointError("non-finite prediction")
    mse = float(np.mean((y_val_true - y_hat_val) ** 2))

    # Both gradient components solve against the same A, so assemble their
    # right-hand sides first and solve the pair in one factorization.
    rhs_cols, dPhi_val_cols = [], []
    for dF_tr, dF_val in ((dF_tr_mu, dF_val_mu), (dF_tr_sigma, dF_val_sigma)):
        dPhi_tr = (_norm_fs_grad(F_tr, r0, dF_tr)[:, :, np.newaxis] * phi_tr[:, np.newaxis, :]).reshape(design_tr.shape)
        rhs_cols.append(dPhi_tr.T @ resid - design_tr.T @ (dPhi_tr @ beta))
        dPhi_val_cols.append(
            (_norm_fs_grad(F_val, r0, dF_val)[:, :, np.newaxis] * phi_val[:, np.newaxis, :]).reshape(design_val.shape)
        )
    rhs = np.column_stack(rhs_cols)
    try:
        dbeta = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        dbeta = np.linalg.lstsq(A, rhs, rcond=None)[0]

    grads = []
    for k, dPhi_val in enumerate(dPhi_val_cols):
        dyhat = dPhi_val @ beta + design_val @ dbeta[:, k]
        grads.append(float(np.mean(2.0 * (y_hat_val - y_val_true) * dyhat)))

    return mse, np.array(grads)


# ---------------------------------------------------------------------------
# Per-variable (block) coordinate descent.
#
# `_optimizers_sub_solve` and `_projected_gradient_solve` (imported above from
# `optimizer_utils`) are the two sub-solve backends used throughout this
# module and `it2_refine.py`/`gt2_refine.py`/`trapz_math.py`/`regression.py`.
# ---------------------------------------------------------------------------

def refine_antecedents_coordinate(
    model: GaussianMixtureModel,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    top_n_todo: list[typing.Any],
    n_output_buckets: int,
    order: str = "full-2nd",
    l2_reg: float = 1e-2,
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
    val_fraction: float = 0.2,
    n_folds: int = 3,
    n_sweeps: int = 3,
    block: int = 2,
    sub_maxfun: int = 25,
    tol: float = 1e-5,
    seed: int = 42,
    norms: NormPair | None = None,
) -> tuple[GaussianMixtureModel, dict]:
    """Refine antecedents by *sequential* per-variable (block) coordinate descent.

    This is the recommended default. It scales to the larger membership models
    (~2*n_MF parameters) far better than a single high-dimensional L-BFGS-B solve:
    each finite-difference gradient of the full solve costs one evaluation per
    parameter, so on a non-smooth objective it burns thousands of evaluations,
    whereas cycling one membership function at a time keeps every sub-problem tiny.
    On concrete (138 params) it reaches essentially the same test R^2 as the full
    L-BFGS-B solve using ~2.3x fewer fitness evaluations.

    Rather than optimize all ~2*n_MF parameters at once -- which forces L-BFGS-B to
    spend one full (2*n_MF)-evaluation finite-difference gradient per step on a
    non-smooth objective -- this cycles through one membership function at a time
    and optimizes just its ``(mu, sigma)`` (a `block`=2 sub-problem) with everything
    else held fixed, repeating for `n_sweeps` passes. Each sub-problem is a cheap,
    low-dimensional local solve, so the total number of fitness evaluations is far
    smaller for comparable quality. `block=1` gives pure scalar coordinate descent.

    ``norms``: when both halves resolve to "probability", each `block=2`
    sub-problem is solved with the analytic bilevel gradient (issue #43) instead
    of L-BFGS-B's default finite-difference estimate, so `sub_maxfun` buys more
    optimizer iterations per fitness evaluation rather than more finite-difference
    evaluations. `None` (the default family, "min/max") keeps the previous
    finite-difference behavior unchanged -- the analytic gradient is only valid
    for a norm family that is smooth everywhere.

    Never returns a model worse than the heuristic start on the CV fitness (the
    running best is only ever updated on a strict improvement).
    """
    folds = _make_folds(len(X_train), n_folds, val_fraction, seed)
    prepared = _prepare_folds(X_train, y_train, folds)
    fitness = _make_kfold_fitness(model, X_train, y_train, folds, top_n_todo,
                                  n_output_buckets, order, l2_reg, basis, cross_pairs,
                                  norms=norms, prepared=prepared)
    bounds = build_param_bounds(model, X_train)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    n_params = len(bounds)

    x = np.clip(extract_gaussian_params(model), lo, hi)
    init_fit = fitness(x)
    cur = init_fit
    n_eval = [1]  # count fitness calls for reporting

    resolved_norms = norms if norms is not None else resolve_norm_pair()
    slots = list(_iter_gaussian_slots(model))
    analytic_ok = _analytic_block_supported(resolved_norms, pin_extremes=False, block=block)

    n_blocks = (n_params + block - 1) // block
    print(f"\nCoordinate-descent antecedent refinement: {n_params} params "
          f"({n_blocks} blocks of {block}), order={order}, {n_folds}-fold "
          f"init val MSE={init_fit:.5f}" + (" (analytic gradient)" if analytic_ok else ""))

    with _single_threaded():
        for sweep in range(n_sweeps):
            prev = cur
            for b in range(n_blocks):
                idx = np.arange(b * block, min((b + 1) * block, n_params))
                sub_bounds = [(lo[k], hi[k]) for k in idx]

                if analytic_ok and b < len(slots):
                    target_feature, target_label, target_mf_index, _mf = slots[b]

                    def f_sub_grad(v, idx=idx, target_feature=target_feature,
                                    target_label=target_label, target_mf_index=target_mf_index):
                        trial = x.copy()
                        trial[idx] = v
                        n_eval[0] += 1
                        candidate = apply_gaussian_params(model, trial)
                        total_f, total_g, n_ok = 0.0, np.zeros(2), 0
                        for X_tr, y_tr, fa_tr, X_val, y_val_true, fa_val in prepared:
                            try:
                                f_i, g_i = _fold_mse_and_grad(
                                    fa_tr, y_tr, fa_val, y_val_true, candidate, top_n_todo,
                                    order, l2_reg, basis, cross_pairs,
                                    target_feature, target_label, target_mf_index,
                                )
                            except Exception:
                                return 1e6, np.zeros(2)
                            total_f += f_i
                            total_g += g_i
                            n_ok += 1
                        return total_f / max(n_ok, 1), total_g / max(n_ok, 1)

                    # This branch's whole point is exploiting the closed-form
                    # bilevel gradient (issue #43), and `optimizers`' local
                    # solve has no way to accept a supplied Jacobian, so it
                    # runs through `_projected_gradient_solve` instead of
                    # `_optimizers_sub_solve` -- an in-house projected-gradient
                    # descent that still uses the exact gradient (and, unlike
                    # `_optimizers_sub_solve`, keeps an exact evaluation cap).
                    res = _projected_gradient_solve(f_sub_grad, x[idx], sub_bounds, max_evals=sub_maxfun)
                else:
                    def f_sub(v, idx=idx):
                        trial = x.copy()
                        trial[idx] = v
                        n_eval[0] += 1
                        return fitness(trial)

                    res = _optimizers_sub_solve(f_sub, x[idx], sub_bounds)
                # This accept-on-strict-improvement test is sensitive to sub-1e-12
                # noise in `res.fun`: a solver change that shifts a fitness by even
                # 1e-13 can flip one accept/reject here, and the trajectory then
                # diverges to a different local optimum (final R^2 drift up to ~1e-3
                # observed when the ridge solve in `_fold_mse_and_grad` moved from
                # SVD-lstsq to LU). The refinement is therefore reproducible only
                # against a fixed numerical stack (BLAS, numpy, hardware); see #177.
                if res.fun < cur - 1e-12:
                    x[idx] = np.clip(res.x, lo[idx], hi[idx])
                    cur = float(res.fun)
            print(f"  sweep {sweep + 1}/{n_sweeps}: val MSE={cur:.5f} (evals={n_eval[0]})")
            if prev - cur < tol:
                break

    print(f"  Coordinate descent done: val MSE {init_fit:.5f} -> {cur:.5f} "
          f"({100 * (init_fit - cur) / max(init_fit, 1e-12):.1f}% lower, {n_eval[0]} evals)")
    return apply_gaussian_params(model, x), {"init_val_mse": init_fit, "val_mse": cur, "n_eval": n_eval[0]}


# ---------------------------------------------------------------------------
# Real-coded genetic algorithm (dependency-light).
# ---------------------------------------------------------------------------

def refine_antecedents_ga(
    model: GaussianMixtureModel,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    top_n_todo: list[typing.Any],
    n_output_buckets: int,
    order: str = "full-2nd",
    l2_reg: float = 1e-2,
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
    val_fraction: float = 0.2,
    n_folds: int = 3,
    n_generations: int = 60,
    pop_size: int = 60,
    elite_frac: float = 0.1,
    tournament_k: int = 3,
    crossover_alpha: float = 0.5,
    mutation_rate: float = 0.15,
    mutation_scale: float = 0.1,
    seed: int = 42,
) -> tuple[GaussianMixtureModel, dict]:
    """Refine antecedents with a real-coded GA seeded from the heuristic model.

    Tournament selection, BLX-alpha crossover, Gaussian mutation (scaled by each
    parameter's box width), and elitism. The heuristic solution is injected into
    the initial population so the GA can only improve on it. Fitness is the mean
    held-out MSE over `n_folds` folds to prevent overfitting a single fold.
    """
    folds = _make_folds(len(X_train), n_folds, val_fraction, seed)
    fitness = _make_kfold_fitness(model, X_train, y_train, folds, top_n_todo,
                                  n_output_buckets, order, l2_reg, basis, cross_pairs)
    bounds = build_param_bounds(model, X_train)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    width = hi - lo
    n_params = len(bounds)

    rng = np.random.default_rng(seed)
    x0 = np.clip(extract_gaussian_params(model), lo, hi)

    # Initial population: heuristic seed + jittered copies + uniform-random fill.
    pop = rng.uniform(lo, hi, size=(pop_size, n_params))
    pop[0] = x0
    n_jitter = max(1, pop_size // 4)
    pop[1:1 + n_jitter] = np.clip(
        x0 + rng.normal(0, 0.1, size=(n_jitter, n_params)) * width, lo, hi
    )

    def evaluate(P):
        return np.array([fitness(ind) for ind in P])

    fit = evaluate(pop)
    init_best = fit.min()
    n_elite = max(1, int(elite_frac * pop_size))
    print(f"\nGA antecedent refinement: {n_params} params, pop={pop_size}, "
          f"gens={n_generations}, order={order}, init val MSE={init_best:.5f}")

    for gen in range(n_generations):
        order_idx = np.argsort(fit)
        elites = pop[order_idx[:n_elite]].copy()
        elite_fit = fit[order_idx[:n_elite]].copy()

        children = []
        while len(children) < pop_size - n_elite:
            # Tournament selection of two parents.
            def pick():
                cand = rng.integers(0, pop_size, size=tournament_k)
                return pop[cand[np.argmin(fit[cand])]]
            p1, p2 = pick(), pick()
            # BLX-alpha crossover.
            cmin = np.minimum(p1, p2)
            cmax = np.maximum(p1, p2)
            span = cmax - cmin
            child = rng.uniform(cmin - crossover_alpha * span, cmax + crossover_alpha * span)
            # Gaussian mutation.
            mask = rng.random(n_params) < mutation_rate
            child[mask] += rng.normal(0, mutation_scale, size=mask.sum()) * width[mask]
            children.append(np.clip(child, lo, hi))

        pop = np.vstack([elites, np.array(children)])
        fit = np.concatenate([elite_fit, evaluate(pop[n_elite:])])

        if (gen + 1) % 10 == 0 or gen == n_generations - 1:
            print(f"  gen {gen + 1:3d}: best val MSE={fit.min():.5f}")

    best_idx = int(np.argmin(fit))
    best_x, best_fit = (pop[best_idx], fit[best_idx]) if fit[best_idx] <= init_best else (x0, init_best)
    print(f"  GA done: val MSE {init_best:.5f} -> {best_fit:.5f} "
          f"({100 * (init_best - best_fit) / max(init_best, 1e-12):.1f}% lower)")
    return apply_gaussian_params(model, best_x), {"init_val_mse": init_best, "val_mse": best_fit}


# ---------------------------------------------------------------------------
# Population + local-polish search via the `optimizers` package.
# ---------------------------------------------------------------------------
#
# The empirical lesson from the earlier DE/GA experiments (see the project
# memory) is that a *global* population search overfits the CV/validation
# estimate, and the only part of DE that actually helped was its L-BFGS
# ``polish`` -- i.e. the local move. The `optimizers` package
# (github.com/fundthmcalculus/optimizers) folds that local move directly into
# every population member: with ``local_grad_optim="single-var-grad"`` each GA
# child / ACO ant / PSO particle is polished by a per-variable gradient descent
# before it competes. That gives us "population diversity + local polish" in one
# optimizer instead of bolting a separate polish onto a global search.
#
# We keep the two guard-rails that made the earlier refinements trustworthy:
#   1. the heuristic solution is *seeded* into (and preserved in) the archive, so
#      the optimizer starts from -- and can never score worse than -- the
#      heuristic; and
#   2. the search box is optionally *localized* around the heuristic
#      (``local_scale``) so the polish-driven population stays in the heuristic's
#      basin rather than wandering into overfit territory.
#
# Reproducibility, and why the `optimizers` pin has a floor
# ---------------------------------------------------------
# `_run_optimizer_search` calls `set_seed(seed)` so that a refinement is a pure
# function of its `seed`. Whether that actually holds is a property of the
# installed `optimizers` revision, not of this file:
#
#   * Before optimizers 3a57f91, `InputContinuousVariable.initial_random_value`
#     fell back to `np.random.default_rng()` with **no argument** -- fresh OS
#     entropy -- so the initial population ignored `set_seed` entirely. Every
#     call returned a different model. Measured here on a fixed 120-row problem
#     with `seed=0`: eight identical calls produced three different validation
#     accuracies (0.8000 / 0.8083 / 0.8167). That is also what made
#     `tests/test_classifier_refine.py` flaky, roughly one run in six.
#   * From 3a57f91 the initial population draws from the package's seeded
#     generator, and the same eight calls agree exactly.
#
# `uv.lock` therefore must not be rolled back past that commit; the constraint
# is restated next to the git source in `pyproject.toml`, which is where someone
# re-pinning would be looking.
#
# One caveat remains upstream: `optimizers` is only reproducible at `n_jobs=1`,
# because above that its parallel workers share a single
# `numpy.random.Generator`, which is not thread-safe and hands out draws in
# scheduler order. `_run_optimizer_search` passes `n_jobs=1` -- for an unrelated
# reason, that the fitness closure is not picklable -- so this path is safe
# today, but a future change to that argument would silently reintroduce the
# nondeterminism. See fundthmcalculus/optimizers#100.

_OPTIMIZER_METHODS = ("ga", "pso", "aco", "multi")


def _localized_bounds(
    bounds: list[tuple[float, float]], x0: np.ndarray, local_scale: float | None,
) -> list[tuple[float, float]]:
    """Intersect the global box `bounds` with a box of half-width
    ``local_scale * width`` centred on `x0`.

    ``local_scale=None`` (or a non-positive value) returns the global bounds
    unchanged (a genuinely global search). A small value (e.g. 0.25) keeps the
    population near the heuristic, which is what reliably improves *test* error.
    """
    if local_scale is None or local_scale <= 0:
        return list(bounds)
    out: list[tuple[float, float]] = []
    for (lo, hi), c in zip(bounds, x0):
        half = local_scale * (hi - lo)
        out.append((max(lo, c - half), min(hi, c + half)))
    return out


def _run_optimizer_search(
    fitness: typing.Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    x0: np.ndarray,
    *,
    method: str = "ga",
    local_grad_optim: str = "single-var-grad",
    population_size: int = 40,
    num_generations: int = 25,
    stop_after_iterations: int = 8,
    local_scale: float | None = 0.25,
    seed: int = 42,
    label: str = "antecedents",
) -> tuple[np.ndarray, float, dict]:
    """Minimise `fitness` over box `bounds`, seeded from `x0`, using the
    `optimizers` package (population search + per-member local gradient polish).

    Returns ``(best_x, best_fit, info)``. Guarantees ``best_fit <= fitness(x0)``
    by seeding and preserving the heuristic in the solution archive and by an
    explicit fallback comparison at the end.
    """
    if method not in _OPTIMIZER_METHODS:
        raise ValueError(f"method={method!r} not in {_OPTIMIZER_METHODS}")

    # Imported lazily so the rest of the module works without the optional dep.
    from optimizers import (
        GeneticAlgorithmOptimizer, GeneticAlgorithmOptimizerConfig,
        ParticleSwarmOptimizer, ParticleSwarmOptimizerConfig,
        AntColonyOptimizer, AntColonyOptimizerConfig,
        MultiTypeOptimizer, IOptimizerConfig,
        set_seed,
    )
    from optimizers.continuous.variables import InputContinuousVariable
    from optimizers.solution_deck import SolutionDeck

    set_seed(seed)

    search_bounds = _localized_bounds(bounds, x0, local_scale)
    lo = np.array([b[0] for b in search_bounds])
    hi = np.array([b[1] for b in search_bounds])
    x0c = np.clip(x0, lo, hi)
    n = len(search_bounds)

    variables = [
        InputContinuousVariable(f"p{i}", float(lo[i]), float(hi[i]))
        for i in range(n)
    ]

    # The optimizers minimise ``fcn(x)``; wrap so out-of-the-loop exceptions in
    # the fuzzy forward pass never crash a whole generation.
    def fcn(x):
        try:
            return float(fitness(np.asarray(x, dtype=float)))
        except Exception:
            return 1e6

    init_fit = fcn(x0c)

    # Seed the heuristic into an archive and preserve it (row 0) so the search
    # can only improve on the heuristic.
    archive_size = max(population_size * 2, n * 2, 30)
    deck = SolutionDeck(archive_size=archive_size, num_vars=n)
    deck.solution_archive[0] = x0c
    deck.solution_value[0] = init_fit
    deck.is_local_optima[0] = False
    preserve = 1.0 / archive_size

    common = dict(
        name=f"{method}-{label}",
        num_generations=num_generations,
        population_size=population_size,
        solution_archive_size=archive_size,
        stop_after_iterations=stop_after_iterations,
        n_jobs=1,                 # fitness closure is not picklable; stay single-process
        joblib_prefer="threads",
        local_grad_optim=local_grad_optim,
    )

    print(f"\n{method.upper()} ({local_grad_optim}) {label} refinement: {n} params, "
          f"pop={population_size}, gens={num_generations}, "
          f"local_scale={local_scale}, init fitness={init_fit:.5f}")

    with _single_threaded():
        if method == "ga":
            opt = GeneticAlgorithmOptimizer(
                config=GeneticAlgorithmOptimizerConfig(**common), fcn=fcn,
                variables=variables, existing_soln_deck=deck)
        elif method == "pso":
            opt = ParticleSwarmOptimizer(
                config=ParticleSwarmOptimizerConfig(**common), fcn=fcn,
                variables=variables, existing_soln_deck=deck)
        elif method == "aco":
            opt = AntColonyOptimizer(
                config=AntColonyOptimizerConfig(**common), fcn=fcn,
                variables=variables, existing_soln_deck=deck)
        else:  # "multi"
            opt = MultiTypeOptimizer(
                config=IOptimizerConfig(**common), fcn=fcn, variables=variables)
        result = opt.solve(preserve_percent=preserve)

    best_x = np.clip(np.asarray(result.solution_vector, dtype=float), lo, hi)
    best_fit = float(result.solution_score)
    if best_fit > init_fit:      # never worse than the heuristic start
        best_x, best_fit = x0c, init_fit
        print("  optimizer did not beat the heuristic start; keeping heuristic.")
    print(f"  {method.upper()} done: fitness {init_fit:.5f} -> {best_fit:.5f} "
          f"({100 * (init_fit - best_fit) / max(init_fit, 1e-12):.1f}% lower, "
          f"stop={result.stop_reason})")
    return best_x, best_fit, {
        "init_fit": init_fit, "fit": best_fit, "stop_reason": result.stop_reason,
        "generations": result.generations_completed,
    }


def refine_antecedents_optimizers(
    model: GaussianMixtureModel,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    top_n_todo: list[typing.Any],
    n_output_buckets: int,
    order: str = "full-2nd",
    l2_reg: float = 1e-2,
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
    val_fraction: float = 0.2,
    n_folds: int = 3,
    method: str = "ga",
    local_grad_optim: str = "single-var-grad",
    population_size: int = 40,
    num_generations: int = 25,
    local_scale: float | None = 0.25,
    seed: int = 42,
) -> tuple[GaussianMixtureModel, dict]:
    """Refine the *regressor* antecedents with the `optimizers` package.

    Same closed-form-consequent CV fitness as the other regressor refiners, but
    the search is a population optimizer whose members are each locally polished
    (``local_grad_optim``). Localised around the heuristic (``local_scale``) and
    seeded from it, so it keeps the productive local move without the overfit-
    prone global wandering. Never returns a model worse than the heuristic on CV.
    """
    folds = _make_folds(len(X_train), n_folds, val_fraction, seed)
    fitness = _make_kfold_fitness(model, X_train, y_train, folds, top_n_todo,
                                  n_output_buckets, order, l2_reg, basis, cross_pairs)
    bounds = build_param_bounds(model, X_train)
    x0 = np.clip(extract_gaussian_params(model),
                 [b[0] for b in bounds], [b[1] for b in bounds])
    best_x, best_fit, info = _run_optimizer_search(
        fitness, bounds, x0, method=method, local_grad_optim=local_grad_optim,
        population_size=population_size, num_generations=num_generations,
        local_scale=local_scale, seed=seed, label=f"regressor-{order}",
    )
    return apply_gaussian_params(model, best_x), {
        "init_val_mse": info["init_fit"], "val_mse": info["fit"], **info,
    }


# ---------------------------------------------------------------------------
# Classifier antecedent refinement.
# ---------------------------------------------------------------------------
#
# A zeroth-order TSK *classifier* has no consequents: the predicted class is
# ``argmax`` of the per-class firing strengths, which are a pure function of the
# Gaussian ``(mu, sigma)`` antecedents. So refining the antecedents *is* the whole
# model -- there is nothing else to learn -- and the heuristic (KMeans +
# ``norm.fit`` per class) only ever fits each feature/label marginal, never the
# discriminative objective. Tuning ``(mu, sigma)`` against a classification loss
# is therefore directly worthwhile.
#
# Overfitting control: because there are no per-fold consequents to refit, a
# k-fold "held-out" score of a single shared parameter vector reduces to the
# full-training score and provides no real held-out signal. Instead we (a) add an
# L2 shrinkage penalty pulling the parameters toward the heuristic start x0
# (ridge / early-stopping-like), (b) do local descent from x0, and (c) select the
# final model on a held-out validation split, keeping the heuristic if the refined
# model does not improve validation loss.


def _normalize_proba(fs: np.ndarray, n_labels: int) -> np.ndarray:
    """Row-normalise firing strengths, with zero-firing rows falling back to
    uniform -- the same rule as ``TribbleClassifier.predict_proba``.

    Written as a masked ``divide`` rather than boolean fancy-indexing
    (``proba[nz] = fs[nz] / row[nz]``), which materialised three whole-array
    temporaries and made two index passes. Identical arithmetic, and it profiled
    at a third of the wide refinement's total runtime before the change.
    """
    row = fs.sum(axis=1, keepdims=True)
    proba = np.full_like(fs, 1.0 / max(n_labels, 1))
    np.divide(fs, row, out=proba, where=row > 0)
    return proba


def _cross_entropy_from_strengths(fs: np.ndarray, y_idx: np.ndarray, n_labels: int) -> float:
    """``_cross_entropy(_normalize_proba(fs, n_labels), y_idx)`` without the matrix.

    The cross-entropy only ever reads one probability per row -- the true
    class's -- so normalising all ``n * L`` of them to throw away all but ``n`` is
    wasted work in a function called thousands of times per refinement. Each
    step is the same floating-point operation on the same operands as the two-call
    form, so the value is bit-identical.

    For the refinement hot loop use :class:`_CrossEntropy` instead, which hoists
    the row-index gather and the scratch buffers out of the call.
    """
    return _CrossEntropy(y_idx, n_labels, fs.shape)(fs)


# Historically, per-solver option spelling and gradient support for five
# distinct SciPy methods (`sub_method` picked one). Measured back then: SLSQP
# 1.14x over L-BFGS-B at equal accuracy, and 1.95x with the gradient; Powell
# was slightly more accurate but 1.5x slower; TNC was 3x slower (see
# `refine_classifier_antecedents`'s docstring).
#
# `_sub_solve` no longer calls scipy at all: there is exactly one backend per
# whether a gradient is supplied (see below), so `jac` -- not `method` --
# now decides which runs. `_SUB_SOLVERS` is kept only to validate `sub_method`
# against the same five names (so a typo still raises the same error) and to
# look up whether the *requested* method supports a gradient at all -- e.g.
# `sub_method="Powell"` with `analytic_gradient=True` still finite-differences,
# matching the old scipy behavior where Powell silently ignored `jac`.
_SUB_SOLVERS: dict[str, dict[str, typing.Any]] = {
    "L-BFGS-B":    {"jac": True},
    "SLSQP":       {"jac": True},
    "TNC":         {"jac": True},
    "Powell":      {"jac": False},
    "Nelder-Mead": {"jac": False},
}


def _sub_solve(method: str, fun, x0, bounds, budget: int, jac: bool):
    """Run one bounded sub-problem, in-house rather than via `scipy.optimize`.

    `method` (`sub_method` at the public API) is retained for backward
    compatibility and is still validated against the same five names, but no
    longer selects a distinct algorithm -- there is exactly one non-scipy
    backend per whether a gradient is supplied:
      - a gradient is supplied and `method` supports one:
        `_projected_gradient_solve` (an in-house projected-gradient descent
        with an exact evaluation budget of `budget`).
      - otherwise: `_optimizers_sub_solve` (finite-difference joint descent
        via the in-house `optimizers` package, which has no evaluation-budget
        knob -- see its docstring).
    """
    try:
        spec = _SUB_SOLVERS[method]
    except KeyError:
        raise ValueError(
            f"sub_method={method!r} not in {sorted(_SUB_SOLVERS)}"
        ) from None
    if jac and spec["jac"]:
        return _projected_gradient_solve(fun, x0, bounds, max_evals=budget)
    return _optimizers_sub_solve(fun, x0, bounds)


# ---------------------------------------------------------------------------
# Acceptance guards: deciding whether a refinement beat its starting point.
#
# The refinement minimises a training objective; whether that helped on unseen
# data is a separate question, answered here on a split the search never saw.
# The naive answer -- "is held-out accuracy higher?" -- is what `legacy` does,
# and it is badly underpowered: on a ~37-row validation split one sample is 2.7
# points against a binomial standard error near 5, so the comparison is mostly
# reading noise. See issue #65.
#
# Each strategy takes the same evidence (per-row correctness and per-row log
# loss, for both models, on the held-out split) and returns accept/reject plus
# whatever diagnostics it used. They are interchangeable, which is what lets
# `benchmarks/guard_bench.py` score them against each other.
# ---------------------------------------------------------------------------

GUARDS: tuple[str, ...] = ("legacy", "effect-size", "mcnemar", "ce", "none")


def _mcnemar_p(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value for `only_a` vs `only_b` discordant pairs.

    McNemar is the right instrument here because both models predict the *same*
    rows: the rows they agree on carry no information about which is better, and
    an unpaired test that ignores that pairing throws away most of the power.
    Under the null the discordant pairs are Binomial(n, 1/2), so the exact tail
    is a sum of binomial coefficients -- no approximation, which matters because
    the discordant counts here are routinely single digits.
    """
    from math import comb

    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _apply_guard(
    strategy: str,
    correct_init: np.ndarray,
    correct_refined: np.ndarray,
    ll_init: np.ndarray,
    ll_refined: np.ndarray,
    alpha: float = 0.10,
) -> tuple[bool, dict]:
    """Decide whether to keep a refinement. Returns ``(accept, diagnostics)``.

    Args:
        correct_init, correct_refined: Per-row boolean correctness on the
            held-out split, for the starting model and the refined one.
        ll_init, ll_refined: Per-row negative log-likelihood on the same rows.
        alpha: Significance level for the tests that use one.
    """
    n = len(correct_init)
    acc_init = float(np.mean(correct_init)) if n else 0.0
    acc_refined = float(np.mean(correct_refined)) if n else 0.0
    ce_init = float(np.mean(ll_init)) if n else 0.0
    ce_refined = float(np.mean(ll_refined)) if n else 0.0
    diag = {"val_acc": acc_refined, "init_val_acc": acc_init,
            "val_ce": ce_refined, "init_val_ce": ce_init}

    if strategy == "none":
        # Route E: no guard at all -- trust the ridge shrinkage toward x0 to
        # bound how far the search can wander, and always keep the refinement.
        return True, {**diag, "guard": "none"}

    if strategy == "legacy":
        accept = (acc_refined > acc_init) or (
            acc_refined == acc_init and ce_refined < ce_init)
        return accept, {**diag, "guard": "legacy"}

    if strategy == "effect-size":
        # Route B: require the accuracy gain to clear one standard error of the
        # *paired difference*, not of either accuracy on its own -- the two
        # estimates are computed on the same rows and are strongly correlated,
        # so the unpaired error bar would be far too wide.
        d = correct_refined.astype(float) - correct_init.astype(float)
        se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        # A single discordant row sits *exactly* on the threshold: for k=1 the
        # paired mean and its standard error are both 1/n analytically, so a bare
        # `>` decides it on floating-point noise. Requiring a hair more than one
        # SE makes that case a deterministic reject, which is the intent -- one
        # row is precisely the noise this rule exists to resist.
        accept = (acc_refined - acc_init) > se * (1.0 + 1e-9)
        if acc_refined == acc_init:
            accept = ce_refined < ce_init
        return accept, {**diag, "guard": "effect-size", "paired_se": se}

    if strategy == "mcnemar":
        # Route A: significance test on the discordant pairs.
        only_refined = int(np.sum(correct_refined & ~correct_init))
        only_init = int(np.sum(correct_init & ~correct_refined))
        p = _mcnemar_p(only_refined, only_init)
        accept = only_refined > only_init and p < alpha
        if only_refined == only_init:
            # Indistinguishable on accuracy; fall back to the continuous signal.
            accept = ce_refined < ce_init
        return accept, {**diag, "guard": "mcnemar", "p_value": p,
                        "only_refined": only_refined, "only_init": only_init}

    if strategy == "ce":
        # Route F from the issue: score on the loss the search actually
        # minimises. Accuracy is a step function that discards most of the
        # signal; cross-entropy is continuous and much lower variance, so the
        # same split resolves a smaller true difference.
        accept = ce_refined < ce_init
        return accept, {**diag, "guard": "ce"}

    raise ValueError(f"guard={strategy!r} not in {GUARDS}")


def _per_row_evidence(X, y, model, norms, labels_hint=None):
    """Per-row correctness and negative log-likelihood under `model`."""
    proba, labels = _classifier_proba(X, model, norms)
    col = {lab: i for i, lab in enumerate(labels)}
    y_arr = np.asarray(y, dtype=object)
    y_idx = np.array([col.get(v, 0) for v in y_arr])
    pred = np.array([labels[i] for i in np.argmax(proba, axis=1)], dtype=object)
    p = np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)
    return (pred == y_arr), -np.log(p)


#: Families whose classifier objective is differentiable everywhere, so a
#: closed-form gradient is the *actual* derivative rather than a subgradient.
#: Only `probability` qualifies among the pairs the kernel has partials for:
#: min/max is piecewise smooth, with a kink wherever the arg-min or arg-max
#: switches. This distinction is not academic -- measured, the analytic gradient
#: is accuracy-neutral under probability (+0.0012 +/- 0.0026) and an accuracy
#: lottery under min/max (mean -0.0091, worst -0.0967).
_SMOOTH_FAMILIES = frozenset({"probability"})


def _smooth_objective(norms: NormPair) -> bool:
    """Whether `norms` makes the objective differentiable everywhere."""
    return (norms.t_norm in _SMOOTH_FAMILIES
            and norms.t_conorm in _SMOOTH_FAMILIES)


class _CrossEntropy:
    """``_cross_entropy_from_strengths`` with its per-call setup hoisted out.

    Profiling the wide refinement after the incremental-fold work put this
    function at roughly *twice* the cost of the forward pass it consumes, which
    was not where the time was supposed to be. Almost none of it was arithmetic:
    every call rebuilt ``np.arange(n)`` for the fancy index, allocated a fresh
    probability buffer, and did a two-array gather. The labels and the shape are
    fixed for a whole refinement, so all of that hoists into the constructor and
    the flat gather becomes a single take.

    Bit-identical to the function form -- the same operations on the same
    operands, only allocated once.
    """

    def __init__(self, y_idx: np.ndarray, n_labels: int, shape: tuple[int, int]):
        n, n_cols = shape
        self.n_labels = n_labels
        self.y_idx = np.asarray(y_idx)
        self.uniform = 1.0 / max(n_labels, 1)
        # Flat indices of the true-class entry of each row, so the per-call
        # gather is one `take` on a ravelled view instead of a two-array
        # fancy-index plus a fresh `arange`.
        self.flat_idx = np.arange(n, dtype=np.intp) * n_cols + np.asarray(y_idx, dtype=np.intp)
        self._p = np.empty(n, dtype=float)
        self._num = np.empty(n, dtype=float)

    def __call__(self, fs: np.ndarray) -> float:
        row = fs.sum(axis=1)
        np.take(fs.reshape(-1), self.flat_idx, out=self._num)
        p = self._p
        p.fill(self.uniform)
        np.divide(self._num, row, out=p, where=row > 0)
        np.clip(p, 1e-12, 1.0, out=p)
        np.log(p, out=p)
        return float(-np.mean(p))

    def with_column_grad(self, fs: np.ndarray, col: int, d_col) -> tuple[float, np.ndarray]:
        """Cross-entropy, and its derivative w.r.t. parameters that move only
        column `col` of `fs`.

        With ``p_i = fs[i, y_i] / S_i`` and only column ``c`` depending on the
        parameter,

            d(-log p_i)/dtheta = -( [y_i == c] / fs[i, c] - 1 / S_i ) * dfs[i, c]/dtheta

        -- the first term appears only for rows whose true class *is* the moved
        column, the second for every row, through the normaliser. Rows that are
        clipped, or whose strengths are all zero (uniform fallback), contribute
        nothing and are masked out.

        `d_col` is a sequence of ``(n,)`` derivative arrays; one output per entry.
        """
        row = fs.sum(axis=1)
        np.take(fs.reshape(-1), self.flat_idx, out=self._num)
        p = self._p
        p.fill(self.uniform)
        np.divide(self._num, row, out=p, where=row > 0)
        np.clip(p, 1e-12, 1.0, out=p)

        fs_c = fs[:, col]
        live = (row > 0) & (p > 1e-12) & (fs_c > 0)
        coef = np.zeros(len(row))
        np.divide(-1.0, row, out=coef, where=live)
        target = live & (self.y_idx == col)
        coef[target] += 1.0 / fs_c[target]

        n = len(row)
        grads = np.array([-float(np.dot(coef, d)) / n for d in d_col])

        np.log(p, out=p)
        return float(-np.mean(p)), grads


def _classifier_proba(X: pd.DataFrame, model: GaussianMixtureModel,
                      norms: NormPair | None = None):
    """Row-normalised firing strengths -> class probabilities, plus the label
    order. Mirrors ``TribbleClassifier.predict_proba`` (zero-firing
    rows fall back to uniform) so the fitness matches the deployed forward pass."""
    fs, labels = tsk_firing_strengths(X, model, norms=norms)
    return _normalize_proba(fs, len(labels)), labels


def _cross_entropy(proba: np.ndarray, y_idx: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class, with probability clipping."""
    p = np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def _make_classifier_fitness(model, X_tr, y_tr, l2_shrink, x0, lo, hi,
                             norms: NormPair | None = None):
    """Ridge-regularised training cross-entropy for a candidate antecedent vector.

    ``fitness(vec) = CE(train) + l2_shrink * mean(((vec - x0) / width) ** 2)``.

    The shrinkage term (scaled by each parameter's box width so it is
    dimensionless) is the real overfitting control: it keeps the tuned antecedents
    close to the data-driven heuristic unless the classification loss strongly
    favours moving them.

    `norms` must be the pair the *deployed* model will use. Refining against a
    different pair optimises a model nobody runs: the firing strengths, and
    therefore the cross-entropy surface, are a function of the operators.
    """
    labels = list(next(iter(model.feature_models.values())).ordered_keys)
    label_to_col = {lab: i for i, lab in enumerate(labels)}
    y_idx_tr = np.array([label_to_col.get(v, -1) for v in np.asarray(y_tr)])
    valid_tr = y_idx_tr >= 0
    y_idx_tr = y_idx_tr[valid_tr]
    X_tr = X_tr.iloc[np.where(valid_tr)[0]] if not valid_tr.all() else X_tr
    width = np.where((hi - lo) > 0, hi - lo, 1.0)
    n_labels = len(labels)
    norms = norms if norms is not None else resolve_norm_pair()

    # Compile the model once, if its shape allows (all-Gaussian, every feature
    # carrying every label). A candidate evaluation is then an in-place write of
    # 2*n_MF floats instead of a full reconstruction of the immutable model tree,
    # and the feature columns are extracted once instead of per call. Profiled at
    # baseline, those two costs were ~17% and ~5% of a refinement respectively.
    # The kernel is bit-exact against `tsk_firing_strengths`, so this changes the
    # cost of the search and not its trajectory.
    compiled = None
    try:
        compiled = compile_model(model, list(X_tr.columns))
    except NotCompilable:
        pass

    if compiled is not None:
        feature_matrix = compiled.feature_matrix(
            {name: X_tr[name].to_numpy() for name in compiled.feature_names}
        )
        return _CompiledClassifierObjective(
            compiled, feature_matrix, norms, y_idx_tr, n_labels, l2_shrink, x0, width
        )

    def fitness(vec: np.ndarray) -> float:
        candidate = apply_gaussian_params(model, vec)
        try:
            proba, cand_labels = _classifier_proba(X_tr, candidate, norms)
        except Exception:
            return 1e6
        # cand_labels order matches `labels` (same model structure), so columns align.
        ce = _cross_entropy(proba, y_idx_tr)
        reg = l2_shrink * float(np.mean(((vec - x0) / width) ** 2)) if l2_shrink else 0.0
        return ce + reg

    return fitness


class _CompiledClassifierObjective:
    """The ridge-shrunk cross-entropy objective over a compiled model.

    Callable like the plain closure it replaces, so every existing caller is
    unaffected. What it adds is :meth:`slot_fitness`: block coordinate descent
    knows it is moving one membership function's ``(mu, sigma)``, and that lets
    the evaluation reuse :class:`~tribblefis.kernel.IncrementalFIS`'s cached
    per-cell folds instead of recomputing the forward pass. Same number, ~15x
    less arithmetic on a wide model.
    """

    def __init__(self, compiled, feature_matrix, norms, y_idx, n_labels,
                 l2_shrink, x0, width):
        self.compiled = compiled
        self.feature_matrix = feature_matrix
        self.norms = norms
        self.y_idx = y_idx
        self.n_labels = n_labels
        self.l2_shrink = l2_shrink
        self.x0 = x0
        self.width = width
        self._incremental: IncrementalFIS | None = None
        self._ce = _CrossEntropy(y_idx, n_labels, (len(y_idx), n_labels))

    def _reg(self, vec: np.ndarray) -> float:
        if not self.l2_shrink:
            return 0.0
        return self.l2_shrink * float(np.mean(((vec - self.x0) / self.width) ** 2))

    def _loss(self, fs: np.ndarray, vec: np.ndarray) -> float:
        return self._ce(fs) + self._reg(vec)

    def __call__(self, vec: np.ndarray) -> float:
        try:
            self.compiled.set_params(vec)
            fs = kernel_firing_strengths(self.compiled, self.feature_matrix, self.norms)
        except Exception:
            return 1e6
        # A full evaluation moved parameters behind the incremental cache's back,
        # so the cache is stale until it is rebuilt.
        if self._incremental is not None:
            self._incremental.refresh()
        return self._loss(fs, vec)

    def _reg_grad(self, vec: np.ndarray, indices) -> np.ndarray:
        """d(reg)/d(vec[indices]) for ``reg = l2 * mean(((vec - x0)/width)**2)``."""
        if not self.l2_shrink:
            return np.zeros(len(indices))
        scale = 2.0 * self.l2_shrink / vec.size
        idx = np.asarray(indices)
        return scale * (vec[idx] - self.x0[idx]) / self.width[idx] ** 2

    def supports_gradient(self) -> bool:
        inc = self._ensure_incremental()
        return inc.supports_gradient()

    def _ensure_incremental(self) -> IncrementalFIS:
        if self._incremental is None:
            self._incremental = IncrementalFIS(
                self.compiled, self.feature_matrix, self.norms
            )
        return self._incremental

    def slot_fitness_with_grad(self, slot: int, template: np.ndarray):
        """Like :meth:`slot_fitness`, but the objective also returns its gradient.

        L-BFGS-B otherwise finite-differences a two-parameter block, which costs
        two extra whole evaluations per gradient -- two thirds of all evaluations
        in a refinement. Here the derivative rides along inside the same fold
        that computes the value.

        Under ``min/max`` this is a subgradient and the search therefore takes a
        *different* path than the finite-difference version; it is not a
        bit-exact substitution, and measurably it is not reliably a better one.
        See ``analytic_gradient`` in :func:`refine_classifier_antecedents` and
        ``docs/analytic-gradient-evaluation.md``.
        """
        inc = self._ensure_incremental()
        vec = template.copy()
        i_mu, i_sigma = 2 * slot, 2 * slot + 1
        col = inc.target_label_index(slot)

        reg_grad = self._reg_grad

        def f_sub(v):
            vec[i_mu] = v[0]
            vec[i_sigma] = v[1]
            try:
                fs, d_mu, d_sigma = inc.evaluate_slot_with_grad(
                    slot, float(v[0]), float(v[1])
                )
            except Exception:
                return 1e6, np.zeros(2)
            ce, grad = self._ce.with_column_grad(fs, col, (d_mu, d_sigma))
            return ce + self._reg(vec), grad + reg_grad(vec, (i_mu, i_sigma))

        def commit(v) -> None:
            inc.evaluate_slot(slot, float(v[0]), float(v[1]))
            inc.commit()

        return f_sub, commit

    def slot_fitness(self, slot: int, template: np.ndarray):
        """A two-argument objective for membership `slot`'s ``(mu, sigma)``.

        `template` is the current full parameter vector; only the shrinkage term
        needs it, and only the two entries this slot owns ever differ from it.
        Returns ``(f_sub, commit)``: call ``commit(v)`` to fold an accepted
        ``v`` into the cache. Nothing is mutated until then, so an L-BFGS-B
        sub-problem that ends up rejecting every trial leaves no trace.
        """
        inc = self._ensure_incremental()
        vec = template.copy()
        i_mu, i_sigma = 2 * slot, 2 * slot + 1

        def f_sub(v) -> float:
            vec[i_mu] = v[0]
            vec[i_sigma] = v[1]
            try:
                fs = inc.evaluate_slot(slot, float(v[0]), float(v[1]))
            except Exception:
                return 1e6
            return self._loss(fs, vec)

        def commit(v) -> None:
            inc.evaluate_slot(slot, float(v[0]), float(v[1]))
            inc.commit()

        return f_sub, commit


def _classifier_accuracy(X, y, model, norms: NormPair | None = None) -> float:
    proba, labels = _classifier_proba(X, model, norms)
    pred = np.array([labels[i] for i in np.argmax(proba, axis=1)], dtype=object)
    return float(np.mean(pred == np.asarray(y, dtype=object)))


def _classifier_val_ce(X, y, model, norms: NormPair | None = None) -> float:
    """Held-out cross-entropy, mapping each true label to its firing-strength column."""
    proba, labels = _classifier_proba(X, model, norms)
    col = {lab: i for i, lab in enumerate(labels)}
    y_idx = np.array([col.get(v, 0) for v in np.asarray(y, dtype=object)])
    return _cross_entropy(proba, y_idx)


def refine_classifier_antecedents(
    model: GaussianMixtureModel,
    X_train: pd.DataFrame,
    y_train: typing.Any,
    *,
    method: str = "coordinate",
    l2_shrink: float = 0.05,
    val_fraction: float = 0.25,
    n_sweeps: int = 3,
    block: int = 2,
    norms: NormPair | None = None,
    incremental: bool = True,
    analytic_gradient: bool | None = None,
    sub_method: str = "SLSQP",
    guard: str = "none",
    sub_maxfun: int = 25,
    population_size: int = 40,
    num_generations: int = 20,
    local_scale: float | None = 0.25,
    optimizer_method: str = "ga",
    local_grad_optim: str = "single-var-grad",
    sigma_min_frac: float = 0.02,
    sigma_max_frac: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[GaussianMixtureModel, dict]:
    """Refine a fuzzy *classifier*'s Gaussian antecedents against cross-entropy.

    ``method="coordinate"`` (default) runs the proven per-membership block
    coordinate descent; ``method="optimizers"`` runs the `optimizers`-package
    population+polish search. Either way the objective is a ridge-shrunk training
    cross-entropy and the result is accepted only if it does not worsen a held-out
    validation split's accuracy *and* cross-entropy (otherwise the heuristic model
    is returned unchanged). Returns ``(refined_model, info)``.

    ``norms`` selects the (t-norm, t-conorm) pair the objective is evaluated
    under, and must match what the deployed model uses -- the firing strengths,
    and so the entire loss surface, depend on it. Callers that hold an operator
    choice (``TribbleClassifier.norm_conorm``) must pass it; the
    default reproduces the library-wide default pair.

    ``incremental=False`` turns off the cached per-cell evaluation that
    ``block=2`` coordinate descent otherwise uses (see
    :class:`~tribblefis.kernel.IncrementalFIS`). The two produce bit-identical
    results and the cache is several times faster on a wide model, so this exists
    to A/B that claim, not because either answer is preferable.

    ``analytic_gradient`` hands the solver a closed-form gradient instead of
    letting it finite-difference each two-parameter block, removing two thirds of
    the evaluations. The default, ``None``, enables it exactly when the operator
    pair makes the objective differentiable everywhere -- which is what the
    default ``probability`` family does. Under a piecewise-smooth pair such as
    ``min/max`` the closed form is only a *subgradient*, and measured it turns
    the search into an accuracy lottery (mean -0.0091, worst -0.0967), so ``None``
    leaves it off there. ``True``/``False`` override the rule. See
    ``docs/analytic-gradient-evaluation.md``.

    ``sub_method`` no longer selects a distinct scipy algorithm -- `_sub_solve`
    routes every block through one of two in-house, non-scipy backends chosen
    by whether a gradient is supplied (`_projected_gradient_solve` or
    `_optimizers_sub_solve`; see `_sub_solve`'s docstring). The parameter is
    kept so a caller's `sub_method` still validates against the same five
    names. The comparison below is historical, from when each name really did
    select a different scipy method: measured under the default family, SLSQP
    was 1.14x over L-BFGS-B at equal accuracy and 1.95x with the gradient;
    Powell was slightly more accurate but 1.5x slower; TNC was 3x slower.

    ``guard`` decides whether a refinement is kept. It defaults to ``"none"`` --
    keep it always -- which is a measured result, not an oversight: across 108
    (dataset x split x configuration) cases, refinements beat their starting
    point 85 times and lost 12, gaining ~4 points when they won and shedding ~2
    when they lost, so *every* rejection rule tested destroys expected accuracy.
    ``"none"`` also lets the search train on all the data instead of reserving
    ``val_fraction`` of it to referee a decision no longer being made. The
    alternatives -- ``"legacy"``, ``"ce"``, ``"effect-size"``, ``"mcnemar"`` --
    remain for anyone who wants a bounded worst case at that price. See
    ``docs/refinement-guard-evaluation.md`` and issue #65.
    """
    if guard not in GUARDS:
        raise ValueError(f"guard={guard!r} not in {GUARDS}")
    y_arr = np.asarray(y_train)
    # The operators the deployed model will use. Refining against a different
    # pair optimises a model nobody runs, because the firing strengths -- and so
    # the whole loss surface -- are a function of them.
    resolved_norms = norms if norms is not None else resolve_norm_pair()
    bounds = build_param_bounds(model, X_train, sigma_min_frac, sigma_max_frac)
    if not bounds:                          # no Gaussian memberships -> nothing to do
        return model, {"refined": False, "reason": "no_gaussian_memberships"}
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0 = np.clip(extract_gaussian_params(model), lo, hi)

    # Held-out split used only to *accept/reject* the refinement (never optimised).
    # With `guard="none"` there is no decision to referee, so withholding a
    # quarter of the training data would be pure loss -- the search gets all of
    # it instead. That is half the value of dropping the guard; see
    # docs/refinement-guard-evaluation.md.
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X_train))
    if guard == "none":
        tr_idx = val_idx = idx
    else:
        strat = y_arr if len(np.unique(y_arr)) > 1 else None
        try:
            tr_idx, val_idx = train_test_split(
                idx, test_size=val_fraction, random_state=seed, stratify=strat)
        except ValueError:                  # too few samples in a class to stratify
            tr_idx, val_idx = train_test_split(idx, test_size=val_fraction,
                                               random_state=seed)
    X_tr, y_tr = X_train.iloc[tr_idx], y_arr[tr_idx]
    X_val, y_val = X_train.iloc[val_idx], y_arr[val_idx]

    fitness = _make_classifier_fitness(model, X_tr, y_tr, l2_shrink, x0, lo, hi,
                                       resolved_norms)
    init_fit = fitness(x0)
    init_val_acc = _classifier_accuracy(X_val, y_val, model, resolved_norms)

    if verbose:
        print(f"\nClassifier antecedent refinement ({method}): {len(bounds)} params, "
              f"l2_shrink={l2_shrink}, init train obj={init_fit:.5f}, "
              f"init val acc={init_val_acc:.4f}")

    if method == "coordinate":
        x = x0.copy()
        cur = init_fit
        n_eval = 1
        n_params = len(bounds)
        n_blocks = (n_params + block - 1) // block
        # A block of 2 starting on an even index *is* one membership function's
        # (mu, sigma), which is the case the incremental evaluator handles. Any
        # other blocking (block=1, block=4, a ragged tail) falls back to the full
        # objective, which is the same function evaluated the slow way.
        use_incremental = (
            incremental
            and block == 2
            and n_params % 2 == 0
            and isinstance(fitness, _CompiledClassifierObjective)
        )
        want_grad = (
            _smooth_objective(resolved_norms)
            if analytic_gradient is None else analytic_gradient
        )
        use_analytic_grad = (
            use_incremental and want_grad
            and _SUB_SOLVERS.get(sub_method, {}).get("jac", False)
            and fitness.supports_gradient()
        )
        with _single_threaded():
            for sweep in range(n_sweeps):
                prev = cur
                for b in range(n_blocks):
                    bidx = np.arange(b * block, min((b + 1) * block, n_params))
                    sub_bounds = [(lo[k], hi[k]) for k in bidx]

                    commit = None
                    jac = False
                    if use_incremental:
                        if use_analytic_grad:
                            slot_f, commit = fitness.slot_fitness_with_grad(b, x)
                            jac = True
                        else:
                            slot_f, commit = fitness.slot_fitness(b, x)

                        def f_sub(v, _slot_f=slot_f):
                            nonlocal n_eval
                            n_eval += 1
                            return _slot_f(v)
                    else:
                        def f_sub(v):
                            nonlocal n_eval
                            trial = x.copy()
                            trial[bidx] = v
                            n_eval += 1
                            return fitness(trial)

                    res = _sub_solve(sub_method, f_sub, x[bidx], sub_bounds,
                                     sub_maxfun, jac)
                    if res.fun < cur - 1e-12:
                        x[bidx] = np.clip(res.x, lo[bidx], hi[bidx])
                        cur = float(res.fun)
                        if commit is not None:
                            # Fold the accepted move into the cache so the next
                            # block starts from it. Rejected blocks never touch
                            # the cache, so they need no undo.
                            commit(x[bidx])
                if verbose:
                    print(f"  sweep {sweep + 1}/{n_sweeps}: train obj={cur:.5f} (evals={n_eval})")
                if prev - cur < 1e-6:
                    break
        best_x, best_fit = x, cur
        info = {"train_obj": best_fit, "n_eval": n_eval}
    elif method == "optimizers":
        best_x, best_fit, info = _run_optimizer_search(
            fitness, bounds, x0, method=optimizer_method, local_grad_optim=local_grad_optim,
            population_size=population_size, num_generations=num_generations,
            local_scale=local_scale, seed=seed, label="classifier",
        )
    else:
        raise ValueError(f"method={method!r} must be 'coordinate' or 'optimizers'")

    refined = apply_gaussian_params(model, best_x)

    # Accept only on a held-out improvement the chosen guard is willing to call
    # real. Both models are scored on the same rows, so the evidence is paired
    # and the guards can use that.
    ok_init, ll_init = _per_row_evidence(X_val, y_val, model, resolved_norms)
    ok_refined, ll_refined = _per_row_evidence(X_val, y_val, refined, resolved_norms)
    accept, guard_info = _apply_guard(guard, ok_init, ok_refined, ll_init, ll_refined)
    val_acc = guard_info["val_acc"]
    val_ce = guard_info["val_ce"]
    init_val_ce = guard_info["init_val_ce"]
    out_model = refined if accept else model
    if verbose:
        verdict = "accepted" if accept else "rejected (kept heuristic)"
        print(f"  refinement {verdict} [{guard}]: val acc {init_val_acc:.4f} -> "
              f"{val_acc:.4f}, val CE {init_val_ce:.4f} -> {val_ce:.4f}")
    return out_model, {
        "refined": bool(accept),
        "init_val_acc": init_val_acc, "val_acc": val_acc,
        "init_val_ce": init_val_ce, "val_ce": val_ce,
        "init_train_obj": init_fit, **guard_info, **info,
    }


# ---------------------------------------------------------------------------
# Ruspini triangular-partition refinement (apex knots).
# ---------------------------------------------------------------------------
#
# A Ruspini model (`tribblefis.ruspini.RuspiniPartitionModel`) is defined entirely
# by its per-feature triangular *apex knots*; the class->term rule assignment is
# frozen. Refining the knots therefore searches a low-dimensional, naturally
# constrained space -- because every candidate is rebuilt from shared knots, the
# partition-of-unity property holds for free and no per-parameter shape constraint
# is needed. We reuse the classifier objective (ridge-shrunk cross-entropy +
# held-out acceptance guard); the only new ingredient is a firing/proba routine
# for the explicit triangular model.


def _ruspini_accuracy(rmodel, X, y) -> float:
    proba, labels = rmodel.class_proba(X)
    pred = np.array([labels[i] for i in np.argmax(proba, axis=1)], dtype=object)
    return float(np.mean(pred == np.asarray(y, dtype=object)))


def _ruspini_row_evidence(rmodel, X, y):
    """Per-row correctness and negative log-likelihood, for the guards."""
    proba, labels = rmodel.class_proba(X)
    col = {lab: i for i, lab in enumerate(labels)}
    y_arr = np.asarray(y, dtype=object)
    y_idx = np.array([col.get(v, 0) for v in y_arr])
    pred = np.array([labels[i] for i in np.argmax(proba, axis=1)], dtype=object)
    p = np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)
    return (pred == y_arr), -np.log(p)


def _ruspini_ce(rmodel, X, y) -> float:
    proba, labels = rmodel.class_proba(X)
    col = {lab: i for i, lab in enumerate(labels)}
    y_idx = np.array([col.get(v, 0) for v in np.asarray(y, dtype=object)])
    return _cross_entropy(proba, y_idx)


def refine_ruspini_partition(
    rmodel,
    X_train: pd.DataFrame,
    y_train: typing.Any,
    *,
    method: str = "coordinate",
    l2_shrink: float = 0.02,
    val_fraction: float = 0.25,
    n_sweeps: int = 3,
    sub_maxfun: int = 25,
    pad_frac: float = 0.05,
    guard: str = "legacy",
    population_size: int = 40,
    num_generations: int = 20,
    local_scale: float | None = 0.3,
    optimizer_method: str = "ga",
    local_grad_optim: str = "perturb",
    seed: int = 42,
    verbose: bool = True,
):
    """Refine a Ruspini triangular partition's apex knots against cross-entropy.

    Searches the concatenated per-feature apex-knot vector (each candidate is
    re-sorted into a valid monotone partition by ``RuspiniPartitionModel.with_knots``,
    so partition-of-unity is preserved automatically). ``method="coordinate"`` moves
    one knot at a time via a coarse-to-fine grid line search (no scipy, no
    `optimizers` -- the knot objective is piecewise-linear, so a gradient step
    stalls; see the loop below); ``method="optimizers"`` uses the
    `optimizers`-package population+polish search. The objective is a ridge-shrunk
    training cross-entropy; the refined knots are accepted only if they do not
    worsen a held-out split's accuracy (CE tiebreak), else the input model is
    returned unchanged. Returns ``(refined_rmodel, info)``.

    ``guard`` keeps its ``"legacy"`` default here, unlike
    :func:`refine_classifier_antecedents`, which dropped its guard entirely.
    That difference is measured, not an oversight or an omission: on this search
    the guard is a wash (``legacy`` beats ``"none"`` by 0.0049 +/- 0.0058,
    t = 0.84 over 48 paired cases), where on the classifier it was a clear loss.
    The base rate differs -- refinement helps 2.2x more often than it hurts here
    against 7.1x there -- so the classifier's answer had no business being
    assumed. With no evidence for a change, the existing behaviour stands.
    ``guard="mcnemar"`` is the one option to avoid: significantly *worse*
    (-0.0240 +/- 0.0094) on both searches. See
    ``docs/refinement-guard-evaluation.md``.
    """
    y_arr = np.asarray(y_train)
    knots0 = rmodel.extract_knots()
    if len(knots0) == 0:
        return rmodel, {"refined": False, "reason": "no_knots"}

    # Per-knot box bounds from each feature's observed range (padded).
    slices = rmodel.knot_slices()
    lo = np.empty(len(knots0))
    hi = np.empty(len(knots0))
    for f in rmodel.feature_order:
        sl = slices[f]
        if f in X_train.columns:
            col = X_train[f].to_numpy(dtype=float)
            flo, fhi = float(np.min(col)), float(np.max(col))
        else:
            flo, fhi = float(np.min(knots0[sl])), float(np.max(knots0[sl]))
        pad = pad_frac * (fhi - flo if fhi > flo else 1.0)
        lo[sl], hi[sl] = flo - pad, fhi + pad
    bounds = list(zip(lo.tolist(), hi.tolist()))
    x0 = np.clip(knots0, lo, hi)
    width = np.where((hi - lo) > 0, hi - lo, 1.0)

    if guard not in GUARDS:
        raise ValueError(f"guard={guard!r} not in {GUARDS}")

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X_train))
    if guard == "none":
        # No decision to referee, so no reason to withhold data from the search.
        tr_idx = val_idx = idx
    else:
        strat = y_arr if len(np.unique(y_arr)) > 1 else None
        try:
            tr_idx, val_idx = train_test_split(idx, test_size=val_fraction, random_state=seed, stratify=strat)
        except ValueError:
            tr_idx, val_idx = train_test_split(idx, test_size=val_fraction, random_state=seed)
    X_tr, y_tr = X_train.iloc[tr_idx].reset_index(drop=True), y_arr[tr_idx]
    X_val, y_val = X_train.iloc[val_idx].reset_index(drop=True), y_arr[val_idx]

    # A class can own more than one rule (see `cluster_joint_terms` in
    # `ruspinize_model`); dedupe the same way `class_proba` does, so this lookup
    # doesn't silently collapse to "last rule for this label wins".
    labels0 = list(dict.fromkeys(consequent for consequent, _ in rmodel.rules))
    label_to_col = {lab: i for i, lab in enumerate(labels0)}
    y_idx_tr = np.array([label_to_col.get(v, -1) for v in y_tr])
    keep = y_idx_tr >= 0
    y_idx_tr = y_idx_tr[keep]
    X_tr_fit = X_tr.iloc[np.where(keep)[0]].reset_index(drop=True) if not keep.all() else X_tr

    def fitness(vec: np.ndarray) -> float:
        try:
            cand = rmodel.with_knots(vec)
            proba, _ = cand.class_proba(X_tr_fit)
        except Exception:
            return 1e6
        ce = _cross_entropy(proba, y_idx_tr)
        reg = l2_shrink * float(np.mean(((vec - x0) / width) ** 2)) if l2_shrink else 0.0
        return ce + reg

    init_fit = fitness(x0)
    init_val_acc = _ruspini_accuracy(rmodel, X_val, y_val)
    if verbose:
        print(f"\nRuspini knot refinement ({method}): {len(knots0)} knots, "
              f"l2_shrink={l2_shrink}, init train obj={init_fit:.5f}, "
              f"init val acc={init_val_acc:.4f}")

    if method == "coordinate":
        # The knot objective is piecewise-linear, so a gradient step (tiny finite
        # difference) sees a near-zero slope and stalls. A per-knot *line search*
        # over a coarse-to-fine grid across the knot's box moves reliably on this
        # landscape. `sub_maxfun` sets the grid resolution.
        x = x0.copy()
        cur = init_fit
        n_eval = 1
        grid_n = max(5, sub_maxfun)
        with _single_threaded():
            for sweep in range(n_sweeps):
                prev = cur
                window = 0.5 ** sweep  # shrink the search window each sweep (coarse -> fine)
                for k in range(len(x)):
                    half = window * (hi[k] - lo[k]) / 2.0
                    grid = np.clip(np.linspace(x[k] - half, x[k] + half, grid_n), lo[k], hi[k])
                    best_v, best_f = x[k], cur
                    for v in grid:
                        trial = x.copy()
                        trial[k] = v
                        n_eval += 1
                        fv = fitness(trial)
                        if fv < best_f - 1e-12:
                            best_v, best_f = float(v), fv
                    if best_f < cur - 1e-12:
                        x[k] = best_v
                        cur = best_f
                if verbose:
                    print(f"  sweep {sweep + 1}/{n_sweeps}: train obj={cur:.5f} (evals={n_eval})")
                if prev - cur < 1e-6:
                    break
        best_x, best_fit = x, cur
        info = {"train_obj": best_fit, "n_eval": n_eval}
    elif method == "optimizers":
        best_x, best_fit, info = _run_optimizer_search(
            fitness, bounds, x0, method=optimizer_method, local_grad_optim=local_grad_optim,
            population_size=population_size, num_generations=num_generations,
            local_scale=local_scale, seed=seed, label="ruspini-knots",
        )
    else:
        raise ValueError(f"method={method!r} must be 'coordinate' or 'optimizers'")

    refined = rmodel.with_knots(best_x)
    ok_init, ll_init = _ruspini_row_evidence(rmodel, X_val, y_val)
    ok_refined, ll_refined = _ruspini_row_evidence(refined, X_val, y_val)
    accept, guard_info = _apply_guard(guard, ok_init, ok_refined, ll_init, ll_refined)
    val_acc, val_ce = guard_info["val_acc"], guard_info["val_ce"]
    init_val_ce = guard_info["init_val_ce"]
    out = refined if accept else rmodel
    if verbose:
        verdict = "accepted" if accept else "rejected (kept initial)"
        print(f"  Ruspini refinement {verdict} [{guard}]: val acc {init_val_acc:.4f} "
              f"-> {val_acc:.4f}, val CE {init_val_ce:.4f} -> {val_ce:.4f}")
    return out, {
        "refined": bool(accept),
        "init_val_acc": init_val_acc, "val_acc": val_acc,
        "init_val_ce": init_val_ce, "val_ce": val_ce,
        "init_train_obj": init_fit, **guard_info, **info,
    }
