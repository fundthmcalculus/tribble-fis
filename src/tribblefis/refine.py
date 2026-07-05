"""Phase 2: post-model refinement of the Gaussian antecedent parameters.

The heuristic membership fit (KMeans + `stats.norm.fit` per output bucket) sets
every membership function's ``(mu, sigma)`` without ever looking at the
regression objective. Those parameters determine the firing strengths, which are
then frozen while the consequents are fit. This module closes that loop: it
searches over the ``(mu, sigma)`` of every Gaussian membership function to
minimize a held-out MSE, using the Phase 1 closed-form consequent solver
(`solve_tsk_consequents`) as the fast, exact inner step. Because the consequents
are solved in closed form for each candidate, the search dimension is just
``2 * n_membership_functions`` -- the consequents are never themselves searched.

Two optimizers are provided:
- `refine_antecedents_de`  -- SciPy differential evolution (global, low-effort).
- `refine_antecedents_ga`  -- a dependency-light real-coded genetic algorithm
                              (tournament + BLX-alpha crossover + Gaussian
                              mutation + elitism), seeded from the heuristic model.

Both hold out an inner validation fold from the training data and select on that
fold, never on the test set, and both guarantee they never return a model worse
(on the validation fold) than the heuristic starting point.
"""

import typing

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from .gauss_data import GaussianMembership, LabelModel, FeatureModel, GaussianMixtureModel
from .regression import solve_tsk_consequents, predict_tsk, _mse, _rsquared


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
        col = X_train[fname].to_numpy()
        lo, hi = float(np.min(col)), float(np.max(col))
        rng = hi - lo if hi > lo else 1.0
        bounds.append((lo, hi))                                  # mu
        bounds.append((sigma_min_frac * rng, sigma_max_frac * rng))  # sigma
    return bounds


# ---------------------------------------------------------------------------
# Fitness: apply candidate antecedents -> closed-form consequents -> val MSE.
# ---------------------------------------------------------------------------

def _make_kfold_fitness(
    model, X_train, y_train, folds, top_n_todo, n_output_buckets, order, l2_reg, basis, cross_pairs,
):
    """Cross-validated fitness: mean held-out MSE over `folds`.

    A single validation fold is far too easy to overfit when the search has
    O(100) free antecedent parameters -- the optimizer drives that one fold's MSE
    down while test error rises. Averaging over k folds forces the antecedents to
    generalize. Each fold pre-slices its train/val DataFrames once (outside the
    hot loop) so a fitness call is just: apply params -> per-fold solve+predict.
    """
    prepared = []
    y_bucket_mean_dummy = np.zeros(n_output_buckets)  # solver ignores this arg
    for tr_idx, val_idx in folds:
        prepared.append((
            X_train.iloc[tr_idx], y_train.iloc[tr_idx],
            X_train.iloc[val_idx], y_train.iloc[val_idx]["y_value"].to_numpy(),
        ))

    def fitness(vec: np.ndarray) -> float:
        candidate = apply_gaussian_params(model, vec)
        total, n = 0.0, 0
        for X_tr, y_tr, X_val, y_val_true in prepared:
            try:
                corr, means = solve_tsk_consequents(
                    X_tr, candidate, top_n_todo, y_bucket_mean_dummy, y_tr,
                    n_output_buckets=n_output_buckets, order=order,
                    l2_reg=l2_reg, basis=basis, cross_pairs=cross_pairs, verbose=False,
                )
                y_hat = predict_tsk(
                    X_val, candidate, top_n_todo, means, corr,
                    order=order, basis=basis, cross_pairs=cross_pairs,
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
    """Refine antecedents with SciPy differential evolution.

    Returns (refined_model, info) where info has the initial/final validation MSE.
    Never returns a model worse than the heuristic start on the CV fitness.
    """
    folds = _make_folds(len(X_train), n_folds, val_fraction, seed)
    fitness = _make_kfold_fitness(model, X_train, y_train, folds, top_n_todo,
                                  n_output_buckets, order, l2_reg, basis, cross_pairs)
    bounds = build_param_bounds(model, X_train)
    x0 = np.clip(extract_gaussian_params(model),
                 [b[0] for b in bounds], [b[1] for b in bounds])
    init_fit = fitness(x0)

    print(f"\nDE antecedent refinement: {len(bounds)} params, order={order}, "
          f"init val MSE={init_fit:.5f}")
    result = differential_evolution(
        fitness, bounds, x0=x0, seed=seed, maxiter=maxiter, popsize=popsize,
        tol=1e-6, mutation=(0.5, 1.0), recombination=0.7, polish=True,
        init="sobol", updating="immediate",
    )

    best_x, best_fit = (result.x, result.fun) if result.fun <= init_fit else (x0, init_fit)
    if result.fun > init_fit:
        print("  DE did not beat the heuristic start; keeping heuristic antecedents.")
    print(f"  DE done: val MSE {init_fit:.5f} -> {best_fit:.5f} "
          f"({100 * (init_fit - best_fit) / max(init_fit, 1e-12):.1f}% lower)")
    return apply_gaussian_params(model, best_x), {"init_val_mse": init_fit, "val_mse": best_fit}


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
    """Refine antecedents by L-BFGS-B *local* descent from the heuristic start.

    This is the recommended default. Empirically, aggressive global search (DE
    without polish, or a long GA) drives the cross-validated fitness down but
    *overfits that CV estimate* -- test error rises. A local refinement stays in
    the heuristic's basin and reliably improves test error. (The forward pass uses
    the min/max t-norm, which is non-smooth, so L-BFGS-B works from a finite-
    difference gradient; this is exactly the local step DE's `polish=True`
    performs, but without the expensive and counter-productive global phase.)

    Never returns a model worse than the heuristic start on the CV fitness.
    """
    folds = _make_folds(len(X_train), n_folds, val_fraction, seed)
    fitness = _make_kfold_fitness(model, X_train, y_train, folds, top_n_todo,
                                  n_output_buckets, order, l2_reg, basis, cross_pairs)
    bounds = build_param_bounds(model, X_train)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0 = np.clip(extract_gaussian_params(model), lo, hi)
    init_fit = fitness(x0)

    print(f"\nLocal (L-BFGS-B) antecedent refinement: {len(bounds)} params, "
          f"order={order}, {n_folds}-fold init val MSE={init_fit:.5f}")
    result = minimize(fitness, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": maxiter, "maxfun": maxfun})

    best_x, best_fit = (result.x, result.fun) if result.fun <= init_fit else (x0, init_fit)
    if result.fun > init_fit:
        print("  Local refine did not beat the heuristic start; keeping heuristic.")
    print(f"  Local refine done: val MSE {init_fit:.5f} -> {best_fit:.5f} "
          f"({100 * (init_fit - best_fit) / max(init_fit, 1e-12):.1f}% lower)")
    return apply_gaussian_params(model, best_x), {"init_val_mse": init_fit, "val_mse": best_fit}


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
