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
from .gauss_math import tsk_firing_strengths
from .regression import solve_tsk_consequents, predict_tsk, _mse, _rsquared

# The refinement fitness runs thousands of tiny (~O(100)-wide) linear solves. On a
# multithreaded BLAS those small matrices thrash on thread-spawn overhead and
# oversubscribe the machine -- pinning BLAS to a single thread roughly halves
# wall-clock here. Wrap the search loops in `_single_threaded()`.
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
    with _single_threaded():
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

    Kept for comparison; `refine_antecedents_coordinate` is the recommended default
    at this scale (this single high-dimensional L-BFGS-B solve spends one evaluation
    per parameter on every finite-difference gradient). Empirically, aggressive
    global search (DE without polish, or a long GA) drives the CV fitness down but
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
    with _single_threaded():
        result = minimize(fitness, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": maxiter, "maxfun": maxfun})

    best_x, best_fit = (result.x, result.fun) if result.fun <= init_fit else (x0, init_fit)
    if result.fun > init_fit:
        print("  Local refine did not beat the heuristic start; keeping heuristic.")
    print(f"  Local refine done: val MSE {init_fit:.5f} -> {best_fit:.5f} "
          f"({100 * (init_fit - best_fit) / max(init_fit, 1e-12):.1f}% lower)")
    return apply_gaussian_params(model, best_x), {"init_val_mse": init_fit, "val_mse": best_fit}


# ---------------------------------------------------------------------------
# Per-variable (block) coordinate descent.
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

    Never returns a model worse than the heuristic start on the CV fitness (the
    running best is only ever updated on a strict improvement).
    """
    folds = _make_folds(len(X_train), n_folds, val_fraction, seed)
    fitness = _make_kfold_fitness(model, X_train, y_train, folds, top_n_todo,
                                  n_output_buckets, order, l2_reg, basis, cross_pairs)
    bounds = build_param_bounds(model, X_train)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    n_params = len(bounds)

    x = np.clip(extract_gaussian_params(model), lo, hi)
    init_fit = fitness(x)
    cur = init_fit
    n_eval = [1]  # count fitness calls for reporting

    n_blocks = (n_params + block - 1) // block
    print(f"\nCoordinate-descent antecedent refinement: {n_params} params "
          f"({n_blocks} blocks of {block}), order={order}, {n_folds}-fold "
          f"init val MSE={init_fit:.5f}")

    with _single_threaded():
        for sweep in range(n_sweeps):
            prev = cur
            for b in range(n_blocks):
                idx = np.arange(b * block, min((b + 1) * block, n_params))
                sub_bounds = [(lo[k], hi[k]) for k in idx]

                def f_sub(v):
                    trial = x.copy()
                    trial[idx] = v
                    n_eval[0] += 1
                    return fitness(trial)

                res = minimize(f_sub, x[idx], method="L-BFGS-B", bounds=sub_bounds,
                               options={"maxfun": sub_maxfun, "maxiter": sub_maxfun})
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
                GeneticAlgorithmOptimizerConfig(**common), fcn, variables, existing_soln_deck=deck)
        elif method == "pso":
            opt = ParticleSwarmOptimizer(
                ParticleSwarmOptimizerConfig(**common), fcn, variables, existing_soln_deck=deck)
        elif method == "aco":
            opt = AntColonyOptimizer(
                AntColonyOptimizerConfig(**common), fcn, variables, existing_soln_deck=deck)
        else:  # "multi"
            opt = MultiTypeOptimizer(
                IOptimizerConfig(**common), fcn, variables, existing_soln_deck=deck)
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


def _classifier_proba(X: pd.DataFrame, model: GaussianMixtureModel):
    """Row-normalised firing strengths -> class probabilities, plus the label
    order. Mirrors ``MixtureOfGaussiansFuzzyClassifier.predict_proba`` (zero-firing
    rows fall back to uniform) so the fitness matches the deployed forward pass."""
    fs, labels = tsk_firing_strengths(X, model)
    row = fs.sum(axis=1, keepdims=True)
    proba = np.full_like(fs, 1.0 / max(len(labels), 1))
    nz = row.flatten() > 0
    proba[nz] = fs[nz] / row[nz]
    return proba, labels


def _cross_entropy(proba: np.ndarray, y_idx: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class, with probability clipping."""
    p = np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def _make_classifier_fitness(model, X_tr, y_tr, l2_shrink, x0, lo, hi):
    """Ridge-regularised training cross-entropy for a candidate antecedent vector.

    ``fitness(vec) = CE(train) + l2_shrink * mean(((vec - x0) / width) ** 2)``.

    The shrinkage term (scaled by each parameter's box width so it is
    dimensionless) is the real overfitting control: it keeps the tuned antecedents
    close to the data-driven heuristic unless the classification loss strongly
    favours moving them.
    """
    labels = list(next(iter(model.feature_models.values())).ordered_keys)
    label_to_col = {lab: i for i, lab in enumerate(labels)}
    y_idx_tr = np.array([label_to_col.get(v, -1) for v in np.asarray(y_tr)])
    valid_tr = y_idx_tr >= 0
    y_idx_tr = y_idx_tr[valid_tr]
    X_tr = X_tr.iloc[np.where(valid_tr)[0]] if not valid_tr.all() else X_tr
    width = np.where((hi - lo) > 0, hi - lo, 1.0)

    def fitness(vec: np.ndarray) -> float:
        candidate = apply_gaussian_params(model, vec)
        try:
            proba, cand_labels = _classifier_proba(X_tr, candidate)
        except Exception:
            return 1e6
        # cand_labels order matches `labels` (same model structure), so columns align.
        ce = _cross_entropy(proba, y_idx_tr)
        reg = l2_shrink * float(np.mean(((vec - x0) / width) ** 2)) if l2_shrink else 0.0
        return ce + reg

    return fitness


def _classifier_accuracy(X, y, model) -> float:
    proba, labels = _classifier_proba(X, model)
    pred = np.array([labels[i] for i in np.argmax(proba, axis=1)], dtype=object)
    return float(np.mean(pred == np.asarray(y, dtype=object)))


def _classifier_val_ce(X, y, model) -> float:
    """Held-out cross-entropy, mapping each true label to its firing-strength column."""
    proba, labels = _classifier_proba(X, model)
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
    """
    y_arr = np.asarray(y_train)
    bounds = build_param_bounds(model, X_train, sigma_min_frac, sigma_max_frac)
    if not bounds:                          # no Gaussian memberships -> nothing to do
        return model, {"refined": False, "reason": "no_gaussian_memberships"}
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0 = np.clip(extract_gaussian_params(model), lo, hi)

    # Held-out split used only to *accept/reject* the refinement (never optimised).
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X_train))
    strat = y_arr if len(np.unique(y_arr)) > 1 else None
    try:
        tr_idx, val_idx = train_test_split(
            idx, test_size=val_fraction, random_state=seed, stratify=strat)
    except ValueError:                      # too few samples in a class to stratify
        tr_idx, val_idx = train_test_split(idx, test_size=val_fraction, random_state=seed)
    X_tr, y_tr = X_train.iloc[tr_idx], y_arr[tr_idx]
    X_val, y_val = X_train.iloc[val_idx], y_arr[val_idx]

    fitness = _make_classifier_fitness(model, X_tr, y_tr, l2_shrink, x0, lo, hi)
    init_fit = fitness(x0)
    init_val_acc = _classifier_accuracy(X_val, y_val, model)

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
        with _single_threaded():
            for sweep in range(n_sweeps):
                prev = cur
                for b in range(n_blocks):
                    bidx = np.arange(b * block, min((b + 1) * block, n_params))
                    sub_bounds = [(lo[k], hi[k]) for k in bidx]

                    def f_sub(v):
                        nonlocal n_eval
                        trial = x.copy()
                        trial[bidx] = v
                        n_eval += 1
                        return fitness(trial)

                    res = minimize(f_sub, x[bidx], method="L-BFGS-B", bounds=sub_bounds,
                                   options={"maxfun": sub_maxfun, "maxiter": sub_maxfun})
                    if res.fun < cur - 1e-12:
                        x[bidx] = np.clip(res.x, lo[bidx], hi[bidx])
                        cur = float(res.fun)
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

    # Accept only on a genuine held-out improvement (accuracy first, CE tiebreak).
    val_acc = _classifier_accuracy(X_val, y_val, refined)
    val_ce = _classifier_val_ce(X_val, y_val, refined)
    init_val_ce = _classifier_val_ce(X_val, y_val, model)
    accept = (val_acc > init_val_acc) or (val_acc == init_val_acc and val_ce < init_val_ce)
    out_model = refined if accept else model
    if verbose:
        verdict = "accepted" if accept else "rejected (kept heuristic)"
        print(f"  refinement {verdict}: val acc {init_val_acc:.4f} -> {val_acc:.4f}, "
              f"val CE {init_val_ce:.4f} -> {val_ce:.4f}")
    return out_model, {
        "refined": bool(accept),
        "init_val_acc": init_val_acc, "val_acc": val_acc,
        "init_val_ce": init_val_ce, "val_ce": val_ce,
        "init_train_obj": init_fit, **info,
    }
