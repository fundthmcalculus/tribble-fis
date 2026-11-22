"""In-house replacements for the small `scipy.optimize.minimize` sub-solves
this project used to scatter across `regression.py`, `refine.py`,
`it2_refine.py`, `gt2_refine.py`, and `trapz_math.py`.

Kept dependency-free of every other `tribblefis` module (only `numpy` and the
`optimizers` package) so it can be imported from anywhere without creating a
cycle -- `regression.py` in particular cannot import from `refine.py`, since
`refine.py` already imports from `regression.py`.

Two sub-problem shapes cover every call site that used to go through
`scipy.optimize.minimize`:

- No analytic gradient available (the objective is treated as a black box,
  usually because it runs a non-smooth t-norm forward pass or a discrete
  type-reduction search): `optimizers_sub_solve`, backed by the `optimizers`
  package's own local descent (`full_grad_optim`).
- An exact analytic gradient is available (the one bilevel-gradient block in
  `refine.refine_antecedents_coordinate`/`refine_classifier_antecedents`,
  issue #43): `projected_gradient_solve`, a small in-house projected-gradient
  descent with an Armijo backtracking line search -- the `optimizers` package
  has no way to accept a supplied Jacobian, so exploiting the gradient means
  not routing through it.

Neither of these makes the project scipy-free: `optimizers` itself imports
`scipy.optimize.minimize` (the very function backing `full_grad_optim`),
`scipy.special`, and `scipy.spatial`/`scipy.stats` internally. What this
module removes is every *direct* `scipy.optimize` import from this project's
own source.
"""

import typing

import numpy as np


class SubSolveResult(typing.NamedTuple):
    x: np.ndarray
    fun: float


def optimizers_sub_solve(fun: typing.Callable, x0: np.ndarray, bounds) -> SubSolveResult:
    """Bounded local descent for a gradient-free box-bounded sub-problem, via
    the in-house `optimizers` package instead of
    `scipy.optimize.minimize(method="L-BFGS-B")`.

    Uses `optimizers.continuous.local.full_grad_optim`, which jointly
    descends every parameter from `x0` -- `single_var_grad_optim` (descend
    one dimension at a time) was benchmarked and lands on a visibly worse
    optimum on a non-smooth block objective, so it is not a like-for-like
    replacement here even though it is the package's more commonly used
    "local polish".

    This path has no `maxfun`/`maxiter` knob (`optimizers`'s local solve does
    not expose one -- it always runs `scipy.optimize.minimize` to its own
    convergence, uncapped internally). Measured on 2-3 parameter box-bounded
    blocks it lands within the same order of magnitude of evaluations as an
    old `sub_maxfun`-capped scipy solve (same or better objective value,
    1x-2.5x the evaluation count).
    """
    from optimizers.continuous.local import full_grad_optim
    from optimizers.continuous.variables import InputContinuousVariable

    variables = [InputContinuousVariable(f"p{i}", float(lo), float(hi))
                 for i, (lo, hi) in enumerate(bounds)]
    x, fun_val = full_grad_optim(fun, np.asarray(x0, dtype=float), variables)
    return SubSolveResult(x=np.asarray(x, dtype=float), fun=float(fun_val))


def projected_gradient_solve(
    fun_grad: typing.Callable, x0: np.ndarray, bounds, max_evals: int = 25,
) -> SubSolveResult:
    """Bounded local descent for a sub-problem that supplies its own exact
    gradient, via projected gradient descent with an Armijo backtracking line
    search -- an in-house substitute for
    `scipy.optimize.minimize(method="L-BFGS-B", jac=True)`.

    `fun_grad(x)` must return `(value, gradient)`. Unlike `optimizers_sub_solve`
    (the finite-difference substitute above), this keeps an explicit, exactly
    enforced evaluation budget (`max_evals`), matching the old scipy call's
    `maxfun`/`maxiter` -- an analytic gradient makes every step informative
    enough that a plain descent-with-backtracking converges in a small,
    boundable number of evaluations, so there is no need to give that up the
    way the finite-difference path above does.

    Never returns a point worse (on `fun_grad`'s value) than `x0`: the
    starting point is the fallback if every line-search step is rejected or
    the budget is exhausted immediately.
    """
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    x = np.clip(np.asarray(x0, dtype=float), lo, hi)
    scale = float(np.mean(hi - lo)) or 1.0

    f, g = fun_grad(x)
    best_x, best_f = x, float(f)
    n_evals = 1
    step = 0.25 * scale
    min_step = 1e-10 * scale

    while n_evals < max_evals:
        gnorm = float(np.linalg.norm(g))
        if gnorm < 1e-12:
            break
        direction = g / gnorm
        t = step
        accepted = False
        while n_evals < max_evals and t >= min_step:
            x_new = np.clip(x - t * direction, lo, hi)
            f_new, g_new = fun_grad(x_new)
            n_evals += 1
            if f_new < f - 1e-4 * t * gnorm:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            break
        x, f, g = x_new, float(f_new), g_new
        if f < best_f:
            best_x, best_f = x, f
        step = min(t * 2.0, scale)

    return SubSolveResult(x=best_x, fun=best_f)


def projected_lbfgs_solve(
    fun_grad: typing.Callable, x0: np.ndarray, bounds, max_evals: int = 300,
    memory: int = 10, tol: float = 1e-9,
) -> SubSolveResult:
    """Box-bounded limited-memory BFGS for a problem with an exact gradient.

    The sibling of `projected_gradient_solve` for problems with more than a
    handful of parameters on very different curvature scales, where
    steepest descent zig-zags and stalls. Added for
    `temporal_aggregation.FuzzyTemporalAggregator` (issue #127): on its
    surrogate objective, steepest descent plateaued at a CV loss ~40% above
    what curvature-aware descent reaches from the same start and budget.

    Directions come from the standard two-loop L-BFGS recursion; each trial
    point is projected onto the box and accepted under an Armijo condition on
    the *projected* step. A direction that is not a descent direction resets
    the memory to steepest descent. Curvature pairs with ``s.y <= 0`` are
    skipped, so the implicit Hessian stays positive definite.

    Same contract as `projected_gradient_solve`: `fun_grad(x)` returns
    `(value, gradient)`, the evaluation budget is exact, and the result is
    never worse than `x0`.
    """
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    x = np.clip(np.asarray(x0, dtype=float), lo, hi)
    scale = float(np.mean(hi - lo)) or 1.0

    f, g = fun_grad(x)
    f = float(f)
    best_x, best_f = x, f
    n_evals = 1
    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []

    while n_evals < max_evals:
        if np.max(np.abs(x - np.clip(x - g, lo, hi))) < tol:
            break  # projected-gradient stationarity
        # Two-loop recursion.
        q = g.copy()
        rhos = [1.0 / float(np.dot(yv, sv)) for sv, yv in zip(s_hist, y_hist)]
        alphas = []
        for sv, yv, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rhos)):
            a = rho * float(np.dot(sv, q))
            alphas.append(a)
            q -= a * yv
        if s_hist:
            q *= float(np.dot(s_hist[-1], y_hist[-1])) / float(np.dot(y_hist[-1], y_hist[-1]))
        else:
            q *= 0.25 * scale / max(float(np.linalg.norm(g)), 1e-300)
        for sv, yv, rho, a in zip(s_hist, y_hist, rhos, reversed(alphas)):
            b = rho * float(np.dot(yv, q))
            q += (a - b) * sv
        d = -q
        # Do not push against an active bound.
        d = np.where(((x <= lo) & (d < 0)) | ((x >= hi) & (d > 0)), 0.0, d)
        if float(np.dot(g, d)) >= 0.0:
            s_hist.clear()
            y_hist.clear()
            d = -g * (0.25 * scale / max(float(np.linalg.norm(g)), 1e-300))

        t = 1.0
        accepted = False
        while n_evals < max_evals:
            x_new = np.clip(x + t * d, lo, hi)
            step = x_new - x
            if np.max(np.abs(step)) < 1e-14 * scale:
                break
            f_new, g_new = fun_grad(x_new)
            f_new = float(f_new)
            n_evals += 1
            if f_new <= f + 1e-4 * float(np.dot(g, step)):
                accepted = True
                break
            t *= 0.5
        if not accepted:
            if s_hist:          # retry once from steepest descent
                s_hist.clear()
                y_hist.clear()
                continue
            break
        yv = g_new - g
        if float(np.dot(step, yv)) > 1e-12:
            s_hist.append(step)
            y_hist.append(yv)
            if len(s_hist) > memory:
                s_hist.pop(0)
                y_hist.pop(0)
        x, f, g = x_new, f_new, g_new
        if f < best_f:
            best_x, best_f = x, f

    return SubSolveResult(x=best_x, fun=best_f)
