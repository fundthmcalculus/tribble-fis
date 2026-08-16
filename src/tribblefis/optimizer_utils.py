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
