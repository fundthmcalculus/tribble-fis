"""Tests for `optimizer_utils` and the project-wide removal of direct
`scipy.optimize` usage (the user's explicit ask: every optimizer call site in
`tribblefis` should route through the in-house `optimizers` package instead).
"""

import ast
from pathlib import Path

import numpy as np

from tribblefis.optimizer_utils import optimizers_sub_solve, projected_gradient_solve

SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "tribblefis"


def _imports_scipy_optimize(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("scipy.optimize"):
            return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("scipy.optimize"):
                    return True
    return False


def test_no_module_imports_scipy_optimize_directly():
    """Every `src/tribblefis/*.py` file must be free of a direct
    `scipy.optimize` import. `optimizers` (the in-house package) may still use
    scipy internally -- that is out of this project's control -- but nothing
    in this codebase should import `scipy.optimize` itself anymore."""
    offenders = [
        str(p.relative_to(SRC_DIR))
        for p in SRC_DIR.rglob("*.py")
        if _imports_scipy_optimize(p)
    ]
    assert offenders == [], f"scipy.optimize imported directly by: {offenders}"


def test_optimizers_sub_solve_finds_quadratic_minimum():
    def quadratic(v):
        return float((v[0] - 0.3) ** 2 + (v[1] + 0.4) ** 2)

    res = optimizers_sub_solve(quadratic, np.array([0.0, 0.0]), [(-1.0, 1.0), (-1.0, 1.0)])
    np.testing.assert_allclose(res.x, [0.3, -0.4], atol=1e-3)
    assert res.fun < 1e-6


def test_optimizers_sub_solve_respects_box_bounds():
    def quadratic(v):
        return float((v[0] - 5.0) ** 2)

    # Unconstrained optimum (5.0) sits well outside the box; the result must
    # stay inside it regardless.
    res = optimizers_sub_solve(quadratic, np.array([0.0]), [(-1.0, 1.0)])
    assert -1.0 - 1e-9 <= res.x[0] <= 1.0 + 1e-9


def test_projected_gradient_solve_finds_quadratic_minimum():
    def quadratic_with_grad(v):
        f = float((v[0] - 0.3) ** 2 + (v[1] + 0.4) ** 2)
        g = np.array([2.0 * (v[0] - 0.3), 2.0 * (v[1] + 0.4)])
        return f, g

    res = projected_gradient_solve(
        quadratic_with_grad, np.array([0.0, 0.0]), [(-1.0, 1.0), (-1.0, 1.0)], max_evals=50,
    )
    np.testing.assert_allclose(res.x, [0.3, -0.4], atol=1e-3)
    assert res.fun < 1e-6


def test_projected_gradient_solve_never_worse_than_start():
    """A pathological non-convex objective must still not leave the solver
    worse off than its starting point -- the function falls back to `x0`
    whenever every line-search step is rejected."""
    def objective(v):
        # A step function-like landscape: any move away from x0 gets penalized
        # hard, so no line-search step should ever be accepted.
        f = 0.0 if np.allclose(v, [0.5, 0.5]) else 1e6
        return f, np.array([1.0, 1.0])  # a nonzero gradient that leads nowhere useful

    res = projected_gradient_solve(
        objective, np.array([0.5, 0.5]), [(-1.0, 1.0), (-1.0, 1.0)], max_evals=20,
    )
    assert res.fun <= 0.0 + 1e-9


def test_projected_gradient_solve_respects_evaluation_budget():
    calls = {"n": 0}

    def quadratic_with_grad(v):
        calls["n"] += 1
        f = float((v[0] - 0.3) ** 2)
        g = np.array([2.0 * (v[0] - 0.3)])
        return f, g

    max_evals = 7
    projected_gradient_solve(quadratic_with_grad, np.array([0.0]), [(-1.0, 1.0)], max_evals=max_evals)
    assert calls["n"] <= max_evals
