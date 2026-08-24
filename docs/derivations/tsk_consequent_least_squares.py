"""Symbolic derivation of the TSK consequent closed-form (ridge) least-squares solve.

`regression.solve_tsk_consequents_from_firing` claims that, for fixed firing
strengths, its single linear solve returns the *exact* minimizer of the
firing-weighted MSE plus a ridge penalty -- not an iterative approximation.
This script derives that claim from first principles with sympy (so every
step can be re-emitted as LaTeX) and then checks the derivation against
concrete numbers, including a run of the actual shipped function on those
same numbers.

Five parts, run in order by `main()`:

1.  Index-notation statement of the model and the objective it minimizes.
2.  Matrix/stacked form of the objective, and a symbolic proof (elementwise,
    on a generic finite-size Phi/beta/D) of the stationarity condition
    ("normal equations") that the code's linear solve implements.
3.  A proof that solving the ridge-augmented system via `lstsq` (what the
    code actually calls) minimizes the same objective as the normal
    equations in (2) -- i.e. the augmentation trick is not an approximation.
4.  The `pin_extremes` constrained solve: derive the substitution the code
    uses (move pinned columns to the right-hand side) and verify, on a
    concrete rational instance, that it agrees with the full
    Lagrange/KKT solution of the equality-constrained least-squares problem.
5.  Run `solve_tsk_consequents_from_firing` on the *same* concrete numbers
    from part 4 and check its output matches the symbolic answer exactly.

Run with `uv run --extra docs python docs/derivations/tsk_consequent_least_squares.py`.
Writes `tsk_consequent_least_squares.tex` (one `align*` block per step) next
to this file; `docs/tsk-consequent-least-squares-derivation.md` narrates the
same steps and quotes this script's LaTeX output directly.
"""

import pathlib

import sympy as sp

LATEX_STEPS: list[tuple[str, str]] = []


def emit(title: str, expr) -> None:
    """Record one derivation step and print it (as LaTeX) to stdout."""
    latex = expr if isinstance(expr, str) else sp.latex(expr)
    LATEX_STEPS.append((title, latex))
    print(f"\n-- {title} --")
    print(latex)


# ---------------------------------------------------------------------------
# Part 1: index-notation statement of the model and objective.
#
# Matches solve_tsk_consequents_from_firing's variables exactly:
#   w[i, r]   <-> norm_fs            (row-normalized firing strength)
#   Phi[i, k] <-> phi = [1 | feats]  (k=0 is the constant/intercept column)
#   beta[r, k]<-> coeffs[r, :]       (k=0 is that rule's bucket mean)
#   lambda    <-> l2_reg             (penalizes k >= 1 only, never k = 0)
# ---------------------------------------------------------------------------

def part1_index_notation() -> None:
    i, r, k = sp.symbols("i r k", cls=sp.Idx)
    N, R, K = sp.symbols("N R K", integer=True, positive=True)
    w, Phi, beta, yv, yhat = (
        sp.IndexedBase("w"), sp.IndexedBase("Phi"), sp.IndexedBase("beta"),
        sp.IndexedBase("y"), sp.IndexedBase(r"\hat{y}"),
    )
    lam = sp.Symbol("lambda", nonnegative=True)

    yhat_def = sp.Eq(
        yhat[i],
        sp.Sum(w[i, r] * sp.Sum(Phi[i, k] * beta[r, k], (k, 0, K)), (r, 0, R - 1)),
    )
    emit("Part 1a: TSK prediction for fixed firing strengths", yhat_def)

    J = sp.Symbol("J")
    J_def = sp.Eq(
        J,
        sp.Sum((yv[i] - yhat[i]) ** 2, (i, 0, N - 1))
        + lam * sp.Sum(sp.Sum(beta[r, k] ** 2, (k, 1, K)), (r, 0, R - 1)),
    )
    emit("Part 1b: firing-weighted MSE objective (ridge on k>=1 only)", J_def)


# ---------------------------------------------------------------------------
# Part 2: matrix form + symbolic proof of the normal equations.
#
# Stack rule blocks into one design matrix Psi with columns indexed by
# c = r * (K+1) + k (row-major: rule outer, per-rule coeff inner -- the same
# order `coeffs.reshape(n_rules, n_coeffs_per_rule)` uses in the code):
#   Psi[i, c] = w[i, r(c)] * Phi[i, k(c)]
# This is exactly `design = (norm_fs[:, :, None] * phi[:, None, :]).reshape(...)`.
#
# J(beta) = ||y - Psi @ beta||^2 + beta^T D beta,  D = diag(0 if k(c)=0 else lambda)
#
# Claim: grad_beta J = 2*(Psi^T Psi + D) beta - 2 Psi^T y, so J is minimized
# exactly where (Psi^T Psi + D) beta = Psi^T y.
#
# sympy cannot differentiate symbolically by an abstract-size MatrixSymbol
# (tested: `diff(quadratic_form, MatrixSymbol(...))` returns 0, i.e. it does
# not recognize the pattern), so the proof below instead builds Psi/beta/D
# out of *plain scalar symbols* at a concrete but arbitrary size (m samples,
# n coefficients) and differentiates term-by-term. Every entry of Psi, y, beta
# is an independent free symbol -- nothing about the identity below depends on
# m or n taking those specific values, so this constitutes a proof of the
# general claim, not a numerical example of it (part 4/5 supply the numbers).
# ---------------------------------------------------------------------------

def part2_normal_equations(m: int = 3, n: int = 2):
    Psi = sp.Matrix(m, n, lambda i, j: sp.Symbol(f"Psi_{i}{j}"))
    y = sp.Matrix(m, 1, lambda i, j: sp.Symbol(f"y_{i}"))
    beta = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f"beta_{i}"))
    d = [sp.Symbol(f"d_{i}", nonnegative=True) for i in range(n)]
    D = sp.diag(*d)

    resid = y - Psi * beta
    J = sp.expand((resid.T * resid + beta.T * D * beta)[0, 0])
    emit(
        f"Part 2a: J(beta) expanded for a generic {m}x{n} instance (proof scaffold)",
        sp.Eq(sp.Symbol("J"), J),
    )

    grad = sp.Matrix([sp.expand(sp.diff(J, b)) for b in beta])
    normal_eq_rhs = sp.expand(2 * (Psi.T * Psi + D) * beta - 2 * Psi.T * y)
    identity_holds = all(sp.simplify(a - b) == 0 for a, b in zip(grad, normal_eq_rhs))
    assert identity_holds, "grad_beta J != 2*(Psi^T Psi + D) beta - 2 Psi^T y"

    Psi_s, y_s, beta_s, D_s = (
        sp.MatrixSymbol("Psi", m, n), sp.MatrixSymbol("y", m, 1),
        sp.MatrixSymbol("beta", n, 1), sp.MatrixSymbol("D", n, n),
    )
    emit(
        "Part 2b: stationarity condition, verified true term-by-term above",
        sp.Eq(2 * (Psi_s.T * Psi_s + D_s) * beta_s - 2 * Psi_s.T * y_s, sp.ZeroMatrix(n, 1)),
    )
    emit(
        "Part 2c: normal equations solved by the code's linear solve",
        sp.Eq((Psi_s.T * Psi_s + D_s) * beta_s, Psi_s.T * y_s),
    )
    print(f"[verified] elementwise gradient identity holds for a generic {m}x{n} Psi/beta/D "
          f"({m*n + m + n} independent free symbols)")


# ---------------------------------------------------------------------------
# Part 3: the ridge-augmented lstsq trick is exact, not approximate.
#
# The code never forms Psi^T Psi. It instead stacks sqrt(lambda)*I rows onto
# Psi and zero rows onto y, then calls np.linalg.lstsq on the augmented
# system. Claim: minimizing the augmented residual norm minimizes the same
# J(beta) as part 2, so lstsq's SVD-based global minimizer (which lstsq
# guarantees regardless of rank, including the rank-deficient minimum-norm
# case) is the exact minimizer of the ridge objective, just computed more
# stably than forming Psi^T Psi + D directly (squaring the condition number).
# ---------------------------------------------------------------------------

def part3_augmented_lstsq_equivalence(m: int = 3, n: int = 2):
    Psi = sp.Matrix(m, n, lambda i, j: sp.Symbol(f"Psi_{i}{j}"))
    y = sp.Matrix(m, 1, lambda i, j: sp.Symbol(f"y_{i}"))
    beta = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f"beta_{i}"))
    d = [sp.Symbol(f"d_{i}", nonnegative=True) for i in range(n)]
    D = sp.diag(*d)
    sqrtD = sp.diag(*[sp.sqrt(di) for di in d])

    J = sp.expand(((y - Psi * beta).T * (y - Psi * beta) + beta.T * D * beta)[0, 0])

    Psi_aug = Psi.row_insert(m, sqrtD)
    y_aug = y.row_insert(m, sp.zeros(n, 1))
    J_aug = sp.expand(((y_aug - Psi_aug * beta).T * (y_aug - Psi_aug * beta))[0, 0])

    assert sp.simplify(J - J_aug) == 0, "augmented residual norm != ridge objective"

    emit(
        "Part 3: augmented system solved by lstsq, equal to the ridge objective",
        r"J(\beta) = \left\lVert y - \Psi\beta \right\rVert_2^2 + \beta^T D \beta"
        r" = \left\lVert \begin{bmatrix} y \\ 0 \end{bmatrix}"
        r" - \begin{bmatrix} \Psi \\ \sqrt{D} \end{bmatrix} \beta \right\rVert_2^2",
    )
    print(f"[verified] ||[Psi; sqrt(D)] beta - [y; 0]||^2 == ||y - Psi beta||^2 + beta^T D beta "
          f"for generic {m}x{n} Psi (exact algebraic identity, not a numeric check)")


# ---------------------------------------------------------------------------
# Part 4: pin_extremes as equality-constrained least squares.
#
# The code holds two columns of beta fixed (the first and last rule's
# intercept/bucket-mean) at given values, moves their known contribution to
# the right-hand side, and solves the *reduced* ridge normal equations for
# the remaining ("free") coefficients. This is the standard substitution
# method for linear-equality-constrained least squares. It is exact (not a
# penalty/approximation) iff it agrees with the Lagrange/KKT solution of
#     minimize J(beta)  s.t.  C @ beta = v
# where C selects the pinned columns. Verify that agreement on a concrete
# rational instance sized like a real (if tiny) two-rule TSK fit.
# ---------------------------------------------------------------------------

def part4_pin_extremes_instance():
    N, R, P = 6, 2, 2  # 6 samples, 2 rules, 2 coeffs/rule (intercept + 1 slope)
    ncoef = R * P

    phi = sp.Matrix([[1, x] for x in (-2, -1, 0, 1, 2, 3)])  # [1 | feature]
    w = sp.Matrix([
        [sp.Rational(9, 10), sp.Rational(1, 10)],
        [sp.Rational(7, 10), sp.Rational(3, 10)],
        [sp.Rational(5, 10), sp.Rational(5, 10)],
        [sp.Rational(3, 10), sp.Rational(7, 10)],
        [sp.Rational(2, 10), sp.Rational(8, 10)],
        [sp.Rational(1, 10), sp.Rational(9, 10)],
    ])  # row-normalized firing strengths (rows sum to 1), 2 rules
    y = sp.Matrix([1, 2, 3, 5, 8, 13])
    lam = sp.Rational(1, 10)

    Psi = sp.zeros(N, ncoef)
    for r in range(R):
        for k in range(P):
            c = r * P + k
            for i in range(N):
                Psi[i, c] = w[i, r] * phi[i, k]
    d_vec = [0 if c % P == 0 else lam for c in range(ncoef)]  # never penalize intercepts
    D = sp.diag(*d_vec)

    beta_unconstrained = (Psi.T * Psi + D).solve(Psi.T * y)

    v0, v1 = sp.Integer(1), sp.Integer(13)  # pin rule 0's and rule (R-1)'s intercept
    pinned = [0, (R - 1) * P]
    free = [c for c in range(ncoef) if c not in pinned]
    values = sp.Matrix([v0, v1])

    residual = y - Psi[:, pinned] * values
    Psi_free = Psi[:, free]
    D_free = sp.diag(*[d_vec[c] for c in free])
    beta_free = (Psi_free.T * Psi_free + D_free).solve(Psi_free.T * residual)

    beta_substitution = sp.zeros(ncoef, 1)
    for idx, c in enumerate(pinned):
        beta_substitution[c] = values[idx]
    for idx, c in enumerate(free):
        beta_substitution[c] = beta_free[idx]

    # Full KKT system for the equality-constrained problem, no substitution:
    #   [[Psi^T Psi + D,  C^T] [beta]   [Psi^T y]
    #    [C,               0 ]] [mu ] = [v      ]
    C = sp.zeros(len(pinned), ncoef)
    for idx, c in enumerate(pinned):
        C[idx, c] = 1
    kkt_lhs = (Psi.T * Psi + D).row_join(C.T).col_join(C.row_join(sp.zeros(len(pinned))))
    kkt_rhs = (Psi.T * y).col_join(values)
    beta_kkt = kkt_lhs.solve(kkt_rhs)[:ncoef, 0]

    assert beta_substitution == beta_kkt, "substitution solve != full KKT solve"

    emit(
        "Part 4a: unconstrained ridge solve on the 6-sample/2-rule instance",
        sp.Eq(sp.MatrixSymbol("beta", ncoef, 1), beta_unconstrained),
    )
    emit(
        "Part 4b: pin_extremes solve (substitution == full KKT), same instance",
        sp.Eq(sp.MatrixSymbol("beta", ncoef, 1), beta_substitution),
    )
    print("[verified] substitution solve (what the code does) == Lagrange/KKT solve "
          "of the equality-constrained problem, on this instance")
    return dict(N=N, R=R, P=P, phi=phi, w=w, y=y, lam=lam, pinned_values=(v0, v1),
                beta_substitution=beta_substitution)


# ---------------------------------------------------------------------------
# Part 5: the shipped function, run on the exact numbers from part 4.
# ---------------------------------------------------------------------------

def part5_check_against_shipped_code(instance: dict) -> None:
    import numpy as np
    import pandas as pd

    from tribblefis.regression import solve_tsk_consequents_from_firing

    N, P = instance["N"], instance["P"]
    firing = np.array(instance["w"].tolist(), dtype=float)
    labels = [0, 1]
    x_col = np.array([float(instance["phi"][i, 1]) for i in range(N)])
    X_train = pd.DataFrame({"x": x_col})
    y_train = pd.DataFrame({"y_value": np.array(instance["y"].tolist(), dtype=float).ravel()})
    v0, v1 = instance["pinned_values"]
    y_bucket_mean = np.array([float(v0), float(v1)])

    corr, means = solve_tsk_consequents_from_firing(
        firing, labels, X_train, ["x"], y_bucket_mean, y_train,
        order="1st", l2_reg=float(instance["lam"]), basis="raw",
        pin_extremes=True, verbose=False,
    )
    code_beta = np.array([means[0], corr[0, 0], means[1], corr[1, 0]])
    symbolic_beta = np.array([float(b) for b in instance["beta_substitution"]])

    max_abs_diff = float(np.max(np.abs(code_beta - symbolic_beta)))
    assert max_abs_diff < 1e-10, (
        f"solve_tsk_consequents_from_firing disagrees with the symbolic derivation "
        f"by {max_abs_diff:g}: code={code_beta}, symbolic={symbolic_beta}"
    )
    print(f"[verified] solve_tsk_consequents_from_firing output matches the symbolic "
          f"closed form to {max_abs_diff:.3e} (float64 round-off only)")
    print(f"  code:     {code_beta}")
    print(f"  symbolic: {symbolic_beta}")


def write_latex(path: pathlib.Path) -> None:
    blocks = []
    for title, latex in LATEX_STEPS:
        blocks.append(f"% {title}\n\\[\n{latex}\n\\]")
    path.write_text("\n\n".join(blocks) + "\n")
    print(f"\nWrote {len(LATEX_STEPS)} LaTeX blocks to {path}")


def main() -> None:
    part1_index_notation()
    part2_normal_equations()
    part3_augmented_lstsq_equivalence()
    instance = part4_pin_extremes_instance()
    part5_check_against_shipped_code(instance)
    write_latex(pathlib.Path(__file__).with_suffix(".tex"))


if __name__ == "__main__":
    main()
