# The TSK consequent solve is an exact least-squares minimizer

**Claim being checked:** `regression.solve_tsk_consequents_from_firing` -- the
consequent half of `TribbleRegressor.fit`'s "full fit" -- says in its own
docstring that its single linear solve returns the *globally optimal* TSK
consequent coefficients, not an iterative approximation. This derives that
claim from the objective it actually minimizes, in `docs/derivations/
tsk_consequent_least_squares.py` (sympy, so every step below is emitted as
LaTeX rather than typeset by hand), and then checks the derivation against
the shipped code on concrete numbers. Run it with:

```
uv run --extra docs --extra dev python docs/derivations/tsk_consequent_least_squares.py
```

## How the full fit is structured

`TribbleRegressor.fit` (`gaussian_regressor.py:198`) is two stages with
different objectives:

1. **Antecedents** -- `(mu, sigma)` of every membership function, set by
   `create_gaussian_membership_dict` (KMeans + per-bucket `stats.norm.fit`).
   Heuristic, not fit against the regression loss.
2. **Consequents** -- the per-rule polynomial coefficients, set by
   `solve_tsk_consequents` -> `solve_tsk_consequents_from_firing`
   (`regression.py:838-1031`). *This* stage is where "least-squares optimal"
   applies, and only here: for **fixed** firing strengths (i.e. with the
   antecedents from stage 1 held constant), the model's output is linear in
   the consequent coefficients, so the exact minimizer of its loss has a
   closed form. The rest of this document derives that closed form and
   confirms the code implements it exactly.

## Part 1 -- what is being minimized

For sample `i`, rule `r` (`r = 0..R-1`), and basis term `k` (`k = 0..K`, with
`k=0` the constant column so `Phi[i,0] = 1`), the code's variables are:

| symbol below | code | meaning |
|---|---|---|
| `w[i,r]` | `norm_fs` | row-normalized firing strength (`_normalize_firing_strengths`) |
| `Phi[i,k]` | `phi = hstack([ones, feats])` | basis features, intercept prepended |
| `beta[r,k]` | `coeffs[r,:]` after `.reshape(n_rules, n_coeffs_per_rule)` | `k=0` is that rule's bucket mean, `k>=1` are the correction coefficients |
| `lambda` | `l2_reg` | ridge strength, applied to `k>=1` only -- `penalty[::n_coeffs_per_rule] = 0` never penalizes the intercept |

The prediction (`apply_tsk_consequents`/`rule_consequent_values`, firing-weighted
sum over rules of each rule's own crisp output):

```math
\hat{y}_i = \sum_{r=0}^{R-1} w_{i,r} \sum_{k=0}^{K} \Phi_{i,k}\,\beta_{r,k}
```

and the objective it is fit against -- firing-weighted MSE plus a ridge
penalty that spares every intercept column:

```math
J = \sum_{i=0}^{N-1} \left(y_i - \hat{y}_i\right)^{2}
    + \lambda \sum_{r=0}^{R-1}\sum_{k=1}^{K} \beta_{r,k}^{2}
```

(`docs/derivations/tsk_consequent_least_squares.py::part1_index_notation`).

## Part 2 -- stacking into one design matrix, and the normal equations

Flatten `(r, k)` into a single column index `c = r*(K+1) + k` -- row-major,
rule outer / per-rule-coefficient inner, i.e. exactly the order
`coeffs.reshape(n_rules, n_coeffs_per_rule)` uses -- and define the stacked
design matrix

```math
\Psi_{i,c} = w_{i,r(c)}\,\Phi_{i,k(c)}
```

This is precisely `design = (norm_fs[:, :, None] * phi[:, None, :]).reshape(N,
R*(K+1))` (`regression.py:952-956`): broadcasting the firing weight of rule
`r(c)` across that rule's whole feature block, for every rule, side by side.
With `D = diag(0 \text{ if } k(c){=}0 \text{ else } \lambda)` (`penalty` in
the code), the objective becomes the textbook ridge form

```math
J(\beta) = \left\lVert y - \Psi\beta \right\rVert_2^2 + \beta^T D \beta
```

**Claim:** `J` is minimized exactly where `(Psi^T Psi + D) beta = Psi^T y` --
the normal equations `solve_tsk_consequents_from_firing` sets up (as
`design`/`penalty`) and solves.

sympy cannot differentiate a quadratic form symbolically by an
abstract-*size* `MatrixSymbol` (`diff(expr, MatrixSymbol(...))` silently
returns `0` -- checked, not assumed), so the proof instead builds `Psi`,
`beta`, `D` out of independent plain scalar symbols at a concrete size (3
samples x 2 coefficients below) and differentiates term-by-term. Nothing in
the resulting identity depends on those particular sizes -- every one of the
11 free symbols is unconstrained -- so this is a proof of the general claim,
not a numeric example of it:

```math
J = \Psi_{00}^{2} \beta_{0}^{2} + 2 \Psi_{00} \Psi_{01} \beta_{0} \beta_{1} - 2 \Psi_{00} \beta_{0} y_{0} + \Psi_{01}^{2} \beta_{1}^{2} - 2 \Psi_{01} \beta_{1} y_{0} + \Psi_{10}^{2} \beta_{0}^{2} + 2 \Psi_{10} \Psi_{11} \beta_{0} \beta_{1} - 2 \Psi_{10} \beta_{0} y_{1} + \Psi_{11}^{2} \beta_{1}^{2} - 2 \Psi_{11} \beta_{1} y_{1} + \Psi_{20}^{2} \beta_{0}^{2} + 2 \Psi_{20} \Psi_{21} \beta_{0} \beta_{1} - 2 \Psi_{20} \beta_{0} y_{2} + \Psi_{21}^{2} \beta_{1}^{2} - 2 \Psi_{21} \beta_{1} y_{2} + \beta_{0}^{2} d_{0} + \beta_{1}^{2} d_{1} + y_{0}^{2} + y_{1}^{2} + y_{2}^{2}
```

`sympy.diff` against each `beta_i` and comparing term-by-term against
`2*(Psi^T Psi + D)*beta - 2*Psi^T y` confirms they are identical (`part2_
normal_equations`, asserted in-script), i.e.

```math
\left(2 D + 2 \Psi^{T} \Psi\right) \beta - 2 \Psi^{T} y = 0
\quad\Longleftrightarrow\quad
\left(D + \Psi^{T} \Psi\right) \beta = \Psi^{T} y
```

That right-hand equation is what `solve_tsk_consequents_from_firing` solves.

## Part 3 -- the augmented `lstsq` call is exact, not a shortcut

The code never forms `Psi^T Psi` directly (squaring it would square its
condition number). Instead, when `l2_reg > 0`, it stacks `sqrt(l2_reg *
penalty)` as extra diagonal rows onto `design` and zero rows onto `y`, then
calls `np.linalg.lstsq` on the augmented system (`regression.py:1013-1021`
and the pinned-column analogue at `995-1007`). This is exact because the
augmented residual norm *is* the ridge objective:

```math
J(\beta) = \left\lVert y - \Psi\beta \right\rVert_2^2 + \beta^T D \beta = \left\lVert \begin{bmatrix} y \\ 0 \end{bmatrix} - \begin{bmatrix} \Psi \\ \sqrt{D} \end{bmatrix} \beta \right\rVert_2^2
```

(`sympy.expand` on both sides and subtracting gives exactly `0` --
`part3_augmented_lstsq_equivalence`.) `lstsq`'s SVD-based solver is a global
minimizer of the left-hand residual norm regardless of rank -- unlike
forming and inverting `Psi^T Psi + D` by hand, it degrades gracefully
(minimum-norm solution) instead of blowing up when the design is
ill-conditioned, which is the whole reason the code takes this path instead
of a direct `solve`.

## Part 4 -- `pin_extremes` is still an exact solve

`pin_extremes=True` holds the first and last rule's intercept (bucket mean)
fixed at given values instead of letting the solve re-derive them
(`regression.py:915-930`). The code's approach: move each pinned column's
known contribution to the right-hand side and solve the *reduced* system for
the remaining ("free") coefficients only (`regression.py:984-1011`) --
substitution, the standard technique for linear-equality-constrained least
squares. That is only "the exact minimizer subject to the constraint" (as
the docstring claims) if it agrees with the full Lagrange/KKT solution of

```math
\text{minimize } J(\beta) \quad\text{s.t.}\quad C\beta = v
```

where `C` selects the pinned columns. Checked on a concrete instance sized
like a real (if tiny) two-rule TSK fit -- 6 samples, 2 rules, 1 feature,
`lambda = 0.1`, firing weights that actually sum to 1 per row:

Unconstrained ridge solve on that instance (`beta = [rule0 intercept, rule0
slope, rule1 intercept, rule1 slope]`):

```math
\beta = \left[\begin{matrix}\frac{11354}{5757}\\\frac{60}{707}\\\frac{146978}{40299}\\\frac{2320}{707}\end{matrix}\right]
```

Pinning `rule0`'s intercept to `1` and `rule1`'s to `13` (the observed
min/max, exactly what `partition_output`'s `y_bucket_mean` supplies) and
solving via substitution:

```math
\beta = \left[\begin{matrix}1\\\frac{13787}{19552}\\13\\- \frac{8421}{19552}\end{matrix}\right]
```

Solving the *same* problem via the full KKT system (stack `C`/`C^T` around
`Psi^T Psi + D` and solve for `[beta; mu]` directly, no substitution) gives
the identical rational vector -- `sympy` compares them exactly, not
approximately (`part4_pin_extremes_instance`). Substitution is not an
approximation of the constrained problem; it is one standard way to solve it.

## Part 5 -- the derivation matches the shipped function, not just the theory

Running `solve_tsk_consequents_from_firing` itself on the *same* firing
weights, feature column, targets, `lambda`, and pin values from Part 4:

```
code:     [ 1.          0.70514525 13.         -0.43069763]
symbolic: [ 1.          0.70514525 13.         -0.43069763]
```

agrees with the symbolic closed form to `5.6e-16` -- float64 round-off, not a
modeling gap. This is the strongest check available: not "the theory says
this closed form is optimal" but "the theory's closed form and the actual
function call return the same sixteen significant figures" (`part5_check_
against_shipped_code`).

## Where this does *not* apply

Everything above holds firing strengths fixed. It says nothing about
`optimize_tsk_coefficients` (the older L-BFGS-B path, still used where a
caller hasn't switched to the closed-form solver) or about the antecedents
-- `create_gaussian_membership_dict`'s `(mu, sigma)` are set by clustering
and per-bucket Gaussian fitting, never by the regression loss, so there is
no least-squares claim to make about them (`consequent-plan.md`'s Phase 2 is
exactly the proposal to eventually optimize that side too). "Least-squares
optimal" in `TribbleRegressor.fit` is a true but *scoped* claim: exact for
the consequents given the antecedents, silent about the antecedents
themselves.
