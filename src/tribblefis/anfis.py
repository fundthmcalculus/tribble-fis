"""ANFIS: a grid-partitioned Sugeno network trained with Jang's (1993) hybrid rule.

Every other TSK model in this package (`gaussian_regressor.py`, `regression.py`)
uses an *implicit* rule base: rules are never enumerated, because the mixture
model has exactly one rule per output label/bucket. That sidesteps the
classic ANFIS problem -- a rule for every *combination* of per-input terms,
which grows combinatorially -- but it also means nothing in the package
implements the textbook algorithm itself. `tribble-tree/HFIS_NOVELTY_REVIEW.md`
names ANFIS (Jang 1993) as the standing point of comparison for this project's
consequent-first design; this module is that comparison made literal and
runnable, not just cited.

**Architecture (Jang's five layers).** For `F` inputs, each partitioned into
`K_f` Gaussian terms, a rule is one specific combination of one term per
input -- the Cartesian product, `R = prod(K_f)` rules in total (`_build_rule_grid`).
Layer 1 evaluates every term's Gaussian membership; Layer 2 takes the rule
firing strength as the *product* t-norm across the chosen terms; Layer 3
row-normalizes; Layer 4 evaluates each rule's (Sugeno/TSK) linear consequent;
Layer 5 sums them, weighted by the Layer-3 weights.

**Why product, and only product.** The rest of the package supports five
De Morgan t-norm families and defaults to `"probability"` (`gauss_data.py`)
specifically because it is the one family that is smooth everywhere, which is
what makes an exact analytic gradient possible (`docs/norm-family-evaluation.md`,
`kernel.IncrementalFIS.supports_gradient`). ANFIS was defined with the product
t-norm (the "probability" family's T-half) from the start, and the batch
gradient below additionally depends on the *grid* structure factoring as a
literal product across features -- that is what lets the backward pass
marginalize a reshaped tensor instead of walking rules one at a time (see
`_premise_gradients`). Min/max or another family would break both properties,
so unlike the rest of the package this module does not expose a norm choice.

**Why the closed-form LSE solver is duplicated, not imported.**
`regression.solve_tsk_consequents` takes a `GaussianMixtureModel` and calls
`tsk_firing_strengths` on it internally; ANFIS's grid-Cartesian firing matrix
has no such model to hand it. `solve_anfis_consequents` below is the same
~20-line ridge normal-equations solve applied to a firing matrix the caller
already has -- the same call `fuzzytree/solve.py` makes for the same reason
(see that module's docstring). Everything else generic -- `build_consequent_features`
for the polynomial/orthogonal consequent basis, `_normalize_firing_strengths`
for the shared zero-firing convention, `_mse`/`_rsquared` for scoring -- is
imported, not re-derived.

**The hybrid rule, precisely.** Per epoch: (a) solve every rule's consequent
in closed form for the *current* premises (`solve_anfis_consequents`) -- exact,
because output is linear in the consequents for fixed firing strengths; then
(b) with those consequents held fixed, take one gradient step on every
premise parameter (`mu`, `sigma`) against the training MSE (`_premise_gradients`
+ `_adam_step`). Holding the consequents fixed during (b) is not an
approximation of the textbook rule, it *is* the textbook rule -- Jang's
"hybrid" name refers to exactly this alternation, LSE then gradient, each
half treating the other's parameters as constant. Contrast
`refine.py`'s `_fold_mse_and_grad`, which re-solves the consequents inside its
own gradient (because it is differentiating a *nested* optimum, envelope
theorem and all) for a fundamentally different reason: this module's gradient
never re-solves anything, so no such subtlety applies here.

**Why full-batch, vectorized, rather than the package's block coordinate
descent.** `refine.py`/`kernel.IncrementalFIS` move one membership function's
`(mu, sigma)` at a time and cache per-cell folds to make that cheap -- a good
fit for a *non-smooth* global search (GA/DE) over `min/max`, where nothing is
differentiable and there's no reason to update every parameter in lock-step.
ANFIS's premises are exactly the numbers a smooth loss can be differentiated
through *simultaneously*: reshaping the raw firing strengths as a
`(n, K_0, K_1, ..., K_{F-1})` tensor (valid precisely because they are a
literal product across features) turns "gradient w.r.t. every term on every
feature" into one reshape, one elementwise divide, and one sum-over-axes per
feature -- no Python loop over rules or membership functions, and no
per-parameter cache to maintain. That is the module's performance story: a
handful of vectorized array ops per epoch, cost `O(n * R * F)` like the
forward pass itself, not `O(n_epochs * n_premise_params)` fitness evaluations.

**The inherent limit.** `R = prod(K_f)` is exactly why the rest of the
package avoids grid partitioning. Past a handful of features this is the
wrong tool -- `init_anfis_model` raises `RuleExplosionError` rather than
silently building a slow, overfit model, and points at
`gaussian_regressor.MixtureOfGaussiansFuzzyRegressor` for the many-feature
case. Do not read this limit as "ANFIS is worse": per
`tribble-tree/HFIS_NOVELTY_REVIEW.md`, the point of this module is literal
correspondence with the textbook algorithm for interoperability and
comparison, and a genuinely different training regime (simultaneous premise
descent) worth having on its own terms -- not a claim that it beats the
mixture model's consequent-first solver, which was never in question here.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

from .regression import build_consequent_features, _normalize_firing_strengths, _mse, _rsquared

# Rules grow as prod(K_f); past this, the grid is the wrong tool (see module
# docstring). 5000 rules x a few thousand samples is already a multi-second
# consequent solve -- comfortably past where a user meant to ask for this.
_MAX_RULES = 5000

TSKOrder = typing.Literal["0th", "1st", "2nd", "3rd", "full-2nd"]


class RuleExplosionError(ValueError):
    """Raised when a grid partition's Cartesian rule count exceeds `_MAX_RULES`."""


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------

@dataclass
class ANFISModel:
    """A grid-partitioned Sugeno network: the trainable state ANFIS optimizes.

    `mu[f]`/`sigma[f]` are the Gaussian premise parameters for feature `f`'s
    `n_terms[f]` terms. `rule_grid` (built in `__post_init__`, not a
    constructor argument) is the `(n_rules, n_features)` Cartesian product of
    term indices -- `rule_grid[r, f]` is which of feature `f`'s terms rule `r`
    uses. `consequent` is `(n_rules, 1 + n_consequent_features)`: column 0 is
    each rule's intercept, the rest its `build_consequent_features` coefficients.
    """

    feature_names: tuple[str, ...]
    n_terms: tuple[int, ...]
    mu: list[np.ndarray]
    sigma: list[np.ndarray]
    consequent: np.ndarray
    order: TSKOrder = "1st"
    consequent_basis: str = "raw"

    def __post_init__(self) -> None:
        self.rule_grid = _build_rule_grid(self.n_terms)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_rules(self) -> int:
        return int(self.rule_grid.shape[0])

    def copy(self) -> "ANFISModel":
        return ANFISModel(
            feature_names=self.feature_names,
            n_terms=self.n_terms,
            mu=[a.copy() for a in self.mu],
            sigma=[a.copy() for a in self.sigma],
            consequent=self.consequent.copy(),
            order=self.order,
            consequent_basis=self.consequent_basis,
        )


def _build_rule_grid(n_terms: tuple[int, ...]) -> np.ndarray:
    """Cartesian product of per-feature term indices, one row per rule.

    Row order matches C-order `reshape(n, *n_terms)` of a flat `(n, R)` array
    -- the last feature varies fastest -- which is what lets
    `_premise_gradients` marginalize a reshaped tensor instead of grouping
    rules by hand.
    """
    if not n_terms:
        return np.zeros((0, 0), dtype=np.intp)
    grids = np.meshgrid(*[np.arange(k) for k in n_terms], indexing="ij")
    return np.stack([g.reshape(-1) for g in grids], axis=1)


# ---------------------------------------------------------------------------
# Forward pass (Layers 1-5).
# ---------------------------------------------------------------------------

def _gaussian(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Layer 1 for one feature: `(n_samples,) x (K,) -> (n_samples, K)`."""
    sigma_safe = np.maximum(sigma, 1e-6)
    z = (x[:, None] - mu[None, :]) / sigma_safe[None, :]
    return np.exp(-0.5 * z * z)


def term_memberships(model: ANFISModel, feature_arrays: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Layer 1: per-feature Gaussian term memberships, each `(n_samples, K_f)`."""
    return [
        _gaussian(np.asarray(feature_arrays[f], dtype=float), model.mu[fi], model.sigma[fi])
        for fi, f in enumerate(model.feature_names)
    ]


def raw_firing_strengths(model: ANFISModel, memberships: list[np.ndarray]) -> np.ndarray:
    """Layer 2: rule firing strengths under the product t-norm, `(n_samples, n_rules)`.

    Product only -- see the module docstring for why that is not a
    simplification here but the property the whole design leans on.
    """
    n = memberships[0].shape[0] if memberships else 0
    acc = np.ones((n, model.n_rules), dtype=float)
    for fi, g in enumerate(memberships):
        acc *= g[:, model.rule_grid[:, fi]]
    return acc


def _per_rule_predictions(model: ANFISModel, X_rule: np.ndarray) -> np.ndarray:
    """Layer 4: every rule's consequent evaluated at every sample, `(n, n_rules)`."""
    feats = build_consequent_features(X_rule, model.order, basis=model.consequent_basis)
    phi = np.hstack([np.ones((X_rule.shape[0], 1)), feats])  # (n, 1 + n_terms)
    return phi @ model.consequent.T


def anfis_predict(model: ANFISModel, X: pd.DataFrame) -> np.ndarray:
    """Layers 1-5 end to end: the model's prediction for `X`."""
    feature_arrays = {f: X[f].to_numpy(dtype=float) for f in model.feature_names}
    memberships = term_memberships(model, feature_arrays)
    raw_fs = raw_firing_strengths(model, memberships)
    norm_fs = _normalize_firing_strengths(raw_fs)
    X_rule = (
        np.column_stack([feature_arrays[f] for f in model.feature_names])
        if model.feature_names else np.empty((len(X), 0))
    )
    per_rule = _per_rule_predictions(model, X_rule)
    return np.sum(norm_fs * per_rule, axis=1)


# ---------------------------------------------------------------------------
# Consequent identification: closed-form ridge LSE (the "L" of the hybrid rule).
# ---------------------------------------------------------------------------

def solve_anfis_consequents(
    model: ANFISModel,
    X_rule: np.ndarray,
    norm_fs: np.ndarray,
    y: np.ndarray,
    l2_reg: float = 1e-6,
) -> np.ndarray:
    """The exact, ridge-regularized weighted-least-squares consequent solve.

    For fixed firing strengths, `y_hat = sum_r norm_fs[:, r] * phi @ consequent[r]`
    is linear in `consequent`, so stacking a per-rule design block
    `norm_fs[:, r, None] * phi` across all rules and solving the regularized
    normal equations gives the exact minimizer -- no iteration needed. See the
    module docstring for why this duplicates (rather than imports)
    `regression.solve_tsk_consequents`'s inner solve.
    """
    feats = build_consequent_features(X_rule, model.order, basis=model.consequent_basis)
    phi = np.hstack([np.ones((X_rule.shape[0], 1)), feats])  # (n, C)
    n_rules, n_coeffs = model.n_rules, phi.shape[1]
    design = (norm_fs[:, :, None] * phi[:, None, :]).reshape(X_rule.shape[0], n_rules * n_coeffs)

    penalty = np.ones(n_rules * n_coeffs)
    penalty[::n_coeffs] = 0.0  # never penalize the intercept

    if l2_reg > 0:
        sqrt_penalty = np.sqrt(l2_reg * penalty)
        design_aug = np.vstack([design, np.diag(sqrt_penalty)])
        y_aug = np.hstack([y, np.zeros_like(sqrt_penalty)])
        beta = np.linalg.lstsq(design_aug, y_aug, rcond=None)[0]
    else:
        beta = np.linalg.lstsq(design, y, rcond=None)[0]

    return beta.reshape(n_rules, n_coeffs)


# ---------------------------------------------------------------------------
# Premise identification: analytic batch gradient (the "GD" of the hybrid rule).
# ---------------------------------------------------------------------------

def _premise_gradients(
    model: ANFISModel,
    feature_cols: list[np.ndarray],
    memberships: list[np.ndarray],
    raw_fs: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
    per_rule_pred: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """`d(MSE)/d(mu_f[k], sigma_f[k])` for every premise parameter, consequents fixed.

    Derivation: let `w[n,r]` be the raw (pre-normalization) firing strength,
    `S[n] = sum_r w[n,r]`, `p[n,r] = w[n,r]/S[n]`, `y_hat[n] = sum_r p[n,r]
    f_r[n]` where `f_r` is rule `r`'s (fixed) consequent prediction. Then
    `d(y_hat[n])/d(w[n,r]) = (f_r[n] - y_hat[n]) / S[n]` (the standard
    normalized-weighted-average identity), so with `L = mean((y_hat - y)^2)`:

        dL/dw[n,r] = (2/n) (y_hat[n]-y[n]) * (f_r[n] - y_hat[n]) / S[n]

    `w[n,r] = prod_f g_f[n, k_f(r)]` is a literal product across features, so
    `dL/dg_f[n,k]`, summed over every rule using term `k` on feature `f`, is
    `sum_{r: k_f(r)=k} dL/dw[n,r] * w[n,r]/g_f[n,k]` (the leave-one-out
    product). Reshaping `dL/dw * w` as `(n, K_0, ..., K_{F-1})` -- valid
    because `_build_rule_grid`'s row order matches this reshape exactly --
    turns that per-term sum into "divide by this feature's axis, sum every
    other axis": one vectorized pass per feature, no rule-by-rule loop.
    """
    n = y.shape[0]
    row_sum = raw_fs.sum(axis=1)
    invalid = row_sum <= 1e-6
    safe_row_sum = np.where(invalid, 1.0, row_sum)

    resid = (2.0 / n) * (y_hat - y)
    dL_dw = resid[:, None] * (per_rule_pred - y_hat[:, None]) / safe_row_sum[:, None]
    dL_dw[invalid] = 0.0  # no rule fired -> this row contributes no gradient

    tp = dL_dw * raw_fs  # (n, R) = dL/dw[n,r] * w[n,r]
    shape = (n, *model.n_terms)
    tp_tensor = tp.reshape(shape)

    d_mu: list[np.ndarray] = []
    d_sigma: list[np.ndarray] = []
    for fi in range(model.n_features):
        g = memberships[fi]  # (n, K_f)
        safe_g = np.maximum(g, 1e-12)

        divisor_shape = [1] * tp_tensor.ndim
        divisor_shape[0] = n
        divisor_shape[fi + 1] = model.n_terms[fi]
        dL_dg = tp_tensor / safe_g.reshape(divisor_shape)
        sum_axes = tuple(a for a in range(1, tp_tensor.ndim) if a != fi + 1)
        dL_dg = dL_dg.sum(axis=sum_axes) if sum_axes else dL_dg  # (n, K_f)

        x = feature_cols[fi]
        mu, sigma = model.mu[fi], model.sigma[fi]
        sigma_safe = np.maximum(sigma, 1e-6)
        diff = x[:, None] - mu[None, :]
        dg_dmu = g * diff / (sigma_safe[None, :] ** 2)
        dg_dsigma = g * (diff ** 2) / (sigma_safe[None, :] ** 3)

        d_mu.append(np.sum(dL_dg * dg_dmu, axis=0))
        d_sigma.append(np.sum(dL_dg * dg_dsigma, axis=0))

    return d_mu, d_sigma


@dataclass
class _AdamState:
    m_mu: list[np.ndarray]
    v_mu: list[np.ndarray]
    m_sigma: list[np.ndarray]
    v_sigma: list[np.ndarray]
    t: int = 0

    @staticmethod
    def zeros_like(model: ANFISModel) -> "_AdamState":
        return _AdamState(
            m_mu=[np.zeros_like(a) for a in model.mu],
            v_mu=[np.zeros_like(a) for a in model.mu],
            m_sigma=[np.zeros_like(a) for a in model.sigma],
            v_sigma=[np.zeros_like(a) for a in model.sigma],
        )


def _adam_step(
    model: ANFISModel,
    state: _AdamState,
    d_mu: list[np.ndarray],
    d_sigma: list[np.ndarray],
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """One in-place Adam update of every premise parameter, simultaneously.

    Adam rather than Jang's original fixed/heuristic step-size rule: it needs
    no hand-tuned decay schedule to converge reliably, and every other
    "pragmatic default over textbook convention" choice in this package
    (the `probability` norm, the closed-form consequent solve itself) is
    justified the same way -- robustness over literal fidelity to 1993.
    """
    state.t += 1
    bias1 = 1 - beta1 ** state.t
    bias2 = 1 - beta2 ** state.t
    for fi in range(model.n_features):
        state.m_mu[fi] = beta1 * state.m_mu[fi] + (1 - beta1) * d_mu[fi]
        state.v_mu[fi] = beta2 * state.v_mu[fi] + (1 - beta2) * d_mu[fi] ** 2
        step_mu = lr * (state.m_mu[fi] / bias1) / (np.sqrt(state.v_mu[fi] / bias2) + eps)
        model.mu[fi] = model.mu[fi] - step_mu

        state.m_sigma[fi] = beta1 * state.m_sigma[fi] + (1 - beta1) * d_sigma[fi]
        state.v_sigma[fi] = beta2 * state.v_sigma[fi] + (1 - beta2) * d_sigma[fi] ** 2
        step_sigma = lr * (state.m_sigma[fi] / bias1) / (np.sqrt(state.v_sigma[fi] / bias2) + eps)
        model.sigma[fi] = np.maximum(model.sigma[fi] - step_sigma, 1e-6)


# ---------------------------------------------------------------------------
# Initialization: the grid-partition initial candidate.
# ---------------------------------------------------------------------------

def init_anfis_model(
    X: pd.DataFrame,
    feature_names: list[str],
    n_terms: int | list[int],
    order: TSKOrder = "1st",
    consequent_basis: str = "raw",
) -> ANFISModel:
    """Grid-partition every feature into `n_terms` quantile-spaced Gaussian terms.

    Centres are evenly spaced over the feature's observed range; sigma is set
    so adjacent terms cross at membership 0.5 (`exp(-0.5*(0.5*gap/s)^2) = 0.5`)
    -- the standard ANFIS "grid partition" initial candidate (Jang 1993,
    section V), the per-feature-independent analogue of
    `ruspini.ruspinize_model`'s shared-knot landmark heuristic. Raises
    `RuleExplosionError` before building anything if the Cartesian rule count
    would exceed `_MAX_RULES` -- see the module docstring.
    """
    if isinstance(n_terms, int):
        n_terms_list = [n_terms] * len(feature_names)
    else:
        n_terms_list = list(n_terms)
    if len(n_terms_list) != len(feature_names):
        raise ValueError("n_terms must be an int or one value per feature")
    if any(k < 1 for k in n_terms_list):
        raise ValueError(f"every feature needs at least 1 term, got {n_terms_list}")

    n_rules = int(np.prod(n_terms_list)) if n_terms_list else 0
    if n_rules > _MAX_RULES:
        raise RuleExplosionError(
            f"grid-partitioning {n_terms_list} terms across {len(feature_names)} "
            f"features would create {n_rules} rules (limit {_MAX_RULES}). ANFIS's "
            f"Cartesian rule base grows combinatorially with feature count -- "
            f"reduce the terms per feature or the feature count, or use "
            f"gaussian_regressor.MixtureOfGaussiansFuzzyRegressor, whose "
            f"implicit per-label rule base does not multiply this way."
        )

    mu: list[np.ndarray] = []
    sigma: list[np.ndarray] = []
    for f, k in zip(feature_names, n_terms_list):
        col = X[f].to_numpy(dtype=float)
        lo, hi = float(np.min(col)), float(np.max(col))
        if hi <= lo:
            hi = lo + 1.0
        centres = np.array([0.5 * (lo + hi)]) if k == 1 else np.linspace(lo, hi, k)
        gap = float(centres[1] - centres[0]) if k > 1 else (hi - lo)
        s = max(0.5 * gap / np.sqrt(2 * np.log(2)), 1e-6)
        mu.append(centres)
        sigma.append(np.full(k, s))

    dummy = np.zeros((1, len(feature_names)))
    n_consequent_terms = build_consequent_features(dummy, order, basis="raw").shape[1]
    consequent = np.zeros((max(n_rules, 1), 1 + n_consequent_terms))

    return ANFISModel(
        feature_names=tuple(feature_names),
        n_terms=tuple(n_terms_list),
        mu=mu,
        sigma=sigma,
        consequent=consequent,
        order=order,
        consequent_basis=consequent_basis,
    )


# ---------------------------------------------------------------------------
# Training: the hybrid learning loop.
# ---------------------------------------------------------------------------

def fit_anfis(
    model: ANFISModel,
    X: pd.DataFrame,
    y: np.ndarray,
    n_epochs: int = 200,
    learning_rate: float = 0.05,
    l2_reg: float = 1e-6,
    val_fraction: float = 0.2,
    patience: int = 15,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[ANFISModel, list[dict]]:
    """Jang's hybrid learning rule: alternate closed-form consequent LSE with
    a batch Adam step on every premise parameter, model-selecting on a held-out
    validation fold.

    The returned model is never worse (on that fold) than the untrained grid
    partition's own first consequent solve -- the same guard-rail invariant
    `refine.py`'s antecedent search enforces on itself: track the best
    validation score seen and return that snapshot, not necessarily the last
    epoch's.
    """
    if n_epochs < 1:
        raise ValueError("n_epochs must be >= 1 (at least one consequent solve is required)")

    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = max(1, int(round(val_fraction * n))) if n > 1 else 0
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if len(train_idx) == 0:  # degenerate tiny-n fallback: no split possible
        train_idx = val_idx = idx

    feature_arrays = {f: X[f].to_numpy(dtype=float) for f in model.feature_names}
    y_arr = np.asarray(y, dtype=float)

    def _rule_matrix(rows: np.ndarray) -> np.ndarray:
        return (
            np.column_stack([feature_arrays[f][rows] for f in model.feature_names])
            if model.feature_names else np.empty((len(rows), 0))
        )

    def _forward(rows: np.ndarray):
        feats = [feature_arrays[f][rows] for f in model.feature_names]
        memberships = [_gaussian(feats[fi], model.mu[fi], model.sigma[fi]) for fi in range(model.n_features)]
        raw_fs = raw_firing_strengths(model, memberships)
        norm_fs = _normalize_firing_strengths(raw_fs)
        return feats, memberships, raw_fs, norm_fs

    train_X_rule, val_X_rule = _rule_matrix(train_idx), _rule_matrix(val_idx)
    y_train, y_val = y_arr[train_idx], y_arr[val_idx]

    adam = _AdamState.zeros_like(model)
    history: list[dict] = []
    best_val = np.inf
    best_snapshot: ANFISModel | None = None
    stall = 0

    for epoch in range(n_epochs):
        train_feats, train_memberships, raw_train, norm_train = _forward(train_idx)
        model.consequent = solve_anfis_consequents(model, train_X_rule, norm_train, y_train, l2_reg=l2_reg)

        per_rule_train = _per_rule_predictions(model, train_X_rule)
        y_hat_train = np.sum(norm_train * per_rule_train, axis=1)
        train_mse = _mse(y_train, y_hat_train)

        _, _, _, norm_val = _forward(val_idx)
        per_rule_val = _per_rule_predictions(model, val_X_rule)
        y_hat_val = np.sum(norm_val * per_rule_val, axis=1)
        val_mse = _mse(y_val, y_hat_val)

        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        if verbose:
            print(f"epoch {epoch}: train_mse={train_mse:.6g} val_mse={val_mse:.6g}")

        if val_mse < best_val - 1e-12:
            best_val = val_mse
            best_snapshot = model.copy()
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

        d_mu, d_sigma = _premise_gradients(
            model, train_feats, train_memberships, raw_train, y_train, y_hat_train, per_rule_train,
        )
        _adam_step(model, adam, d_mu, d_sigma, lr=learning_rate)

    assert best_snapshot is not None  # epoch 0 always scores and snapshots
    return best_snapshot, history


def _describe_rules(model: ANFISModel) -> list[str]:
    """Human-readable `IF ... THEN ...` listing of every rule (interpretability)."""
    lines = []
    for r in range(model.n_rules):
        antecedent = " AND ".join(
            f"{f} is term_{model.rule_grid[r, fi]}" for fi, f in enumerate(model.feature_names)
        )
        coeffs = model.consequent[r]
        if model.order == "1st":
            terms = " + ".join(
                f"{c:.4g}*{f}" for c, f in zip(coeffs[1:], model.feature_names)
            )
            consequent = f"y = {coeffs[0]:.4g}" + (f" + {terms}" if terms else "")
        elif model.order == "0th":
            consequent = f"y = {coeffs[0]:.4g}"
        else:
            consequent = f"y = {coeffs[0]:.4g} + poly({model.order}, {model.n_features} features)"
        lines.append(f"IF {antecedent} THEN {consequent}")
    return lines


# ---------------------------------------------------------------------------
# scikit-learn estimator.
# ---------------------------------------------------------------------------

class ANFISRegressor(BaseEstimator, RegressorMixin):
    """Adaptive Neuro-Fuzzy Inference System (Jang, 1993).

    A grid-partitioned, Gaussian-premise Sugeno network trained with the
    canonical hybrid learning rule: closed-form consequent least squares
    alternated with a batch gradient step on every premise parameter. See the
    module docstring for the full derivation and how this relates to (and
    differs from) `gaussian_regressor.MixtureOfGaussiansFuzzyRegressor`.

    Because rules are the Cartesian product of per-feature terms, this scales
    to a handful of features with a few terms each -- past that, prefer the
    mixture regressor (or reduce `n_terms`); `fit` raises `RuleExplosionError`
    rather than silently building something impractically slow.
    """

    def __init__(
        self,
        n_terms: int | list[int] = 2,
        tsk_order: TSKOrder = "1st",
        consequent_basis: str = "raw",
        n_epochs: int = 200,
        learning_rate: float = 0.05,
        l2_reg: float = 1e-6,
        val_fraction: float = 0.2,
        patience: int = 15,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.n_terms = n_terms
        self.tsk_order = tsk_order
        self.consequent_basis = consequent_basis
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.val_fraction = val_fraction
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y) -> "ANFISRegressor":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            X = pd.DataFrame(np.asarray(X))
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
            X.columns = self.feature_names_in_

        y_array = np.asarray(y, dtype=float).flatten()
        X_array, y_array = check_X_y(X, y_array, multi_output=False, y_numeric=True)
        X_df = pd.DataFrame(X_array, columns=self.feature_names_in_)

        init_model = init_anfis_model(
            X_df, self.feature_names_in_, self.n_terms,
            order=self.tsk_order, consequent_basis=self.consequent_basis,
        )
        self.model_, self.history_ = fit_anfis(
            init_model, X_df, y_array,
            n_epochs=self.n_epochs, learning_rate=self.learning_rate,
            l2_reg=self.l2_reg, val_fraction=self.val_fraction,
            patience=self.patience, seed=self.random_state, verbose=self.verbose,
        )
        self.n_rules_ = self.model_.n_rules
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        return anfis_predict(self.model_, X_df)

    def score(self, X, y, sample_weight=None) -> float:
        return _rsquared(np.asarray(y, dtype=float), self.predict(X))

    def describe_rules(self) -> list[str]:
        """Human-readable `IF ... THEN ...` listing of every rule."""
        check_is_fitted(self)
        return _describe_rules(self.model_)
