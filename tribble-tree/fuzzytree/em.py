"""Expectation-Maximization refinement for ``HierarchicalFuzzyExperts*`` models.

Implements the design in ``EM_REFINEMENT.md``: replace the HME's one-shot
greedy build (gate-only responsibilities, soft-inclusion training subsets)
with an EM loop that alternates:

  E-step -- posterior leaf responsibilities that combine the gate probability
            *and* the expert's likelihood of the observed target, so a
            well-fitting expert can "claim" a sample a neighbouring expert
            would otherwise keep purely on gate proximity.
  M-step -- refit every expert and every gate to maximize the expected
            complete-data log-likelihood under those posteriors, holding the
            tree *structure* (which variable gates where, and how many
            leaves) fixed -- only parameters move. Structural EM (growing or
            pruning gates mid-refinement) is explicitly out of scope; see
            EM_REFINEMENT.md Sec.9.

Only ``fuzzytree`` is touched; nothing under ``src/tribblefis`` changes.

Expert M-step, by expert kind:
  * Regression sub-FIS (``TribbleRegressor``): the antecedent
    Gaussian memberships are frozen (re-fitting them weighted would need a
    weighted GMM, which the upstream estimator has no hook for); only the
    closed-form TSK consequents are refit, weighted by the posterior, mirroring
    ``solve.solve_leaf_consequents`` but reusing the *frozen* rule firing
    matrix from the expert's own ``model_`` via ``tsk_firing_strengths``. This
    is "Option A" from EM_REFINEMENT.md Sec.5, restricted to the cheap
    consequents-only variant the doc recommends.
  * Constant-fallback experts (tiny leaves): a straight posterior-weighted mean.
  * Classification sub-FIS (``TribbleClassifier``): zeroth-order
    -- the antecedents *are* the whole model, so there is no separate
    consequent to weight-refit. Its M-step is "Option B": a posterior-weighted
    bootstrap resample followed by an ordinary (unmodified) refit.

Gate M-step: each internal node's linguistic terms are re-estimated from the
posterior branch responsibilities. Gaussian gate terms get the closed-form
weighted mean/variance update (EM_REFINEMENT.md Sec.4.2); trapezoid gate terms
keep their shape but their knots are recomputed from responsibility-weighted
quantiles (Sec.5, "Option B" for gates). A branch whose responsibility mass
falls below ``min_mass`` is frozen (with a warning) rather than collapsed or
reinitialised -- full structural pruning/reinit is future work (Sec.9).
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from tribblefis.gauss_data import GaussianMembership
from tribblefis.gauss_math import tsk_firing_strengths
from tribblefis.regression import _normalize_firing_strengths, build_consequent_features

from .hme import _AlignedClassifier, _ConstantClassifier, _ConstantRegressor, compute_responsibilities
from .node import FuzzyTreeNode
from .terms import _trapezoid_terms, _weighted_quantiles

_EPS = 1e-12


# --------------------------------------------------------------------------
# E-step
# --------------------------------------------------------------------------
def _regression_log_lik(experts, sigma2, X_full, y, n_leaves):
    n = len(X_full)
    loglik = np.full((n, n_leaves), -np.inf)
    for leaf_id, expert in experts.items():
        mu = np.asarray(expert.predict(X_full), dtype=float)
        var = max(float(sigma2.get(leaf_id, 1.0)), _EPS)
        loglik[:, leaf_id] = -0.5 * np.log(2 * np.pi * var) - 0.5 * (y - mu) ** 2 / var
    return loglik


def _classification_log_lik(experts, X_full, y_idx, n_leaves):
    n = len(X_full)
    loglik = np.full((n, n_leaves), np.log(_EPS))
    rows = np.arange(n)
    for leaf_id, expert in experts.items():
        proba = expert.predict_proba_aligned(X_full)
        p = np.clip(proba[rows, y_idx], _EPS, 1.0)
        loglik[:, leaf_id] = np.log(p)
    return loglik


def e_step(R: np.ndarray, log_lik: np.ndarray) -> tuple[np.ndarray, float]:
    """Posterior responsibilities ``H`` and incomplete-data log-likelihood ``L``.

    ``R`` is the gate-only path product (``compute_responsibilities``); the
    posterior multiplies it by the expert likelihood and row-normalises in log
    space (log-sum-exp) so deep trees don't underflow.
    """
    log_pi = np.log(np.clip(R, _EPS, 1.0))
    log_num = log_pi + log_lik
    log_norm = logsumexp(log_num, axis=1, keepdims=True)
    H = np.exp(log_num - log_norm)
    L = float(log_norm.sum())
    return H, L


# --------------------------------------------------------------------------
# Gate M-step: rebuild every internal node's terms from branch responsibilities
# --------------------------------------------------------------------------
def _rebuild_gate_tree(
    node: FuzzyTreeNode, X_gate: pd.DataFrame, H: np.ndarray, var_floor: float, min_mass: float
) -> FuzzyTreeNode:
    if node.is_leaf:
        return node

    child_leaf_ids = [[ln.leaf_id for ln in child.iter_leaves()] for child in node.children]
    taus = [H[:, ids].sum(axis=1) for ids in child_leaf_ids]
    gamma = sum(taus)
    col = X_gate[node.split_var].to_numpy(dtype=float)
    labels = [lbl for lbl, _ in node.terms]
    old_mfs = [mf for _, mf in node.terms]
    is_gaussian = isinstance(old_mfs[0], GaussianMembership)

    new_mfs = []
    if is_gaussian:
        for tau, old_mf in zip(taus, old_mfs):
            m = float(tau.sum())
            if m <= min_mass:
                warnings.warn(
                    f"EM gate update: branch at path {node.path + (labels[len(new_mfs)],)} "
                    f"has responsibility mass {m:.3g} <= min_mass; freezing its term."
                )
                new_mfs.append(old_mf)
                continue
            mu = float((tau * col).sum() / m)
            var = float((tau * (col - mu) ** 2).sum() / m)
            sigma = max(float(np.sqrt(var)), float(np.sqrt(var_floor)))
            new_mfs.append(GaussianMembership.create(mu=mu, sigma=sigma))
    else:
        # Trapezoid gate M-step: Use smooth trapezoid optimization then extract crisp parameters.
        # The smooth approximation avoids the mode-hugging pathology of piecewise-linear objectives
        # while preserving crisp trapezoid behavior in the final gates.
        #
        # We weight the data using the gate responsibilities (gamma) by resampling, then fit
        # smooth trapezoids, which gives us better knot optimization than quantile-based approaches.
        n_terms = len(old_mfs)
        total_gamma = float(gamma.sum())
        if total_gamma <= min_mass:
            new_mfs = old_mfs
        else:
            # Weighted data: resample according to gamma (normalized to probabilities)
            from tribblefis.trapz_math_smooth import fit_smooth_trapezoids_em

            p = gamma / total_gamma
            n_resample = max(int(round(total_gamma)), 100)  # At least 100 samples
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            idx_resampled = rng.choice(len(col), size=n_resample, replace=True, p=p)
            col_resampled = col[idx_resampled]

            # Fit smooth trapezoids to weighted sample
            memberships, weights, _ = fit_smooth_trapezoids_em(
                col_resampled, n_components=n_terms, n_bins=50, max_iter=20, tol=1e-4, shape="trapezoid"
            )
            new_mfs = memberships

    new_terms = list(zip(labels, new_mfs))
    new_children = [
        _rebuild_gate_tree(child, X_gate, H, var_floor, min_mass) for child in node.children
    ]
    return FuzzyTreeNode.create_internal(
        depth=node.depth,
        path=node.path,
        split_var=node.split_var,
        terms=new_terms,
        children=new_children,
        soft_mass=float(gamma.sum()),
    )


# --------------------------------------------------------------------------
# Expert M-step: regression (Option A -- weighted consequent-only refit)
# --------------------------------------------------------------------------
def _refit_regression_expert(expert, X_full: pd.DataFrame, y: np.ndarray, h: np.ndarray, var_floor: float) -> float:
    m = float(h.sum())
    if m <= _EPS:
        return var_floor

    if isinstance(expert, _ConstantRegressor):
        expert.value = float((h * y).sum() / m)
        resid = y - expert.predict(X_full)
        return max(float((h * resid**2).sum() / m), var_floor)

    X_top = X_full[expert.top_features_]
    firing, _labels = tsk_firing_strengths(X_top, expert.model_, norms=expert._norms())
    norm_fs = _normalize_firing_strengths(firing)
    n_rules = norm_fs.shape[1]

    feats = build_consequent_features(X_top.to_numpy(), expert.tsk_order, basis=expert.consequent_basis)
    n_terms = feats.shape[1]
    n_coeffs = 1 + n_terms
    phi = np.hstack([np.ones((len(X_top), 1)), feats])
    design = (norm_fs[:, :, np.newaxis] * phi[:, np.newaxis, :]).reshape(len(X_top), n_rules * n_coeffs)

    sw = np.sqrt(np.clip(h, 0.0, None))
    design_w = design * sw[:, None]
    y_w = y * sw

    penalty = np.ones(n_rules * n_coeffs)
    penalty[::n_coeffs] = 0.0
    l2_reg = getattr(expert, "l2_reg", 1e-6)
    sqrt_penalty = np.sqrt(l2_reg * penalty)
    design_aug = np.vstack([design_w, np.diag(sqrt_penalty)])
    y_aug = np.hstack([y_w, np.zeros_like(sqrt_penalty)])
    beta = np.linalg.lstsq(design_aug, y_aug, rcond=None)[0]

    coeffs = beta.reshape(n_rules, n_coeffs)
    expert.y_bucket_mean_ = coeffs[:, 0].copy()
    expert.corr_terms_ = coeffs[:, 1:].copy() if n_terms > 0 else np.zeros((n_rules, 0))

    resid = y - expert.predict(X_full)
    return max(float((h * resid**2).sum() / m), var_floor)


# --------------------------------------------------------------------------
# Expert M-step: classification (Option B -- weighted importance resampling)
# --------------------------------------------------------------------------
def _refit_classification_expert(expert, X_full, y_idx, classes, h, rng, min_mass):
    m = float(h.sum())

    if isinstance(expert, _ConstantClassifier):
        n_classes = len(classes)
        proba = np.array([h[y_idx == c].sum() for c in range(n_classes)], dtype=float)
        total = proba.sum()
        expert.proba_ = proba / total if total > _EPS else np.full(n_classes, 1.0 / n_classes)
        return expert

    if m <= min_mass:
        warnings.warn(
            f"EM expert update: leaf has responsibility mass {m:.3g} <= min_mass; "
            "freezing its expert."
        )
        return expert

    p = h / m
    n_resample = max(int(round(m)), 30)
    idx = rng.choice(len(X_full), size=n_resample, replace=True, p=p)
    y_resampled = classes[y_idx[idx]]
    if len(np.unique(y_resampled)) < 2:
        return expert  # can't refit on a single-class resample; keep frozen

    new_clf = clone(expert.clf)
    new_clf.fit(X_full.iloc[idx], y_resampled)
    return _AlignedClassifier(new_clf, expert.global_classes)


# --------------------------------------------------------------------------
# Top-level drivers
# --------------------------------------------------------------------------
def refine_em_regressor(
    model,
    X,
    y,
    max_iter: int = 15,
    tol: float = 1e-4,
    var_floor: float = 1e-6,
    min_mass: float = 1.0,
    verbose: bool = False,
):
    """Refine a fitted ``HierarchicalFuzzyExpertsRegressor`` in place via EM.

    Warm-starts from the model's current (greedy-built) tree and experts.
    Mutates and returns ``model``; also sets ``model.sigma2_``,
    ``model.em_log_likelihood_`` (per-iteration incomplete-data log-lik, for
    the Sec.10 monotonicity check) and ``model.em_iterations_``.

    Each M-step here is an exact weighted refit (closed-form ridge consequents
    and, for Gaussian gates, closed-form weighted moments), so the
    incomplete-data log-likelihood should be non-decreasing in practice. As a
    safeguard against an approximate/degenerate step regardless (e.g. a
    starved branch, or trapezoid gates whose knot-based ramps are only a loose
    stand-in for a true weighted MLE), the model's parameters are rolled back
    to whichever iteration had the best observed log-likelihood before
    returning -- the Sec.10 "no-worse guarantee".
    """
    check_is_fitted(model)
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=model.feature_names_in_)
    X_df = X_df.reset_index(drop=True)
    y_value = np.asarray(y, dtype=float).flatten()
    X_gate = X_df[model.gate_features_]

    if not hasattr(model, "sigma2_") or not model.sigma2_:
        R0 = compute_responsibilities(model.tree_, X_gate, model.n_leaves_)
        model.sigma2_ = {}
        for leaf_id, expert in model.experts_.items():
            resid = y_value - expert.predict(X_df)
            w = R0[:, leaf_id]
            wsum = w.sum()
            var = float((w * resid**2).sum() / wsum) if wsum > _EPS else float(np.var(resid))
            model.sigma2_[leaf_id] = max(var, var_floor)

    history: list[float] = []
    best_L = -np.inf
    best_state = None
    for it in range(max_iter):
        R = compute_responsibilities(model.tree_, X_gate, model.n_leaves_)
        log_lik = _regression_log_lik(model.experts_, model.sigma2_, X_df, y_value, model.n_leaves_)
        H, L = e_step(R, log_lik)
        history.append(L)
        if verbose:
            print(f"EM iter {it}: L={L:.4f}")
        if L > best_L:
            best_L = L
            best_state = (model.tree_, copy.deepcopy(model.experts_), dict(model.sigma2_))
        if it > 0 and (L - history[-2]) < tol * max(abs(history[-2]), 1.0):
            break

        for leaf_id, expert in model.experts_.items():
            model.sigma2_[leaf_id] = _refit_regression_expert(
                expert, X_df, y_value, H[:, leaf_id], var_floor
            )

        model.tree_ = _rebuild_gate_tree(model.tree_, X_gate, H, var_floor, min_mass)

    if best_state is not None:
        model.tree_, model.experts_, model.sigma2_ = best_state

    model.em_log_likelihood_ = history
    model.em_iterations_ = len(history)
    return model


def refine_em_classifier(
    model,
    X,
    y,
    max_iter: int = 15,
    tol: float = 1e-4,
    min_mass: float = 1.0,
    random_state: int | None = None,
    verbose: bool = False,
):
    """Refine a fitted ``HierarchicalFuzzyExpertsClassifier`` in place via EM.

    Mutates and returns ``model``; sets ``model.em_log_likelihood_`` and
    ``model.em_iterations_``. Expert refits use importance resampling (see
    module docstring) since the classifier sub-FIS has no separable consequent
    to weight-refit while freezing antecedents -- that M-step is stochastic
    and only *approximates* the weighted MLE, so unlike the regression path it
    can occasionally decrease the log-likelihood. As a safeguard the model's
    parameters are rolled back to whichever iteration had the best observed
    log-likelihood before returning -- the Sec.10 "no-worse guarantee".
    """
    check_is_fitted(model)
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=model.feature_names_in_)
    X_df = X_df.reset_index(drop=True)
    y_arr = np.asarray(y).flatten()
    class_to_idx = {c: i for i, c in enumerate(model.classes_)}
    y_idx = np.array([class_to_idx[c] for c in y_arr])
    X_gate = X_df[model.gate_features_]
    rng = np.random.default_rng(random_state if random_state is not None else model.random_state)

    history: list[float] = []
    best_L = -np.inf
    best_state = None
    for it in range(max_iter):
        R = compute_responsibilities(model.tree_, X_gate, model.n_leaves_)
        log_lik = _classification_log_lik(model.experts_, X_df, y_idx, model.n_leaves_)
        H, L = e_step(R, log_lik)
        history.append(L)
        if verbose:
            print(f"EM iter {it}: L={L:.4f}")
        if L > best_L:
            best_L = L
            best_state = (model.tree_, copy.deepcopy(model.experts_))
        if it > 0 and (L - history[-2]) < tol * max(abs(history[-2]), 1.0):
            break

        for leaf_id in list(model.experts_.keys()):
            model.experts_[leaf_id] = _refit_classification_expert(
                model.experts_[leaf_id], X_df, y_idx, model.classes_, H[:, leaf_id], rng, min_mass
            )

        model.tree_ = _rebuild_gate_tree(model.tree_, X_gate, H, 1e-6, min_mass)

    if best_state is not None:
        model.tree_, model.experts_ = best_state

    model.em_log_likelihood_ = history
    model.em_iterations_ = len(history)
    return model
