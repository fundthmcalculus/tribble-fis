"""Head registry for Experiment B: fuzzy and non-fuzzy estimators on frozen embeddings.

Every head takes a ``(n_samples, n_dims)`` embedding matrix and a target. The
non-fuzzy heads (1-2) are the numbers to beat; the fuzzy heads (3-8) are the
subject of the experiment.

Alongside accuracy, each head reports **complexity**: rule count and mean
antecedents per rule. Those columns matter as much as accuracy -- a head that
wins by a point with 400 rules has lost the thing the experiment is about.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# fuzzytree lives outside the installed package (see tribble-tree/README.md).
_TREE_DIR = Path(__file__).resolve().parents[2] / "tribble-tree"
if str(_TREE_DIR) not in sys.path:
    sys.path.insert(0, str(_TREE_DIR))


@dataclass
class HeadResult:
    name: str
    task: str
    fit_seconds: float
    predict_seconds: float
    metrics: dict[str, float]
    n_rules: int | None = None
    mean_antecedents: float | None = None
    selected_features: list[str] = field(default_factory=list)
    error: str | None = None


def as_frame(X: np.ndarray) -> pd.DataFrame:
    """TRIBBLE's estimators key their rules off column names, so name the dims."""
    return pd.DataFrame(np.asarray(X), columns=[f"dim_{i}" for i in range(X.shape[1])])


# --------------------------------------------------------------------------
# Complexity introspection
# --------------------------------------------------------------------------

def describe_complexity(model, task: str) -> tuple[int | None, float | None, list[str]]:
    """Best-effort ``(n_rules, mean_antecedents, selected_features)``.

    Deliberately duck-typed: the estimator families in this repo expose their
    structure differently (implicit one-rule-per-class, explicit ``Rule`` lists,
    leaf counts), and this reporting layer should not need editing every time a
    new head is added.
    """
    feats = list(getattr(model, "top_features_", []) or [])

    # Fuzzy trees / HME: one rule per leaf, antecedents bounded by depth.
    n_leaves = getattr(model, "n_leaves_", None)
    if n_leaves is not None:
        depth = getattr(model, "max_depth", None)
        gate_feats = list(getattr(model, "gate_features_", []) or [])
        return int(n_leaves), float(depth) if depth else None, feats or gate_feats

    # Regressor: exposes its rule count directly.
    n_rules = getattr(model, "n_rules_", None)
    if n_rules is not None:
        return int(n_rules), float(len(feats)) if feats else None, feats

    # Ruspini: explicit Rule objects on the derived model.
    inner = getattr(model, "model_", None)
    rules = getattr(inner, "rules", None)
    if rules is not None:
        counts = [len(getattr(r, "antecedents", []) or []) for r in rules]
        mean = float(np.mean(counts)) if counts else None
        return len(rules), mean, feats

    # Flat MixtureOfGaussians classifier: rules are implicit, one per class,
    # each an AND over every selected feature.
    classes = getattr(model, "classes_", None)
    if classes is not None and feats:
        n_ante = getattr(model, "top_n_actual_", None) or len(feats)
        return int(len(classes)), float(n_ante), feats

    return None, None, feats


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def score(task: str, y_true, y_pred) -> dict[str, float]:
    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        }
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score
    y_pred = np.asarray(y_pred, dtype=float)
    rho = spearmanr(y_true, y_pred).statistic
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        # Spearman is the headline for a graded target: it measures whether the
        # ordering of sentiment intensity is right, which is what a fuzzy
        # membership over a bipolar axis is actually claiming to capture.
        "spearman": float(rho) if rho == rho else 0.0,
    }


# --------------------------------------------------------------------------
# Head builders
# --------------------------------------------------------------------------

def _baseline_heads(task: str) -> dict:
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    if task == "classification":
        return {
            "linear_probe": lambda: LogisticRegression(max_iter=2000, n_jobs=-1),
            "mlp": lambda: MLPClassifier(hidden_layer_sizes=(128,), max_iter=400,
                                         random_state=42),
        }
    return {
        "linear_probe": lambda: RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "mlp": lambda: MLPRegressor(hidden_layer_sizes=(128,), max_iter=400,
                                     random_state=42),
    }


def _fuzzy_heads(task: str, top_n: int) -> dict:
    from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
    from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
    from tribblefis.ruspini import RuspiniFuzzyClassifier
    from fuzzytree import (
        FuzzyClassificationTree,
        FuzzyRegressionTree,
        HierarchicalFuzzyExpertsClassifier,
        HierarchicalFuzzyExpertsRegressor,
    )

    if task == "classification":
        return {
            "tribble_flat": lambda: MixtureOfGaussiansFuzzyClassifier(top_n=top_n),
            "tribble_flat_refined": lambda: MixtureOfGaussiansFuzzyClassifier(
                top_n=top_n, refine=True, refine_method="coordinate"),
            "fuzzy_tree": lambda: FuzzyClassificationTree(
                criterion="ambiguity", top_n=top_n, max_depth=3, n_terms=2),
            "hme": lambda: HierarchicalFuzzyExpertsClassifier(
                criterion="ambiguity", top_n=top_n, max_depth=2, n_gate_terms=2,
                expert_kwargs={"top_n": 5}),
            "ruspini": lambda: RuspiniFuzzyClassifier(top_n=top_n),
        }
    return {
        "tribble_flat": lambda: MixtureOfGaussiansFuzzyRegressor(
            top_n=top_n, n_output_buckets=3, tsk_order="1st"),
        "tribble_flat_0th": lambda: MixtureOfGaussiansFuzzyRegressor(
            top_n=top_n, n_output_buckets=3, tsk_order="0th"),
        "fuzzy_tree": lambda: FuzzyRegressionTree(
            tsk_order="1st", top_n=top_n, max_depth=3, n_terms=2),
        "hme": lambda: HierarchicalFuzzyExpertsRegressor(
            criterion="variance", top_n=top_n, max_depth=2, n_gate_terms=2,
            expert_kwargs={"n_output_buckets": 3, "tsk_order": "1st"}),
    }


def build_heads(task: str, top_n: int = 20, include: list[str] | None = None) -> dict:
    """Return ``{name: factory}``.

    ``top_n`` caps how many embedding dimensions the fuzzy heads select. This is
    the single most important knob in the experiment: a flat FIS over 768 raw
    dimensions is not interpretable at any rule count, and TRIBBLE's
    differentiation-based ``take_top_features`` is what keeps the antecedent
    count human-readable. Sweep it.
    """
    heads = {**_baseline_heads(task), **_fuzzy_heads(task, top_n)}
    if include:
        missing = set(include) - set(heads)
        if missing:
            raise ValueError(f"unknown heads: {sorted(missing)}; have {sorted(heads)}")
        heads = {k: v for k, v in heads.items() if k in include}
    return heads


def run_head(name: str, factory, X_train, y_train, X_test, y_test,
             task: str) -> HeadResult:
    """Fit and evaluate one head. Failures are captured, not raised.

    A head that blows up on a particular width should not abort a sweep -- the
    failure is itself a reportable result.
    """
    Xtr, Xte = as_frame(X_train), as_frame(X_test)
    model = factory()
    try:
        t0 = time.perf_counter()
        model.fit(Xtr, y_train)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(Xte)
        pred_s = time.perf_counter() - t0

        n_rules, mean_ante, feats = describe_complexity(model, task)
        return HeadResult(name, task, fit_s, pred_s, score(task, y_test, y_pred),
                          n_rules, mean_ante, feats)
    except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
        return HeadResult(name, task, float("nan"), float("nan"), {},
                          error=f"{type(exc).__name__}: {exc}")
