"""Grow a model one ranked feature at a time until a held-out target is met (tribble-fis #242).

Choosing ``top_n`` by hand means guessing, or sweeping: refit at ``top_n = 1, 2, 3, ...`` and read off
where quality is good enough. A sweep re-scores every feature and refits every membership at every size.
Two properties of the model make that unnecessary:

* a feature's memberships are fitted per ``(feature, label)`` and do not depend on which other
  features are present (a feature fitted alone is bit-identical to the same feature in a joint fit), and
* the firing strength folds over features -- per label, ``t_norm(running, conorm-fold of this
  feature's memberships)`` starting from 1.0 -- so adding a feature is ONE more fold step.

So `grow_features` scores/ranks once (the caller does), fits each feature's memberships once, keeps one
running firing-strength matrix on the held-out rows, and adds a feature per step. Everything here
is *evaluation*; deciding which rows are held out and refitting the final model is the estimator's job
(`TribbleClassifier.fit_growing`).
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from . import gauss_math as _gm
from .gauss_math import t_conorm, t_norm

__all__ = [
    "RunningFiringStrengths",
    "metric_function",
    "regression_metric_function",
    "growth_order",
    "grow_features",
]


class RunningFiringStrengths:
    """Per-label firing strengths on a fixed row set, extended one feature at a time.

    This is the inner loop of `gauss_math.tsk_firing_strengths` unrolled over features: the same
    start (ones), the same conorm fold per cell, the same t-norm across features, in the same
    order -- so after adding features ``f1..fn`` the strengths equal
    ``tsk_firing_strengths(X, model_with_f1..fn)`` (NumPy path), at the cost of one step instead
    of a full pass.
    """

    def __init__(self, X: pd.DataFrame, labels: list, norms):
        self._X = X
        self.labels = list(labels)
        self._norms = norms
        self.strengths = np.ones((len(X), len(self.labels)))
        self.n_features = 0

    def add(self, name: Any, feature_model) -> None:
        data = np.asarray(self._X[name].values)
        for j, label in enumerate(self.labels):
            label_model = feature_model.label_models.get(label)
            if label_model is None:  # same skip as tsk_firing_strengths
                continue
            membership = np.zeros(len(data))
            for mf in label_model.memberships:
                membership = t_conorm(membership, mf.evaluate(data), self._norms.t_conorm)
            self.strengths[:, j] = t_norm(self.strengths[:, j], membership, self._norms.t_norm)
        self.n_features += 1

    def predict(self) -> np.ndarray:
        winners = np.argmax(self.strengths, axis=1)
        return np.array([self.labels[i] for i in winners])


def metric_function(metric: str | Callable, pos_label=None) -> Callable:
    """``metric(y_true, y_pred) -> float`` from a name or a callable.

    Names: ``f1_macro`` (default elsewhere), ``f1_weighted``, ``f1_micro``, ``accuracy``,
    ``balanced_accuracy``, and ``f1`` (binary, needs ``pos_label`` unless the labels are {0, 1}).
    """
    if callable(metric):
        return metric
    table = {
        "f1_macro": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        "f1_weighted": lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0),
        "f1_micro": lambda yt, yp: f1_score(yt, yp, average="micro", zero_division=0),
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "f1": lambda yt, yp: f1_score(yt, yp, pos_label=1 if pos_label is None else pos_label, zero_division=0),
    }
    if metric not in table:
        raise ValueError(f"metric must be a callable or one of {sorted(table)}, got {metric!r}")
    return table[metric]


def regression_metric_function(metric: str | Callable, higher_is_better: bool | None = None) -> tuple[Callable, bool]:
    """``(metric(y_true, y_pred) -> float, higher_is_better)`` for a regressor.

    Names: ``r2`` (higher is better), ``rmse`` and ``mae`` (lower is better). A callable needs
    ``higher_is_better`` stated, because the target is a floor for some metrics and a ceiling for others.
    """
    if callable(metric):
        if higher_is_better is None:
            raise ValueError("a callable metric needs higher_is_better=True/False (is the target a floor or a ceiling?)")
        return metric, bool(higher_is_better)
    table = {
        "r2": (r2_score, True),
        "rmse": (lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))), False),
        "mae": (mean_absolute_error, False),
    }
    if metric not in table:
        raise ValueError(f"metric must be a callable or one of {sorted(table)}, got {metric!r}")
    fn, direction = table[metric]
    if higher_is_better is not None and bool(higher_is_better) != direction:
        raise ValueError(f"metric {metric!r} is {'higher' if direction else 'lower'}-is-better; got higher_is_better={higher_is_better}")
    return fn, direction


def growth_order(
    X_train: pd.DataFrame,
    y_rank,
    *,
    method: str,
    correlation_threshold: float,
    user_ranking=None,
    user_weight: float = 0.5,
) -> tuple[list, list, list]:
    """The ONE ranking growth walks: ``(order, skipped, scored)``.

    ``scored`` is the data ranking (``calculate_gaussian_correlation`` on the TRAINING rows, looked
    up at call time). With a ``user_ranking`` the order is ``(1 - w) * data + w * user`` -- for a fixed
    weight a single order whose top-k is a prefix, which growth needs. The redundancy filter
    then skips a candidate correlated with one already kept (as ``top_n`` does), and what it skipped is returned.
    """
    from .feature_selection import normalize_user_ranking

    scored = list(_gm.calculate_gaussian_correlation(X_train, y_rank, method=method, top_n=-1))
    ranked = scored
    if user_ranking is not None:
        user = normalize_user_ranking(user_ranking, [f for f, _ in ranked])
        ranked = sorted(
            ranked,
            key=lambda fs: (-((1 - user_weight) * fs[1] + user_weight * user[fs[0]]), -fs[1], -user[fs[0]]),
        )
    skipped: list = []
    if 0.0 < correlation_threshold < 1.0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kept = _gm._select_uncorrelated_top_n(X_train, ranked, len(ranked), correlation_threshold)
        order = [f for f, _ in kept]
        kept_set = set(order)
        skipped = [f for f, _ in ranked if f not in kept_set]
    else:
        order = [f for f, _ in ranked]
    return order, skipped, scored


def grow_features(
    order: list,
    *,
    target: float,
    evaluate: Callable[[int, Any], float],
    max_n: int | None = None,
    patience: int | None = None,
    higher_is_better: bool = True,
) -> dict:
    """Add ``order``'s features one at a time; stop at the first size whose held-out score meets ``target``.

    ``evaluate(n, name)`` does the whole step -- fit the new feature on the TRAINING rows, extend the
    model by it, and return the held-out score -- so the classifier (a running firing-strength fold)
    and the regressor (re-solving its consequents) share this loop. With ``higher_is_better`` the
    target is a floor (``score >= target``); otherwise a ceiling (``score <= target``), and "best"
    means lowest.

    Returns ``history`` (n, feature, score, best_so_far, step_seconds, cumulative_seconds),
    ``n_selected`` and ``target_reached``. Stopping: the smallest size meeting the target wins. If it is
    never met the best size is returned (earliest on ties) with ``target_reached=False`` and a
    warning. ``patience`` (optional) ends growth early, without reaching the target, after that many
    sizes in a row with no new best.
    """
    limit = len(order) if max_n is None else min(int(max_n), len(order))
    if limit < 1:
        raise ValueError("there is nothing to grow: no candidate features (or max_n < 1)")
    sign = 1.0 if higher_is_better else -1.0

    rows: list[dict] = []
    best, best_n, stale = -np.inf, 0, 0  # `best` is in sign-adjusted units: larger is always better
    reached_n: int | None = None
    started = time.perf_counter()

    for n, name in enumerate(order[:limit], start=1):
        step = time.perf_counter()
        score = float(evaluate(n, name))
        adjusted = sign * score
        if adjusted > best:
            best, best_n, stale = adjusted, n, 0
        else:
            stale += 1
        now = time.perf_counter()
        rows.append(
            dict(
                n=n,
                feature=name,
                score=score,
                best_so_far=sign * best,
                step_seconds=now - step,
                cumulative_seconds=now - started,
            )
        )
        if adjusted >= sign * target:
            reached_n = n
            break
        if patience is not None and stale >= patience:
            break

    reached = reached_n is not None
    if not reached:
        warnings.warn(
            f"target {target:g} not reached on the held-out rows (best {sign * best:.4f} at n={best_n}, "
            f"tried {len(rows)} of {len(order)} ranked features); returning the best size seen.",
            UserWarning,
            stacklevel=3,
        )
    return dict(
        history=pd.DataFrame(rows),
        n_selected=reached_n if reached else best_n,
        target_reached=reached,
    )
