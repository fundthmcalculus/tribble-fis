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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .gauss_math import create_gaussian_membership_dict, t_conorm, t_norm

__all__ = ["RunningFiringStrengths", "metric_function", "grow_features"]


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


def grow_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val,
    order: list,
    *,
    target: float,
    metric: Callable,
    norms,
    max_n: int | None = None,
    patience: int | None = None,
    n_gaussians=0,
    max_samples=None,
    random_state: int = 42,
) -> dict:
    """Add ``order``'s features one at a time; stop at the first size whose held-out score >= ``target``.

    Memberships are fitted on ``X_train`` only and scored on ``X_val``. Returns a dict:
    ``history`` (one row per size: n, feature, score, best_so_far, step_seconds,
    cumulative_seconds), ``n_selected``, ``target_reached``, ``feature_models`` (the fitted
    memberships of every feature tried) and ``labels``.

    Stopping: the smallest size meeting ``target`` wins and growth stops there. If the target is
    never met the best-scoring size is returned (earliest on ties) with ``target_reached=False``
    and a warning. ``patience`` (optional) ends growth early, without reaching the target, after
    that many sizes in a row with no new best score.
    """
    limit = len(order) if max_n is None else min(int(max_n), len(order))
    if limit < 1:
        raise ValueError("there is nothing to grow: no candidate features (or max_n < 1)")
    y_val = np.asarray(y_val)

    feature_models: dict = {}
    rows: list[dict] = []
    running: RunningFiringStrengths | None = None
    best, best_n, stale = -np.inf, 0, 0
    reached_n: int | None = None
    started = time.perf_counter()

    for n, name in enumerate(order[:limit], start=1):
        step = time.perf_counter()
        # One feature's memberships, fitted once. (A feature fitted alone equals the same
        # feature in a joint fit, so growing does not change what a fresh fit would build.)
        fitted = create_gaussian_membership_dict(
            X_train,
            y_train,
            top_n_var_names=[name],
            n_gaussians=n_gaussians,
            max_samples=max_samples,
            random_state=random_state,
        )
        feature_models[name] = fitted.feature_models[name]
        if running is None:
            running = RunningFiringStrengths(X_val, feature_models[name].ordered_keys, norms)
        running.add(name, feature_models[name])
        score = float(metric(y_val, running.predict()))

        if score > best:
            best, best_n, stale = score, n, 0
        else:
            stale += 1
        now = time.perf_counter()
        rows.append(
            dict(
                n=n,
                feature=name,
                score=score,
                best_so_far=best,
                step_seconds=now - step,
                cumulative_seconds=now - started,
            )
        )
        if score >= target:
            reached_n = n
            break
        if patience is not None and stale >= patience:
            break

    reached = reached_n is not None
    if not reached:
        warnings.warn(
            f"target {target:g} not reached on the held-out rows (best {best:.4f} at n={best_n}, "
            f"tried {len(rows)} of {len(order)} ranked features); returning the best size seen.",
            UserWarning,
            stacklevel=3,
        )
    return dict(
        history=pd.DataFrame(rows),
        n_selected=reached_n if reached else best_n,
        target_reached=reached,
        feature_models=feature_models,
        labels=list(running.labels) if running is not None else [],
    )
