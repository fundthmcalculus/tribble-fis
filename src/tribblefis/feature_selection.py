"""Pareto-optimal feature selection over a data-driven score and a user ranking.

Two objectives per feature, both maximised: the differentiation score from
`calculate_gaussian_correlation` and an optional user-supplied importance.
Without a user ranking this degenerates to the plain top-N by score.

Two views are provided:

* `pareto_front` -- non-dominated *features*, plus successive fronts (rank 0 is
  the Pareto front, rank 1 the front once rank 0 is removed, ...).
* `pareto_select_features` -- candidate *subsets* of size `top_n`. Subset
  objectives are the sums of per-feature objectives. Candidates are the
  supported Pareto-optimal subsets (found by sweeping the weight between the
  two objectives); each is non-dominated, and different weights can give
  genuinely different subsets. The caller picks one via `index`, or inspects
  all of them. Non-supported (concave-region) Pareto subsets are not
  enumerated -- documented limitation, exact enumeration is combinatorial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from optimizers.archive.metrics import non_dominated_mask

__all__ = [
    "ParetoSelection",
    "normalize_user_ranking",
    "pareto_front",
    "pareto_select_features",
    "rank_features",
    "find_cross_pairs",
    "add_cross_columns",
    "cross_name",
]


@dataclass(frozen=True)
class ParetoSelection:
    """One candidate subset: features, summed objectives, and the weight that found it."""

    features: tuple
    data_score: float
    user_score: float
    weight: float  # weight on the data objective; 1 - weight on the user objective


def normalize_user_ranking(user_ranking, features: Sequence[Any]) -> dict[Any, float]:
    """Coerce a user ranking to {feature: importance in [0, 1]} over `features`.

    Accepts a mapping feature->importance (higher = more important), a list of
    (feature, importance) pairs, or a plain ordered list of names (first = most
    important; scored by rank). Features the user omitted get 0. Unknown names
    raise, so a typo cannot silently vanish.
    """
    feats = list(features)
    if isinstance(user_ranking, Mapping):
        raw = dict(user_ranking)
    else:
        items = list(user_ranking)
        if items and all(isinstance(i, (tuple, list)) and len(i) == 2 for i in items):
            raw = {k: v for k, v in items}
        else:
            n = len(items)
            raw = {name: float(n - i) for i, name in enumerate(items)}
    unknown = [k for k in raw if k not in set(feats)]
    if unknown:
        raise ValueError(f"user_ranking names features not in the data: {unknown}")
    vals = np.array([float(raw.get(f, 0.0)) for f in feats])
    if not np.all(np.isfinite(vals)):
        raise ValueError("user_ranking contains non-finite importances")
    lo, hi = vals.min(), vals.max()
    scaled = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)
    # Omitted features must not rank above explicitly-scored ones just because
    # min-max stretched the scored range; pin omitted to the floor.
    omitted = np.array([f not in raw for f in feats])
    scaled[omitted] = 0.0
    return dict(zip(feats, scaled.tolist()))


def _objectives(data_scores, user_scores):
    names = [n for n, _ in data_scores]
    d = np.array([s for _, s in data_scores], dtype=float)
    u = np.array([user_scores[n] for n in names], dtype=float) if user_scores is not None else np.zeros(len(d))
    return names, d, u


def _nondominated_ranks(objs: np.ndarray) -> np.ndarray:
    """Front index per row (0 = Pareto front); maximisation on every column.

    Peels successive fronts using ``optimizers.archive.metrics.non_dominated_mask`` -- the
    dominance test this project already ships, which minimises, so the objectives are negated.
    """
    cost = -np.asarray(objs, dtype=float)
    ranks = np.full(len(cost), -1)
    remaining = np.arange(len(cost))
    rank = 0
    while len(remaining):
        front = non_dominated_mask(cost[remaining])
        ranks[remaining[front]] = rank
        remaining = remaining[~front]
        rank += 1
    return ranks


def pareto_front(data_scores, user_ranking=None) -> list[tuple[Any, float, float, int]]:
    """(feature, data_score, user_score, front_rank), sorted by front then data score."""
    feats = [n for n, _ in data_scores]
    user = normalize_user_ranking(user_ranking, feats) if user_ranking is not None else None
    names, d, u = _objectives(data_scores, user)
    ranks = _nondominated_ranks(np.column_stack([d, u]))
    order = sorted(range(len(names)), key=lambda i: (ranks[i], -d[i], -u[i]))
    return [(names[i], float(d[i]), float(u[i]), int(ranks[i])) for i in order]


def pareto_select_features(
    data_scores,
    user_ranking=None,
    top_n: int = 5,
    n_weights: int = 101,
) -> list[ParetoSelection]:
    """Distinct Pareto-optimal size-`top_n` subsets, ordered from data-led to user-led.

    Parameters
    ----------
    data_scores : list[tuple[feature, float]]
        Output of `calculate_gaussian_correlation`.
    user_ranking : optional
        See `normalize_user_ranking`. None -> a single subset, the plain top-N.
    top_n : int
        Subset size (clipped to the number of features).
    n_weights : int
        Resolution of the weight sweep.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    feats = [n for n, _ in data_scores]
    user = normalize_user_ranking(user_ranking, feats) if user_ranking is not None else None
    names, d, u = _objectives(data_scores, user)
    k = min(top_n, len(names))
    if user is None:
        idx = np.argsort(-d, kind="stable")[:k]
        return [ParetoSelection(tuple(names[i] for i in idx), float(d[idx].sum()), 0.0, 1.0)]

    found: dict[frozenset, ParetoSelection] = {}
    for w in np.linspace(1.0, 0.0, n_weights):
        # Stable tie-break on data then user score keeps the result deterministic.
        idx = np.lexsort((-u, -d, -(w * d + (1 - w) * u)))[:k]
        key = frozenset(idx.tolist())
        if key not in found:
            order = sorted(idx.tolist(), key=lambda i: (-d[i], -u[i]))
            found[key] = ParetoSelection(
                tuple(names[i] for i in order), float(d[idx].sum()), float(u[idx].sum()), float(w)
            )
    # Weighted-sum optima are non-dominated, but ties can yield a dominated one; filter.
    cands = list(found.values())
    return [
        c
        for c in cands
        if not any(
            o.data_score >= c.data_score
            and o.user_score >= c.user_score
            and (o.data_score > c.data_score or o.user_score > c.user_score)
            for o in cands
        )
    ]


def rank_features(
    X,
    y,
    method: str = "wasserstein",
    top_n: int = -1,
    correlation_threshold: float = 0.85,
    user_ranking=None,
    pareto_choice: int = 0,
):
    """Feature ranking for the estimators: plain scoring, or Pareto against a user ranking.

    Returns ``(differentiators, candidates)``. Without `user_ranking` this is
    exactly ``calculate_gaussian_correlation(...)`` and `candidates` is ``None``,
    so existing behaviour is unchanged. With one, every feature is scored, the
    Pareto-optimal size-`top_n` subsets are enumerated, and subset
    `pareto_choice` (0 = data-led ... last = user-led) becomes the result.
    The redundancy filter (`correlation_threshold`) is not applied on that path:
    the subset is chosen by the two objectives alone.
    """
    from .gauss_math import calculate_gaussian_correlation

    if user_ranking is None:
        return calculate_gaussian_correlation(
            X, y, method=method, top_n=top_n, correlation_threshold=correlation_threshold
        ), None
    if top_n <= 0:
        raise ValueError("user_ranking requires top_n > 0 (the size of the subset to choose)")
    scored = calculate_gaussian_correlation(X, y, method=method, top_n=-1)
    candidates = pareto_select_features(scored, user_ranking, top_n=top_n)
    if not -len(candidates) <= pareto_choice < len(candidates):
        raise ValueError(
            f"pareto_choice={pareto_choice} out of range: {len(candidates)} Pareto-optimal subset(s) available"
        )
    chosen = set(candidates[pareto_choice].features)
    return [(f, v) for f, v in scored if f in chosen], candidates


def cross_name(a, b) -> str:
    """Column name of the product feature built from ``a`` and ``b``."""
    return f"{a}*{b}"


def find_cross_pairs(
    X,
    y,
    max_cross: int = 3,
    n_permutations: int = 20,
    random_state: int = 0,
) -> list[tuple]:
    """Feature pairs whose joint class separation is real, for building product (crossover) columns.

    Pair lifts come from the vectorised engine of `calculate_interaction_scores`
    (``engine="gaussian"``), RAW and not normalised. A relative threshold cannot say whether the
    best pair is an interaction or merely the largest of many noise lifts, so the floor is a
    permutation null: shuffle ``y`` ``n_permutations`` times, take the largest lift seen in any
    shuffle, and keep only real pairs strictly above it (at most ``max_cross``, strongest first).
    With no interaction present a pair clears the floor with probability about
    ``1 / (n_permutations + 1)`` per dataset, so the default 20 gives ~5 % false alarms.
    """
    from .gauss_math import calculate_interaction_scores

    if max_cross <= 0 or X.shape[1] < 2:
        return []
    pool = [(c, 0.0) for c in X.columns]  # the gaussian engine computes its own marginals

    def raw(yy):
        return calculate_interaction_scores(X, yy, pool, engine="gaussian", normalize=False)

    real = raw(y)
    if not real:
        return []
    rng = np.random.default_rng(random_state)
    yv = y.to_numpy()
    floor = 0.0
    for _ in range(max(0, n_permutations)):
        null = raw(pd.Series(rng.permutation(yv), index=y.index))
        if null:
            floor = max(floor, null[0][2])
    return [(fi, fj) for fi, fj, lift in real[:max_cross] if lift > floor]


def add_cross_columns(X, pairs):
    """``X`` plus one product column per ``(a, b)`` pair (original columns untouched)."""
    if not pairs:
        return X
    return X.assign(**{cross_name(a, b): X[a] * X[b] for a, b in pairs})
