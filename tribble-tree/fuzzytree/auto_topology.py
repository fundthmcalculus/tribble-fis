"""Derive a deconstruction topology from the data when none is supplied.

`deconstruct.py` is built on the premise that "structure comes from domain
knowledge, not from the data" (`DECONSTRUCTED_TREE_FINDINGS.md`), and on
N-CMAPSS -- where a real turbofan-station topology exists -- that premise pays:
R^2 0.593 against 0.405 flat and 0.370 for HME.

The gap this module fills is the case where the caller has no such knowledge.
Before it, `DeconstructedHierarchicalRegressor.fit` required a fully-specified
topology dict and had no fallback at all, so a dataset with no known physical
structure could not use the estimator without someone inventing a grouping by
hand -- which is what `grad-school`'s `table_tribbletree_tsk_order.py` had to do
for Concrete and Body Fat, with no automated alternative to compare against.
See tribble-fis#226.

**These are a fallback, not a replacement.** A derived topology is a guess about
structure made from correlations in one sample; a domain topology is a
statement about how the system actually works. The two are not the same kind of
object even when they produce the same dict, and nothing here should be read as
making the hand-authored path optional for a domain that has one.

Three strategies, matching #226's list
--------------------------------------

:func:`affinity_topology` (strategy 1)
    Cluster features by an affinity matrix -- agglomerative, average linkage,
    on ``1 - |corr|`` -- into ``n_groups`` groups, and wrap each group as a
    leaf under a common root. The intuition is that features which move
    together describe the same underlying thing, so they belong on the same
    leaf; the deconstruction then re-solves one consequent per group rather
    than one over everything.

:func:`select_topology` (strategy 2)
    Generate several candidates (varying ``n_groups``, plus the degenerate
    floor) and pick the one with the best held-out score. This is the strategy
    that makes the other two honest: an affinity grouping at some fixed ``k``
    is an assumption, and ``k`` is not knowable a priori.

:func:`per_feature_topology` (strategy 3)
    One leaf per feature. The zero-knowledge floor -- it asserts no grouping at
    all, so it is what a smarter strategy has to beat to have earned anything.

Everything here emits a plain ``dict[str, list[str]]`` and hands it to
``topology.parse_topology`` unchanged, so there is exactly one validation path
and a derived topology is subject to every rule a hand-written one is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Node names are generated, so they can collide with a feature column -- which
# `parse_topology` rejects outright, and rightly, since `{"G1": ["G1"]}` is
# unreadable. Rather than fail on a dataset that happens to have a column called
# "group_0", generated names get underscores appended until they are free.
_ROOT_STEM = "root"
_GROUP_STEM = "group"
_FEATURE_STEM = "feat"


def _unique_name(stem: str, taken: set[str]) -> str:
    name = stem
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def feature_affinity(X: pd.DataFrame) -> pd.DataFrame:
    """Absolute Pearson correlation between every pair of features.

    Absolute because a pair moving in opposite directions is as much "the same
    underlying quantity" as a pair moving together -- sign is a convention of
    how each column happens to be oriented, and grouping should not depend on
    it.

    A constant column correlates with nothing (``np.corrcoef`` yields NaN, and
    pandas propagates it). Those become affinity 0 to everything except
    themselves, which puts each constant feature on its own until the cut count
    forces it to join something. That is the right default: a column with no
    variance carries no evidence about what it belongs with, and guessing would
    be worse than isolating it.
    """
    corr = X.corr(numeric_only=True).abs().fillna(0.0)
    # `.values` on a freshly-reduced frame can be a read-only view, so the
    # diagonal is set through a copy rather than in place. A constant column's
    # self-correlation is NaN -> 0 above, and a feature must be perfectly
    # affine with itself or the distance matrix has a non-zero diagonal and the
    # clustering is nonsense.
    values = np.array(corr.to_numpy(dtype=float), copy=True)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)


def cluster_features(X: pd.DataFrame, n_groups: int, linkage: str = "average") -> list[list[str]]:
    """Group feature columns into ``n_groups`` clusters by affinity.

    Agglomerative rather than k-means, because the input is a precomputed
    pairwise distance (``1 - affinity``) with no coordinates behind it -- there
    is no space for a centroid to live in.

    Groups come back ordered by their first member's position in ``X.columns``,
    and members keep column order within a group. Determinism matters here
    beyond tidiness: the group names end up in ``node_state_`` and in any plot
    of the tree, and a topology that renames its nodes between runs on the same
    data is unreadable.
    """
    from sklearn.cluster import AgglomerativeClustering

    columns = list(X.columns)
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1, got {n_groups}")
    if n_groups > len(columns):
        raise ValueError(
            f"n_groups={n_groups} exceeds the {len(columns)} feature(s) available."
        )
    if n_groups == 1:
        return [columns]
    if n_groups == len(columns):
        return [[c] for c in columns]

    distance = 1.0 - feature_affinity(X).to_numpy()
    np.fill_diagonal(distance, 0.0)
    # Numerical asymmetry in `corr` would make the "precomputed" metric invalid.
    distance = np.clip((distance + distance.T) / 2.0, 0.0, None)

    labels = AgglomerativeClustering(
        n_clusters=n_groups, metric="precomputed", linkage=linkage
    ).fit_predict(distance)

    groups: dict[int, list[str]] = {}
    for column, label in zip(columns, labels):
        groups.setdefault(int(label), []).append(column)
    return [groups[k] for k in sorted(groups, key=lambda k: columns.index(groups[k][0]))]


def _wrap_groups(groups: list[list[str]], taken: set[str], stem: str) -> dict[str, list[str]]:
    """A two-level ``root -> group -> features`` dict from a list of groups."""
    taken = set(taken)
    topology: dict[str, list[str]] = {}
    group_names = []
    for i, members in enumerate(groups):
        name = _unique_name(f"{stem}_{i}", taken)
        topology[name] = list(members)
        group_names.append(name)
    root = _unique_name(_ROOT_STEM, taken)
    # Root first, purely so a printed dict reads top-down.
    return {root: group_names, **topology}


def affinity_topology(
    X: pd.DataFrame, n_groups: int = 3, linkage: str = "average"
) -> dict[str, list[str]]:
    """Strategy 1: cluster features by affinity, one leaf per cluster."""
    return _wrap_groups(
        cluster_features(X, n_groups, linkage), set(X.columns), _GROUP_STEM
    )


def per_feature_topology(X: pd.DataFrame) -> dict[str, list[str]]:
    """Strategy 3: one leaf per feature. The zero-knowledge floor.

    Asserts no grouping at all, which is what makes it useful: any strategy
    claiming to have found structure has to beat the topology that claims none.
    It is also the cheapest sanity check that the deconstruction machinery is
    working, since every leaf owns exactly one feature and the branch combiner
    is doing all of the work.
    """
    return _wrap_groups([[c] for c in X.columns], set(X.columns), _FEATURE_STEM)


def candidate_topologies(
    X: pd.DataFrame, n_groups: tuple[int, ...] = (2, 3, 4)
) -> dict[str, dict[str, list[str]]]:
    """The candidate set :func:`select_topology` chooses among.

    Named rather than a bare list, so the winner can be reported as
    ``"affinity_k3"`` instead of as an index into a list the caller cannot see.

    Cut counts above the feature count are dropped rather than raising: the
    caller asked for a sweep, and "3 groups over 2 features" is not a request
    they made, it is an artefact of the default. The floor is always included,
    both as strategy 3 and so the set is never empty on a one-feature frame.

    Candidates that describe the *same grouping* collapse to the first one
    named. They are not hypothetical: at ``k == n_features`` the affinity cut
    puts every feature in its own group, which is exactly ``per_feature`` under
    a different node prefix. Scoring both spends a full k-fold sweep to learn
    that a topology ties with itself, and then reports the winner under
    whichever name sorted first -- so "affinity_k4 won" would be describing the
    no-knowledge floor.
    """
    n_features = X.shape[1]
    candidates: dict[str, dict[str, list[str]]] = {}
    seen: dict[frozenset, str] = {}

    def add(name: str, topology: dict[str, list[str]]) -> None:
        # Node names differ between strategies, so identity is the partition of
        # feature columns itself -- a set of frozensets, order-insensitive on
        # both levels.
        root = next(iter(topology))
        key = frozenset(frozenset(topology[g]) for g in topology[root])
        if key in seen:
            return
        seen[key] = name
        candidates[name] = topology

    for k in n_groups:
        if 1 <= k <= n_features:
            add(f"affinity_k{k}", affinity_topology(X, k))
    add("per_feature", per_feature_topology(X))
    return candidates


def select_topology(
    X: pd.DataFrame,
    y: pd.Series,
    fit_score,
    n_groups: tuple[int, ...] = (2, 3, 4),
    n_splits: int = 3,
    random_state: int | None = 0,
) -> tuple[str, dict[str, list[str]], dict[str, float]]:
    """Strategy 2: score each candidate out-of-fold and return the best.

    Args:
        X, y: training data.
        fit_score: ``(X_tr, y_tr, X_va, y_va, topology) -> float``, higher is
            better. Injected rather than importing the estimator, because
            `deconstruct.py` imports this module and the reverse import would
            be a cycle -- and because it lets the selection be tested against a
            cheap stand-in instead of a full hierarchical fit.
        n_splits: K-fold splits. Cross-validated rather than a single holdout
            because the candidates differ by a single hyperparameter (`k`) and
            their scores land close together; one split's noise is comparable
            to the gap being measured.

    Returns ``(name, topology, all_scores)``. The scores come back too: which
    candidate won matters less than whether it won by anything, and a caller
    who cannot see the spread cannot tell "structure found" from "arbitrary
    pick among ties".
    """
    from sklearn.model_selection import KFold

    candidates = candidate_topologies(X, n_groups)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(X))

    scores: dict[str, float] = {}
    for name, topology in candidates.items():
        fold_scores = []
        for train_idx, val_idx in folds:
            fold_scores.append(
                fit_score(
                    X.iloc[train_idx].reset_index(drop=True),
                    y.iloc[train_idx].reset_index(drop=True),
                    X.iloc[val_idx].reset_index(drop=True),
                    y.iloc[val_idx].reset_index(drop=True),
                    topology,
                )
            )
        scores[name] = float(np.mean(fold_scores))

    # Ties break toward the earlier candidate, and `candidate_topologies` orders
    # them by ascending `k` with the floor last -- so a tie prefers the simpler
    # grouping, and prefers any real grouping over the no-knowledge floor only
    # when it actually scored better.
    best = max(scores, key=lambda name: scores[name])
    return best, candidates[best], scores
