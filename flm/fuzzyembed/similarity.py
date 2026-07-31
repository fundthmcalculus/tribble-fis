"""Similarity for membership vectors.

**Do not use cosine.** A membership vector is not a point in a linear space --
coordinates are degrees of truth, not signed magnitudes along orthogonal axes.
Cosine on them is a category error that silently costs accuracy: it rewards
matching *ratios* between dimensions when what matters is matching *degrees*, and
it treats a shared zero (both spans lack the concept) as informative.

The fuzzy Jaccard index is the right instrument -- the set-theoretic ratio
generalised to graded membership::

    sim(A, B) = sum_c min(mu_c^A, mu_c^B) / sum_c max(mu_c^A, mu_c^B)

``hierarchy_jaccard`` extends it with partial credit when mass lands on a *sibling*
rather than the same node, which is what makes the tree structure pay off: "dog" and
"wolf" share no leaf but share ``canine``, and a flat metric scores that 0.
"""

from __future__ import annotations

import numpy as np

from .hierarchy import FuzzyHierarchy


def fuzzy_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Graded Jaccard. The honest headline metric."""
    num = np.minimum(a, b).sum()
    den = np.maximum(a, b).sum()
    return float(num / den) if den > 0 else 0.0


def fuzzy_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Graded Dice -- less punitive than Jaccard on sparse vectors."""
    den = a.sum() + b.sum()
    return float(2 * np.minimum(a, b).sum() / den) if den > 0 else 0.0


#: The level whose Jaccard actually discriminates, measured rather than assumed.
#: On the children's-story corpus, per-level Jaccard for synonym vs unrelated pairs
#: came out::
#:
#:     pair                                      L1     L2     L3     L4     L5
#:     happy child / joyful boy      (similar)  .333   .333   .000   .000   .000
#:     dog barked  / wolf howled     (similar)  .581   .235   .016   .004   .000
#:     dog barked  / king spoke      (differ)   .453   .062   .033   .018   .001
#:     girl ate bread / boy ate food (similar)  .974   .407   .300   .270   .267
#:     girl ate bread / mountain tall(differ)   .421   .000   .000   .000   .000
#:
#: L1 (4 POS dimensions) is too coarse to separate anything; L3+ is so sparse that
#: near-synonyms share no coordinate at all. L2 -- the ~45 supersenses -- is the only
#: level that orders the pairs correctly, confirming the plan's prediction that the
#: supersense width is the design centre.
DISCRIMINATIVE_LEVEL = 2


def hierarchy_jaccard(a: np.ndarray, b: np.ndarray, h: FuzzyHierarchy,
                      level: int, peak_level: int = DISCRIMINATIVE_LEVEL,
                      decay: float = 0.5) -> float:
    """Jaccard blended across resolutions, giving partial credit for near misses.

    Weights peak at ``peak_level`` and decay geometrically in both directions. An
    earlier version decayed from the *finest* level, which put full weight on the
    sparsest and least informative resolution: near-synonyms share no leaf synset, so
    that term is 0 and it dominated the average, collapsing the ordering the metric
    exists to produce.

    Rolling up is what buys the partial credit: "dog" and "wolf" share no leaf but do
    share ``canine``, and a flat metric scores that exactly 0.
    """
    score = 0.0
    weight = 0.0
    for lvl in range(1, level + 1):
        w = decay ** abs(lvl - peak_level)
        av = a if lvl == level else h.rollup(a, level, lvl)
        bv = b if lvl == level else h.rollup(b, level, lvl)
        score += w * fuzzy_jaccard(av, bv)
        weight += w
    return score / weight if weight else 0.0


def pairwise(vectors: np.ndarray, metric=fuzzy_jaccard) -> np.ndarray:
    """Full similarity matrix. O(n^2) -- for evaluation sets, not for retrieval."""
    n = vectors.shape[0]
    out = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = metric(vectors[i], vectors[j])
    return out


def nearest(query: np.ndarray, corpus: np.ndarray, k: int = 5,
            metric=fuzzy_jaccard) -> list[tuple[int, float]]:
    scores = [(i, metric(query, corpus[i])) for i in range(corpus.shape[0])]
    scores.sort(key=lambda kv: -kv[1])
    return scores[:k]
