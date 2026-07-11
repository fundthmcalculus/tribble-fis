"""Split criteria: score a candidate variable's fuzzy partition at a node.

Every criterion works on *fuzzy path weights* ``w`` -- the accumulated membership
of each sample down the path so far -- and on the candidate variable's linguistic
``terms`` (from ``terms.build_split_terms``). Higher score = better split; the
builder takes the ``argmax``.

Criteria
--------
* ``variance_reduction`` (default, regression): fraction of firing-weighted output
  variance explained by the split. Directly aligned with the leaf objective
  (the consequent solver minimises firing-weighted squared error).
* ``ambiguity`` / fuzzy information gain (Yuan & Shaw 1995; Janikow 1998): treats
  a discrete target (true classes, or output quantile buckets for regression) as
  fuzzy classes and maximises fuzzy information gain. Native to the classifier.
* ``differentiation``: a weight-aware adaptation of
  ``tribblefis.gauss_math.calculate_gaussian_correlation`` -- a cheap *relevance*
  score, best used to prefilter wide input spaces, not as the sole per-node
  splitter.

References:
    Y. Yuan, M. J. Shaw. "Induction of fuzzy decision trees." Fuzzy Sets and
        Systems 69(2):125-139, 1995.
    C. Z. Janikow. "Fuzzy decision trees: issues and methods." IEEE Trans. SMC-B
        28(1):1-14, 1998.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

from tribblefis.gauss_data import AnyMembership

_EPS = 1e-9


def _membership_matrix(
    x_col: np.ndarray, terms: list[tuple[str, AnyMembership]]
) -> np.ndarray:
    """(n_samples, n_terms) membership of each sample in each term."""
    return np.column_stack([mf.evaluate(x_col) for _, mf in terms])


def _weighted_mean(y: np.ndarray, w: np.ndarray) -> float:
    m = w.sum()
    if m <= _EPS:
        return 0.0
    return float((w * y).sum() / m)


def _weighted_var(y: np.ndarray, w: np.ndarray) -> float:
    m = w.sum()
    if m <= _EPS:
        return 0.0
    mu = (w * y).sum() / m
    return float((w * (y - mu) ** 2).sum() / m)


# --------------------------------------------------------------------------
# 1. Weighted variance reduction (default for regression)
# --------------------------------------------------------------------------
def variance_reduction(
    x_col: np.ndarray,
    y_value: np.ndarray,
    y_bucket: np.ndarray,
    w: np.ndarray,
    terms: list[tuple[str, AnyMembership]],
) -> float:
    """Fraction of firing-weighted variance explained by splitting on this var.

    ``dVar = var_w(y) - sum_c (M_c / sum M) * var_w(y, w * mu_c)``, normalised by
    ``var_w(y)`` so scores are comparable across nodes and give a clean stop rule.
    """
    parent_var = _weighted_var(y_value, w)
    if parent_var <= _EPS:
        return 0.0

    M = _membership_matrix(x_col, terms)  # (n, k)
    child_w = w[:, None] * M  # (n, k)
    child_mass = child_w.sum(axis=0)  # (k,)
    total = child_mass.sum()
    if total <= _EPS:
        return 0.0

    weighted_child = 0.0
    for c in range(M.shape[1]):
        if child_mass[c] <= _EPS:
            continue
        weighted_child += (child_mass[c] / total) * _weighted_var(y_value, child_w[:, c])

    return float((parent_var - weighted_child) / parent_var)


# --------------------------------------------------------------------------
# 2. Fuzzy information gain / ambiguity (Yuan-Shaw, Janikow)
# --------------------------------------------------------------------------
def fuzzy_information_gain(
    x_col: np.ndarray,
    y_value: np.ndarray,
    y_bucket: np.ndarray,
    w: np.ndarray,
    terms: list[tuple[str, AnyMembership]],
) -> float:
    """Fuzzy information gain of a split (Janikow fuzzy ID3 form).

    ``y_bucket`` holds the discrete (pseudo-)class of each sample. For regression
    this is the output quantile bucket; for classification it is the true label
    (encoded to ints). Gain = node fuzzy entropy - weighted child fuzzy entropy.
    """
    classes = np.unique(y_bucket)
    if len(classes) < 2:
        return 0.0

    # Node entropy over the soft class distribution.
    node_mass = np.array([w[y_bucket == c].sum() for c in classes], dtype=float)
    node_total = node_mass.sum()
    if node_total <= _EPS:
        return 0.0
    pn = node_mass[node_mass > _EPS] / node_total
    h_node = float(-(pn * np.log2(pn)).sum())

    M = _membership_matrix(x_col, terms)
    child_w = w[:, None] * M  # (n, k)
    term_mass = child_w.sum(axis=0)
    total = term_mass.sum()
    if total <= _EPS:
        return 0.0

    expected_h = 0.0
    for j in range(M.shape[1]):
        if term_mass[j] <= _EPS:
            continue
        d_jl = np.array([child_w[y_bucket == c, j].sum() for c in classes], dtype=float)
        p = d_jl / d_jl.sum()
        p = p[p > _EPS]
        h_j = float(-(p * np.log2(p)).sum())
        expected_h += (term_mass[j] / total) * h_j

    return h_node - expected_h


def _nonspecificity(p: np.ndarray) -> float:
    """Higashi-Klir U-uncertainty of the possibility distribution ``pi=p/max p``:
    ``g(pi) = sum (pi*_l - pi*_{l+1}) ln l`` on descending-sorted possibilities."""
    if p.sum() <= _EPS:
        return 0.0
    pi = p / p.max() if p.max() > _EPS else p
    pi_sorted = np.sort(pi)[::-1]
    pi_ext = np.append(pi_sorted, 0.0)
    levels = np.arange(1, len(pi_sorted) + 1)
    return float(((pi_ext[:-1] - pi_ext[1:]) * np.log(levels)).sum())


def classification_ambiguity(
    x_col: np.ndarray,
    y_value: np.ndarray,
    y_bucket: np.ndarray,
    w: np.ndarray,
    terms: list[tuple[str, AnyMembership]],
) -> float:
    """Yuan-Shaw classification *ambiguity reduction* (higher = better).

    Ambiguity of a class distribution is the nonspecificity (U-uncertainty) of its
    possibility distribution. We return ``g(node) - sum_j (mass_j/total)*g(A_j)``
    -- the reduction in ambiguity from the split -- so the builder's argmax picks
    the most ambiguity-reducing variable and the ``min_gain`` stop rule applies
    with its usual non-negative-gain semantics.
    """
    classes = np.unique(y_bucket)
    if len(classes) < 2:
        return 0.0

    node_mass = np.array([w[y_bucket == c].sum() for c in classes], dtype=float)
    if node_mass.sum() <= _EPS:
        return 0.0
    g_node = _nonspecificity(node_mass)

    M = _membership_matrix(x_col, terms)
    child_w = w[:, None] * M
    term_mass = child_w.sum(axis=0)
    total = term_mass.sum()
    if total <= _EPS:
        return 0.0

    expected_g = 0.0
    for j in range(M.shape[1]):
        if term_mass[j] <= _EPS:
            continue
        d_jl = np.array([child_w[y_bucket == c, j].sum() for c in classes], dtype=float)
        expected_g += (term_mass[j] / total) * _nonspecificity(d_jl)

    return g_node - expected_g


# --------------------------------------------------------------------------
# 3. Weight-aware differentiation score (relevance prefilter)
# --------------------------------------------------------------------------
def _weighted_moments(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    m = w.sum()
    if m <= _EPS:
        return 0.0, 1e-6
    mu = (w * x).sum() / m
    sigma = np.sqrt(max((w * (x - mu) ** 2).sum() / m, 0.0))
    return float(mu), float(max(sigma, 1e-6))


def differentiation(
    x_col: np.ndarray,
    y_value: np.ndarray,
    y_bucket: np.ndarray,
    w: np.ndarray,
    terms: list[tuple[str, AnyMembership]],
) -> float:
    """Weight-aware distributional separability of the variable across classes.

    Mirrors the metric block of ``calculate_gaussian_correlation`` (Bhattacharyya,
    Jensen-Shannon, overlap) but with weighted moments so it reflects the soft
    subset at this node. Ignores the candidate ``terms`` -- it scores variable
    *relevance*, not partition quality, so it is a prefilter, not a sole splitter.
    """
    classes = np.unique(y_bucket)
    if len(classes) < 2:
        return 0.0

    lo, hi = float(np.min(x_col)), float(np.max(x_col))
    if hi - lo <= _EPS:
        return 0.0
    xr = np.linspace(lo, hi, 100)

    score = 0.0
    for a in range(len(classes)):
        for b in range(a + 1, len(classes)):
            wa = w * (y_bucket == classes[a])
            wb = w * (y_bucket == classes[b])
            if wa.sum() <= _EPS or wb.sum() <= _EPS:
                continue
            mua, sa = _weighted_moments(x_col, wa)
            mub, sb = _weighted_moments(x_col, wb)
            pa = stats.norm.pdf(xr, mua, sa)
            pb = stats.norm.pdf(xr, mub, sb)
            sa_sum, sb_sum = pa.sum(), pb.sum()
            if sa_sum <= _EPS or sb_sum <= _EPS:
                continue
            pa /= sa_sum
            pb /= sb_sum
            bhatta = 1.0 - float(np.sum(np.sqrt(pa * pb)))
            js = float(jensenshannon(pa, pb))
            overlap = 1.0 - float(np.sum(np.minimum(pa, pb)))
            vals = np.array([bhatta, js, overlap])
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            score += (vals.mean() + np.prod(vals) ** (1.0 / len(vals))) / 2.0
    return float(score)


CRITERIA = {
    "variance": variance_reduction,
    "ambiguity": classification_ambiguity,
    "info_gain": fuzzy_information_gain,
    "differentiation": differentiation,
}


def get_criterion(name: str):
    if name not in CRITERIA:
        raise ValueError(
            f"Unknown criterion {name!r}; expected one of {sorted(CRITERIA)}"
        )
    return CRITERIA[name]
