"""General type-2 (GT2) FIS kernel via alpha-plane decomposition.

This is the alpha-plane approach surveyed in ``docs/gt2-evaluation.md``
(issue #122's research spike): a GT2 secondary membership grade is decomposed
into ``K`` nested alpha-cuts, each of which is an ordinary IT2 set
(``GT2GaussianMembership.alpha_cut`` -- see ``gauss_data.py``), so **every**
forward-inference and type-reduction call in this module is
``it2_kernel.it2_firing_strengths``/``it2_kernel.karnik_mendel_tsk`` run
unchanged, once per alpha-plane. The only genuinely new computation is the
cross-plane combination: an alpha-weighted average (Mendel, Liu 2008) of the
``K`` per-plane results, weight = the alpha level itself, so a *more certain*
(higher-alpha, narrower) plane counts for more than a boundary
(lower-alpha, wider) one.

Two distinct problems, mirroring ``it2_kernel``'s own module docstring:

1. **Classification** (`gt2_firing_strengths`): each alpha-plane's own
   per-class crisp score (`it2_firing_strengths`'s per-rule midpoint
   reduction) is combined across planes into one alpha-weighted score per
   class.
2. **Regression** (`gt2_rule_firing` + `gt2_karnik_mendel_tsk`): each
   alpha-plane's own cross-rule Karnik-Mendel interval
   (`karnik_mendel_tsk`) is combined across planes into one alpha-weighted
   output interval.

``alpha=0`` (today's IT2 footprint) always carries zero weight in the
alpha-weighted average -- it is the "no confidence information used yet"
boundary case, not a plane that ever participates in the sum -- so
`default_alpha_levels` never includes it; the plane at ``alpha=0`` is used
below only as an independent bound for the containment property every
GT2-combined output must satisfy (see `gt2_rule_firing`'s docstring).
"""

import numpy as np
import pandas as pd

from .gauss_data import (
    GT2GaussianMixtureModel,
    IT2FeatureModel,
    IT2GaussianMixtureModel,
    IT2LabelModel,
    NormPair,
)
from .gauss_math import tsk_firing_strengths
from .it2_kernel import (
    it2_firing_strengths,
    karnik_mendel_tsk,
    _extract_upper_model,
    _extract_lower_model,
)


def default_alpha_levels(n_alpha_planes: int) -> np.ndarray:
    """``n_alpha_planes`` evenly spaced alpha levels in ``(0, 1]``.

    Excludes ``alpha=0``: it always carries zero weight in the alpha-weighted
    combination below, so including it would be a wasted forward pass (see
    the module docstring).
    """
    if n_alpha_planes < 1:
        raise ValueError(f"n_alpha_planes must be >= 1, got {n_alpha_planes}")
    return np.linspace(1.0 / n_alpha_planes, 1.0, n_alpha_planes)


def extract_alpha_plane_model(
    model: GT2GaussianMixtureModel, alpha: float
) -> IT2GaussianMixtureModel:
    """The IT2-shaped model at alpha-level ``alpha``, across every
    (feature, label) -- ``GT2GaussianMembership.alpha_cut`` applied slot by
    slot."""
    feature_models = {}
    for feature_name, gt2_feature_model in model.feature_models.items():
        label_models = {}
        for label, gt2_label_model in gt2_feature_model.label_models.items():
            label_models[label] = IT2LabelModel(
                memberships=[mf.alpha_cut(alpha) for mf in gt2_label_model.memberships]
            )
        feature_models[feature_name] = IT2FeatureModel(label_models)
    return IT2GaussianMixtureModel(feature_models)


def gt2_firing_strengths(
    X: pd.DataFrame,
    model: GT2GaussianMixtureModel,
    norms: NormPair,
    n_alpha_planes: int = 5,
    km_iterations: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Classification-shaped GT2 inference: the alpha-weighted combination of
    ``n_alpha_planes`` per-plane IT2 crisp firing strengths.

    Each plane's own reduction is exactly `it2_kernel.it2_firing_strengths`,
    called unchanged -- this function's only new work is the weighted sum.

    Args:
        X: Input feature matrix.
        model: GT2GaussianMixtureModel.
        norms: (t_norm, t_conorm) pair for fuzzy operations.
        n_alpha_planes: number of alpha-planes to combine (see
            `default_alpha_levels`); more planes trade cost for fidelity to
            the underlying triangular secondary membership grade
            (`docs/gt2-evaluation.md` measured this cost as linear in
            ``n_alpha_planes``, with no combination overhead).
        km_iterations: passed through to each plane's `it2_firing_strengths`
            call (per-rule reduction, not the cross-rule Karnik-Mendel search
            -- see that function's own docstring for why ``None``/midpoint
            and any truthy value agree exactly for this reduction).

    Returns:
        firing_crisp: (n_samples, n_labels) alpha-combined type-reduced output.
        labels: output labels, `it2_firing_strengths`'s own convention
            (``sorted(model.all_output_labels)``).
    """
    alphas = default_alpha_levels(n_alpha_planes)
    weighted_sum = None
    labels: list[int] = []
    for alpha in alphas:
        it2_model = extract_alpha_plane_model(model, float(alpha))
        _, _, firing_crisp, labels = it2_firing_strengths(
            X, it2_model, norms, km_iterations=km_iterations
        )
        contribution = alpha * firing_crisp
        weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
    return weighted_sum / np.sum(alphas), labels


def gt2_rule_firing(
    gt2_model: GT2GaussianMixtureModel,
    X: pd.DataFrame,
    top_n_todo: list,
    norms: NormPair,
    n_alpha_planes: int = 5,
    feature_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, list]:
    """Per-alpha-plane ``(firing_upper, firing_lower)`` pairs, for the
    regressor's Karnik-Mendel combination (`gt2_karnik_mendel_tsk`).

    Each plane's firing bounds come from `it2_kernel._extract_upper_model`/
    `_extract_lower_model` plus `gauss_math.tsk_firing_strengths`, exactly
    the machinery `it2_regressor._it2_rule_firing` already uses for a single
    (non-alpha-planed) IT2 model -- applied here once per plane.

    Returns:
        (firing_uppers, firing_lowers, alphas, labels): the first two are
        lists of length ``n_alpha_planes``, each ``(n_samples, n_rules)``;
        ``alphas`` is `default_alpha_levels`'s own array (so a caller can
        recompute the same weighting `gt2_karnik_mendel_tsk` uses); ``labels``
        is the rule ordering, shared across every plane.
    """
    alphas = default_alpha_levels(n_alpha_planes)
    firing_uppers: list[np.ndarray] = []
    firing_lowers: list[np.ndarray] = []
    labels: list = []
    for alpha in alphas:
        it2_model = extract_alpha_plane_model(gt2_model, float(alpha))
        upper_model = _extract_upper_model(it2_model)
        lower_model = _extract_lower_model(it2_model)
        fu, labels = tsk_firing_strengths(
            X[top_n_todo], upper_model, norms=norms, feature_arrays=feature_arrays
        )
        fl, _ = tsk_firing_strengths(
            X[top_n_todo], lower_model, norms=norms, feature_arrays=feature_arrays
        )
        firing_uppers.append(fu)
        firing_lowers.append(fl)
    return firing_uppers, firing_lowers, alphas, labels


def gt2_karnik_mendel_tsk(
    rule_values: np.ndarray,
    firing_uppers: list[np.ndarray],
    firing_lowers: list[np.ndarray],
    alphas: np.ndarray,
    max_iterations: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Alpha-weighted combination of ``len(alphas)`` per-plane Karnik-Mendel
    output intervals into one GT2 output interval ``(y_l, y_r)``.

    Each plane's own interval comes from `it2_kernel.karnik_mendel_tsk`,
    called unchanged on that plane's firing bounds; only the weighted average
    across planes is new. ``rule_values`` (each rule's own TSK consequent
    output) does not vary with alpha -- consequents are a function of ``X``
    alone, not of the antecedent footprint -- so it is passed once and reused
    across every plane, exactly as `it2_regressor` reuses one `rule_values`
    array across the upper/lower halves of a single (non-alpha-planed) IT2
    model.

    **Containment.** Every alpha-plane's firing bounds are a *subset*
    (`GT2GaussianMembership.alpha_cut`'s narrowing property) of the
    ``alpha=0`` plane's own (widest) bounds, and `karnik_mendel_tsk` is
    monotonic in its firing-bound arguments (narrower bounds cannot widen its
    output interval), so every individual plane's ``(y_l, y_r)`` lies inside
    the ``alpha=0`` plane's own -- and since the combined output here is a
    convex (weight-normalized) combination of those nested intervals, it
    inherits the same containment. Tested in
    ``tests/test_gt2_kernel.py``.

    Returns:
        (y_l, y_r): (n_samples,) arrays, the alpha-combined output interval.
    """
    total_alpha = float(np.sum(alphas))
    y_l_sum = None
    y_r_sum = None
    for alpha, fu, fl in zip(alphas, firing_uppers, firing_lowers):
        y_l, y_r = karnik_mendel_tsk(rule_values, fl, fu, max_iterations=max_iterations)
        y_l_sum = alpha * y_l if y_l_sum is None else y_l_sum + alpha * y_l
        y_r_sum = alpha * y_r if y_r_sum is None else y_r_sum + alpha * y_r
    return y_l_sum / total_alpha, y_r_sum / total_alpha
