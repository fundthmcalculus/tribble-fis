"""Error-driven adaptive rule partitioning for the TSK consequent pipeline.

`partition_output` (regression.py) fixes the number of rules up front and
splits `y` into equal-frequency buckets via `qcut` -- every region of the
target gets the same resolution regardless of how well the model actually
fits there. `grow_adaptive_partition` replaces that fixed partition with an
iterative one: start from a single rule covering all of `y`, fit, measure
each rule's local R^2 against its own y-mean, and split the worst-fitting
rule into two (`split_method="median"` or `"sse"`). Repeat until every rule
clears an R^2 threshold or a rule budget is hit, skipping any rule whose last
split didn't actually reduce error there (`guard_stalled_splits`) so the loop
can't fixate on a single unproductive region.

Because rules are just contiguous y-ranges (`partition_output_by_edges`),
splitting one only ever adds a new label -- it does not require touching the
membership functions of unrelated rules, though this implementation still
refits every rule's memberships each iteration for simplicity/correctness.
"""

import typing
from typing import NamedTuple

import numpy as np
import pandas as pd

from .gauss_data import GaussianMixtureModel, NormPair
from .gauss_math import create_gaussian_membership_dict
from .regression import (
    partition_output_by_edges,
    solve_tsk_consequents,
    predict_tsk,
    _bucket_r2,
    _rsquared,
)


class AdaptivePartitionResult(NamedTuple):
    model: GaussianMixtureModel
    y_part: pd.DataFrame
    y_bucket_mean: np.ndarray
    corr_terms: np.ndarray
    edges: list[float]
    history: list[dict]


def _intervals_from_edges(edges: list[float]) -> list[tuple[float, float]]:
    """Bucket (lo, hi) y-ranges implied by `edges`, in ascending bucket-id order."""
    bounds = [-np.inf, *sorted(edges), np.inf]
    return list(zip(bounds[:-1], bounds[1:]))


def _best_sse_split(y_bucket: np.ndarray, min_leaf: int = 5) -> float | None:
    """CART-style 1D split: the threshold minimizing combined child SSE.

    Scans every candidate split of the bucket's sorted y-values into a left
    and right run, each scored against its own mean (`SSE = sum(y^2) -
    sum(y)^2/n`, the usual cumulative-sum trick so the whole scan is O(n) after
    an O(n log n) sort). Returns the midpoint between the two values straddling
    the best split, or None if the bucket is too small to leave `min_leaf`
    points on each side -- the caller treats that as "can't be split further."
    """
    y_sorted = np.sort(y_bucket)
    n = len(y_sorted)
    if n < 2 * min_leaf:
        return None

    cs1 = np.cumsum(y_sorted)
    cs2 = np.cumsum(y_sorted.astype(np.float64) ** 2)
    total_sum, total_sumsq = cs1[-1], cs2[-1]

    i = np.arange(min_leaf, n - min_leaf + 1)  # split after index i-1 (1-indexed count on the left)
    s1_l, s2_l, n_l = cs1[i - 1], cs2[i - 1], i
    s1_r, s2_r, n_r = total_sum - s1_l, total_sumsq - s2_l, n - i
    sse = (s2_l - s1_l ** 2 / n_l) + (s2_r - s1_r ** 2 / n_r)

    best_i = int(i[np.argmin(sse)])
    return float((y_sorted[best_i - 1] + y_sorted[best_i]) / 2.0)


def grow_adaptive_partition(
    X_train: pd.DataFrame,
    y_raw_train: pd.Series,
    top_n_todo: list[typing.Any],
    n_gaussians: int | dict = 0,
    tsk_order: str = "1st",
    l2_reg: float = 0.0,
    basis: str = "raw",
    cross_pairs: list[tuple[int, int]] | None = None,
    pin_extremes: bool = True,
    norms: NormPair | None = None,
    max_rules: int = 8,
    r2_threshold: float = 0.9,
    min_bucket_samples: int = 20,
    guard_stalled_splits: bool = True,
    split_method: typing.Literal["median", "sse"] = "median",
    verbose: bool = False,
) -> AdaptivePartitionResult:
    """Grow a TSK rule partition by repeatedly splitting the worst-fitting rule.

    Starts from a single rule (all of `y_raw_train`) and, each iteration,
    fits the full model (memberships + closed-form consequents), scores every
    rule's local R^2, and bisects the lowest-scoring rule (see `split_method`
    below) -- provided that rule still has at least `min_bucket_samples` rows,
    its R^2 is below `r2_threshold`, and (see `guard_stalled_splits` below) it
    isn't a region that already proved unproductive to split. Stops when no
    rule is eligible to split or the rule count reaches `max_rules`.

    `split_method`: 'median' (default) bisects at the bucket's median y,
    giving two equal-frequency children regardless of how the values are
    distributed. 'sse' instead scans every possible split of the bucket's
    y-values and picks the one minimizing the two children's combined SSE
    against their own means (the standard CART 1D split) -- a data-driven
    choice of *where* to cut, rather than always cutting at the midpoint by
    count. It costs an O(n log n) sort per candidate bucket per iteration
    instead of an O(n) median, negligible next to the membership refit.

    `guard_stalled_splits` (default True): splitting a bucket can, itself,
    make that region's fit worse rather than better -- a narrower y-range
    gives its per-feature Gaussian antecedents fewer points to fit, and
    firing-strength routing can degrade enough that the split doesn't help
    (or actively hurts) even though a child is still below `r2_threshold`.
    Without this guard, "always split the globally worst bucket" can fixate
    on such a region and keep re-bisecting it forever (observed empirically
    on the Concrete dataset), starving other under-threshold buckets of a
    turn.

    The comparison that decides "did this split help" is the total squared
    error over exactly the parent bucket's rows, before vs. after the split
    -- not R^2. Two R^2 numbers computed over different-sized sub-populations
    aren't comparable (a narrower slice has less of its own variance to
    explain, so its local R^2 is a strictly harder bar even for an equally
    good fit), and the *global* train R^2 is nearly always non-decreasing as
    rules are added regardless of whether this particular split helped, since
    the joint closed-form solve is weakly monotonic in free parameters. SSE
    over the same fixed row set, before and after, has neither problem. A
    split that doesn't reduce it has its child buckets excluded from future
    splitting; a candidate whose median coincides with an existing edge
    (can't be divided further) is excluded outright rather than stopping the
    whole loop.

    The stopping criterion is evaluated on `X_train`/`y_raw_train` directly
    (no internal validation split) -- this can overfit the training set if
    `r2_threshold` is pushed high with a small `min_bucket_samples`; that
    tradeoff is left to the caller, not guarded against here.

    `pin_extremes` is deliberately NOT applied while growing: pinning forces
    the current first/last bucket's constant to the observed global min/max
    of `y`, which is a sensible constraint on the final, wide outer buckets of
    a finished model but actively misleads the split decision while growing
    -- right after the very first split, both halves would be judged not
    against their own local mean but against the dataset's single most
    extreme value, which most of their rows aren't anywhere near. Growth
    always fits unpinned; if `pin_extremes` is True the final returned
    consequents get one last pinned solve after the edge set is fixed.
    """
    edges: list[float] = []
    history: list[dict] = []
    blocked: set[tuple[float, float]] = set()
    pending_split: tuple[tuple[float, float], float] | None = None  # (parent_interval, pre_split_sse)

    model = None
    y_part = None
    y_bucket_mean_raw = None
    y_bucket_mean = None
    corr_terms = None
    n_rules = 1

    while True:
        y_part, y_bucket_mean_raw = partition_output_by_edges(y_raw_train, edges)
        n_rules = len(edges) + 1
        intervals = _intervals_from_edges(edges)

        model = create_gaussian_membership_dict(
            X_train, y_part["y_bucket"], top_n_var_names=top_n_todo, n_gaussians=n_gaussians
        )
        corr_terms, y_bucket_mean = solve_tsk_consequents(
            X_train, model, top_n_todo, y_bucket_mean_raw, y_part,
            n_output_buckets=n_rules, order=tsk_order, l2_reg=l2_reg, basis=basis,
            cross_pairs=cross_pairs, pin_extremes=False, norms=norms, verbose=False,
        )
        y_pred = predict_tsk(
            X_train, model, top_n_todo, y_bucket_mean, corr_terms,
            order=tsk_order, basis=basis, cross_pairs=cross_pairs, norms=norms,
        )

        bucket_r2: dict[int, float] = {}
        bucket_counts: dict[int, int] = {}
        for rule_id in range(n_rules):
            mask = (y_part["y_bucket"] == rule_id).values
            bucket_counts[rule_id] = int(mask.sum())
            bucket_r2[rule_id] = _bucket_r2(y_part["y_value"].values[mask], y_pred[mask])

        train_r2 = _rsquared(y_part["y_value"].values, y_pred)
        y_values = y_part["y_value"].values

        if guard_stalled_splits and pending_split is not None:
            parent_interval, pre_split_sse = pending_split
            lo, hi = parent_interval
            # Same fixed row set the parent bucket covered -- rows don't move
            # between splits, so this is the exact "before" population.
            parent_rows = (y_values > lo) & (y_values <= hi)
            post_split_sse = float(np.sum((y_values[parent_rows] - y_pred[parent_rows]) ** 2))
            if post_split_sse >= pre_split_sse - 1e-9:
                for rule_id, interval in enumerate(intervals):
                    if interval[0] >= lo and interval[1] <= hi and interval != parent_interval:
                        blocked.add(interval)
            pending_split = None

        if verbose:
            print(f"  [adaptive] n_rules={n_rules} train_r2={train_r2:.4f} bucket_r2={bucket_r2} blocked={blocked}")
        history.append({
            "n_rules": n_rules,
            "edges": list(edges),
            "bucket_r2": dict(bucket_r2),
            "bucket_counts": dict(bucket_counts),
            "train_r2": train_r2,
            "blocked": len(blocked),
        })

        eligible = [
            rule_id for rule_id in range(n_rules)
            if bucket_r2[rule_id] < r2_threshold
            and bucket_counts[rule_id] >= min_bucket_samples
            and intervals[rule_id] not in blocked
        ]
        if not eligible or n_rules >= max_rules:
            break

        chosen_rule = None
        new_edge = None
        for rule_id in sorted(eligible, key=lambda rid: bucket_r2[rid]):
            mask = (y_part["y_bucket"] == rule_id).values
            bucket_y = y_part["y_value"].values[mask]
            if split_method == "sse":
                candidate_edge = _best_sse_split(bucket_y, min_leaf=max(1, min_bucket_samples // 2))
            else:
                candidate_edge = float(np.median(bucket_y))
            if candidate_edge is None or candidate_edge in edges:
                # This bucket can't be divided further (too few points on one
                # side, or a degenerate median); never reconsider it, and try
                # the next-worst candidate instead of giving up on the whole loop.
                blocked.add(intervals[rule_id])
                continue
            chosen_rule, new_edge = rule_id, candidate_edge
            break

        if chosen_rule is None:
            break

        chosen_mask = (y_part["y_bucket"] == chosen_rule).values
        pre_split_sse = float(np.sum((y_values[chosen_mask] - y_pred[chosen_mask]) ** 2))
        pending_split = (intervals[chosen_rule], pre_split_sse)
        edges = sorted(edges + [new_edge])

    if pin_extremes:
        # One last solve against the fixed, final edge set -- now that no more
        # splitting decisions depend on it, pin the outer buckets to the
        # observed range as requested.
        corr_terms, y_bucket_mean = solve_tsk_consequents(
            X_train, model, top_n_todo, y_bucket_mean_raw, y_part,
            n_output_buckets=n_rules, order=tsk_order, l2_reg=l2_reg, basis=basis,
            cross_pairs=cross_pairs, pin_extremes=True, norms=norms, verbose=False,
        )

    return AdaptivePartitionResult(
        model=model, y_part=y_part, y_bucket_mean=y_bucket_mean,
        corr_terms=corr_terms, edges=edges, history=history,
    )
