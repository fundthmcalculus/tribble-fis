"""Recursive fuzzy partitioning: grow the tree structure.

The builder only decides *structure* (which variable each node splits on and its
linguistic terms). Leaf consequents are solved globally afterwards by
``solve.solve_leaf_consequents`` once every leaf and the full leaf firing matrix
are known -- exactly as the flat model solves all rule consequents in one shot.

Splitting stops at any of: ``max_depth``; soft sample mass below
``min_soft_count``; best auto criterion gain below ``min_gain``; a degenerate
variable (no spread, ``build_split_terms`` returns ``[]``); an exhausted candidate
pool; or the ``max_leaves`` cap. The ``max_leaves`` bound is exact: a k-way split
turns one prospective leaf into k, so we track the projected leaf count and refuse
a split that would exceed the cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .node import FuzzyTreeNode
from .plan import AUTO, VariablePlan, candidate_pool, resolve_split_variable
from .splitter import get_criterion
from .terms import build_split_terms


def build_tree(
    X_df: pd.DataFrame,
    y_value: np.ndarray,
    y_bucket: np.ndarray,
    plan: VariablePlan,
    t_norm_name: str,
    min_soft_count: float = 5.0,
    min_gain: float = 1e-3,
    max_leaves: int = 64,
) -> tuple[FuzzyTreeNode, int]:
    """Grow a fuzzy tree; return ``(root, n_leaves)``."""
    all_vars = list(X_df.columns)
    criterion_fn = get_criterion(plan.criterion)

    var_by_path: dict[tuple[str, ...], str] = {}
    leaf_counter = [0]
    projected_leaves = [1]  # a bare root would be one leaf

    y_value = np.asarray(y_value, dtype=float)
    y_bucket = np.asarray(y_bucket)

    def make_leaf(depth: int, path: tuple[str, ...], w: np.ndarray) -> FuzzyTreeNode:
        leaf_id = leaf_counter[0]
        leaf_counter[0] += 1
        return FuzzyTreeNode.create_leaf(
            depth=depth, path=path, leaf_id=leaf_id, soft_mass=float(w.sum())
        )

    def choose_auto(path, w, pool, n_terms):
        """Score every candidate variable and return the best (var, terms, gain)."""
        best_var, best_terms, best_score = None, None, -np.inf
        for var in pool:
            col = X_df[var].to_numpy(dtype=float)
            terms = build_split_terms(col, w, n_terms, plan.term_labels, plan.term_style)
            if not terms:
                continue
            score = criterion_fn(col, y_value, y_bucket, w, terms)
            if score > best_score:
                best_var, best_terms, best_score = var, terms, score
        return best_var, best_terms, best_score

    def recurse(path: tuple[str, ...], depth: int, w: np.ndarray) -> FuzzyTreeNode:
        soft_mass = float(w.sum())
        decision, n_terms = resolve_split_variable(plan, path, depth, all_vars, var_by_path)

        if decision is None or soft_mass < min_soft_count:
            return make_leaf(depth, path, w)

        # Determine the split variable and its terms.
        if decision is AUTO:
            pool = candidate_pool(plan, path, all_vars, var_by_path)
            chosen_var, terms, score = choose_auto(path, w, pool, n_terms)
            if chosen_var is None or score < min_gain:
                return make_leaf(depth, path, w)
        else:
            chosen_var = decision
            col = X_df[chosen_var].to_numpy(dtype=float)
            terms = build_split_terms(col, w, n_terms, plan.term_labels, plan.term_style)
            if not terms:  # degenerate variable at this node
                return make_leaf(depth, path, w)

        # Enforce the exact max_leaves bound: a k-way split adds (k-1) leaves.
        k = len(terms)
        if projected_leaves[0] + (k - 1) > max_leaves:
            return make_leaf(depth, path, w)
        projected_leaves[0] += k - 1

        var_by_path[path] = chosen_var
        col = X_df[chosen_var].to_numpy(dtype=float)
        children = []
        for label, mf in terms:
            child_w = w * mf.evaluate(col)
            children.append(recurse(path + (label,), depth + 1, child_w))

        return FuzzyTreeNode.create_internal(
            depth=depth,
            path=path,
            split_var=chosen_var,
            terms=terms,
            children=children,
            soft_mass=soft_mass,
        )

    root = recurse((), 0, np.ones(len(X_df), dtype=float))
    return root, leaf_counter[0]
