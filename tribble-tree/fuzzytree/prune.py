"""Post-hoc split-gain pruning for fuzzy (soft) trees.

CART's minimal cost-complexity pruning scores a subtree by an *additive* sum
of per-leaf costs, R(T) = sum_leaf (N_leaf/N) * impurity_leaf, which relies on
leaves partitioning the training set disjointly (each sample belongs to
exactly one leaf). Fuzzy/soft leaves do not have that property: firing
weights overlap (a sample has partial membership in many leaves at once), so
summing per-leaf weighted SSE/Gini across a subtree does not correctly
decompose the parent's impurity -- collapsing an 8-leaf soft tree to 1 leaf
was measured to look "cheaper" than keeping the split even with zero penalty,
which would be a nonsensical pruning rule.

Instead, pruning here re-scores each internal node's *own* split with the same
normalized gain criterion used to build it (`splitter.get_criterion`) -- the
same reduction-in-variance / reduction-in-ambiguity score `build_tree` already
uses and stops on via `min_gain`. A node whose recomputed gain falls below
``ccp_alpha`` is collapsed into a leaf; since the criterion is purely local
(a function of this node's own terms and the samples reaching it), the
decision does not depend on what is below it, and pruning is a simple
top-down threshold sweep: any subtree under a collapsed node is dropped
without being visited.

Because `build_tree` already refuses to create a split scoring below its own
``min_gain``, pruning is only useful with ``ccp_alpha`` set *higher* than the
``min_gain`` used at build time: grow a deeper tree with a lenient
``min_gain`` (and/or a generous ``max_depth``/``max_leaves``), then prune back
to whatever gain threshold you actually want, as an independent knob from the
build-time stopping rules. ``ccp_alpha=0`` (the default everywhere it's
threaded in) is a no-op, since every surviving split already scored above the
build's own (strictly positive, by default) ``min_gain``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from tribblefis.gauss_math import t_norm

from .node import FuzzyTreeNode


def _prune(node, criterion_fn, X_top, y_value, y_bucket, w, ccp_alpha, t_norm_name):
    if node.is_leaf:
        return node

    col = X_top[node.split_var].to_numpy(dtype=float)
    gain = criterion_fn(col, y_value, y_bucket, w, node.terms)
    if gain < ccp_alpha:
        return FuzzyTreeNode.create_leaf(
            depth=node.depth, path=node.path, leaf_id=-1, soft_mass=float(w.sum())
        )

    new_children = [
        _prune(child, criterion_fn, X_top, y_value, y_bucket, t_norm(w, mf.evaluate(col), t_norm_name), ccp_alpha, t_norm_name)
        for (_label, mf), child in zip(node.terms, node.children)
    ]
    return FuzzyTreeNode.create_internal(
        depth=node.depth,
        path=node.path,
        split_var=node.split_var,
        terms=node.terms,
        children=new_children,
        soft_mass=node.soft_mass,
    )


def _renumber(node: FuzzyTreeNode, counter: list[int]) -> FuzzyTreeNode:
    if node.is_leaf:
        leaf_id = counter[0]
        counter[0] += 1
        return node._replace(leaf_id=leaf_id)
    return node._replace(children=[_renumber(c, counter) for c in node.children])


def prune_tree(
    tree: FuzzyTreeNode,
    X_top,
    criterion_fn: Callable,
    y_value: np.ndarray,
    y_bucket: np.ndarray,
    ccp_alpha: float,
    t_norm_name: str = "probability",
) -> tuple[FuzzyTreeNode, int]:
    """Collapse any internal node whose own split gain (under ``criterion_fn``,
    e.g. ``splitter.variance_reduction`` or ``splitter.classification_ambiguity``)
    falls below ``ccp_alpha``. Returns ``(new_tree, n_leaves)`` with leaves
    renumbered contiguously from 0. Callers must re-solve leaf consequents /
    class distributions against this new leaf set -- pruning only changes
    structure. ``ccp_alpha <= 0`` is a no-op.
    """
    if ccp_alpha <= 0:
        return tree, (tree.n_leaves if not tree.is_leaf else 1)

    n = len(X_top)
    w0 = np.ones(n, dtype=float)
    pruned = _prune(tree, criterion_fn, X_top, y_value, y_bucket, w0, ccp_alpha, t_norm_name)
    counter = [0]
    renumbered = _renumber(pruned, counter)
    return renumbered, counter[0]
