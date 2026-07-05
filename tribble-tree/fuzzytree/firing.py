"""Leaf firing strengths for a fuzzy tree.

A leaf's firing strength for a sample is the t-norm (AND) of the branch
memberships along the root->leaf path. With the product t-norm this is just the
product of the memberships; a sample flows into *every* leaf with some partial
mass. The resulting ``(n_samples, n_leaves)`` matrix is the exact analogue of the
flat model's ``(n_samples, n_rules)`` firing matrix, so it plugs straight into the
same normalisation and consequent-solve machinery.

The *same* t-norm must be used for split-weight propagation during fitting and for
this matrix at both fit and predict time, or training and evaluation silently
disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tribblefis.gauss_math import t_norm

# Product t-norm ("probability") gives smooth path weights and is the standard
# choice for soft trees; callers may override.
DEFAULT_T_NORM = "probability"


def compute_leaf_firing(
    tree,
    X_df: pd.DataFrame,
    n_leaves: int,
    t_norm_name: str = DEFAULT_T_NORM,
) -> np.ndarray:
    """Return the ``(n_samples, n_leaves)`` leaf firing-strength matrix."""
    n = len(X_df)
    firing = np.zeros((n, n_leaves), dtype=float)

    def recurse(node, w: np.ndarray) -> None:
        if node.is_leaf:
            firing[:, node.leaf_id] = w
            return
        col = X_df[node.split_var].to_numpy(dtype=float)
        for (_label, mf), child in zip(node.terms, node.children):
            mu = mf.evaluate(col)
            recurse(child, t_norm(w, mu, t_norm_name))

    recurse(tree, np.ones(n, dtype=float))
    return firing
