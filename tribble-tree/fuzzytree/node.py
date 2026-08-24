"""Fuzzy tree node data model.

A ``FuzzyTreeNode`` is an immutable ``NamedTuple`` (matching the convention used
throughout ``tribblefis.gauss_data``). Internal nodes split on a single input
variable using fuzzy linguistic terms; a sample flows down *all* branches with a
partial membership degree, so the tree is a *soft* (fuzzy) tree rather than a
crisp decision tree. Leaves carry a local TSK consequent (constant for
``tsk_order='0th'`` or linear/polynomial for higher orders) or, for the
classifier, fuzzy class votes.
"""

from __future__ import annotations

import uuid
from typing import NamedTuple, Optional

import numpy as np

from tribblefis.gauss_data import AnyMembership


class FuzzyTreeNode(NamedTuple):
    """A node in a fuzzy regression/classification tree.

    Attributes:
        depth: Distance from the root (root has depth 0).
        path: Branch labels taken from the root to reach this node, e.g.
            ``("Low", "High")``. The root is ``()``.
        split_var: The input variable this node splits on, or ``None`` for a leaf.
        terms: For an internal node, the ordered ``(label, membership)`` pairs
            that define the branches (sorted left-to-right by membership centre,
            so ``terms[0]`` is the leftmost/"Low" branch). Empty for a leaf.
        children: Child nodes, aligned index-for-index with ``terms``. Empty for
            a leaf.
        leaf_id: For a leaf, its column index in the leaf firing-strength matrix
            (and therefore into the solved consequent arrays). ``None`` for an
            internal node.
        soft_mass: Sum of fuzzy path weights of the training samples that reached
            this node (a "soft" sample count).
        id: Unique identifier.
    """

    depth: int
    path: tuple[str, ...]
    split_var: Optional[str]
    terms: list[tuple[str, AnyMembership]]
    children: list["FuzzyTreeNode"]
    leaf_id: Optional[int] = None
    soft_mass: float = 0.0
    id: Optional[uuid.UUID] = None

    @staticmethod
    def create_leaf(
        depth: int,
        path: tuple[str, ...],
        leaf_id: int,
        soft_mass: float,
    ) -> "FuzzyTreeNode":
        return FuzzyTreeNode(
            depth=depth,
            path=path,
            split_var=None,
            terms=[],
            children=[],
            leaf_id=leaf_id,
            soft_mass=soft_mass,
            id=uuid.uuid4(),
        )

    @staticmethod
    def create_internal(
        depth: int,
        path: tuple[str, ...],
        split_var: str,
        terms: list[tuple[str, AnyMembership]],
        children: list["FuzzyTreeNode"],
        soft_mass: float,
    ) -> "FuzzyTreeNode":
        return FuzzyTreeNode(
            depth=depth,
            path=path,
            split_var=split_var,
            terms=terms,
            children=children,
            leaf_id=None,
            soft_mass=soft_mass,
            id=uuid.uuid4(),
        )

    @property
    def is_leaf(self) -> bool:
        return self.split_var is None

    def iter_leaves(self):
        """Yield every leaf under this node, left-to-right."""
        if self.is_leaf:
            yield self
            return
        for child in self.children:
            yield from child.iter_leaves()

    def iter_nodes(self):
        """Yield this node and all descendants (pre-order)."""
        yield self
        for child in self.children:
            yield from child.iter_nodes()

    @property
    def n_leaves(self) -> int:
        return sum(1 for _ in self.iter_leaves())
