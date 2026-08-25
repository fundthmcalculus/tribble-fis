"""User-supplied tree topology for :mod:`fuzzytree.deconstruct`.

A topology is a plain ``dict[str, list[str]]`` mapping a node name to its
children. A node is a **branch** if every one of its children is itself a
key of the dict, or a **leaf** if every one of its children is a column name
of ``X`` (mixing the two on one node is rejected -- see ``parse_topology``).
The root is the one key that never appears as anyone's child.

This is deliberately much simpler than :class:`fuzzytree.plan.VariablePlan`:
``VariablePlan`` drives *automatic* structure discovery (one variable per
split, chosen by a criterion, user-overridable). A topology here is instead
*fully* user-specified up front -- the whole point of the deconstruction
approach is that structure comes from domain knowledge, not from the data.
"""

from __future__ import annotations

from typing import NamedTuple


class TopologyNode(NamedTuple):
    """A node in a user-supplied tree topology.

    Attributes:
        name: This node's key in the topology dict (or, for the root, its
            key too -- there is no separate "root" concept).
        children: Child nodes. Empty for a leaf.
        own_features: For a leaf, the feature-column names it owns. Empty
            for a branch.
    """

    name: str
    children: list["TopologyNode"]
    own_features: list[str]

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def iter_nodes(self):
        """Yield this node and all descendants (pre-order)."""
        yield self
        for child in self.children:
            yield from child.iter_nodes()

    def iter_leaves(self):
        """Yield every leaf under this node, left-to-right."""
        if self.is_leaf:
            yield self
            return
        for child in self.children:
            yield from child.iter_leaves()


def parse_topology(topology: dict[str, list[str]], feature_names: list[str]) -> TopologyNode:
    """Validate and resolve a user-supplied topology dict into a `TopologyNode` tree.

    Args:
        topology: node name -> list of children. A branch node's children
            must all be other keys of this dict; a leaf node's children must
            all be entries of ``feature_names``. A node may not mix the two.
        feature_names: The columns of the training `X`.

    Raises:
        ValueError: on anything that would make the topology ambiguous or
            malformed -- a node name colliding with a feature name, zero or
            more than one root, a child that is neither a node nor a
            feature, a node mixing branch/leaf children, a cycle or reused
            node (this is a tree, not a DAG), or a node unreachable from the
            root.
    """
    keys = set(topology.keys())
    feature_set = set(feature_names)

    overlap = keys & feature_set
    if overlap:
        raise ValueError(
            f"Topology node name(s) {sorted(overlap)} collide with feature "
            f"column names; rename the node(s)."
        )

    all_children = [c for children in topology.values() for c in children]
    roots = keys - set(all_children)
    if len(roots) != 1:
        raise ValueError(
            f"Topology must have exactly one root (a key that is never a "
            f"child of another key); found {sorted(roots)}."
        )
    root_name = next(iter(roots))

    visited: set[str] = set()

    def build(name: str) -> TopologyNode:
        if name in visited:
            raise ValueError(
                f"Topology node {name!r} is reached more than once (a cycle "
                f"or a reused node) -- this must be a tree, not a DAG."
            )
        visited.add(name)
        children_names = topology[name]
        if not children_names:
            raise ValueError(f"Topology node {name!r} has no children.")

        child_keys = [c for c in children_names if c in keys]
        child_features = [c for c in children_names if c in feature_set]
        unknown = [c for c in children_names if c not in keys and c not in feature_set]
        if unknown:
            raise ValueError(
                f"Topology node {name!r} lists {unknown!r}, which are "
                f"neither other topology nodes nor columns of X."
            )
        if child_keys and child_features:
            raise ValueError(
                f"Topology node {name!r} mixes sub-nodes {child_keys!r} with "
                f"direct feature columns {child_features!r}; a node must be "
                f"a pure branch (all children are other nodes) or a pure "
                f"leaf (all children are feature columns)."
            )

        if child_features:
            return TopologyNode(name=name, children=[], own_features=child_features)
        return TopologyNode(
            name=name,
            children=[build(c) for c in child_keys],
            own_features=[],
        )

    root = build(root_name)
    unreached = keys - visited
    if unreached:
        raise ValueError(
            f"Topology node(s) {sorted(unreached)} are unreachable from root "
            f"{root_name!r}."
        )
    return root
