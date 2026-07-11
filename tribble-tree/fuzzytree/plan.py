"""User-facing variable plan: how variables are assigned to tree nodes.

This satisfies both halves of the request: variables are ordered *automatically*
by a split criterion, but the user may also *pin* a specific variable to a
specific node (identified by its branch path) or fix a global per-level ordering.

``resolve_split_variable`` implements the precedence:

    path-pin  >  level-order  >  auto-by-criterion

Hard global gates (``exclude``, ``max_depth``, ``max_terms_per_var``) apply
regardless of where the choice came from. A pin that names an excluded or missing
variable is dropped (with a warning) rather than silently violating a constraint.

The plan is a plain immutable value object, so it round-trips trivially to/from a
dict/JSON for a config-file front end (see ``to_dict``/``from_dict``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Optional

# Sentinel returned by resolve_split_variable when the builder should pick the
# split variable itself using the configured criterion.
AUTO = "__auto__"


@dataclass(frozen=True)
class NodePin:
    """Pin a variable to the node reached by exactly this branch path.

    ``path=()`` targets the root; ``path=("Low",)`` targets the child under the
    root's "Low" branch; ``path=("Low", "High")`` a grandchild, and so on.
    """

    path: tuple[str, ...]
    variable: str
    n_terms: Optional[int] = None


@dataclass(frozen=True)
class VariablePlan:
    """Declarative control over tree structure.

    Attributes:
        level_order: Global per-level ordering. At depth ``d`` every node splits
            on ``level_order[d]``. Use ``None`` in a slot to leave that level to
            auto selection. An empty tuple means "fully automatic".
        pins: Exact node-path overrides (highest precedence).
        criterion: Split criterion for auto-selected nodes -- one of
            ``"variance"``, ``"ambiguity"``, ``"differentiation"``.
        exclude: Variables that must never be split on.
        max_depth: Maximum tree depth (root is depth 0).
        default_n_terms: Linguistic terms per split variable when unspecified.
        max_terms_per_var: Hard cap on terms per split.
        term_labels: Labels applied left-to-right to a split's terms.
        no_reuse_on_path: If True, a variable may split at most once on any path.
        term_style: ``"trapezoid"`` (default) or ``"gaussian"``.
    """

    level_order: tuple[Optional[str], ...] = ()
    pins: tuple[NodePin, ...] = ()
    criterion: str = "variance"
    exclude: frozenset[str] = field(default_factory=frozenset)
    max_depth: int = 3
    default_n_terms: int = 3
    max_terms_per_var: int = 3
    term_labels: tuple[str, ...] = ("Low", "Med", "High")
    no_reuse_on_path: bool = False
    term_style: str = "trapezoid"

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "level_order": list(self.level_order),
            "pins": [
                {"path": list(p.path), "variable": p.variable, "n_terms": p.n_terms}
                for p in self.pins
            ],
            "criterion": self.criterion,
            "exclude": sorted(self.exclude),
            "max_depth": self.max_depth,
            "default_n_terms": self.default_n_terms,
            "max_terms_per_var": self.max_terms_per_var,
            "term_labels": list(self.term_labels),
            "no_reuse_on_path": self.no_reuse_on_path,
            "term_style": self.term_style,
        }

    @staticmethod
    def from_dict(d: dict) -> "VariablePlan":
        pins = tuple(
            NodePin(path=tuple(p["path"]), variable=p["variable"], n_terms=p.get("n_terms"))
            for p in d.get("pins", [])
        )
        return VariablePlan(
            level_order=tuple(d.get("level_order", [])),
            pins=pins,
            criterion=d.get("criterion", "variance"),
            exclude=frozenset(d.get("exclude", [])),
            max_depth=d.get("max_depth", 3),
            default_n_terms=d.get("default_n_terms", 3),
            max_terms_per_var=d.get("max_terms_per_var", 3),
            term_labels=tuple(d.get("term_labels", ("Low", "Med", "High"))),
            no_reuse_on_path=d.get("no_reuse_on_path", False),
            term_style=d.get("term_style", "trapezoid"),
        )

    def with_criterion(self, criterion: str) -> "VariablePlan":
        return replace(self, criterion=criterion)


def path_variables(path: tuple[str, ...], var_by_path: dict[tuple[str, ...], str]) -> set[str]:
    """Variables already used on the ancestors of ``path``.

    ``var_by_path`` maps each ancestor node path to the variable it split on.
    """
    used: set[str] = set()
    for d in range(len(path)):
        ancestor = path[:d]
        if ancestor in var_by_path:
            used.add(var_by_path[ancestor])
    return used


def candidate_pool(
    plan: VariablePlan,
    path: tuple[str, ...],
    available_vars: list[str],
    var_by_path: dict[tuple[str, ...], str],
) -> list[str]:
    """Variables eligible to split on at ``path`` (honours exclude / no-reuse)."""
    pool = [v for v in available_vars if v not in plan.exclude]
    if plan.no_reuse_on_path:
        used = path_variables(path, var_by_path)
        pool = [v for v in pool if v not in used]
    return pool


def resolve_split_variable(
    plan: VariablePlan,
    path: tuple[str, ...],
    depth: int,
    available_vars: list[str],
    var_by_path: dict[tuple[str, ...], str],
) -> tuple[Optional[str], int]:
    """Decide what to split on at a node.

    Returns ``(decision, n_terms)`` where ``decision`` is:
        * a variable name  -> split on it (pin or level-order),
        * ``AUTO``         -> builder chooses via the criterion,
        * ``None``         -> make this node a leaf.
    """
    if depth >= plan.max_depth:
        return None, 0

    # Candidate pool honours global exclusions and (optionally) no-reuse.
    pool = candidate_pool(plan, path, available_vars, var_by_path)
    if not pool:
        return None, 0

    default_terms = min(plan.default_n_terms, plan.max_terms_per_var)

    # 1) Path pin (highest precedence, exact branch match).
    for pin in plan.pins:
        if tuple(pin.path) == tuple(path):
            if pin.variable in plan.exclude:
                warnings.warn(
                    f"NodePin at path {path} names excluded variable "
                    f"{pin.variable!r}; falling back to auto selection."
                )
                break
            if pin.variable not in pool:
                warnings.warn(
                    f"NodePin at path {path} names variable {pin.variable!r} "
                    f"which is unavailable here; falling back to auto selection."
                )
                break
            n_terms = min(pin.n_terms or plan.default_n_terms, plan.max_terms_per_var)
            return pin.variable, n_terms

    # 2) Global per-level ordering.
    if depth < len(plan.level_order) and plan.level_order[depth] is not None:
        v = plan.level_order[depth]
        if v in pool:
            return v, default_terms
        warnings.warn(
            f"level_order[{depth}]={v!r} is unavailable/excluded at path {path}; "
            f"falling back to auto selection."
        )

    # 3) Auto selection by criterion.
    return AUTO, default_terms
