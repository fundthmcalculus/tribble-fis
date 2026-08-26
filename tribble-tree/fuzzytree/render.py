"""Human-readable rendering of a fitted fuzzy tree.

Views:
    * ``render_tree_text``       -- an indented IF-THEN rule tree with per-leaf stats.
    * ``plot_fuzzy_tree``        -- a matplotlib node-box diagram (no graphviz needed).
    * ``plot_deconstructed_tree`` -- the same kind of diagram for a
      ``DeconstructedHierarchicalRegressor`` (or a bare, unfitted
      ``TopologyNode``), so a user-supplied topology can be eyeballed --
      which sensors actually reached each leaf, whether a leaf starved down
      to a constant, and each branch's fitted combiner weights -- before or
      after fitting.

``plot_fuzzy_tree``/``plot_hme`` accept a fitted ``FuzzyRegressionTree`` or
``FuzzyClassificationTree`` and introspect the leaf consequents / class
distributions to annotate leaves.
"""

from __future__ import annotations

import numpy as np

from .terms import _OPEN_SHOULDER
from tribblefis.gauss_data import TrapezoidMembership


def _term_range_str(mf) -> str:
    """A compact numeric description of a linguistic term's core region."""
    if isinstance(mf, TrapezoidMembership):
        left_open = mf.b <= -_OPEN_SHOULDER / 2
        right_open = mf.c >= _OPEN_SHOULDER / 2
        if left_open and not right_open:
            return f"≤ {mf.c:.3g}"
        if right_open and not left_open:
            return f"≥ {mf.b:.3g}"
        return f"[{mf.b:.3g}, {mf.c:.3g}]"
    return f"~{mf.mu:.3g}"


def _leaf_label_regressor(est, leaf_id: int) -> str:
    mean = est.leaf_mean_[leaf_id]
    order = est.tsk_order
    if order == "0th" or est.corr_terms_.shape[1] == 0:
        return f"y ≈ {mean:.4g}"
    coeffs = est.corr_terms_[leaf_id]
    feats = est.top_features_
    parts = [f"{mean:.4g}"]
    # First len(feats) correction columns are the linear terms (order '1st'+).
    for j, name in enumerate(feats[: len(coeffs)]):
        c = coeffs[j]
        if abs(c) > 1e-6:
            parts.append(f"{c:+.3g}·{name}")
    tail = " ..." if len(coeffs) > len(feats) else ""
    return "y ≈ " + " ".join(parts) + tail


def _leaf_label_classifier(est, leaf_id: int) -> str:
    dist = est.leaf_class_dist_[leaf_id]
    top = int(np.argmax(dist))
    return f"class = {est.classes_[top]} (p={dist[top]:.2f})"


def _leaf_label(est, leaf_id: int) -> str:
    if hasattr(est, "leaf_class_dist_"):
        return _leaf_label_classifier(est, leaf_id)
    return _leaf_label_regressor(est, leaf_id)


def _leaf_label_short(est, leaf_id: int) -> str:
    """Compact leaf label for the diagram (full detail lives in the text view)."""
    if hasattr(est, "leaf_class_dist_"):
        return _leaf_label_classifier(est, leaf_id)
    mean = est.leaf_mean_[leaf_id]
    if est.tsk_order != "0th" and est.corr_terms_.shape[1] > 0:
        return f"y≈{mean:.3g}\n(+linear)"
    return f"y≈{mean:.3g}"


def render_tree_text(est, indent: str = "  ") -> str:
    """Return an indented IF-THEN rendering of the fitted tree."""
    lines: list[str] = []
    kind = "classification" if hasattr(est, "leaf_class_dist_") else "regression"
    lines.append(
        f"FuzzyTree ({kind}): {est.n_leaves_} leaves, "
        f"features={list(est.top_features_)}"
    )

    def recurse(node, depth: int, prefix: str) -> None:
        pad = indent * depth
        if node.is_leaf:
            lines.append(
                f"{pad}{prefix}=> {_leaf_label(est, node.leaf_id)} "
                f"(soft n={node.soft_mass:.1f})"
            )
            return
        for i, ((label, mf), child) in enumerate(zip(node.terms, node.children)):
            cond = f"IF {node.split_var} is {label} {_term_range_str(mf)}:"
            lines.append(f"{pad}{cond}")
            recurse(child, depth + 1, prefix="")

    recurse(est.tree_, 0, prefix="")
    return "\n".join(lines)


def _layout(node, next_x, positions):
    """Assign (x, y) positions: leaves get sequential x, internals the mean of
    their children; y = -depth."""
    if node.is_leaf:
        x = next_x[0]
        next_x[0] += 1
        positions[node.id] = (x, -node.depth)
        return x
    child_xs = [_layout(c, next_x, positions) for c in node.children]
    x = float(np.mean(child_xs))
    positions[node.id] = (x, -node.depth)
    return x


def plot_fuzzy_tree(est, ax=None, figsize=(11, 7), title: str | None = None):
    """Draw the tree as a labelled node-box diagram. Returns the matplotlib Figure.

    Matplotlib is imported lazily so the core library has no plotting dependency.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    positions: dict = {}
    _layout(est.tree_, [0], positions)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    def draw(node):
        x, y = positions[node.id]
        if node.is_leaf:
            text = _leaf_label_short(est, node.leaf_id)
            color = "#dbeafe"
        else:
            text = node.split_var
            color = "#fef3c7"
        box = FancyBboxPatch(
            (x - 0.45, y - 0.22),
            0.9,
            0.44,
            boxstyle="round,pad=0.02",
            linewidth=1.0,
            edgecolor="#334155",
            facecolor=color,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=8, zorder=3, wrap=True)

        for (label, mf), child in zip(node.terms, node.children):
            cx, cy = positions[child.id]
            ax.plot([x, cx], [y - 0.22, cy + 0.22], color="#94a3b8", lw=0.9, zorder=1)
            mx, my = (x + cx) / 2, (y - 0.22 + cy + 0.22) / 2
            ax.text(
                mx,
                my,
                f"{label}\n{_term_range_str(mf)}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="#475569",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8),
                zorder=3,
            )
            draw(child)

    draw(est.tree_)

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.axis("off")
    ax.set_title(title or "Fuzzy Tree", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hierarchical mixture of fuzzy experts (HME) rendering
# ---------------------------------------------------------------------------
def _hme_leaf_label(est, leaf_id: int) -> str:
    info = est.leaf_info_[leaf_id]
    if info["kind"] == "constant":
        return f"expert: constant (n={info['n_train']})"
    feats = info["features"][:3]
    tail = ", ..." if len(info["features"]) > 3 else ""
    return f"expert sub-FIS (n={info['n_train']})\non [{', '.join(feats)}{tail}]"


def render_hme_text(est, indent: str = "  ") -> str:
    """Indented rendering of an HME: gate (routing) nodes + expert leaves."""
    kind = "classification" if hasattr(est, "classes_") else "regression"
    lines = [
        f"HierarchicalFuzzyExperts ({kind}): {est.n_leaves_} experts, "
        f"gate features={list(est.gate_features_)}"
    ]

    def recurse(node, depth: int) -> None:
        pad = indent * depth
        if node.is_leaf:
            lines.append(f"{pad}=> {_hme_leaf_label(est, node.leaf_id)}")
            return
        for (label, mf), child in zip(node.terms, node.children):
            lines.append(
                f"{pad}ROUTE {node.split_var} is {label} {_term_range_str(mf)}:"
            )
            recurse(child, depth + 1)

    recurse(est.tree_, 0)
    return "\n".join(lines)


def plot_hme(est, ax=None, figsize=(12, 7), title: str | None = None):
    """Draw an HME: yellow gate (routing) nodes, green expert leaves."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    positions: dict = {}
    _layout(est.tree_, [0], positions)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    def draw(node):
        x, y = positions[node.id]
        if node.is_leaf:
            text = _hme_leaf_label(est, node.leaf_id)
            color = "#dcfce7"
        else:
            text = f"route on\n{node.split_var}"
            color = "#fef3c7"
        box = FancyBboxPatch(
            (x - 0.48, y - 0.24), 0.96, 0.48,
            boxstyle="round,pad=0.02", linewidth=1.0,
            edgecolor="#334155", facecolor=color, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5, zorder=3)
        for (label, mf), child in zip(node.terms, node.children):
            cx, cy = positions[child.id]
            ax.plot([x, cx], [y - 0.24, cy + 0.24], color="#94a3b8", lw=0.9, zorder=1)
            mx, my = (x + cx) / 2, (y - 0.24 + cy + 0.24) / 2
            ax.text(
                mx, my, f"{label}\n{_term_range_str(mf)}",
                ha="center", va="center", fontsize=6.5, color="#475569",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8),
                zorder=3,
            )
            draw(child)

    draw(est.tree_)
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
    ax.axis("off")
    ax.set_title(title or "Hierarchical Fuzzy Experts", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Deconstructed hierarchical FIS rendering (fuzzytree.deconstruct)
# ---------------------------------------------------------------------------
_MAX_FEATURES_SHOWN = 4


def _feature_preview(names) -> str:
    names = list(names)
    shown = ", ".join(names[:_MAX_FEATURES_SHOWN])
    return shown + (", ..." if len(names) > _MAX_FEATURES_SHOWN else "")


def _layout_topology(node, depth, next_x, positions):
    """Same idea as ``_layout``, but keyed by ``node.name`` -- a `TopologyNode`
    has no ``.id``/``.depth`` field, and its uniqueness (one name per tree,
    enforced by ``parse_topology``) makes the name a valid position key."""
    if node.is_leaf:
        x = next_x[0]
        next_x[0] += 1
        positions[node.name] = (x, -depth)
        return x
    child_xs = [_layout_topology(c, depth + 1, next_x, positions) for c in node.children]
    x = float(np.mean(child_xs))
    positions[node.name] = (x, -depth)
    return x


def _deconstructed_leaf_text(node, state) -> str:
    declared = node.own_features
    if state is None:
        return f"{node.name}\n[{len(declared)}] {_feature_preview(declared)}"
    if state["kind"] == "constant":
        return f"{node.name}\nCONSTANT (0/{len(declared)} features survived)"
    used = state["top_n_todo"]
    if len(used) < len(declared):
        return f"{node.name}\nused {len(used)}/{len(declared)}: {_feature_preview(used)}"
    return f"{node.name}\n[{len(used)}] {_feature_preview(used)}"


def _deconstructed_branch_text(node, state) -> str:
    if state is None:
        return f"{node.name}\n({len(node.children)} children)"
    if "per_class" in state:
        # Classifier branch: one (corr_terms, y_bucket_mean) pair per class,
        # not a single intercept -- name the thing there is one of instead.
        return f"{node.name}\n{len(state['per_class'])} classes"
    a0 = state["y_bucket_mean"][0]
    return f"{node.name}\na0={a0:+.3g}"


def _deconstructed_edge_label(state, child_name: str) -> str | None:
    if state is None or state.get("kind") != "branch":
        return None
    try:
        idx = state["children"].index(child_name)
    except ValueError:
        return None
    if "per_class" in state:
        # Mean of that child's per-class combiner weight -- a compact stand-in
        # for the n_classes individual weights, clearly flagged as an average.
        weights = [corr_terms[0][idx] for corr_terms, _ in state["per_class"].values()]
        return f"×{float(np.mean(weights)):+.3g} avg"
    coeff = state["corr_terms"][0][idx]
    return f"×{coeff:+.3g}"


def plot_deconstructed_tree(model, ax=None, figsize=(11, 7), title: str | None = None, save_path: str | None = None):
    """Draw a `DeconstructedHierarchicalRegressor`'s tree as a node-box diagram.

    Accepts either a *fitted* estimator (leaf boxes then show how many of the
    declared features actually survived the flat model's own feature
    selection, a constant-fallback flag, and each branch's fitted combiner
    weight on every child edge) or a bare, unfitted `TopologyNode` (e.g. from
    `parse_topology` directly) -- so a topology can be sanity-checked before
    ever fitting anything. Blue leaves are ordinary leaves, amber nodes are
    branches, and a leaf that starved down to a constant is flagged in red.

    ``save_path``, if given, is passed straight to ``Figure.savefig`` --
    matplotlib infers the format from the extension (``.svg``, ``.png``,
    ``.pdf``, ...). The figure is returned either way.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    root = model.root_ if hasattr(model, "root_") else model
    node_state = getattr(model, "node_state_", None)

    positions: dict = {}
    _layout_topology(root, 0, [0], positions)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    def state_of(name):
        return node_state.get(name) if node_state is not None else None

    def draw(node):
        x, y = positions[node.name]
        state = state_of(node.name)
        if node.is_leaf:
            text = _deconstructed_leaf_text(node, state)
            color = "#fee2e2" if (state and state["kind"] == "constant") else "#dbeafe"
        else:
            text = _deconstructed_branch_text(node, state)
            color = "#fef3c7"
        box = FancyBboxPatch(
            (x - 0.52, y - 0.24), 1.04, 0.48,
            boxstyle="round,pad=0.02", linewidth=1.0,
            edgecolor="#334155", facecolor=color, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5, zorder=3)

        for child in node.children:
            cx, cy = positions[child.name]
            ax.plot([x, cx], [y - 0.24, cy + 0.24], color="#94a3b8", lw=0.9, zorder=1)
            label = _deconstructed_edge_label(state, child.name)
            if label:
                mx, my = (x + cx) / 2, (y - 0.24 + cy + 0.24) / 2
                ax.text(
                    mx, my, label, ha="center", va="center", fontsize=7,
                    color="#475569",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85),
                    zorder=3,
                )
            draw(child)

    draw(root)

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
    ax.axis("off")
    ax.set_title(title or "Deconstructed Hierarchical FIS", fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)

    return fig
