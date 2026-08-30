"""fuzzytree -- a hierarchical fuzzy (soft) decision/regression tree built on the
TRIBBLE TSK fuzzy inference primitives.

Each internal node splits one input variable into fuzzy linguistic terms; samples
flow down all branches with partial membership; leaves hold local TSK consequents
(regression) or fuzzy class votes (classification). Variable-to-node assignment is
automatic by a split criterion but user-overridable via ``VariablePlan``.
"""

from .plan import AUTO, NodePin, VariablePlan
from .node import FuzzyTreeNode
from .regressor import FuzzyRegressionTree, MimoFuzzyTreeRegressor
from .classifier import FuzzyClassificationTree
from .hme import (
    HierarchicalFuzzyExpertsClassifier,
    HierarchicalFuzzyExpertsRegressor,
)
from .em import refine_em_classifier, refine_em_regressor
from .render import (
    render_tree_text,
    plot_fuzzy_tree,
    render_hme_text,
    plot_hme,
    plot_deconstructed_tree,
)
from .topology import TopologyNode, parse_topology
from .auto_topology import (
    affinity_topology,
    candidate_topologies,
    cluster_features,
    feature_affinity,
    per_feature_topology,
    select_topology,
)
from .deconstruct import DeconstructedHierarchicalRegressor, DeconstructedHierarchicalClassifier

__all__ = [
    "VariablePlan",
    "NodePin",
    "AUTO",
    "FuzzyTreeNode",
    "FuzzyRegressionTree",
    "MimoFuzzyTreeRegressor",
    "FuzzyClassificationTree",
    "HierarchicalFuzzyExpertsRegressor",
    "HierarchicalFuzzyExpertsClassifier",
    "refine_em_regressor",
    "refine_em_classifier",
    "render_tree_text",
    "plot_fuzzy_tree",
    "render_hme_text",
    "plot_hme",
    "plot_deconstructed_tree",
    "TopologyNode",
    "affinity_topology",
    "candidate_topologies",
    "cluster_features",
    "feature_affinity",
    "per_feature_topology",
    "select_topology",
    "parse_topology",
    "DeconstructedHierarchicalRegressor",
    "DeconstructedHierarchicalClassifier",
]
