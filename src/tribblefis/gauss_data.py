import uuid
from math import prod
from typing import NamedTuple, Literal, Optional

import numpy as np

# norm/conorm pairings
NormConorm = Literal["min/max", "probability", "luk", "hamacher"]
MemberFunction = Literal["gaussian", "triangular", "trap"]
DefaultNormCornorm: NormConorm = "min/max"
DefaultMemberFunction: MemberFunction = "gaussian"


class AnomalyParameters(NamedTuple):
    """Parameters for anomaly detection"""

    include_anomaly: bool = True
    threshold: float = 0.5
    label: str = "anomaly"
    norm_conorm: NormConorm = DefaultNormCornorm
    member_function: MemberFunction = DefaultMemberFunction


class GaussianMembership(NamedTuple):
    """A single Gaussian membership function."""
    mu: float
    sigma: float
    id: Optional[uuid.UUID] = None

    @staticmethod
    def create(mu: float, sigma: float) -> "GaussianMembership":
        return GaussianMembership(mu=mu, sigma=sigma, id=uuid.uuid4())


class Rule(NamedTuple):
    """A fuzzy rule mapping input membership functions to an output label."""

    antecedents: dict[str, list[uuid.UUID]]  # maps feature name to ids of GaussianMembership
    consequent: int | str  # output label


class SimpleGaussianClassifierModel(NamedTuple):
    """A simple Gaussian classifier model with explicit rules."""

    input_mfs: list[GaussianMembership]
    rules: list[Rule]
    anomaly_params: Optional[AnomalyParameters] = None

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    @property
    def all_features(self) -> list[str]:
        return list(set().union(*[rule.antecedents.keys() for rule in self.rules]))

    @property
    def n_features(self) -> int:
        return len(self.all_features)

    def get_mfs(self, ids: list[uuid.UUID]) -> list[GaussianMembership]:
        return [mf for mf in self.input_mfs if mf.id in ids]

    def get_mfs_for_feature(self, feature_name: str) -> list[GaussianMembership]:
        return list(set([mf for rule in self.rules for mf in self.get_mfs(rule.antecedents.get(feature_name, []))]))


class LabelModel(NamedTuple):
    """A collection of Gaussian membership functions for a specific output class label."""

    gaussians: list[GaussianMembership]

    def augment(self, other_label_model) -> "LabelModel":
        """Augment this LabelModel with another LabelModel, combining Gaussian membership."""
        new_gaussian_memberships = self.gaussians.copy()
        new_gaussian_memberships.extend(other_label_model.gaussians)
        return LabelModel(new_gaussian_memberships)


class FeatureModel(NamedTuple):
    """A collection of LabelModels for a specific feature, mapping labels to their respective models."""

    label_models: dict[int, LabelModel]

    @property
    def ordered_keys(self) -> list[int]:
        return list(sorted(self.label_models.keys()))

    def augment(self, other_feature_model) -> "FeatureModel":
        """Augment this FeatureModel with another FeatureModel, combining label models."""
        new_label_models = self.label_models.copy()
        for label, other_label_model in other_feature_model.label_models.items():
            if label in new_label_models:
                new_label_models[label] = new_label_models[label].augment(other_label_model)
            else:
                new_label_models[label] = other_label_model
        return FeatureModel(new_label_models)


class GaussianMixtureModel(NamedTuple):
    """A collection of FeatureModels mapping feature names to their models."""

    feature_models: dict[str, FeatureModel]
    anomaly_params: Optional[AnomalyParameters] = None

    @property
    def n_rules(self) -> int:
        return len(list(self.feature_models.values())[0].label_models.keys())

    @property
    def possible_rules(self) -> float:
        """Compute the total possible rules based upon number of membership functions on each input variable regardless of output class."""
        return prod(
            len(gaussian.gaussians)
            for label_model in self.feature_models.values()
            for gaussian in label_model.label_models.values()
        )

    @property
    def n_membership_functions(self) -> int:
        return sum(
            len(gaussian.gaussians)
            for label_model in self.feature_models.values()
            for gaussian in label_model.label_models.values()
        )

    @property
    def rule_ids(self) -> list[int]:
        return list(list(self.feature_models.values())[0].label_models.keys())

    @property
    def n_classes(self) -> int:
        return list(self.feature_models.values())[0].ordered_keys[-1] + 1

    @property
    def all_membership_fcns(self) -> list[GaussianMembership]:
        """Gets all membership functions across all features and labels."""
        return [
            g
            for label_model in self.feature_models.values()
            for gaussian in label_model.label_models.values()
            for g in gaussian.gaussians
        ]

    @property
    def all_output_labels(self) -> list[int]:
        return list(set([label for label_model in self.feature_models.values() for label in label_model.ordered_keys]))

    def identify_duplicate_membership_fcns(self) -> list[tuple[str, int, GaussianMembership, GaussianMembership]]:
        duplicates = []
        for feature_name, feature_model in self.feature_models.items():
            for label, label_model in feature_model.label_models.items():
                for i, gaussian in enumerate(label_model.gaussians):
                    for j, other_gaussian in enumerate(label_model.gaussians):
                        if i < j:
                            if _is_close(gaussian, other_gaussian):
                                duplicates.append((feature_name, label, other_gaussian, gaussian))
        return duplicates

    def get_deduplicated_membership_fcns(self) -> dict[GaussianMembership, GaussianMembership]:
        """ Returns a dictionary of [to_replace, with_this] membership functions."""
        to_replace = dict()
        all_mfs = self.all_membership_fcns
        for idx, mf in enumerate(all_mfs):
            for other_mf in all_mfs[idx + 1 :]:
                if _is_close(mf, other_mf):
                    to_replace[other_mf] = mf

        # Recursively apply replacement chains to get final targets
        for key in list(to_replace.keys()):
            current = to_replace[key]
            while current in to_replace:
                current = to_replace[current]
            to_replace[key] = current

        return to_replace

    def remove_duplicate_membership_fcns(self):
        duplicate_mfcns = self.identify_duplicate_membership_fcns()
        for _, (feature_name, label, dup_mf, src_mf) in enumerate(duplicate_mfcns):
            # Replace the dup_mf with the src_mf
            try:
                self.feature_models[feature_name].label_models[label].gaussians.remove(dup_mf)
            except ValueError:
                pass

    def augment(self, other) -> "GaussianMixtureModel":
        """Augment this GaussianMixtureModel with another GaussianMixtureModel, combining feature models."""
        new_feature_models = self.feature_models.copy()
        for feature_name, other_feature_model in other.feature_models.items():
            if feature_name in new_feature_models:
                new_feature_models[feature_name] = new_feature_models[feature_name].augment(other_feature_model)
            else:
                new_feature_models[feature_name] = other_feature_model
        return GaussianMixtureModel(new_feature_models)

    def to_simple_model(self, details: AnomalyParameters | None = None) -> SimpleGaussianClassifierModel:
        dedup_mfs = self.get_deduplicated_membership_fcns()
        rules: list[Rule] = []

        for label in self.all_output_labels:
            antecedent_ids: dict[str, list[uuid.UUID]] = {}
            for feature_name, feature_model in self.feature_models.items():
                label_model = feature_model.label_models.get(label, None)
                if label_model is None:
                    continue
                antecedent_ids[feature_name] = [dedup_mfs.get(mf, mf).id for mf in label_model.gaussians]  # type: ignore[misc]
            rules.append(Rule(antecedents=antecedent_ids, consequent=label))

        # Get the input membership functions from the rules
        required_mf_ids = set([u for r in rules for u_lst in r.antecedents.values() for u in u_lst])
        input_mfs = [mf for mf in self.all_membership_fcns if mf.id in required_mf_ids]
        return SimpleGaussianClassifierModel(
            input_mfs=input_mfs,
            rules=rules,
            anomaly_params=details
        )

def _is_close(g1: GaussianMembership, g2: GaussianMembership, rtol: float = 1e-2, atol: float = 1e-3) -> bool:
    return bool(
        np.isclose(g1.mu, g2.mu, rtol=rtol, atol=atol)
        and np.isclose(g1.sigma, g2.sigma, rtol=rtol, atol=atol)
    )