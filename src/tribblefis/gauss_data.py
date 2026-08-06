import uuid
from math import prod
from typing import NamedTuple, Literal, Optional

import numpy as np

# norm/conorm pairings
#
# Each name selects a FAMILY. A family's t-norm and t-conorm are De Morgan duals
# under the standard negation N(x) = 1 - x, i.e. S(x, y) = 1 - T(1-x, 1-y):
#
#   min/max       T = min(x, y)              S = max(x, y)
#   probability   T = xy                     S = x + y - xy
#   luk           T = max(0, x + y - 1)      S = min(1, x + y)
#   hamacher      T = xy / (x + y - xy)      S = (x + y - 2xy) / (1 - xy)
#   einstein      T = xy / (2 - (x+y-xy))    S = (x + y) / (1 + xy)
#
# Taking both operators from one family is the default and the supported case.
# Mixing families is possible but must be asked for explicitly -- see
# `resolve_norm_pair`.
NormConorm = Literal["min/max", "probability", "luk", "hamacher", "einstein"]
MemberFunction = Literal["gaussian", "triangular", "trap"]
# The default family. `probability` rather than the textbook `min/max` because
# min/max measured as the *worst* of the four De Morgan families on classification
# accuracy -- see docs/norm-family-evaluation.md. Over 18 dataset x split
# combinations, refined:
#
#   min/max      0.7881   (baseline)
#   hamacher     0.8029   +0.0148 +/- 0.0078
#   probability  0.8135   +0.0254 +/- 0.0063   <- default
#   einstein     0.8175   +0.0294 +/- 0.0061
#
# Einstein edges it out, but not separably (the gap between the two is well
# inside their error bars) and it costs two divisions per operation. Probability
# is the cheapest of the three, is the one family whose objective is smooth
# everywhere -- which is what makes an exact analytic gradient possible -- and is
# the most familiar (product / probabilistic sum).
DefaultNormCornorm: NormConorm = "probability"
DefaultMemberFunction: MemberFunction = "gaussian"

NORM_FAMILIES: tuple[NormConorm, ...] = (
    "min/max", "probability", "luk", "hamacher", "einstein",
)


class NormPair(NamedTuple):
    """A resolved (t-norm, t-conorm) selection.

    ``is_de_morgan`` is True when both halves come from the same family, which is
    the only configuration in which De Morgan's laws hold. Code that builds a
    complement out of a conorm -- the anomaly rule is exactly
    ``1 - S(mu_1, ..., mu_k)`` -- depends on that duality for its interpretation,
    so a mixed pair silently changes what such a rule means.
    """

    t_norm: NormConorm
    t_conorm: NormConorm

    @property
    def is_de_morgan(self) -> bool:
        return self.t_norm == self.t_conorm


def resolve_norm_pair(
    norm_conorm: Optional[NormConorm] = None,
    t_norm: Optional[NormConorm] = None,
    t_conorm: Optional[NormConorm] = None,
    allow_mixed_norms: bool = False,
) -> NormPair:
    """Resolve the operator selection into an explicit (t-norm, t-conorm) pair.

    ``norm_conorm`` picks a family for both halves and is the ordinary way to
    configure this. ``t_norm`` / ``t_conorm`` override one half each and are an
    advanced setting: a pair drawn from two different families is not a De Morgan
    dual pair, so it is rejected unless ``allow_mixed_norms=True`` says the caller
    means it. Defaulting to "whatever combination was typed" would let a mismatch
    reach the anomaly rule -- whose complement construction assumes duality --
    without anything being said about it.
    """
    # `is None` rather than a falsy test: None means "not specified", but an empty
    # string is a value, and a wrong one. `or` would quietly swap it for the
    # default and hand back a pair the caller never asked for.
    base: NormConorm = norm_conorm if norm_conorm is not None else DefaultNormCornorm
    resolved = NormPair(
        t_norm=t_norm if t_norm is not None else base,
        t_conorm=t_conorm if t_conorm is not None else base,
    )

    for field, value in (("t_norm", resolved.t_norm), ("t_conorm", resolved.t_conorm)):
        if value not in NORM_FAMILIES:
            raise ValueError(
                f"Invalid {field} {value!r}; expected one of {list(NORM_FAMILIES)}"
            )

    if not resolved.is_de_morgan and not allow_mixed_norms:
        raise ValueError(
            f"t_norm={resolved.t_norm!r} and t_conorm={resolved.t_conorm!r} are from "
            f"different families and so are not De Morgan duals. Pass "
            f"allow_mixed_norms=True to opt in to a mixed pair, or set "
            f"norm_conorm={resolved.t_norm!r} to use one family for both."
        )
    return resolved


class AnomalyParameters(NamedTuple):
    """Parameters for anomaly detection"""

    include_anomaly: bool = True
    threshold: float = 0.5
    label: str = "anomaly"
    norm_conorm: NormConorm = DefaultNormCornorm
    member_function: MemberFunction = DefaultMemberFunction
    # Advanced: override one half of the pair. Leave as None to take both
    # operators from `norm_conorm`, which is the De Morgan-consistent default.
    t_norm: Optional[NormConorm] = None
    t_conorm: Optional[NormConorm] = None
    allow_mixed_norms: bool = False

    def norms(self) -> NormPair:
        """The resolved (t-norm, t-conorm) pair for these parameters."""
        return resolve_norm_pair(
            self.norm_conorm, self.t_norm, self.t_conorm, self.allow_mixed_norms
        )


class GaussianMembership(NamedTuple):
    """A single Gaussian membership function."""
    mu: float
    sigma: float
    id: Optional[uuid.UUID] = None

    @staticmethod
    def create(mu: float, sigma: float) -> "GaussianMembership":
        return GaussianMembership(mu=mu, sigma=sigma, id=uuid.uuid4())

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Gaussian membership function at given points."""
        x = np.asarray(x, dtype=float)
        sigma = max(self.sigma, 1e-6)
        return np.exp(-0.5 * ((x - self.mu) / sigma) ** 2)


class TrapezoidMembership(NamedTuple):
    """A single trapezoidal membership function with parameters a <= b <= c <= d."""
    a: float
    b: float
    c: float
    d: float
    id: Optional[uuid.UUID] = None

    @staticmethod
    def create(a: float, b: float, c: float, d: float) -> "TrapezoidMembership":
        return TrapezoidMembership(a=a, b=b, c=c, d=d, id=uuid.uuid4())

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate trapezoidal membership function at given points.

        Membership is 0 outside [a,d], rises linearly from 0 to 1 over [a,b],
        remains at 1 over [b,c], and falls linearly from 1 to 0 over [c,d].
        """
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x, dtype=float)

        # Rising slope [a, b]
        ab_width = self.b - self.a
        if ab_width > 0:
            mask = (x > self.a) & (x < self.b)
            y[mask] = (x[mask] - self.a) / ab_width

        # Flat top [b, c]
        y[(x >= self.b) & (x <= self.c)] = 1.0

        # Falling slope [c, d]
        cd_width = self.d - self.c
        if cd_width > 0:
            mask = (x > self.c) & (x < self.d)
            y[mask] = (self.d - x[mask]) / cd_width

        return y


class TriangularMembership(NamedTuple):
    """A triangular membership function with apex at ``b``, rising from ``a`` and
    falling to ``c`` (``a <= b <= c``).

    Shoulders are expressed with infinities: ``a = -inf`` makes a *left shoulder*
    (membership is 1 for every ``x <= b``) and ``c = +inf`` makes a *right
    shoulder* (membership is 1 for every ``x >= b``). Triangular terms placed on a
    shared set of apex knots -- with a left shoulder on the first term and a right
    shoulder on the last -- form a **Ruspini partition**: their memberships sum to
    exactly 1 at every point of the axis. See :mod:`tribblefis.ruspini`.
    """

    a: float
    b: float
    c: float
    id: Optional[uuid.UUID] = None

    @staticmethod
    def create(a: float, b: float, c: float) -> "TriangularMembership":
        return TriangularMembership(a=a, b=b, c=c, id=uuid.uuid4())

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the triangular membership at ``x`` (handles ``+/-inf`` shoulders)."""
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x, dtype=float)

        # Rising side / left shoulder: x <= b
        left = x <= self.b
        if np.isneginf(self.a):
            y[left] = 1.0
        else:
            width = self.b - self.a
            if width > 0:
                m = left & (x > self.a)
                y[m] = (x[m] - self.a) / width
            # x <= a stays 0

        # Falling side / right shoulder: x > b
        right = x > self.b
        if np.isposinf(self.c):
            y[right] = 1.0
        else:
            width = self.c - self.b
            if width > 0:
                m = right & (x < self.c)
                y[m] = (self.c - x[m]) / width
            # x >= c stays 0

        return y


AnyMembership = GaussianMembership | TrapezoidMembership | TriangularMembership


class IT2GaussianMembership(NamedTuple):
    """An interval type-2 Gaussian membership function.

    Consists of an upper membership function (UMF) and lower membership function (LMF),
    both Gaussians. The region between them is the footprint of uncertainty (FoU).
    """
    upper_mf: GaussianMembership
    lower_mf: GaussianMembership
    id: Optional[uuid.UUID] = None

    @staticmethod
    def create(
        upper_mu: float, upper_sigma: float,
        lower_mu: float, lower_sigma: float
    ) -> "IT2GaussianMembership":
        return IT2GaussianMembership(
            upper_mf=GaussianMembership(mu=upper_mu, sigma=upper_sigma),
            lower_mf=GaussianMembership(mu=lower_mu, sigma=lower_sigma),
            id=uuid.uuid4()
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate both upper and lower MFs at given points.

        Returns:
            (y_upper, y_lower) - both are numpy arrays of same shape as x
        """
        y_upper = self.upper_mf.evaluate(x)
        y_lower = self.lower_mf.evaluate(x)
        return y_upper, y_lower


class Rule(NamedTuple):
    """A fuzzy rule mapping input membership functions to an output label."""

    antecedents: dict[str, list[uuid.UUID]]  # maps feature name to ids of GaussianMembership
    consequent: int | str  # output label


class SimpleGaussianClassifierModel(NamedTuple):
    """A simple classifier model with explicit rules (supports any membership function type)."""

    input_mfs: list[AnyMembership]
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

    def get_mfs(self, ids: list[uuid.UUID]) -> list[AnyMembership]:
        return [mf for mf in self.input_mfs if mf.id in ids]

    def get_mfs_for_feature(self, feature_name: str) -> list[AnyMembership]:
        return list(set([mf for rule in self.rules for mf in self.get_mfs(rule.antecedents.get(feature_name, []))]))


class LabelModel(NamedTuple):
    """A collection of membership functions for a specific output class label."""

    memberships: list[AnyMembership]

    def augment(self, other_label_model) -> "LabelModel":
        """Augment this LabelModel with another LabelModel, combining membership functions."""
        new_memberships = self.memberships.copy()
        new_memberships.extend(other_label_model.memberships)
        return LabelModel(new_memberships)


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
            len(label_model.memberships)
            for feature_model in self.feature_models.values()
            for label_model in feature_model.label_models.values()
        )

    @property
    def n_membership_functions(self) -> int:
        return sum(
            len(label_model.memberships)
            for feature_model in self.feature_models.values()
            for label_model in feature_model.label_models.values()
        )

    @property
    def rule_ids(self) -> list[int]:
        return list(list(self.feature_models.values())[0].label_models.keys())

    @property
    def n_classes(self) -> int:
        return list(self.feature_models.values())[0].ordered_keys[-1] + 1

    @property
    def all_membership_fcns(self) -> list[AnyMembership]:
        """Gets all membership functions across all features and labels."""
        return [
            g
            for feature_model in self.feature_models.values()
            for label_model in feature_model.label_models.values()
            for g in label_model.memberships
        ]

    @property
    def all_output_labels(self) -> list[int]:
        return list(set([label for label_model in self.feature_models.values() for label in label_model.ordered_keys]))

    def identify_duplicate_membership_fcns(self) -> list[tuple[str, int, AnyMembership, AnyMembership]]:
        duplicates = []
        for feature_name, feature_model in self.feature_models.items():
            for label, label_model in feature_model.label_models.items():
                for i, mf in enumerate(label_model.memberships):
                    for j, other_mf in enumerate(label_model.memberships):
                        if i < j:
                            if _is_close(mf, other_mf):
                                duplicates.append((feature_name, label, other_mf, mf))
        return duplicates

    def get_deduplicated_membership_fcns(self) -> dict[AnyMembership, AnyMembership]:
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
                self.feature_models[feature_name].label_models[label].memberships.remove(dup_mf)
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
                antecedent_ids[feature_name] = [dedup_mfs.get(mf, mf).id for mf in label_model.memberships]  # type: ignore[misc]
            rules.append(Rule(antecedents=antecedent_ids, consequent=label))

        # Get the input membership functions from the rules
        required_mf_ids = set([u for r in rules for u_lst in r.antecedents.values() for u in u_lst])
        input_mfs = [mf for mf in self.all_membership_fcns if mf.id in required_mf_ids]
        return SimpleGaussianClassifierModel(
            input_mfs=input_mfs,
            rules=rules,
            anomaly_params=details
        )

def _is_close(g1: AnyMembership, g2: AnyMembership, rtol: float = 1e-2, atol: float = 1e-3) -> bool:
    """Check if two membership functions are numerically close.

    Only returns True if both objects are the same type and their parameters match.
    """
    if type(g1) != type(g2):
        return False

    if isinstance(g1, GaussianMembership):
        return bool(
            np.isclose(g1.mu, g2.mu, rtol=rtol, atol=atol)
            and np.isclose(g1.sigma, g2.sigma, rtol=rtol, atol=atol)
        )
    elif isinstance(g1, TrapezoidMembership):
        return bool(
            np.allclose(
                [g1.a, g1.b, g1.c, g1.d],
                [g2.a, g2.b, g2.c, g2.d],
                rtol=rtol,
                atol=atol
            )
        )
    return False


# Interval Type-2 FIS Data Structures
class IT2LabelModel(NamedTuple):
    """A collection of IT2 membership functions for a specific output class label."""

    memberships: list[IT2GaussianMembership]

    def augment(self, other_label_model: "IT2LabelModel") -> "IT2LabelModel":
        """Augment this IT2LabelModel with another, combining membership functions."""
        new_memberships = self.memberships.copy()
        new_memberships.extend(other_label_model.memberships)
        return IT2LabelModel(new_memberships)


class IT2FeatureModel(NamedTuple):
    """A collection of IT2LabelModels for a specific feature."""

    label_models: dict[int, IT2LabelModel]

    @property
    def ordered_keys(self) -> list[int]:
        return list(sorted(self.label_models.keys()))

    def augment(self, other_feature_model: "IT2FeatureModel") -> "IT2FeatureModel":
        """Augment this IT2FeatureModel with another."""
        new_label_models = self.label_models.copy()
        for label, other_label_model in other_feature_model.label_models.items():
            if label in new_label_models:
                new_label_models[label] = new_label_models[label].augment(other_label_model)
            else:
                new_label_models[label] = other_label_model
        return IT2FeatureModel(new_label_models)


class IT2GaussianMixtureModel(NamedTuple):
    """An interval type-2 Gaussian mixture model with IT2 memberships."""

    feature_models: dict[str, IT2FeatureModel]

    @property
    def n_rules(self) -> int:
        return len(list(self.feature_models.values())[0].label_models.keys())

    @property
    def n_features(self) -> int:
        return len(self.feature_models)

    @property
    def all_membership_fcns(self) -> list[IT2GaussianMembership]:
        """Get all IT2 membership functions across all features and labels."""
        return [
            mf
            for feature_model in self.feature_models.values()
            for label_model in feature_model.label_models.values()
            for mf in label_model.memberships
        ]

    @property
    def all_output_labels(self) -> list[int]:
        return list(set([label for fm in self.feature_models.values() for label in fm.ordered_keys]))

    @property
    def n_classes(self) -> int:
        if not self.feature_models:
            return 0
        return list(self.feature_models.values())[0].ordered_keys[-1] + 1