import itertools
import uuid
from math import isinf, prod
from typing import NamedTuple, Literal, Optional

import numpy as np

# Numeric thresholds for numerical stability
_SIGMA_FLOOR = 1e-6  # Minimum variance/sigma to avoid numerical issues

# Norm/conorm families: each name selects a De Morgan dual pair T(x,y), S(x,y).
# Taking both from one family (default). Mixing requires explicit opt-in (resolve_norm_pair).
#   min/max       T = min, S = max
#   probability   T = xy, S = x+y-xy (product/sum)
#   luk           T = max(0,x+y-1), S = min(1,x+y)
#   hamacher      T = xy/(x+y-xy), S = (x+y-2xy)/(1-xy)
#   einstein      T = xy/(2-(x+y-xy)), S = (x+y)/(1+xy)
NormConorm = Literal["min/max", "probability", "luk", "hamacher", "einstein"]
MemberFunction = Literal["gaussian", "triangular", "trap"]
# Default: probability (not min/max) for smooth gradients; see norm-family-evaluation.md.
DefaultNormCornorm: NormConorm = "probability"
DefaultMemberFunction: MemberFunction = "gaussian"

# Conservative MF deduplication tolerances; issue #85, see _is_close.
DEFAULT_DEDUP_RTOL = 1e-2
DEFAULT_DEDUP_ATOL = 1e-3

# Below this total firing strength, a row is treated as "no rule meaningfully
# covers this point" and predictions fall back to a fixed default (0) rather
# than trusting a firing-weighted average of near-noise-level weights (see
# `regression._normalize_firing_strengths`'s docstring for the extrapolation
# rationale). Every zero-firing gate in the package -- Type-1's own
# normalization, the TSK consequent solver, and IT2/GT2's Karnik-Mendel search
# -- must share this single value: two different thresholds for "no rule
# fires" is what silently broke the "IT2/GT2 converges to Type-1 as the
# footprint of uncertainty vanishes" invariant: `karnik_mendel_tsk` used to
# gate at 1e-9 of its own, three decades stricter than this one, so a row deep
# in that gap got a real Karnik-Mendel answer while Type-1 returned its 0
# fallback for the exact same point (found while investigating a GT2 regressor
# RMSE gap that turned out to reproduce on plain IT2 too).
ZERO_FIRING_THRESHOLD = 1e-6

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
        sigma = max(self.sigma, _SIGMA_FLOOR)
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


def widen_membership(
    mf: AnyMembership, uncertainty_width: float, min_val: float = 1e-4,
) -> tuple[AnyMembership, AnyMembership]:
    """``(upper, lower)``: footprint-of-uncertainty widened/narrowed versions
    of ``mf``, holding its peak fixed and scaling its spread by
    ``(1 + uncertainty_width)`` (upper, wider, more permissive) /
    ``max(0.1, 1 - uncertainty_width)`` (lower, narrower, more restrictive) --
    the one transform every IT2/GT2 conversion (`it2_classifier.py`,
    `it2_regressor.py`, `gt2_classifier.py`, `gt2_regressor.py`) applies,
    now shared instead of duplicated once per membership type per file.

    "Spread" is type-specific: Gaussian's ``sigma``; trapezoid's two slope
    half-widths, ``b - a`` and ``d - c``, scaled independently so the flat top
    ``[b, c]`` is untouched (the trapezoid analogue of holding ``mu`` fixed);
    triangular's two leg half-widths, ``b - a`` and ``c - b``, scaled
    independently around the fixed apex ``b``. Trapezoid/triangular shoulders
    (``a = -inf`` / ``c = +inf``, see `TriangularMembership`'s Ruspini-
    partition use) have no finite spread to scale and are not supported here --
    conversion only ever sees memberships from `create_gaussian_membership_dict`/
    `create_trapz_membership_dict[_fast]`, which never produce one.
    """
    w = uncertainty_width
    if isinstance(mf, GaussianMembership):
        base_sigma = max(mf.sigma, min_val)
        return (
            GaussianMembership(mu=mf.mu, sigma=base_sigma * (1.0 + w), id=mf.id),
            GaussianMembership(mu=mf.mu, sigma=base_sigma * max(0.1, 1.0 - w), id=mf.id),
        )
    if isinstance(mf, TrapezoidMembership):
        if not (np.isfinite(mf.a) and np.isfinite(mf.d)):
            raise ValueError("widen_membership does not support trapezoid shoulders (a=-inf/d=+inf)")
        left = max(mf.b - mf.a, min_val)
        right = max(mf.d - mf.c, min_val)
        left_wide, left_narrow = left * (1.0 + w), left * max(0.1, 1.0 - w)
        right_wide, right_narrow = right * (1.0 + w), right * max(0.1, 1.0 - w)
        return (
            TrapezoidMembership(a=mf.b - left_wide, b=mf.b, c=mf.c, d=mf.c + right_wide, id=mf.id),
            TrapezoidMembership(a=mf.b - left_narrow, b=mf.b, c=mf.c, d=mf.c + right_narrow, id=mf.id),
        )
    if isinstance(mf, TriangularMembership):
        if not (np.isfinite(mf.a) and np.isfinite(mf.c)):
            raise ValueError("widen_membership does not support triangular shoulders (a=-inf/c=+inf)")
        left = max(mf.b - mf.a, min_val)
        right = max(mf.c - mf.b, min_val)
        left_wide, left_narrow = left * (1.0 + w), left * max(0.1, 1.0 - w)
        right_wide, right_narrow = right * (1.0 + w), right * max(0.1, 1.0 - w)
        return (
            TriangularMembership(a=mf.b - left_wide, b=mf.b, c=mf.b + right_wide, id=mf.id),
            TriangularMembership(a=mf.b - left_narrow, b=mf.b, c=mf.b + right_narrow, id=mf.id),
        )
    raise TypeError(f"Unsupported membership type for widen_membership: {type(mf)!r}")


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


class IT2TrapezoidMembership(NamedTuple):
    """An interval type-2 trapezoidal membership function.

    Consists of upper and lower trapezoidal membership functions.
    """
    upper_mf: TrapezoidMembership
    lower_mf: TrapezoidMembership
    id: Optional[uuid.UUID] = None

    @staticmethod
    def create(
        upper_a: float, upper_b: float, upper_c: float, upper_d: float,
        lower_a: float, lower_b: float, lower_c: float, lower_d: float,
    ) -> "IT2TrapezoidMembership":
        return IT2TrapezoidMembership(
            upper_mf=TrapezoidMembership(a=upper_a, b=upper_b, c=upper_c, d=upper_d),
            lower_mf=TrapezoidMembership(a=lower_a, b=lower_b, c=lower_c, d=lower_d),
            id=uuid.uuid4()
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate both upper and lower MFs at given points."""
        y_upper = self.upper_mf.evaluate(x)
        y_lower = self.lower_mf.evaluate(x)
        return y_upper, y_lower


class IT2TriangularMembership(NamedTuple):
    """An interval type-2 triangular membership function.

    Consists of upper and lower triangular membership functions.
    """
    upper_mf: TriangularMembership
    lower_mf: TriangularMembership
    id: Optional[uuid.UUID] = None

    @staticmethod
    def create(
        upper_a: float, upper_b: float, upper_c: float,
        lower_a: float, lower_b: float, lower_c: float,
    ) -> "IT2TriangularMembership":
        return IT2TriangularMembership(
            upper_mf=TriangularMembership(a=upper_a, b=upper_b, c=upper_c),
            lower_mf=TriangularMembership(a=lower_a, b=lower_b, c=lower_c),
            id=uuid.uuid4()
        )

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate both upper and lower MFs at given points."""
        y_upper = self.upper_mf.evaluate(x)
        y_lower = self.lower_mf.evaluate(x)
        return y_upper, y_lower


IT2AnyMembership = IT2GaussianMembership | IT2TrapezoidMembership | IT2TriangularMembership

_IT2_MEMBERSHIP_BY_TYPE1: dict[type, type] = {
    GaussianMembership: IT2GaussianMembership,
    TrapezoidMembership: IT2TrapezoidMembership,
    TriangularMembership: IT2TriangularMembership,
}


def to_it2_membership(
    upper_mf: AnyMembership, lower_mf: AnyMembership, id: Optional[uuid.UUID] = None,
) -> IT2AnyMembership:
    """Wrap a ``(upper_mf, lower_mf)`` pair (e.g. from `widen_membership`) in
    the matching `IT2AnyMembership` container for their shared Type-1 type."""
    it2_cls = _IT2_MEMBERSHIP_BY_TYPE1.get(type(upper_mf))
    if it2_cls is None:
        raise TypeError(f"Unsupported membership type for to_it2_membership: {type(upper_mf)!r}")
    return it2_cls(upper_mf=upper_mf, lower_mf=lower_mf, id=id)


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

    def identify_duplicate_membership_fcns(
        self, rtol: float = DEFAULT_DEDUP_RTOL, atol: float = DEFAULT_DEDUP_ATOL
    ) -> list[tuple[str, int, AnyMembership, AnyMembership]]:
        """Find near-duplicate membership functions within each (feature, label).

        Two memberships in the same (feature, label) list only ever feed the same
        conorm fold, so a duplicate found here can be dropped without changing any
        prediction (exactly, at ``rtol=atol=0``; within measurement noise at looser
        tolerances -- see issue #85).

        Args:
            rtol, atol: Passed through to `_is_close`. The defaults match the
                historical hardcoded tolerance; pass looser values to trade more
                reduction for more (noise-level, per the issue #85 measurement)
                risk.
        """
        duplicates = []
        for feature_name, feature_model in self.feature_models.items():
            for label, label_model in feature_model.label_models.items():
                for i, mf in enumerate(label_model.memberships):
                    for j, other_mf in enumerate(label_model.memberships):
                        if i < j:
                            if _is_close(mf, other_mf, rtol=rtol, atol=atol):
                                duplicates.append((feature_name, label, other_mf, mf))
        return duplicates

    def get_deduplicated_membership_fcns(
        self, rtol: float = DEFAULT_DEDUP_RTOL, atol: float = DEFAULT_DEDUP_ATOL
    ) -> dict[AnyMembership, AnyMembership]:
        """Returns a dictionary of [to_replace, with_this] membership functions.

        Unlike `identify_duplicate_membership_fcns`, this compares *every*
        membership function against every other, regardless of feature or label --
        it is meant for `to_simple_model`'s flat, explicit-rule representation,
        where a duplicate found across labels is still a real duplicate id to
        collapse.

        Args:
            rtol, atol: Passed through to `_is_close`.
        """
        to_replace = dict()
        all_mfs = self.all_membership_fcns
        for idx, mf in enumerate(all_mfs):
            for other_mf in all_mfs[idx + 1 :]:
                if _is_close(mf, other_mf, rtol=rtol, atol=atol):
                    to_replace[other_mf] = mf

        # Recursively apply replacement chains to get final targets
        for key in list(to_replace.keys()):
            current = to_replace[key]
            while current in to_replace:
                current = to_replace[current]
            to_replace[key] = current

        return to_replace

    def remove_duplicate_membership_fcns(
        self, rtol: float = DEFAULT_DEDUP_RTOL, atol: float = DEFAULT_DEDUP_ATOL
    ) -> int:
        """Remove near-duplicate membership functions in place.

        Args:
            rtol, atol: Passed through to `identify_duplicate_membership_fcns`.

        Returns:
            The number of membership functions actually removed.
        """
        duplicate_mfcns = self.identify_duplicate_membership_fcns(rtol=rtol, atol=atol)
        removed = 0
        for _, (feature_name, label, dup_mf, src_mf) in enumerate(duplicate_mfcns):
            # Replace the dup_mf with the src_mf
            try:
                self.feature_models[feature_name].label_models[label].memberships.remove(dup_mf)
                removed += 1
            except ValueError:
                pass
        return removed

    def augment(self, other) -> "GaussianMixtureModel":
        """Augment this GaussianMixtureModel with another GaussianMixtureModel, combining feature models."""
        new_feature_models = self.feature_models.copy()
        for feature_name, other_feature_model in other.feature_models.items():
            if feature_name in new_feature_models:
                new_feature_models[feature_name] = new_feature_models[feature_name].augment(other_feature_model)
            else:
                new_feature_models[feature_name] = other_feature_model
        return GaussianMixtureModel(new_feature_models)

    def to_simple_model(
        self,
        details: AnomalyParameters | None = None,
        convex_clauses_only: bool = False,
        rtol: float = DEFAULT_DEDUP_RTOL,
        atol: float = DEFAULT_DEDUP_ATOL,
    ) -> SimpleGaussianClassifierModel:
        """Materialise this (implicit) mixture into an explicit rule base.

        When ``convex_clauses_only`` is set, any feature clause whose OR'd
        membership functions cover disjoint (non-touching) intervals of that
        feature's axis is split into one rule per convex sub-clause, via
        :func:`split_convex_clauses`. This runs strictly after
        deduplication (``dedup_mfs`` below), so it always operates on
        already-deduplicated membership ids.

        Args:
            details: Anomaly parameters carried onto the resulting
                `SimpleGaussianClassifierModel`, if any.
            rtol, atol: Passed through to `get_deduplicated_membership_fcns` --
                see that method and issue #85 for the reduction-vs-risk tradeoff
                these control.
        """
        dedup_mfs = self.get_deduplicated_membership_fcns(rtol=rtol, atol=atol)
        rules: list[Rule] = []

        mf_lookup: dict[uuid.UUID, AnyMembership] = {}
        if convex_clauses_only:
            for mf in self.all_membership_fcns:
                resolved = dedup_mfs.get(mf, mf)
                mf_lookup[resolved.id] = resolved  # type: ignore[misc]

        for label in self.all_output_labels:
            antecedent_ids: dict[str, list[uuid.UUID]] = {}
            for feature_name, feature_model in self.feature_models.items():
                label_model = feature_model.label_models.get(label, None)
                if label_model is None:
                    continue
                antecedent_ids[feature_name] = [dedup_mfs.get(mf, mf).id for mf in label_model.memberships]  # type: ignore[misc]
            if convex_clauses_only:
                for combo in split_convex_clauses(antecedent_ids, mf_lookup):
                    rules.append(Rule(antecedents=combo, consequent=label))
            else:
                rules.append(Rule(antecedents=antecedent_ids, consequent=label))

        # Get the input membership functions from the rules
        required_mf_ids = set([u for r in rules for u_lst in r.antecedents.values() for u in u_lst])
        input_mfs = [mf for mf in self.all_membership_fcns if mf.id in required_mf_ids]
        return SimpleGaussianClassifierModel(
            input_mfs=input_mfs,
            rules=rules,
            anomaly_params=details
        )

def _close_scalar(a: float, b: float, rtol: float, atol: float) -> bool:
    """``np.isclose`` for two Python floats, without building arrays to do it.

    Semantics match numpy's default (``equal_nan=False``) exactly, including the
    cases the bare ``|a - b| <= atol + rtol * |b|`` formula gets wrong: NaN is
    close to nothing including itself, same-signed infinities are close, and an
    infinity is not close to a finite value (the formula would evaluate
    ``inf <= inf`` and say True).

    This exists because `_is_close` is called from an O(n^2) dedup scan and
    dominates `MembershipDict.to_simple_model`. On one RT-IOT2022 fold that is
    2.9M calls; ``np.isclose`` on two scalars allocates arrays, enters an
    errstate context and runs two ufunc reductions, at 8.06 us against 0.12 us
    here -- 23.4 s of a 55.8 s fold, against 0.3 s.
    """
    if a == b:  # exact equality, and the same-signed-infinity case
        return True
    if a != a or b != b:  # NaN is close to nothing, itself included
        return False
    if isinf(a) or isinf(b):  # opposite infinities, or inf vs finite
        return False
    return bool(abs(a - b) <= atol + rtol * abs(b))


def _is_close(
    g1: AnyMembership, g2: AnyMembership, rtol: float = DEFAULT_DEDUP_RTOL, atol: float = DEFAULT_DEDUP_ATOL
) -> bool:
    """Check if two membership functions are numerically close.

    Only returns True if both objects are the same type and their parameters match.
    """
    if type(g1) != type(g2):
        return False

    if isinstance(g1, GaussianMembership):
        return _close_scalar(g1.mu, g2.mu, rtol, atol) and _close_scalar(g1.sigma, g2.sigma, rtol, atol)
    elif isinstance(g1, TrapezoidMembership):
        return all(
            _close_scalar(x, y, rtol, atol)
            for x, y in zip((g1.a, g1.b, g1.c, g1.d), (g2.a, g2.b, g2.c, g2.d))
        )
    elif isinstance(g1, TriangularMembership):
        return all(
            _close_scalar(x, y, rtol, atol)
            for x, y in zip((g1.a, g1.b, g1.c), (g2.a, g2.b, g2.c))
        )
    return False


def mf_interval(mf: AnyMembership, gaussian_k: float = 3.0) -> tuple[float, float]:
    """The "effective support" of a membership function, as a ``(low, high)`` interval.

    Exact for :class:`TriangularMembership` (``a, c``) and
    :class:`TrapezoidMembership` (``a, d``). For :class:`GaussianMembership`
    -- which has infinite support -- this uses a symmetric
    ``mu +/- gaussian_k * sigma`` cutoff: a conservative "the tails are
    negligible" convention for deciding whether two clauses' regions are
    disjoint, independent of (and answering a different question from) the
    MAE-optimal triangle-fit width in :mod:`tribblefis.triangle_fit`.
    """
    if isinstance(mf, TriangularMembership):
        return mf.a, mf.c
    elif isinstance(mf, TrapezoidMembership):
        return mf.a, mf.d
    elif isinstance(mf, GaussianMembership):
        return mf.mu - gaussian_k * mf.sigma, mf.mu + gaussian_k * mf.sigma
    else:
        raise TypeError(f"mf_interval does not support membership type {type(mf)!r}")


def split_convex_clauses(
    antecedents: dict[str, list[uuid.UUID]],
    mf_lookup: dict[uuid.UUID, AnyMembership],
    gaussian_k: float = 3.0,
) -> list[dict[str, list[uuid.UUID]]]:
    """Expand a rule's antecedents into convex (contiguous-interval) sub-clauses.

    Per feature, every OR'd membership id's effective interval
    (:func:`mf_interval`) is sorted by lower bound and greedily merged
    wherever consecutive intervals overlap or touch. A feature that already
    reduces to a single merged interval is left as a one-element list, so a
    rule with no disjoint clauses at all comes back as its own sole
    combination, unchanged. When more than one feature has a disjoint
    clause, the Cartesian product of every feature's convex groups is
    returned, so "AND across features, OR (convexly) within a feature"
    semantics are preserved exactly -- just spread across more rules.
    """

    def merged_groups(ids: list[uuid.UUID]) -> list[list[uuid.UUID]]:
        if not ids:
            # Preserve today's behaviour for a feature with no memberships:
            # one combination, carrying the empty list through unchanged.
            return [[]]
        scored = sorted(ids, key=lambda i: mf_interval(mf_lookup[i], gaussian_k)[0])
        groups: list[list[uuid.UUID]] = []
        current: list[uuid.UUID] = []
        current_hi: float | None = None
        for mf_id in scored:
            lo, hi = mf_interval(mf_lookup[mf_id], gaussian_k)
            if current_hi is None or lo <= current_hi:
                current.append(mf_id)
                current_hi = hi if current_hi is None else max(current_hi, hi)
            else:
                groups.append(current)
                current = [mf_id]
                current_hi = hi
        if current:
            groups.append(current)
        return groups

    features = list(antecedents.keys())
    per_feature_groups = [merged_groups(antecedents[f]) for f in features]
    return [
        dict(zip(features, combo))
        for combo in itertools.product(*per_feature_groups)
    ]


# Interval Type-2 FIS Data Structures
class IT2LabelModel(NamedTuple):
    """A collection of IT2 membership functions for a specific output class label."""

    memberships: list[IT2AnyMembership]

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


# General Type-2 (GT2) FIS Data Structures -- alpha-plane representation
# (Mendel, Liu 2008; see docs/gt2-evaluation.md for the survey this implements).
class GT2GaussianMembership(NamedTuple):
    """A general type-2 Gaussian membership function via the alpha-plane
    representation.

    Extends `IT2GaussianMembership`'s ``(upper_mf, lower_mf)`` footprint of
    uncertainty with one more Gaussian, ``principal_mf``: the single
    most-likely membership function within that footprint, whose ``sigma``
    must lie in ``[lower_mf.sigma, upper_mf.sigma]``. The secondary
    membership grade at each primary point is modeled as triangular over
    sigma, apex at ``principal_mf.sigma``, base spanning ``[lower_mf.sigma,
    upper_mf.sigma]`` -- the simplest closed-form secondary-membership shape
    consistent with the KISS design philosophy in ``IT2_GUIDE.md``, and the
    one ``alpha_cut`` assumes.

    ``mu`` is shared across all three Gaussians, mirroring
    `IT2GaussianMembership`'s own invariant (see
    ``it2_refine._iter_it2_gaussian_slots``'s docstring for why a shared
    peak matters): a secondary membership that also varied ``mu`` would need
    a 2-D alpha-cut instead of the 1-D sigma interval this module's kernel
    assumes.
    """

    upper_mf: GaussianMembership
    lower_mf: GaussianMembership
    principal_mf: GaussianMembership
    id: Optional[uuid.UUID] = None

    @staticmethod
    def create(
        upper_mu: float, upper_sigma: float,
        lower_mu: float, lower_sigma: float,
        principal_sigma: float | None = None,
    ) -> "GT2GaussianMembership":
        """``principal_sigma`` defaults to the midpoint of ``[lower_sigma,
        upper_sigma]`` when omitted -- a neutral starting point equivalent to
        assuming a *uniform* secondary grade, i.e. today's IT2 midpoint
        reduction, until a real principal value is known."""
        if principal_sigma is None:
            principal_sigma = 0.5 * (lower_sigma + upper_sigma)
        return GT2GaussianMembership(
            upper_mf=GaussianMembership(mu=upper_mu, sigma=upper_sigma),
            lower_mf=GaussianMembership(mu=lower_mu, sigma=lower_sigma),
            principal_mf=GaussianMembership(mu=upper_mu, sigma=principal_sigma),
            id=uuid.uuid4(),
        )

    def alpha_cut(self, alpha: float) -> IT2GaussianMembership:
        """The IT2-shaped alpha-plane at level ``alpha`` in ``[0, 1]``.

        Linear interpolation from each side of the footprint toward the
        principal value -- the alpha-cut of a triangular secondary grade.
        ``alpha=0`` returns exactly today's IT2 footprint (``[lower_mf.sigma,
        upper_mf.sigma]``); ``alpha=1`` collapses both bounds onto
        ``principal_mf.sigma`` (``upper_mf == lower_mf == principal_mf``).
        """
        sigma_lo = self.lower_mf.sigma + alpha * (self.principal_mf.sigma - self.lower_mf.sigma)
        sigma_hi = self.upper_mf.sigma - alpha * (self.upper_mf.sigma - self.principal_mf.sigma)
        mu = self.principal_mf.mu
        return IT2GaussianMembership(
            upper_mf=GaussianMembership(mu=mu, sigma=sigma_hi, id=self.upper_mf.id),
            lower_mf=GaussianMembership(mu=mu, sigma=sigma_lo, id=self.lower_mf.id),
            id=self.id,
        )


class GT2TrapezoidMembership(NamedTuple):
    """A general type-2 trapezoidal membership function via the alpha-plane
    representation -- the trapezoidal analogue of `GT2GaussianMembership`
    (see its docstring for the general alpha-plane/secondary-grade design).

    The flat top ``[b, c]`` is shared across all three trapezoids (mirroring
    `GT2GaussianMembership`'s shared ``mu``); the two outer slopes -- left
    half-width ``b - a`` and right half-width ``d - c`` -- each carry their
    own independent triangular secondary grade over ``[lower, upper]``, apex
    at ``principal``.
    """

    upper_mf: TrapezoidMembership
    lower_mf: TrapezoidMembership
    principal_mf: TrapezoidMembership
    id: Optional[uuid.UUID] = None

    @staticmethod
    def create(
        b: float, c: float,
        upper_a: float, upper_d: float,
        lower_a: float, lower_d: float,
        principal_a: float | None = None,
        principal_d: float | None = None,
    ) -> "GT2TrapezoidMembership":
        """``principal_a``/``principal_d`` default to the midpoint of their
        ``[lower, upper]`` range when omitted -- see `GT2GaussianMembership.create`."""
        if principal_a is None:
            principal_a = 0.5 * (lower_a + upper_a)
        if principal_d is None:
            principal_d = 0.5 * (lower_d + upper_d)
        return GT2TrapezoidMembership(
            upper_mf=TrapezoidMembership(a=upper_a, b=b, c=c, d=upper_d),
            lower_mf=TrapezoidMembership(a=lower_a, b=b, c=c, d=lower_d),
            principal_mf=TrapezoidMembership(a=principal_a, b=b, c=c, d=principal_d),
            id=uuid.uuid4(),
        )

    def alpha_cut(self, alpha: float) -> IT2TrapezoidMembership:
        """The IT2-shaped alpha-plane at level ``alpha`` in ``[0, 1]`` -- see
        `GT2GaussianMembership.alpha_cut`; here each outer edge (``a``, ``d``)
        is interpolated toward its own principal value independently."""
        a_upper = self.upper_mf.a - alpha * (self.upper_mf.a - self.principal_mf.a)
        a_lower = self.lower_mf.a + alpha * (self.principal_mf.a - self.lower_mf.a)
        d_upper = self.upper_mf.d - alpha * (self.upper_mf.d - self.principal_mf.d)
        d_lower = self.lower_mf.d + alpha * (self.principal_mf.d - self.lower_mf.d)
        b, c = self.principal_mf.b, self.principal_mf.c
        return IT2TrapezoidMembership(
            upper_mf=TrapezoidMembership(a=a_upper, b=b, c=c, d=d_upper, id=self.upper_mf.id),
            lower_mf=TrapezoidMembership(a=a_lower, b=b, c=c, d=d_lower, id=self.lower_mf.id),
            id=self.id,
        )


class GT2TriangularMembership(NamedTuple):
    """A general type-2 triangular membership function via the alpha-plane
    representation -- the triangular analogue of `GT2GaussianMembership`
    (see its docstring for the general alpha-plane/secondary-grade design).

    The apex ``b`` is shared across all three triangles (mirroring
    `GT2GaussianMembership`'s shared ``mu``); the two legs -- left half-width
    ``b - a`` and right half-width ``c - b`` -- each carry their own
    independent triangular secondary grade over ``[lower, upper]``, apex at
    ``principal``. Shoulder legs (``a = -inf`` / ``c = +inf``, see
    `TriangularMembership`) have no finite spread to cut and are passed
    through unchanged by `alpha_cut`.
    """

    upper_mf: TriangularMembership
    lower_mf: TriangularMembership
    principal_mf: TriangularMembership
    id: Optional[uuid.UUID] = None

    @staticmethod
    def create(
        b: float,
        upper_a: float, upper_c: float,
        lower_a: float, lower_c: float,
        principal_a: float | None = None,
        principal_c: float | None = None,
    ) -> "GT2TriangularMembership":
        """``principal_a``/``principal_c`` default to the midpoint of their
        ``[lower, upper]`` range when omitted -- see `GT2GaussianMembership.create`."""
        if principal_a is None:
            principal_a = 0.5 * (lower_a + upper_a)
        if principal_c is None:
            principal_c = 0.5 * (lower_c + upper_c)
        return GT2TriangularMembership(
            upper_mf=TriangularMembership(a=upper_a, b=b, c=upper_c),
            lower_mf=TriangularMembership(a=lower_a, b=b, c=lower_c),
            principal_mf=TriangularMembership(a=principal_a, b=b, c=principal_c),
            id=uuid.uuid4(),
        )

    def alpha_cut(self, alpha: float) -> IT2TriangularMembership:
        """The IT2-shaped alpha-plane at level ``alpha`` in ``[0, 1]`` -- see
        `GT2GaussianMembership.alpha_cut`; each leg (``a``, ``c``) is
        interpolated toward its own principal value independently, with an
        infinite (shoulder) leg passed through unchanged."""
        def _cut(lower_val: float, principal_val: float, upper_val: float) -> tuple[float, float]:
            if not np.isfinite(principal_val):
                return principal_val, principal_val
            return (
                lower_val + alpha * (principal_val - lower_val),
                upper_val - alpha * (upper_val - principal_val),
            )

        a_lower, a_upper = _cut(self.lower_mf.a, self.principal_mf.a, self.upper_mf.a)
        c_lower, c_upper = _cut(self.lower_mf.c, self.principal_mf.c, self.upper_mf.c)
        b = self.principal_mf.b
        return IT2TriangularMembership(
            upper_mf=TriangularMembership(a=a_upper, b=b, c=c_upper, id=self.upper_mf.id),
            lower_mf=TriangularMembership(a=a_lower, b=b, c=c_lower, id=self.lower_mf.id),
            id=self.id,
        )


GT2AnyMembership = GT2GaussianMembership | GT2TrapezoidMembership | GT2TriangularMembership

_GT2_MEMBERSHIP_BY_TYPE1: dict[type, type] = {
    GaussianMembership: GT2GaussianMembership,
    TrapezoidMembership: GT2TrapezoidMembership,
    TriangularMembership: GT2TriangularMembership,
}


def to_gt2_membership(
    upper_mf: AnyMembership, lower_mf: AnyMembership, principal_mf: AnyMembership,
    id: Optional[uuid.UUID] = None,
) -> GT2AnyMembership:
    """Wrap a ``(upper_mf, lower_mf, principal_mf)`` triple (e.g. ``upper``/
    ``lower`` from `widen_membership`, ``principal`` the original Type-1 fit)
    in the matching `GT2AnyMembership` container for their shared Type-1 type."""
    gt2_cls = _GT2_MEMBERSHIP_BY_TYPE1.get(type(upper_mf))
    if gt2_cls is None:
        raise TypeError(f"Unsupported membership type for to_gt2_membership: {type(upper_mf)!r}")
    return gt2_cls(upper_mf=upper_mf, lower_mf=lower_mf, principal_mf=principal_mf, id=id)


class GT2LabelModel(NamedTuple):
    """A collection of GT2 membership functions for a specific output class label."""

    memberships: list[GT2AnyMembership]

    def augment(self, other_label_model: "GT2LabelModel") -> "GT2LabelModel":
        """Augment this GT2LabelModel with another, combining membership functions."""
        new_memberships = self.memberships.copy()
        new_memberships.extend(other_label_model.memberships)
        return GT2LabelModel(new_memberships)


class GT2FeatureModel(NamedTuple):
    """A collection of GT2LabelModels for a specific feature."""

    label_models: dict[int, GT2LabelModel]

    @property
    def ordered_keys(self) -> list[int]:
        return list(sorted(self.label_models.keys()))

    def augment(self, other_feature_model: "GT2FeatureModel") -> "GT2FeatureModel":
        """Augment this GT2FeatureModel with another."""
        new_label_models = self.label_models.copy()
        for label, other_label_model in other_feature_model.label_models.items():
            if label in new_label_models:
                new_label_models[label] = new_label_models[label].augment(other_label_model)
            else:
                new_label_models[label] = other_label_model
        return GT2FeatureModel(new_label_models)


class GT2GaussianMixtureModel(NamedTuple):
    """A general type-2 Gaussian mixture model, alpha-plane represented."""

    feature_models: dict[str, GT2FeatureModel]

    @property
    def n_rules(self) -> int:
        return len(list(self.feature_models.values())[0].label_models.keys())

    @property
    def n_features(self) -> int:
        return len(self.feature_models)

    @property
    def all_membership_fcns(self) -> list[GT2GaussianMembership]:
        """Get all GT2 membership functions across all features and labels."""
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