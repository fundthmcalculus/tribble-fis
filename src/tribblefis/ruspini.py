"""Ruspini-partitioned triangular fuzzy models.

The models produced by the rest of the package use an *implicit* layout
(:class:`GaussianMixtureModel`): every ``(feature, class)`` pair carries its own
private set of Gaussian membership functions, and the "rules" are implicit -- one
per class, ``argmax`` of the per-class firing strengths. Those per-class Gaussians
do **not** partition the input axis: they overlap arbitrarily and their
memberships do not sum to anything in particular.

This module derives, from such a trained model, an *explicit* layout
(:class:`SimpleGaussianClassifierModel`: a flat list of membership functions plus
explicit :class:`Rule` objects) in which each feature is covered by a single
shared **Ruspini partition** of *triangular* terms. A Ruspini partition is a
family of fuzzy sets whose memberships sum to exactly 1 at every point (a fuzzy
"partition of unity"). Triangular terms placed on shared apex knots -- with a left
shoulder on the first term and a right shoulder on the last -- have exactly this
property, so the partition is a genuine, interpretable set of linguistic terms
("the value is around knot k") that tile the axis.

Pipeline ("TRIBBLE a strong initial candidate, then refine"):

1. :func:`ruspinize_model` builds the initial candidate. Per feature it collects
   every class's Gaussian centres as *landmarks*, merges near-duplicates, and uses
   the survivors as the triangular apex knots. It then writes one explicit rule per
   class by matching each class -- feature by feature -- to the partition term whose
   apex is nearest that class's Gaussian centre(s). (This is a deliberately simple,
   fast "membership-function matching" heuristic; it is easy to swap out.)
2. :func:`tribblefis.refine.refine_ruspini_partition` then moves the *apex knots*
   against a classification objective using the `optimizers` package. Because the
   terms are always rebuilt from shared knots, the partition-of-unity property is
   preserved for free -- the search space is just the (monotone) knot vector.

:class:`RuspiniFuzzyClassifier` wraps the whole flow behind the scikit-learn API.
"""

import uuid
import typing

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, ClassifierMixin

from .gauss_data import (
    TriangularMembership,
    GaussianMembership,
    GaussianMixtureModel,
    SimpleGaussianClassifierModel,
    Rule,
    AnomalyParameters,
)


# ---------------------------------------------------------------------------
# Triangular Ruspini partition of a single axis.
# ---------------------------------------------------------------------------

def build_triangular_partition(
    apexes: typing.Sequence[float],
    term_ids: typing.Sequence[uuid.UUID] | None = None,
) -> list[TriangularMembership]:
    """Build a Ruspini partition of triangular terms from sorted apex knots.

    Given apex knots ``c_0 < c_1 < ... < c_{k-1}`` returns ``k`` triangular
    membership functions whose memberships sum to exactly 1 at every point:

    * term 0 is a *left shoulder* (1 for ``x <= c_0``, falling to 0 at ``c_1``);
    * term ``i`` (``0 < i < k-1``) is a triangle apexed at ``c_i``, rising from
      ``c_{i-1}`` and falling to ``c_{i+1}``;
    * term ``k-1`` is a *right shoulder* (rising from ``c_{k-2}``, then 1 for
      ``x >= c_{k-1}``).

    ``term_ids`` (optional) supplies stable ids per term index so a partition can
    be rebuilt with moved knots without changing the ids the rules reference.
    """
    apexes = np.asarray(apexes, dtype=float)
    k = len(apexes)
    if k == 0:
        raise ValueError("A partition needs at least one apex.")
    if term_ids is None:
        term_ids = [uuid.uuid4() for _ in range(k)]
    if len(term_ids) != k:
        raise ValueError("term_ids length must match the number of apexes.")

    terms: list[TriangularMembership] = []
    for i in range(k):
        a = -np.inf if i == 0 else float(apexes[i - 1])
        b = float(apexes[i])
        c = np.inf if i == k - 1 else float(apexes[i + 1])
        terms.append(TriangularMembership(a=a, b=b, c=c, id=term_ids[i]))
    return terms


def _merge_close(values: typing.Sequence[float], tol: float) -> list[float]:
    """Greedily merge sorted values whose gap is ``<= tol`` into their mean."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return []
    clusters: list[list[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


# ---------------------------------------------------------------------------
# Explicit Ruspini model: shared knots + frozen class->term rule assignment.
# ---------------------------------------------------------------------------

@dataclass
class RuspiniPartitionModel:
    """An explicit, Ruspini-partitioned triangular classifier.

    The *free parameters* are the per-feature apex knots (``apexes``); everything
    else is structure. ``rules`` freezes which partition term(s) each class fires
    on each feature (by term *index*), so moving the knots keeps the linguistic
    rule base intact while the term shapes change -- this is what makes knot
    refinement well behaved. ``term_ids`` keeps the per-term ids stable across
    rebuilds so the generated :class:`Rule` objects stay consistent.
    """

    feature_order: list[str]
    apexes: dict[str, np.ndarray]
    term_ids: dict[str, list[uuid.UUID]]
    # Each rule is (consequent_label, {feature_name: [term_index, ...]}).
    rules: list[tuple[typing.Any, dict[str, list[int]]]]
    anomaly_params: AnomalyParameters | None = None
    # Inference t-norm/conorm. The default is the *product* ("probability") norm:
    # unlike min/max it is smooth (every knot affects every rule's firing), which
    # both lets the knot refinement see a gradient and -- as a product of
    # per-feature term memberships -- gives a naive-Bayes-like joint score that
    # classifies better than the min t-norm.
    norm_conorm: str = "probability"

    def feature_terms(self) -> dict[str, list[TriangularMembership]]:
        return {
            f: build_triangular_partition(self.apexes[f], self.term_ids[f])
            for f in self.feature_order
        }

    def class_proba(self, X: pd.DataFrame) -> tuple[np.ndarray, list]:
        """Row-normalised per-class firing strengths and the class labels.

        Each rule fires the AND (``norm_conorm`` t-norm) across features of the OR
        (t-conorm) over that class's assigned terms on the feature; rows are then
        normalised to a probability over the rules' consequents."""
        from .gauss_math import t_norm, t_conorm

        n = len(X)
        terms = self.feature_terms()
        feat_eval = {
            f: [t.evaluate(X[f].to_numpy(dtype=float)) if f in X.columns else np.zeros(n)
                for t in terms[f]]
            for f in self.feature_order
        }
        labels = [consequent for consequent, _ in self.rules]
        fs = np.zeros((n, len(labels)))
        for r, (_consequent, antecedent_idx) in enumerate(self.rules):
            firing = np.ones(n)
            for f, idxs in antecedent_idx.items():
                feature_membership = np.zeros(n)
                for i in idxs:
                    feature_membership = t_conorm(feature_membership, feat_eval[f][i], self.norm_conorm)
                firing = t_norm(firing, feature_membership, self.norm_conorm)
            fs[:, r] = firing
        row = fs.sum(axis=1, keepdims=True)
        proba = np.full_like(fs, 1.0 / max(len(labels), 1))
        nz = row.flatten() > 0
        proba[nz] = fs[nz] / row[nz]
        return proba, labels

    def to_simple_model(self) -> SimpleGaussianClassifierModel:
        """Materialise the current knots as an explicit :class:`SimpleGaussianClassifierModel`
        (flat triangular MFs + explicit rules) for inspection / interoperability."""
        terms = self.feature_terms()
        input_mfs = [mf for f in self.feature_order for mf in terms[f]]
        rules: list[Rule] = []
        for consequent, antecedent_idx in self.rules:
            antecedents = {
                f: [terms[f][i].id for i in idxs]
                for f, idxs in antecedent_idx.items()
            }
            rules.append(Rule(antecedents=antecedents, consequent=consequent))
        # Carry the model's inference t-norm into the explicit form (without adding
        # an anomaly class) so ``simple_gaussian_predict`` on the serialized model
        # matches this model's own ``predict``.
        ap = self.anomaly_params
        if ap is None:
            ap = AnomalyParameters(include_anomaly=False, norm_conorm=self.norm_conorm)
        return SimpleGaussianClassifierModel(
            input_mfs=input_mfs, rules=rules, anomaly_params=ap
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba, labels = self.class_proba(X)
        # Let numpy infer the natural dtype (int for int labels, str for str), so
        # downstream metrics don't see an ``object`` array.
        return np.array([labels[i] for i in np.argmax(proba, axis=1)])

    # -- knot vector <-> flat parameter vector (for refinement) ---------------

    def extract_knots(self) -> np.ndarray:
        """Flatten every feature's apex knots into one vector (feature order)."""
        return np.concatenate([np.asarray(self.apexes[f], float) for f in self.feature_order])

    def knot_slices(self) -> dict[str, slice]:
        slices, k = {}, 0
        for f in self.feature_order:
            n = len(self.apexes[f])
            slices[f] = slice(k, k + n)
            k += n
        return slices

    def with_knots(self, vec: np.ndarray) -> "RuspiniPartitionModel":
        """Return a copy with apex knots taken from ``vec`` (each feature's knots
        are sorted so the result is always a valid monotone partition)."""
        vec = np.asarray(vec, dtype=float)
        slices = self.knot_slices()
        new_apexes = {}
        for f in self.feature_order:
            a = np.sort(vec[slices[f]])
            # Keep knots strictly increasing so triangle widths never collapse.
            a = _dedupe_increasing(a)
            new_apexes[f] = a
        return RuspiniPartitionModel(
            feature_order=list(self.feature_order),
            apexes=new_apexes,
            term_ids={f: list(self.term_ids[f]) for f in self.feature_order},
            rules=[(c, {f: list(idxs) for f, idxs in ant.items()}) for c, ant in self.rules],
            anomaly_params=self.anomaly_params,
            norm_conorm=self.norm_conorm,
        )

    @property
    def n_terms_total(self) -> int:
        return sum(len(v) for v in self.apexes.values())


def _dedupe_increasing(a: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Nudge a sorted array to be strictly increasing (avoids zero-width triangles)."""
    a = np.array(a, dtype=float)
    for i in range(1, len(a)):
        if a[i] <= a[i - 1]:
            a[i] = a[i - 1] + eps
    return a


# ---------------------------------------------------------------------------
# TRIBBLE the initial candidate: implicit Gaussian model -> explicit Ruspini.
# ---------------------------------------------------------------------------

def ruspinize_model(
    model: GaussianMixtureModel,
    X: pd.DataFrame,
    y: typing.Any | None = None,
    *,
    merge_tol_frac: float = 0.05,
    max_terms: int | None = None,
    assign_frac: float = 0.5,
    sigma_knots: float = 0.0,
    anomaly_params: AnomalyParameters | None = None,
) -> RuspiniPartitionModel:
    """Convert a derived (implicit) Gaussian classifier into an explicit,
    Ruspini-partitioned triangular classifier -- the strong initial candidate.

    Per feature, the union of every class's Gaussian centres becomes the pool of
    apex-knot *landmarks*; landmarks closer than ``merge_tol_frac`` of the feature
    range are merged. The survivors are the triangular apex knots (a Ruspini
    partition of that axis).

    **Membership-function matching (data-driven).** One explicit rule per class is
    written by matching the class, feature by feature, to the partition term(s) it
    actually occupies. When ``y`` is supplied, each term's *mean activation over the
    class's own samples* is measured and every term scoring at least ``assign_frac``
    of the class's best term is OR'd into the rule (always at least the best term).
    This makes a rule cover a class's real data footprint on each axis, so a single
    off-centre feature can't drop the rule's ``min`` firing to zero -- far more
    robust than matching only the term nearest the Gaussian mean (the fallback used
    when ``y`` is ``None``).

    Args:
        model: a trained :class:`GaussianMixtureModel` (Gaussian memberships).
        X: training features (observed range + data-driven term assignment).
        y: training labels; enables the data-driven matching above.
        merge_tol_frac: landmark-merge tolerance as a fraction of the feature range.
        max_terms: optional cap on the number of terms per feature (quantile thinning).
        assign_frac: activation threshold (fraction of a class's best term) for OR-ing
            a term into that class's rule.
        anomaly_params: anomaly parameters for the explicit model (defaults to the
            source model's).
    """
    feature_order = list(model.feature_models.keys())
    labels = sorted({lab for fm in model.feature_models.values() for lab in fm.label_models})
    y_arr = np.asarray(y) if y is not None else None

    apexes: dict[str, np.ndarray] = {}
    term_ids: dict[str, list[uuid.UUID]] = {}
    # class_feature_term[feature][label] = [term_index, ...]
    class_feature_term: dict[str, dict[typing.Any, list[int]]] = {}

    for f in feature_order:
        fmodel = model.feature_models[f]
        if f in X.columns:
            col = X[f].to_numpy(dtype=float)
            lo, hi = float(np.min(col)), float(np.max(col))
        else:
            col = None
            lo, hi = 0.0, 1.0
        rng = hi - lo if hi > lo else 1.0

        per_class_mus: dict[typing.Any, list[float]] = {}
        landmarks: list[float] = []
        for label, lm in fmodel.label_models.items():
            mus = [float(mf.mu) for mf in lm.memberships if isinstance(mf, GaussianMembership)]
            per_class_mus[label] = sorted(mus)
            landmarks.extend(mus)
            # Encode each Gaussian's *width* as extra apex knots at mu +/- k*sigma,
            # so the triangular terms resolve class spread rather than only centres.
            # (Clamped to the observed range; merged with the centres below.)
            if sigma_knots > 0:
                for mf in lm.memberships:
                    if isinstance(mf, GaussianMembership) and mf.sigma > 1e-6:
                        for s in (-sigma_knots, sigma_knots):
                            v = mf.mu + s * mf.sigma
                            if col is None or (lo <= v <= hi):
                                landmarks.append(float(v))

        merged = _merge_close(landmarks, merge_tol_frac * rng)
        if len(merged) < 2:  # a single (or no) landmark can't partition -- span the range
            merged = [lo, hi] if hi > lo else [lo, lo + 1.0]
        if max_terms is not None and len(merged) > max_terms:
            # Thin to `max_terms` by evenly spaced quantiles over the landmark list.
            qs = np.linspace(0, len(merged) - 1, max_terms).round().astype(int)
            merged = [merged[i] for i in sorted(set(qs))]

        apex = _dedupe_increasing(np.array(sorted(merged), dtype=float))
        apexes[f] = apex
        term_ids[f] = [uuid.uuid4() for _ in apex]
        terms = build_triangular_partition(apex, term_ids[f])
        mid = 0.5 * (lo + hi)

        class_feature_term[f] = {}
        for label in labels:
            if y_arr is not None and col is not None:
                # Data-driven: OR every term the class's own samples activate well.
                mask = y_arr == label
                if np.any(mask):
                    acts = np.array([terms[i].evaluate(col[mask]).mean() for i in range(len(apex))])
                    best = float(acts.max())
                    if best > 0:
                        idxs = sorted(int(i) for i in np.where(acts >= assign_frac * best)[0])
                    else:
                        idxs = [int(np.argmax(acts))]
                    class_feature_term[f][label] = idxs
                    continue
            # Fallback (no labels): match the term nearest the class's Gaussian mean(s).
            mus = per_class_mus.get(label, [])
            if not mus:
                idxs = [int(np.argmin(np.abs(apex - mid)))]
            else:
                idxs = sorted({int(np.argmin(np.abs(apex - mu))) for mu in mus})
            class_feature_term[f][label] = idxs

    rules = [
        (label, {f: class_feature_term[f][label] for f in feature_order})
        for label in labels
    ]
    return RuspiniPartitionModel(
        feature_order=feature_order,
        apexes=apexes,
        term_ids=term_ids,
        rules=rules,
        anomaly_params=anomaly_params if anomaly_params is not None else model.anomaly_params,
    )


# ---------------------------------------------------------------------------
# scikit-learn estimator.
# ---------------------------------------------------------------------------

class RuspiniFuzzyClassifier(BaseEstimator, ClassifierMixin):
    """A triangular, Ruspini-partitioned fuzzy classifier.

    Fits a Gaussian :class:`TribbleClassifier` to derive the
    per-class landmarks, converts it to an explicit Ruspini triangular model
    (:func:`ruspinize_model`), and -- optionally -- refines the partition's apex
    knots against a cross-entropy objective with the `optimizers` package
    (see :func:`tribblefis.refine.refine_ruspini_partition`).
    """

    def __init__(
        self,
        top_n=-1,
        top_p=0.95,
        n_gaussians=0,
        merge_tol_frac=0.05,
        max_terms=None,
        refine=False,
        refine_method="coordinate",
        refine_l2_shrink=0.02,
        random_state=42,
    ):
        self.top_n = top_n
        self.top_p = top_p
        self.n_gaussians = n_gaussians
        self.merge_tol_frac = merge_tol_frac
        self.max_terms = max_terms
        self.refine = refine
        self.refine_method = refine_method
        self.refine_l2_shrink = refine_l2_shrink
        self.random_state = random_state

    def fit(self, X, y):
        from .gaussian_classifier import TribbleClassifier

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_df = X.reset_index(drop=True)
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
            X_df = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        base = TribbleClassifier(
            top_n=self.top_n, top_p=self.top_p, n_gaussians=self.n_gaussians,
            member_function="gaussian",
            random_state=self.random_state,
        )
        base.fit(X_df, y_arr)
        self.base_model_ = base.model_
        self.top_features_ = base.top_features_

        # Only the selected features carry membership models; restrict X to them.
        X_feat = X_df[[f for f in base.top_features_ if f in X_df.columns]]
        self.ruspini_model_ = ruspinize_model(
            base.model_, X_feat, y_arr,
            merge_tol_frac=self.merge_tol_frac, max_terms=self.max_terms,
        )
        self.refine_info_ = None
        if self.refine:
            from .refine import refine_ruspini_partition
            self.ruspini_model_, self.refine_info_ = refine_ruspini_partition(
                self.ruspini_model_, X_feat, y_arr,
                method=self.refine_method, l2_shrink=self.refine_l2_shrink,
                seed=self.random_state, verbose=False,
            )
        self.is_fitted_ = True
        return self

    def _prep(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)

    def predict(self, X):
        return self.ruspini_model_.predict(self._prep(X))

    @property
    def simple_model_(self) -> SimpleGaussianClassifierModel:
        """The explicit (flat MFs + rules) form of the current Ruspini model."""
        return self.ruspini_model_.to_simple_model()
