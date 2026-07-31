"""A TSK rule learner for inputs that are *already* membership degrees.

Why this exists
---------------
Feeding the fuzzy embedding into ``MixtureOfGaussiansFuzzyClassifier`` produced a
model at chance. Measured, on identical features and splits:

    target        logreg AUC   FIS/gaussian AUC   FIS/trapezoid AUC
    OPEN_NOUN        0.673          0.488            0.500 (spread 0.000)
    OPEN_VERB        0.752          0.511            0.500 (spread 0.000)
    DETERMINER       0.730          0.515            0.500 (spread 0.000)

Logistic regression learns; the FIS does not. So the features carry signal and the
*antecedent representation* was the problem.

The cause: the feature matrix is **93% zeros**. TRIBBLE fits a Gaussian (or
trapezoid) membership function per ``(feature, class)``, which assumes a continuous,
unimodal, reasonably-spread variable -- true of its benchmark data (concrete
strength, turbine power, wine chemistry), false of a sparse membership vector. Fit a
Gaussian to ``{0 w.p. 0.95, 1 w.p. 0.05}`` and you get a narrow near-zero Gaussian
for *both* classes; per-feature memberships then barely differ, the t-norm product of
near-identical values is near-identical, and ``predict_proba`` collapses to 0.5. The
trapezoid variant degenerates completely and emits a constant.

The insight
-----------
**When the inputs are already fuzzy memberships, there is no membership function to
fit -- the input value *is* the membership degree.** The whole antecedent-fitting
layer is not just ill-suited here, it is redundant. What remains to learn is rule
*structure* (which conjunctions matter) and *consequents*.

So this is a zero-order TSK system where:

- an antecedent is a conjunction of named terms, e.g.
  ``prev1:DETERMINER AND prev2:OPEN_VERB``;
- its firing strength is the t-norm of the corresponding input values -- read
  straight off, not through a fitted curve;
- its consequent is the firing-weighted mean of the target;
- prediction is the firing-weighted blend over rules, with a default rule carrying
  the global mean so an all-quiet input still predicts something sensible.

Interpretability is *better* than the Gaussian version, not worse: an antecedent is
literally "the previous token is a determiner, to degree 0.9", with no fitted centre
or width to explain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

T_NORMS = ("product", "min")


@dataclass
class Rule:
    """One fuzzy rule: a named conjunction and a consequent value."""

    features: tuple[int, ...]
    names: tuple[str, ...]
    consequent: float
    support: float
    lift: float = 0.0

    def render(self, target: str = "y") -> str:
        ante = " AND ".join(self.names)
        return (f"IF {ante} THEN {target} ~ {self.consequent:.3f}"
                f"   (support={self.support:.1f}, lift={self.lift:+.3f})")


class MembershipRuleRegressor:
    """Zero-order TSK over membership-valued inputs. Targets in [0, 1].

    ``fit`` mines single-term rules, then grows higher orders level-wise from the
    strongest singles, keeping a beam of ``beam`` survivors per level. ``max_order``
    trades readability for expressiveness: a 2-conjunct rule is readable aloud, and
    3 conjuncts is about the limit before a rule stops being an explanation.
    """

    def __init__(self, max_rules: int = 24, max_order: int = 2,
                 min_support: float = 8.0, t_norm: str = "product",
                 top_singles: int = 24, min_interaction: float = 0.02,
                 beam: int = 24, must_include: set[int] | None = None,
                 order_quota: dict[int, float] | None = None,
                 seed_features: set[int] | None = None):
        if t_norm not in T_NORMS:
            raise ValueError(f"t_norm must be one of {T_NORMS}")
        self.max_rules = max_rules
        self.max_order = max_order
        self.min_support = min_support
        self.t_norm = t_norm
        self.top_singles = top_singles
        self.min_interaction = min_interaction
        self.beam = beam
        # Feature indices of which every rule must use at least one. Needed by the
        # joint ranker: a rule over context features *only* assigns the same score to
        # every candidate word, so it cannot affect a ranking at all -- it would just
        # consume rule slots while contributing a constant.
        self.must_include = set(must_include) if must_include else None
        # Fraction of the rule budget reserved per antecedent order. Needed because
        # ranking by |lift| = |consequent - default| * sqrt(support) systematically
        # favours high-support single-term rules: on the joint ranking task the whole
        # 40-rule base filled with candidate-only marginals and not one
        # context-x-candidate interaction survived, so the scorer could not depend on
        # context at all. Quotas guarantee interactions a share of the budget.
        self.order_quota = order_quota
        # Feature indices always admitted to the growth seed pool regardless of their
        # marginal lift. Required for **pure-interaction** structure, which lift-based
        # seeding provably cannot find: on the joint ranking task a position's positive
        # and all its negatives share one context vector, so every `ctx:` feature fires
        # identically on both and its firing-weighted target equals the base rate --
        # marginal lift exactly zero. Those features never reached the top-k seed pool,
        # so no context-x-candidate rule was ever generated and the ranker could only
        # learn candidate marginals. This is the XOR problem for greedy selection:
        # informative jointly, invisible marginally.
        self.seed_features = set(seed_features) if seed_features else None
        self.rules_: list[Rule] = []
        self.default_: float = 0.0
        self.feature_names_: list[str] = []

    # -- firing ------------------------------------------------------------

    def _fire(self, X: np.ndarray, features: tuple[int, ...]) -> np.ndarray:
        if len(features) == 1:
            return X[:, features[0]]
        cols = X[:, list(features)]
        return cols.prod(axis=1) if self.t_norm == "product" else cols.min(axis=1)

    def _admissible(self, features: tuple[int, ...]) -> bool:
        """Whether a rule may enter the base (see ``must_include``)."""
        if self.must_include is None:
            return True
        return any(f in self.must_include for f in features)

    @property
    def max_order_used(self) -> int:
        return max((len(r.features) for r in self.rules_), default=0)

    @staticmethod
    def _consequent(fire: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        s = fire.sum()
        if s <= 0:
            return 0.0, 0.0
        return float((fire * y).sum() / s), float(s)

    # -- fit ---------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: list[str] | None = None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]
        self.feature_names_ = (list(feature_names) if feature_names
                               else [f"f{i}" for i in range(n_features)])
        self.default_ = float(y.mean())

        # Order 1.
        singles: list[Rule] = []
        for j in range(n_features):
            fire = X[:, j]
            cons, sup = self._consequent(fire, y)
            if sup < self.min_support:
                continue
            # Lift weighted by sqrt(support): a rule firing on 9 samples with a wild
            # consequent is noise, and unweighted lift ranks exactly those first.
            lift = (cons - self.default_) * np.sqrt(sup)
            singles.append(Rule((j,), (self.feature_names_[j],), cons, sup, lift))

        singles.sort(key=lambda r: -abs(r.lift))
        candidates = [r for r in singles if self._admissible(r.features)]

        # Higher orders by level-wise growth (Apriori-style): extend each surviving
        # order-(k-1) rule by one strong single. A beam bounds the cost, which would
        # otherwise be combinatorial in the number of features.
        seeds = singles[: self.top_singles]
        if self.seed_features:
            have = {r.features[0] for r in seeds}
            seeds = seeds + [r for r in singles
                             if r.features[0] in self.seed_features
                             and r.features[0] not in have]
        # Carry each frontier rule's firing vector alongside it. Extending a rule is
        # then a single vectorised multiply against one column, instead of
        # re-multiplying all k columns from scratch -- the naive version recomputed
        # O(beam x top_singles x k) full-length products per output per level and was
        # too slow to finish an order-3 sweep at all.
        frontier: list[tuple[Rule, np.ndarray]] = [(r, X[:, r.features[0]])
                                                   for r in seeds]
        use_min = self.t_norm == "min"
        for _order in range(2, self.max_order + 1):
            if not frontier:
                break
            grown: list[tuple[Rule, np.ndarray]] = []
            seen: set[frozenset[int]] = set()
            for base, base_fire in frontier:
                base_gain = abs(base.consequent - self.default_)
                for single in seeds:
                    j = single.features[0]
                    if j in base.features:
                        continue
                    feats = tuple(sorted(base.features + (j,)))
                    key = frozenset(feats)
                    if key in seen:
                        continue
                    seen.add(key)
                    col = X[:, j]
                    fire = np.minimum(base_fire, col) if use_min else base_fire * col
                    cons, sup = self._consequent(fire, y)
                    if sup < self.min_support:
                        continue
                    # Keep an extension only if the conjunction says something
                    # *neither part* says alone -- it must land further from the
                    # default than both its parents. Without this the rule base fills
                    # with redundant conjunctions of correlated aliases (the observed
                    # case was `prev1:adj.all AND prev1:OPEN_ADJ`, two names for "the
                    # previous token is an adjective": no information, and it costs
                    # the reader's attention, which is the whole budget an
                    # interpretable model spends).
                    gain = abs(cons - self.default_)
                    parent = max(base_gain,
                                 abs(single.consequent - self.default_))
                    if gain <= parent + self.min_interaction:
                        continue
                    lift = (cons - self.default_) * np.sqrt(sup)
                    grown.append((Rule(
                        feats, tuple(self.feature_names_[f] for f in feats),
                        cons, sup, lift), fire))
            grown.sort(key=lambda rf: -abs(rf[0].lift))
            frontier = grown[: self.beam]
            candidates.extend(r for r, _ in frontier
                              if self._admissible(r.features))

        candidates.sort(key=lambda r: -abs(r.lift))
        self.rules_ = self._apply_quota(candidates)
        return self

    def _apply_quota(self, candidates: list[Rule]) -> list[Rule]:
        """Select the rule base, honouring ``order_quota`` if given."""
        if not self.order_quota:
            return candidates[: self.max_rules]
        by_order: dict[int, list[Rule]] = {}
        for rule in candidates:
            by_order.setdefault(len(rule.features), []).append(rule)
        chosen: list[Rule] = []
        for order, frac in sorted(self.order_quota.items()):
            take = int(round(self.max_rules * frac))
            chosen.extend(by_order.get(order, [])[:take])
        # Backfill any unused quota (an order may have too few candidates) from the
        # global ranking, so a quota never shrinks the rule base.
        if len(chosen) < self.max_rules:
            have = {r.features for r in chosen}
            for rule in candidates:
                if rule.features not in have:
                    chosen.append(rule)
                    if len(chosen) >= self.max_rules:
                        break
        chosen.sort(key=lambda r: -abs(r.lift))
        return chosen[: self.max_rules]

    def order_histogram(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for rule in self.rules_:
            out[len(rule.features)] = out.get(len(rule.features), 0) + 1
        return dict(sorted(out.items()))

    # -- predict -----------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not self.rules_:
            return np.full(len(X), self.default_)

        num = np.zeros(len(X))
        den = np.zeros(len(X))
        for rule in self.rules_:
            fire = self._fire(X, rule.features)
            num += fire * rule.consequent
            den += fire

        # Default rule with a small fixed firing strength: without it, an input that
        # matches no antecedent divides by zero. With it, "nothing fired" degrades to
        # the prior instead of to a NaN.
        eps = 0.05
        num += eps * self.default_
        den += eps
        return np.clip(num / den, 0.0, 1.0)

    # -- inspection --------------------------------------------------------

    def render(self, target: str = "y", k: int = 10) -> str:
        lines = [f"{len(self.rules_)} rules (default {target} = {self.default_:.3f}):"]
        for rule in self.rules_[:k]:
            lines.append("  " + rule.render(target))
        if len(self.rules_) > k:
            lines.append(f"  ... {len(self.rules_) - k} more")
        return "\n".join(lines)

    @property
    def mean_antecedents(self) -> float:
        if not self.rules_:
            return 0.0
        return float(np.mean([len(r.features) for r in self.rules_]))
