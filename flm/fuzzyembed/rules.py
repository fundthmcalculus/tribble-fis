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

    ``fit`` mines single-term rules, then pairs built from the strongest singles.
    Depth is capped at 2 deliberately: with named terms a 2-conjunct rule is still
    readable aloud, and the candidate space grows quadratically per level.
    """

    def __init__(self, max_rules: int = 24, max_order: int = 2,
                 min_support: float = 8.0, t_norm: str = "product",
                 top_singles: int = 24, min_interaction: float = 0.02):
        if t_norm not in T_NORMS:
            raise ValueError(f"t_norm must be one of {T_NORMS}")
        self.max_rules = max_rules
        self.max_order = max_order
        self.min_support = min_support
        self.t_norm = t_norm
        self.top_singles = top_singles
        self.min_interaction = min_interaction
        self.rules_: list[Rule] = []
        self.default_: float = 0.0
        self.feature_names_: list[str] = []

    # -- firing ------------------------------------------------------------

    def _fire(self, X: np.ndarray, features: tuple[int, ...]) -> np.ndarray:
        if len(features) == 1:
            return X[:, features[0]]
        cols = X[:, list(features)]
        return cols.prod(axis=1) if self.t_norm == "product" else cols.min(axis=1)

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
        candidates = list(singles)

        # Order 2, built only from the strongest singles.
        if self.max_order >= 2:
            seeds = singles[: self.top_singles]
            for a_i in range(len(seeds)):
                for b_i in range(a_i + 1, len(seeds)):
                    ra, rb = seeds[a_i], seeds[b_i]
                    ja, jb = ra.features[0], rb.features[0]
                    feats = (ja, jb)
                    fire = self._fire(X, feats)
                    cons, sup = self._consequent(fire, y)
                    if sup < self.min_support:
                        continue
                    # Keep a pair only if the conjunction says something *neither*
                    # conjunct says alone -- i.e. it lands further from the default
                    # than both parents. Without this the rule base fills with
                    # redundant conjunctions of correlated aliases (the observed case
                    # was `prev1:adj.all AND prev1:OPEN_ADJ`, two names for
                    # "the previous token is an adjective", which adds no information
                    # and costs a reader's attention).
                    gain = abs(cons - self.default_)
                    parent = max(abs(ra.consequent - self.default_),
                                 abs(rb.consequent - self.default_))
                    if gain <= parent + self.min_interaction:
                        continue
                    lift = (cons - self.default_) * np.sqrt(sup)
                    candidates.append(Rule(
                        feats, (self.feature_names_[ja], self.feature_names_[jb]),
                        cons, sup, lift))

        candidates.sort(key=lambda r: -abs(r.lift))
        self.rules_ = candidates[: self.max_rules]
        return self

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
