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
                 seed_features: set[int] | None = None,
                 reserved_features: set[int] | None = None,
                 reserved_quota: float = 0.0,
                 dtype=np.float64, significance: float | None = None,
                 holdout: float = 0.0, holdout_seed: int = 0):
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
        # Feature indices guaranteed ``reserved_quota`` of the rule budget. See
        # ``_apply_quota`` for the measurement that motivated it (E24.3).
        self.reserved_features = set(reserved_features) if reserved_features else None
        self.reserved_quota = reserved_quota
        # Working precision for the design matrix. float64 by default because the GEMM
        # exactness test asserts agreement with per-candidate computation to rel=1e-9, which
        # float32 accumulation over tens of thousands of rows cannot hold. float32 exists for
        # wide context windows, where memory -- not time -- is the binding constraint: cost is
        # ``rows * dims * 2 * itemsize`` (X and Xy both materialised), so a 32-token window at
        # 8,613 columns needs ~137 KB/row in float64 and half that in float32 (E26).
        self.dtype = dtype
        # -- width-aware false-discovery control (E27) ------------------------------
        # E26.1 found the failure mode a wider context creates: the order-2 candidate pool
        # grows from ~0.2M pairs at window 2 to ~2.2M at window 32, spurious correlations
        # scale with the pool while real ones do not, and `min_support`/`min_interaction`
        # were tuned at window 2 and never tighten. 5,111 of 6,060 rules referenced a lag > 2
        # at weak lift, and `predict`'s firing-weighted blend dragged every prediction toward
        # the base rate.
        #
        # `significance` is a Bonferroni-corrected gate, and it costs nothing to compute
        # because **|lift| is already a z-statistic up to a constant**. Under the null that a
        # rule is irrelevant, its consequent is a mean of `support` draws with
        # SE = sqrt(p(1-p)/support), so
        #
        #     z = (consequent - default) / SE = lift / sqrt(p(1-p))
        #
        # Setting alpha/m with m = candidates *actually tested at that level* makes the
        # threshold tighten automatically as the search space grows -- which is exactly the
        # width-awareness E26.1 said was missing.
        self.significance = significance
        # `holdout` is the empirical alternative: mine on part of the rows, then require each
        # rule's effect to reproduce on rows never used to find it. Makes no distributional
        # assumption, and the survival rate is a direct measurement of the false-discovery
        # rate rather than an estimate of it.
        self.holdout = holdout
        self.holdout_seed = holdout_seed
        self.n_tested_: int = 0
        self.n_gated_: int = 0
        self.replication_: dict[str, float] = {}
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

    # -- significance ------------------------------------------------------

    def _z_critical(self, n_tested: int) -> float:
        """Bonferroni-corrected two-sided z threshold for ``n_tested`` comparisons."""
        from statistics import NormalDist
        alpha = self.significance / max(n_tested, 1)
        # Guard the tail: alpha/m underflows for very large m, and inv_cdf(1.0) is inf.
        q = min(1.0 - alpha / 2.0, 1.0 - 1e-15)
        return float(NormalDist().inv_cdf(q))

    def _z_of(self, consequent: float, support: float) -> float:
        """|z| for a rule, from quantities already computed. See ``significance``."""
        p = self.default_
        var = max(p * (1.0 - p), 1e-12)
        return abs(consequent - p) * np.sqrt(max(support, 0.0)) / np.sqrt(var)

    def _replicates(self, X, y, rules: list[Rule]) -> list[Rule]:
        """Keep rules whose effect reproduces on rows never used to mine them.

        Direction *and* magnitude must survive: a rule found because 24 of 54,000 rows
        happened to line up will not reproduce its sign on a disjoint sample, whereas
        ``IF prev1:DETERMINER AND cand:OPEN_NOUN`` will. Requiring only |z| again on the
        holdout would re-test the same statistic on less data; requiring the *sign* of the
        deviation to agree is the part that actually discriminates.
        """
        n = X.shape[0]
        rng = np.random.default_rng(self.holdout_seed)
        mask = rng.random(n) < self.holdout
        if mask.sum() < 50 or (~mask).sum() < 50:
            return rules
        Xv, yv = X[mask], y[mask]
        default_v = float(yv.mean())
        kept = []
        for rule in rules:
            fire = self._fire(Xv, rule.features)
            sup = float(fire.sum())
            if sup <= 0:
                continue
            cons = float((fire * yv).sum() / sup)
            same_sign = ((cons - default_v) * (rule.consequent - self.default_)) > 0
            if same_sign and sup >= max(self.min_support * self.holdout, 3.0):
                kept.append(rule)
        self.replication_ = {
            "tested": float(len(rules)),
            "replicated": float(len(kept)),
            "rate": len(kept) / max(len(rules), 1),
        }
        return kept

    # -- fit ---------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: list[str] | None = None):
        X = np.asarray(X, dtype=self.dtype)
        y = np.asarray(y, dtype=self.dtype)
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
        # Order-1 tests one hypothesis per feature.
        self.n_tested_ = n_features
        if self.significance is not None:
            zc = self._z_critical(n_features)
            before = len(candidates)
            candidates = [r for r in candidates
                          if self._z_of(r.consequent, r.support) >= zc]
            self.n_gated_ += before - len(candidates)

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
        seed_cols = np.array([r.features[0] for r in seeds], dtype=np.intp)
        seed_gain = np.abs(np.array([r.consequent for r in seeds])
                           - self.default_)
        Xy = X * y[:, None] if self.t_norm == "product" else None

        frontier_rules = list(seeds)
        frontier_fire = X[:, seed_cols].copy()
        for _order in range(2, self.max_order + 1):
            if not frontier_rules:
                break
            # Every (frontier rule x seed) cell is one hypothesis tested, so this is the
            # multiplicity that has to be corrected for -- and it is what grows with the
            # context window (E26.1).
            self.n_tested_ += len(frontier_rules) * len(seeds)
            frontier_rules, frontier_fire = self._grow_level(
                X, y, Xy, frontier_rules, frontier_fire, seeds, seed_cols,
                seed_gain)
            grown = [r for r in frontier_rules if self._admissible(r.features)]
            if self.significance is not None:
                zc = self._z_critical(self.n_tested_)
                before = len(grown)
                grown = [r for r in grown
                         if self._z_of(r.consequent, r.support) >= zc]
                self.n_gated_ += before - len(grown)
            candidates.extend(grown)

        if self.holdout > 0:
            candidates = self._replicates(X, y, candidates)

        candidates.sort(key=lambda r: -abs(r.lift))
        self.rules_ = self._apply_quota(candidates)
        return self

    def _grow_level(self, X, y, Xy, frontier_rules, frontier_fire, seeds,
                    seed_cols, seed_gain):
        """One level of rule growth, computed as two matrix products.

        With the **product** t-norm, every candidate extension's two required
        statistics factor into GEMMs over the frontier firing matrix ``F`` and the
        seed columns ``S``::

            support   = F.T @ S
            weighted  = F.T @ (S * y)      ->  consequent = weighted / support

        so a whole level costs two BLAS calls instead of one Python-level numpy call
        per candidate. That is what makes order-3 and large beams affordable: at
        beam 200 a level has ~30k candidates, and the per-call overhead of the naive
        loop -- not the arithmetic -- was the bottleneck.

        This is also why the hot path is *not* a Cython candidate. The work is
        already dense linear algebra dispatched to an optimised BLAS; hand-writing
        the loop in C would at best match one thread of it.

        The ``min`` t-norm does not factor this way (``min`` is not bilinear), so it
        falls back to the per-candidate loop and is correspondingly slower.
        """
        n_front = len(frontier_rules)
        n_seed = len(seeds)

        if self.t_norm == "product":
            S = X[:, seed_cols]
            support = frontier_fire.T @ S                    # (n_front, n_seed)
            weighted = frontier_fire.T @ Xy[:, seed_cols]
            with np.errstate(divide="ignore", invalid="ignore"):
                cons = np.where(support > 0, weighted / np.maximum(support, 1e-12),
                                0.0)
        else:
            support = np.empty((n_front, n_seed))
            cons = np.empty((n_front, n_seed))
            for i in range(n_front):
                fire = np.minimum(frontier_fire[:, i][:, None], X[:, seed_cols])
                support[i] = fire.sum(axis=0)
                w = (fire * y[:, None]).sum(axis=0)
                cons[i] = np.where(support[i] > 0,
                                   w / np.maximum(support[i], 1e-12), 0.0)

        gain = np.abs(cons - self.default_)
        base_gain = np.abs(np.array([r.consequent for r in frontier_rules])
                           - self.default_)
        # An extension must say something *neither part* says alone: it has to land
        # further from the default than both its parents. Without this the base fills
        # with redundant conjunctions of correlated aliases.
        parent = np.maximum(base_gain[:, None], seed_gain[None, :])
        valid = (support >= self.min_support) & (gain > parent + self.min_interaction)

        # A seed already present in the base rule is not an extension.
        for i, rule in enumerate(frontier_rules):
            for f in rule.features:
                valid[i, seed_cols == f] = False
        if not valid.any():
            return [], np.empty((X.shape[0], 0))

        lift = (cons - self.default_) * np.sqrt(np.maximum(support, 0.0))
        score = np.where(valid, np.abs(lift), -np.inf)

        # Take enough of the ranking to survive de-duplication (distinct (i, j) can
        # denote the same feature set), then trim to the beam.
        take = min(int(valid.sum()), self.beam * 4)
        flat = np.argpartition(score.ravel(), -take)[-take:]
        flat = flat[np.argsort(score.ravel()[flat])[::-1]]

        out_rules: list[Rule] = []
        out_cols: list[np.ndarray] = []
        seen: set[frozenset[int]] = set()
        for idx in flat:
            i, j = divmod(int(idx), n_seed)
            if not valid[i, j]:
                continue
            feats = tuple(sorted(frontier_rules[i].features + (int(seed_cols[j]),)))
            key = frozenset(feats)
            if key in seen:
                continue
            seen.add(key)
            out_rules.append(Rule(
                feats, tuple(self.feature_names_[f] for f in feats),
                float(cons[i, j]), float(support[i, j]), float(lift[i, j])))
            # Materialise the firing vector only for survivors -- the GEMM gave us
            # every candidate's statistics without ever forming its firing vector.
            out_cols.append(self._fire(X, feats))
            if len(out_rules) >= self.beam:
                break

        fire_mat = (np.stack(out_cols, axis=1) if out_cols
                    else np.empty((X.shape[0], 0)))
        return out_rules, fire_mat

    def _select_by_order(self, candidates: list[Rule], budget: int) -> list[Rule]:
        """Top ``budget`` rules, honouring ``order_quota`` within this pool."""
        if budget <= 0:
            return []
        if not self.order_quota:
            return candidates[:budget]
        by_order: dict[int, list[Rule]] = {}
        for rule in candidates:
            by_order.setdefault(len(rule.features), []).append(rule)
        chosen: list[Rule] = []
        for order, frac in sorted(self.order_quota.items()):
            chosen.extend(by_order.get(order, [])[: int(round(budget * frac))])
        # Backfill unused quota (an order may have too few candidates) from the global
        # ranking, so a quota never shrinks the rule base.
        if len(chosen) < budget:
            have = {r.features for r in chosen}
            for rule in candidates:
                if rule.features not in have:
                    chosen.append(rule)
                    if len(chosen) >= budget:
                        break
        chosen.sort(key=lambda r: -abs(r.lift))
        return chosen[:budget]

    def _apply_quota(self, candidates: list[Rule]) -> list[Rule]:
        """Select the rule base, honouring the order and reserved-feature quotas.

        ``reserved_quota`` exists because of a *measured* budget misallocation (E24.3).
        Ranking by ``|lift| = |consequent - default| * sqrt(support)`` follows support, and
        on the joint ranking task the highest-support features are closed-class by an order
        of magnitude (``cand:CONJUNCTION`` alone has support 17,250). The whole rule base
        therefore filled with function-word syntax -- correct rules, `IF prev1:PREPOSITION
        AND cand:DETERMINER` carries lift +10.09 -- and nothing that selects a *content*
        word, so generation produced only function words.

        This is the same failure shape as ``order_quota`` (lift favouring high-support
        singles until interactions were guaranteed a share) applied to a different axis, so
        it takes the same remedy: reserve part of the budget and let the reserved pool
        compete only against itself.
        """
        if not self.reserved_features or self.reserved_quota <= 0:
            return self._select_by_order(candidates, self.max_rules)

        res = self.reserved_features
        hit = [r for r in candidates if any(f in res for f in r.features)]
        rest = [r for r in candidates if not any(f in res for f in r.features)]
        n_res = int(round(self.max_rules * self.reserved_quota))
        chosen = self._select_by_order(hit, n_res)
        chosen += self._select_by_order(rest, self.max_rules - len(chosen))
        # Backfill, so a quota can never shrink the base below what it would have been.
        if len(chosen) < self.max_rules:
            have = {r.features for r in chosen}
            for rule in candidates:
                if rule.features not in have:
                    chosen.append(rule)
                    if len(chosen) >= self.max_rules:
                        break
        chosen.sort(key=lambda r: -abs(r.lift))
        return chosen[: self.max_rules]

    def reserved_histogram(self) -> dict[str, int]:
        """How much of the final base actually touches a reserved feature."""
        res = self.reserved_features or set()
        hit = sum(1 for r in self.rules_ if any(f in res for f in r.features))
        return {"reserved": hit, "other": len(self.rules_) - hit}

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
