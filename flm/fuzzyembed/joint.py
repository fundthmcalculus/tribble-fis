"""Joint next-token ranking -- scoring ``(context, candidate)`` pairs directly.

The problem this fixes
----------------------
``sequence.py`` predicts a membership degree per named dimension *independently*, then
``decode.py`` matches the resulting vector against the lexeme atlas. Nothing in that
pipeline enforces that the prediction corresponds to **one actual word**: a vector with
``OPEN_NOUN=0.5`` and ``OPEN_VERB=0.5`` is a perfectly good marginal prediction and a
description of no word at all. The marginals are individually plausible and jointly
incoherent, and its aggregate skill stayed marginal (balanced accuracy 0.527 +- 0.010)
after context width, rule order, the antecedent representation, the decode metric, and
feature hygiene were each ruled out.

So the target is reframed. Instead of "what degree does dimension *c* have next?", ask
**"is *w* the next word here?"** -- one binary, genuinely joint question. Features are
the context window's memberships *concatenated with the candidate's own*, so a rule can
say

    IF ctx:prev1:DETERMINER AND cand:OPEN_NOUN THEN P(next) ~ 0.71

which is exactly the context-times-candidate interaction the marginal formulation could
not express. This is noise-contrastive estimation with an interpretable scorer.

Two consequences worth the rewrite:

* **A real metric.** Ranking the true next token against sampled distractors gives MRR
  and hits@k -- a language-modelling measurement, not balanced accuracy on marginals.
* **Context-only rules are provably useless here** and are excluded via the rule
  learner's ``must_include``. Such a rule fires identically for every candidate, so it
  shifts all scores by the same constant and cannot change a ranking. Leaving them in
  would silently consume most of the rule budget, since context features outnumber
  candidate features by the window size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .corpus import Corpus
from .rules import MembershipRuleRegressor


@dataclass
class RankingResult:
    """Held-out ranking performance against sampled distractors."""

    name: str
    mrr: float
    hits1: float
    hits5: float
    hits10: float
    n_eval: int
    candidate_set: int
    n_rules: int = 0

    def row(self) -> str:
        return (f"{self.name:<26}{self.mrr:>8.3f}{self.hits1:>9.3f}"
                f"{self.hits5:>9.3f}{self.hits10:>10.3f}{self.n_rules:>8}")

    @staticmethod
    def header() -> str:
        return (f"{'model':<26}{'MRR':>8}{'hits@1':>9}{'hits@5':>9}"
                f"{'hits@10':>10}{'rules':>8}")


class JointNextTokenRanker:
    """Interpretable joint scorer for ``(context, candidate)`` pairs.

    Wraps a fitted-or-unfitted ``FuzzySequenceModel`` purely for its featuriser
    (``_token_vector`` / ``_output_names``), so the semantic + syntactic joint space is
    shared with the rest of the stack.
    """

    def __init__(self, featuriser, window: int = 2, n_negatives: int = 8,
                 max_rules: int = 300, max_order: int = 2, seed: int = 42,
                 order_quota: dict[int, float] | None = None, beam: int = 200):
        self.f = featuriser
        self.window = window
        self.n_negatives = n_negatives
        self.max_rules = max_rules
        self.max_order = max_order
        # Reserve most of the budget for context-x-candidate interactions. Without a
        # quota the base filled entirely with candidate-only marginals, which encode
        # "which word types are common" and nothing about context.
        self.order_quota = order_quota or {1: 0.15, 2: 0.85}
        # `beam` caps how many order-2 rules survive per growth level, so it -- not
        # `max_rules` -- is what actually limits the interaction supply. At the old
        # default of 24 the rule base saturated at 81 rules no matter how large
        # max_rules was set. 200 is the measured MRR optimum: 24 -> 0.244,
        # 80 -> 0.246, 200 -> 0.257, 400 -> 0.254 (hits@10 keeps rising to 0.680 at
        # 400, so raise it if recall matters more than top-1).
        self.beam = beam
        self.seed = seed
        self.model_: MembershipRuleRegressor | None = None
        self.feature_names_: list[str] = []
        self.cand_offset_: int = 0

    # -- featurisation -----------------------------------------------------

    def _names(self) -> list[str]:
        base = self.f._output_names()
        ctx = [f"ctx:prev{lag}:{n}"
               for lag in range(self.window, 0, -1) for n in base]
        self.cand_offset_ = len(ctx)
        return ctx + [f"cand:{n}" for n in base]

    def _context_vector(self, tokens: list[str], i: int) -> np.ndarray:
        """Window ending just before position ``i``, padded with the boundary token."""
        parts = []
        for lag in range(self.window, 0, -1):
            j = i - lag
            parts.append(self.f._token_vector(tokens[j] if j >= 0 else ""))
        return np.concatenate(parts)

    # -- data --------------------------------------------------------------

    def _sampler(self, corpus: Corpus, vocab: list[str]):
        """Frequency-weighted distractor sampler.

        Frequency-weighted rather than uniform, for both training negatives and
        evaluation distractors: uniform sampling from a Zipfian vocabulary draws almost
        only rare words, which makes the task trivially easy (any frequency signal
        separates them) and the reported numbers meaningless.
        """
        rng = np.random.default_rng(self.seed)
        w = np.array([corpus.counts.get(v, 1) for v in vocab], dtype=float)
        p = w / w.sum()
        return rng, p

    def build(self, corpus: Corpus, vocab: list[str], max_positions: int,
              seed_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng, p = self._sampler(corpus, vocab)
        rng = np.random.default_rng(self.seed + seed_offset)
        vocab_arr = np.asarray(vocab, dtype=object)
        vocab_set = set(vocab)

        sents = [s for s in corpus.sentences if len(s) > self.window]
        order = rng.permutation(len(sents))

        X, y = [], []
        for si in order:
            sent = sents[si]
            for i in range(self.window, len(sent)):
                true_tok = sent[i]
                if true_tok not in vocab_set:
                    continue
                cvec = self._context_vector(sent, i)
                tvec = self.f._token_vector(true_tok)
                if tvec.sum() <= 0:
                    continue     # undecodable target; nothing to rank toward
                X.append(np.concatenate([cvec, tvec]))
                y.append(1.0)
                negs = vocab_arr[rng.choice(len(vocab_arr), size=self.n_negatives,
                                            p=p)]
                for neg in negs:
                    if neg == true_tok:
                        continue
                    X.append(np.concatenate([cvec, self.f._token_vector(neg)]))
                    y.append(0.0)
            if len(X) >= max_positions * (1 + self.n_negatives):
                break
        if not X:
            raise RuntimeError("no training pairs; check vocabulary coverage")
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    # -- fit ---------------------------------------------------------------

    def fit(self, corpus: Corpus, vocab: list[str], max_positions: int = 1500,
            verbose: bool = True):
        self.feature_names_ = self._names()
        X, y = self.build(corpus, vocab, max_positions)
        n_feat = X.shape[1]
        cand_idx = set(range(self.cand_offset_, n_feat))

        self.model_ = MembershipRuleRegressor(
            max_rules=self.max_rules, max_order=self.max_order,
            # Every rule must touch the candidate, else it cannot change a ranking.
            must_include=cand_idx, order_quota=self.order_quota, beam=self.beam,
            # Context features have exactly zero marginal lift here (they fire
            # identically on a position's positive and its negatives), so they must be
            # force-seeded or no ctx-x-cand interaction is ever generated.
            seed_features=set(range(self.cand_offset_)),
        ).fit(X, y, self.feature_names_)

        if verbose:
            pos = int(y.sum())
            print(f"  joint ranker: {len(y)} pairs ({pos} positive, "
                  f"1:{self.n_negatives} neg), {n_feat} features, "
                  f"{len(self.model_.rules_)} rules "
                  f"by order {self.model_.order_histogram()}")
        return self

    # -- scoring / evaluation ---------------------------------------------

    def score(self, context_vec: np.ndarray, cand_vecs: np.ndarray) -> np.ndarray:
        """Score a shared context against many candidate vectors at once."""
        n = cand_vecs.shape[0]
        X = np.hstack([np.repeat(context_vec[None, :], n, axis=0), cand_vecs])
        return self.model_.predict(X)

    def evaluate(self, corpus: Corpus, vocab: list[str], n_eval: int = 400,
                 candidate_set: int = 20, seed_offset: int = 991,
                 unigram_baseline: bool = True) -> list[RankingResult]:
        """Rank the true next token against frequency-sampled distractors."""
        rng = np.random.default_rng(self.seed + seed_offset)
        _, p = self._sampler(corpus, vocab)
        vocab_arr = np.asarray(vocab, dtype=object)
        vocab_set = set(vocab)
        freq = {v: corpus.counts.get(v, 0) for v in vocab}

        sents = [s for s in corpus.sentences if len(s) > self.window]
        order = rng.permutation(len(sents))

        ranks_model: list[int] = []
        ranks_unigram: list[int] = []
        for si in order:
            sent = sents[si]
            for i in range(self.window, len(sent)):
                true_tok = sent[i]
                if true_tok not in vocab_set:
                    continue
                tvec = self.f._token_vector(true_tok)
                if tvec.sum() <= 0:
                    continue
                distract = [w for w in vocab_arr[rng.choice(
                    len(vocab_arr), size=candidate_set * 2, p=p)]
                    if w != true_tok][: candidate_set - 1]
                cands = [true_tok] + list(distract)
                cvecs = np.vstack([self.f._token_vector(w) for w in cands])

                scores = self.score(self._context_vector(sent, i), cvecs)
                # Rank of the true token (index 0). Ties broken pessimistically --
                # counting tied candidates as ahead -- so a model that outputs a
                # constant scores at chance rather than perfectly.
                ranks_model.append(1 + int((scores[1:] >= scores[0]).sum()))

                if unigram_baseline:
                    u = np.array([freq[w] for w in cands], dtype=float)
                    ranks_unigram.append(1 + int((u[1:] >= u[0]).sum()))

                if len(ranks_model) >= n_eval:
                    break
            if len(ranks_model) >= n_eval:
                break

        out = [_summarise("joint rules", ranks_model, candidate_set,
                          len(self.model_.rules_))]
        if unigram_baseline:
            out.append(_summarise("unigram frequency", ranks_unigram,
                                  candidate_set))
        return out

    def render_rules(self, k: int = 12) -> str:
        return self.model_.render("P(next)", k=k)


def _summarise(name: str, ranks: list[int], candidate_set: int,
               n_rules: int = 0) -> RankingResult:
    r = np.asarray(ranks, dtype=float)
    return RankingResult(
        name=name, mrr=float((1.0 / r).mean()),
        hits1=float((r <= 1).mean()), hits5=float((r <= 5).mean()),
        hits10=float((r <= 10).mean()),
        n_eval=len(ranks), candidate_set=candidate_set, n_rules=n_rules,
    )


def marginal_pipeline_baseline(seq_model, corpus: Corpus, vocab: list[str],
                               window: int, n_eval: int = 400,
                               candidate_set: int = 20, seed: int = 42,
                               seed_offset: int = 991) -> RankingResult:
    """The old marginal-then-match pipeline, on the *same* ranking task.

    This is the A/B that matters: same evaluation, same distractors, so the only
    difference is marginal-then-match versus joint scoring.
    """
    from .decode import LexemeAtlas

    rng = np.random.default_rng(seed + seed_offset)
    w = np.array([corpus.counts.get(v, 1) for v in vocab], dtype=float)
    p = w / w.sum()
    vocab_arr = np.asarray(vocab, dtype=object)
    vocab_set = set(vocab)

    atlas = LexemeAtlas(seq_model.emb, seq_model.level, vocabulary=vocab,
                        vectorizer=seq_model._token_vector,
                        names=seq_model._output_names())
    index = {word: i for i, word in enumerate(atlas.words)}

    sents = [s for s in corpus.sentences if len(s) > window]
    order = rng.permutation(len(sents))
    ranks: list[int] = []
    for si in order:
        sent = sents[si]
        for i in range(window, len(sent)):
            true_tok = sent[i]
            if true_tok not in vocab_set or true_tok not in index:
                continue
            distract = [x for x in vocab_arr[rng.choice(len(vocab_arr),
                                                        size=candidate_set * 2, p=p)]
                        if x != true_tok and x in index][: candidate_set - 1]
            cands = [true_tok] + list(distract)
            target = seq_model.predict_next(sent[max(0, i - window):i])
            all_scores = atlas.score(target, metric="coverage")
            s = np.array([all_scores[index[c]] for c in cands])
            ranks.append(1 + int((s[1:] >= s[0]).sum()))
            if len(ranks) >= n_eval:
                break
        if len(ranks) >= n_eval:
            break
    return _summarise("marginal then match", ranks, candidate_set)
