"""Fuzzy language generation, built on the joint ranker.

The marginal-then-match decoder in ``decode.py`` generates category-appropriate word
salad, because it predicts each named dimension independently and nothing ties the
result to one actual word. The joint ranker (``joint.py``) scores ``(context,
candidate)`` pairs and does beat a unigram baseline, so it -- not the marginal
pipeline -- is the thing to generate from.

Two design points.

**Scores become a real distribution.** Normalising the joint scores over the whole
vocabulary gives ``p(w | context)``, which makes **perplexity** available -- the
standard language-modelling metric -- and therefore makes this directly comparable to
an n-gram LM or GPT-2 on identical held-out text. Without that step there is no
common yardstick against any conventional LM.

**Temperature is a Zadeh hedge.** Sharpening is concentration (``mu ** e``, e > 1),
flattening is dilation. No separate temperature mechanism: ``e = 2`` is literally
"*very* like the prediction". Perplexity is reported at ``e = 1`` (the unhedged
distribution), since hedging is a decoding choice and would otherwise make the metric
depend on a knob.

**Efficiency.** A generation step scores the entire vocabulary, so the naive route --
build an ``(n_vocab, n_features)`` matrix and re-evaluate every rule -- repeats work
that does not depend on the context. Each rule's antecedent splits into a context part
and a candidate part, and under the product t-norm the firing factorises::

    firing(rule, w) = prod(ctx values)  x  prod(cand values for w)
                      \\_____________/     \\____________________/
                       scalar per step      precomputed per (rule, w)

so the candidate half is computed **once** into an ``(n_vocab, n_rules)`` matrix and a
step is one scalar-vector scaling plus two reductions. See ``_precompute``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .corpus import Corpus


@dataclass
class GenStep:
    """One generation step, with the named rules that drove it."""

    chosen: str
    prob: float
    top: list[tuple[str, float]] = field(default_factory=list)
    firing_rules: list[tuple[str, float]] = field(default_factory=list)

    def render(self) -> str:
        alts = ", ".join(f"{w}({p:.3f})" for w, p in self.top[:5])
        why = "; ".join(f"{r} [{f:.2f}]" for r, f in self.firing_rules[:2])
        return f"{self.chosen:<14} p={self.prob:.3f}  alts: {alts}" + (
            f"\n{'':<16}why: {why}" if why else "")


class FuzzyGenerator:
    """Turns a fitted ``JointNextTokenRanker`` into a language model."""

    def __init__(self, ranker, vocab: list[str], counts: dict[str, int] | None = None,
                 seed: int = 0):
        self.r = ranker
        self.f = ranker.f
        self.window = ranker.window
        self.rng = np.random.default_rng(seed)
        self._precompute(vocab)
        self._set_noise_prior(counts)

    def _set_noise_prior(self, counts: dict[str, int] | None) -> None:
        """The noise distribution *q* the ranker's negatives were sampled from.

        Must match ``JointNextTokenRanker._sampler`` -- frequency-weighted -- or the
        NCE inversion in ``distribution`` corrects with the wrong prior.
        """
        if counts:
            w = np.array([max(counts.get(t, 1), 1) for t in self.words], dtype=float)
        else:
            w = np.ones(len(self.words))
        self.noise_prior = w / w.sum()

    # -- setup -------------------------------------------------------------

    def _precompute(self, vocab: list[str]) -> None:
        """Cache the candidate half of every rule's firing, once."""
        model = self.r.model_
        if model is None:
            raise RuntimeError("fit the ranker first")
        offset = self.r.cand_offset_

        words, rows = [], []
        for w in vocab:
            v = self.f._token_vector(w)
            if v.sum() > 0:
                words.append(w)
                rows.append(v)
        if not rows:
            raise RuntimeError("no decodable vocabulary")
        self.words = words
        self.index = {w: i for i, w in enumerate(words)}
        cand = np.vstack(rows)                       # (n_vocab, cand_width)

        self.rules = model.rules_
        self.default = model.default_
        n_vocab, n_rules = len(words), len(self.rules)
        # Candidate-side firing per (word, rule), and which context columns each rule
        # needs. Rules with no candidate feature would be constant across words; the
        # ranker's `must_include` already excludes them.
        self.cand_part = np.ones((n_vocab, n_rules))
        self.ctx_cols: list[np.ndarray] = []
        self.consequents = np.array([r.consequent for r in self.rules])
        for j, rule in enumerate(self.rules):
            ctx_idx = [f for f in rule.features if f < offset]
            cnd_idx = [f - offset for f in rule.features if f >= offset]
            if cnd_idx:
                self.cand_part[:, j] = cand[:, cnd_idx].prod(axis=1)
            self.ctx_cols.append(np.asarray(ctx_idx, dtype=np.intp))

    # -- scoring -----------------------------------------------------------

    def _context_features(self, tokens: list[str]) -> np.ndarray:
        parts = []
        for lag in range(self.window, 0, -1):
            j = len(tokens) - lag
            parts.append(self.f._token_vector(tokens[j] if j >= 0 else ""))
        return np.concatenate(parts)

    def score_all(self, tokens: list[str]) -> np.ndarray:
        """Joint score for every decodable vocabulary word. One step of generation."""
        ctx = self._context_features(tokens)
        # Scalar context factor per rule; zero-firing rules drop out entirely.
        ctx_factor = np.array([
            ctx[cols].prod() if cols.size else 1.0 for cols in self.ctx_cols])
        live = ctx_factor > 0
        if not live.any():
            return np.full(len(self.words), self.default)

        fire = self.cand_part[:, live] * ctx_factor[live][None, :]
        num = fire @ self.consequents[live]
        den = fire.sum(axis=1)
        eps = 0.05                          # default rule, as in MembershipRuleRegressor
        return np.clip((num + eps * self.default) / (den + eps), 0.0, 1.0)

    def distribution(self, tokens: list[str], hedge: float = 1.0,
                     nce_correct: bool = True) -> np.ndarray:
        """``p(w | context)`` over decodable vocabulary.

        ``hedge`` is the Zadeh concentration exponent -- the temperature analogue.
        Perplexity should always be measured at ``hedge=1``.

        ``nce_correct`` applies the noise-contrastive inversion, and it matters
        enormously. The ranker was trained to answer "is this the true next word rather
        than one of *k* words drawn from the noise distribution *q*?", so its output is

            s  =  p(w|ctx) / (p(w|ctx) + k*q(w))

        Simply normalising ``s`` treats it as if it were ``p`` and is badly wrong: ``s``
        is bounded in [0, 1] and spans under one order of magnitude across candidates,
        while a language model needs ratios of 10^3 between likely and unlikely words.
        Normalising the raw scores gave perplexity 2477 against a uniform floor of 2897
        -- almost no information -- purely from that flatness.

        Solving the NCE relation for ``p`` gives the correct conversion::

            p(w|ctx)  proportional to  q(w) * s / (1 - s)

        The odds ratio ``s/(1-s)`` is unbounded, so it can express the dynamic range a
        distribution needs, and multiplying by the noise prior ``q`` restores the
        unigram frequency information the contrastive objective deliberately factored
        out (the model was never asked to learn it -- negatives were drawn *from* it).
        """
        s = self.score_all(tokens)
        if nce_correct:
            s = np.clip(s, 1e-6, 1.0 - 1e-6)
            w = self.noise_prior * (s / (1.0 - s))
        else:
            w = s
        if hedge != 1.0:
            w = w ** hedge
        total = w.sum()
        if total <= 0:
            return np.full(len(w), 1.0 / len(w))
        return w / total

    # -- generation --------------------------------------------------------

    def generate(self, prompt: list[str], n_tokens: int = 12, hedge: float = 3.0,
                 top_k: int = 20, block_repeat: int = 2,
                 explain: bool = False) -> tuple[list[str], list[GenStep]]:
        out = list(prompt)
        steps: list[GenStep] = []
        for _ in range(n_tokens):
            p = self.distribution(out, hedge=hedge)
            if block_repeat:
                for w in out[-block_repeat:]:
                    i = self.index.get(w)
                    if i is not None:
                        p[i] = 0.0
            k = min(top_k, len(p))
            top_idx = np.argpartition(p, -k)[-k:]
            top_idx = top_idx[np.argsort(p[top_idx])[::-1]]
            q = p[top_idx]
            if q.sum() <= 0:
                break
            q = q / q.sum()
            pick = int(self.rng.choice(len(top_idx), p=q))
            chosen = self.words[top_idx[pick]]

            step = GenStep(chosen, float(q[pick]),
                           [(self.words[i], float(p[i])) for i in top_idx[:5]])
            if explain:
                step.firing_rules = self._why(out, chosen)
            steps.append(step)
            out.append(chosen)
        return out, steps

    def _why(self, tokens: list[str], word: str) -> list[tuple[str, float]]:
        """The strongest-firing named rules behind one choice."""
        ctx = self._context_features(tokens)
        i = self.index.get(word)
        if i is None:
            return []
        scored = []
        for j, rule in enumerate(self.rules):
            cols = self.ctx_cols[j]
            cf = ctx[cols].prod() if cols.size else 1.0
            fire = cf * self.cand_part[i, j]
            if fire > 0:
                scored.append((rule.render("P(next)"), float(fire)))
        scored.sort(key=lambda kv: -kv[1])
        return scored[:3]

    # -- evaluation --------------------------------------------------------

    def perplexity(self, corpus: Corpus, max_positions: int = 2000,
                   seed: int = 7, nce_correct: bool = True) -> dict:
        """Perplexity over held-out positions whose true token is decodable.

        Positions whose gold token has no membership are **skipped, not scored**, and
        the covered fraction is returned alongside. Assigning them some floor
        probability would let the floor set the number; excluding them measures the
        model on what it can actually represent, and ``coverage`` says how much of the
        text that is. Any comparison must be run on the *same* restriction -- see
        ``baselines.py``.
        """
        rng = np.random.default_rng(seed)
        sents = [s for s in corpus.sentences if len(s) > self.window]
        order = rng.permutation(len(sents))
        logs, n, skipped = [], 0, 0
        for si in order:
            sent = sents[si]
            for i in range(self.window, len(sent)):
                gold = sent[i]
                gi = self.index.get(gold)
                if gi is None:
                    skipped += 1
                    continue
                p = self.distribution(sent[:i], nce_correct=nce_correct)
                logs.append(np.log(max(p[gi], 1e-12)))
                n += 1
                if n >= max_positions:
                    break
            if n >= max_positions:
                break
        nll = -float(np.mean(logs))
        return {"perplexity": float(np.exp(nll)), "nll": nll, "n": n,
                "coverage": n / max(n + skipped, 1),
                "vocab": len(self.words)}


def render_generation(prompt: list[str], steps: list[GenStep]) -> str:
    lines = [f"prompt: {' '.join(prompt)!r}"]
    lines.extend("  " + s.render() for s in steps)
    lines.append("  -> " + " ".join(prompt + [s.chosen for s in steps]))
    return "\n".join(lines)
