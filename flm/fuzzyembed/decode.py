"""The fuzzy decoder -- membership vector back to a surface word.

This is Zadeh's **linguistic approximation**: the retranslation step that closes the
computing-with-words loop (explicitation -> fuzzy inference -> retranslation). The
sequence model predicts a fuzzy set over named hierarchy nodes; the decoder finds the
lexeme whose own membership vector best matches it, under the fuzzy Jaccard of
``similarity.py``.

Two things fall out of doing it this way rather than with a softmax over logits.

**Temperature is a Zadeh hedge.** Sharpening the candidate distribution is exactly
concentration (``mu ** e``, e > 1) and flattening it is dilation (``e < 1``). There
is no separate temperature parameter bolted on -- the knob is the hedge exponent
already in the algebra, and ``e = 2`` is literally "*very* like the prediction".

**Every choice is auditable.** A decode step reports which named dimensions drove
it and which lexemes competed, with degrees. There is no vocabulary-sized opaque
distribution to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .embedder import FuzzyEmbedder
from .similarity import fuzzy_jaccard


@dataclass
class DecodeStep:
    """One decode decision, with its justification."""

    chosen: str
    degree: float
    candidates: list[tuple[str, float]] = field(default_factory=list)
    driving_dims: list[tuple[str, float]] = field(default_factory=list)

    def render(self) -> str:
        cands = ", ".join(f"{w}({d:.2f})" for w, d in self.candidates[:5])
        dims = ", ".join(f"{n}={v:.2f}" for n, v in self.driving_dims[:3])
        return f"{self.chosen:<14} [{dims}]  <- {cands}"


class LexemeAtlas:
    """Precomputed level-``L`` membership vectors for every vocabulary word.

    Built once. Decoding is then a similarity scan over this matrix, which is the
    fuzzy analogue of an output embedding table -- except every column is named.
    """

    def __init__(self, embedder: FuzzyEmbedder, level: int,
                 vocabulary: list[str] | None = None, verbose: bool = False,
                 vectorizer=None, names: list[str] | None = None):
        """``vectorizer``/``names`` let the atlas live in the sequence model's *joint*
        semantic+syntactic space rather than semantics alone.

        This is what makes function words decodable. With a semantics-only atlas,
        ``the`` / ``of`` / ``was`` have all-zero vectors and were skipped outright, so
        generation could never emit them and the output could not be grammatical.
        """
        self.emb = embedder
        self.level = level
        self.vectorizer = vectorizer or (lambda w: embedder.embed(w, level))
        self.names = names
        vocab = vocabulary if vocabulary is not None else embedder.lexicon.vocabulary
        rows, words = [], []
        for w in vocab:
            v = self.vectorizer(w)
            if v.sum() <= 0:
                continue      # no membership anywhere; nothing to match against
            rows.append(v)
            words.append(w)
        if not rows:
            raise RuntimeError("no decodable vocabulary; check hierarchy coverage")
        self.words = words
        self.matrix = np.vstack(rows)
        if verbose:
            print(f"  lexeme atlas: {len(words)} decodable words "
                  f"x {self.matrix.shape[1]} dims (L{level})")

    def score(self, target: np.ndarray, metric: str = "coverage") -> np.ndarray:
        """Score every lexeme against a predicted membership vector. Vectorised.

        ``metric="coverage"`` (default) is **asymmetric**: how much of this word's own
        membership mass sits in categories the prediction called for,
        ``sum_c min(t_c, w_c) / sum_c w_c``.

        ``metric="jaccard"`` is the symmetric fuzzy Jaccard of ``similarity.py``, and
        it is the *wrong* asymmetry for decoding -- kept only for comparison. The
        predicted vector is a marginal (a degree per category, spread across many),
        whereas a lexeme's vector is one specific peaked pattern. Symmetric Jaccard
        therefore rewards words whose pattern resembles the *marginal distribution*,
        i.e. bland words with membership spread thinly everywhere, and penalises the
        pure noun the prediction actually asked for: with ``t[OPEN_NOUN]=0.44``,
        ``cat`` (mass 1.0 concentrated in OPEN_NOUN) scored below ``jolly`` and
        ``fourth`` purely for being less spread out. Normalising by the word's own
        mass removes that bias.
        """
        t = target[None, :]
        if metric == "jaccard":
            num = np.minimum(self.matrix, t).sum(axis=1)
            den = np.maximum(self.matrix, t).sum(axis=1)
            return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        if metric != "coverage":
            raise ValueError(f"unknown metric {metric!r}")
        num = np.minimum(self.matrix, t).sum(axis=1)
        mass = self.matrix.sum(axis=1)
        return np.where(mass > 0, num / np.maximum(mass, 1e-12), 0.0)


class FuzzyDecoder:
    """Turns a predicted membership vector into a word."""

    def __init__(self, atlas: LexemeAtlas, hedge: float = 2.0, top_k: int = 10,
                 seed: int = 0, metric: str = "coverage"):
        """``hedge`` is the concentration exponent -- the temperature analogue.

        ``hedge > 1`` concentrates (more decisive, "*very* like the prediction");
        ``hedge < 1`` dilates (more diverse); ``hedge = 1`` samples proportional to
        raw similarity.
        """
        self.atlas = atlas
        self.metric = metric
        self.hedge = hedge
        self.top_k = top_k
        self.rng = np.random.default_rng(seed)

    def decode(self, target: np.ndarray, sample: bool = True,
               exclude: set[str] | None = None) -> DecodeStep:
        scores = self.atlas.score(target, metric=self.metric)
        if exclude:
            for i, w in enumerate(self.atlas.words):
                if w in exclude:
                    scores[i] = 0.0

        k = min(self.top_k, len(scores))
        top = np.argsort(scores)[::-1][:k]
        cands = [(self.atlas.words[i], float(scores[i])) for i in top
                 if scores[i] > 0]
        if not cands:
            return DecodeStep("<none>", 0.0)

        if sample:
            # Concentration/dilation *is* the temperature. Normalising afterwards
            # turns the hedged membership degrees into a sampling distribution;
            # the hedge shapes it, it is not a second mechanism.
            w = np.array([d for _, d in cands]) ** self.hedge
            total = w.sum()
            idx = (self.rng.choice(len(cands), p=w / total) if total > 0
                   else int(np.argmax([d for _, d in cands])))
        else:
            idx = 0

        if self.atlas.names is not None:
            labels = self.atlas.names
        else:
            h = self.atlas.emb.h
            labels = [h.name(k) for k in h.level_keys(self.atlas.level)]
        dims = [(labels[i], float(target[i]))
                for i in np.argsort(target)[::-1][:4] if target[i] > 0]
        return DecodeStep(cands[idx][0], cands[idx][1], cands, dims)


def generate(seq_model, decoder: FuzzyDecoder, prompt: list[str], n_tokens: int = 8,
             sample: bool = True, avoid_repeats: bool = True
             ) -> tuple[list[str], list[DecodeStep]]:
    """Roll the sequence model forward, decoding each step.

    Output is a *semantic-class* walk, not grammatical text: function words carry no
    hierarchy membership and so can never be emitted. Read it as "what kind of thing
    comes next", which is what this stack currently models.
    """
    out = list(prompt)
    steps: list[DecodeStep] = []
    for _ in range(n_tokens):
        target = seq_model.predict_next(out)
        # Blocking the immediately-preceding tokens is a crude repetition guard.
        # Without it a self-similar prediction is a fixed point and generation
        # collapses to one word repeated.
        exclude = set(out[-3:]) if avoid_repeats else None
        step = decoder.decode(target, sample=sample, exclude=exclude)
        if step.chosen == "<none>":
            break
        steps.append(step)
        out.append(step.chosen)
    return out, steps


def render_generation(prompt: list[str], steps: list[DecodeStep]) -> str:
    lines = [f"prompt: {' '.join(prompt)!r}", "decode trace:"]
    lines.extend("  " + s.render() for s in steps)
    lines.append("result: " + " ".join(prompt + [s.chosen for s in steps]))
    return "\n".join(lines)
