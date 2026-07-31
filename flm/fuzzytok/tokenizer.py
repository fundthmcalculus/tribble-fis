"""A fuzzy tokenizer: surface string -> graded set over units.

Standard tokenizers commit to one segmentation. BPE merges greedily (Sennrich et al.,
2016), WordPiece maximises likelihood (Schuster & Nakajima, 2012), and the unigram-LM
tokenizer (Kudo, 2018) can enumerate segmentations with probabilities but *samples* one
per step for subword regularisation.

This keeps every plausible segmentation simultaneously, with **membership degrees**, and
never samples. Two consequences a hard tokenizer cannot offer: misspellings get partial
membership in the correct unit natively rather than through a bolted-on corrector, and the
ambiguity is *reportable* -- "this was read as `un + happy` to degree 0.72" is auditable.

Vocabulary design is measured, not guessed (``../LOG.md`` E20). On the children's corpus,
**54 word types carry 50% of all tokens** and 415 carry 80%. The head is therefore tiny and
idiosyncratic -- no decomposition helps with ``the``, ``of``, ``said`` -- while the tail is
large and morphologically regular. Hence a **hybrid vocabulary**: whole words for the head,
affixes and character n-grams for the tail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Common English affixes, kept deliberately short and hand-listed. A learned BPE merge
# table would cover more, but these are *nameable* -- a rule can say `-ly` or `un-`, which
# is the whole point of an interpretable unit inventory. Ordered longest-first so greedy
# stripping prefers the more specific affix.
SUFFIXES = tuple(sorted((
    "ness", "ment", "tion", "sion", "able", "ible", "ally", "ical", "ious", "eous",
    "ing", "ers", "est", "ies", "ful", "ish", "ist", "ity", "ive", "ize", "ise",
    "ly", "ed", "er", "es", "en", "al", "ic", "ty", "ous",
    "s", "y",
), key=len, reverse=True))

PREFIXES = tuple(sorted((
    "under", "over", "about", "back",
    "un", "re", "in", "im", "dis", "mis", "non", "pre", "sub", "out", "up",
), key=len, reverse=True))


@dataclass
class Unit:
    """One vocabulary unit and how it was produced."""

    text: str
    kind: str          # "word" | "stem" | "prefix" | "suffix" | "ngram" | "unk"
    degree: float

    def label(self) -> str:
        if self.kind == "prefix":
            return f"{self.text}-"
        if self.kind == "suffix":
            return f"-{self.text}"
        return self.text


@dataclass
class Segmentation:
    """One reading of a token, with its overall degree."""

    units: list[Unit]
    degree: float
    source: str = ""

    def render(self) -> str:
        return (" + ".join(u.label() for u in self.units)
                + f"  [{self.degree:.2f}{', ' + self.source if self.source else ''}]")


@dataclass
class FuzzyTokenization:
    """The fuzzy set of readings for one surface token."""

    surface: str
    readings: list[Segmentation] = field(default_factory=list)

    def unit_degrees(self) -> dict[str, float]:
        """Degree per unit label, aggregated by t-conorm (max) across readings.

        Max, not sum: two readings both containing ``happy`` is a *disjunction* ("this
        token is about happy"), not accumulated evidence. Same convention as
        ``fuzzyembed/senses.py``, and it keeps degrees in [0, 1] without renormalising --
        these are memberships, not probabilities.
        """
        out: dict[str, float] = {}
        for seg in self.readings:
            for u in seg.units:
                d = min(seg.degree, u.degree)          # t-norm along a reading
                out[u.label()] = max(out.get(u.label(), 0.0), d)
        return out

    def best(self) -> Segmentation | None:
        return self.readings[0] if self.readings else None

    def render(self) -> str:
        lines = [f"{self.surface!r}"]
        lines.extend("  " + s.render() for s in self.readings)
        return "\n".join(lines)


class FuzzyTokenizer:
    """Hybrid whole-word + affix + character-n-gram tokenizer with graded output."""

    def __init__(self, head_words: list[str], stems: set[str] | None = None,
                 max_readings: int = 4, ngram_n: int = 3,
                 lexicon=None, min_degree: float = 0.25):
        """``head_words`` are the frequency-ordered whole-word units (the Zipf head).

        ``lexicon`` is an optional ``fuzzyembed.lexical.FuzzyLexicon``; when supplied, an
        out-of-vocabulary surface form gets a *graded* reading through fuzzy lexical
        access, which is what makes misspelling robustness intrinsic to tokenization
        rather than a preprocessing step.
        """
        self.head = list(head_words)
        self.head_set = set(self.head)
        self.stems = set(stems) if stems else set(self.head)
        self.max_readings = max_readings
        self.ngram_n = ngram_n
        self.lexicon = lexicon
        self.min_degree = min_degree
        self._cache: dict[str, FuzzyTokenization] = {}

    # -- vocabulary induction ---------------------------------------------

    @classmethod
    def from_corpus(cls, corpus, head_size: int = 500, stem_size: int = 3000,
                    **kwargs) -> FuzzyTokenizer:
        """Induce the hybrid vocabulary from a corpus.

        ``head_size=500`` covers ~82% of token mass on the children's corpus; the
        measured Zipf curve is what sets the default rather than a round number.
        """
        vocab = corpus.vocabulary
        return cls(head_words=vocab[:head_size],
                   stems=set(vocab[:stem_size]), **kwargs)

    def unit_inventory(self) -> list[str]:
        """Every nameable unit. Small on purpose -- this is the 'simple vocabulary'."""
        return ([w for w in self.head]
                + [f"{p}-" for p in PREFIXES]
                + [f"-{s}" for s in SUFFIXES])

    # -- segmentation -----------------------------------------------------

    def _affix_readings(self, token: str) -> list[Segmentation]:
        """Strip one prefix and/or one suffix, keeping the reading only if a real stem
        remains. Depth 1 each side: deeper stripping produces spurious stems far more
        often than real ones on a vocabulary this size."""
        out: list[Segmentation] = []
        for pre in ("",) + PREFIXES:
            if pre and not token.startswith(pre):
                continue
            body = token[len(pre):]
            for suf in ("",) + SUFFIXES:
                if suf and not body.endswith(suf):
                    continue
                stem = body[: len(body) - len(suf)] if suf else body
                if len(stem) < 3 or not (pre or suf):
                    continue
                # A stem in the vocabulary is strong evidence; one that merely looks
                # word-like is weak. Restoring a dropped 'e' ("hoping" -> "hope") is the
                # one orthographic repair common enough to be worth special-casing.
                # Both the bare stem and the e-restored stem can be real words, so
                # emit whichever apply as *competing graded readings* rather than
                # letting the first match win. Short-circuiting here made "hoping"
                # resolve only to `hop + -ing` (because "hop" is in the vocabulary)
                # and never offer `hope + -ing` at all -- exactly the kind of
                # premature commitment a fuzzy tokenizer exists to avoid.
                variants: list[tuple[str, float, str]] = []
                if stem in self.stems:
                    variants.append((stem, 0.9, "stem in vocabulary"))
                if stem + "e" in self.stems:
                    variants.append((stem + "e", 0.85, "stem+e in vocabulary"))
                if not variants and len(stem) >= 4:
                    variants.append((stem, 0.35, "stem shape only"))
                for stem_v, deg, why in variants:
                    units = []
                    if pre:
                        units.append(Unit(pre, "prefix", 0.9))
                    units.append(Unit(stem_v, "stem", deg))
                    if suf:
                        units.append(Unit(suf, "suffix", 0.9))
                    out.append(Segmentation(units, deg, why))
        out.sort(key=lambda s: -s.degree)
        return out

    def _ngram_reading(self, token: str) -> Segmentation:
        """Last resort: character n-grams. Always available, never confident.

        Degree is capped low so an n-gram reading loses to any lexical or affix reading;
        its role is to guarantee *some* representation for a wholly unknown string rather
        than to compete."""
        n = self.ngram_n
        pad = f"^{token}$"
        grams = [pad[i:i + n] for i in range(max(len(pad) - n + 1, 1))]
        return Segmentation([Unit(g, "ngram", 0.3) for g in grams], 0.3,
                            "character n-grams")

    def tokenize(self, surface: str) -> FuzzyTokenization:
        if surface in self._cache:
            return self._cache[surface]
        # Case-fold for vocabulary lookup while keeping the surface form for the
        # caller. Without this a capitalised in-vocabulary word missed the head set
        # entirely and fell through to fuzzy lexical access -- "Margery" matched
        # itself at 0.73 instead of 1.0, and could be outranked by an unrelated
        # neighbour. Orthographic case is a *parameter* (see params.Shape), not a
        # reason to fail lookup.
        token = surface.lower()
        readings: list[Segmentation] = []

        # 1. Whole-word head unit. Exact and unambiguous, so degree 1.
        if token in self.head_set:
            readings.append(Segmentation([Unit(token, "word", 1.0)], 1.0, "head word"))

        # 2. Morphological readings.
        readings.extend(self._affix_readings(token))

        # 3. Fuzzy lexical access for out-of-vocabulary forms -- this is where a
        #    misspelling acquires partial membership in the correct unit.
        if self.lexicon is not None and token not in self.head_set:
            for m in self.lexicon.match(token)[:2]:
                if m.lexeme != token and m.degree >= self.min_degree:
                    readings.append(Segmentation(
                        [Unit(m.lexeme, "word", m.degree)], m.degree,
                        "fuzzy lexical access"))

        readings = [r for r in readings if r.degree >= self.min_degree]
        readings.sort(key=lambda s: -s.degree)
        readings = readings[: self.max_readings]
        if not readings:
            readings = [self._ngram_reading(token)]

        out = FuzzyTokenization(surface, readings)
        self._cache[surface] = out
        return out

    def tokenize_all(self, tokens: list[str]) -> list[FuzzyTokenization]:
        return [self.tokenize(t) for t in tokens]

    # -- diagnostics ------------------------------------------------------

    def coverage(self, corpus, max_tokens: int = 50_000) -> dict:
        """How much of a corpus each unit kind accounts for.

        Reports token mass by best-reading kind, which is the number that decides whether
        the hybrid split is sized correctly.
        """
        kinds: dict[str, int] = {}
        n = 0
        for sent in corpus.sentences:
            for tok in sent:
                best = self.tokenize(tok).best()
                kind = best.units[0].kind if best else "unk"
                if best and len(best.units) > 1:
                    kind = "decomposed"
                kinds[kind] = kinds.get(kind, 0) + 1
                n += 1
                if n >= max_tokens:
                    break
            if n >= max_tokens:
                break
        return {"n_tokens": n, "units": len(self.unit_inventory()),
                "by_kind": {k: v / n for k, v in sorted(
                    kinds.items(), key=lambda kv: -kv[1])}}
