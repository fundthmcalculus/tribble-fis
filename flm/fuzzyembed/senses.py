"""Stage 2 -- lexeme -> graded membership in hierarchy nodes.

WordNet already gives this as a crisp multiset: a lemma belongs to *k* synsets, and
that *k* is its polysemy. Fuzzifying it means deciding *how much* each sense counts.

Three signals, in decreasing reliability:

1. **Sense frequency priors.** WordNet lemmas carry SemCor tagged-occurrence counts.
   This is a real corpus-estimated prior, which is strictly better than the
   "estimate it yourself from the corpus" the plan proposed -- it comes annotated.
   Note the counts are sparse: many lemmas have all-zero counts, in which case this
   degrades to WordNet's own sense ordering (sense 1 is the lexicographer's most
   central), which is a weaker but non-arbitrary fallback.
2. **Polarity**, from the opinion lexicon. Roget's would have supplied this for free
   through its antonymous opposed pairs (648 Goodness / 649 Badness); WordNet has no
   such pairing, so it has to be attached externally. This is the concrete cost of
   the Roget's -> WordNet substitution.
3. **Context relaxation.** Other tokens in the span vote softly for which supersenses
   are live. Capped at two rounds -- this is word-sense disambiguation, an unsolved
   problem, and unbounded relaxation buys drift rather than accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coverage import FUNCTION_WORDS
from .hierarchy import FuzzyHierarchy


@dataclass
class SenseAssignment:
    """Per-lexeme fuzzy set over leaf hierarchy nodes."""

    lexeme: str
    leaf_degrees: dict[str, float]     # leaf key -> degree
    polarity: float = 0.0              # -1 negative .. +1 positive, 0 neutral

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.leaf_degrees.items(), key=lambda kv: -kv[1])[:k]


class SenseAssigner:
    """Turns lexemes into fuzzy sets over the hierarchy's leaf nodes."""

    def __init__(self, hierarchy: FuzzyHierarchy, lemma_synsets: dict[str, list],
                 wn=None, use_polarity: bool = True):
        if wn is None:
            from nltk.corpus import wordnet as wn  # noqa: PLC0415
        self.h = hierarchy
        self.wn = wn
        self.lemma_synsets = lemma_synsets
        self.polarity = _load_polarity() if use_polarity else {}
        self._cache: dict[str, SenseAssignment] = {}
        self._antonym_cache: dict[str, list[str]] = {}

    def _sense_prior(self, lemma: str, synsets: list) -> np.ndarray:
        """Normalised prior over a lemma's senses."""
        counts = []
        for syn in synsets:
            n = 0
            for lem in syn.lemmas():
                if lem.name().lower() == lemma.lower():
                    n = max(n, lem.count())
            counts.append(n)
        counts = np.asarray(counts, dtype=float)

        if counts.sum() > 0:
            # Add-one smoothing: a zero-count sense of a lemma with counts elsewhere
            # is unlikely, not impossible, and a hard zero would make it unreachable.
            w = counts + 0.5
        else:
            # No tagged counts anywhere -- fall back to WordNet's sense order via a
            # reciprocal-rank prior, which encodes the lexicographer's centrality
            # judgement.
            w = 1.0 / (1.0 + np.arange(len(synsets)))
        return w / w.sum()

    def antonym_keys(self, node_key: str) -> list[str]:
        """Hierarchy keys of a synset's WordNet antonyms, if present in the tree.

        This is what makes negation work: ``not happy`` moves mass to ``unhappy``
        rather than complementing a sparse vector. WordNet records antonymy on
        *lemmas*, not synsets, so every lemma of the synset is consulted.

        Roget's would have given this structurally through its opposed category
        pairs; with WordNet it has to be looked up, and coverage is partial --
        antonymy is well populated for adjectives and thin for nouns.
        """
        if node_key in self._antonym_cache:
            return self._antonym_cache[node_key]
        out: list[str] = []
        if node_key.startswith("wn:"):
            try:
                syn = self.wn.synset(node_key[3:])
                for lem in syn.lemmas():
                    for anti in lem.antonyms():
                        key = f"wn:{anti.synset().name()}"
                        if key in self.h.nodes and key not in out:
                            out.append(key)
            except Exception:  # noqa: BLE001 - malformed/absent synset
                out = []
        self._antonym_cache[node_key] = out
        return out

    def assign(self, lemma: str) -> SenseAssignment | None:
        if lemma in self._cache:
            return self._cache[lemma]
        if lemma in FUNCTION_WORDS:
            # WordNet has junk entries for closed-class forms -- synsets("was")
            # returns WAS = Washington (the state abbreviation), and it carries the
            # top sense prior, so "the rabbit was happy" acquired a strong
            # "administrative district" dimension. Function words carry no
            # hierarchy membership worth representing.
            return None
        synsets = self.lemma_synsets.get(lemma)
        if not synsets:
            return None

        prior = self._sense_prior(lemma, synsets)
        degrees: dict[str, float] = {}
        for syn, p in zip(synsets, prior):
            key = f"wn:{syn.name()}"
            if key not in self.h.nodes:
                continue
            # Max, not sum: two senses mapping to one node is a disjunction ("this
            # word is about X"), not an accumulation of evidence.
            degrees[key] = max(degrees.get(key, 0.0), float(p))

        # Rescale so the most likely sense has degree 1. Memberships are relative
        # to the lexeme's own sense distribution: an unambiguous word should reach
        # full membership in its single sense, not 1/k of it.
        if degrees:
            top = max(degrees.values())
            degrees = {k: v / top for k, v in degrees.items()}

        out = SenseAssignment(lemma, degrees, self.polarity.get(lemma, 0.0))
        self._cache[lemma] = out
        return out

    def context_relax(self, assignments: list[SenseAssignment], rounds: int = 1,
                      strength: float = 0.5) -> list[SenseAssignment]:
        """Reweight senses toward supersenses the rest of the span agrees on.

        Cheap Lesk-flavoured relaxation at the *supersense* (level 2) granularity
        rather than the synset level: supersense agreement is a much denser signal
        than synset overlap, so it survives short spans where synset-level voting
        sees almost no evidence.
        """
        if not assignments or rounds <= 0:
            return assignments

        lex_level = 2
        for _ in range(rounds):
            votes: dict[str, float] = {}
            for a in assignments:
                for key, deg in a.leaf_degrees.items():
                    sup = self.h.project(key, lex_level)
                    votes[sup] = votes.get(sup, 0.0) + deg
            if not votes:
                break
            hi = max(votes.values()) or 1.0

            updated = []
            for a in assignments:
                new = {}
                for key, deg in a.leaf_degrees.items():
                    sup = self.h.project(key, lex_level)
                    # Exclude this token's own vote so a word cannot confirm itself.
                    support = (votes.get(sup, 0.0) - deg) / hi
                    new[key] = float(deg * (1.0 - strength + strength * max(support, 0.0)))
                top = max(new.values()) if new else 0.0
                if top > 0:
                    new = {k: v / top for k, v in new.items()}
                updated.append(SenseAssignment(a.lexeme, new, a.polarity))
            assignments = updated
        return assignments


def _load_polarity() -> dict[str, float]:
    """+1 / -1 per word from the NLTK opinion lexicon (Hu & Liu).

    Supplies the polarity that Roget's opposed pairs would have given structurally.
    Returns empty rather than raising when the corpus is absent -- polarity is an
    enrichment, and the embedding is well-defined without it.
    """
    try:
        from nltk.corpus import opinion_lexicon
        pos = set(opinion_lexicon.positive())
        neg = set(opinion_lexicon.negative())
    except Exception:  # noqa: BLE001 - optional corpus
        return {}
    out = {w: 1.0 for w in pos}
    out.update({w: -1.0 for w in neg})
    return out
