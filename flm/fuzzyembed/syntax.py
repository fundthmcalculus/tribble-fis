"""A fuzzy model of syntax, to sit beside the fuzzy model of semantics.

Why this module exists
----------------------
The first sequence model (semantics only) scored **balanced accuracy 0.528 against
chance 0.500** -- essentially no skill. The diagnosis was structural rather than a
hyperparameter: what predicts the next word is mostly *syntax and function words*,
and the semantic embedding discards both by construction. Function words carry no
WordNet membership, so ``the``, ``of``, ``was`` are invisible to it. A two-token
window of supersenses can say "an animal was just mentioned"; it cannot say "a
determiner was just used, so a noun is due".

So: a second, small, **named** feature space of closed-class syntactic categories,
concatenated onto the semantic one. Interpretability survives because these
dimensions are named too -- a rule can read ``IF prev1[DETERMINER] is High THEN
next[OPEN_NOUN] is High``, which is a checkable claim about English.

Syntax is genuinely fuzzy here, which is a pleasant confirmation rather than a
convenience. ``to`` is both a preposition and an infinitive marker; ``that`` is a
determiner, a pronoun, and a complementiser; ``her`` is a pronoun and a possessive;
``no`` is a determiner and a negator. Those readings are represented as simultaneous
partial memberships, not forced into one label.

Membership convention
---------------------
Closed-class memberships are **possibilistic**: a word gets degree 1.0 in every
category it *can* belong to, and they are not normalised to sum to 1. This matches
the convention in ``lexical.py`` -- these are degrees of possibility, not a
probability distribution over readings, and disambiguating them would need a tagger
this module deliberately does not depend on. Open-class memberships *are* graded, by
the fraction of a lemma's WordNet senses falling in each part of speech.
"""

from __future__ import annotations

import numpy as np

from .coverage import FUNCTION_WORDS

# Curated closed-class categories. Overlaps are intentional -- see module docstring.
CLOSED_CLASS: dict[str, frozenset[str]] = {
    "DETERMINER": frozenset("""the a an this that these those every each some any no
        all both either neither another such""".split()),
    "PRONOUN": frozenset("""i you he she it we they me him her us them who whom
        myself yourself himself herself itself ourselves themselves someone something
        anyone anything everyone everything nobody nothing one ones somebody anybody
        everybody none""".split()),
    "POSSESSIVE": frozenset("""my your his her its our their mine yours hers ours
        theirs whose""".split()),
    "AUXILIARY": frozenset("""is am are was were be been being do does did done have
        has had having will would shall should can could may might must ought""".split()),
    "PREPOSITION": frozenset("""of in on at to from by with without within into onto
        upon about above below under over between among through during before after
        since until against for beside toward towards across behind near round""".split()),
    "CONJUNCTION": frozenset("""and or but nor yet so because if then than while when
        where as although though unless whether that""".split()),
    "NEGATOR": frozenset("not no never none cannot nothing nobody".split()),
    "QUANTIFIER": frozenset("""many much more most less least few several enough
        little lot lots plenty""".split()),
    "WH_WORD": frozenset("who what where when why how which whose whom".split()),
    "INTENSIFIER": frozenset("""very too quite rather somewhat extremely really just
        only even still already almost nearly so""".split()),
    "INFINITIVE_TO": frozenset(["to"]),
}

OPEN_CLASS = ("OPEN_NOUN", "OPEN_VERB", "OPEN_ADJ", "OPEN_ADV")

#: Marker for a padded / sentence-initial position, so "nothing precedes this" is a
#: representable state rather than an all-zero vector indistinguishable from "unknown
#: word".
BOUNDARY = "BOUNDARY"

SYNTAX_CATEGORIES: tuple[str, ...] = (
    tuple(CLOSED_CLASS) + OPEN_CLASS + (BOUNDARY,)
)

_POS_TO_CATEGORY = {"n": "OPEN_NOUN", "v": "OPEN_VERB",
                    "a": "OPEN_ADJ", "s": "OPEN_ADJ", "r": "OPEN_ADV"}


class SyntaxTagger:
    """Maps a token to a fuzzy membership vector over named syntactic categories."""

    def __init__(self, lemma_synsets: dict[str, list] | None = None):
        """``lemma_synsets`` supplies the open-class part (from the hierarchy build).

        Without it only closed-class categories fire, which still helps: the closed
        class is where the predictive signal for word order lives.
        """
        self.lemma_synsets = lemma_synsets or {}
        self.index = {c: i for i, c in enumerate(SYNTAX_CATEGORIES)}
        self._cache: dict[str, np.ndarray] = {}

    @property
    def width(self) -> int:
        return len(SYNTAX_CATEGORIES)

    def zeros(self) -> np.ndarray:
        return np.zeros(self.width, dtype=np.float32)

    def tag(self, token: str) -> np.ndarray:
        if token in self._cache:
            return self._cache[token]
        vec = self.zeros()

        if not token:
            vec[self.index[BOUNDARY]] = 1.0
            self._cache[token] = vec
            return vec

        for cat, members in CLOSED_CLASS.items():
            if token in members:
                vec[self.index[cat]] = 1.0

        # Open-class marking must respect the same closed-class filter
        # SenseAssigner uses. Without this, closed-class words leaked in with a pure
        # OPEN_NOUN=1.0 and nothing else -- and since the coverage decode metric
        # normalises by a word's own mass, those single-coordinate vectors topped
        # every OPEN_NOUN retrieval ("somebody", "o", "t" outranking "cat").
        synsets = None if token in FUNCTION_WORDS else self.lemma_synsets.get(token)
        if synsets:
            # Graded by sense distribution: a word that is 80% noun senses and 20%
            # verb senses says so, rather than being forced to one tag.
            counts: dict[str, int] = {}
            for syn in synsets:
                cat = _POS_TO_CATEGORY.get(syn.pos())
                if cat:
                    counts[cat] = counts.get(cat, 0) + 1
            total = sum(counts.values())
            for cat, n in counts.items():
                vec[self.index[cat]] = n / total

        self._cache[token] = vec
        return vec

    def names(self) -> list[str]:
        return list(SYNTAX_CATEGORIES)

    def explain(self, token: str) -> str:
        vec = self.tag(token)
        live = [(SYNTAX_CATEGORIES[i], float(v)) for i, v in enumerate(vec) if v > 0]
        live.sort(key=lambda kv: -kv[1])
        if not live:
            return f"{token!r}: no syntactic category (unknown open-class word)"
        return f"{token!r}: " + ", ".join(f"{c}={v:.2f}" for c, v in live)
