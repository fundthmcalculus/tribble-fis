"""Relational context slots instead of lag positions (E27, option B).

The problem this attacks
------------------------
E26 swept context windows 2 to 32 and found perplexity degrades **monotonically** (355.5 ->
395.2), with 5,111 of 6,060 rules referencing a lag greater than 2 at weak lift
(``IF prev31:QUANTIFIER AND cand:INFINITIVE_TO``, support 24, lift +1.70, against +8.41 for
``IF prev1:DETERMINER AND cand:OPEN_NOUN``). Two things go wrong at once with lag indexing:

1. **Dimensions grow linearly with reach.** Window 32 costs 8,613 columns, so the order-2
   candidate pool goes from ~0.2M pairs to ~2.2M and spurious correlations scale with it.
2. **A lag is not a linguistic relation.** "The token 13 positions back" is a different
   *thing* in every sentence, so it cannot generalise; `prev1` works precisely because
   adjacency is the one lag that reliably means something.

Relational slots decouple reach from dimension. Each slot is a *function* of the left
context -- "the nearest preceding verb", "the governing determiner" -- filled by scanning
back for the first token in a category. Reach is unbounded; dimension is fixed at
``n_slots * base``. A rule then reads ``IF ctx:verb:OPEN_VERB AND cand:noun.person``, which
is a statement about structure rather than about counting backwards.

Honest limitation, stated up front
----------------------------------
**This is not a parse.** Proper dependency relations need a trained parser, and none is
installable in this environment (no spaCy, stanza, or benepar; nltk ships no POS-tagger data
here either). So slots are filled by scanning for the nearest token whose *fuzzy syntactic
category* -- from ``syntax.SyntaxTagger``, already part of the representation -- matches. That
is a shallow heuristic approximation of the relations a parser would give, in the spirit of
"nearest preceding verb is probably the governing verb". It will be wrong across clause
boundaries, on coordination, and wherever English word order is not canonical.

The approximation is worth measuring anyway: if *approximate* relational slots beat exact lag
slots, the mechanism is confirmed and a real parser is the obvious upgrade. If they do not,
adding a parser to the same architecture is unlikely to rescue it, and that is worth knowing
before investing in one.
"""

from __future__ import annotations

import numpy as np

from .joint import JointNextTokenRanker
from .syntax import CLOSED_CLASS

#: Slot definitions: ``(name, predicate over the token's syntax categories)``.
#:
#: `adj1`/`adj2` keep the two lags that E26 showed actually carry signal -- the point is to
#: *add* structure, not to discard the one thing that works. The rest reach back without
#: bound. Ordered so the fixed-lag slots come first, which keeps a window-2 model a strict
#: prefix of this one and makes the comparison a clean addition rather than a substitution.
SLOTS: tuple[str, ...] = (
    "adj1",        # immediately preceding token (lag 1)
    "adj2",        # lag 2
    "verb",        # nearest preceding verb or auxiliary  ~ governing predicate
    "subj",        # nearest preceding noun/pronoun before that verb  ~ subject
    "noun",        # nearest preceding noun  ~ head of the current phrase
    "prep",        # nearest preceding preposition  ~ attachment site
    "det",         # nearest preceding determiner/possessive  ~ current NP opener
    "start",       # first token of the sentence  ~ clause type (question, imperative)
)

_VERBISH = ("OPEN_VERB", "AUXILIARY")
_NOUNISH = ("OPEN_NOUN",)
_SUBJISH = ("OPEN_NOUN", "PRONOUN")
_DETISH = ("DETERMINER", "POSSESSIVE")


class RelationalNextTokenRanker(JointNextTokenRanker):
    """Joint ranker whose context slots are shallow relations, not lag offsets.

    Overrides only the three slot methods factored out of ``JointNextTokenRanker``, so
    training, evaluation, and generation all pick up the new context definition from one
    place. ``window`` is retained as the *lookback bound* for the unbounded slots -- a cap on
    how far the backward scan goes, defaulting to the whole sentence.
    """

    def __init__(self, featuriser, lookback: int = 64, **kwargs):
        kwargs.setdefault("window", 2)
        super().__init__(featuriser, **kwargs)
        self.lookback = lookback
        self._tagger = None
        self._cat_names: list[str] = []
        # Slot filling touches every token of every context, so the category lookup is on the
        # hottest path in the whole pipeline; cache it per type as everything else does.
        self._cat_cache: dict[str, set[str]] = {}

    # -- slot machinery ----------------------------------------------------

    def n_slots(self) -> int:
        return len(SLOTS)

    def slot_names(self) -> list[str]:
        return list(SLOTS)

    def _categories(self, token: str) -> set[str]:
        """Closed-class category of a token, plus a coarse open-class guess.

        Uses the same closed-class word lists the representation already uses, so the slot
        filler and the features cannot disagree about what a determiner is.
        """
        cached = self._cat_cache.get(token)
        if cached is not None:
            return cached
        if self._tagger is None:
            from .syntax import SYNTAX_CATEGORIES, SyntaxTagger
            senses = getattr(getattr(self.f, "emb", None), "senses", None)
            lemma_synsets = getattr(senses, "lemma_synsets", {}) or {}
            self._tagger = SyntaxTagger(lemma_synsets)
            self._cat_names = list(SYNTAX_CATEGORIES)
        vec = self._tagger.tag(token)
        cats = {self._cat_names[k] for k in np.nonzero(vec > 0.3)[0]}
        self._cat_cache[token] = cats
        return cats

    def slot_tokens(self, tokens: list[str], i: int) -> list[str]:
        """Fill every slot for a prediction at position ``i``.

        One left-to-right pass over the visible context, recording the most recent token in
        each category, so cost is O(lookback) per position rather than O(slots x lookback).
        """
        lo = max(0, i - self.lookback)
        last: dict[str, str] = {}
        verb_at = -1
        for j in range(lo, i):
            tok = tokens[j]
            cats = self._categories(tok)
            if cats & set(_VERBISH):
                last["verb"] = tok
                verb_at = j
            if cats & set(_NOUNISH):
                last["noun"] = tok
            if cats & set(_DETISH):
                last["det"] = tok
            if any(c in CLOSED_CLASS for c in cats) and "PREPOSITION" in cats:
                last["prep"] = tok
            # Subject: the most recent noun-or-pronoun that precedes the governing verb.
            # Recorded *before* the verb is updated for this token, so a noun sitting to the
            # right of the verb (an object) cannot claim the slot.
            if cats & set(_SUBJISH) and j < max(verb_at, i):
                last.setdefault("subj_pending", tok)
            if j == verb_at:
                if "subj_pending" in last:
                    last["subj"] = last.pop("subj_pending")

        out = []
        for slot in SLOTS:
            if slot == "adj1":
                out.append(tokens[i - 1] if i >= 1 else "")
            elif slot == "adj2":
                out.append(tokens[i - 2] if i >= 2 else "")
            elif slot == "start":
                out.append(tokens[lo] if i > lo else "")
            else:
                out.append(last.get(slot, ""))
        return out

    def slot_report(self, tokens: list[str], i: int) -> str:
        """Which token filled each slot -- the readable form of the context."""
        filled = self.slot_tokens(tokens, i)
        return ", ".join(f"{s}={t!r}" for s, t in zip(SLOTS, filled) if t)
