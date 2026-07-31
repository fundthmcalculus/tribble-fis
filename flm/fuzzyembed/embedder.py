"""Stage 3 -- span composition, and the assembled fuzzy embedding model.

Token-level node memberships become a span embedding by:

1. **Aggregating across tokens** with an OWA operator. OWA earns its extra
   parameter: it interpolates between "any token mentions this concept" (max) and
   "most tokens do" (mean), and the weight is a single readable number.
2. **Applying modifier operators.** Zadeh's *hedges* work exactly as advertised and
   need no training::

       very X      -> concentration    mu ** 2
       somewhat X  -> dilation         mu ** 0.5

   Zadeh's *complement* for negation does **not** work here, and this was found
   empirically rather than assumed. A lexeme's sense vector is sparse and
   peaked -- ``happy`` is ``happy.a.01@1.0, felicitous@0.07, glad@0.01`` -- so
   ``1 - mu`` zeroes the intended sense while promoting its weakly-activated
   siblings to ~0.95. "not happy" came out asserting *strongly felicitous and
   glad*. The complement of a sparse membership vector is a dense vector of
   near-ones, which is semantically meaningless.

   Negation is therefore **suppression plus antonym transfer**: zero the negated
   concept, and move its mass to a WordNet antonym when one exists in the
   hierarchy. That is the semantically right operation, and it reintroduces
   precisely the structure Roget's antonymous opposed pairs (648 Goodness / 649
   Badness) would have supplied natively -- see ``hierarchy.py`` on the cost of the
   Roget's -> WordNet substitution.

3. **Deriving every coarse level from the finest** by t-conorm rollup, which makes
   fuzzy subsumption (C1) hold *by construction* rather than by penalty. This is
   the property that distinguishes the readout from Matryoshka truncation, and
   ``tests/test_hierarchy.py`` asserts it exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .corpus import Corpus, tokenize
from .hierarchy import FuzzyHierarchy, build_wordnet_hierarchy
from .lexical import FuzzyLexicon, train_aggregator
from .senses import SenseAssigner

# Hedges: surface cue -> exponent on the membership degree. 1.0 is a no-op.
HEDGES: dict[str, float] = {
    "very": 2.0, "extremely": 3.0, "really": 2.0, "so": 2.0, "quite": 1.5,
    "somewhat": 0.5, "slightly": 0.5, "rather": 0.7, "fairly": 0.7, "little": 0.5,
}
NEGATORS = frozenset({"not", "no", "never", "none", "nothing", "cannot", "n't"})

# How far a hedge or negator reaches to its right. A dependency parse would scope
# this properly; a fixed window is the honest cheap approximation and is stated as
# such rather than dressed up.
SCOPE_WINDOW = 3

#: For an out-of-vocabulary token, keep only lexeme matches within this fraction of the best
#: match's degree. See ``FuzzyEmbedder._prune_matches`` for the measurement that set it.
OOV_RELATIVE_KEEP = 0.8


@dataclass
class Explanation:
    """Why a span embedded the way it did -- the audit trail."""

    text: str
    tokens: list[str] = field(default_factory=list)
    lexical: list[str] = field(default_factory=list)
    top_nodes: list[tuple[str, float]] = field(default_factory=list)
    hedges_applied: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f'"{self.text}"']
        if self.lexical:
            lines.append("  lexical access:")
            lines.extend(f"    {s}" for s in self.lexical)
        if self.hedges_applied:
            lines.append("  hedges: " + "; ".join(self.hedges_applied))
        if self.top_nodes:
            lines.append("  top dimensions:")
            lines.extend(f"    {name:<28} {deg:.3f}" for name, deg in self.top_nodes)
        return "\n".join(lines)


class FuzzyEmbedder:
    """A text embedding whose coordinates are memberships in named hierarchy nodes.

    ``embed`` returns the finest level; ``embed_levels`` returns every resolution at
    once, and the coarse ones are exact aggregations of the fine one.
    """

    def __init__(self, hierarchy: FuzzyHierarchy, lexicon: FuzzyLexicon,
                 senses: SenseAssigner, owa_weight: float = 0.7,
                 relax_rounds: int = 1, rollup: str = "max"):
        self.h = hierarchy
        self.lexicon = lexicon
        self.senses = senses
        self.owa_weight = owa_weight
        self.relax_rounds = relax_rounds
        self.rollup = rollup

    # -- composition -------------------------------------------------------

    def _owa(self, values: np.ndarray) -> float:
        """Blend max and mean. ``w=1`` is pure max (existential), ``w=0`` pure mean."""
        if values.size == 0:
            return 0.0
        w = self.owa_weight
        return float(w * values.max() + (1.0 - w) * values.mean())

    @staticmethod
    def _prune_matches(matches: list) -> list:
        """Keep only lexeme matches close to the best one.

        Merging *every* candidate a fuzzy lookup returns produces a blend of unrelated
        senses rather than a reading. Measured on an out-of-vocabulary word (E23.5):

            'wooden'  noun.substance=0.57, noun.group=0.56, noun.person=0.51,
                      adj.all=0.49, verb.cognition=0.46

        Four unrelated supersenses at roughly equal degree is not graded ambiguity, it is
        noise -- and the rule learner then has to fit it. A person reading an unfamiliar word
        entertains the one or two nearest real words, not the eight nearest strings.

        Relative rather than absolute: genuine ambiguity between two close candidates is
        exactly what this representation should keep (``littel`` -> ``little`` and possibly
        ``list``), while a long tail at half the best degree is not evidence of anything.
        Exact vocabulary hits already return a single match, so this only affects OOV forms.
        """
        if len(matches) <= 1:
            return matches
        floor = OOV_RELATIVE_KEEP * matches[0].degree
        return [m for m in matches if m.degree >= floor]

    @staticmethod
    def _scope_modifiers(tokens: list[str]) -> tuple[list[float], list[bool], list[str]]:
        """Per-token hedge exponent and negation flag, from a right-scoped window."""
        n = len(tokens)
        exps = [1.0] * n
        negs = [False] * n
        notes: list[str] = []
        for i, tok in enumerate(tokens):
            if tok in NEGATORS:
                for j in range(i + 1, min(i + 1 + SCOPE_WINDOW, n)):
                    negs[j] = True
                notes.append(f"'{tok}' negates next {SCOPE_WINDOW}")
            elif tok in HEDGES:
                e = HEDGES[tok]
                for j in range(i + 1, min(i + 1 + SCOPE_WINDOW, n)):
                    exps[j] *= e
                notes.append(f"'{tok}' -> exponent {e:g}")
        return exps, negs, notes

    def embed_levels(self, text: str, explain: bool = False
                     ) -> tuple[dict[int, np.ndarray], Explanation | None]:
        """Every resolution level for ``text``, coarse derived exactly from fine."""
        finest = self.h.n_levels - 1
        tokens = [t for sent in tokenize(text) for t in sent]
        exp = Explanation(text, tokens) if explain else None

        # token index -> {leaf key: degree}
        per_token: list[dict[str, float]] = []
        assignments = []
        keep_idx = []
        for i, tok in enumerate(tokens):
            matches = self._prune_matches(self.lexicon.match(tok))
            if explain:
                exp.lexical.append(self.lexicon.explain(tok))
            if not matches:
                continue
            # A token's node memberships are its lexeme matches' memberships,
            # discounted by how strongly the surface form matched that lexeme.
            merged: dict[str, float] = {}
            for m in matches:
                sa = self.senses.assign(m.lexeme)
                if sa is None:
                    continue
                for key, deg in sa.leaf_degrees.items():
                    merged[key] = max(merged.get(key, 0.0), deg * m.degree)
            if merged:
                from .senses import SenseAssignment
                assignments.append(SenseAssignment(tok, merged))
                keep_idx.append(i)

        if self.relax_rounds and len(assignments) > 1:
            assignments = self.senses.context_relax(assignments, self.relax_rounds)
        per_token = [a.leaf_degrees for a in assignments]

        exps, negs, notes = self._scope_modifiers(tokens)
        if explain and notes:
            exp.hedges_applied = notes

        # Accumulate per-node values across tokens, then OWA.
        buckets: dict[str, list[float]] = {}
        for local_i, degrees in enumerate(per_token):
            tok_i = keep_idx[local_i]
            e, neg = exps[tok_i], negs[tok_i]
            for key, deg in degrees.items():
                v = deg ** e if e != 1.0 else deg
                if neg:
                    # Suppress, do not complement (see module docstring).
                    for anti in self.senses.antonym_keys(key):
                        buckets.setdefault(anti, []).append(v)
                    v = 0.0
                buckets.setdefault(key, []).append(v)

        leaf = self.h.zeros(finest)
        for key, vals in buckets.items():
            leaf[self.h.index(self.h.project(key, finest), finest)] = self._owa(
                np.asarray(vals, dtype=float))

        levels = self.h.enforce_subsumption({finest: leaf}, op=self.rollup)

        if explain:
            order = np.argsort(leaf)[::-1][:8]
            keys = self.h.level_keys(finest)
            exp.top_nodes = [(self.h.name(keys[i]), float(leaf[i]))
                             for i in order if leaf[i] > 0]
        return levels, exp

    def embed(self, text: str, level: int | None = None) -> np.ndarray:
        levels, _ = self.embed_levels(text)
        return levels[self.h.n_levels - 1 if level is None else level]

    def embed_batch(self, texts: list[str], level: int) -> np.ndarray:
        return np.vstack([self.embed(t, level) for t in texts])

    def explain(self, text: str) -> str:
        _, exp = self.embed_levels(text, explain=True)
        return exp.render()


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def build_embedder(corpus: Corpus, n_levels: int = 6, max_types: int | None = None,
                   train_lexical: bool = True, verbose: bool = True,
                   ) -> tuple[FuzzyEmbedder, dict]:
    """Build the whole stack from a corpus. Returns ``(embedder, info)``.

    ``max_types=None`` (the default) gives sense and lexicon coverage over the **whole**
    vocabulary. Truncating it was the E23.5 bug: any type outside the prefix was treated as
    a possible misspelling and resolved by fuzzy lexical access, so 24,044 of 27,044 types on
    the 1M-token corpus got a blend of neighbours' senses instead of their own. That cost
    ~293s of a 325s fit *and* fed the rule learner noise -- the larger the corpus, the larger
    the fraction of context that was blur.

    Full coverage is close to free, which is what makes truncation indefensible rather than
    merely suboptimal:

    * **Dimensionality does not change.** The model reads level 2, and level 2 is WordNet's 45
      lexicographer files regardless of vocabulary size -- measured widths went
      ``[1, 4, 45, 4983, ...]`` at 3,000 types to ``[1, 4, 45, 12628, ...]`` at 27,044. Only
      the levels below the one in use grow, so perplexity stays comparable.
    * **Build time does not grow.** 3.7s for 27,044 types against 5.6s for 3,000.

    Pass an integer only to *deliberately* study a truncated lexicon.
    """
    vocab = corpus.vocabulary if max_types is None else corpus.vocabulary[:max_types]
    if verbose:
        print(f"building hierarchy over {len(vocab)} vocabulary types...")
    hierarchy, lemma_synsets = build_wordnet_hierarchy(vocab, n_levels=n_levels)
    if verbose:
        print(hierarchy.describe())
        print(f"  lemmas with senses: {len(lemma_synsets)}/{len(vocab)}")

    aggregator, lex_info = (None, {})
    if train_lexical:
        if verbose:
            print("training fuzzy lexical-access aggregator...")
        aggregator, lex_info = train_aggregator(
            vocab, n_words=900, counts=corpus.counts, verbose=verbose)

    lexicon = FuzzyLexicon(vocab, aggregator=aggregator, counts=corpus.counts)
    senses = SenseAssigner(hierarchy, lemma_synsets)
    embedder = FuzzyEmbedder(hierarchy, lexicon, senses)
    return embedder, {
        "widths": hierarchy.widths(),
        "n_nodes": len(hierarchy.nodes),
        "vocab": len(vocab),
        "lemmas_with_senses": len(lemma_synsets),
        "lexical": lex_info,
    }


def save_hierarchy(embedder: FuzzyEmbedder, path: Path) -> None:
    embedder.h.save(path)
