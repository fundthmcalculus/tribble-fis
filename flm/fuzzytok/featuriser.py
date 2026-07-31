"""Featurisers that plug the parameter space into the existing joint ranker.

``JointNextTokenRanker`` and ``FuzzyGenerator`` need only three things from a featuriser:
``_token_vector(token)``, ``_output_names()``, and (optionally) a ``lexemes`` list so the
candidate-side lexeme mask knows how wide the identity block is. Matching that duck-typed
interface is enough to swap the representation without touching the ranker, which is why
the A/B below is a clean comparison rather than a rewrite.

Three featurisers, so the comparison can separate "does the parameter space help?" from
"does it help *instead of* or *in addition to* the existing space?":

``ParameterFeaturiser``   linguistic parameters only (replaces the semantic+syntax space)
``CombinedFeaturiser``    both spaces concatenated (tests whether they are complementary)

The distinction matters because of E19.4: the fuzzy model lost to a bigram head-to-head yet
*improved a mixture with it*. Two representations can be individually comparable and still
carry different information, so replacing is not the same experiment as adding.

Every token goes through the **fuzzy tokenizer**, so a misspelling arrives as partial
membership in the correct unit rather than as an unknown -- the robustness path is intrinsic
to featurisation here rather than bolted on upstream.
"""

from __future__ import annotations

import numpy as np

from .params import PARAMETERS, LinguisticParameterEncoder
from .tokenizer import FuzzyTokenizer

#: Padding marker for positions before the start of a sentence. Needed because the ranker
#: featurises ``""`` for out-of-range context slots, and an all-zero vector there would be
#: indistinguishable from "a token with no recognised parameters".
BOUNDARY = "Boundary"


class ParameterFeaturiser:
    """Encodes tokens as graded linguistic parameters, via the fuzzy tokenizer."""

    def __init__(self, tokenizer: FuzzyTokenizer,
                 encoder: LinguisticParameterEncoder,
                 lexeme_top_k: int = 0, vocabulary: list[str] | None = None,
                 use_fuzzy_readings: bool = True):
        """``use_fuzzy_readings=True`` encodes the whole fuzzy tokenization (all readings,
        degree-weighted). ``False`` encodes the surface form directly and is the ablation
        that isolates what the tokenizer contributes.
        """
        self.tok = tokenizer
        self.enc = encoder
        self.use_fuzzy_readings = use_fuzzy_readings
        # Same lexicalisation knob as FuzzySequenceModel, and for the same measured
        # reason (E19/E20): 54 types carry half the token mass, and identity for the Zipf
        # head is information no category can express.
        self.lexemes = list(vocabulary or [])[:lexeme_top_k] if lexeme_top_k else []
        self.lexeme_index = {w: i for i, w in enumerate(self.lexemes)}
        self._cache: dict[str, np.ndarray] = {}

    def _output_names(self) -> list[str]:
        return list(PARAMETERS) + [BOUNDARY] + [f"={w}" for w in self.lexemes]

    def _token_vector(self, token: str) -> np.ndarray:
        cached = self._cache.get(token)
        if cached is not None:
            return cached

        if not token:
            v = np.zeros(len(PARAMETERS) + 1 + len(self.lexemes), dtype=np.float32)
            v[len(PARAMETERS)] = 1.0                      # BOUNDARY
            self._cache[token] = v
            return v

        params = (self.enc.encode_fuzzy(self.tok.tokenize(token))
                  if self.use_fuzzy_readings else self.enc.encode(token))
        parts = [params.astype(np.float32), np.zeros(1, dtype=np.float32)]
        if self.lexemes:
            ident = np.zeros(len(self.lexemes), dtype=np.float32)
            i = self.lexeme_index.get(token.lower())
            if i is not None:
                ident[i] = 1.0
            parts.append(ident)
        v = np.concatenate(parts)
        self._cache[token] = v
        return v

    def explain(self, token: str) -> str:
        names = self._output_names()
        v = self._token_vector(token)
        live = sorted(((names[i], float(x)) for i, x in enumerate(v) if x > 0),
                      key=lambda kv: -kv[1])
        return f"{token!r}: " + ", ".join(f"{n}={d:.2f}" for n, d in live[:8])


class CombinedFeaturiser:
    """Concatenates two featurisers, so their contributions can be tested together.

    Names are prefixed to keep rules readable and unambiguous about provenance -- a rule
    mentioning ``wn:noun.animal`` versus ``lp:NOUN`` says which space it came from, which
    matters when the whole claim is that the features are interpretable.
    """

    def __init__(self, first, second, first_tag: str = "wn", second_tag: str = "lp"):
        self.a, self.b = first, second
        self.ta, self.tb = first_tag, second_tag
        # Identity dims must live at the very end for JointNextTokenRanker.cand_vector to
        # mask them, so only one side may own them.
        if getattr(first, "lexemes", ()) and getattr(second, "lexemes", ()):
            raise ValueError("only one sub-featuriser may carry lexeme identity dims")
        self.lexemes = list(getattr(second, "lexemes", ())
                            or getattr(first, "lexemes", ()))
        self._owner_is_b = bool(getattr(second, "lexemes", ()))
        self._cache: dict[str, np.ndarray] = {}

    def _output_names(self) -> list[str]:
        n_lex = len(self.lexemes)
        a = [f"{self.ta}:{n}" for n in self.a._output_names()]
        b = [f"{self.tb}:{n}" for n in self.b._output_names()]
        if n_lex:
            # Strip the identity block from whichever side owns it, re-append unprefixed
            # at the end so the mask offset stays correct.
            if self._owner_is_b:
                b = b[:-n_lex]
            else:
                a = a[:-n_lex]
        return a + b + [f"={w}" for w in self.lexemes]

    def _token_vector(self, token: str) -> np.ndarray:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        va, vb = self.a._token_vector(token), self.b._token_vector(token)
        n_lex = len(self.lexemes)
        if n_lex:
            if self._owner_is_b:
                ident, vb = vb[-n_lex:], vb[:-n_lex]
            else:
                ident, va = va[-n_lex:], va[:-n_lex]
            v = np.concatenate([va, vb, ident])
        else:
            v = np.concatenate([va, vb])
        v = v.astype(np.float32)
        self._cache[token] = v
        return v


def build_parameter_featuriser(corpus, vocabulary: list[str], lemma_synsets=None,
                               lexeme_top_k: int = 200, head_size: int = 500,
                               lexicon=None, use_fuzzy_readings: bool = True
                               ) -> ParameterFeaturiser:
    """Assemble tokenizer + encoder + featuriser from a corpus."""
    tok = FuzzyTokenizer.from_corpus(corpus, head_size=head_size,
                                    stem_size=len(vocabulary), lexicon=lexicon)
    enc = LinguisticParameterEncoder(lemma_synsets=lemma_synsets or {})
    return ParameterFeaturiser(tok, enc, lexeme_top_k=lexeme_top_k,
                               vocabulary=vocabulary,
                               use_fuzzy_readings=use_fuzzy_readings)
