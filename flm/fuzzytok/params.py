"""The linguistic parameter space: a small, named, graded feature basis.

Replaces the WordNet-derived space used in ``../fuzzyembed/`` for three measured reasons:

* **The WordNet ladder is unbalanced** -- adjectives have no hypernyms, so the resolution
  ladder jumps 45 -> 4527 with no usable middle (``../fuzzyembed/hierarchy.py``).
* **No morphology.** Supersenses cannot express number, tense, or degree, which is much of
  what predicts the next word.
* **No proper nouns.** Names were the dominant residual coverage gap at 96.7%
  (``../LOG.md`` E1), and a single orthographic feature -- capitalisation -- recovers most
  of them.

Four blocks, ~70 dimensions, every one named before training:

    UPOS       17   Universal Dependencies universal POS tags (de Marneffe et al., 2021)
    FEATS      ~20  UD morphological features: Number, Tense, Degree, Person, Polarity
    Shape      ~12  orthographic: capitalisation, length band, affix class, digits
    Semantics  ~14  coarse supersense groups (entity/act/state/quality/relation/...)
    Affect      3   Osgood's semantic differential: evaluation, potency, activity

Design notes worth stating rather than burying:

* **UD is a designed, balanced inventory**, which is exactly what the WordNet hypernym DAG
  was not. Its 17 UPOS tags are stable and cross-lingual, so the space ports to other
  languages where a Roget's/WordNet backbone would not.
* **Osgood's three affective axes** carry the polarity that Roget's antonymous opposed pairs
  would have supplied structurally. Evaluation is the axis a sentiment task needs.
* **Everything is graded.** A word that is 80% noun senses and 20% verb senses says so; a
  word that is ambiguous between singular and plural (``deer``) holds both.
* **No tagger dependency.** Features come from the lexicon, orthography, and suffix
  evidence, so this runs without a trained POS model. That costs accuracy on genuinely
  ambiguous tokens and is the main limitation of the block -- stated here, not hidden.
"""

from __future__ import annotations

import numpy as np

# --- UPOS: the 17 Universal Dependencies tags -----------------------------
UPOS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
        "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X")

# --- FEATS: a working subset of UD morphological features -----------------
FEATS = ("Number=Sing", "Number=Plur", "Tense=Past", "Tense=Pres", "Tense=Fut",
         "Degree=Cmp", "Degree=Sup", "Person=1", "Person=2", "Person=3",
         "Polarity=Neg", "VerbForm=Part", "VerbForm=Inf", "Poss=Yes",
         "PronType=Prs", "PronType=Int", "PronType=Dem", "Mood=Imp",
         "Aspect=Prog", "Voice=Pass")

# --- Shape: orthography, which is where proper nouns live -----------------
SHAPE = ("Shape=Capitalised", "Shape=AllCaps", "Shape=HasDigit", "Shape=Short",
         "Shape=Medium", "Shape=Long", "Shape=HasHyphen", "Shape=SuffixLy",
         "Shape=SuffixIng", "Shape=SuffixEd", "Shape=SuffixS", "Shape=SuffixTion")

# --- Semantics: coarse supersense groups ----------------------------------
SEM = ("Sem=Entity", "Sem=Animate", "Sem=Person", "Sem=Artifact", "Sem=Place",
       "Sem=Time", "Sem=Act", "Sem=Motion", "Sem=Communication", "Sem=Cognition",
       "Sem=State", "Sem=Quality", "Sem=Quantity", "Sem=Relation")

# --- Affect: Osgood's semantic differential -------------------------------
AFFECT = ("Affect=Evaluation", "Affect=Potency", "Affect=Activity")

PARAMETERS: tuple[str, ...] = UPOS + FEATS + SHAPE + SEM + AFFECT

#: WordNet lexname prefix -> coarse semantic group. Deliberately many-to-few: the point of
#: this block is a *small* basis, so 45 supersenses collapse to 14 groups.
_LEXNAME_TO_SEM = {
    "noun.animal": ("Sem=Entity", "Sem=Animate"),
    "noun.person": ("Sem=Entity", "Sem=Animate", "Sem=Person"),
    "noun.artifact": ("Sem=Entity", "Sem=Artifact"),
    "noun.location": ("Sem=Entity", "Sem=Place"),
    "noun.time": ("Sem=Time",),
    "noun.act": ("Sem=Act",),
    "noun.event": ("Sem=Act",),
    "noun.communication": ("Sem=Communication",),
    "noun.cognition": ("Sem=Cognition",),
    "noun.feeling": ("Sem=State",),
    "noun.state": ("Sem=State",),
    "noun.attribute": ("Sem=Quality",),
    "noun.quantity": ("Sem=Quantity",),
    "noun.relation": ("Sem=Relation",),
    "noun.body": ("Sem=Entity",),
    "noun.food": ("Sem=Entity", "Sem=Artifact"),
    "noun.plant": ("Sem=Entity", "Sem=Animate"),
    "noun.substance": ("Sem=Entity",),
    "noun.object": ("Sem=Entity",),
    "noun.group": ("Sem=Entity",),
    "noun.possession": ("Sem=Relation",),
    "noun.phenomenon": ("Sem=Act",),
    "noun.process": ("Sem=Act",),
    "noun.motive": ("Sem=Cognition",),
    "noun.shape": ("Sem=Quality",),
    "noun.Tops": ("Sem=Entity",),
    "verb.motion": ("Sem=Act", "Sem=Motion"),
    "verb.communication": ("Sem=Act", "Sem=Communication"),
    "verb.cognition": ("Sem=Act", "Sem=Cognition"),
    "verb.emotion": ("Sem=State",),
    "verb.stative": ("Sem=State",),
    "verb.change": ("Sem=Act",),
    "verb.contact": ("Sem=Act",),
    "verb.creation": ("Sem=Act",),
    "verb.body": ("Sem=Act",),
    "verb.consumption": ("Sem=Act",),
    "verb.competition": ("Sem=Act",),
    "verb.possession": ("Sem=Relation",),
    "verb.social": ("Sem=Act",),
    "verb.perception": ("Sem=Cognition",),
    "verb.weather": ("Sem=Act",),
    "adj.all": ("Sem=Quality",),
    "adj.pert": ("Sem=Relation",),
    "adj.ppl": ("Sem=Quality",),
    "adv.all": ("Sem=Quality",),
}

_WN_POS_TO_UPOS = {"n": "NOUN", "v": "VERB", "a": "ADJ", "s": "ADJ", "r": "ADV"}


class LinguisticParameterEncoder:
    """Encodes a token as a graded vector over ``PARAMETERS``."""

    def __init__(self, lemma_synsets: dict[str, list] | None = None,
                 closed_class: dict[str, frozenset[str]] | None = None,
                 polarity: dict[str, float] | None = None,
                 potency: dict[str, float] | None = None):
        self.lemma_synsets = lemma_synsets or {}
        if closed_class is None:
            from ..fuzzyembed.syntax import CLOSED_CLASS
            closed_class = CLOSED_CLASS
        self.closed_class = closed_class
        if polarity is None:
            from ..fuzzyembed.senses import _load_polarity
            polarity = _load_polarity()
        self.polarity = polarity or {}
        self.potency = potency or {}
        self.index = {p: i for i, p in enumerate(PARAMETERS)}
        self._cache: dict[str, np.ndarray] = {}

    @property
    def width(self) -> int:
        return len(PARAMETERS)

    def names(self) -> list[str]:
        return list(PARAMETERS)

    # -- blocks ------------------------------------------------------------

    def _closed_class_upos(self, token: str, v: np.ndarray) -> bool:
        """Map curated closed classes onto UPOS/FEATS. Returns True if it fired."""
        mapping = {
            "DETERMINER": ("DET",), "PRONOUN": ("PRON",), "POSSESSIVE": ("PRON",),
            "AUXILIARY": ("AUX",), "PREPOSITION": ("ADP",), "CONJUNCTION": ("CCONJ",),
            "NEGATOR": ("PART",), "QUANTIFIER": ("DET",), "WH_WORD": ("PRON",),
            "INTENSIFIER": ("ADV",), "INFINITIVE_TO": ("PART",),
        }
        fired = False
        for cat, members in self.closed_class.items():
            if token in members:
                for tag in mapping.get(cat, ()):
                    v[self.index[tag]] = 1.0
                    fired = True
                if cat == "POSSESSIVE":
                    v[self.index["Poss=Yes"]] = 1.0
                if cat == "NEGATOR":
                    v[self.index["Polarity=Neg"]] = 1.0
                if cat == "WH_WORD":
                    v[self.index["PronType=Int"]] = 1.0
                if cat == "PRONOUN":
                    v[self.index["PronType=Prs"]] = 1.0
        return fired

    def _shape(self, surface: str, v: np.ndarray) -> None:
        low = surface.lower()
        if surface[:1].isupper() and not surface.isupper():
            v[self.index["Shape=Capitalised"]] = 1.0
            # Capitalisation is the single cheapest proper-noun signal, and names were
            # the dominant coverage gap. Degree 0.6 not 1.0: sentence-initial words are
            # capitalised too, and this encoder has no sentence position.
            v[self.index["PROPN"]] = max(v[self.index["PROPN"]], 0.6)
        if surface.isupper() and len(surface) > 1:
            v[self.index["Shape=AllCaps"]] = 1.0
        if any(c.isdigit() for c in surface):
            v[self.index["Shape=HasDigit"]] = 1.0
            v[self.index["NUM"]] = max(v[self.index["NUM"]], 0.7)
        if "-" in surface:
            v[self.index["Shape=HasHyphen"]] = 1.0
        n = len(low)
        v[self.index["Shape=Short" if n <= 4 else
                     "Shape=Medium" if n <= 8 else "Shape=Long"]] = 1.0

        # Suffix evidence for UPOS/FEATS. Graded, because every one of these has
        # exceptions ("only" is not an adverb-forming -ly; "ring" is not progressive).
        for suf, shape_key, extra in (
                ("ly", "Shape=SuffixLy", (("ADV", 0.8),)),
                ("ing", "Shape=SuffixIng", (("VERB", 0.6), ("Aspect=Prog", 0.7),
                                            ("VerbForm=Part", 0.6))),
                ("ed", "Shape=SuffixEd", (("VERB", 0.6), ("Tense=Past", 0.7),
                                          ("VerbForm=Part", 0.5))),
                ("tion", "Shape=SuffixTion", (("NOUN", 0.8),)),
                ("s", "Shape=SuffixS", (("Number=Plur", 0.5),)),
        ):
            if low.endswith(suf) and len(low) > len(suf) + 2:
                v[self.index[shape_key]] = 1.0
                for key, deg in extra:
                    v[self.index[key]] = max(v[self.index[key]], deg)

    def _semantics(self, token: str, v: np.ndarray) -> None:
        synsets = self.lemma_synsets.get(token)
        if not synsets:
            return
        pos_counts: dict[str, int] = {}
        sem_counts: dict[str, int] = {}
        for syn in synsets:
            upos = _WN_POS_TO_UPOS.get(syn.pos())
            if upos:
                pos_counts[upos] = pos_counts.get(upos, 0) + 1
            for key in _LEXNAME_TO_SEM.get(syn.lexname(), ()):
                sem_counts[key] = sem_counts.get(key, 0) + 1
        # Graded by sense distribution rather than argmax: an 80/20 noun/verb word says so.
        total_pos = sum(pos_counts.values()) or 1
        for tag, k in pos_counts.items():
            v[self.index[tag]] = max(v[self.index[tag]], k / total_pos)
        top_sem = max(sem_counts.values()) if sem_counts else 1
        for key, k in sem_counts.items():
            v[self.index[key]] = max(v[self.index[key]], k / top_sem)

    def _affect(self, token: str, v: np.ndarray) -> None:
        pol = self.polarity.get(token)
        if pol is not None:
            # Evaluation is signed, mapped to [0, 1] with 0.5 as neutral, so a single
            # dimension carries both poles -- the bipolar-axis idea Roget's opposed pairs
            # would have given structurally.
            v[self.index["Affect=Evaluation"]] = 0.5 + 0.5 * float(np.clip(pol, -1, 1))
        else:
            v[self.index["Affect=Evaluation"]] = 0.5
        pot = self.potency.get(token)
        if pot is not None:
            v[self.index["Affect=Potency"]] = float(np.clip(pot, 0, 1))
        # Activity is left at 0 unless supplied: it needs elicited ratings (Osgood's
        # method) or a norms lexicon, and guessing it from orthography would be invention.

    # -- public ------------------------------------------------------------

    def encode(self, surface: str) -> np.ndarray:
        if surface in self._cache:
            return self._cache[surface]
        v = np.zeros(self.width, dtype=np.float32)
        token = surface.lower()
        closed = self._closed_class_upos(token, v)
        self._shape(surface, v)
        if not closed:
            self._semantics(token, v)
        self._affect(token, v)
        self._cache[surface] = v
        return v

    def encode_fuzzy(self, tokenization) -> np.ndarray:
        """Encode a ``FuzzyTokenization`` by degree-weighted t-conorm over its units.

        This is where the fuzzy tokenizer pays off: a misspelling's partial membership in
        the correct unit propagates into the parameter vector as a partial activation,
        instead of the whole token becoming unknown.
        """
        v = np.zeros(self.width, dtype=np.float32)
        for label, degree in tokenization.unit_degrees().items():
            unit = label.strip("-")
            if not unit:
                continue
            np.maximum(v, degree * self.encode(unit), out=v)
        return v

    def explain(self, surface: str, k: int = 8) -> str:
        v = self.encode(surface)
        live = [(PARAMETERS[i], float(x)) for i, x in enumerate(v) if x > 0]
        live.sort(key=lambda kv: -kv[1])
        body = ", ".join(f"{n}={d:.2f}" for n, d in live[:k]) or "(no parameters)"
        return f"{surface!r}: {body}"
