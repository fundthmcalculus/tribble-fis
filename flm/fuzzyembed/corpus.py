"""Corpus loading and tokenisation for the fuzzy embedding experiments.

TinyStories was the requested corpus and is the right choice -- deliberately
simple vocabulary, narrative register, small. It is distributed on Hugging Face,
which is outside this environment's egress allowlist, so it cannot be fetched
here.

The substitute is the closest reachable register: children's narrative prose from
the ``nltk_data`` Gutenberg sample (mirrored on GitHub, which *is* reachable) --

    bryant-stories.txt        "Stories to Tell to Children"   ~249K chars
    burgess-busterbrown.txt   Thornton Burgess animal story   ~ 85K chars
    carroll-alice.txt         Alice in Wonderland             ~144K chars

~478K characters of simple narrative. Register-wise this is a fair stand-in;
be aware it is early-20th-century prose, so it carries some archaic vocabulary
TinyStories would not, and it is ~50x smaller than a TinyStories subset one would
normally train on.

``load_local`` reads a real TinyStories dump from disk when you have one, so the
swap costs one flag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Children's narrative -- the TinyStories stand-in. ~90K tokens.
TINY_LIKE = ("bryant-stories.txt", "burgess-busterbrown.txt", "carroll-alice.txt")

# All the narrative prose in the nltk Gutenberg sample, ~14x TINY_LIKE. E16 established
# that corpus size is the binding constraint on this stack, so this exists to test the one
# lever that was never pushed.
#
# Deliberately *narrative prose only*. Excluded: bible-kjv (1.0M tokens of archaic register
# and verse numbering, which would dominate the vocabulary), milton-paradise and
# whitman-leaves and blake-poems (verse -- line breaks defeat the regex sentence splitter,
# so "sentences" would be arbitrary spans), and the three Shakespeare plays (dialogue with
# speaker headings, a different token distribution entirely). Adding them would grow the
# token count fastest and confound the measurement worst: a perplexity change could then be
# either "more data" or "different language", and the point of this corpus is to isolate the
# former.
NARRATIVE = TINY_LIKE + (
    "austen-emma.txt", "austen-persuasion.txt", "austen-sense.txt",
    "chesterton-ball.txt", "chesterton-brown.txt", "chesterton-thursday.txt",
    "edgeworth-parents.txt", "melville-moby_dick.txt",
)

_WORD = re.compile(r"[a-z]+")
_SENT_SPLIT = re.compile(r"[.!?]+[\s\"']*")

# Contractions and possessives must be expanded *before* word extraction. Left
# alone they show up as uncovered types ("don't", "i'll", "brown's"), which reads
# as a lexicon gap when it is really a tokenisation artifact -- the M0 coverage run
# surfaced these as top misses.
_CONTRACTIONS = {
    "won't": "will not", "can't": "can not", "shan't": "shall not",
    "n't": " not", "'ll": " will", "'re": " are", "'ve": " have",
    "'m": " am", "'d": " would", "'s": "",  # possessive/copula: drop the clitic
}


@dataclass
class Corpus:
    name: str
    sentences: list[list[str]]   # tokenised, lowercased
    vocabulary: list[str]        # frequency-ordered
    counts: dict[str, int]

    @property
    def n_tokens(self) -> int:
        return sum(len(s) for s in self.sentences)

    def summary(self) -> str:
        top = ", ".join(self.vocabulary[:8])
        return (f"{self.name}: {len(self.sentences):,} sentences, "
                f"{self.n_tokens:,} tokens, {len(self.vocabulary):,} types "
                f"(top: {top})")

    def split(self, test_frac: float = 0.2, seed: int = 0) -> tuple[Corpus, Corpus]:
        """Sentence-level train/test split, sharing one vocabulary.

        Splitting by *sentence* rather than by position is essential and was missing
        at first: the joint ranker's ``build`` and ``evaluate`` both iterated the full
        sentence list under different shuffles, so held-out positions came from
        sentences the model had trained on. Every ranking number measured before this
        existed is optimistic.

        Vocabulary and counts stay shared, deliberately -- they are properties of the
        lexicon and the frequency prior, not of the split, and re-deriving them per
        side would change the candidate set between train and test and make the two
        incomparable.
        """
        import random
        idx = list(range(len(self.sentences)))
        random.Random(seed).shuffle(idx)
        n_test = max(1, int(len(idx) * test_frac))
        test_idx, train_idx = set(idx[:n_test]), idx[n_test:]
        train = [self.sentences[i] for i in sorted(train_idx)]
        test = [self.sentences[i] for i in sorted(test_idx)]
        return (Corpus(f"{self.name}/train", train, self.vocabulary, self.counts),
                Corpus(f"{self.name}/test", test, self.vocabulary, self.counts))

    def truncate_vocabulary(self, max_types: int | None) -> Corpus:
        """Keep only the ``max_types`` most frequent types (a small-model knob)."""
        if max_types is None or max_types >= len(self.vocabulary):
            return self
        keep = set(self.vocabulary[:max_types])
        sents = [[t for t in s if t in keep] for s in self.sentences]
        sents = [s for s in sents if s]
        return Corpus(self.name, sents, self.vocabulary[:max_types],
                      {w: self.counts[w] for w in self.vocabulary[:max_types]})


def tokenize(text: str) -> list[list[str]]:
    """Sentence-split then word-tokenise. Deliberately simple and dependency-free.

    Regex splitting is crude, but the fuzzy embedding aggregates over a span with a
    commutative operator, so sentence-boundary errors cost far less here than they
    would in a parser-driven pipeline.
    """
    out = []
    for raw in _SENT_SPLIT.split(text.lower()):
        raw = raw.replace("’", "'")
        for pat, sub in _CONTRACTIONS.items():
            raw = raw.replace(pat, sub)
        toks = _WORD.findall(raw)
        if toks:
            out.append(toks)
    return out


def _build(name: str, sentences: list[list[str]]) -> Corpus:
    # Single characters are dropped: contraction expansion and OCR noise leave stray
    # "o"/"t"/"m" tokens that carry no meaning but do acquire spurious WordNet senses
    # (and then dominate decode retrieval, since their membership vector is a single
    # coordinate).
    sentences = [[t for t in s if len(t) > 1] for s in sentences]
    sentences = [s for s in sentences if s]
    counts: dict[str, int] = {}
    for sent in sentences:
        for tok in sent:
            counts[tok] = counts.get(tok, 0) + 1
    vocab = sorted(counts, key=lambda w: (-counts[w], w))
    return Corpus(name, sentences, vocab, counts)


def load_tiny_like(fileids: tuple[str, ...] = TINY_LIKE) -> Corpus:
    """The TinyStories stand-in from the nltk Gutenberg sample."""
    from nltk.corpus import gutenberg
    sentences: list[list[str]] = []
    for fid in fileids:
        sentences.extend(tokenize(gutenberg.raw(fid)))
    # Dedupe author prefixes: three Austen novels should read "austen", not
    # "austen,austen,austen", or the corpus name in every results table is unreadable.
    authors = list(dict.fromkeys(f.split("-")[0] for f in fileids))
    return _build(f"gutenberg({','.join(authors)})", sentences)


def load_brown(categories: tuple[str, ...] | None = ("news", "fiction", "romance")
               ) -> Corpus:
    """Balanced modern English -- the harder coverage test of the two."""
    from nltk.corpus import brown
    sents = [[w.lower() for w in s if _WORD.fullmatch(w.lower())]
             for s in brown.sents(categories=list(categories) if categories else None)]
    return _build(f"brown({','.join(categories) if categories else 'all'})",
                  [s for s in sents if s])


def load_local(path: Path, max_docs: int | None = None) -> Corpus:
    """Load a plain-text or JSONL corpus from disk -- use this for real TinyStories.

    JSONL is read as one document per line with a ``text`` (or ``story``) field,
    which is the shape TinyStories ships in.
    """
    path = Path(path)
    sentences: list[list[str]] = []
    if path.suffix == ".jsonl":
        import json
        with path.open() as fh:
            for i, line in enumerate(fh):
                if max_docs is not None and i >= max_docs:
                    break
                doc = json.loads(line)
                sentences.extend(tokenize(doc.get("text") or doc.get("story") or ""))
    else:
        sentences = tokenize(path.read_text(errors="replace"))
    return _build(f"local({path.name})", sentences)


def load_corpus(spec: str, max_types: int | None = None) -> Corpus:
    """``"tiny"`` | ``"narrative"`` | ``"brown"`` | ``"brown-all"`` | any filesystem path."""
    if spec == "tiny":
        corpus = load_tiny_like()
    elif spec == "narrative":
        corpus = load_tiny_like(NARRATIVE)
    elif spec == "brown":
        corpus = load_brown()
    elif spec == "brown-all":
        corpus = load_brown(None)
    else:
        corpus = load_local(Path(spec))
    return corpus.truncate_vocabulary(max_types)
