"""Character-level noise generators for the robustness axis of Experiment B.

These are *internal sweep* tools. For anything published, swap them for the
perturbation generators from the robustness literature (AdvGLUE / TextBugger) --
beating your own noise model is not a result. See
``../FIS_ON_EMBEDDINGS_PLAN.md`` section 3.

The five operations mirror the error classes the fuzzy lexical-access layer of
Experiment A is designed to absorb (see ``../FUZZY_EMBEDDING_PLAN.md`` 2.1), so a
degradation curve measured here is directly comparable to one measured there.
"""

from __future__ import annotations

import random

# QWERTY physical adjacency. Keyboard-adjacent substitution is the dominant
# real-world typo class, and the one a uniform-random character substitution
# badly misrepresents.
_ADJACENT = {
    "a": "qwsxz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}

OPS = ("substitute", "transpose", "delete", "insert", "double")


def perturb_word(word: str, rng: random.Random, op: str | None = None) -> str:
    """Apply one character-level edit to ``word``.

    Words of length < 3 are returned unchanged: a one- or two-character token is
    usually a function word or punctuation, and editing it changes the sentence's
    meaning rather than adding noise.
    """
    if len(word) < 3:
        return word
    op = op or rng.choice(OPS)
    i = rng.randrange(len(word))
    ch = word[i].lower()

    if op == "substitute":
        pool = _ADJACENT.get(ch)
        if not pool:
            return word
        return word[:i] + rng.choice(pool) + word[i + 1:]
    if op == "transpose":
        # Interior transposition only -- keeps first/last char, the classic
        # "cdhrimbaega" effect that humans read straight through.
        if len(word) < 4:
            return word
        j = rng.randrange(1, len(word) - 2)
        return word[:j] + word[j + 1] + word[j] + word[j + 2:]
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "insert":
        pool = _ADJACENT.get(ch, "aeiou")
        return word[:i] + rng.choice(pool) + word[i:]
    if op == "double":
        return word[:i] + word[i] + word[i:]
    raise ValueError(f"unknown op: {op}")


def perturb_text(text: str, rate: float, seed: int = 0) -> str:
    """Perturb each whitespace-delimited token with probability ``rate``.

    ``rate`` is the *per-token* probability of one edit, not a per-character
    rate -- report it that way, since the two differ by roughly the mean token
    length and the confusion makes noise levels incomparable across papers.
    """
    if rate <= 0:
        return text
    rng = random.Random(seed)
    return " ".join(
        perturb_word(tok, rng) if rng.random() < rate else tok
        for tok in text.split()
    )


def perturb_corpus(texts, rate: float, seed: int = 0) -> list[str]:
    """Perturb a corpus, deriving a distinct but deterministic seed per document."""
    return [perturb_text(t, rate, seed=seed + i) for i, t in enumerate(texts)]
