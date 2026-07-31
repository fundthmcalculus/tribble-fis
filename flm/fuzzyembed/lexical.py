"""Stage 1 -- fuzzy lexical access: surface form -> graded membership in lexemes.

``recieve`` should come out ~0.95 ``receive`` and ~0 everything else. This is the
most natural application of fuzzy sets in the whole pipeline: *is this that word?*
is genuinely a matter of degree, and existing approaches force it to be either
crisp (spell-correct, then tokenise) or implicit (shaped subword loss, as in
Misspelling Oblivious Embeddings).

Putting the fuzziness here, *before* tokenisation, is the mechanism that matters:
a subword tokeniser has no graceful degradation path for ``recieve`` -- it simply
becomes a different token sequence. Membership is computed before anything can
shatter.

The aggregator over the scoring channels is itself a **TSK fuzzy system**
(``MixtureOfGaussiansFuzzyRegressor``), which is the recursive punchline: the
layer that makes the embedding robust is itself an interpretable fuzzy model, so
it can *report why* it matched. No shaped-loss approach can produce that
attribution, because it has nothing to report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# QWERTY adjacency, for cost-weighted substitution.
_ADJ = {
    "a": "qwsxz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}

# The five *similarity* channels the FIS aggregates. Candidate frequency is
# deliberately NOT here: it is a prior, not a similarity, and inside a t-norm
# product an atypical frequency value can veto an obviously-correct match (see
# FREQ_PRIOR_WEIGHT below). It is applied outside the FIS instead.
CHANNELS = ("dl_sim", "trigram_dice", "kb_sim", "phonetic", "prefix")

# How much the frequency prior may modulate a similarity score:
#   degree = sim * (1 - w + w * freq)
# At w=0.3 a maximally rare candidate keeps 70% of its similarity score, so the
# prior nudges the ranking without ever gating a match to zero.
FREQ_PRIOR_WEIGHT = 0.3


# --------------------------------------------------------------------------
# Channels
# --------------------------------------------------------------------------

def _dl_simple(a: str, b: str) -> int:
    """Optimal string alignment distance -- Levenshtein plus adjacent transposition.

    Transpositions matter disproportionately: ``teh``/``the`` is one of the most
    common real typos, and plain Levenshtein charges it 2 -- the same as two
    unrelated errors. Full matrix rather than rolling rows: the transposition case
    needs row ``i-2``, and clarity is worth more here than the constant factor.
    """
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def keyboard_distance(a: str, b: str) -> float:
    """Levenshtein where substituting an adjacent key is cheap (0.35 vs 1.0).

    This is the channel that buys *explainability*: a match driven by keyboard
    proximity can be reported as "fat-finger", which is a claim a human can check.
    """
    la, lb = len(a), len(b)
    d = [[0.0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = float(i)
    for j in range(lb + 1):
        d[0][j] = float(j)
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            ca, cb = a[i - 1], b[j - 1]
            if ca == cb:
                sub = 0.0
            elif cb in _ADJ.get(ca, ""):
                sub = 0.35
            else:
                sub = 1.0
            d[i][j] = min(d[i - 1][j] + 1.0, d[i][j - 1] + 1.0, d[i - 1][j - 1] + sub)
    return d[la][lb]


def trigram_dice(a: str, b: str, n: int = 3) -> float:
    """Dice coefficient over padded character n-grams -- a subword signal, untrained."""
    def grams(s):
        s = f"__{s}__"
        return {s[i:i + n] for i in range(len(s) - n + 1)}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return 2 * len(ga & gb) / (len(ga) + len(gb))


_PHON_GROUPS = {
    **dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
    **dict.fromkeys("dt", "3"), "l": "4", **dict.fromkeys("mn", "5"), "r": "6",
}


def phonetic_key(word: str) -> str:
    """A Soundex-family consonant-group key.

    Deliberately *not* Double Metaphone -- that is ~500 lines of English
    orthography rules. This catches the phonetic-error class that matters here
    (``fone``/``phone``, ``definately``/``definitely``) at a fraction of the
    complexity. It will not catch homophone confusions like ``their``/``there``,
    which are word-choice errors rather than spelling errors and belong to a
    different layer.
    """
    if not word:
        return ""
    word = word.lower()
    # ph -> f before grouping, the one digraph worth special-casing.
    word = word.replace("ph", "f")
    out = [word[0]]
    last = _PHON_GROUPS.get(word[0], "")
    for ch in word[1:]:
        code = _PHON_GROUPS.get(ch, "")
        if code and code != last:
            out.append(code)
        if ch not in "hw":
            last = code
    return "".join(out)[:6]


def common_prefix_ratio(a: str, b: str) -> float:
    """Shared-prefix length over the shorter word. Typos are rarer word-initially."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(n, 1)


def channel_features(surface: str, candidate: str) -> dict[str, float]:
    """The five interpretable similarity channels for one (surface, candidate) pair.

    Two channels were tried here and removed. An ``in_vocab`` flag was dead weight:
    every BK-tree candidate is drawn *from* the vocabulary, so it was constant 1.0,
    and it made every explanation claim an "exact vocabulary hit". A ``cand_freq``
    channel was actively harmful: as one term in a t-norm product it let an
    atypical frequency veto a correct match (``littel -> little`` scored 0.003 with
    features otherwise indistinguishable from ``freind -> friend`` at 0.605).
    Frequency is now a prior applied outside the FIS.
    """
    m = max(len(surface), len(candidate), 1)
    return {
        "dl_sim": 1.0 - _dl_simple(surface, candidate) / m,
        "trigram_dice": trigram_dice(surface, candidate),
        "kb_sim": 1.0 - keyboard_distance(surface, candidate) / m,
        "phonetic": 1.0 if phonetic_key(surface) == phonetic_key(candidate) else 0.0,
        "prefix": common_prefix_ratio(surface, candidate),
    }


# --------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------

def _perturb():
    """The noise generators from ``flm/exp_b/perturb.py``, reused for typo synthesis.

    ``exp_b`` is a script directory rather than a package, so it needs a path
    insertion; doing it lazily keeps importing this module side-effect-free.
    """
    import sys
    from pathlib import Path
    d = str(Path(__file__).resolve().parents[1] / "exp_b")
    if d not in sys.path:
        sys.path.insert(0, d)
    import perturb  # noqa: PLC0415
    return perturb


def _normalized_log_freq(vocabulary: list[str],
                         counts: dict[str, int] | None) -> dict[str, float]:
    """Map each word to a [0, 1] log-frequency score.

    Falls back to rank-based frequency when no counts are supplied, since
    ``Corpus.vocabulary`` is already frequency-ordered.
    """
    if counts:
        vals = {w: np.log1p(counts.get(w, 0)) for w in vocabulary}
        hi = max(vals.values()) or 1.0
        return {w: float(v / hi) for w, v in vals.items()}
    n = max(len(vocabulary), 1)
    return {w: 1.0 - i / n for i, w in enumerate(vocabulary)}


class BKTree:
    """Metric tree over edit distance, for sub-linear candidate retrieval.

    Without a candidate index this layer is a toy: scoring a token against a full
    vocabulary is O(|V|) edit-distance computations per token. A BK-tree prunes by
    the triangle inequality and keeps the vocabulary side of the cost near-constant
    for small radii.
    """

    def __init__(self, words: list[str]):
        self.root: tuple | None = None
        for w in words:
            self.add(w)

    def add(self, word: str) -> None:
        if self.root is None:
            self.root = (word, {})
            return
        node = self.root
        while True:
            pivot, children = node
            d = _dl_simple(word, pivot)
            if d == 0:
                return
            if d in children:
                node = children[d]
            else:
                children[d] = (word, {})
                return

    def query(self, word: str, max_dist: int) -> list[tuple[str, int]]:
        if self.root is None:
            return []
        out: list[tuple[str, int]] = []
        stack = [self.root]
        while stack:
            pivot, children = stack.pop()
            d = _dl_simple(word, pivot)
            if d <= max_dist:
                out.append((pivot, d))
            for edge, child in children.items():
                if d - max_dist <= edge <= d + max_dist:
                    stack.append(child)
        out.sort(key=lambda kv: kv[1])
        return out


# --------------------------------------------------------------------------
# The lexical access layer
# --------------------------------------------------------------------------

@dataclass
class Match:
    lexeme: str
    degree: float
    features: dict[str, float] = field(default_factory=dict)

    def explain(self) -> str:
        """Why this match fired -- the attribution a shaped subword loss cannot give."""
        f = self.features
        drivers = []
        if f.get("phonetic", 0) > 0.5:
            drivers.append("phonetic key agrees")
        if f.get("kb_sim", 0) > f.get("dl_sim", 0) + 0.02:
            drivers.append("keyboard-adjacent (fat-finger)")
        if f.get("prefix", 0) >= 0.8:
            drivers.append("shared word-initial prefix")
        if f.get("trigram_dice", 0) > 0.6:
            drivers.append("high trigram overlap")
        if f.get("exact", 0) > 0.5:
            drivers.append("exact vocabulary hit")
        elif f.get("cand_freq", 0) > 0.7:
            drivers.append("frequent word")
        return f"{self.lexeme} @ {self.degree:.2f}" + (
            " (" + "; ".join(drivers) + ")" if drivers else "")


class FuzzyLexicon:
    """Maps a surface token to a fuzzy set over vocabulary lexemes.

    ``aggregator`` is an optional fitted TSK regressor over ``CHANNELS``. Without
    one, a t-norm-flavoured heuristic blend is used, which is a reasonable but
    unlearned fallback -- ``train_aggregator`` fits the real thing.
    """

    def __init__(self, vocabulary: list[str], max_dist: int = 2,
                 max_candidates: int = 25, threshold: float = 0.35,
                 aggregator=None, counts: dict[str, int] | None = None,
                 max_matches: int = 5):
        self.max_matches = max_matches
        self.vocabulary = list(vocabulary)
        self.vocab_set = set(self.vocabulary)
        self.max_dist = max_dist
        self.max_candidates = max_candidates
        self.threshold = threshold
        self.aggregator = aggregator
        self.freq = _normalized_log_freq(self.vocabulary, counts)
        # Short words are excluded from the index: at length <= 3 an edit-distance-2
        # neighbourhood covers most of the lexicon, so every match is noise.
        self.tree = BKTree([w for w in self.vocabulary if len(w) > 3])
        self._cache: dict[str, list[Match]] = {}

    # -- scoring -----------------------------------------------------------

    def _heuristic(self, f: dict[str, float]) -> float:
        """Unlearned fallback: weighted blend, then a phonetic/prefix bonus."""
        base = (0.45 * f["dl_sim"] + 0.30 * f["trigram_dice"] + 0.25 * f["kb_sim"])
        bonus = 0.06 * f["phonetic"] + 0.06 * f["prefix"]
        return float(np.clip(base + bonus, 0.0, 1.0))

    def _score(self, feats: list[dict[str, float]]) -> np.ndarray:
        """Similarity degree per candidate, from the FIS (or the fallback blend)."""
        if self.aggregator is None:
            return np.array([self._heuristic(f) for f in feats], dtype=float)
        import pandas as pd
        X = pd.DataFrame(feats, columns=list(CHANNELS))
        # The aggregator is a fuzzy *classifier* over match/no-match, so the
        # membership degree is the positive-class probability. A regressor on a
        # 0/1 target cannot be used here: partition_output() quantile-bins the
        # target and pd.qcut rejects the duplicate edges a binary target produces.
        proba = self.aggregator.predict_proba(X)
        pos = list(self.aggregator.classes_).index(1)
        return np.clip(np.asarray(proba, dtype=float)[:, pos], 0.0, 1.0)

    def match(self, surface: str) -> list[Match]:
        """Fuzzy set over lexemes for one surface token, highest degree first."""
        if surface in self._cache:
            return self._cache[surface]

        # Guard (a): an in-vocabulary token is itself with degree 1. Without this
        # the layer happily "corrects" rare-but-real words into common ones, which
        # is a regression on clean text -- the dominant failure mode of fuzzy
        # lexical access.
        if surface in self.vocab_set:
            out = [Match(surface, 1.0, {"exact": 1.0})]
            self._cache[surface] = out
            return out

        if len(surface) <= 3:
            self._cache[surface] = []
            return []

        cands = self.tree.query(surface, self.max_dist)[: self.max_candidates]
        if not cands:
            self._cache[surface] = []
            return []

        feats = [channel_features(surface, c) for c, _ in cands]
        sims = self._score(feats)

        # Frequency prior, applied *outside* the FIS so it can only re-rank, never
        # veto. This is over-correction guard (b): prefer correcting toward a
        # common word rather than a rare one.
        w = FREQ_PRIOR_WEIGHT
        scores = [
            s * (1.0 - w + w * self.freq.get(c, 0.0))
            for (c, _), s in zip(cands, sims)
        ]

        out = [
            Match(c, float(s), dict(f, cand_freq=self.freq.get(c, 0.0)))
            for (c, _), s, f in zip(cands, scores, feats) if s >= self.threshold
        ]
        out.sort(key=lambda m: -m.degree)
        # Degrees are NOT normalised to sum to 1. These are fuzzy memberships, not
        # probabilities -- a token can be 0.9 "receive" and 0.8 "relieve" at once,
        # and rescaling to a simplex would destroy exactly the graded information
        # the layer exists to produce. Total contributed mass is bounded later, in
        # composition, where it belongs.
        out = out[: self.max_matches]
        self._cache[surface] = out
        return out

    def explain(self, surface: str) -> str:
        matches = self.match(surface)
        if not matches:
            return f"{surface!r}: no lexeme match (out of vocabulary)"
        return f"{surface!r} -> " + " | ".join(m.explain() for m in matches)


# --------------------------------------------------------------------------
# Training the aggregator
# --------------------------------------------------------------------------

def build_training_pairs(vocabulary: list[str], ops: tuple[str, ...],
                         n_words: int = 1500, seed: int = 0,
                         counts: dict[str, int] | None = None
                         ) -> tuple[list[dict], list[int]]:
    """Generate (features, target) pairs from synthetic typos.

    Positives are (perturbed, true word) -> 1.0; negatives are (perturbed, another
    BK-tree candidate) -> 0.0. Restricting to ``ops`` is what makes a held-out
    *error-class* split possible, which matters: training and testing on the same
    noise generator measures nothing but self-consistency, so
    ``train_aggregator`` trains on some operations and evaluates on others.
    """
    import random
    perturb_word = _perturb().perturb_word

    rng = random.Random(seed)
    long_words = [w for w in vocabulary if len(w) > 3]
    freq = _normalized_log_freq(vocabulary, counts)

    # Sample training words with probability proportional to log frequency, NOT
    # uniformly. Uniform sampling draws mostly rare words (a Zipfian vocabulary is
    # nearly all tail), so the positive class's `cand_freq` distribution ends up
    # concentrated near 0 -- while at inference the true correction is usually a
    # *frequent* word. The resulting train/test mismatch on that one channel put
    # real corrections in the tail of its Gaussian, and because rule firing is a
    # t-norm product across channels, a single tail value collapsed the whole
    # score: `littel -> little` scored 0.003 while the near-identical
    # `freind -> friend` scored 0.605.
    weights = [freq.get(w, 0.0) + 0.05 for w in long_words]
    words = rng.choices(long_words, weights=weights, k=min(n_words, len(long_words)))
    tree = BKTree(long_words)

    feats: list[dict] = []
    targets: list[int] = []
    for word in words:
        typo = perturb_word(word, rng, op=rng.choice(ops))
        if typo == word:
            continue
        feats.append(channel_features(typo, word))
        targets.append(1)
        for cand, _ in tree.query(typo, 2)[:6]:
            if cand != word:
                feats.append(channel_features(typo, cand))
                targets.append(0)
    return feats, targets


def train_aggregator(vocabulary: list[str], n_words: int = 1500, seed: int = 0,
                     counts: dict[str, int] | None = None, verbose: bool = True):
    """Fit the TSK aggregator, holding out two entire error classes.

    Train on substitute/delete/double, evaluate on **transpose/insert**. Training
    and evaluating on the same noise generator would measure nothing but
    self-consistency, so the split is by *error class*: a model that only works on
    the operations it saw is memorising a generator, not learning robustness.

    A fuzzy **classifier** over match/no-match, not a regressor -- the target is
    binary, and ``MixtureOfGaussiansFuzzyRegressor`` quantile-bins its target,
    which ``pd.qcut`` rejects for binary input. ``predict_proba`` supplies the
    graded degree.
    """
    from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
    import pandas as pd
    OPS = _perturb().OPS

    cols = list(CHANNELS)

    def _fit(feats, targets):
        model = MixtureOfGaussiansFuzzyClassifier(top_n=len(CHANNELS))
        model.fit(pd.DataFrame(feats, columns=cols), np.asarray(targets))
        return model

    def _separation(model, feats, targets) -> tuple[float, float]:
        proba = model.predict_proba(pd.DataFrame(feats, columns=cols))
        pos = list(model.classes_).index(1)
        pred = np.clip(np.asarray(proba)[:, pos], 0.0, 1.0)
        t = np.asarray(targets)
        # Separation, not accuracy: the layer only has to rank the true lexeme
        # above its distractors, and at ~6:1 imbalance accuracy is dominated by
        # the negatives.
        return (float(pred[t == 1].mean() - pred[t == 0].mean()),
                float(((pred >= 0.5).astype(int) == t).mean()))

    # (1) Generalisation diagnostic: train on three error classes, evaluate on the
    # two held out. This is the honest robustness number, and it is *not* the
    # model that ships -- crippling the deployed model to preserve a clean split
    # would be a false economy.
    train_ops = ("substitute", "delete", "double")
    test_ops = ("transpose", "insert")
    Xtr, ytr = build_training_pairs(vocabulary, train_ops, n_words, seed, counts)
    Xho, yho = build_training_pairs(vocabulary, test_ops, max(n_words // 3, 50),
                                    seed + 999, counts)
    if not Xtr or not Xho:
        raise RuntimeError("no training pairs generated; vocabulary too small?")
    held_sep, held_acc = _separation(_fit(Xtr, ytr), Xho, yho)

    # (2) The shipped model: all five error classes, evaluated on a fresh sample.
    Xall, yall = build_training_pairs(vocabulary, OPS, n_words, seed + 7, counts)
    Xval, yval = build_training_pairs(vocabulary, OPS, max(n_words // 3, 50),
                                      seed + 4242, counts)
    model = _fit(Xall, yall)
    val_sep, val_acc = _separation(model, Xval, yval)

    if verbose:
        print(f"  shipped model: {len(yall)} pairs over all ops {OPS}")
        print(f"    held-out sample : separation={val_sep:+.3f} acc={val_acc:.3f}")
        print(f"  generalisation diagnostic (train {train_ops} -> test {test_ops}):")
        print(f"    held-out error classes: separation={held_sep:+.3f} acc={held_acc:.3f}")
        if held_sep < val_sep - 0.05:
            print(f"    NOTE: {val_sep - held_sep:+.3f} separation drop across error "
                  "classes -- transposition lowers trigram overlap while keeping\n"
                  "          edit distance at 1, so channels learned on other ops "
                  "transfer imperfectly.")
    return model, {
        "separation": val_sep, "accuracy": val_acc,
        "heldout_class_separation": held_sep, "heldout_class_accuracy": held_acc,
        "train_ops": OPS, "diagnostic_train_ops": train_ops,
        "diagnostic_test_ops": test_ops,
    }
