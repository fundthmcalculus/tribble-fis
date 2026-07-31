"""Tests for the fuzzy embedding stack.

The rollup-exactness tests are the important ones: exact multi-resolution readout is
*the* claim that separates this from Matryoshka truncation, so it belongs in CI
rather than in a notebook. If these fail, fuzzy subsumption is not actually being
enforced and the central claim is void.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flm.fuzzyembed.corpus import Corpus, tokenize  # noqa: E402
from flm.fuzzyembed.hierarchy import (  # noqa: E402
    FuzzyHierarchy, Node, ROOT_KEY, build_wordnet_hierarchy,
)
from flm.fuzzyembed.lexical import (  # noqa: E402
    BKTree, _dl_simple, common_prefix_ratio, keyboard_distance, phonetic_key,
    trigram_dice,
)
from flm.fuzzyembed.similarity import fuzzy_jaccard, hierarchy_jaccard  # noqa: E402

wn = pytest.importorskip("nltk.corpus.reader.wordnet", reason="needs nltk")


# --------------------------------------------------------------------------
# A hand-built hierarchy, so structural tests do not depend on WordNet
# --------------------------------------------------------------------------

def toy_hierarchy() -> FuzzyHierarchy:
    """A 4-level tree with a deliberately *short* branch, to exercise clamping.

        *
        |- pos:n -- lex:animal -- dog
        |                      \\- cat
        |- pos:a -- lex:quality           <- depth 2 only; clamps at levels 3+
    """
    paths = {
        ROOT_KEY: (ROOT_KEY,),
        "pos:n": (ROOT_KEY, "pos:n"),
        "lex:animal": (ROOT_KEY, "pos:n", "lex:animal"),
        "dog": (ROOT_KEY, "pos:n", "lex:animal", "dog"),
        "cat": (ROOT_KEY, "pos:n", "lex:animal", "cat"),
        "pos:a": (ROOT_KEY, "pos:a"),
        "lex:quality": (ROOT_KEY, "pos:a", "lex:quality"),
    }
    nodes = {k: Node(k, k.split(":")[-1], v) for k, v in paths.items()}
    # Only these three keys are places a sense can land; the rest are interior.
    return FuzzyHierarchy(nodes, n_levels=4,
                          terminals={"dog", "cat", "lex:quality"})


def test_toy_widths_and_clamping():
    h = toy_hierarchy()
    assert h.widths() == [1, 2, 2, 3]
    # A shallow branch clamps: lex:quality is its own level-3 representative.
    assert h.project("lex:quality", 3) == "lex:quality"
    assert h.project("dog", 2) == "lex:animal"
    assert h.project("dog", 0) == ROOT_KEY


def test_rollup_is_exact_max():
    """THE claim: a coarse level is an exact t-conorm aggregation of the fine one."""
    h = toy_hierarchy()
    leaf = h.zeros(3)
    leaf[h.index("dog", 3)] = 0.8
    leaf[h.index("cat", 3)] = 0.3
    leaf[h.index("lex:quality", 3)] = 0.5

    l2 = h.rollup(leaf, 3, 2)
    assert l2[h.index("lex:animal", 2)] == pytest.approx(0.8)   # max(0.8, 0.3)
    assert l2[h.index("lex:quality", 2)] == pytest.approx(0.5)

    l1 = h.rollup(leaf, 3, 1)
    assert l1[h.index("pos:n", 1)] == pytest.approx(0.8)
    assert l1[h.index("pos:a", 1)] == pytest.approx(0.5)

    # Rolling up one level at a time must equal rolling up directly -- otherwise
    # "resolution dial" is not a coherent notion.
    assert np.allclose(h.rollup(l2, 2, 1), l1)
    assert np.allclose(h.rollup(leaf, 3, 0), h.rollup(l1, 1, 0))


def test_rollup_ops_agree_on_single_child():
    h = toy_hierarchy()
    leaf = h.zeros(3)
    leaf[h.index("dog", 3)] = 0.6
    for op in ("max", "sum", "probor"):
        rolled = h.rollup(leaf, 3, 2, op=op)
        assert rolled[h.index("lex:animal", 2)] == pytest.approx(0.6)


def test_subsumption_holds_after_enforcement():
    """(C1): a parent's membership never falls below its most present child."""
    h = toy_hierarchy()
    leaf = h.zeros(3)
    leaf[h.index("dog", 3)] = 0.9
    leaf[h.index("cat", 3)] = 0.4
    levels = h.enforce_subsumption({3: leaf})

    for level in range(1, h.n_levels):
        for key in h.level_keys(level):
            child = levels[level][h.index(key, level)]
            parent_key = h.project(key, level - 1)
            parent = levels[level - 1][h.index(parent_key, level - 1)]
            assert parent >= child - 1e-6, f"{parent_key} < {key} at L{level}"


def test_rollup_rejects_downward():
    h = toy_hierarchy()
    with pytest.raises(ValueError):
        h.rollup(h.zeros(1), 1, 3)


def test_hierarchy_roundtrip(tmp_path):
    h = toy_hierarchy()
    p = tmp_path / "h.json"
    h.save(p)
    back = FuzzyHierarchy.load(p)
    assert back.widths() == h.widths()
    assert back.project("dog", 2) == "lex:animal"


# --------------------------------------------------------------------------
# Lexical channels
# --------------------------------------------------------------------------

def test_transposition_costs_one():
    """The reason to use Damerau-Levenshtein rather than plain Levenshtein."""
    assert _dl_simple("teh", "the") == 1
    assert _dl_simple("hosue", "house") == 1
    assert _dl_simple("littel", "little") == 1
    assert _dl_simple("abc", "abc") == 0


def test_keyboard_distance_prefers_adjacent_keys():
    # 'q' is adjacent to 'w'; 'p' is not.
    assert keyboard_distance("world", "qorld") < keyboard_distance("world", "porld")


def test_phonetic_key_groups_homophones():
    assert phonetic_key("fone") == phonetic_key("phone")
    assert phonetic_key("cat") != phonetic_key("dog")


def test_trigram_and_prefix():
    assert trigram_dice("rabbit", "rabbit") == pytest.approx(1.0)
    assert trigram_dice("rabbit", "xyzqqq") == pytest.approx(0.0)
    assert common_prefix_ratio("little", "littel") == pytest.approx(4 / 6)


def test_bktree_finds_near_neighbours():
    tree = BKTree(["house", "horse", "mouse", "rabbit", "elephant"])
    got = dict(tree.query("hosue", 2))
    assert got.get("house") == 1
    assert "elephant" not in got


# --------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------

def test_fuzzy_jaccard_bounds():
    a = np.array([1.0, 0.5, 0.0])
    assert fuzzy_jaccard(a, a) == pytest.approx(1.0)
    assert fuzzy_jaccard(a, np.zeros(3)) == pytest.approx(0.0)
    assert 0 < fuzzy_jaccard(a, np.array([0.5, 1.0, 0.0])) < 1


def test_hierarchy_jaccard_gives_sibling_credit():
    """Siblings are partially similar; a flat metric cannot express that."""
    h = toy_hierarchy()
    dog = h.zeros(3); dog[h.index("dog", 3)] = 1.0
    cat = h.zeros(3); cat[h.index("cat", 3)] = 1.0
    qual = h.zeros(3); qual[h.index("lex:quality", 3)] = 1.0

    assert fuzzy_jaccard(dog, cat) == pytest.approx(0.0)      # no shared leaf
    sib = hierarchy_jaccard(dog, cat, h, 3)
    far = hierarchy_jaccard(dog, qual, h, 3)
    assert sib > far, "siblings must score above unrelated nodes"
    assert sib > 0.0


# --------------------------------------------------------------------------
# Corpus / tokenisation
# --------------------------------------------------------------------------

def test_contractions_expand():
    """Contractions were the top 'uncovered' types until this was fixed."""
    flat = [t for s in tokenize("I don't think it's Brown's. I'll go.") for t in s]
    assert "not" in flat and "will" in flat
    assert not any("'" in t for t in flat)


def test_truncate_vocabulary_drops_rare_types():
    sents = [["a", "b", "c"], ["a", "b"], ["a"]]
    c = Corpus("t", sents, ["a", "b", "c"], {"a": 3, "b": 2, "c": 1})
    small = c.truncate_vocabulary(2)
    assert small.vocabulary == ["a", "b"]
    assert all("c" not in s for s in small.sentences)


# --------------------------------------------------------------------------
# WordNet-backed hierarchy (skipped when the corpus is absent)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wn_corpus():
    try:
        from nltk.corpus import wordnet
        wordnet.synsets("dog")
    except Exception:  # pragma: no cover
        pytest.skip("wordnet data not downloaded")
    return wordnet


def test_wordnet_paths_stay_inside_lexname(wn_corpus):
    h, lemma_synsets = build_wordnet_hierarchy(["dog", "cat", "run", "happy"],
                                               n_levels=6, wn=wn_corpus)
    dog_key = "wn:dog.n.01"
    assert dog_key in h.nodes
    path = h.nodes[dog_key].path
    assert path[0] == ROOT_KEY and path[1] == "pos:n"
    assert path[2] == "lex:noun.animal"
    # Every interior synset on the path shares the leaf's lexname, which is what
    # makes the path a coherent chain rather than a walk through "entity".
    for key in path[3:]:
        assert wn_corpus.synset(key[3:]).lexname() == "noun.animal"


def test_wordnet_rollup_exact_on_real_hierarchy(wn_corpus):
    h, _ = build_wordnet_hierarchy(
        ["dog", "cat", "wolf", "run", "walk", "happy", "sad"], n_levels=6,
        wn=wn_corpus)
    rng = np.random.default_rng(0)
    finest = h.n_levels - 1
    for _ in range(5):
        leaf = rng.random(h.width(finest)).astype(np.float32)
        levels = h.enforce_subsumption({finest: leaf})
        for lo in range(finest):
            for mid in range(lo + 1, finest):
                assert np.allclose(levels[lo], h.rollup(levels[mid], mid, lo),
                                   atol=1e-6), f"L{lo} != rollup(L{mid})"


def test_widths_decrease_toward_the_root(wn_corpus):
    h, _ = build_wordnet_hierarchy(["dog", "cat", "wolf", "happy", "run"],
                                   n_levels=6, wn=wn_corpus)
    widths = h.widths()
    assert widths[0] == 1
    assert widths == sorted(widths), f"widths must be non-decreasing: {widths}"
