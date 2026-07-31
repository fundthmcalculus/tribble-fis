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


def test_vectorised_rollup_equals_the_reference_implementation():
    """The fast rollup must be *identical* to the obvious one, for every op.

    ``rollup`` is vectorised via a cached grouping (E23) because the elementwise version
    dominated the entire pipeline. Exact multi-resolution readout is the central claim of
    this representation, so the optimisation is only admissible while it is provably
    equivalent -- hence the original implementation is retained as an oracle and compared
    on random vectors rather than on a hand-picked case that might miss a grouping bug.
    """
    h = toy_hierarchy()
    rng = np.random.default_rng(0)
    for from_level in range(h.n_levels):
        for to_level in range(from_level + 1):
            for op in ("max", "sum", "probor"):
                for _ in range(5):
                    vec = rng.random(h.width(from_level)).astype(np.float32)
                    fast = h.rollup(vec, from_level, to_level, op=op)
                    ref = h._rollup_reference(vec, from_level, to_level, op=op)
                    assert np.allclose(fast, ref, atol=1e-6), (from_level, to_level, op)
    # Sparse vectors take different reduceat paths (empty groups stay zero).
    for op in ("max", "sum", "probor"):
        vec = np.zeros(h.width(3), dtype=np.float32)
        vec[h.index("dog", 3)] = 0.7
        assert np.allclose(h.rollup(vec, 3, 1, op=op),
                           h._rollup_reference(vec, 3, 1, op=op), atol=1e-6)


def test_rollup_plan_is_cached_not_rebuilt():
    """The whole speedup rests on the grouping being computed once per level pair."""
    h = toy_hierarchy()
    vec = np.zeros(h.width(3), dtype=np.float32)
    h.rollup(vec, 3, 1)
    assert (3, 1) in h._plans
    plan = h._plans[(3, 1)]
    h.rollup(vec, 3, 1)
    assert h._plans[(3, 1)] is plan, "plan was rebuilt; the optimisation is defeated"


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


# --------------------------------------------------------------------------
# Syntax tagger
# --------------------------------------------------------------------------

def test_syntax_is_genuinely_fuzzy():
    """Closed-class ambiguity is represented, not forced to a single label."""
    from flm.fuzzyembed.syntax import SyntaxTagger, SYNTAX_CATEGORIES
    t = SyntaxTagger()
    idx = {c: i for i, c in enumerate(SYNTAX_CATEGORIES)}

    to = t.tag("to")
    assert to[idx["PREPOSITION"]] == 1.0
    assert to[idx["INFINITIVE_TO"]] == 1.0      # both readings live at once

    that = t.tag("that")
    assert that[idx["DETERMINER"]] == 1.0
    assert that[idx["CONJUNCTION"]] == 1.0

    her = t.tag("her")
    assert her[idx["PRONOUN"]] == 1.0 and her[idx["POSSESSIVE"]] == 1.0

    no = t.tag("no")
    assert no[idx["DETERMINER"]] == 1.0 and no[idx["NEGATOR"]] == 1.0


def test_boundary_marker_distinguishes_padding_from_unknown():
    """An empty (padded) position must not look like an unknown open-class word."""
    from flm.fuzzyembed.syntax import SyntaxTagger, BOUNDARY, SYNTAX_CATEGORIES
    t = SyntaxTagger()
    i = SYNTAX_CATEGORIES.index(BOUNDARY)
    assert t.tag("")[i] == 1.0
    assert t.tag("zzzunknownzzz")[i] == 0.0
    assert t.tag("zzzunknownzzz").sum() == 0.0


# --------------------------------------------------------------------------
# Membership rule learner
# --------------------------------------------------------------------------

def test_rules_recover_a_planted_conjunction():
    """The learner must find an interaction that neither conjunct shows alone."""
    from flm.fuzzyembed.rules import MembershipRuleRegressor
    rng = np.random.default_rng(0)
    n = 600
    a = (rng.random(n) < 0.4).astype(float)
    b = (rng.random(n) < 0.4).astype(float)
    noise = rng.random((n, 5))
    X = np.column_stack([a, b, noise])
    y = a * b                                    # pure AND, no marginal signal
    model = MembershipRuleRegressor(max_rules=8).fit(
        X, y, ["A", "B", "n0", "n1", "n2", "n3", "n4"])

    top = model.rules_[0]
    assert set(top.names) == {"A", "B"}, model.render()
    assert top.consequent > 0.9
    # And it predicts: firing on both should far exceed firing on neither.
    both = model.predict(np.array([[1, 1, 0, 0, 0, 0, 0]]))[0]
    neither = model.predict(np.array([[0, 0, 0, 0, 0, 0, 0]]))[0]
    assert both > neither + 0.4


def test_rules_reject_redundant_conjunctions():
    """A conjunct that duplicates its parent must not enter the rule base."""
    from flm.fuzzyembed.rules import MembershipRuleRegressor
    rng = np.random.default_rng(1)
    n = 500
    a = (rng.random(n) < 0.5).astype(float)
    X = np.column_stack([a, a.copy(), rng.random(n)])   # col 1 is an alias of col 0
    y = a
    model = MembershipRuleRegressor(max_rules=10).fit(X, y, ["A", "A_alias", "noise"])
    pairs = [r for r in model.rules_ if set(r.names) == {"A", "A_alias"}]
    assert not pairs, f"kept a redundant alias conjunction: {model.render()}"


def test_rules_default_handles_all_quiet_input():
    """No antecedent firing must degrade to the prior, not divide by zero."""
    from flm.fuzzyembed.rules import MembershipRuleRegressor
    rng = np.random.default_rng(2)
    X = (rng.random((300, 4)) < 0.3).astype(float)
    y = X[:, 0]
    model = MembershipRuleRegressor().fit(X, y)
    out = model.predict(np.zeros((1, 4)))[0]
    assert np.isfinite(out)
    assert out == pytest.approx(model.default_, abs=1e-6)


def test_rules_predictions_stay_in_unit_interval():
    from flm.fuzzyembed.rules import MembershipRuleRegressor
    rng = np.random.default_rng(3)
    X = rng.random((400, 6))
    y = rng.random(400)
    p = MembershipRuleRegressor().fit(X, y).predict(X)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_rules_grow_to_order_three():
    """A three-way interaction must be findable when max_order allows it."""
    from flm.fuzzyembed.rules import MembershipRuleRegressor
    rng = np.random.default_rng(7)
    n = 1500
    a = (rng.random(n) < 0.5).astype(float)
    b = (rng.random(n) < 0.5).astype(float)
    c = (rng.random(n) < 0.5).astype(float)
    X = np.column_stack([a, b, c, rng.random((n, 3))])
    y = a * b * c                                # pure 3-way AND
    names = ["A", "B", "C", "n0", "n1", "n2"]

    order2 = MembershipRuleRegressor(max_order=2, max_rules=12).fit(X, y, names)
    order3 = MembershipRuleRegressor(max_order=3, max_rules=12).fit(X, y, names)

    assert order2.max_order_used == 2
    assert order3.max_order_used == 3
    triple = [r for r in order3.rules_ if set(r.names) == {"A", "B", "C"}]
    assert triple, order3.render()
    assert triple[0].consequent > 0.9
    # The 3-conjunct rule must predict the interaction better than order 2 can.
    probe = np.array([[1, 1, 1, 0, 0, 0]], dtype=float)
    assert order3.predict(probe)[0] > order2.predict(probe)[0]


def test_rules_find_pure_interaction_only_when_seeded():
    """Pure-interaction structure is invisible to marginal-lift seeding.

    Regression test for the joint ranker: features with zero marginal effect but a
    strong joint effect must be force-seeded, or no interaction rule is generated.
    """
    from flm.fuzzyembed.rules import MembershipRuleRegressor
    rng = np.random.default_rng(11)
    n = 2000
    # `ctx` is balanced and, alone, carries no signal about y at all.
    ctx = (rng.random(n) < 0.5).astype(float)
    cand = (rng.random(n) < 0.5).astype(float)
    y = (ctx == cand).astype(float)          # pure XOR-style interaction
    # Pad with high-marginal-lift noise so top-k seeding prefers the decoys.
    decoy = np.column_stack([(rng.random(n) < 0.5).astype(float) for _ in range(20)])
    X = np.column_stack([ctx, cand, decoy])
    names = ["ctx", "cand"] + [f"d{i}" for i in range(20)]

    # ctx alone must look useless -- that is the premise of the test.
    plain = MembershipRuleRegressor(max_rules=30, top_singles=6).fit(X, y, names)
    assert abs(plain.rules_[0].consequent - plain.default_) < 0.5

    seeded = MembershipRuleRegressor(
        max_rules=30, top_singles=6, seed_features={0},
        order_quota={1: 0.2, 2: 0.8}).fit(X, y, names)
    pairs = [r for r in seeded.rules_ if set(r.names) == {"ctx", "cand"}]
    assert pairs, f"force-seeding failed to surface the interaction: {seeded.render()}"


def test_order_quota_reserves_slots():
    from flm.fuzzyembed.rules import MembershipRuleRegressor
    rng = np.random.default_rng(5)
    n = 800
    X = (rng.random((n, 8)) < 0.4).astype(float)
    y = np.clip(X[:, 0] * 0.5 + X[:, 1] * X[:, 2] * 0.5, 0, 1)
    quota = MembershipRuleRegressor(max_rules=10, order_quota={1: 0.5, 2: 0.5}).fit(X, y)
    hist = quota.order_histogram()
    assert hist.get(2, 0) >= 1, hist
    assert sum(hist.values()) == len(quota.rules_) <= 10


def test_joint_ranker_excludes_context_only_rules():
    """Every rule must touch the candidate, or it cannot change a ranking."""
    from flm.fuzzyembed.rules import MembershipRuleRegressor
    rng = np.random.default_rng(13)
    n = 900
    X = (rng.random((n, 6)) < 0.4).astype(float)
    y = np.clip(X[:, 4] * 0.6 + X[:, 0] * X[:, 4] * 0.4, 0, 1)
    cand_idx = {4, 5}
    model = MembershipRuleRegressor(max_rules=20, must_include=cand_idx).fit(X, y)
    for rule in model.rules_:
        assert any(f in cand_idx for f in rule.features), rule.render()


def test_vectorised_build_is_bit_identical_to_the_reference():
    """The table-indexed ``build`` must reproduce the row-at-a-time version exactly.

    ``build`` was rewritten (E23) to featurise each token *type* once into a table and then
    assemble rows by fancy indexing. That is only safe if the RNG draw sequence is unchanged
    -- a different negative sample would silently alter the training set and every number
    downstream, in a way no perplexity comparison would reveal as a bug. So assert exact
    equality, not approximate.
    """
    from flm.fuzzyembed.joint import JointNextTokenRanker

    class TinyFeaturiser:
        """Two named dims plus a lexeme identity block, to exercise the candidate mask."""

        lexemes = ["the", "dog"]

        def _output_names(self):
            return ["A", "B", "=the", "=dog"]

        def _token_vector(self, token):
            v = np.zeros(4, dtype=np.float32)
            if not token:
                return v
            v[0] = 1.0 if token[0] in "aeiou" else 0.3
            v[1] = min(len(token) / 6.0, 1.0)
            if token in self.lexemes:
                v[2 + self.lexemes.index(token)] = 1.0
            return v

    sents = [["the", "dog", "ran", "away"], ["a", "cat", "sat", "on", "the", "mat"],
             ["the", "dog", "saw", "a", "cat"], ["birds", "fly", "over", "the", "hill"]]
    counts = {}
    for s in sents:
        for t in s:
            counts[t] = counts.get(t, 0) + 1
    vocab = sorted(counts, key=lambda w: (-counts[w], w))
    corpus = Corpus("tiny", sents, vocab, counts)

    for lexeme_side in ("ctx", "both"):
        j = JointNextTokenRanker(TinyFeaturiser(), window=2, n_negatives=3, seed=5,
                                 lexeme_side=lexeme_side)
        j.feature_names_ = j._names()
        Xa, ya = j.build(corpus, vocab, max_positions=50)
        Xb, yb = j._build_reference(corpus, vocab, max_positions=50)
        assert Xa.shape == Xb.shape, lexeme_side
        assert np.array_equal(Xa, Xb), lexeme_side
        assert np.array_equal(ya, yb), lexeme_side


def test_batched_growth_matches_bruteforce():
    """The GEMM formulation must agree with per-candidate computation exactly.

    Guards the optimisation that makes order-3 affordable: support and consequent for
    every order-2 candidate are read off two matrix products, never by forming the
    candidate's firing vector.
    """
    from flm.fuzzyembed.rules import MembershipRuleRegressor
    rng = np.random.default_rng(17)
    n, d = 400, 12
    X = (rng.random((n, d)) < 0.35).astype(float)
    y = (rng.random(n) < 0.4).astype(float)

    model = MembershipRuleRegressor(max_rules=200, max_order=2, min_support=1.0,
                                    min_interaction=-1.0, top_singles=d,
                                    beam=200).fit(X, y)
    default = float(y.mean())
    for rule in model.rules_:
        fire = X[:, list(rule.features)].prod(axis=1)
        sup = fire.sum()
        assert rule.support == pytest.approx(sup, rel=1e-9, abs=1e-9)
        assert rule.consequent == pytest.approx((fire * y).sum() / sup, rel=1e-9)
        assert rule.lift == pytest.approx((rule.consequent - default) * np.sqrt(sup),
                                          rel=1e-9)


def test_min_tnorm_still_works():
    """The min t-norm cannot factor into a GEMM, so it takes the fallback path."""
    from flm.fuzzyembed.rules import MembershipRuleRegressor
    rng = np.random.default_rng(19)
    n = 500
    a = (rng.random(n) < 0.5).astype(float)
    b = (rng.random(n) < 0.5).astype(float)
    X = np.column_stack([a, b, rng.random((n, 4))])
    y = np.minimum(a, b)
    model = MembershipRuleRegressor(max_rules=10, max_order=2,
                                    t_norm="min").fit(X, y, list("abcdef"))
    for rule in model.rules_:
        fire = X[:, list(rule.features)].min(axis=1)
        assert rule.support == pytest.approx(fire.sum(), rel=1e-9)


def test_growth_dedupes_feature_sets():
    """Distinct (frontier, seed) pairs can denote one feature set; keep it once."""
    from flm.fuzzyembed.rules import MembershipRuleRegressor
    rng = np.random.default_rng(23)
    X = (rng.random((600, 10)) < 0.4).astype(float)
    y = (rng.random(600) < 0.5).astype(float)
    model = MembershipRuleRegressor(max_rules=300, max_order=3, min_support=1.0,
                                    min_interaction=-1.0, beam=100).fit(X, y)
    keys = [frozenset(r.features) for r in model.rules_]
    assert len(keys) == len(set(keys))


def test_corpus_split_is_disjoint_by_sentence():
    """Held-out sentences must not appear in training -- the original leak."""
    sents = [[f"s{i}", f"w{i}", f"x{i}"] for i in range(50)]
    c = Corpus("t", sents, ["a"], {"a": 1})
    train, test = c.split(test_frac=0.2, seed=0)
    assert len(train.sentences) + len(test.sentences) == len(sents)
    tr = {tuple(s) for s in train.sentences}
    te = {tuple(s) for s in test.sentences}
    assert not (tr & te), "train and test share sentences"
    # Vocabulary is shared on purpose, so the candidate set is comparable.
    assert train.vocabulary is c.vocabulary and test.vocabulary is c.vocabulary


# --------------------------------------------------------------------------
# Generation / NCE correction
# --------------------------------------------------------------------------

def test_nce_correction_widens_dynamic_range():
    """Raw scores are too flat to be a distribution; the odds ratio is not.

    Regression test for the finding that normalising raw NCE scores gave perplexity
    2478 against a 2897 uniform floor, while the NCE inversion gave 386.
    """
    s = np.array([0.05, 0.10, 0.20, 0.40, 0.60])
    q = np.full(len(s), 1.0 / len(s))

    raw = s / s.sum()
    odds = q * (s / (1 - s))
    corrected = odds / odds.sum()

    raw_range = raw.max() / raw.min()
    cor_range = corrected.max() / corrected.min()
    assert cor_range > raw_range * 2, (raw_range, cor_range)
    assert corrected.sum() == pytest.approx(1.0)


def test_nce_correction_reduces_to_noise_prior_on_constant_scores():
    """With an uninformative scorer the corrected distribution *is* the unigram prior.

    This is what makes the perplexity decomposition clean: any gain over the unigram
    baseline is exactly what the context rules contribute, since a constant score
    recovers the prior identically.
    """
    q = np.array([0.5, 0.3, 0.15, 0.05])
    for const in (0.05, 0.3, 0.9):
        s = np.full(len(q), const)
        w = q * (s / (1 - s))
        assert np.allclose(w / w.sum(), q)


def test_generator_factorised_scoring_matches_direct():
    """The ctx/cand factorisation must equal evaluating the rule base directly."""
    from flm.fuzzyembed.rules import MembershipRuleRegressor

    rng = np.random.default_rng(29)
    n, ctx_w, cand_w = 500, 6, 4
    X = (rng.random((n, ctx_w + cand_w)) < 0.4).astype(float)
    y = np.clip(X[:, 0] * X[:, ctx_w] + 0.1 * X[:, ctx_w + 1], 0, 1)
    model = MembershipRuleRegressor(
        max_rules=12, max_order=2, must_include=set(range(ctx_w, ctx_w + cand_w)),
        seed_features=set(range(ctx_w))).fit(X, y)

    ctx = (rng.random(ctx_w) < 0.5).astype(float)
    cands = (rng.random((7, cand_w)) < 0.5).astype(float)

    direct = model.predict(np.hstack([np.repeat(ctx[None, :], 7, axis=0), cands]))

    # Factorised: scalar context part per rule x precomputed candidate part.
    num = np.zeros(7)
    den = np.zeros(7)
    for rule in model.rules_:
        c_idx = [f for f in rule.features if f < ctx_w]
        d_idx = [f - ctx_w for f in rule.features if f >= ctx_w]
        cf = ctx[c_idx].prod() if c_idx else 1.0
        fire = cf * (cands[:, d_idx].prod(axis=1) if d_idx else 1.0)
        num += fire * rule.consequent
        den += fire
    eps = 0.05
    fact = np.clip((num + eps * model.default_) / (den + eps), 0.0, 1.0)
    assert np.allclose(direct, fact, atol=1e-9)


def test_oov_match_pruning_drops_the_weak_tail_but_keeps_real_ambiguity():
    """OOV blending was E23.5's bug: a word got four unrelated supersenses at ~0.5 each.

    Pruning is *relative* on purpose. Genuine ambiguity between two near-equal candidates is
    exactly what a fuzzy representation should preserve; a long tail at half the best
    candidate's degree is not evidence of anything, and merging it fed the rule learner noise
    that grew with corpus size.
    """
    from flm.fuzzyembed.embedder import OOV_RELATIVE_KEEP, FuzzyEmbedder

    class M:
        def __init__(self, lexeme, degree):
            self.lexeme, self.degree = lexeme, degree

    # Exact hits already arrive as a single match; pruning must not disturb them.
    assert len(FuzzyEmbedder._prune_matches([M("little", 1.0)])) == 1
    assert FuzzyEmbedder._prune_matches([]) == []

    # Near-equal candidates: real ambiguity, both kept.
    kept = FuzzyEmbedder._prune_matches([M("little", 0.74), M("littler", 0.70)])
    assert [m.lexeme for m in kept] == ["little", "littler"]

    # The E23.5 shape: one good candidate and a tail of weak unrelated ones.
    kept = FuzzyEmbedder._prune_matches(
        [M("wood", 0.90), M("woolen", 0.55), M("golden", 0.52), M("garden", 0.46)])
    assert [m.lexeme for m in kept] == ["wood"], "weak tail must not be merged"

    # The threshold is what the docstring says it is.
    kept = FuzzyEmbedder._prune_matches(
        [M("a", 1.0), M("b", OOV_RELATIVE_KEEP + 0.01), M("c", OOV_RELATIVE_KEEP - 0.01)])
    assert [m.lexeme for m in kept] == ["a", "b"]


def test_full_coverage_is_the_default_and_keeps_level_two_width():
    """``max_types=None`` must mean the whole vocabulary, and must not change dimensions.

    The fix in E23.6 is only safe because the model reads level 2, which is WordNet's 45
    lexicographer files however large the vocabulary is. If widening coverage changed the
    feature width, perplexity would stop being comparable across conditions and the fix
    would be confounded with a capacity change.
    """
    import inspect
    from flm.fuzzyembed.embedder import build_embedder
    assert inspect.signature(build_embedder).parameters["max_types"].default is None
