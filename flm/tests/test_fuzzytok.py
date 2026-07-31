"""Tests for the fuzzy tokenizer and the linguistic parameter encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flm.fuzzytok.params import (  # noqa: E402
    AFFECT, FEATS, PARAMETERS, SEM, SHAPE, UPOS, LinguisticParameterEncoder,
)
from flm.fuzzytok.tokenizer import FuzzyTokenizer  # noqa: E402


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------

def tok() -> FuzzyTokenizer:
    head = ["the", "and", "a", "happy", "receive", "hope", "run", "little", "house"]
    return FuzzyTokenizer(head_words=head,
                          stems=set(head) | {"quick", "kind", "friend"})


def test_head_word_is_exact_and_unambiguous():
    t = tok().tokenize("the")
    assert t.best().degree == 1.0
    assert [u.label() for u in t.best().units] == ["the"]


def test_affix_decomposition_prefers_known_stems():
    """`quickly` -> quick + -ly, and the stem-in-vocabulary reading must win."""
    readings = tok().tokenize("quickly").readings
    labels = [[u.label() for u in r.units] for r in readings]
    assert ["quick", "-ly"] in labels
    top = readings[0]
    assert top.degree >= 0.9 and "vocabulary" in top.source


def test_prefix_and_suffix_together():
    t = tok().tokenize("unkindly")
    labels = [[u.label() for u in r.units] for r in t.readings]
    assert ["un-", "kind", "-ly"] in labels, labels


def test_dropped_e_is_restored():
    """`hoping` -> hope + -ing: the one orthographic repair worth special-casing."""
    labels = [[u.label() for u in r.units] for r in tok().tokenize("hoping").readings]
    assert ["hope", "-ing"] in labels, labels


def test_multiple_readings_are_kept_with_degrees():
    """The point of a *fuzzy* tokenizer: competing segmentations coexist, graded."""
    t = tok().tokenize("friendless")
    assert len(t.readings) >= 1
    degs = [r.degree for r in t.readings]
    assert degs == sorted(degs, reverse=True)
    assert all(0.0 < d <= 1.0 for d in degs)


def test_unknown_string_falls_back_to_ngrams_and_never_empties():
    t = tok().tokenize("xqzzyv")
    assert t.readings, "must always produce some representation"
    assert t.best().units[0].kind == "ngram"
    assert t.best().degree <= 0.4, "n-gram readings must never outrank lexical ones"


def test_unit_degrees_aggregate_by_max_not_sum():
    """Two readings containing a unit is a disjunction, not accumulated evidence."""
    t = tok().tokenize("unkindly")
    for label, d in t.unit_degrees().items():
        assert 0.0 <= d <= 1.0, (label, d)


def test_unit_inventory_stays_small():
    """This is the 'simple vocabulary' -- head words plus nameable affixes only."""
    inv = tok().unit_inventory()
    assert len(inv) < 100
    assert "-ly" in inv and "un-" in inv


def test_fuzzy_lexical_access_gives_misspellings_partial_membership():
    """Robustness is intrinsic to tokenization here, not a preprocessing step."""
    from flm.fuzzyembed.lexical import FuzzyLexicon
    vocab = ["receive", "relieve", "believe", "house", "the", "happy"]
    lex = FuzzyLexicon(vocab, threshold=0.3)
    t = FuzzyTokenizer(head_words=vocab, stems=set(vocab), lexicon=lex)
    reading = t.tokenize("recieve")
    labels = {u.label() for r in reading.readings for u in r.units}
    assert "receive" in labels, reading.render()
    assert 0.0 < reading.best().degree < 1.0, "a misspelling must not be certain"


# --------------------------------------------------------------------------
# Parameter space
# --------------------------------------------------------------------------

def test_parameter_space_is_small_and_unique():
    assert len(PARAMETERS) == len(set(PARAMETERS))
    assert len(PARAMETERS) < 100, "the point is a *small* named basis"
    for block in (UPOS, FEATS, SHAPE, SEM, AFFECT):
        assert set(block) <= set(PARAMETERS)
    assert len(UPOS) == 17, "the 17 Universal Dependencies UPOS tags"


def test_closed_class_maps_to_upos():
    e = LinguisticParameterEncoder(polarity={})
    i = {p: k for k, p in enumerate(PARAMETERS)}
    assert e.encode("the")[i["DET"]] == 1.0
    assert e.encode("of")[i["ADP"]] == 1.0
    assert e.encode("not")[i["Polarity=Neg"]] == 1.0
    assert e.encode("his")[i["Poss=Yes"]] == 1.0


def test_capitalisation_signals_proper_noun():
    """Names were the dominant coverage gap; orthography recovers most of them."""
    e = LinguisticParameterEncoder(polarity={})
    i = {p: k for k, p in enumerate(PARAMETERS)}
    up = e.encode("Margery")
    assert up[i["Shape=Capitalised"]] == 1.0
    assert up[i["PROPN"]] > 0.0
    # Graded, not certain: sentence-initial words are capitalised too.
    assert up[i["PROPN"]] < 1.0
    assert e.encode("margery")[i["Shape=Capitalised"]] == 0.0


def test_suffix_evidence_is_graded_not_certain():
    e = LinguisticParameterEncoder(polarity={})
    i = {p: k for k, p in enumerate(PARAMETERS)}
    v = e.encode("walking")
    assert 0.0 < v[i["Aspect=Prog"]] < 1.0
    assert 0.0 < v[i["VERB"]] < 1.0, "'-ing' is evidence, not proof"


def test_evaluation_axis_is_bipolar_around_neutral():
    e = LinguisticParameterEncoder(polarity={"good": 1.0, "bad": -1.0})
    i = {p: k for k, p in enumerate(PARAMETERS)}
    assert e.encode("good")[i["Affect=Evaluation"]] > 0.9
    assert e.encode("bad")[i["Affect=Evaluation"]] < 0.1
    assert e.encode("table")[i["Affect=Evaluation"]] == pytest.approx(0.5)


def test_activity_is_left_unset_rather_than_invented():
    """Osgood's activity axis needs elicited ratings; guessing it would be invention."""
    e = LinguisticParameterEncoder(polarity={})
    i = {p: k for k, p in enumerate(PARAMETERS)}
    assert e.encode("running")[i["Affect=Activity"]] == 0.0


def test_encode_fuzzy_propagates_partial_membership():
    """A misspelling should partially activate the correct unit's parameters."""
    from flm.fuzzyembed.lexical import FuzzyLexicon
    vocab = ["happy", "the", "house", "receive"]
    lex = FuzzyLexicon(vocab, threshold=0.3)
    t = FuzzyTokenizer(head_words=vocab, stems=set(vocab), lexicon=lex)
    e = LinguisticParameterEncoder(polarity={"happy": 1.0})
    i = {p: k for k, p in enumerate(PARAMETERS)}

    clean = e.encode_fuzzy(t.tokenize("happy"))
    noisy = e.encode_fuzzy(t.tokenize("hapy"))
    assert clean[i["Affect=Evaluation"]] > 0.9
    # Partial, not zero and not full -- the graded path survived tokenization.
    assert 0.0 < noisy[i["Affect=Evaluation"]] <= clean[i["Affect=Evaluation"]]


def test_encoder_output_stays_in_unit_interval():
    e = LinguisticParameterEncoder(polarity={"good": 1.0})
    for w in ("the", "Margery", "walking", "good", "xqzzyv", "12"):
        v = e.encode(w)
        assert v.min() >= 0.0 and v.max() <= 1.0, w


def test_competing_stem_variants_both_survive():
    """`hoping` must offer BOTH `hop + -ing` and `hope + -ing` when both stems exist.

    Regression test: the first version short-circuited on the bare stem, so with "hop"
    in the vocabulary the `hope` reading was never emitted — premature commitment,
    which is exactly what a fuzzy tokenizer exists to avoid.
    """
    t = FuzzyTokenizer(head_words=["hop", "hope", "the"],
                       stems={"hop", "hope", "the"}, max_readings=6)
    labels = [[u.label() for u in r.units] for r in t.tokenize("hoping").readings]
    assert ["hop", "-ing"] in labels, labels
    assert ["hope", "-ing"] in labels, labels


def test_lookup_is_case_folded_but_surface_is_preserved():
    """A capitalised in-vocabulary word must still hit the head set exactly.

    Regression test: without case folding, "Margery" missed the head set and fell
    through to fuzzy lexical access, matching itself at 0.73 and risking being
    outranked by an unrelated neighbour. Case belongs in the parameter space, not in
    the lookup key.
    """
    t = FuzzyTokenizer(head_words=["margery", "the"], stems={"margery", "the"})
    tk = t.tokenize("Margery")
    assert tk.surface == "Margery", "surface form must survive for shape features"
    assert tk.best().degree == 1.0
    assert [u.label() for u in tk.best().units] == ["margery"]


# --------------------------------------------------------------------------
# Featuriser: interface compatibility with the joint ranker
# --------------------------------------------------------------------------

def _featuriser(lexeme_top_k=0, vocab=None):
    from flm.fuzzytok.featuriser import ParameterFeaturiser
    vocab = vocab or ["the", "and", "happy", "rabbit", "not"]
    t = FuzzyTokenizer(head_words=vocab, stems=set(vocab))
    e = LinguisticParameterEncoder(polarity={"happy": 1.0})
    return ParameterFeaturiser(t, e, lexeme_top_k=lexeme_top_k, vocabulary=vocab)


def test_featuriser_matches_the_ranker_interface():
    """The ranker needs exactly _token_vector, _output_names, and lexemes."""
    f = _featuriser(lexeme_top_k=3)
    assert callable(f._token_vector) and callable(f._output_names)
    assert len(f._output_names()) == len(f._token_vector("happy"))
    assert isinstance(f.lexemes, list) and len(f.lexemes) == 3


def test_featuriser_widths_are_consistent_across_tokens():
    f = _featuriser(lexeme_top_k=2)
    widths = {len(f._token_vector(t)) for t in ("the", "happy", "", "xqzzyv", "Rabbit")}
    assert len(widths) == 1, widths


def test_boundary_is_distinguishable_from_unknown():
    """Padding must not look like 'a token with no recognised parameters'."""
    from flm.fuzzytok.featuriser import BOUNDARY
    f = _featuriser()
    names = f._output_names()
    i = names.index(BOUNDARY)
    assert f._token_vector("")[i] == 1.0
    assert f._token_vector("xqzzyv")[i] == 0.0


def test_identity_block_is_last_so_the_candidate_mask_works():
    """JointNextTokenRanker.cand_vector masks the FINAL n_lex dims; order matters."""
    f = _featuriser(lexeme_top_k=3, vocab=["the", "and", "happy", "rabbit"])
    names = f._output_names()
    assert all(n.startswith("=") for n in names[-3:]), names[-3:]
    v = f._token_vector("the")
    assert v[-3:].sum() == 1.0, "the identity dim for 'the' should fire"


def test_combined_featuriser_concatenates_and_prefixes():
    from flm.fuzzytok.featuriser import CombinedFeaturiser
    a, b = _featuriser(), _featuriser(lexeme_top_k=2)
    c = CombinedFeaturiser(a, b, first_tag="x", second_tag="y")
    assert len(c._output_names()) == len(c._token_vector("happy"))
    names = c._output_names()
    assert any(n.startswith("x:") for n in names)
    assert any(n.startswith("y:") for n in names)
    # Identity dims must still be last, unprefixed.
    assert all(n.startswith("=") for n in names[-2:]), names[-2:]


def test_combined_rejects_identity_on_both_sides():
    """Two identity blocks would break the candidate mask's offset assumption."""
    from flm.fuzzytok.featuriser import CombinedFeaturiser
    with pytest.raises(ValueError):
        CombinedFeaturiser(_featuriser(lexeme_top_k=2), _featuriser(lexeme_top_k=2))


def test_fuzzy_readings_ablation_changes_the_vector():
    """use_fuzzy_readings=False is the ablation isolating the tokenizer's contribution."""
    from flm.fuzzyembed.lexical import FuzzyLexicon
    from flm.fuzzytok.featuriser import ParameterFeaturiser
    vocab = ["happy", "the", "rabbit"]
    lex = FuzzyLexicon(vocab, threshold=0.3)
    t = FuzzyTokenizer(head_words=vocab, stems=set(vocab), lexicon=lex)
    e = LinguisticParameterEncoder(polarity={"happy": 1.0})
    on = ParameterFeaturiser(t, e, use_fuzzy_readings=True)
    off = ParameterFeaturiser(t, e, use_fuzzy_readings=False)
    # A misspelling should reach the correct unit only through the fuzzy readings.
    assert on._token_vector("hapy").sum() > off._token_vector("hapy").sum()
