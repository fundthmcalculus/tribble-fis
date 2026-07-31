"""A/B the linguistic parameter space against the WordNet semantic+syntax space."""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from flm.fuzzyembed.corpus import load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.lexical import FuzzyLexicon
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.generate import FuzzyGenerator
from flm.fuzzyembed.baselines import NgramLM, ngram_perplexity
from flm.fuzzytok.featuriser import (
    ParameterFeaturiser, CombinedFeaturiser, build_parameter_featuriser)

NPOS, TRAINPOS, K = 600, 5000, 200
c = load_corpus('tiny'); train, test = c.split(test_frac=0.2, seed=0)
emb, _ = build_embedder(c, max_types=3000, train_lexical=False, verbose=False)
vocab = c.vocabulary[:3000]
lex = FuzzyLexicon(vocab, counts=c.counts)

def wn_featuriser(k=K):
    f = FuzzySequenceModel(emb, level=2, window=2, n_outputs=12, use_syntax=True,
                           lexeme_top_k=k, vocabulary=vocab)
    f._out_names_ = None
    return f

def lp_featuriser(k=K, fuzzy_readings=True):
    return build_parameter_featuriser(
        c, vocab, lemma_synsets=emb.senses.lemma_synsets, lexeme_top_k=k,
        head_size=500, lexicon=lex, use_fuzzy_readings=fuzzy_readings)

cases = [
    ("WordNet sem+syntax (baseline)",      lambda: wn_featuriser()),
    ("linguistic parameters",              lambda: lp_featuriser()),
    ("linguistic params, no fuzzy tok",    lambda: lp_featuriser(fuzzy_readings=False)),
    ("combined (WN + params)",             lambda: CombinedFeaturiser(
                                                wn_featuriser(k=0), lp_featuriser())),
]

print(f"{'representation':<34}{'dims':>6}{'ppl':>9}{'rules':>7}{'fit_s':>7}", flush=True)
rows = []
for label, make in cases:
    f = make()
    j = JointNextTokenRanker(f, window=2, n_negatives=8, max_rules=2500,
                             max_order=2, beam=800, lexeme_side='ctx')
    t0 = time.perf_counter()
    j.fit(train, vocab, max_positions=TRAINPOS, verbose=False)
    dt = time.perf_counter() - t0
    g = FuzzyGenerator(j, vocab, counts=c.counts, seed=1)
    r = g.perplexity(test, max_positions=NPOS)
    dims = len(f._output_names())
    print(f"{label:<34}{dims:>6}{r['perplexity']:>9.1f}"
          f"{len(j.model_.rules_):>7}{dt:>7.0f}", flush=True)
    rows.append((label, r['perplexity'], j, g, f))

allowed = rows[0][3].words
for order in (1, 2, 3):
    lm = NgramLM(order=order).fit(train, allowed)
    ppl = ngram_perplexity(lm, test, allowed, 2, max_positions=NPOS)['perplexity']
    print(f"{f'{order}-gram (same data, tuned)':<34}{'-':>6}{ppl:>9.1f}{'-':>7}{'-':>7}",
          flush=True)

best = min(rows, key=lambda r: r[1])
print(f"\nbest representation: {best[0]} (ppl {best[1]:.1f})", flush=True)
print("\n--- sample rules from the parameter space ---", flush=True)
lp_row = next(r for r in rows if r[0] == "linguistic parameters")
shown = [x for x in lp_row[2].model_.rules_ if len(x.features) == 2][:10]
for rr in shown:
    print("  " + rr.render("P(next)"), flush=True)
print("DONE", flush=True)
