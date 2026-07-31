"""The size/quality frontier: how small can the rule base get before perplexity degrades?

Goal is a model trainable on local compute, so report the three costs that actually decide
that -- learned parameters, fit wall-clock, and inference wall-clock -- next to perplexity,
at matched data and split.
"""
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
from flm.fuzzytok.featuriser import build_parameter_featuriser

NPOS, TRAINPOS, K, W = 600, 5000, 200, 2
c = load_corpus('tiny'); train, test = c.split(test_frac=0.2, seed=0)
emb, _ = build_embedder(c, max_types=3000, train_lexical=False, verbose=False)
vocab = c.vocabulary[:3000]
lex = FuzzyLexicon(vocab, counts=c.counts)


def params(model):
    """Learned parameters = one consequent per rule + its antecedent indices.

    Antecedent indices are learned (which features, from the search) so they count; the
    membership degrees themselves are *not* learned -- that is the whole reason this model
    is small (E10: inputs are already memberships, so no membership function is fitted).
    """
    return sum(len(r.features) + 1 for r in model.rules_)


def run(label, make, max_rules):
    f = make()
    j = JointNextTokenRanker(f, window=W, n_negatives=8, max_rules=max_rules,
                             max_order=2, beam=800, lexeme_side='ctx')
    t0 = time.perf_counter(); j.fit(train, vocab, max_positions=TRAINPOS, verbose=False)
    fit_s = time.perf_counter() - t0
    g = FuzzyGenerator(j, vocab, counts=c.counts, seed=1)
    t0 = time.perf_counter(); r = g.perplexity(test, max_positions=NPOS)
    infer_s = time.perf_counter() - t0
    n_rules = len(j.model_.rules_)
    print(f"{label:<14}{max_rules:>7}{n_rules:>7}{params(j.model_):>9}"
          f"{r['perplexity']:>9.1f}{fit_s:>8.1f}{1000*infer_s/NPOS:>9.2f}", flush=True)
    return r['perplexity']


def wn():
    return FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                              lexeme_top_k=K, vocabulary=vocab)


def lp():
    return build_parameter_featuriser(c, vocab, lemma_synsets=emb.senses.lemma_synsets,
                                      lexeme_top_k=K, head_size=500, lexicon=lex)


print(f"{'space':<14}{'budget':>7}{'rules':>7}{'params':>9}{'ppl':>9}"
      f"{'fit_s':>8}{'ms/tok':>9}", flush=True)
for budget in (100, 250, 500, 1000, 2500):
    run("params (lp)", lp, budget)
print(flush=True)
for budget in (100, 500, 2500):
    run("wordnet (wn)", wn, budget)

print("\n--- reference: n-gram tables on the same data ---", flush=True)
allowed = FuzzyGenerator(JointNextTokenRanker(lp(), window=W).fit(
    train, vocab, max_positions=200, verbose=False), vocab, counts=c.counts).words
for order in (1, 2, 3):
    lm = NgramLM(order=order).fit(train, allowed)
    n_par = sum(len(d) for d in lm.counts) + sum(len(d) for d in lm.totals)
    ppl = ngram_perplexity(lm, test, allowed, W, max_positions=NPOS)['perplexity']
    print(f"  {order}-gram  ppl={ppl:>7.1f}  stored counts={n_par}", flush=True)
print("DONE", flush=True)
