"""Does the *smallest* rule base still carry information a bigram lacks?

E22.2 found the size/quality knee is very sharp: 100 rules / 285 learned parameters gets
within 5% of the 860-rule model. The claim that matters for local compute is not "small and
almost as good alone" but "small and still complementary" -- 285 parameters improving a
34,021-count bigram table would be the real result.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from flm.fuzzyembed.corpus import load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.generate import FuzzyGenerator
from flm.fuzzyembed.baselines import NgramLM, interpolated_perplexity

NPOS, TRAINPOS, K, W = 600, 5000, 200, 2
c = load_corpus('tiny'); train, test = c.split(test_frac=0.2, seed=0)
emb, _ = build_embedder(c, max_types=3000, train_lexical=False, verbose=False)
vocab = c.vocabulary[:3000]

for budget in (100, 250, 860):
    f = FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                           lexeme_top_k=K, vocabulary=vocab)
    j = JointNextTokenRanker(f, window=W, n_negatives=8, max_rules=budget,
                             max_order=2, beam=800, lexeme_side='ctx')
    j.fit(train, vocab, max_positions=TRAINPOS, verbose=False)
    g = FuzzyGenerator(j, vocab, counts=c.counts, seed=1)
    n_par = sum(len(r.features) + 1 for r in j.model_.rules_)
    lm = NgramLM(order=2).fit(train, g.words)
    rows = interpolated_perplexity(g, lm, test, g.words, W,
                                   lambdas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0),
                                   max_positions=NPOS)
    best = min(rows, key=lambda r: r['perplexity'])
    alone = next(r for r in rows if r['lambda'] == 1.0)['perplexity']
    bigram = next(r for r in rows if r['lambda'] == 0.0)['perplexity']
    print(f"rules={len(j.model_.rules_):>4} params={n_par:>5}  alone={alone:>7.1f}  "
          f"bigram={bigram:>7.1f}  best_mix={best['perplexity']:>7.1f} "
          f"@lam={best['lambda']:.1f}  gain={100*(bigram-best['perplexity'])/bigram:>5.2f}%",
          flush=True)
    print("   sweep: " + "  ".join(f"{r['lambda']:.1f}={r['perplexity']:.1f}"
                                   for r in rows), flush=True)
print("DONE", flush=True)
