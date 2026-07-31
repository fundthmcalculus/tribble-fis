"""E28.3: is the standalone first-order failure just undersmoothing + selection bias?

`explain` showed `IF prev2:=the AND prev1:QUANTIFIER -> jackal(0.105)` -- "jackal" as the top
prediction after "the little". Two suspected causes, both testable:
  * alpha=0.5 pseudo-counts is far too little smoothing for a class with ~20 firing units
  * information gain REWARDS low entropy, so class selection is biased toward exactly the
    under-observed classes that look sharp by accident (the E26 lesson, again)
Sweep both. If standalone perplexity moves a long way, the architecture is fine and the
estimator was wrong; if it does not, word-level consequents genuinely do not work standalone.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from flm.fuzzyembed.corpus import load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.generate import FuzzyGenerator
from flm.fuzzyembed.firstorder import ContextClassMiner
from flm.fuzzyembed.baselines import NgramLM

NPOS, TRAINPOS, NCAND, K, MIN_CTX, W = 1000, 6000, 3000, 200, 32, 2
c = load_corpus('narrative')
train, test = c.split(test_frac=0.2, seed=0)
cand_vocab = c.vocabulary[:NCAND]
emb, _ = build_embedder(c, train_lexical=False, verbose=False)
f = FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                       lexeme_top_k=K, vocabulary=cand_vocab)
j = JointNextTokenRanker(f, window=W, n_negatives=8, max_rules=20000, max_order=2,
                         beam=6000, lexeme_side='ctx', dtype=np.float32)
j.fit(train, cand_vocab, max_positions=TRAINPOS, verbose=False)
g0 = FuzzyGenerator(j, cand_vocab, counts=c.counts, seed=1)
lm = NgramLM(order=2).fit(train, g0.words)
index = {w: i for i, w in enumerate(g0.words)}
rng = np.random.default_rng(7)
sents = [s for s in test.sentences if len(s) > MIN_CTX]
ctxs, gold = [], []
for si in rng.permutation(len(sents)):
    sent = sents[si]
    for i in range(MIN_CTX, len(sent)):
        if sent[i] not in index:
            continue
        ctxs.append(sent[:i]); gold.append(index[sent[i]])
        if len(gold) >= NPOS:
            break
    if len(gold) >= NPOS:
        break
gold = np.asarray(gold); rows = np.arange(len(gold))
PB = np.vstack([lm.distribution(x, g0.words) for x in ctxs])
pplB = float(np.exp(-np.mean(np.log(np.maximum(PB[rows, gold], 1e-12)))))
print(f"bigram {pplB:.1f}\n")
print(f"{'alpha':>7}{'min_mass':>10}{'classes':>9}{'ppl':>9}{'bestmix':>9}{'lam':>5}",
      flush=True)
for alpha in (0.5, 5.0, 50.0, 200.0):
    for min_mass in (20.0, 150.0):
        fo = ContextClassMiner(j, cand_vocab, counts=c.counts, alpha=alpha,
                               min_mass=min_mass, max_order=2)
        fo.fit(train, max_positions=20000)
        P = np.vstack([fo.distribution(x) for x in ctxs])
        p = float(np.exp(-np.mean(np.log(np.maximum(P[rows, gold], 1e-12)))))
        mix = [(float(np.exp(-np.mean(np.log(np.maximum(
            (l * P + (1 - l) * PB)[rows, gold], 1e-12))))), round(l, 1))
            for l in np.arange(0, 0.85, 0.1)]
        bm = min(mix)
        print(f"{alpha:>7.1f}{min_mass:>10.0f}{len(fo.ctx_cols):>9}{p:>9.1f}"
              f"{bm[0]:>9.1f}{bm[1]:>5.1f}", flush=True)
print("DONE", flush=True)
