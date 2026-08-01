"""How much of the mixture gain is lambda overfitting the evaluation positions?

Every mixture number in this project picked lambda by minimising perplexity on the SAME held-out
positions it then reported -- a 1-D grid search over 11 values on 1,000 points. The standalone
numbers are unaffected (no lambda), but the mixture numbers are optimistic by some amount, and
"some amount" is not good enough. Measure it: split the positions, tune on one half, report on the
other, and compare against the tuned-on-test figure.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from flm.fuzzyembed.corpus import load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.generate import FuzzyGenerator
from flm.fuzzyembed.firstorder import ContextClassMiner
from flm.fuzzyembed.baselines import NgramLM

NPOS, MIN_CTX = 2000, 32          # 2,000 so each half is the usual 1,000
c = load_corpus('narrative'); train, test = c.split(test_frac=0.2, seed=0)
cand = c.vocabulary[:3000]
emb, _ = build_embedder(c, train_lexical=False, verbose=False)
f = FuzzySequenceModel(emb, level=2, window=2, n_outputs=12, use_syntax=True,
                       lexeme_top_k=200, vocabulary=cand)
j = JointNextTokenRanker(f, window=2, n_negatives=8, max_rules=20000, max_order=2,
                         beam=6000, lexeme_side='ctx', dtype=np.float32)
j.fit(train, cand, max_positions=6000, verbose=False)
g0 = FuzzyGenerator(j, cand, counts=c.counts, seed=1)
lm2 = NgramLM(order=2).fit(train, g0.words)
lm3 = NgramLM(order=3).fit(train, g0.words)
m = ContextClassMiner(j, cand, counts=c.counts, alpha=0.5, min_mass=8.0, max_order=2,
                      top_singles=10**6, max_classes=10**9, n_jobs=4)
m.fit(train, max_positions=1_200_000)

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
        if len(gold) >= NPOS: break
    if len(gold) >= NPOS: break
gold = np.asarray(gold)
PF = np.vstack([m.distribution(x) for x in ctxs])
PB2 = np.vstack([lm2.distribution(x, g0.words) for x in ctxs])
PB3 = np.vstack([lm3.distribution(x, g0.words) for x in ctxs])
half = len(gold) // 2
dev, tst = np.arange(half), np.arange(half, len(gold))


def ppl(M, idx):
    return float(np.exp(-np.mean(np.log(np.maximum(M[idx, gold[idx]], 1e-12)))))


LAMS = np.arange(0, 1.01, 0.1)
print(f"n = {len(gold)} positions, split {len(dev)} dev / {len(tst)} test\n", flush=True)
print(f"{'partner':<10}{'tuned-on-test':>15}{'honest (dev->test)':>21}{'lam':>6}{'inflation':>11}",
      flush=True)
for name, PB in (("bigram", PB2), ("trigram", PB3)):
    on_test = min(ppl(l * PF + (1 - l) * PB, tst) for l in LAMS)
    l_dev = min(LAMS, key=lambda l: ppl(l * PF + (1 - l) * PB, dev))
    honest = ppl(l_dev * PF + (1 - l_dev) * PB, tst)
    print(f"{name:<10}{on_test:>15.1f}{honest:>21.1f}{l_dev:>6.1f}"
          f"{100 * (honest - on_test) / on_test:>10.2f}%", flush=True)
print(f"\nstandalone (no lambda, unaffected): {ppl(PF, tst):.1f} on the test half", flush=True)
print("DONE", flush=True)
