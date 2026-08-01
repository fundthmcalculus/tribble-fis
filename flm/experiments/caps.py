"""E32.1/E32.2: lift the caps that E31's speedup made affordable.

`max_classes=3000` and `top_singles=140` were both cost-driven choices, and E29.5's frontier was
still improving at the class cap. Protocol identical to E26-E29 so numbers are comparable:
1M-token corpus, same split, positions >= 32, bigram and trigram controls.
"""
import sys, time, resource
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
ESTPOS = 624_325
c = load_corpus('narrative'); train, test = c.split(test_frac=0.2, seed=0)
cand = c.vocabulary[:NCAND]
emb, _ = build_embedder(c, train_lexical=False, verbose=False)
f = FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                       lexeme_top_k=K, vocabulary=cand)
j = JointNextTokenRanker(f, window=W, n_negatives=8, max_rules=20000, max_order=2,
                         beam=6000, lexeme_side='ctx', dtype=np.float32)
j.fit(train, cand, max_positions=TRAINPOS, verbose=False)
g0 = FuzzyGenerator(j, cand, counts=c.counts, seed=1)
lm2 = NgramLM(order=2).fit(train, g0.words)
lm3 = NgramLM(order=3).fit(train, g0.words)
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
gold = np.asarray(gold); rows = np.arange(len(gold))
PB2 = np.vstack([lm2.distribution(x, g0.words) for x in ctxs])
PB3 = np.vstack([lm3.distribution(x, g0.words) for x in ctxs])
ppl = lambda M: float(np.exp(-np.mean(np.log(np.maximum(M[rows, gold], 1e-12)))))
best_mix = lambda PF, PB: min(((ppl(l * PF + (1 - l) * PB), round(float(l), 1))
                               for l in np.arange(0, 1.01, 0.1)), key=lambda t: t[0])
print(f"bigram {ppl(PB2):.1f}   trigram {ppl(PB3):.1f}   "
      f"E29 best first-order 219.9\n", flush=True)

# Round 1 found `max_classes` was never the binding cap -- unlimited admits only 4,646 classes,
# which is simply how many pass `min_mass`. The supply is set by `top_singles` (the seed pool) and
# `min_mass` (the floor), so round 2 pushes those two and watches for E26's dilution turning point.
CASES = [
    ("seeds 140, mass 20  (E29)", dict(top_singles=140, min_mass=20.0)),
    ("seeds 280, mass 20", dict(top_singles=280, min_mass=20.0)),
    ("seeds 560, mass 20", dict(top_singles=560, min_mass=20.0)),
    ("seeds 280, mass 8", dict(top_singles=280, min_mass=8.0)),
    ("seeds 560, mass 8", dict(top_singles=560, min_mass=8.0)),
    ("seeds 560, mass 3", dict(top_singles=560, min_mass=3.0)),
]
print(f"{'condition':<32}{'classes':>9}{'params':>10}{'alone':>8}{'vs3g':>8}{'lam':>5}"
      f"{'fit_s':>7}{'GB':>6}", flush=True)
for label, kw in CASES:
    m = ContextClassMiner(j, cand, counts=c.counts, alpha=0.5, max_classes=10**9,
                          max_order=2, n_jobs=4, **kw)
    t0 = time.perf_counter(); m.fit(train, max_positions=ESTPOS)
    dt = time.perf_counter() - t0
    P = np.vstack([m.distribution(x) for x in ctxs])
    a = ppl(P); m3, l3 = best_mix(P, PB3)
    gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"{label:<32}{len(m.ctx_cols):>9}{m.sparse_parameters(20):>10}{a:>8.1f}"
          f"{m3:>8.1f}{l3:>5.1f}{dt:>7.0f}{gb:>6.1f}", flush=True)
print("DONE", flush=True)
