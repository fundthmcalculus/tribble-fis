"""E33: does the first-order model want a wider context, where the zero-order model did not?

E32.2 exhausted the seed pool -- at window 2 there are only 522 context features and the quality
curve was still climbing when they ran out. More supply means a wider window or more lexeme dims.
E26 found width monotonically harmful, but on the *zero-order* model, and that class of conclusion
already inverted once (E23.4 -> E29.3).

All caps lifted so feature supply is the only variable. 300,000 estimation positions, so rows are
comparable to each other but not to E32's 624,325-position numbers.
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

NPOS, TRAINPOS, NCAND, MIN_CTX, ESTPOS = 1000, 6000, 3000, 32, 300_000
c = load_corpus('narrative'); train, test = c.split(test_frac=0.2, seed=0)
cand = c.vocabulary[:NCAND]
emb, _ = build_embedder(c, train_lexical=False, verbose=False)

# Shared held-out positions and n-gram controls, built once from a window-2 model.
f0 = FuzzySequenceModel(emb, level=2, window=2, n_outputs=12, use_syntax=True,
                        lexeme_top_k=200, vocabulary=cand)
j0 = JointNextTokenRanker(f0, window=2, n_negatives=8, lexeme_side='ctx', dtype=np.float32)
j0.fit(train, cand, max_positions=TRAINPOS, verbose=False)
g0 = FuzzyGenerator(j0, cand, counts=c.counts, seed=1)
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
print(f"controls: bigram {ppl(PB2):.1f}  trigram {ppl(PB3):.1f}   "
      f"(constant across rows; if they move, the comparison is broken)\n", flush=True)

print(f"{'window':>7}{'lex_k':>7}{'ctxfeat':>9}{'classes':>9}{'alone':>8}{'vs3g':>8}"
      f"{'lam':>5}{'fit_s':>7}{'GB':>6}", flush=True)
for W, K in ((2, 200), (3, 200), (4, 200), (2, 500), (3, 500)):
    try:
        f = FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                               lexeme_top_k=K, vocabulary=cand)
        j = JointNextTokenRanker(f, window=W, n_negatives=8, max_rules=20000, max_order=2,
                                 beam=6000, lexeme_side='ctx', dtype=np.float32)
        j.fit(train, cand, max_positions=TRAINPOS, verbose=False)
        m = ContextClassMiner(j, cand, counts=c.counts, alpha=0.5, min_mass=8.0,
                              max_order=2, top_singles=10**6, max_classes=10**9, n_jobs=4)
        t0 = time.perf_counter(); m.fit(train, max_positions=ESTPOS)
        dt = time.perf_counter() - t0
        P = np.vstack([m.distribution(x) for x in ctxs])
        a = ppl(P); m3, l3 = best_mix(P, PB3)
        gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f"{W:>7}{K:>7}{j.cand_offset_:>9}{len(m.ctx_cols):>9}{a:>8.1f}{m3:>8.1f}"
              f"{l3:>5.1f}{dt:>7.0f}{gb:>6.1f}", flush=True)
        del m, P, j, f
    except MemoryError:
        print(f"{W:>7}{K:>7}   OUT OF MEMORY", flush=True)
print("DONE", flush=True)
