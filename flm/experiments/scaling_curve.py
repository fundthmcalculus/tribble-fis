"""Produce ONE internally consistent scaling curve for the E29 write-up.

The E29.3 table was assembled from two runs with different `backoff` settings -- the 60K/150K/300K
points came from the parent-backoff config and 20K/600K/624K from unigram. The two agree by 300K
(244.6 vs 244.2) but differ a lot at 20K (735.2 vs 601.2), so plotting them as a single curve
would be misleading. This refits the missing points under one configuration and emits JSON.

Also recomputes the class-count frontier from the *same* model, rather than from whichever model
happened to be in scope at the end of the previous script.
"""
import sys, json, time
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
ALPHA, MIN_MASS, BACKOFF = 0.5, 20.0, "unigram"
OUT = "/tmp/claude-0/-home-user-tribble-fis/f164b19e-5483-5d37-886d-cc2a0a0c3bda/scratchpad/curve.json"

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
        if len(gold) >= NPOS:
            break
    if len(gold) >= NPOS:
        break
gold = np.asarray(gold); rows = np.arange(len(gold))
PB2 = np.vstack([lm2.distribution(x, g0.words) for x in ctxs])
PB3 = np.vstack([lm3.distribution(x, g0.words) for x in ctxs])
ppl = lambda M: float(np.exp(-np.mean(np.log(np.maximum(M[rows, gold], 1e-12)))))
P0 = np.vstack([g0.distribution(x) for x in ctxs])


def best_mix(PF, PB):
    return min(((ppl(l * PF + (1 - l) * PB), round(float(l), 1))
                for l in np.arange(0, 1.01, 0.1)), key=lambda t: t[0])


out = {"bigram": ppl(PB2), "trigram": ppl(PB3), "zero_order": ppl(P0),
       "alpha": ALPHA, "backoff": BACKOFF, "n_eval": int(len(gold)),
       "scaling": [], "frontier": []}
print(f"bigram {out['bigram']:.1f} trigram {out['trigram']:.1f} "
      f"zero-order {out['zero_order']:.1f}", flush=True)

for npos in (20_000, 60_000, 150_000, 300_000):
    t0 = time.perf_counter()
    m = ContextClassMiner(j, cand_vocab, counts=c.counts, alpha=ALPHA,
                          min_mass=MIN_MASS, max_order=2,
                          backoff=BACKOFF).fit(train, max_positions=npos)
    P = np.vstack([m.distribution(x) for x in ctxs])
    a = ppl(P)
    m2, l2 = best_mix(P, PB2)
    m3, l3 = best_mix(P, PB3)
    row = {"positions": int(m.n_positions_), "classes": len(m.ctx_cols), "alone": a,
           "mix_bigram": m2, "lam_bigram": l2, "mix_trigram": m3, "lam_trigram": l3,
           "fit_s": round(time.perf_counter() - t0, 1)}
    out["scaling"].append(row)
    print(f"  {row}", flush=True)
    last = m

# Frontier from the same model, so the two panels are consistent.
full = last.P, last.ctx_cols, last.info_gain_, last.n_ctx, last.mass_
for n_keep in (25, 50, 100, 200, 400, 800, 1600, len(full[1])):
    if n_keep > len(full[1]):
        continue
    order = np.argsort(full[2] * full[4])[::-1][:n_keep]
    last.P, last.ctx_cols = full[0][order], [full[1][k] for k in order]
    last.info_gain_, last.n_ctx, last.mass_ = full[2][order], full[3][order], full[4][order]
    P = np.vstack([last.distribution(x) for x in ctxs])
    m2, l2 = best_mix(P, PB2)
    out["frontier"].append({"classes": int(n_keep), "params": int(n_keep * 21),
                            "alone": ppl(P), "mix_bigram": m2, "lam": l2})
    print(f"  frontier {out['frontier'][-1]}", flush=True)
last.P, last.ctx_cols, last.info_gain_, last.n_ctx, last.mass_ = full

with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1)
print(f"wrote {OUT}\nDONE", flush=True)
