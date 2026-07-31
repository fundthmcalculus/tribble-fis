"""Item 4b: is the parallel sweep exact, and what does it buy?"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from flm.fuzzyembed.corpus import load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.firstorder import ContextClassMiner

c = load_corpus('narrative'); train, _ = c.split(test_frac=0.2, seed=0)
cand = c.vocabulary[:3000]
emb, _ = build_embedder(c, train_lexical=False, verbose=False)
f = FuzzySequenceModel(emb, level=2, window=2, n_outputs=12, use_syntax=True,
                       lexeme_top_k=200, vocabulary=cand)
j = JointNextTokenRanker(f, window=2, n_negatives=8, max_rules=20000, max_order=2,
                         beam=6000, lexeme_side='ctx', dtype=np.float32)
j.fit(train, cand, max_positions=6000, verbose=False)

if __name__ == "__main__":
    ref = None
    for npos in (300_000, 624_325):
        for nj in (1, 2, 4):
            m = ContextClassMiner(j, cand, counts=c.counts, alpha=0.5, min_mass=20.0,
                                  max_order=2, n_jobs=nj)
            t0 = time.perf_counter(); m.fit(train, max_positions=npos)
            dt = time.perf_counter() - t0
            d = {tuple(k): m.P[i] for i, k in enumerate(m.ctx_cols)}
            note = ""
            if nj == 1:
                ref = d
            else:
                same = (set(d) == set(ref)
                        and max(np.abs(d[k] - ref[k]).max() for k in d) == 0.0)
                note = f"  bit-identical={same}"
            print(f"{npos:>7} positions  n_jobs={nj}  {dt:>6.1f}s"
                  f"  classes={len(m.ctx_cols)}{note}", flush=True)
    print("DONE", flush=True)
