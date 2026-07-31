"""Isolate F construction: per-position _context_vector vs table gather. 3 reps each."""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from flm.fuzzyembed.corpus import load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.firstorder import ContextClassMiner

NPOS = 300_000
c = load_corpus('narrative'); train, _ = c.split(test_frac=0.2, seed=0)
cand = c.vocabulary[:3000]
emb, _ = build_embedder(c, train_lexical=False, verbose=False)
f = FuzzySequenceModel(emb, level=2, window=2, n_outputs=12, use_syntax=True,
                       lexeme_top_k=200, vocabulary=cand)
j = JointNextTokenRanker(f, window=2, n_negatives=8, max_rules=20000, max_order=2,
                         beam=6000, lexeme_side='ctx', dtype=np.float32)
j.fit(train, cand, max_positions=6000, verbose=False)
m = ContextClassMiner(j, cand, counts=c.counts, alpha=0.5, min_mass=20.0, max_order=2)
sents = [s for s in train.sentences if len(s) > 2]
order = np.random.default_rng(11).permutation(len(sents))
n = 0
for si in order:
    for i in range(2, len(sents[si])):
        if m.index.get(sents[si][i]) is not None:
            n += 1
            if n >= NPOS: break
    if n >= NPOS: break
print(f"n = {n}", flush=True)


def old_way():
    F = np.empty((n, j.cand_offset_), dtype=np.float32)
    r = 0
    for si in order:
        sent = sents[si]
        for i in range(2, len(sent)):
            if m.index.get(sent[i]) is None:
                continue
            F[r] = j._context_vector(sent, i)[:j.cand_offset_]
            r += 1
            if r >= n: break
        if r >= n: break
    return F


ctx_table, ctx_id, _ = j._tables(train, m.words)
boundary = len(ctx_table) - 1
n_slots, base = j.n_slots(), ctx_table.shape[1]


def new_way(contig):
    ids = np.empty((n, n_slots), dtype=np.intp)
    r = 0
    for si in order:
        sent = sents[si]
        for i in range(2, len(sent)):
            if m.index.get(sent[i]) is None:
                continue
            for k, t in enumerate(j.slot_tokens(sent, i)):
                ids[r, k] = ctx_id.get(t, boundary) if t else boundary
            r += 1
            if r >= n: break
        if r >= n: break
    if contig:                      # gather each slot into its own contiguous block, then hstack
        return np.hstack([ctx_table[ids[:, k]] for k in range(n_slots)])
    F = np.empty((n, n_slots * base), dtype=np.float32)
    for k in range(n_slots):
        np.take(ctx_table, ids[:, k], axis=0, out=F[:, k * base:(k + 1) * base])
    return F


for name, fn in (("per-position _context_vector", old_way),
                 ("table gather, strided out=", lambda: new_way(False)),
                 ("table gather, hstack", lambda: new_way(True))):
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); F = fn(); ts.append(time.perf_counter() - t0)
    print(f"{name:<32} {min(ts):>6.2f}s  (reps {' '.join(f'{t:.2f}' for t in ts)})",
          flush=True)
print("DONE", flush=True)
