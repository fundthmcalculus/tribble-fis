"""Evaluate multi-threaded / multi-process training, after the two vectorisations.

The order of these measurements matters. Profiling first (E23) showed the cost was NOT where
E22.3 assumed: 88% of fit was cold per-token-type featurisation, dominated by a pure-Python
rollup loop. So the parallelism question has to be re-asked *after* fixing that, or it
optimises a cost that no longer exists.

Four things measured here:
  1. the new time breakdown -- what is even left to parallelise
  2. BLAS thread scaling -- the rule-search GEMMs are already threaded, so the question is
     how much that is actually worth, not whether to add it
  3. thread-level parallelism over token types -- expected to fail (GIL), measured anyway
  4. process-level parallelism over token types -- works, but on what is now a small cost

Run with ``OMP_NUM_THREADS`` unset; the script sets thread counts itself via threadpoolctl.
"""
import sys, time, os
sys.path.insert(0, '.')
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from flm.fuzzyembed.corpus import load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.rules import MembershipRuleRegressor

TRAINPOS, K, W = 5000, 200, 2
c = load_corpus('tiny'); train, _ = c.split(test_frac=0.2, seed=0)
emb, _ = build_embedder(c, max_types=3000, train_lexical=False, verbose=False)
vocab = c.vocabulary[:3000]


def fresh():
    return FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                              lexeme_top_k=K, vocabulary=vocab)


def ranker(f):
    return JointNextTokenRanker(f, window=W, n_negatives=8, max_rules=2500,
                                max_order=2, beam=800, lexeme_side='ctx')


print("=== 1. time breakdown after vectorising rollup and build ===", flush=True)
f = fresh()
t0 = time.perf_counter()
for w in vocab:
    f._token_vector(w)
t_feat = time.perf_counter() - t0
j = ranker(f); j.feature_names_ = j._names()
t0 = time.perf_counter(); X, y = j.build(train, vocab, TRAINPOS)
t_build = time.perf_counter() - t0
cand_idx = set(range(j.cand_offset_, X.shape[1]))
t0 = time.perf_counter()
m = MembershipRuleRegressor(max_rules=2500, max_order=2, must_include=cand_idx,
                            order_quota=j.order_quota, beam=j.beam,
                            seed_features=set(range(j.cand_offset_))).fit(
                                X, y, j.feature_names_)
t_search = time.perf_counter() - t0
tot = t_feat + t_build + t_search
print(f"  featurise {len(vocab)} types  {t_feat:>6.2f}s  {100*t_feat/tot:>4.0f}%   "
      f"(was 58.8s before vectorising rollup)")
print(f"  build() X={X.shape}   {t_build:>6.2f}s  {100*t_build/tot:>4.0f}%")
print(f"  rule search ({len(m.rules_)} rules)  {t_search:>6.2f}s  {100*t_search/tot:>4.0f}%")
print(f"  TOTAL                   {tot:>6.2f}s   (was 65.9s)", flush=True)

print("\n=== 2. how much is BLAS threading in the rule search worth? ===", flush=True)
try:
    from threadpoolctl import threadpool_limits
    for nt in (1, 2, 4):
        with threadpool_limits(limits=nt, user_api='blas'):
            t0 = time.perf_counter()
            MembershipRuleRegressor(max_rules=2500, max_order=2, must_include=cand_idx,
                                    order_quota=j.order_quota, beam=j.beam,
                                    seed_features=set(range(j.cand_offset_))).fit(
                                        X, y, j.feature_names_)
            dt = time.perf_counter() - t0
        print(f"  rule search, {nt} BLAS thread(s): {dt:>6.2f}s", flush=True)
except ImportError:
    print("  threadpoolctl not installed; skipped", flush=True)

print("\n=== 3. threads over token types (expected to fail: GIL) ===", flush=True)
fs = fresh()
t0 = time.perf_counter()
for w in vocab:
    fs._token_vector(w)
serial = time.perf_counter() - t0
for nthread in (2, 4):
    ft = fresh()
    shards = [vocab[i::nthread] for i in range(nthread)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=nthread) as ex:
        list(ex.map(lambda sh: [ft._token_vector(w) for w in sh], shards))
    dt = time.perf_counter() - t0
    print(f"  {nthread} threads: {dt:>6.2f}s  speedup {serial/dt:.2f}x", flush=True)

print("\n=== 4. processes over token types (fork, COW-inherited embedder) ===", flush=True)
import multiprocessing as mp
_G = {}


def _worker(shard):
    return [_G['f']._token_vector(w) for w in shard]


if __name__ == "__main__":
    ctx = mp.get_context("fork")
    for nproc in (2, 4):
        _G['f'] = fresh()
        shards = [vocab[i::nproc] for i in range(nproc)]
        t0 = time.perf_counter()
        with ctx.Pool(nproc) as pool:
            got = pool.map(_worker, shards)
        dt = time.perf_counter() - t0
        print(f"  {nproc} procs:   {dt:>6.2f}s  speedup {serial/dt:.2f}x", flush=True)
    _G['f'] = fresh()
    shards = [vocab[i::4] for i in range(4)]
    with ctx.Pool(4) as pool:
        got = pool.map(_worker, shards)
    ref = fresh()
    ok = all(np.array_equal(v, ref._token_vector(w))
             for sh, g in zip(shards, got) for w, v in zip(sh, g))
    print(f"  parallel result == serial, exactly: {ok}", flush=True)
    print("DONE", flush=True)
