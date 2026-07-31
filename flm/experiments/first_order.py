"""E28: does a word-level consequent break the categorical floor?

E27.5 concluded the ceiling is the *shape* of the consequent: a zero-order TSK rule assigns one
scalar to every word matching its antecedent, so it can say "a noun follows" but not "which
noun". This replaces the scalar with a word distribution per rule -- a fuzzy class-based LM.

Same protocol as E26/E27 so numbers drop into those tables: 1M-token corpus, same split,
TRAINPOS=6000 for rule mining, positions >= 32, bigram control.

The controls matter more than the headline here. A rule whose context side is `ctx:prev1:=the`
has p_r(w) = p(w | prev=the), which is literally a bigram row, so:
  * `no identity ctx` drops every identity-context rule -- what the *categorical* rules do alone
  * `bigram` and `zero-order` are the two things it has to beat to mean anything
  * the mixture sweep is reported too, since E19.4-style complementarity is a separate question
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from flm.fuzzyembed.corpus import load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.generate import FuzzyGenerator
from flm.fuzzyembed.firstorder import ContextClassMiner, FirstOrderConsequents
from flm.fuzzyembed.baselines import NgramLM
from flm.fuzzyembed.syntax import CLOSED_CLASS

NPOS, TRAINPOS, NCAND, K, MIN_CTX, W = 1000, 6000, 3000, 200, 32, 2
FUNCTION = set().union(*CLOSED_CLASS.values())
PROMPTS = [["the", "little"], ["she", "was"], ["he", "did"], ["the", "old"]]

c = load_corpus('narrative')
train, test = c.split(test_frac=0.2, seed=0)
cand_vocab = c.vocabulary[:NCAND]
emb, _ = build_embedder(c, train_lexical=False, verbose=False)
print(c.summary(), flush=True)

f = FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                       lexeme_top_k=K, vocabulary=cand_vocab)
j = JointNextTokenRanker(f, window=W, n_negatives=8, max_rules=20000, max_order=2,
                         beam=6000, lexeme_side='ctx', dtype=np.float32)
t0 = time.perf_counter()
j.fit(train, cand_vocab, max_positions=TRAINPOS, verbose=False)
print(f"rule base: {len(j.model_.rules_)} rules, mined in {time.perf_counter()-t0:.0f}s "
      f"(shared by every first-order condition below)\n", flush=True)

# --- shared evaluation positions --------------------------------------------
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
gold = np.asarray(gold)
rows = np.arange(len(gold))
print(f"{len(gold)} shared held-out positions\n", flush=True)


def ppl_of(dist_fn):
    P = np.vstack([dist_fn(ctx) for ctx in ctxs])
    return P, float(np.exp(-np.mean(np.log(np.maximum(P[rows, gold], 1e-12)))))


P0, ppl0 = ppl_of(lambda ctx: g0.distribution(ctx))
PB, pplB = ppl_of(lambda ctx: lm.distribution(ctx, g0.words))
print(f"{'model':<40}{'ppl':>9}{'params':>10}{'rules':>7}{'cont%':>8}{'fit_s':>8}",
      flush=True)
print(f"{'zero-order TSK (scalar consequents)':<40}{ppl0:>9.1f}"
      f"{sum(len(r.features)+1 for r in j.model_.rules_):>10}"
      f"{len(j.model_.rules_):>7}{'':>8}{'':>8}", flush=True)
print(f"{'2-gram (same data, tuned)':<40}{pplB:>9.1f}"
      f"{sum(len(d) for d in lm.counts)+sum(len(d) for d in lm.totals):>10}"
      f"{'-':>7}{'':>8}{'':>8}", flush=True)

def reuse(**kw):
    return lambda: FirstOrderConsequents(j, cand_vocab, counts=c.counts, alpha=0.5, **kw)


def mined(**kw):
    return lambda: ContextClassMiner(j, cand_vocab, counts=c.counts, alpha=0.5, **kw)


CASES = [
    # Reusing the zero-order rule base: every class is a SINGLE context feature, because
    # must_include + max_order=2 leaves exactly one context term per rule (E28.1).
    ("reuse rules, firing weights", reuse(weighting="firing")),
    ("reuse rules, infogain weights", reuse(weighting="infogain")),
    ("reuse rules, no identity ctx", reuse(weighting="infogain",
                                           exclude_identity_context=True)),
    # Mining context-only classes by information gain, which is what word-level consequents
    # actually want -- must_include was correct for scalar consequents and wrong here.
    ("mined classes, order 1", mined(max_order=1)),
    ("mined classes, order 2", mined(max_order=2)),
    ("mined classes, order 2, no identity", mined(max_order=2, exclude_identity=True)),
]
best = None
for label, make in CASES:
    t0 = time.perf_counter()
    fo = make()
    fo.fit(train, max_positions=20000)
    dt = time.perf_counter() - t0
    P, p = ppl_of(lambda ctx: fo.distribution(ctx))
    toks = []
    for pr in PROMPTS:
        toks.extend(fo.generate(pr, n_tokens=14)[len(pr):])
    cont = sum(1 for t in toks if t not in FUNCTION) / len(toks)
    print(f"{label:<40}{p:>9.1f}{fo.sparse_parameters(20):>10}"
          f"{len(fo.ctx_cols):>7}{100*cont:>7.1f}%{dt:>8.0f}", flush=True)
    if best is None or p < best[1]:
        best = (label, p, fo, P)

label, p, fo, PF = best
print(f"\nbest: {label} (ppl {p:.1f})", flush=True)

print("\n--- mixture with the bigram (complementarity, as in E19.4) ---", flush=True)
for lam in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
    M = lam * PF + (1 - lam) * PB
    q = float(np.exp(-np.mean(np.log(np.maximum(M[rows, gold], 1e-12)))))
    print(f"  lam={lam:.1f}  ppl={q:.1f}", flush=True)

print("\n=== SAMPLE TEXT (first-order) ===", flush=True)
for pr in PROMPTS + [["there", "was"], ["it", "is"]]:
    print(f"  {' '.join(pr)} | {' '.join(fo.generate(pr, n_tokens=14)[len(pr):])}",
          flush=True)

print("\n=== WHY: firing classes and their word preferences ===", flush=True)
for pr in (["the", "little"], ["he", "did"], ["she", "was", "very"]):
    print(fo.explain(pr), flush=True)
print("DONE", flush=True)
