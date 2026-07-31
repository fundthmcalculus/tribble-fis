"""E29: execute the plan recorded in ../LOG.md (E29.1-E29.6).

E29.1 held-out class selection      E29.4 stronger mixture partner (honesty check)
E29.2 hierarchical parent backoff   E29.5 parameter frontier
E29.3 class-estimation data         E29.6 a number for "not grammatical"

Protocol identical to E26/E27/E28: 1M-token corpus, same split, positions >= 32, bigram control.
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
from flm.fuzzyembed.syntax import CLOSED_CLASS, SyntaxTagger

NPOS, TRAINPOS, NCAND, K, MIN_CTX, W = 1000, 6000, 3000, 200, 32, 2
FUNCTION = set().union(*CLOSED_CLASS.values())
PROMPTS = [["the", "little"], ["she", "was"], ["he", "did"], ["the", "old"],
           ["there", "was"], ["it", "is"]]

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
print(c.summary(), flush=True)

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


def ppl(M):
    return float(np.exp(-np.mean(np.log(np.maximum(M[rows, gold], 1e-12)))))


def best_mix(PF, PB):
    return min(((ppl(l * PF + (1 - l) * PB), round(float(l), 1))
                for l in np.arange(0, 0.95, 0.1)), key=lambda t: t[0])


print(f"\nbigram {ppl(PB2):.1f}   trigram {ppl(PB3):.1f}   "
      f"zero-order {ppl(np.vstack([g0.distribution(x) for x in ctxs])):.1f}", flush=True)

# ---------------------------------------------------------------- E29.1 / E29.2
print("\n=== E29.1 held-out selection x E29.2 parent backoff (alpha sweep) ===", flush=True)
print(f"{'sel':>5}{'backoff':>9}{'alpha':>7}{'classes':>8}{'alone':>8}"
      f"{'mix2':>8}{'lam':>5}{'fit_s':>7}", flush=True)
grid, cache = [], {}
for sel in (0.0, 0.3):
    for backoff in ("unigram", "parent"):
        for alpha in (0.5, 50.0):
            t0 = time.perf_counter()
            m = ContextClassMiner(j, cand_vocab, counts=c.counts, alpha=alpha,
                                  min_mass=20.0, max_order=2, selection_holdout=sel,
                                  backoff=backoff).fit(train, max_positions=20000)
            dt = time.perf_counter() - t0
            P = np.vstack([m.distribution(x) for x in ctxs])
            a, (mx, lam) = ppl(P), best_mix(P, PB2)
            print(f"{sel:>5.1f}{backoff:>9}{alpha:>7.1f}{len(m.ctx_cols):>8}{a:>8.1f}"
                  f"{mx:>8.1f}{lam:>5.1f}{dt:>7.0f}", flush=True)
            grid.append((sel, backoff, alpha, a, mx, lam))
            cache[(sel, backoff, alpha)] = (m, P)

best_alone = min(grid, key=lambda t: t[3])
best_mixed = min(grid, key=lambda t: t[4])
print(f"\nbest standalone: sel={best_alone[0]} {best_alone[1]} alpha={best_alone[2]} "
      f"-> {best_alone[3]:.1f}   (E28 best was 343.5)", flush=True)
print(f"best mixture:    sel={best_mixed[0]} {best_mixed[1]} alpha={best_mixed[2]} "
      f"-> {best_mixed[4]:.1f} @ lam={best_mixed[5]}   (E28 best was 256.2)", flush=True)

# ---------------------------------------------------------------------- E29.3
print("\n=== E29.3 class-estimation data ===", flush=True)
cfg = dict(alpha=best_mixed[2], min_mass=20.0, max_order=2,
           selection_holdout=best_mixed[0], backoff=best_mixed[1])
for npos in (60000, 150000, 300000):
    t0 = time.perf_counter()
    m = ContextClassMiner(j, cand_vocab, counts=c.counts, **cfg).fit(train,
                                                                    max_positions=npos)
    P = np.vstack([m.distribution(x) for x in ctxs])
    mx, lam = best_mix(P, PB2)
    print(f"  positions={npos:>7}  classes={len(m.ctx_cols):>5}  alone={ppl(P):>7.1f}  "
          f"mix={mx:>7.1f} @lam={lam:.1f}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    bestm = (m, P)

# ---------------------------------------------------------------------- E29.4
m, PF = bestm
print("\n=== E29.4 stronger partner (honesty check) ===", flush=True)
for name, PB in (("bigram", PB2), ("trigram", PB3)):
    mx, lam = best_mix(PF, PB)
    base = ppl(PB)
    print(f"  vs {name:<8} alone={base:>7.1f}  best mix={mx:>7.1f} @lam={lam:.1f}  "
          f"gain={100*(base-mx)/base:>5.1f}%", flush=True)
best3 = min(((ppl(a * PF + b * PB2 + (1 - a - b) * PB3), round(float(a), 1), round(float(b), 1))
             for a in np.arange(0, 0.8, 0.1) for b in np.arange(0, 0.9 - a, 0.1)),
            key=lambda t: t[0])
print(f"  3-way fuzzy/bigram/trigram: ppl={best3[0]:.1f} at "
      f"fuzzy={best3[1]}, bigram={best3[2]}, trigram={round(1-best3[1]-best3[2],1)}", flush=True)

# ---------------------------------------------------------------------- E29.5
print("\n=== E29.5 parameter frontier (fewest classes retaining the gain) ===", flush=True)
full = m.P, m.ctx_cols, m.info_gain_, m.n_ctx, m.mass_
for n_keep in (50, 200, 800, len(full[1])):
    order = np.argsort(full[2] * full[4])[::-1][:n_keep]
    m.P, m.ctx_cols = full[0][order], [full[1][k] for k in order]
    m.info_gain_, m.n_ctx, m.mass_ = full[2][order], full[3][order], full[4][order]
    P = np.vstack([m.distribution(x) for x in ctxs])
    mx, lam = best_mix(P, PB2)
    print(f"  classes={n_keep:>5}  params~{n_keep*21:>7}  alone={ppl(P):>7.1f}  "
          f"mix={mx:>7.1f} @lam={lam:.1f}", flush=True)
m.P, m.ctx_cols, m.info_gain_, m.n_ctx, m.mass_ = full

# ---------------------------------------------------------------------- E29.6
print("\n=== E29.6 a number for 'not grammatical' ===", flush=True)
tagger = SyntaxTagger(getattr(emb.senses, "lemma_synsets", {}) or {})
from flm.fuzzyembed.syntax import SYNTAX_CATEGORIES
cats = list(SYNTAX_CATEGORIES)


def cat_of(tok):
    v = tagger.tag(tok)
    return cats[int(np.argmax(v))] if v.max() > 0 else "NONE"


seen = set()
for s in train.sentences[:20000]:
    cs = [cat_of(t) for t in s]
    seen.update(zip(cs, cs[1:]))


def plausibility(token_seqs):
    ok = tot = 0
    for toks in token_seqs:
        cs = [cat_of(t) for t in toks]
        for pair in zip(cs, cs[1:]):
            tot += 1
            ok += pair in seen
    return ok / max(tot, 1)


real = [s for s in test.sentences[:400] if len(s) > 4]
fuzzy = [m.generate(p, n_tokens=14)[len(p):] for p in PROMPTS]
bigram_gen = []
rg = np.random.default_rng(3)
for p in PROMPTS:
    toks = list(p)
    for _ in range(14):
        d = lm2.distribution(toks, g0.words)
        toks.append(g0.words[rg.choice(len(g0.words), p=d / d.sum())])
    bigram_gen.append(toks[len(p):])
print(f"  adjacent-category-pair plausibility (share of pairs the corpus produces):", flush=True)
print(f"    real held-out text     {100*plausibility(real):>5.1f}%", flush=True)
print(f"    first-order fuzzy      {100*plausibility(fuzzy):>5.1f}%", flush=True)
print(f"    bigram, same data      {100*plausibility(bigram_gen):>5.1f}%", flush=True)

print("\n=== samples (best config) ===", flush=True)
for p, out in zip(PROMPTS, fuzzy):
    print(f"  {' '.join(p)} | {' '.join(out)}", flush=True)
print("\n=== classes ===", flush=True)
print(m.explain(["he", "did"]), flush=True)
print(m.explain(["the", "little"]), flush=True)
print("DONE", flush=True)
