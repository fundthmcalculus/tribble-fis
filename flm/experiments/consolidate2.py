"""E29 follow-ups: push the data axis to exhaustion, and fix the grammaticality metric.

E29.3 found class estimation is strongly data-bound and had NOT flattened at 300,000 positions
(standalone 456.6 -> 308.2 -> 244.6). So the headline number depends on where that curve stops.

E29.6's metric failed: real text, fuzzy generation, and a bigram all scored 100% on
"share of adjacent category pairs the corpus produces". With ~17 categories there are only ~289
possible pairs and nearly all occur somewhere in 20,000 sentences, so the metric cannot
discriminate anything. Replaced with **category-sequence perplexity** under a category bigram
fitted on training text -- a graded measure that a degenerate sequence cannot saturate.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from collections import defaultdict
from flm.fuzzyembed.corpus import load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.generate import FuzzyGenerator
from flm.fuzzyembed.firstorder import ContextClassMiner
from flm.fuzzyembed.baselines import NgramLM
from flm.fuzzyembed.syntax import SYNTAX_CATEGORIES, SyntaxTagger

NPOS, TRAINPOS, NCAND, K, MIN_CTX, W = 1000, 6000, 3000, 200, 32, 2
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
best_mix = lambda PF, PB: min(((ppl(l * PF + (1 - l) * PB), round(float(l), 1))
                               for l in np.arange(0, 1.01, 0.1)), key=lambda t: t[0])
print(f"bigram {ppl(PB2):.1f}   trigram {ppl(PB3):.1f}\n", flush=True)

print("=== E29.3b data axis to exhaustion ===", flush=True)
print(f"{'positions':>10}{'classes':>8}{'alone':>8}{'vs2g':>8}{'lam':>5}"
      f"{'vs3g':>8}{'lam':>5}{'fit_s':>7}", flush=True)
last = None
for npos in (300_000, 600_000, 1_200_000):
    t0 = time.perf_counter()
    m = ContextClassMiner(j, cand_vocab, counts=c.counts, alpha=0.5, min_mass=20.0,
                          max_order=2).fit(train, max_positions=npos)
    dt = time.perf_counter() - t0
    P = np.vstack([m.distribution(x) for x in ctxs])
    a = ppl(P)
    m2, l2 = best_mix(P, PB2)
    m3, l3 = best_mix(P, PB3)
    print(f"{m.n_positions_:>10}{len(m.ctx_cols):>8}{a:>8.1f}{m2:>8.1f}{l2:>5.1f}"
          f"{m3:>8.1f}{l3:>5.1f}{dt:>7.0f}", flush=True)
    last = (m, P)
    if m.n_positions_ < npos:
        print(f"  (corpus exhausted at {m.n_positions_} positions)", flush=True)
        break

m, PF = last
best3 = min(((ppl(a * PF + b * PB2 + (1 - a - b) * PB3), round(float(a), 1),
              round(float(b), 1))
             for a in np.arange(0, 1.01, 0.1) for b in np.arange(0, 1.01 - a, 0.1)),
            key=lambda t: t[0])
print(f"\n3-way fuzzy/bigram/trigram: ppl={best3[0]:.1f} at fuzzy={best3[1]}, "
      f"bigram={best3[2]}, trigram={round(1-best3[1]-best3[2], 1)}", flush=True)

# ---------------------------------------------------------------- E29.6b
print("\n=== E29.6b category-sequence perplexity (the metric that discriminates) ===",
      flush=True)
tagger = SyntaxTagger(getattr(emb.senses, "lemma_synsets", {}) or {})
cats = list(SYNTAX_CATEGORIES)
_cache = {}


def cat_of(tok):
    v = _cache.get(tok)
    if v is None:
        a = tagger.tag(tok)
        v = cats[int(np.argmax(a))] if a.max() > 0 else "NONE"
        _cache[tok] = v
    return v


# Category bigram on training text.
cc, ct = defaultdict(float), defaultdict(float)
for s in train.sentences[:30000]:
    cs = [cat_of(t) for t in s]
    for a, b in zip(cs, cs[1:]):
        cc[(a, b)] += 1.0
        ct[a] += 1.0
V = len(cats) + 1


def cat_ppl(seqs):
    nll, n = 0.0, 0
    for toks in seqs:
        cs = [cat_of(t) for t in toks]
        for a, b in zip(cs, cs[1:]):
            p = (cc.get((a, b), 0.0) + 0.5) / (ct.get(a, 0.0) + 0.5 * V)
            nll -= np.log(p); n += 1
    return float(np.exp(nll / max(n, 1)))


real = [s for s in test.sentences[:400] if len(s) > 4]
fuzzy = [m.generate(p, n_tokens=14)[len(p):] for p in PROMPTS for _ in (0,)]
fuzzy += [m.generate(p, n_tokens=14, seed=s)[len(p):] for s in (2, 3) for p in PROMPTS]
bg = []
rg = np.random.default_rng(3)
for _ in range(3):
    for p in PROMPTS:
        toks = list(p)
        for _ in range(14):
            d = lm2.distribution(toks, g0.words)
            toks.append(g0.words[rg.choice(len(g0.words), p=d / d.sum())])
        bg.append(toks[len(p):])
uni = []
pu = np.array([max(c.counts.get(w, 1), 1) for w in g0.words], dtype=float)
pu /= pu.sum()
for _ in range(18):
    uni.append([g0.words[i] for i in rg.choice(len(g0.words), size=14, p=pu)])
print("  lower is more syntactically plausible; real text is the target", flush=True)
print(f"    real held-out text     {cat_ppl(real):>7.2f}", flush=True)
print(f"    first-order fuzzy      {cat_ppl(fuzzy):>7.2f}", flush=True)
print(f"    bigram, same data      {cat_ppl(bg):>7.2f}", flush=True)
print(f"    unigram (floor)        {cat_ppl(uni):>7.2f}", flush=True)

print("\n=== samples ===", flush=True)
for p in PROMPTS:
    print(f"  {' '.join(p)} | {' '.join(m.generate(p, n_tokens=14)[len(p):])}", flush=True)
print("\n=== classes ===", flush=True)
print(m.explain(["he", "did"]), flush=True)
print(m.explain(["the", "little"]), flush=True)
print("DONE", flush=True)
