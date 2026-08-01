"""E33.2/E34: why more features fail, then the headline at the best known configuration.

E33 refuted the hypothesis that first-order would want a wider context: window 2 beats 3 and 4 on
the mixed metric (194.8 / 201.4 / 212.5). Suspected mechanism -- *mass fragmentation*, not the false
discovery E26.1 wrongly blamed: more features means far more candidate classes, each firing on fewer
positions, so each distribution is smoothed harder toward the unigram and the mixture fills with
weakly-informative classes. Measured below as mean mass and mean information gain per class.

Then the final measurement at the best configuration found (E32.2): window 2, lexeme_top_k 200,
seed pool unlimited, min_mass 8, full-corpus estimation.
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

NPOS, TRAINPOS, NCAND, MIN_CTX = 1000, 6000, 3000, 32
PROMPTS = [["the", "little"], ["she", "was"], ["he", "did"], ["the", "old"],
           ["there", "was"], ["it", "is"]]
c = load_corpus('narrative'); train, test = c.split(test_frac=0.2, seed=0)
cand = c.vocabulary[:NCAND]
emb, _ = build_embedder(c, train_lexical=False, verbose=False)


def make(W, K):
    f = FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                           lexeme_top_k=K, vocabulary=cand)
    j = JointNextTokenRanker(f, window=W, n_negatives=8, max_rules=20000, max_order=2,
                             beam=6000, lexeme_side='ctx', dtype=np.float32)
    j.fit(train, cand, max_positions=TRAINPOS, verbose=False)
    return j


j = make(2, 200)
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

print("=== E33.2 mass fragmentation: is that why more features hurt? ===", flush=True)
print(f"{'window':>7}{'ctxfeat':>9}{'classes':>9}{'mean mass':>11}{'median mass':>13}"
      f"{'mean gain':>11}", flush=True)
for W in (2, 3, 4):
    jj = make(W, 200)
    mm = ContextClassMiner(jj, cand, counts=c.counts, alpha=0.5, min_mass=8.0,
                           max_order=2, top_singles=10**6, max_classes=10**9, n_jobs=4)
    mm.fit(train, max_positions=300_000)
    print(f"{W:>7}{jj.cand_offset_:>9}{len(mm.ctx_cols):>9}{mm.mass_.mean():>11.1f}"
          f"{np.median(mm.mass_):>13.1f}{mm.info_gain_.mean():>11.3f}", flush=True)
    del mm, jj

print("\n=== E34 headline at the best known configuration ===", flush=True)
t0 = time.perf_counter()
m = ContextClassMiner(j, cand, counts=c.counts, alpha=0.5, min_mass=8.0, max_order=2,
                      top_singles=10**6, max_classes=10**9, n_jobs=4)
m.fit(train, max_positions=1_200_000)
fit_s = time.perf_counter() - t0
PF = np.vstack([m.distribution(x) for x in ctxs])
a = ppl(PF)
print(f"  window 2, lexeme_top_k 200, all seeds, min_mass 8, "
      f"{m.n_positions_} positions, {len(m.ctx_cols)} classes, fit {fit_s:.0f}s", flush=True)
print(f"  standalone      {a:>7.1f}   (bigram {ppl(PB2):.1f}, trigram {ppl(PB3):.1f})", flush=True)
for name, PB in (("bigram", PB2), ("trigram", PB3)):
    b = min(((ppl(l * PF + (1 - l) * PB), round(float(l), 1))
             for l in np.arange(0, 1.01, 0.1)), key=lambda t: t[0])
    print(f"  + {name:<8}    {b[0]:>7.1f} @ lam={b[1]}   "
          f"({100*(ppl(PB)-b[0])/ppl(PB):.1f}% better than {name} alone)", flush=True)
best3 = min(((ppl(x * PF + y * PB2 + (1 - x - y) * PB3), round(float(x), 1), round(float(y), 1))
             for x in np.arange(0, 1.01, 0.1) for y in np.arange(0, 1.01 - x, 0.1)),
            key=lambda t: t[0])
print(f"  3-way           {best3[0]:>7.1f} at fuzzy={best3[1]}, bigram={best3[2]}, "
      f"trigram={round(1-best3[1]-best3[2],1)}", flush=True)
print(f"  parameters      {m.sparse_parameters(20):>7} (top-20 sparse)", flush=True)

# generation: category-sequence perplexity, as E29.6b
tagger = SyntaxTagger(getattr(emb.senses, "lemma_synsets", {}) or {})
cats = list(SYNTAX_CATEGORIES); _cc = {}
def cat_of(t):
    v = _cc.get(t)
    if v is None:
        a2 = tagger.tag(t); v = cats[int(np.argmax(a2))] if a2.max() > 0 else "NONE"; _cc[t] = v
    return v
cc, ct = defaultdict(float), defaultdict(float)
for s in train.sentences[:30000]:
    cs = [cat_of(t) for t in s]
    for x, y in zip(cs, cs[1:]):
        cc[(x, y)] += 1.0; ct[x] += 1.0
V = len(cats) + 1
def cat_ppl(seqs):
    nll, n = 0.0, 0
    for toks in seqs:
        cs = [cat_of(t) for t in toks]
        for x, y in zip(cs, cs[1:]):
            nll -= np.log((cc.get((x, y), 0.0) + 0.5) / (ct.get(x, 0.0) + 0.5 * V)); n += 1
    return float(np.exp(nll / max(n, 1)))
real = [s for s in test.sentences[:400] if len(s) > 4]
fz = [m.generate(p, n_tokens=14, seed=s)[len(p):] for s in (1, 2, 3) for p in PROMPTS]
bg, rg = [], np.random.default_rng(3)
for _ in range(3):
    for p in PROMPTS:
        tk = list(p)
        for _ in range(14):
            d = lm2.distribution(tk, g0.words); tk.append(g0.words[rg.choice(len(g0.words), p=d/d.sum())])
        bg.append(tk[len(p):])
print(f"\n  category-sequence ppl: real {cat_ppl(real):.2f}  fuzzy {cat_ppl(fz):.2f}  "
      f"bigram {cat_ppl(bg):.2f}   (E29: real 8.18, fuzzy 10.10, bigram 12.84)", flush=True)
print("\n=== samples ===", flush=True)
for p in PROMPTS:
    print(f"  {' '.join(p)} | {' '.join(m.generate(p, n_tokens=14)[len(p):])}", flush=True)
print("\n=== classes ===", flush=True)
print(m.explain(["he", "did"]), flush=True)
print(m.explain(["the", "little"]), flush=True)
print("DONE", flush=True)
