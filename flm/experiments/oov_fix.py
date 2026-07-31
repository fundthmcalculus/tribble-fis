"""Verify the E23.5 OOV fix, then generate sample text.

Two changes under test, both from E23.5/E23.6:
  1. sense + lexicon coverage over the whole vocabulary, not a truncated prefix, so a
     rare-but-real word gets its own senses instead of a blend of its spelling neighbours
  2. relative pruning of OOV lexeme matches, so a weak tail is not merged into a reading

Baseline to beat, same corpus / split / candidate set (E23.4, E23.5): perplexity 320.6,
bigram 253.3, mixture weight 0.0, cold fit 324.9s.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from flm.fuzzyembed.corpus import load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.generate import FuzzyGenerator
from flm.fuzzyembed.baselines import NgramLM

NPOS, TRAINPOS, W, NCAND, K = 1000, 20000, 2, 3000, 200
c = load_corpus('narrative')
train, test = c.split(test_frac=0.2, seed=0)
cand_vocab = c.vocabulary[:NCAND]
print(c.summary(), flush=True)

t0 = time.perf_counter()
emb, info = build_embedder(c, train_lexical=False, verbose=False)     # full coverage
setup = time.perf_counter() - t0
print(f"embedder: {setup:.1f}s, widths={info['widths']}, "
      f"sensed={info['lemmas_with_senses']}/{info['vocab']}\n", flush=True)

print("--- what rare-but-real words resolve to now (E23.5 showed blends) ---", flush=True)
f = FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                       lexeme_top_k=K, vocabulary=cand_vocab)
names = f._output_names()
for w in ("wooden", "arriving", "doorway", "harpooneer", "wodden"):
    v = f._token_vector(w)
    live = sorted(((names[i], float(x)) for i, x in enumerate(v) if x > 0),
                  key=lambda kv: -kv[1])[:4]
    print(f"  {w:<12} " + (", ".join(f"{n}={d:.2f}" for n, d in live) or "(empty)"),
          flush=True)

j = JointNextTokenRanker(f, window=W, n_negatives=8, max_rules=2500, max_order=2,
                         beam=800, lexeme_side='ctx')
t0 = time.perf_counter()
j.fit(train, cand_vocab, max_positions=TRAINPOS, verbose=False)
fit_s = time.perf_counter() - t0
g = FuzzyGenerator(j, cand_vocab, counts=c.counts, seed=1)
lm = NgramLM(order=2).fit(train, g.words)

index = {w: i for i, w in enumerate(g.words)}
rng = np.random.default_rng(7)
sents = [s for s in test.sentences if len(s) > W]
PF, PN, gold, n = [], [], [], 0
for si in rng.permutation(len(sents)):
    sent = sents[si]
    for i in range(W, len(sent)):
        if sent[i] not in index:
            continue
        PF.append(g.distribution(sent[:i]))
        PN.append(lm.distribution(sent[:i], g.words))
        gold.append(index[sent[i]]); n += 1
        if n >= NPOS:
            break
    if n >= NPOS:
        break
PF, PN, gold = np.vstack(PF), np.vstack(PN), np.asarray(gold)
rows = np.arange(len(gold))


def ppl(M):
    return float(np.exp(-np.mean(np.log(np.maximum(M[rows, gold], 1e-12)))))


sweep = [(round(l, 1), ppl(l * PF + (1 - l) * PN)) for l in np.arange(0, 0.65, 0.1)]
best = min(sweep, key=lambda t: t[1])
print(f"\n--- after the fix (cold) ---", flush=True)
print(f"  fuzzy ppl   {ppl(PF):>8.1f}   (E23 baseline 320.6)", flush=True)
print(f"  bigram      {ppl(PN):>8.1f}   (E23 baseline 253.3)", flush=True)
print(f"  best mix    {best[1]:>8.1f} @ lam={best[0]}   (E23 baseline 253.3 @ lam=0.0)",
      flush=True)
print("  sweep: " + "  ".join(f"{l}={p:.1f}" for l, p in sweep), flush=True)
print(f"  cold fit    {fit_s:>8.1f}s  (E23 baseline 324.9s)", flush=True)
print(f"  rules {len(j.model_.rules_)}, dims {len(names)}", flush=True)

# ------------------------------------------------------------------ samples
from flm.fuzzyembed.generate import render_generation

print("\n=== SAMPLE TEXT ===", flush=True)
prompts = [["the", "little"], ["she", "was"], ["he", "did"], ["it", "is"],
           ["there", "was"], ["the", "old"]]
for hedge in (1.0, 3.0):
    print(f"\n-- hedge={hedge} "
          f"({'as learned' if hedge == 1.0 else 'concentrated: mu**3, Zadeh hedge'}) --",
          flush=True)
    for p in prompts:
        out, _ = g.generate(p, n_tokens=14, hedge=hedge)
        print(f"  {' '.join(p)} | {' '.join(out[len(p):])}", flush=True)

print("\n=== same prompts, bigram on the same data, for comparison ===", flush=True)
rng2 = np.random.default_rng(3)
for p in prompts:
    toks = list(p)
    for _ in range(14):
        d = lm.distribution(toks, g.words)
        toks.append(g.words[rng2.choice(len(g.words), p=d / d.sum())])
    print(f"  {' '.join(p)} | {' '.join(toks[len(p):])}", flush=True)

print("\n=== WHY: the named rules behind each choice ===", flush=True)
out, steps = g.generate(["the", "little"], n_tokens=6, hedge=3.0, explain=True)
print(render_generation(["the", "little"], steps), flush=True)
print("DONE", flush=True)
