"""Is the linguistic parameter space COMPLEMENTARY to the WordNet space?

The A/B says it loses head-to-head (360.0 vs 324.8). E19.4 showed that is not the same
question as whether it carries information the other does not: the fuzzy model lost to a
bigram and still improved a mixture with it. So sweep every pair on IDENTICAL positions.
"""
import sys, time, itertools
sys.path.insert(0, '.')
import numpy as np
from flm.fuzzyembed.corpus import load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.lexical import FuzzyLexicon
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.generate import FuzzyGenerator
from flm.fuzzyembed.baselines import NgramLM
from flm.fuzzytok.featuriser import build_parameter_featuriser

NPOS, TRAINPOS, K, W = 600, 5000, 200, 2
c = load_corpus('tiny'); train, test = c.split(test_frac=0.2, seed=0)
emb, _ = build_embedder(c, max_types=3000, train_lexical=False, verbose=False)
vocab = c.vocabulary[:3000]
lex = FuzzyLexicon(vocab, counts=c.counts)


def fit(featuriser):
    j = JointNextTokenRanker(featuriser, window=W, n_negatives=8, max_rules=2500,
                             max_order=2, beam=800, lexeme_side='ctx')
    j.fit(train, vocab, max_positions=TRAINPOS, verbose=False)
    return FuzzyGenerator(j, vocab, counts=c.counts, seed=1)


t0 = time.perf_counter()
wn_f = FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                          lexeme_top_k=K, vocabulary=vocab)
g_wn = fit(wn_f)
print(f"wn fitted {time.perf_counter()-t0:.0f}s", flush=True)
t0 = time.perf_counter()
g_lp = fit(build_parameter_featuriser(c, vocab, lemma_synsets=emb.senses.lemma_synsets,
                                      lexeme_top_k=K, head_size=500, lexicon=lex))
print(f"lp fitted {time.perf_counter()-t0:.0f}s", flush=True)

# The two generators restrict to different decodable vocabularies (a word the WordNet
# space cannot represent is not necessarily one the parameter space cannot), so their
# distributions are over different column sets and cannot be mixed elementwise. Project
# everything onto the shared vocabulary and renormalise -- otherwise the mixture silently
# compares different words, or, as it did here, raises on the shape mismatch.
shared = [w for w in g_wn.words if w in set(g_lp.words)]
index = {w: i for i, w in enumerate(shared)}
take = {"wn": np.array([g_wn.index[w] for w in shared]),
        "lp": np.array([g_lp.index[w] for w in shared])}
lm2 = NgramLM(order=2).fit(train, shared)
print(f"vocab: wn={len(g_wn.words)} lp={len(g_lp.words)} shared={len(shared)}", flush=True)

# --- collect distributions on identical positions -------------------------
rng = np.random.default_rng(7)
sents = [s for s in test.sentences if len(s) > W]
P = {"wn": [], "lp": [], "2g": []}
gold_idx, n = [], 0


def proj(d, cols):
    v = d[cols]
    return v / max(v.sum(), 1e-12)


for si in rng.permutation(len(sents)):
    sent = sents[si]
    for i in range(W, len(sent)):
        gold = sent[i]
        if gold not in index:
            continue
        ctx = sent[:i]
        P["wn"].append(proj(g_wn.distribution(ctx), take["wn"]))
        P["lp"].append(proj(g_lp.distribution(ctx), take["lp"]))
        P["2g"].append(lm2.distribution(ctx, shared))
        gold_idx.append(index[gold]); n += 1
        if n >= NPOS:
            break
    if n >= NPOS:
        break
M = {k: np.vstack(v) for k, v in P.items()}
g = np.asarray(gold_idx)
print(f"\nshared positions n={n}\n", flush=True)


def ppl(mix):
    pick = mix[np.arange(len(g)), g]
    return float(np.exp(-np.mean(np.log(np.maximum(pick, 1e-12)))))


print("--- alone ---", flush=True)
for k in ("wn", "lp", "2g"):
    print(f"  {k:<4} {ppl(M[k]):>8.1f}", flush=True)

print("\n--- pairwise mixtures  p = lam*A + (1-lam)*B ---", flush=True)
for a, b in (("wn", "2g"), ("lp", "2g"), ("lp", "wn")):
    row = [f"  {a}/{b}: "]
    best = (1e9, None)
    for lam in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        p = ppl(lam * M[a] + (1 - lam) * M[b])
        row.append(f"{lam:.1f}={p:.1f}")
        if p < best[0]:
            best = (p, lam)
    print(" ".join(row) + f"   BEST lam={best[1]:.1f} ppl={best[0]:.1f}", flush=True)

print("\n--- 3-way grid  a*wn + b*lp + (1-a-b)*2g ---", flush=True)
best = (1e9, None)
for a in np.arange(0, 0.65, 0.1):
    for b in np.arange(0, 0.65 - a, 0.1):
        p = ppl(a * M["wn"] + b * M["lp"] + (1 - a - b) * M["2g"])
        if p < best[0]:
            best = (p, (round(float(a), 2), round(float(b), 2), round(1 - float(a) - float(b), 2)))
print(f"  BEST wn={best[1][0]} lp={best[1][1]} 2g={best[1][2]}  ppl={best[0]:.1f}", flush=True)
print(f"  vs bigram alone {ppl(M['2g']):.1f}, vs wn+2g best, vs lp+2g best above", flush=True)

# Does LP add anything on top of the already-good wn+2g mixture?
base = 0.3 * M["wn"] + 0.7 * M["2g"]
print("\n--- LP on top of a fixed wn+2g mixture ---", flush=True)
for lam in (0.0, 0.1, 0.2, 0.3):
    print(f"  lam_lp={lam:.1f}  ppl={ppl(lam * M['lp'] + (1 - lam) * base):.1f}", flush=True)
print("DONE", flush=True)
