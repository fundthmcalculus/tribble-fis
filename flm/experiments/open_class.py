"""Does reserving rule budget for open-class candidates fix generation? (E25)

E24.3 diagnosed the failure by reading the rules: every high-lift rule was closed-class
(`IF prev1:PREPOSITION AND cand:DETERMINER`, lift +10.09), because |lift| follows support and
closed-class features carry ~10x the support of anything semantic. The model learned function
word syntax and nothing that selects a content word, so it generated function-word soup.

Sweeping the reserved fraction, reporting BOTH numbers that matter, because they can move in
opposite directions and reporting only one would be misleading:
  * perplexity -- the aggregate; a quota forces lower-lift rules in, so it may well cost some
  * content-word rate in free generation -- the thing actually diagnosed as broken

The content rate is measured against the closed-class inventory, not eyeballed.
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
from flm.fuzzyembed.syntax import CLOSED_CLASS

NPOS, TRAINPOS, W, NCAND, K = 1000, 20000, 2, 3000, 200
FUNCTION = set().union(*CLOSED_CLASS.values())
QUOTAS = (0.0, 0.25, 0.5, 0.75)
PROMPTS = [["the", "little"], ["she", "was"], ["he", "did"], ["it", "is"],
           ["there", "was"], ["the", "old"]]

c = load_corpus('narrative')
train, test = c.split(test_frac=0.2, seed=0)
cand_vocab = c.vocabulary[:NCAND]
emb, _ = build_embedder(c, train_lexical=False, verbose=False)
print(c.summary(), flush=True)


def content_rate(gen: FuzzyGenerator, hedge: float = 1.0) -> tuple[float, list[str]]:
    """Fraction of freely generated tokens that are not function words."""
    toks, samples = [], []
    for p in PROMPTS:
        out, _ = gen.generate(p, n_tokens=14, hedge=hedge)
        new = out[len(p):]
        toks.extend(new)
        samples.append(f"{' '.join(p)} | {' '.join(new)}")
    n_content = sum(1 for t in toks if t not in FUNCTION)
    return n_content / max(len(toks), 1), samples


rows = []
print(f"\n{'quota':>7}{'ppl':>9}{'2gram':>8}{'mix':>8}{'lam':>6}{'rules':>7}"
      f"{'open-cls rules':>16}{'content%':>10}{'fit_s':>8}", flush=True)
for q in QUOTAS:
    f = FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                           lexeme_top_k=K, vocabulary=cand_vocab)
    j = JointNextTokenRanker(f, window=W, n_negatives=8, max_rules=2500, max_order=2,
                             beam=800, lexeme_side='ctx', open_class_quota=q)
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
    r = np.arange(len(gold))

    def ppl(M):
        return float(np.exp(-np.mean(np.log(np.maximum(M[r, gold], 1e-12)))))

    best = min(((ppl(l * PF + (1 - l) * PN), round(l, 1))
                for l in np.arange(0, 0.65, 0.1)), key=lambda t: t[0])
    rate, samples = content_rate(g)
    hist = j.model_.reserved_histogram()
    print(f"{q:>7.2f}{ppl(PF):>9.1f}{ppl(PN):>8.1f}{best[0]:>8.1f}{best[1]:>6.1f}"
          f"{len(j.model_.rules_):>7}{hist['reserved']:>16}{100*rate:>9.1f}%{fit_s:>8.1f}",
          flush=True)
    rows.append((q, samples, j, g))

# Reference: what fraction of real text is content words?
real = [t for s in test.sentences[:400] for t in s]
print(f"\nreference: real held-out text is "
      f"{100*sum(1 for t in real if t not in FUNCTION)/len(real):.1f}% content words",
      flush=True)

for q, samples, j, g in rows:
    print(f"\n--- quota={q:.2f} samples ---", flush=True)
    for s in samples:
        print("  " + s, flush=True)

print("\n--- rules gained at the highest quota ---", flush=True)
best_j = rows[-1][2]
res = best_j.open_class_features()
shown = [r for r in best_j.model_.rules_
         if any(fi in res for fi in r.features) and len(r.features) == 2][:12]
for rr in shown:
    print("  " + rr.render("P(next)"), flush=True)
print("DONE", flush=True)
