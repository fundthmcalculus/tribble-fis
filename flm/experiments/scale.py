"""Push the one constraint that was never pushed: training-corpus size.

E16 concluded corpus size is binding and E22.4 reinforced it, but every measurement so far
used the same ~87K-token children's corpus. This trains on nested subsets of an 11.7x larger
narrative corpus and evaluates every condition on the **same held-out test set with the same
3,000-word candidate vocabulary**, so the only thing varying is how much text the model saw.

Why that design rather than "tiny corpus vs big corpus": those have different test sets, so
their perplexities are not comparable and a difference could be "easier text" rather than
"more data". Nested subsets of one corpus isolate the variable.

Two knobs move together with data and are reported separately, because conflating them would
confound the result:
  * ``frac``     -- how much *text* the model sees (the actual question)
  * ``TRAINPOS`` -- how many training pairs are drawn from it (E16 found ranking quality
                    rises monotonically in this and had not flattened at 12,000)

Counts come from each subset, not the full corpus: the frequency prior q(w) is part of the
model, and giving a small-data condition the full corpus's unigram statistics would leak.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from flm.fuzzyembed.corpus import Corpus, load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.generate import FuzzyGenerator
from flm.fuzzyembed.baselines import NgramLM, ngram_perplexity

NPOS, K, W, NVOCAB = 1000, 200, 2, 3000
FRACS = (0.085, 0.25, 0.5, 1.0)     # 0.085 ~ the old corpus's token count
TRAINPOS = (5000, 20000)

c = load_corpus('narrative')
print(c.summary(), flush=True)
train_full, test = c.split(test_frac=0.2, seed=0)
vocab = c.vocabulary[:NVOCAB]
t0 = time.perf_counter()
emb, _ = build_embedder(c, max_types=NVOCAB, train_lexical=False, verbose=False)
print(f"embedder built in {time.perf_counter()-t0:.1f}s", flush=True)


def subset(frac: float) -> Corpus:
    """First ``frac`` of the training sentences, with its own counts."""
    n = max(1, int(len(train_full.sentences) * frac))
    sents = train_full.sentences[:n]
    counts: dict[str, int] = {}
    for s in sents:
        for t in s:
            counts[t] = counts.get(t, 0) + 1
    return Corpus(f"train[{frac:.3f}]", sents, vocab, counts)


print(f"\n{'frac':>6}{'tokens':>10}{'trainpos':>10}{'ppl':>9}{'2gram':>8}"
      f"{'mix':>8}{'lam':>6}{'rules':>7}{'fit_s':>8}", flush=True)
for frac in FRACS:
    sub = subset(frac)
    n_tok = sub.n_tokens
    for tp in TRAINPOS:
        f = FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                               lexeme_top_k=K, vocabulary=vocab)
        j = JointNextTokenRanker(f, window=W, n_negatives=8, max_rules=2500,
                                 max_order=2, beam=800, lexeme_side='ctx')
        t0 = time.perf_counter()
        j.fit(sub, vocab, max_positions=tp, verbose=False)
        fit_s = time.perf_counter() - t0
        g = FuzzyGenerator(j, vocab, counts=sub.counts, seed=1)
        lm = NgramLM(order=2).fit(sub, g.words)

        # Shared positions for all three numbers, so the mixture is self-calibrating.
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

        best = min(((ppl(l * PF + (1 - l) * PN), l)
                    for l in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)), key=lambda t: t[0])
        print(f"{frac:>6.3f}{n_tok:>10,}{tp:>10}{ppl(PF):>9.1f}{ppl(PN):>8.1f}"
              f"{best[0]:>8.1f}{best[1]:>6.1f}{len(j.model_.rules_):>7}{fit_s:>8.1f}",
              flush=True)
print("DONE", flush=True)
