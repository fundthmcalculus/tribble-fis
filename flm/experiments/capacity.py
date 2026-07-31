"""Why did more data help the bigram more than the fuzzy model?

E23 scaling: over 64K -> 811K training tokens the fuzzy model went 438.8 -> 332.1 while the
bigram went 389.3 -> 253.3, and the mixture weight on the fuzzy model fell to zero. Two
explanations, and they imply opposite next steps:

  (a) the METHOD plateaus -- fuzzy rules extract what they can and further text adds nothing
  (b) the FEATURISER is capacity-bound -- the embedder senses only the top ``max_types``
      words, and every other token falls through to fuzzy lexical access, which returns a
      *blend* of several neighbours. Measured on the small corpus: `wooden` comes back as
      noun.substance 0.57 + noun.group 0.56 + noun.person 0.51 + verb.cognition 0.46
      simultaneously. On the 1M-token corpus 24,000 of 27,044 types take that path, so the
      bigger the corpus, the larger the fraction of context that is diffuse noise.

The candidate set must stay FIXED at 3,000 across conditions or perplexity is not comparable
(more candidates is a harder task). Only the featuriser's capacity varies -- how many types
get real sense assignment, and how many get context-side lexical identity. That isolates (b).
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

NPOS, TRAINPOS, W, NCAND = 1000, 20000, 2, 3000
c = load_corpus('narrative')
train, test = c.split(test_frac=0.2, seed=0)
cand_vocab = c.vocabulary[:NCAND]          # FIXED candidate set, all conditions
print(c.summary(), flush=True)
print(f"candidate vocabulary fixed at {NCAND}; OOV context types = "
      f"{len(c.vocabulary) - NCAND:,}\n", flush=True)

CASES = [
    ("emb 3000, ctx-lex 200  (E23 baseline)", 3000, 200),
    ("emb 12000, ctx-lex 200", 12000, 200),
    ("emb 12000, ctx-lex 600", 12000, 600),
]

print(f"{'condition':<40}{'dims':>6}{'ppl':>9}{'2gram':>8}{'mix':>8}{'lam':>6}"
      f"{'rules':>7}{'setup_s':>9}{'fit_s':>8}", flush=True)
for label, max_types, k in CASES:
    t0 = time.perf_counter()
    emb, _ = build_embedder(c, max_types=max_types, train_lexical=False, verbose=False)
    f = FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                           lexeme_top_k=k, vocabulary=c.vocabulary[:max(k, NCAND)])
    setup_s = time.perf_counter() - t0
    j = JointNextTokenRanker(f, window=W, n_negatives=8, max_rules=2500,
                             max_order=2, beam=800, lexeme_side='ctx')
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

    best = min(((ppl(l * PF + (1 - l) * PN), l)
                for l in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)), key=lambda t: t[0])
    print(f"{label:<40}{len(f._output_names()):>6}{ppl(PF):>9.1f}{ppl(PN):>8.1f}"
          f"{best[0]:>8.1f}{best[1]:>6.1f}{len(j.model_.rules_):>7}"
          f"{setup_s:>9.1f}{fit_s:>8.1f}", flush=True)
print("DONE", flush=True)
