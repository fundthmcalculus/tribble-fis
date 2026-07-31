"""Context window sweep: 2, 4, 8, 16, 32 tokens (E26).

E12 ruled out context width, but that was on the *marginal* formulation, a 87K-token corpus,
and before the joint ranker, the NCE inversion, and the E25.2 decoding fix. Worth re-asking on
the current stack: generation is now plausible word-by-word but has no agreement or clause
structure, and a 2-token window cannot supply either.

Sizing, because this sweep is memory-bound rather than time-bound. Features grow linearly with
the window: ``dims = (window + 1) * base``, and with ``base = 261`` a 32-token window is 8,613
columns. ``MembershipRuleRegressor.fit`` materialises ``Xy`` alongside ``X``, so cost is
``rows * dims * 2 * itemsize`` -- ~137 KB/row at window 32 in float64, half that in float32.
Hence:

* ``TRAINPOS`` is held at 6,000 for **every** condition. Constant across conditions is what
  matters for the comparison; it is lower than the 20,000 used elsewhere, so absolute
  perplexities here are not comparable with E24/E25 numbers -- only the rows against each other.
* ``max_rules`` and ``beam`` are set generously, per the instruction not to worry about
  saturating the rule base. The order-2 candidate supply grows with the window (more context
  features to pair with), so unlike E25.1 the budget may actually bind here.
* ``float32`` throughout, for every condition, so precision is not a variable across rows.
  float64 at window 32 needed ~22 GB and the OOM killer took the process outright.
* Each condition is wrapped so a failure at the wide end still leaves the narrow rows -- though
  note SIGKILL cannot be caught, so a hard OOM shows up as missing output, not as a message.
"""
import sys, time, resource, gc, traceback
sys.path.insert(0, '.')
import numpy as np
from flm.fuzzyembed.corpus import load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.generate import FuzzyGenerator
from flm.fuzzyembed.baselines import NgramLM
from flm.fuzzyembed.syntax import CLOSED_CLASS

NPOS, TRAINPOS, NCAND, K = 1000, 6000, 3000, 200
MAX_RULES, BEAM = 20000, 6000
WINDOWS = (2, 4, 8, 16, 32)
#: Every condition is scored at positions with at least this much left context, so all rows
#: see IDENTICAL held-out positions. The first attempt skipped ``i < window`` per condition,
#: which silently gave each window a different test set -- caught because the bigram control
#: column moved (253.3 / 277.1 / 240.8 / 265.5) when the same bigram on the same training data
#: must score the same everywhere. A moving control is a broken comparison.
MIN_CTX = max(WINDOWS)
#: float32 halves the design matrix. At window 32 (8,613 columns) float64 needed ~22 GB and the
#: OOM killer took the process -- SIGKILL, so no catchable MemoryError and no partial output.
DTYPE = np.float32
FUNCTION = set().union(*CLOSED_CLASS.values())
PROMPTS = [["the", "little"], ["she", "was"], ["he", "did"], ["the", "old"]]

c = load_corpus('narrative')
train, test = c.split(test_frac=0.2, seed=0)
cand_vocab = c.vocabulary[:NCAND]
emb, _ = build_embedder(c, train_lexical=False, verbose=False)
print(c.summary(), flush=True)
print(f"TRAINPOS={TRAINPOS} (constant), max_rules={MAX_RULES}, beam={BEAM}, "
      f"dtype={np.dtype(DTYPE).name}, scored at positions >= {MIN_CTX}\n", flush=True)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


print(f"{'win':>4}{'dims':>7}{'ppl':>9}{'2gram':>8}{'mix':>8}{'lam':>6}{'rules':>7}"
      f"{'order2':>8}{'content%':>10}{'fit_s':>8}{'peakGB':>8}", flush=True)
results = []
for W in WINDOWS:
    try:
        f = FuzzySequenceModel(emb, level=2, window=W, n_outputs=12, use_syntax=True,
                               lexeme_top_k=K, vocabulary=cand_vocab)
        j = JointNextTokenRanker(f, window=W, n_negatives=8, max_rules=MAX_RULES,
                                 max_order=2, beam=BEAM, lexeme_side='ctx',
                                 dtype=DTYPE)
        t0 = time.perf_counter()
        j.fit(train, cand_vocab, max_positions=TRAINPOS, verbose=False)
        fit_s = time.perf_counter() - t0
        g = FuzzyGenerator(j, cand_vocab, counts=c.counts, seed=1)
        lm = NgramLM(order=2).fit(train, g.words)

        index = {w: i for i, w in enumerate(g.words)}
        rng = np.random.default_rng(7)
        sents = [s for s in test.sentences if len(s) > MIN_CTX]
        PF, PN, gold, n = [], [], [], 0
        for si in rng.permutation(len(sents)):
            sent = sents[si]
            for i in range(MIN_CTX, len(sent)):
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
        toks, samples = [], []
        for p in PROMPTS:
            out, _ = g.generate(p, n_tokens=14)
            toks.extend(out[len(p):])
            samples.append(f"{' '.join(p)} | {' '.join(out[len(p):])}")
        rate = sum(1 for t in toks if t not in FUNCTION) / len(toks)
        hist = j.model_.order_histogram()
        print(f"{W:>4}{len(j.feature_names_):>7}{ppl(PF):>9.1f}{ppl(PN):>8.1f}"
              f"{best[0]:>8.1f}{best[1]:>6.1f}{len(j.model_.rules_):>7}"
              f"{hist.get(2, 0):>8}{100*rate:>9.1f}%{fit_s:>8.1f}{rss_gb():>8.1f}",
              flush=True)
        results.append((W, samples, j))
        del PF, PN, g, lm, j, f
        gc.collect()
    # numpy's _ArrayMemoryError subclasses MemoryError, so this catches allocation
    # failures too (its import path moved in numpy 2.x, so do not name it directly).
    except MemoryError as e:
        print(f"{W:>4}  OUT OF MEMORY: {type(e).__name__} "
              f"(peak {rss_gb():.1f} GB) -- wider contexts need the design matrix "
              f"kept in float32 or built in blocks", flush=True)
        gc.collect()
    except Exception:
        traceback.print_exc()
        gc.collect()

for W, samples, j in results:
    print(f"\n--- window={W} samples ---", flush=True)
    for s in samples:
        print("  " + s, flush=True)

if results:
    W, _, j = results[-1]
    print(f"\n--- longest-lag rules actually learned at window={W} ---", flush=True)
    far = [r for r in j.model_.rules_
           if any(nm.startswith("ctx:prev") and int(nm.split(":")[1][4:]) > 2
                  for nm in r.names)]
    print(f"  {len(far)} of {len(j.model_.rules_)} rules reference a lag > 2", flush=True)
    for rr in far[:12]:
        print("  " + rr.render("P(next)"), flush=True)
print("DONE", flush=True)
