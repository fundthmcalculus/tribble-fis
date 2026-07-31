"""E27: evaluate the two candidate fixes for E26's dilution failure.

**Option A -- width-aware significance.** E26.1's claim was that the rule learner's
false-discovery rate rises with the search space while `min_support`/`min_interaction` stay
fixed. Two gates test it:
  A1 `significance=alpha`  Bonferroni on |z| = |lift|/sqrt(p(1-p)), with the multiplicity
                           counted as candidates actually tested, so it tightens with width.
  A2 `holdout=frac`        require each rule's effect to reproduce on rows never used to mine
                           it. Assumption-free, and the survival rate *measures* the FDR
                           instead of estimating it -- which also tests E26.1 directly.

**Option B -- relational slots.** Replace lag-indexed context with 8 functionally-defined
slots (nearest preceding verb, subject, determiner, ...). Unbounded reach at fixed dimension.
A shallow approximation of dependency relations, since no parser is installable here.

Everything is scored on the E26 protocol so the numbers drop straight into that table:
1M-token corpus, same split, TRAINPOS=6000, positions >= 32, float32, bigram control.
"""
import sys, time, gc, traceback
sys.path.insert(0, '.')
import numpy as np
from flm.fuzzyembed.corpus import load_corpus
from flm.fuzzyembed.embedder import build_embedder
from flm.fuzzyembed.sequence import FuzzySequenceModel
from flm.fuzzyembed.joint import JointNextTokenRanker
from flm.fuzzyembed.relations import RelationalNextTokenRanker
from flm.fuzzyembed.generate import FuzzyGenerator
from flm.fuzzyembed.baselines import NgramLM
from flm.fuzzyembed.syntax import CLOSED_CLASS

NPOS, TRAINPOS, NCAND, K, MIN_CTX = 1000, 6000, 3000, 200, 32
MAX_RULES, BEAM, DTYPE = 20000, 6000, np.float32
FUNCTION = set().union(*CLOSED_CLASS.values())
PROMPTS = [["the", "little"], ["she", "was"], ["he", "did"], ["the", "old"]]

c = load_corpus('narrative')
train, test = c.split(test_frac=0.2, seed=0)
cand_vocab = c.vocabulary[:NCAND]
emb, _ = build_embedder(c, train_lexical=False, verbose=False)
print(c.summary(), flush=True)
print(f"protocol: TRAINPOS={TRAINPOS}, positions>={MIN_CTX}, float32, "
      f"max_rules={MAX_RULES}, beam={BEAM}\n", flush=True)


def featuriser(win):
    return FuzzySequenceModel(emb, level=2, window=win, n_outputs=12, use_syntax=True,
                              lexeme_top_k=K, vocabulary=cand_vocab)


COMMON = dict(n_negatives=8, max_rules=MAX_RULES, max_order=2, beam=BEAM,
              lexeme_side='ctx', dtype=DTYPE)

CASES = [
    # label,                          builder
    ("A0 lag w2 (E26 baseline)",      lambda: JointNextTokenRanker(featuriser(2), window=2, **COMMON)),
    ("A0 lag w32 (E26 baseline)",     lambda: JointNextTokenRanker(featuriser(32), window=32, **COMMON)),
    ("A1 lag w2  + signif 0.05",      lambda: JointNextTokenRanker(featuriser(2), window=2, significance=0.05, **COMMON)),
    ("A1 lag w8  + signif 0.05",      lambda: JointNextTokenRanker(featuriser(8), window=8, significance=0.05, **COMMON)),
    ("A1 lag w32 + signif 0.05",      lambda: JointNextTokenRanker(featuriser(32), window=32, significance=0.05, **COMMON)),
    ("A2 lag w8  + holdout 0.3",      lambda: JointNextTokenRanker(featuriser(8), window=8, holdout=0.3, **COMMON)),
    ("A2 lag w32 + holdout 0.3",      lambda: JointNextTokenRanker(featuriser(32), window=32, holdout=0.3, **COMMON)),
    ("B1 relational slots",           lambda: RelationalNextTokenRanker(featuriser(2), lookback=64, **COMMON)),
    ("B2 relational + signif 0.05",   lambda: RelationalNextTokenRanker(featuriser(2), lookback=64, significance=0.05, **COMMON)),
    ("B3 relational + holdout 0.3",   lambda: RelationalNextTokenRanker(featuriser(2), lookback=64, holdout=0.3, **COMMON)),
]

print(f"{'condition':<30}{'dims':>6}{'ppl':>8}{'2gram':>8}{'mix':>8}{'lam':>5}"
      f"{'rules':>7}{'gated':>8}{'repl%':>7}{'far%':>6}{'cont%':>7}{'fit_s':>7}",
      flush=True)
rows = []
for label, make in CASES:
    try:
        j = make()
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
        cont = sum(1 for t in toks if t not in FUNCTION) / len(toks)

        rules = j.model_.rules_
        # Fraction of rules reaching beyond adjacency -- lag > 2, or a non-adjacent slot.
        def is_far(rule):
            for nm in rule.names:
                if not nm.startswith("ctx:"):
                    continue
                slot = nm.split(":")[1]
                if slot.startswith("prev") and slot[4:].isdigit():
                    if int(slot[4:]) > 2:
                        return True
                elif slot not in ("adj1", "adj2"):
                    return True
            return False
        far = sum(1 for rr in rules if is_far(rr)) / max(len(rules), 1)
        repl = j.model_.replication_.get("rate", float('nan'))
        print(f"{label:<30}{len(j.feature_names_):>6}{ppl(PF):>8.1f}{ppl(PN):>8.1f}"
              f"{best[0]:>8.1f}{best[1]:>5.1f}{len(rules):>7}{j.model_.n_gated_:>8}"
              f"{100*repl:>6.0f}%{100*far:>5.0f}%{100*cont:>6.1f}%{fit_s:>7.0f}",
              flush=True)
        rows.append((label, samples, j, ppl(PF)))
        del PF, PN, g, lm
        gc.collect()
    except Exception:
        traceback.print_exc()
        gc.collect()

print("\n(gated = rules rejected by the significance gate; repl% = fraction of mined rules "
      "reproducing on held-out rows; far% = rules reaching beyond adjacency)", flush=True)

for label, samples, j, _ in rows:
    if label.startswith(("B1", "A1 lag w32", "A0 lag w2")):
        print(f"\n--- {label} samples ---", flush=True)
        for s in samples:
            print("  " + s, flush=True)

rel = next((j for lb, _, j, _ in rows if lb.startswith("B1")), None)
if rel is not None:
    print("\n--- what the relational model learned (non-adjacent slots only) ---", flush=True)
    shown = [rr for rr in rel.model_.rules_
             if any(nm.startswith("ctx:") and nm.split(":")[1] not in ("adj1", "adj2")
                    for nm in rr.names)][:14]
    for rr in shown:
        print("  " + rr.render("P(next)"), flush=True)
    sent = "the little rabbit that lived in the old wood was very happy".split()
    print("\n--- slot filling, and where the heuristic is wrong ---", flush=True)
    for i in (3, 8, 12):
        print(f"  after {' '.join(sent[:i])!r}\n      {rel.slot_report(sent, i)}", flush=True)
print("DONE", flush=True)
