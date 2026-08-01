# Handoff — what to do next

Written at the end of the session that produced E1–E35. Read [`LOG.md`](LOG.md)'s **last** standing
summary first; this file is the actionable residue of it.

The single most valuable thing in this repo is not the model — it is the **"do not redo" list at the
bottom**. Fourteen things have been ruled out with measurements, six of them contradicting a
prediction I had made confidently. Check that list before starting anything.

---

## Where it stands

Best configuration, all of it in [`fuzzyembed/firstorder.py`](fuzzyembed/firstorder.py):

```python
j = JointNextTokenRanker(featuriser, window=2, n_negatives=8, max_rules=20000,
                         max_order=2, beam=6000, lexeme_side='ctx', dtype=np.float32)
j.fit(train, cand_vocab, max_positions=6000)
m = ContextClassMiner(j, cand_vocab, counts=corpus.counts, alpha=0.5, min_mass=8.0,
                      max_order=2, top_singles=10**6, max_classes=10**9, n_jobs=4)
m.fit(train, max_positions=1_200_000)      # exhausts a 1M-token corpus at 624,325
```

| | perplexity |
|---|---|
| bigram / tuned trigram, same data | 286.4 / 284.4 |
| **first-order TSK standalone** | **194.9** (31.5% better than the trigram) |
| + trigram, λ=0.8 | **180.4** |
| generation, category-sequence ppl | 9.33 (real text 8.18, bigram 12.84) |

~39s for the whole pipeline on 4 CPU cores. 82 tests pass.

**Running anything:** `uv run --with nltk --with scipy --with threadpoolctl python flm/experiments/<x>.py`
from the repo root. `scipy` is required for the fast paths (there are `np.add.at` fallbacks, ~10x
slower). Tests: `uv run --with nltk python -m pytest flm/tests -q`.

**Protocol to keep.** 1M-token corpus, `split(test_frac=0.2, seed=0)`, 3,000-word candidate
vocabulary, held-out positions with **≥32 tokens of left context**, and an n-gram control column in
every table. That last one is not ceremony: a moving control is what caught a broken comparison in
E26.2. If a baseline that cannot legitimately change does change, stop and find out why.

---

## P0 — cheap, high confidence, no external dependencies

**1. More training text. This is the highest-confidence quality lever left.**
E29.3 found class estimation is strongly data-bound and the curve **only flattened when the corpus
ran out** (601.2 → 219.9 over 20K → 624K positions). Nothing suggests 1M tokens is a natural
stopping point. Concretely: `NARRATIVE + brown-all` is ~2M tokens from data already on disk, and
`corpus.load_local` reads a real TinyStories dump if you can get one. Watch two things — the
candidate vocabulary should stay at 3,000 so perplexity stays comparable, and E33.2's mass
fragmentation predicts more data should *also* make order-3 classes viable, which is item 11.

**2. Redo the held-out class-selection experiment properly.** E29.1 concluded held-out gain does not
help, but the experiment was confounded and the log says so: holding out 30% of rows also made
`min_mass` ~3.3× stricter, so criterion and threshold moved together (class count 1,670 → 397). Fix
is one line — scale the floor by `1/(1 - selection_holdout)` — and the question is still open, since
E33.2 showed selection admits redundant classes, which is exactly what a held-out criterion should
catch. Cheap.

**3. Persistence.** There is no `save`/`load` for the first-order model. 483K sparse parameters is a
few MB; without this, every use re-trains. Straightforward, and needed before anything downstream.

**4. Recency / cache features — the one untried *kind* of feature.** Global incoherence is the
central open problem, and it is **not** a context-width problem: that has now failed twice for two
different measured reasons (E26 zero-order dilution, E33.2 first-order mass fragmentation). So stop
widening the window. A cache feature — "this word already occurred in the current document" (Kuhn &
de Mori, *A Cache-Based Natural Language Model for Speech Recognition*, IEEE PAMI 1990) — is a
different axis entirely, is known to help n-grams, is cheap, and is *nameable*, which matters here:
`IF ctx:recently_mentioned AND cand:noun.person` is a readable class. Add it as a context dimension
and re-run E34's measurement.

**5. Larger candidate vocabulary, clearly labelled as a different task.** The candidate set is capped
at 3,000 of 27,044 types. Perplexity is only comparable within a fixed candidate set, so this cannot
go in the same table as anything above — report it as its own measurement. A real LM has to handle the
full vocabulary, and the model's cost is linear in it.

---

## P1 — blocked on this environment, do them where the network allows

Every item here is blocked by the egress allowlist, not by the code. The code exists in each case.

**6. GPT-2 comparison.** `baselines.gpt2_perplexity` is implemented and unrun. Keep the caveat
attached: GPT-2 saw ~40GB against this model's 1M tokens, so the comparison measures a data gap. The
controlled baseline at this scale is the n-gram, which is why that is what every table uses.

**7. Neural-embedding comparison** (EmbeddingGemma, gte-small). Never run. The interesting question
is not accuracy but whether a FIS over a *named* space loses much to one over a dense space.

**8. Experiment B on real data.** [`exp_b/`](exp_b/) is built and smoke-tested; the real encoder and
SST paths have never executed. It is the control that motivated the whole project (see §5 of its
plan) and it is still unrun. Also the natural home for the sentiment arm, which was requested in the
very first message of this project and never delivered.

**9. Verify every citation.** All are marked ✎ — from search snippets, no PDF opened, no DOI
resolved. Two numbers specifically need re-deriving: the Roget's level cardinalities
(edition-dependent) and the Wikipedia→Philosophy convergence rate. Do this before any of it is
quoted anywhere.

**10. Roget's Thesaurus backend.** WordNet was a *forced substitution*, not a design choice —
gutenberg.org and sites.google.com were both blocked. Roget's gives a balanced named ladder and,
critically, **antonymous opposed pairs** (648 Goodness / 649 Badness) that collapse into signed
bipolar variables, which is what makes the sentiment head nearly free. `hierarchy.py` was written to
be backend-pluggable for this.

---

## P2 — research bets, in descending order of my confidence

**11. Order-3 context classes, but only after item 1.** Deferred twice, the second time with a
measured reason (E33.2: a third conjunct fragments mass further, which is the mechanism that makes
extra features hurt). Fragmentation is a *data* problem, so more text is the precondition. Do not
attempt before item 1.

**12. `fuzzytok` as an explanation layer only.** E22 measured it as strictly dominated as a *feature
space* — mixture weight zero against every alternative — but it is a better *reporting* language:
`Polarity=Neg` is something the WordNet space cannot say. Rendering existing classes in that
vocabulary costs nothing predictive.

**13. Sentence-level reranking for coherence.** Generate *k* candidates and rerank by a global fuzzy
score. Attacks incoherence without touching the local model. Speculative, and it needs a global score
worth optimising, which does not exist yet.

**14. First-order consequents for the *zero-order* rule base's antecedents.** The current classes are
mined context-only by information gain. The zero-order base has richer antecedents (context ×
candidate interactions) that the class miner discards. Whether those make better classes is untested.

---

## Methodological debts — one settled, one standing

**Settled (E35): λ is not tuned on the test set in any harmful sense.** Every mixture number picked
λ by minimising perplexity on the same positions it reported, which is test-set tuning. Measured on a
2,000-position split, tuning λ on one half and reporting on the other: **0.00% inflation**, λ=0.8
either way. The optimum is flat enough that an 11-point 1-D grid cannot overfit 1,000 points. No
action needed — recorded so nobody re-worries.

**Standing: single-split, single-seed everywhere.** Every number in `LOG.md` comes from one
train/test split (`seed=0`) and one evaluation sample of 1,000 positions. No confidence intervals
anywhere. Differences of a few percent between adjacent configurations should not be trusted as
ordering. Cheap fix: repeat the headline over 3–5 seeds and report a spread. Do this before
publishing any of it.

---

## Do NOT redo — ruled out with measurements

Each has a log entry. Six of these contradicted a prediction I had made confidently, which is the
reason this list exists.

| Ruled out | Where | Note |
|---|---|---|
| Wider context windows | E12, **E26**, **E33** | failed for *two different measured reasons*; stop widening |
| More lexeme-identity dimensions | E33 | a wash |
| `max_classes` as a binding cap | E32.1 | lifting it entirely is worth 0.7% |
| Corpus scale hurting the model | E23.4 → **E29.3** | true of zero-order only; it *inverts* for first-order |
| Order-3 rules on the ranking task | E12, E15 | different from item 11 (order-3 *classes*) |
| Gaussian / trapezoid antecedents | E9, E10 | inputs are already memberships; nothing to fit |
| Symmetric decode similarity | E11 | wrong asymmetry |
| Cython | E14, **E30.3** | no scalar-Python hot loop exists to compile |
| GPU | **E30.4** | memory-bound, and the pipeline is 39s on one core |
| Thread-based parallelism | E23.3 | 0.47× — GIL; use processes, and only after removing serial work |
| BLAS threading | E23.3 | worth nothing at these matrix sizes |
| Raw score normalisation | E18.2 | needs the NCE inversion |
| Candidate-side lexicalisation | E19 | double-counts q(w) — but note the NCE inversion is *gone* in first-order, so this could be re-examined |
| The linguistic parameter space as a feature space | E22 | strictly dominated; keep it for reporting (item 12) |
| Open-class rule quota | E25.1 | the budget was never scarce |
| Significance gating, relational slots | E27 | long-range structure is real but redundant |
| Held-out class selection | E29.1 | **but the experiment was confounded — see item 2** |
| Hierarchical parent backoff | E29.2 | the parents are already in the mixture |

---

## Two habits worth keeping

**Measure the intermediate quantity; do not infer it from the aggregate.** Six wrong diagnoses in
this project came from reasoning about mechanism from a symptom. The ones that worked came from
measuring the thing in between — mass on content words before and after truncation (E25.2), mean
information gain per class across windows (E33.2), a profile before a speed plan (E30.1).

**Keep the obvious implementation as a test oracle when you optimise.** `_rollup_reference`,
`_build_reference`, `_pair_blocks_reference` each exist because the fast path they guard produces
every number downstream, and three of the four optimisations in this project were only provably
correct because the slow version was still there to compare against.
