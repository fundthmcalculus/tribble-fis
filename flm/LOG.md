# FLM engineering log

Running record of what was tried, what worked, what didn't, and **why**. Newest entries
at the bottom. Results and rationale live here; the module READMEs carry the settled
conclusions.

Convention: **WORKED** / **FAILED** / **PARTIAL**, each with a why.

> **The current status is the LAST "Standing summary" section in this file.** Earlier ones
> are kept in place, marked superseded, because the *sequence* of conclusions is part of
> the record -- several were later corrected and hiding that would misrepresent how the
> work went.

## Index

| # | Topic | Verdict |
|---|---|---|
| E-B | Experiment B: FIS heads on frozen neural embeddings (predates this log) | harness built, unrun |
| E0 | Environment reconnaissance; forced substitutions | — |
| E1 | M0 coverage gate | PASS (96.7%) |
| E2 | Fuzzy lexical access | WORKED after 5 fixes |
| E3 | Sense assignment; Zadeh complement for negation | complement FAILED |
| E4 | Hierarchy structure; prefix consistency | restriction REQUIRED |
| E5 | Which level discriminates | L2, as predicted |
| E6 | Fuzzy sequence model, first attempt | FAILED (chance) |
| E7 | Fuzzy decoder | WORKED |
| E8 | Adding fuzzy syntax | no help alone |
| E9 | Control: logreg vs FIS on same features | found the real blocker |
| E10 | `MembershipRuleRegressor` | WORKED |
| E11 | Decode metric asymmetry; tagger leak | both FIXED |
| E12 | Testing my own predictions | 2 of 3 FAILED; a headline corrected |
| E13 | Joint next-token ranking | WORKED — first real win |
| E14 | Efficiency: GEMMs, not Cython | WORKED |
| E15 | Order-3 rules; more interactions | order-3 NO, beam YES |
| E16 | Corpus size | CONFIRMED binding |
| E17 | Evaluation leak in my own numbers | FOUND AND FIXED |
| E18 | Generation + perplexity vs n-gram LMs | beats unigram, loses to bigram |
| E19 | Closing the bigram gap: lexicalise the context, not the candidate | WORKED; mixture beats bigram |
| E20 | Why n-grams work; a correction to my own trigram claim | diagnosis + correction |
| E21 | Fuzzy tokenizer and linguistic parameter space (`fuzzytok/`) | built; 2 bugs found by running it |
| E22 | The parameter space wired into the ranker; size/quality frontier | space FAILED (dominated); smallness WORKED |
| E23 | Corpus scale and parallel training | profile overturned the premise; scale went the wrong way |
| E24 | The OOV fix | correct + 5.1x faster; did NOT improve the model |
| E25 | Open-class rule quota | FAILED (no effect); found the real bug was `top_k=20` |
| E26 | Wider context windows (2-32) | FAILED monotonically; spurious long-range rules dilute |
| E27 | Significance gating and relational slots | both FAILED; long-range rules are real but redundant |
| E28 | Word-level consequents (first-order TSK) | **WORKED** — first mixture to beat the bigram (-10.5%) |
| E29 | Consolidating first-order: data, selection, backoff, metrics | **beats bigram AND trigram head-to-head** (219.9 vs 284.4) |
| E30 | Training speed: plan + reformulated pair counting | **4.0x measured**; Cython and GPU both argued against |

---

## E-B — Experiment B: FIS heads on frozen neural embeddings

Predates this log; recorded here for completeness. Full write-up in
[`FIS_ON_EMBEDDINGS_PLAN.md`](FIS_ON_EMBEDDINGS_PLAN.md) and [`exp_b/README.md`](exp_b/README.md).

**What.** A harness sweeping fuzzy and non-fuzzy heads (linear probe, MLP, flat TSK,
fuzzy tree, HME, Ruspini) over cached frozen embeddings, with rule count and antecedent
count reported next to accuracy, plus MRR-style width and character-noise sweeps and
post-hoc dimension naming.

**Status: plumbing verified, science unrun.** The synthetic classification, regression, and
atlas paths all pass; the real encoder and SST paths were never executed because Hugging
Face is outside the egress allowlist.

**Why it still mattered.** Framing it revealed the argument that drove everything after:
a rule like `IF dim_417 is High THEN positive` is *transparent but not interpretable*,
because `dim_417` has no name. The interpretability bottleneck is the representation, not
the inference engine — which is what made Experiment A worth building. Also flagged that
this architecture is already published (Fuzzy Fingerprints), so it is a
replication-plus-comparison, not a contribution.

---

## E0 — Environment reconnaissance

**What.** Probed outbound network before planning anything.

**Result.** Egress is allowlisted. Reachable: GitHub (raw + API, repo-scoped), PyPI.
Blocked (403): arXiv, ACL Anthology, Springer, IEEE, `huggingface.co`,
`gutenberg.org`, `sites.google.com`, `export.arxiv.org`.

**Why it mattered.** Two planned inputs died immediately: **TinyStories** (HF-only)
and **Roget's Thesaurus** (Open Roget's on sites.google.com, 1911 edition on
Gutenberg). Rather than stub them out, I searched for reachable substitutes and found
`nltk/nltk_data` **is** on GitHub — which carries WordNet *and* several corpora.
Doing this probe first avoided writing a pipeline against data I could never load.

**Substitutions chosen**, both behind pluggable interfaces:
- Roget's → **WordNet**. Its 45 lexicographer files play almost exactly the role of
  Roget's 39 Sections.
- TinyStories → **children's narrative prose** (`bryant-stories` +
  `burgess-busterbrown` + `carroll-alice`, ~478K chars) — closest reachable register.
  `corpus.load_local()` reads a real TinyStories JSONL when available.

---

## E1 — M0 coverage gate

**What.** Built `hierarchy.py` + `corpus.py` + `coverage.py` and ran the go/no-go
first, before any embedding code. The plan made this a hard gate because a hierarchy
that cannot name the corpus's words cannot embed them — coverage is an upper bound on
how much text the representation can see at all.

**Round 1 — PARTIAL.** 92.6% content-token coverage (tiny), 91.4% (Brown). Passing,
but the top "uncovered" list was diagnostic: `don't`(110), `i'll`(86), `didn't`(82),
`brown's`(98), plus `himself`, `something`, `anything`.

**Diagnosis.** Not a lexicon gap at all. Two separate bugs of mine:
1. The tokenizer regex `[a-z]+(?:'[a-z]+)?` kept apostrophes, so contractions became
   single unknown types.
2. I had under-listed closed-class words — reflexives and indefinites are exactly the
   kind of word WordNet correctly omits, so they belong in the function-word
   denominator exclusion.

**Round 2 — WORKED.** Expanded contractions before word extraction; extended the
closed-class list. **96.7% (tiny), 94.7% (Brown)** — comfortably over the 85% gate.

**Why the residual matters.** What is left is almost entirely **proper names**:
`alice`(398), `joe`(112), `margery`(73), `william`(62). That is precisely the
named-entity gap the plan predicted, and it is the concrete reason the
Wikipedia-category graft is mandatory rather than optional. WordNet and Roget's both
have zero named entities by design.

---

## E2 — Fuzzy lexical access (stage 1)

**What.** BK-tree candidate retrieval + five similarity channels (Damerau-Levenshtein,
trigram Dice, keyboard-weighted edit, phonetic key, common prefix) aggregated by a
TRIBBLE fuzzy model.

### E2.1 Regressor on a 0/1 target — **FAILED**

`MixtureOfGaussiansFuzzyRegressor` raised `ValueError: Bin edges must be unique`.

**Why.** `partition_output()` quantile-bins the target with `pd.qcut`, and a binary
target produces duplicate bin edges. Structural, not a tuning issue.

**Fix that worked.** Use `MixtureOfGaussiansFuzzyClassifier` over match/no-match and
read `predict_proba` as the membership degree. Also the better *semantic* fit — "is
this surface form that lexeme, and how much?" is a graded membership question, and a
fuzzy classifier's positive-class probability is exactly that.

### E2.2 The `in_vocab` channel — **FAILED (dead feature)**

Explanations claimed "exact vocabulary hit" for *every* candidate.

**Why.** Every BK-tree candidate is drawn *from* the vocabulary, so the flag was
constant 1.0 — zero information, and actively misleading in the audit trail. Removed.

### E2.3 The `cand_freq` channel inside the FIS — **FAILED (veto pathology)**

`littel → little` scored **0.003**; near-identical `freind → friend` scored **0.605**.

**Why.** Two compounding problems, found by printing the feature vectors side by side
rather than guessing:
1. **Sampling bias.** `build_training_pairs` drew training words *uniformly* from the
   vocabulary. A Zipfian vocabulary is nearly all tail, so positives were almost all
   rare words and the positive class's `cand_freq` Gaussian centred near 0 — while at
   inference the true correction is usually a *frequent* word. Classic train/test
   mismatch on one channel.
2. **The deeper error.** Frequency is a **prior, not a similarity**. Rule firing is a
   t-norm *product* across channels, so one tail value collapses the whole score. An
   atypical frequency could veto an obviously-correct match.

**Fixes.** (a) Sample training words proportional to log frequency — helped
(separation +0.67 → +0.70, `hosue → house` started working) but did not fix `littel`,
because `little` sits at the *sparse top* of the frequency distribution where no
sampling scheme gives dense coverage. (b) The real fix: move frequency **outside** the
FIS as a bounded modulator, `degree = sim · (1 − w + w·freq)` with w=0.3. It can
re-rank but never gate. `littel → little` now 0.58. **WORKED.**

### E2.4 Normalising match degrees to sum to 1 — **FAILED (conceptual)**

**Why.** That is probability thinking. A fuzzy set over lexemes has *independent*
degrees — a token can be 0.9 `receive` and 0.8 `relieve` simultaneously, and
rescaling to a simplex destroys exactly the graded information the layer exists to
produce. Removed; total mass is bounded later in composition, where it belongs.

### E2.5 Held-out error-class split — **WORKED (as a diagnostic)**

Train on substitute/delete/double, test on transpose/insert. Separation
**+0.70** in-distribution, **+0.37** across held-out error classes.

**Why the gap.** Transposition lowers trigram overlap while keeping edit distance at
1, so channels learned on other error classes transfer imperfectly. Worth knowing:
training and testing on one noise generator measures only self-consistency.

**Decision.** Ship a model trained on **all five** ops; keep the split purely to
*measure* the gap. Crippling the deployed model to preserve a clean split would be a
false economy.

---

## E3 — Sense assignment and composition (stages 2–3)

### E3.1 `wn.synsets("was")` → **WAS = Washington** — **FAILED**

*"the rabbit was happy"* acquired a strong `administrative district` dimension.

**Why.** WordNet has junk entries for closed-class surface forms, and the state
abbreviation carried the *top* sense prior. Fixed by filtering function words in
`SenseAssigner.assign` (reusing the coverage stoplist). Cheap, and it removed a whole
class of spurious dimensions.

### E3.2 Zadeh's complement for negation — **FAILED (the most interesting one)**

The plan said `not X → 1 − μ`: one line, no training, fuzzy logic earning its keep.
In practice *"not very happy"* came out asserting **strongly felicitous and glad**.

**Why.** A lexeme's sense vector is sparse and peaked — `happy` is
`happy.a.01@1.0, felicitous@0.07, glad@0.01`. The complement zeroes the intended
sense and promotes its weakly-activated siblings to ~0.95. **The complement of a
sparse membership vector is a dense vector of near-ones**, which is semantically
meaningless. This is a real limit on naive fuzzy-logic composition over sparse
representations, and I only saw it by printing the specific node values for negated
vs unnegated versions of one sentence.

**Fix that worked.** Negation = **suppression + antonym transfer**: zero the concept,
move its mass to a WordNet antonym. `not very happy` → `unhappy = 1.00`.

**Why this is satisfying.** It reintroduces exactly the structure **Roget's
antonymous opposed pairs** (648 Goodness / 649 Badness) would have supplied natively —
independent support for the literature review's argument that Roget's is the better
scaffold. WordNet records antonymy on *lemmas*, not synsets, so all lemmas are
consulted; coverage is good for adjectives, thin for nouns.

**Scope of the correction.** The *hedges* (concentration `μ²` for "very", dilation
`μ^0.5` for "somewhat") work exactly as specified. Only the complement breaks. The
docs now say that precisely rather than claiming all of Zadeh's operators transfer.

---

## E4 — Hierarchy structure

### E4.1 Interior nodes as dimensions — **FAILED**

Widths came out `[1, 3, 5, 7]` on a toy tree where `[1, 2, 2, 3]` was right. Caught
by a unit test whose expectation I had written wrong — investigating the mismatch
found a real design wart rather than a bad assertion.

**Why.** Levels were derived from *every registered node*. Since short paths clamp
(which is what makes rollup exact), every ancestor clamped into all deeper levels as
an extra near-always-on coordinate. The root was worst: present at every resolution
carrying the max of everything, inflating similarity denominators for zero
information. Fixed by deriving levels from **terminal** paths only.

### E4.2 Unrestricted hypernym chain for a balanced ladder — **FAILED**

The same-lexname restriction gives an ugly ladder: `[1, 4, 45, 4526, 7257, 9862]` — a
~100× jump from L2 to L3 with no usable middle resolution. Dropping the restriction
improved it to `[1, 4, 45, 3245, 4582, 5804]`. So I tried it.

**It broke rollup exactness** — `run_flm --stage embed` reported FAIL, and L2
similarities scrambled as a downstream consequence (mean SIM fell *below* mean DIF,
gap −0.189).

**Why.** Exact rollup needs node paths to be **prefix-consistent**: every prefix of a
terminal's path must be the registered path of that ancestor. `dog.n.01` is
`lex:noun.animal`, but its ancestor `entity.n.01` is `lex:noun.Tops`, so `entity`'s
own canonical path `(*, pos:n, lex:noun.Tops, entity)` is not a prefix of `dog`'s.
Projection then disagrees between levels.

**Decision.** Reverted. The same-lexname restriction is **required for correctness,
not a stylistic preference**, and the docstring now says so. Exactness is the entire
claim that separates this from Matryoshka truncation; a prettier ladder is not worth
voiding it.

**Why the residual imbalance is itself a result.** It has two structural causes:
same-supersense chains are short, and **WordNet has no adjective hypernyms at all**
(adjectives are organised by antonymy and similarity, not subsumption), so every
adjective and adverb clamps at depth 3. A uniform ladder over all parts of speech is
therefore impossible in WordNet. Roget's designed 6/39/79/596/990 tree would not have
this problem. This is the sharpest *measured* argument for Roget's in the whole
project.

---

## E5 — Which level actually discriminates

**What.** Rather than assume, measured per-level fuzzy Jaccard on hand-built
similar/unrelated sentence pairs.

| pair | | L1 | **L2** | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| happy child / joyful boy | sim | .333 | **.333** | .000 | .000 | .000 |
| dog barked / wolf howled | sim | .581 | **.235** | .016 | .004 | .000 |
| girl ate bread / boy ate food | sim | .974 | **.407** | .300 | .270 | .267 |
| dog barked / king spoke | dif | .453 | **.062** | .033 | .018 | .001 |
| happy child / stone was cold | dif | .697 | **.187** | .000 | .000 | .000 |
| girl ate bread / mountain tall | dif | .421 | **.000** | .000 | .000 | .000 |

**WORKED — and confirmed a prediction.** L1 (4 dims) is too coarse to separate
anything; L3+ is so sparse that near-synonyms share no coordinate. **L2 (the 45
supersenses) is the only level that orders the pairs correctly.** The plan predicted
the supersense width would be the design centre *before* this was measured, so this is
confirmation rather than post-hoc rationalisation.

Final at L2: mean SIM **0.344**, mean DIF **0.066**, gap **+0.278**, complete
separation.

### E5.1 `hierarchy_jaccard` weighting — **FAILED, then fixed**

It originally decayed weights geometrically from the *finest* level, i.e. full weight
on the sparsest and least informative resolution. Near-synonyms share no leaf synset,
so that term is 0 and it dominated the average, collapsing the ordering the metric
exists to produce. Now weights peak at the measured discriminative level and decay in
both directions.

---

## E6 — Fuzzy sequence model, first attempt

**What.** Windowed level-2 memberships of the previous *k* tokens → membership vector
of the next. `MimoGaussianPredictor` over the most active output dimensions.

### E6.1 MIMO regressor — **FAILED (same qcut wall as E2.1)**

Membership coordinates are mostly zero, so `pd.qcut` hit duplicate bin edges again.
Replaced with one fuzzy classifier per output dimension, `predict_proba` as the
degree. Consistent with E2.1 and, again, the better semantic framing.

### E6.2 The model itself — **FAILED (negative result, reported as one)**

```
windows=2500  features=90  outputs=8/45
binarised task : separation=+0.015  balanced-acc=0.528   (chance=0.500)
continuous MAE : test=0.4295        mean-baseline=0.1493  -> LOSES TO baseline
```

**Why report both metrics.** They disagree and each alone misleads. A probability
calibrated near 0.5 *cannot* beat a mean baseline on MAE when targets are sparse, so
MAE understates the model — but separation of +0.015 confirms there is almost no skill
to understate. Reporting only MAE would look like a calibration problem; reporting
only separation would hide that `predict_proba` is not a drop-in membership degree.

**Why it fails — diagnosis, not a shrug.** Structural, not hyperparameter. What
predicts the next word is mostly **syntax and function words**, and the semantic
embedding discards both by construction: `the`, `of`, `was` carry no WordNet
membership and are invisible. A two-token window of supersenses can say "an animal was
just mentioned"; it cannot say "a determiner was just used, so a noun is due".

**Why that is useful anyway.** It is the sharpest evidence yet for the plan's §8
claim: a fuzzy LM needs a fuzzy model of **syntax** beside its fuzzy model of
**semantics**. The decoder machinery works; it is being fed a near-uninformative
signal.

---

## E7 — Fuzzy decoder — **WORKED**

Zadeh **linguistic approximation** over a precomputed lexeme atlas (2757 decodable
words × 45 named dims): score every vocabulary lexeme by fuzzy Jaccard against the
predicted vector, sample.

Two properties worth the architecture:
- **Temperature is a Zadeh hedge.** Sharpening is concentration (`μ^e`, e>1),
  flattening is dilation. No bolted-on temperature parameter — `e = 2` is literally
  "*very* like the prediction".
- **Every step is auditable**: which named dimensions drove it, which lexemes
  competed, at what degree. No vocabulary-sized opaque distribution.

Needed a crude repetition guard (block the last 3 emitted tokens): without it a
self-similar prediction is a fixed point and generation collapses to one word
repeated.

**Honest scope.** Output is a *semantic-class walk*, not grammatical text. Function
words carry no membership so they can never be emitted. That is the representation's
property, and it is the same root cause as E6.2.

---

## E8 — Adding fuzzy syntax (in progress)

**What and why.** Acting on the E6.2 diagnosis rather than tuning around it. New
`syntax.py`: a small **named** feature space of closed-class syntactic categories
(DETERMINER, PRONOUN, POSSESSIVE, AUXILIARY, PREPOSITION, CONJUNCTION, NEGATOR,
QUANTIFIER, WH_WORD, INTENSIFIER, INFINITIVE_TO) plus graded open-class markers
(OPEN_NOUN/VERB/ADJ/ADV) and a BOUNDARY marker, concatenated onto the semantic
vector.

Interpretability is preserved because these dimensions are named too — a rule can
read `IF prev1[DETERMINER] is High THEN next[OPEN_NOUN] is High`, a checkable claim
about English.

**Pleasant confirmation:** syntax turns out to be genuinely fuzzy. `to` is both
preposition and infinitive marker; `that` is determiner, pronoun, and complementiser;
`her` is pronoun and possessive; `no` is determiner and negator. Represented as
simultaneous partial memberships rather than forced to one label.

Closed-class degrees are **possibilistic** (1.0 in every category a word can take,
not normalised) — consistent with E2.4, and disambiguating them would need a tagger
this module deliberately avoids depending on. Open-class degrees *are* graded, by the
fraction of a lemma's senses in each part of speech.

*(Result pending — see next entry.)*


---

## E9 — The control that found the real blocker

**What.** Same features, same splits, three models on the binarised
"does category C come next?" task. This is the experiment that should have come before
adding syntax.

| target | logreg AUC | FIS/gaussian AUC | FIS/trapezoid AUC |
|---|---|---|---|
| OPEN_NOUN | **0.673** | 0.488 | 0.500 (spread 0.000) |
| OPEN_VERB | **0.752** | 0.511 | 0.500 (spread 0.000) |
| DETERMINER | **0.730** | 0.515 | 0.500 (spread 0.000) |

**Decisive.** Logistic regression learns; the FIS is at chance, and the trapezoid
variant emits a literal constant (standard deviation 0.000). So **the features carried
signal all along and the antecedent representation was the blocker.** Without this
control I would have concluded "syntax does not help language modelling", which is
false.

**Why the FIS fails.** The feature matrix is **93% zeros**. TRIBBLE fits a Gaussian (or
trapezoid) membership function per `(feature, class)`, which presumes a continuous,
unimodal, reasonably-spread variable — true of its benchmark data (concrete strength,
turbine power, wine chemistry), false of a sparse membership vector. Fit a Gaussian to
`{0 w.p. 0.95, 1 w.p. 0.05}` and you get a narrow near-zero Gaussian for *both*
classes; per-feature memberships then barely differ, the t-norm product of
near-identical values is near-identical, and `predict_proba` collapses to 0.5.

**This is a finding about TRIBBLE, not just about this experiment.** Any attempt to feed
sparse fuzzy membership vectors into a Gaussian-antecedent TSK system will hit it.

---

## E10 — `MembershipRuleRegressor` — **WORKED**

**The insight.** When the inputs are *already* membership degrees, there is no
membership function to fit — **the input value *is* the membership degree**. The whole
antecedent-fitting layer is not merely ill-suited, it is redundant. What is left to
learn is rule *structure* and *consequents*.

So (`rules.py`): a zero-order TSK where an antecedent is a named conjunction, its
firing strength is the t-norm of the raw input values, its consequent is the
firing-weighted target mean, and prediction is the firing-weighted blend plus a default
rule at the prior.

**Result — matches or beats logistic regression:**

| target | logreg AUC | rules AUC |
|---|---|---|
| OPEN_NOUN | 0.664 | 0.643 |
| OPEN_VERB | 0.745 | 0.692 |
| DETERMINER | 0.742 | **0.747** |
| PREPOSITION | 0.776 | **0.784** |
| adj.all | 0.634 | 0.623 |

**And the rules are linguistically correct**, recovered from data:

```
IF prev1:DETERMINER                        THEN next[OPEN_NOUN] ~ 0.675  (default 0.364)
IF prev1:adj.all AND prev2:DETERMINER      THEN next[OPEN_NOUN] ~ 0.788
IF prev1:POSSESSIVE                        THEN next[OPEN_NOUN] ~ 0.754
IF prev1:OPEN_VERB                         THEN next[OPEN_NOUN] ~ 0.243
```

Determiners precede nouns; "the red ___" is a noun; "my ___" is a noun; after a verb a
noun is *less* likely. That is English noun-phrase syntax, in rules a human can read,
at parity with a linear model. Interpretability is *better* than the Gaussian version,
not worse — an antecedent is literally "the previous token is a determiner, to degree
0.9", with no fitted centre or width to explain.

### E10.1 Redundant conjunctions — **FIXED**

The rule base contained `IF prev1:adj.all AND prev1:OPEN_ADJ` — two names for "the
previous token is an adjective". Harmless numerically, but it wastes the reader's
attention, which is the entire budget an interpretable model is spending. Now a pair is
kept only if its consequent lands *further from the default than both parents*, i.e. it
encodes a genuine interaction.

### E10.2 Both changes were necessary — and neither alone sufficed

| model | features | separation | bal-acc |
|---|---|---|---|
| Gaussian FIS, semantics only | 90 | +0.015 | 0.528 |
| Gaussian FIS, semantics+syntax | 122 | +0.017 | 0.534 |
| rule learner, semantics only | 90 | +0.078 | 0.505 |
| **rule learner, semantics+syntax** | 122 | **+0.129** | **0.569** |

Syntax without the right antecedents does nothing; the right antecedents without syntax
help a little. Together: separation **+0.015 → +0.129** (8.6×), balanced accuracy
**0.528 → 0.569**. **Testing either change in isolation would have given the wrong
conclusion** — the honest reading is that they are jointly necessary.

Predictions are now context-sensitive and sensible: `she was very` →
`adj.all 0.416, OPEN_ADJ 0.332`; `she ran to` → `DETERMINER 0.438`; `the little rabbit`
→ `OPEN_VERB 0.304`.

MAE still loses to the mean baseline (0.195 vs 0.188) but the gap narrowed a lot. Same
calibration caveat as E6.2 — judge by separation, not MAE.

---

## E11 — Decoder metric: symmetric Jaccard was the wrong asymmetry — **FIXED**

**Symptom.** With `t[OPEN_NOUN]=0.44` the top decode candidates were `jolly`, `fourth`,
`in`, `high` — not nouns.

**Why.** The predicted vector is a **marginal** (a degree per category, spread over
many); a lexeme's vector is **one peaked pattern**. Symmetric fuzzy Jaccard
(`Σmin/Σmax`) therefore rewards words whose pattern resembles the marginal
*distribution* — bland words with membership spread thinly everywhere — and penalises
the pure noun the prediction asked for. `cat` (mass 1.0, concentrated) lost to `jolly`
purely for being less spread out.

**Fix.** An asymmetric **coverage** score, `Σ_c min(t_c, w_c) / Σ_c w_c`: how much of
*this word's own* membership mass sits in categories the prediction called for.
Normalising by the word's mass removes the spread bias. Symmetric Jaccard is retained
behind a flag for comparison, and `similarity.py` still uses it for
sentence-to-sentence similarity, where the symmetry *is* correct.

### E11.1 Open-class leak in the tagger — **FIXED**

Even after the metric fix, `OPEN_NOUN` retrieved `somebody`, `o`, `t`.

**Why.** `SyntaxTagger` read `lemma_synsets` directly and did not apply the
function-word filter that `SenseAssigner` does. Closed-class words therefore acquired a
pure `OPEN_NOUN=1.0` and nothing else — and because the coverage metric normalises by a
word's own mass, those single-coordinate vectors scored a perfect 1.0 and topped every
retrieval. A filter inconsistency between two modules, surfaced only by the new metric.
Also dropped single-character tokens, artifacts of contraction expansion.

**Retrieval after both fixes** (pure-category probes) — all correct:

```
OPEN_NOUN   -> mile, chimney, fame, paris, street, hen, city
OPEN_VERB   -> declare, argued, doubted, remarked, remarking, petted
OPEN_ADJ    -> alive, headstrong, hateful, harsh, famous, hotter
DETERMINER  -> those, these, the, another, this, an, no
PREPOSITION -> from, of, upon, through, in, since, during
AUXILIARY   -> must, can, were, did, am, shall, are
```

The decoder is now correct at the category level, and function words are decodable at
all (they were skipped entirely under the semantics-only atlas).

---

## Where it stands

**Working and measured:** coverage gate (96.7%), exact multi-resolution rollup, L2
semantic similarity (gap +0.278), explainable typo robustness, named-rule syntax
learning at logreg parity, category-correct decoding. 24 tests green.

**Not working:** end-to-end generation is still not grammatical. Balanced accuracy 0.569
is real skill but nowhere near enough to chain 8 tokens into a sentence, and the
generated text remains a category-appropriate word salad.

**What the evidence says to do next**, in order:
1. **Widen the context.** A 2-token window cannot represent a clause. Rules over
   `prev3`/`prev4` and a sentence-position feature are cheap and the obvious first move.
2. **Predict the full joint target, not 12 dimensions.** Truncating to the 12 most
   active outputs means most of the vector is the corpus prior at decode time, which
   flattens exactly the distinctions the decoder needs.
3. **Order-3 rules.** Capped at 2 for readability; English NP/VP patterns plainly need
   three-term conjunctions ("determiner, adjective, then noun").
4. **Do not** reach for the Gaussian-antecedent estimators on this data again (E9).

---

## E12 — Testing my own three predictions — **2 of 3 FAILED**

E11 ended with a ranked list of what to try next. Testing them mattered, because the
reasoning sounded convincing and two of the three were simply wrong.

| prediction | reasoning given | measured |
|---|---|---|
| widen context 2 → 3 | "a 2-token window cannot represent a clause" | **no effect**: bal-acc 0.527±0.010 → 0.533±0.005 (3 seeds) |
| allow order-3 rules | "English NP patterns need three-term conjunctions" | **no effect**: sep +0.094 → +0.096, bal-acc 0.527 → 0.530 |
| predict full joint target | truncating to 12 outputs flattens the decode signal | **untested** — too slow at 45+ outputs to finish here |

Order-3 required implementing general level-wise growth first (the previous code silently
treated `max_order=3` as 2), so the null result is a real measurement, not a no-op.

**Why record a null result this prominently.** Both predictions were mine, both were
plausible, and acting on either without measuring would have wasted effort and left a
false claim in the README. Whatever the sequence model is missing, it is not context
width or rule order.

### E12.1 The 0.569 headline was inflated by a bug — **CORRECTED**

Seed variance turned out small (sd ≈ 0.005-0.010 over 3 seeds), which made a discrepancy
impossible to dismiss: the same nominal configuration measured **0.569** before the E11.1
tagger fix and **0.527 ± 0.010** after. Four standard deviations apart, so not noise.

**Why.** The leaked pure-`OPEN_NOUN=1.0` signature on closed-class words was accidentally
acting as a *distinctive marker* for function words, and the rule learner was using it.
Fixing the leak was still right — it was required for the decoder to retrieve nouns at
all — but it removed a real (if illegitimate) feature.

**Corrected claim.** The rule learner's advantage over the Gaussian-antecedent FIS is
genuine but smaller than first reported: separation ~6x better (+0.015 → +0.094), while
balanced accuracy is **barely above chance**. The README and the previous commit message
overstated it; both are now corrected.

**Lesson.** A fix in one module silently re-benchmarked another. Measuring seed variance
is what surfaced it — without an error bar, 0.569 versus 0.527 looks like ordinary run-to-run
drift and the inflated number stands.

### E12.2 Level-wise growth was too slow to sweep — **FIXED**

The first order-3 implementation recomputed each candidate's firing vector from scratch,
`O(beam x top_singles x k)` full-length products per output per level. It could not finish
a single 3-seed sweep. Extending a rule is now one vectorised multiply against the parent's
cached firing vector; `beam` also dropped 40 → 24.

Separately, `_windows` rebuilt every token's embedding on each fit — the full compose
pipeline per token, several times per corpus pass — which dominated fit time. Token vectors
are now cached on the instance. Fit time is ~20s per configuration, which is what made the
3-seed variance measurement above affordable at all.

---

## Standing summary (SUPERSEDED — see the final one)

**Working, measured, and tested:** coverage gate (96.7%), exact multi-resolution rollup
(in CI), L2 semantic similarity (gap +0.278, complete separation), explainable typo
robustness, linguistically-correct named rules at logistic-regression parity per target
(AUC 0.64-0.78), category-correct decoding across six probed categories. 25 tests green.

**Not working:** aggregate next-token skill is marginal (bal-acc 0.527 ± 0.010) and
generation is not grammatical.

**Ruled out as the cause:** context width (E12), rule order (E12), the antecedent
representation (E9/E10 — fixed), the decode metric (E11 — fixed), feature hygiene
(E11.1 — fixed).

**Still open, in the order I would test:** corpus size (~90K tokens is ~50x under a
TinyStories subset); the independent-per-dimension target space (nothing enforces that
the prediction corresponds to *one* word); and the absence of any clause-level state.

---

## E13 — Joint target prediction — **WORKED.** The first real win over a baseline

**What.** The one untested item from E12, and the one I had flagged as possibly the
deeper design flaw. `sequence.py` predicts a degree per named dimension
*independently*; nothing enforces that the result describes **one actual word**. A
vector with `OPEN_NOUN=0.5, OPEN_VERB=0.5` is a fine marginal prediction and a
description of nothing.

So the target was reframed (`joint.py`): instead of "what degree does dimension *c*
have next?", ask **"is *w* the next word here?"** — one binary, genuinely joint
question, with features = context-window memberships ⊕ the candidate's own. This is
noise-contrastive estimation with an interpretable scorer, and it buys a real metric:
rank the true next token against frequency-sampled distractors → MRR, hits@k.

Distractors are **frequency-weighted**, not uniform. Uniform sampling from a Zipfian
vocabulary draws almost only rare words, which makes the task trivially easy and the
numbers meaningless.

### E13.1 Measuring the ceiling first — **the representation was NOT the bottleneck**

Before tuning anything, I checked whether the task was winnable at all. At L2 the joint
space has 61 dims; among 2897 decodable words there are **2133 distinct signatures**
(largest collision buckets 156, 123, 99). A perfect scorer, with pessimistic tie-breaking
on identical signatures, would reach **MRR 0.822, hits@1 0.698**.

**Why this mattered.** My working hypothesis had been that L2 was too coarse to rank
individual words. It is not. That killed a plausible and expensive line of work
(re-decoding at L3/L5) before I started it, and pointed the blame at the learner.

### E13.2 First run — **FAILED, and the rules said exactly why**

MRR 0.191 vs unigram-frequency 0.196 vs chance 0.180. Worse than a frequency baseline.

The rule dump was unambiguous: **every top rule was `cand:`-only.** The base had learned
the marginal prior over candidate types ("which supersenses tend to be next tokens") and
not one context-dependent rule. With no `ctx:`×`cand:` interaction, the scorer literally
could not depend on context.

### E13.3 Two separate causes, found in order

**Cause 1 — marginals crowd out interactions.** `lift = (consequent − default) ×
√support` systematically favours high-support single-term rules. Fixed with
`order_quota`, reserving a share of the rule budget per antecedent order.

That helped less than expected: the quota asked for 75–85% order-2 and got **2 rules**.
So cross rules were being rejected at *generation*, not selection.

**Cause 2 — pure-interaction structure is invisible to lift-based seeding.** Growth seeds
from the top-*k* singles by |lift|. But for a given position, the positive and all eight
negatives **share one context vector**, so every `ctx:` feature fires identically on both
and its firing-weighted target equals the base rate — marginal lift *exactly zero*. Context
features could never enter the seed pool, so no ctx×cand pair was ever generated. The two
order-2 rules that did appear were cand×cand.

This is the **XOR problem for greedy feature selection**: informative jointly, invisible
marginally. Fixed with `seed_features`, which force-admits given indices to the seed pool
regardless of lift. `test_rules_find_pure_interaction_only_when_seeded` is the regression
test, and it asserts both halves — that the interaction is missed without seeding and found
with it.

**Also encoded:** every rule must touch a candidate feature (`must_include`). A
context-only rule fires identically for all candidates, shifts every score by the same
constant, and therefore *cannot* change a ranking — it would silently consume budget,
and context features outnumber candidate features by the window size.

### E13.4 After the fixes — real grammar, learned from data

```
IF ctx:prev1:PRONOUN       AND cand:AUXILIARY       THEN P(next) ~ 0.373   (default 0.112)
IF ctx:prev1:AUXILIARY     AND cand:NEGATOR         THEN P(next) ~ 0.559
IF ctx:prev1:DETERMINER    AND cand:noun.food       THEN P(next) ~ 0.596
IF ctx:prev1:INFINITIVE_TO AND cand:verb.cognition  THEN P(next) ~ 0.503
IF ctx:prev1:verb.motion   AND cand:PREPOSITION     THEN P(next) ~ 0.326
IF ctx:prev2:WH_WORD       AND cand:AUXILIARY       THEN P(next) ~ 0.364
```

"he ___" takes an auxiliary; "was ___" takes *not*; "the ___" takes a noun; "to ___"
takes a verb; "ran ___" takes a preposition; wh-questions invert to an auxiliary. That is
English grammar in readable rules, at 3–5x the base rate.

### E13.5 Beam, not `max_rules`, was the binding constraint

Scaling `max_rules` saturated at 81 rules regardless (40 → 0.235, 80 → 0.244, 160/320/640 →
0.244). Cause: `frontier = grown[:beam]` with `beam=24` caps how many order-2 rules survive
per growth level, so the interaction supply was fixed at 24 no matter the rule budget.

| config | MRR | hits@1 | hits@5 | hits@10 | rules |
|---|---|---|---|---|---|
| chance | 0.180 | — | — | — | — |
| unigram frequency | 0.196 | 0.075 | 0.280 | 0.492 | 0 |
| joint, beam 24 | 0.244 | **0.100** | 0.343 | 0.578 | 80 |
| **joint, beam 200** | **0.257** | 0.095 | 0.403 | 0.635 | 257 |
| joint, beam 400 | 0.254 | 0.085 | **0.415** | **0.680** | 457 |
| oracle ceiling | 0.822 | 0.698 | — | — | — |

Defaults are now `beam=200`, the measured MRR optimum. Note the tradeoff: hits@1 peaks at
the *smallest* rule base (0.100 at 80 rules) and declines as rules are added, while hits@5
and hits@10 keep improving. With 457 rules the firing-weighted blend averages over more
rules, which sharpens mid-rank ordering and smooths the top. Raise the beam if recall
matters more than top-1.

### E13.6 Honest accounting

**MRR 0.196 → 0.257 over the unigram baseline (+31% relative); hits@10 0.492 → 0.680.**
That is the first result in this project that clearly beats a real baseline on a real
language-modelling metric, and the scorer is a readable rule base rather than a black box.

But it captures only **(0.257 − 0.180) / (0.822 − 0.180) ≈ 12%** of the available headroom
above chance, and hits@1 stalls near 0.10. The remaining gap is a coverage problem: 257
category-level rules over a 122 × 61 space of possible interactions is sparse, and rules at
the supersense level cannot separate `bread` from `cake`. The oracle can, because it may use
the full 61-dim signature.

**Next, in order:** (1) more interactions — the beam curve had not flattened on hits@k;
(2) order-3 rules *on this task* — worth retrying even though E12 found them useless for
the marginal formulation, since ctx×ctx×cand is a different structure; (3) a bigger corpus,
still the leading suspect at ~90K tokens; (4) finer candidate features for within-category
discrimination — the one thing that can lift hits@1.

---

## E14 — Efficiency: two GEMMs instead of a candidate loop — **WORKED**

**The ask** was order-3 rules plus more interactions, with Cython if needed. Cython was
not needed and would have been the wrong tool.

**Why.** With the **product** t-norm, an entire growth level's statistics factor into
matrix products over the frontier firing matrix ``F`` and the seed columns ``S``::

    support  = F.T @ S
    weighted = F.T @ (S * y)        ->  consequent = weighted / support

The old code made one Python-level numpy call per candidate; at beam 200 that is ~30k
calls per level, and **call overhead, not arithmetic, was the bottleneck**. Hand-writing
that loop in C would at best match one thread of an optimised BLAS. Expressing it as
linear algebra instead is both faster and shorter.

Only survivors get their firing vector materialised — the GEMM yields every candidate's
statistics without ever forming its firing vector.

Measured on a joint-task-shaped problem (13500 x 183):

| order | beam | fit time | rules |
|---|---|---|---|
| 2 | 24 | 0.10s | 68 |
| 2 | 200 | 0.14s | 115 |
| 2 | 800 | 0.44s | 244 |
| 3 | 200 | 0.24s | 181 |
| 3 | 800 | **0.95s** | 474 |

Order-3 at beam 800 in under a second; the previous implementation could not finish a
single 3-seed sweep at order 3 at all.

``min`` does not factor this way (it is not bilinear), so it keeps the per-candidate
loop and is correspondingly slower — documented, and covered by
``test_min_tnorm_still_works``. ``test_batched_growth_matches_bruteforce`` asserts the
GEMM path reproduces per-candidate support, consequent, and lift exactly.

## E15 — Order-3 on the joint task — **FAILED (again)**; more interactions — **WORKED**

Both asks, measured together. 500 held-out positions, 20 candidates, 3000 training
positions.

| config | MRR | hits@1 | hits@5 | hits@10 | rules |
|---|---|---|---|---|---|
| unigram frequency | 0.189 | 0.068 | 0.266 | 0.486 | 0 |
| order 2, beam 200 | 0.265 | 0.116 | 0.378 | 0.620 | 259 |
| **order 2, beam 800** | **0.279** | **0.126** | 0.416 | 0.660 | 859 |
| order 2, beam 2000 | 0.277 | 0.124 | 0.414 | 0.656 | 894 |
| order 3, beam 200 | 0.265 | 0.114 | 0.382 | 0.622 | 459 |
| order 3, beam 800 | 0.275 | 0.118 | **0.430** | **0.662** | 1659 |
| order 3, beam 2000 | 0.277 | 0.120 | 0.424 | 0.654 | 1761 |

**More interactions: yes.** beam 200 -> 800 lifts MRR 0.265 -> 0.279 and hits@1
0.116 -> 0.126. It saturates at 800 for a reason worth noting: the *candidate supply*
runs out (~835 admissible order-2 rules exist), so beam 2000 changes nothing. The
ceiling is the interaction space, not the beam.

**Order-3: no.** 0.275 vs 0.279 at matched beam — slightly worse on MRR and hits@1,
marginally better on hits@5 (0.430 vs 0.416), for **double** the rule count. Same verdict
as E12 for the marginal formulation, now confirmed for the joint one, so it was worth
retesting on the different structure rather than assuming the earlier null carried over.

**But order-3 gives better *explanations* at no ranking cost**, which is a genuine
interpretability-vs-accuracy split worth recording:

```
IF ctx:prev2:CONJUNCTION AND ctx:prev1:PRONOUN     AND cand:AUXILIARY     -> 0.473
IF ctx:prev2:OPEN_NOUN   AND ctx:prev1:PREPOSITION AND cand:POSSESSIVE    -> 0.676
IF ctx:prev2:WH_WORD     AND ctx:prev1:PRONOUN     AND cand:AUXILIARY     -> 0.550
IF ctx:prev1:verb.change AND ctx:prev1:OPEN_VERB   AND cand:INFINITIVE_TO -> 0.456
```

"and he ___" takes an auxiliary; "house of ___" takes a possessive; "what he ___" takes
an auxiliary; "began ___" takes *to*. Richer grammar than any order-2 rule states, at
4-6x the base rate. Use order 3 when the rule base is the deliverable, order 2 when the
ranking is.

Caveat: some order-3 rules conjoin two features at the *same* position
(``prev1:verb.change AND prev1:OPEN_VERB``). Not redundant — ``verb.change`` is strictly
more specific than ``OPEN_VERB`` — but it reads like padding, and the interaction gate
admits it because the conjunction genuinely narrows.

## E16 — Corpus size is the binding constraint — **CONFIRMED, and it is the biggest lever**

Fits are near-free after E14, so the standing "corpus too small" hypothesis became cheap
to test in-sample.

| training positions | MRR | hits@1 | hits@5 | hits@10 | rules |
|---|---|---|---|---|---|
| unigram frequency | 0.189 | 0.068 | 0.266 | 0.486 | 0 |
| 750 | 0.260 | 0.112 | 0.400 | 0.614 | 470 |
| 1500 | 0.262 | 0.098 | 0.410 | 0.688 | 569 |
| 3000 | 0.279 | 0.126 | 0.416 | 0.660 | 859 |
| 6000 | 0.293 | 0.140 | 0.422 | 0.704 | 860 |
| **12000** | **0.319** | **0.142** | **0.504** | **0.784** | 860 |
| oracle ceiling | 0.822 | 0.698 | — | — | — |

Monotonic, and **still rising when the corpus runs out** — 12000 positions is a large
fraction of the ~90K-token children's corpus. Headroom captured above chance goes from
12% to **(0.319 − 0.180) / (0.822 − 0.180) ≈ 22%**.

**Why this is the important finding.** Rule count saturates at ~860 across the entire
range, so the gain is *not* more rules — it is better-estimated **consequents**. The
rules were already discovered at 750 positions; their consequents were just noisy. So the
bottleneck is data, and the single highest-value action for this project is getting the
real TinyStories corpus (blocked here by egress) rather than any further architectural
change.

Defaults are now the measured optimum: ``beam=800``, ``max_positions=12000``,
``max_order=2``.

## Standing summary (SUPERSEDED — see the final one)

**Best joint result:** MRR **0.319** vs 0.189 unigram (**+69% relative**), hits@1 0.142
vs 0.068 (**+109%**), hits@10 0.784 vs 0.486 — with a readable 860-rule base that
recovers English grammar, on an oracle-capped-0.822 task.

**Ruled out:** context width (E12), rule order (E12 marginal, E15 joint), Gaussian
antecedents (E9/E10), symmetric decode similarity (E11), feature-filter inconsistency
(E11.1), representation coarseness (E13.1 — oracle 0.822 disproved it), Cython (E14 —
the work is BLAS-bound).

**Confirmed binding:** corpus size (E16).

**Still open:** within-category discrimination — supersense rules cannot separate
``bread`` from ``cake``, which is what caps hits@1 near 0.14; and generation, still not
grammatical.

### E16.1 Replication on an independent, larger corpus — **trend confirmed**

The children's corpus ran out of text at 12000 positions, so the rising curve could have
been an artifact of that one corpus. Re-run on Brown (news+fiction+romance, 191K tokens,
17.8K types, 6000-type vocabulary):

| brown positions | MRR | hits@1 | hits@5 | hits@10 |
|---|---|---|---|---|
| unigram frequency | 0.193 | 0.077 | 0.248 | 0.458 |
| 6000 | 0.253 | 0.090 | 0.408 | 0.688 |
| 12000 | 0.273 | 0.117 | 0.432 | 0.722 |
| 25000 | **0.285** | **0.123** | **0.437** | **0.750** |

Monotonic and **still rising at 25000 positions** — twice as far as the children's corpus
allowed. The data-scaling finding is therefore a property of the method, not of one
corpus.

**Absolute values are lower than the children's corpus at matched positions** (0.273 vs
0.319 at 12000) and that is expected, not a regression: Brown has 17.8K types against
5.3K, a 6000-type vocabulary against 3000, and a far more varied register. More
vocabulary diversity makes the 20-candidate ranking harder. So the *level* is
corpus-dependent while the *trend* replicates — which is exactly the pair of claims worth
making.

---

## E17 — Evaluation leak in my own ranking numbers — **FOUND AND FIXED**

Discovered while planning the perplexity comparison, not by a failing test.

`JointNextTokenRanker.build` and `.evaluate` both iterated `corpus.sentences` under
different shuffles. Different *positions*, but the **same sentences** — so held-out
positions came from text the model had trained on. **Every ranking number reported before
this (E13-E16, including MRR 0.319) is optimistic.**

Fixed with `Corpus.split`, a sentence-level train/test split. Vocabulary and counts stay
shared on purpose: they are properties of the lexicon and the frequency prior, not of the
split, and re-deriving them per side would change the candidate set between train and test
and make the two incomparable. `test_corpus_split_is_disjoint_by_sentence` guards it.

**Lesson.** "Different random seed" is not a split. I had reached for a seed offset in
`evaluate` and mistaken it for held-out data; it only reorders the same material.

## E18 — Fuzzy generation and the comparison to conventional LMs

**What.** `generate.py` — a language model built on the joint ranker rather than the
marginal decoder, plus `baselines.py` for the comparison.

**On GPT-2.** It cannot run here: the weights come from Hugging Face, outside the egress
allowlist. `baselines.gpt2_perplexity` implements it for a machine that can reach HF,
under the same restriction so the number is comparable. But the comparison is also
scientifically weak on its own: GPT-2 saw ~40GB, this model sees ~90K tokens, so GPT-2
wins by a margin that measures the *data* gap, not either method. **The controlled
baseline at this scale is an n-gram LM on the same corpus**, which is what was run — and
n-grams are genuinely strong on 90K tokens.

### E18.1 Making the comparison possible at all

A joint *score* is not a distribution, so there was no shared yardstick with any
conventional LM. Normalising the scores over the whole vocabulary gives `p(w | context)`
and therefore perplexity. Comparability rules applied to every model: same sentence
split, same vocabulary, and the same restriction to positions whose gold token the fuzzy
model can represent (94.5% of held-out tokens; the rest are skipped and the coverage is
reported, because a floor probability on unrepresentable words would let the floor set the
number).

### E18.2 Raw score normalisation — **FAILED badly**

| model | perplexity |
|---|---|
| uniform floor | 2897.0 |
| **fuzzy, raw normalised** | **2477.8** |
| 2-gram | 279.2 |

Almost no information over uniform, and **9x worse than a bigram**. The generation trace
showed why immediately: every alternative sat at p ≈ 0.001-0.002. The distribution was
nearly flat.

**Why.** The scores are bounded in [0, 1] and span less than one order of magnitude
across candidates, while a language model needs ratios of ~10^3 between likely and
unlikely words. No amount of rule-learning fixes that: it is the *output transform* that
was wrong.

**This also reframes E13-E16.** MRR 0.319 looked respectable because ranking against 19
frequency-sampled distractors mostly rewards getting the **category** right. Perplexity
over 2897 candidates demands the **word**, and exposed that the model had very little
word-level information. Two metrics, very different verdicts, and the harsher one is the
more honest for a generative model.

### E18.3 The NCE inversion — **WORKED, 6.4x**

The ranker was trained contrastively, so its output is not `p` but

    s = p(w|ctx) / (p(w|ctx) + k*q(w))

with `q` the noise distribution the negatives were drawn from. Solving for `p`:

    p(w|ctx)  proportional to  q(w) * s / (1 - s)

The odds ratio `s/(1-s)` is unbounded, so it can express the needed dynamic range, and
multiplying by `q` restores the unigram information the contrastive objective deliberately
factored out — the model was never asked to learn frequency, because negatives were drawn
*from* frequency.

| model | perplexity | nll |
|---|---|---|
| uniform (floor) | 2897.0 | 7.971 |
| 1-gram (same data) | 472.9 | 6.159 |
| **2-gram (same data)** | **279.2** | 5.632 |
| 3-gram (same data) | 370.2 | 5.914 |
| fuzzy, raw normalised | 2477.8 | 7.815 |
| **fuzzy, NCE-corrected** | **385.8** | 5.955 |

**Honest positioning: the fuzzy LM beats a unigram (386 vs 473), is level with a trigram
(386 vs 370), and loses to a well-smoothed bigram (386 vs 279).** On 90K tokens, a
2-token window, and 860 human-readable rules.

**The gain over unigram is exactly what the rules contribute, and that is provable rather
than asserted:** a constant score makes the NCE inversion reduce *identically* to `q`, i.e.
the unigram model (`test_nce_correction_reduces_to_noise_prior_on_constant_scores`). So
473 -> 386 is precisely the context rules' contribution, with no double-counting of
frequency.

Generation is visibly better too — function words now appear at realistic rates
(`the` 0.165, `and` 0.118, `to` 0.094 as top alternatives) instead of the earlier
category-salad — though still not grammatical.

### E18.4 Efficiency of generation

A step scores the whole vocabulary. Under the product t-norm each rule's antecedent splits
into a context part and a candidate part, so firing factorises:

    firing(rule, w) = prod(ctx values) x prod(cand values for w)

The candidate half is precomputed once into an `(n_vocab, n_rules)` matrix and a step is
one scalar-vector scaling plus two reductions.
`test_generator_factorised_scoring_matches_direct` asserts it equals evaluating the rule
base directly.

## Standing summary (SUPERSEDED — see the final one)

**Beats a real baseline:** ranking MRR over unigram; perplexity 386 vs 473 unigram.
**Loses to:** a smoothed bigram (279). **Untested:** GPT-2 (blocked; and a data-gap
comparison anyway).

**Ruled out:** context width, rule order, Gaussian antecedents, symmetric decode
similarity, representation coarseness, Cython, raw score normalisation (E18.2).

**Confirmed binding:** corpus size (E16); within-category discrimination — supersense
rules cannot separate `bread` from `cake`, which caps both hits@1 and perplexity.

**Corrected:** all pre-E17 ranking numbers were measured with train/test sentence overlap.

---

## E19 — Closing the bigram gap: lexicalise the context, not the candidate

**The gap.** Fuzzy 385.8 vs bigram 279.2. Diagnosis: the fuzzy model is effectively a
**category bigram** -- it has `p(cat(w) | cat(prev))` where a real bigram has
`p(w | prev)`. Supersense rules cannot separate `bread` from `cake`.

**The idea.** English is Zipfian, so the head of the distribution is a few hundred words
that behave idiosyncratically and belong to no useful category (`the`, `of`, `said`).
**Lexicalise the head, generalise over the tail**: add identity dimensions for the top-K
most frequent words, keeping categories for everything else. Rules stay readable -- they
just gain terms like `ctx:prev1:=the`.

### E19.1 Identity on both sides — **FAILED, monotonically**

| config | perplexity |
|---|---|
| no lexeme features | **385.7** |
| top-50 identity, both sides | 394.6 |
| top-200 identity, both sides | 399.0 |

Worse, and consistently worse as K grows. Adding information hurt, which meant the
*interaction* with something else was wrong rather than the features themselves.

### E19.2 The mechanism, and the fix — **WORKED**

The NCE inversion (E18.3) computes `p ∝ q(w) · s/(1−s)`, where `q` is the frequency
prior. So a **candidate-side** rule like `cand:=the → high P(next)` re-learns frequency
that `q(w)` already supplies, and the two compound into an over-weighted head. The
candidate-side identity features were not adding information; they were **double-counting
the prior**.

**Context-side** identity is different in kind. `q(w)` says nothing about what follows
*the*, so `ctx:prev1:=the` is genuinely new -- it is exactly the bigram conditioning the
categories cannot express.

Prediction: masking identity on the candidate side while keeping it on the context side
should reverse the sign of the effect. It did.

| config | perplexity | vs no-lexeme |
|---|---|---|
| no lexeme features | 385.7 | — |
| top-50 identity, **candidate** side | 394.6 | **+8.9 worse** |
| top-50 identity, **context** side | **369.0** | **−16.7 better** |

Same features, same K, same rule budget -- only which half of the vector carries them.
The default is now `lexeme_side="ctx"`, and `must_include` still forces every rule to
touch a candidate feature, so a context-lexeme rule necessarily pairs with a candidate
*category*: `IF ctx:prev1:=the AND cand:noun.animal THEN ...`. Lexicalised where it pays,
categorical where it generalises.

**Progress against the gap.** 385.8 -> 369.0 closes ~15% of the distance to the bigram
(279.2), and now beats the trigram (370.2) as well as the unigram (472.9).

**Why this is the more interesting kind of result.** The failure was not "the feature is
useless" but "the feature is redundant with a transform applied downstream". That is only
findable by reasoning about *why* the first attempt failed rather than sweeping K harder --
sweeping would have shown a monotone worsening and suggested abandoning lexicalisation
entirely, which was the opposite of the right conclusion.

### E19.3 Context-side K keeps helping, and saturates

Run at the headline setting (1500 evaluation positions, 12000 training positions), and
replicated at a lighter one (600/8000) with baselines re-measured to match. **Compare
within a column** -- the two settings are not comparable to each other.

| config | ppl (1500/12000) | ppl (600/8000) |
|---|---|---|
| 1-gram | 472.9 | 433.3 |
| no lexeme features | 385.7 | — |
| 3-gram | 370.2 | — |
| top-50, context side | 369.0 | — |
| top-200, context side | 364.5 | 339.1 |
| **top-500, context side** | **363.4** | **338.5** |
| top-200, **both** sides | 399.0 | 365.1 |
| 2-gram | 279.2 | 236.9 |

Monotone improvement on the context side, flattening after ~200 identity dimensions
(364.5 -> 363.4 for 2.5x the K), consistent with the Zipfian premise: the top couple
hundred words are where idiosyncratic behaviour lives, and beyond that the categories do
the work. Both-sides is worse at every K in *both* settings, so E19.2's double-counting
mechanism holds throughout rather than being an artifact of one configuration.

Net **385.7 -> 363.4**: beats the unigram (472.9) and the trigram (370.2), still loses to
the bigram (279.2) by ~30%. The learner spends budget on lexicalised rules willingly --
**425 of 860** -- chosen on lift, not by quota.

### E19.4 Complementarity with the bigram — **CONFIRMED, replicated. The best result so far.**

The more informative test, positive in both settings. Interpolating
`p = lam * p_fuzzy + (1 - lam) * p_bigram` on identical positions:

| lambda | ppl (1500/12000) | ppl (600/8000) |
|---|---|---|
| 0.0 (pure 2-gram) | 276.1 | 236.9 |
| 0.20 | 264.6 | — |
| **0.30-0.35 (best)** | **263.1** | **228.1** |
| 0.50 | 266.8 | 233.6 |
| 0.70 | 281.3 | 249.4 |
| 1.0 (pure fuzzy) | 364.0 | 338.5 |

**The mixture beats the bigram alone in both settings -- 263.1 vs 276.1 (-4.7%) and 228.1
vs 236.9 (-3.7%) -- even though the fuzzy model loses head-to-head by ~30%.** An interior
optimum near lambda = 0.3 in two independent settings is not a fluke.

(The lambda=0 cell reads 276.1 where the standalone bigram measured 279.2: same model, ~1%
jitter from independent position sampling inside `interpolated_perplexity`. That is exactly
why lambda=0 is *measured* rather than assumed -- the within-column comparison is the valid
one.)

**Why this is the result that matters most.** Every previous number said the fuzzy model was
*worse than* a conventional baseline, which leaves open the reading that the categorical
representation is just a weaker way of doing what n-grams already do. This says otherwise:
the fuzzy features carry information a bigram **does not have**, and the two are additive.
Losing head-to-head while improving the ensemble is the signature of complementary
information rather than a strictly-dominated model.

It also changes the honest framing of the whole line of work. The claim is not "fuzzy rules
can replace an n-gram LM at this scale" -- they cannot, by ~30%. It is "fuzzy rules capture
something n-grams miss, in a form a human can read", and the interior optimum quantifies how
much: about a third of a mixture's weight is worth giving to the fuzzy model.

### E19.5 The lexicalised rules

Exactly the intended hybrid -- one specific frequent word, one general category:

```
IF ctx:prev1:=the    AND cand:OPEN_NOUN     THEN P(next) ~ 0.313   (support=1730.9)
IF ctx:prev1:=did    AND cand:NEGATOR       THEN P(next) ~ 1.000   (support=28.0)
IF ctx:prev1:=the    AND cand:noun.animal   THEN P(next) ~ 0.415   (support=224.0)
IF ctx:prev1:=said   AND cand:DETERMINER    THEN P(next) ~ 0.554   (support=101.0)
IF ctx:prev1:=his    AND cand:OPEN_NOUN     THEN P(next) ~ 0.341   (support=295.1)
IF ctx:prev1:=little AND cand:OPEN_NOUN     THEN P(next) ~ 0.340   (support=266.7)
IF ctx:prev1:=of     AND cand:DETERMINER    THEN P(next) ~ 0.325   (support=271.0)
IF ctx:prev1:=came   AND cand:INFINITIVE_TO THEN P(next) ~ 1.000   (support=11.0)
IF ctx:prev1:=buster AND cand:verb.body     THEN P(next) ~ 0.927   (support=11.2)
```

"the <noun>", "did not", "said the", "his <noun>", "little <noun>", "of the", "came to" --
English collocations recovered from data, at consequents up to 1.000. The last is
corpus-specific (Buster Bear is a character in one of the source stories), a useful reminder
that part of what the rule base learns is about *this corpus* rather than English.

**This is probably where the complementarity in E19.4 comes from.** A bigram encodes
"did -> not" too, but only as a table entry. Here it is a word paired with a *category*, so
it generalises to negators the bigram never saw after "did" -- information a bigram
structurally cannot hold.

---

## Standing summary (SUPERSEDED — see the final one)

**Best results.** As a language model: perplexity **363.4**, beating unigram (472.9) and
trigram (370.2), losing to bigram (279.2). **Mixed with a bigram it beats the bigram alone**
(263.1 vs 276.1), replicated at a second setting. Ranking (leaky pre-E17 numbers aside):
beats a unigram baseline on MRR. Representation: coverage 96.7%, exact multi-resolution
rollup asserted in CI, L2 similarity gap +0.278, explainable typo robustness.

**Honest framing.** Fuzzy rules do not replace an n-gram LM at this scale. They capture
something n-grams miss, in a readable form, worth ~a third of a mixture's weight.

**Ruled out:** context width (E12), rule order for ranking (E12, E15), Gaussian antecedents
(E9/E10), symmetric decode similarity (E11), representation coarseness (E13.1), Cython
(E14 -- BLAS-bound), raw score normalisation (E18.2), candidate-side lexicalisation
(E19.1/E19.2 -- double-counts the NCE prior).

**Confirmed binding:** corpus size (E16, replicated on Brown); within-category
discrimination, partly relieved by context lexicalisation (E19).

**Corrected along the way:** the 0.569 balanced accuracy (E12.1, bug-inflated); all pre-E17
ranking numbers (train/test sentence overlap).

**Open:** generation is still not grammatical; hits@1 caps near 0.14; GPT-2 untested
(blocked, and a data-gap comparison anyway).

---

## E20 — Why n-grams work, and a correction to my own trigram claim

**Why the 1-gram works.** Unigram entropy is 6.047 nats, so frequency alone collapses the
effective vocabulary from 3000 to **423**. The reason is how extreme the Zipf tail is:

| coverage of token mass | types needed | % of vocabulary |
|---|---|---|
| 50% | **54** | 1.8% |
| 80% | 415 | 13.8% |
| 90% | 943 | 31.4% |

**54 words carry half of all tokens.** This retroactively explains E19: lexicalising the
top ~200 words covers ~72% of token mass, and the measured saturation of `lexeme_top_k`
around 200 is exactly where the Zipf head stops paying. The design premise was right for a
quantifiable reason, not just a plausible one.

**Why higher orders should work — the information is there.** Train-side conditional
entropy (in-sample, so optimistic, but it bounds what is *available*):

| context | H(next \| context) | implied ppl |
|---|---|---|
| none | 6.047 nats | 422.9 |
| 1 word | 3.504 | 33.3 |
| 2 words | 1.222 | **3.4** |

Longer context carries enormous information. The training data is nearly deterministic given
two words.

**Why they nevertheless fail here — context sparsity.** The information is available but not
*estimable* at 67K training tokens:

| order | test contexts seen in train | seen >= 5 times |
|---|---|---|
| 1 (bigram context) | 99.3% | 92.3% |
| 2 (trigram context) | 66.6% | 39.5% |
| 3 (4-gram context) | 25.4% | **6.2%** |

For a trigram, three quarters of test positions have a context never seen in training, and
only 6% have one seen often enough to estimate. The gap between "ppl 3.4 available" and
"ppl ~300 achievable" is pure estimation error. **This is the same finding as E16 (corpus
size binds), arrived at from the baseline side.**

### E20.1 CORRECTION: my 3-gram baseline was mis-weighted, and I over-claimed against it

`NgramLM` used fixed Jelinek-Mercer weights proportional to order, so an order-3 model put
**half its mixture weight on the trigram term** -- the term whose context is unseen 75% of
the time. That is why my trigram scored *worse* than my bigram, which is not how trigrams
normally behave and which I should have questioned earlier instead of reporting it as a
baseline.

| model | perplexity |
|---|---|
| 2-gram, fixed weights | 278.7 |
| **3-gram, weights favouring low order** | **297.5** |
| 3-gram, balanced | 298.9 |
| 3-gram, bigram-dominant | 306.9 |
| 3-gram, fixed weights (as previously reported) | 370.3 |

**The consequence for my claims.** I repeatedly wrote that the fuzzy model at 363.4 "beats
the trigram (370.2)". That comparison was against a badly-weighted trigram. A
properly-weighted trigram reaches **297.5**, so **the fuzzy model does not beat a competent
trigram** -- it loses to it, as it loses to the bigram.

Corrected standing: the fuzzy LM **beats the unigram (472.9)** and **loses to both the
bigram (278.7) and a well-weighted trigram (297.5)**.

**Unaffected:** the complementarity result (E19.4) interpolated with the *bigram*, never the
trigram, so mixing still beats the bigram alone (263.1 vs 276.1). That remains the strongest
result in the project.

**Why I missed it.** The README itself flagged `NgramLM` as "a credible reference, not a
competitive n-gram implementation" -- I wrote that caveat and then quoted its number as a
beaten baseline anyway. A baseline being deliberately simple is a reason to distrust
*favourable* comparisons against it, which is the opposite of how I used it.

Even well-weighted, no trigram beats the bigram here, so the ordering bigram < trigram is
itself a real finding about 67K tokens rather than an artifact -- but the *magnitude* was.

---

## E21 — A fuzzy tokenizer and a linguistic parameter space (`flm/fuzzytok/`)

**Why.** Two joints in `fuzzyembed/` were the weakest links, and E20 quantified how to fix
one of them. The word-level vocabulary has no subword generalisation and cannot see proper
names; the WordNet feature space has an unbalanced ladder, no morphology, and no names.

### E21.1 Vocabulary sizing came from the measurement, not a guess

E20 found **54 types carry 50% of token mass, 415 carry 80%**. So the head is tiny and
idiosyncratic (no decomposition helps `the`, `of`, `said`) while the tail is large and
morphologically regular -- which argues for a **hybrid** vocabulary rather than pure BPE or
pure whole-word. Measured coverage by best-reading kind:

| head size | nameable units | whole-word | decomposed |
|---|---|---|---|
| 100 | 147 | 57.0% | 43.0% |
| 200 | 247 | 67.1% | 32.9% |
| **500** | **547** | **79.5%** | 20.5% |
| 1000 | 1047 | 87.6% | 12.4% |

head=500 giving 79.5% whole-word matches the independent Zipf prediction (415 types = 80%)
almost exactly, which is a useful consistency check on both measurements. **547 nameable
units** total -- genuinely a "simple vocabulary", and every unit is something a rule can say
(`un-`, `-ly`, `rabbit`).

### E21.2 The fuzzy part, and how it differs from prior art

BPE (Sennrich et al. 2016) merges greedily; WordPiece (Schuster & Nakajima 2012) maximises
likelihood; the unigram-LM tokenizer (Kudo 2018) *can* enumerate segmentations with
probabilities and subword regularisation **samples** one per training step. This keeps all of
them at once with membership degrees and never samples:

```
'quickly'   quick + -ly   [0.90, stem in vocabulary]
            quickl + -y   [0.35, stem shape only]
'unkindly'  unkind + -ly  [0.90]   un- + kindly [0.90]   un- + kind + -ly [0.90]
'littel'    little        [0.74, fuzzy lexical access]
            listen        [0.44, fuzzy lexical access]
'xqzzyv'    ^xq + xqz + qzz + zzy + zyv + yv$  [0.30, character n-grams]
```

Two things follow that a hard tokenizer cannot give: misspelling robustness is *intrinsic*
(the graded lexical-access layer now runs inside tokenization rather than after it), and the
ambiguity is **reportable** -- `unkindly` is genuinely three-ways ambiguous and the
tokenizer says so instead of picking one.

Versus tokenizer-free byte/character models (CANINE, ByT5), which also avoid segmentation
commitment: this keeps units *nameable*, which a byte model cannot.

### E21.3 Two real bugs, both found by running it rather than by tests

The unit tests passed while both bugs were live, because the test vocabularies happened to
avoid them. Running on the real corpus exposed both.

**Premature commitment on stem variants.** `hoping` resolved *only* to `hop + -ing`, at 0.90,
with `hope + -ing` never emitted. The affix code short-circuited: if the bare stem was in the
vocabulary it stopped, and "hop" is a word. So the one case the dropped-`e` rule existed for
was silently unreachable whenever the truncated stem was also real. Both variants are now
emitted as competing graded readings. **This is exactly the failure mode a fuzzy tokenizer is
supposed to prevent**, which made it worth a regression test rather than a quiet fix.

**Case folding.** `Margery` missed the head set and fell through to fuzzy lexical access,
matching *itself* at 0.73 and competing with `larger` at 0.47. Lookup is now case-folded while
the surface form is preserved for the shape features -- orthographic case is a *parameter*,
not a reason to fail lookup.

### E21.4 The parameter space

66 named dimensions in four blocks: UD's 17 UPOS tags and ~20 FEATS (de Marneffe et al.
2021), 12 orthographic shape features, 14 coarse semantic groups collapsed from WordNet's 45
supersenses, and Osgood's 3 affective axes (Osgood et al. 1957).

```
'the':      DET=1.00, Shape=Short=1.00, Affect=Evaluation=0.50
'not':      PART=1.00, Polarity=Neg=1.00
'his':      PRON=1.00, Poss=Yes=1.00
'Margery':  Shape=Capitalised=1.00, PROPN=0.60
'walking':  Shape=SuffixIng=1.00, Sem=Act=1.00, VERB=0.83, Aspect=Prog=0.70
'rabbit':   Sem=Entity=1.00, NOUN=0.75, Sem=Animate=0.33
'happy':    ADJ=1.00, Sem=Quality=1.00, Affect=Evaluation=1.00
```

Three deliberate choices:

* **UD instead of WordNet** for the syntactic backbone, because it is a *designed, balanced*
  inventory -- the thing WordNet's hypernym DAG demonstrably is not (E4.2: 45 -> 4527).
* **Capitalisation at degree 0.6, not 1.0.** It is the cheapest proper-noun signal and names
  were the dominant coverage gap (E1), but sentence-initial words are capitalised too and this
  encoder has no sentence position. Overclaiming here would manufacture proper nouns.
* **Osgood's activity axis is left at 0** rather than inferred. It needs elicited ratings or a
  norms lexicon; deriving it from orthography would be invention dressed as a feature. Only
  evaluation is populated (from the opinion lexicon), which is the axis sentiment needs.

Misspelling robustness survives the whole path: parameter-vector overlap 0.944 for
`happy`/`hapy` and 0.975 for `rabbit`/`rabit`.

### E21.5 What this is NOT yet

**Not wired into the joint ranker, so there is no perplexity number for it.** Everything above
is component-level validation. The comparison that matters -- 66 named linguistic parameters
versus the current 61-dimensional semantic+syntax space, at matched rule budget and on the
same split -- has not been run. Until it is, the honest claim is only that the encoder
produces sensible graded features on inspection, not that it improves the language model.

54 tests pass (17 new for this module, including regression tests for both bugs above).

---

## E22 — The parameter space wired into the ranker — **FAILED (strictly dominated)**

E21.5 said the honest claim was only that the encoder produces sensible features on
inspection, and that the perplexity comparison had not been run. It has now been run.

### E22.1 The A/B: worse alone, at every budget

Same split, same rule ceiling (2500), same held-out positions, same candidate vocabulary.
Wiring cost nothing structural: `JointNextTokenRanker` and `FuzzyGenerator` need only
`_token_vector`, `_output_names`, and an optional `lexemes` list, so `ParameterFeaturiser`
swaps the representation without the ranker changing at all.

| representation | dims | ppl | rules | fit |
|---|---|---|---|---|
| WordNet sem+syntax (baseline) | 261 | **324.8** | 860 | 68s |
| linguistic parameters | 267 | 360.0 | 841 | **14s** |
| linguistic params, fuzzy tokenizer OFF | 267 | 354.9 | 841 | 10s |
| combined (WN + params) | 328 | 364.2 | 901 | 64s |
| 2-gram, same data, tuned | — | 256.3 | — | — |

Replicated at a larger evaluation budget with identical ordering (365.0 / 397.0 / 394.3 /
410.4), so the ranking is not an artefact of one eval sample.

**Not a budget artefact.** Both spaces saturate at ~850 rules out of the same 2500 ceiling,
so the parameter space is not simply being starved of rules relative to its dimensionality.

**The fuzzy tokenizer made it slightly worse** (360.0 with graded readings vs 354.9 encoding
the surface form directly). The multi-reading machinery costs accuracy here rather than
paying for itself. Two readings of `hoping` mean the parameter vector is a blend of two
lemmas' features, and blending is exactly what a next-token decision does not want when one
reading is right.

### E22.2 Complementarity: none, in any combination

This is the test that mattered, because E19.4 established that losing head-to-head is a
different question from carrying no information -- the fuzzy model lost to a bigram and
still improved a mixture with it. So every pair was swept on **identical positions**.

```
alone:   wn 325.0    lp 346.9    2g 256.3
wn/2g:   BEST lam=0.2  ppl=253.9      <- beats the bigram alone; replicates E19.4 in sign
lp/2g:   BEST lam=0.0  ppl=256.3      <- adds nothing
lp/wn:   BEST lam=0.0  ppl=325.0      <- adds nothing
lp on top of wn+2g:   0.0=254.6  0.1=257.7  0.2=261.4  0.3=265.7   (monotonically worse)
```

**Optimal weight is zero, three times out of three.** That is a much stronger negative than
"loses head-to-head": the space is *strictly dominated*, carrying no information the
WordNet space or the bigram does not already have.

**Why, most likely.** The parameter space collapses WordNet's 45 supersenses into 14 coarse
semantic groups. It buys morphology, orthographic shape, and proper nouns, and pays in
semantic resolution -- and on this task the semantic block is where the win lives. The
purchase was real; the price was higher than the goods. Note this does *not* contradict
E13.1 (representation coarseness was not the blocker): E13.1 tested whether a *finer*
representation helps, this tests whether a *coarser* one hurts, and the answer to both can
be yes without inconsistency once discrimination is already near its ceiling.

**A bug in my own sweep, worth recording.** The two generators restrict to different
decodable vocabularies (2897 vs 3000 -- a word the WordNet space cannot represent is not
necessarily one the parameter space cannot), so their distributions are over different
column sets. Mixing them elementwise raised on the shape mismatch, which was lucky: had the
widths matched by coincidence it would have silently compared different words and produced
a plausible wrong number. Distributions are now projected onto the shared vocabulary and
renormalised explicitly.

### E22.3 The size/quality frontier — the knee is very sharp

Motivated by the requirement that this train on local compute without servers. Learned
parameters counted honestly: one consequent per rule plus its antecedent indices. The
membership degrees are *not* learned -- that is precisely why the model is small (E10:
inputs are already memberships, so no membership function is fitted).

| space | budget | rules | learned params | ppl | fit | ms/tok |
|---|---|---|---|---|---|---|
| params (lp) | 100 | 100 | 285 | 375.3 | 10.5s | 0.82 |
| params (lp) | 250 | 250 | 712 | 365.4 | 6.8s | 0.98 |
| params (lp) | 1000 | 841 | 2482 | 360.0 | 6.9s | 3.56 |
| wordnet (wn) | 100 | 100 | **285** | **341.2** | 65.0s | 1.45 |
| wordnet (wn) | 500 | 500 | 1440 | 329.8 | 58.5s | 3.58 |
| wordnet (wn) | 2500 | 860 | 2520 | 324.8 | 61.7s | 4.64 |
| 1-gram | — | — | 2,966 stored counts | 448.4 | — | — |
| 2-gram | — | — | 34,021 stored counts | 264.2 | — | — |
| 3-gram | — | — | 108,511 stored counts | 269.6 | — | — |

**285 learned parameters reach 341.2, within 5% of the 860-rule model.** An 8.8x increase in
parameters (285 -> 2520) buys 4.8% perplexity. So the rule base can be shrunk hard almost
for free, and against the n-gram column that is the compression result: 285 parameters land
within 29% of a bigram needing **34,021 stored counts**, two orders of magnitude more.
Fuzzy rules generalise across words where an n-gram stores each context separately.

**Shrinking the rule base does NOT shrink training time** (65.0s at 100 rules, 61.7s at
2500). Fit cost is dominated by featurisation and the candidate-growth GEMM over the full
seed pool, not by how many rules survive. The lever for training time is the seed pool and
negatives per position, not `max_rules`. Inference *does* scale with rule count (1.45 vs
4.64 ms/token), so the small end is the right place to be for serving.

### E22.4 Complementarity survives shrinking, but the effect is small — and a correction

| rules | params | alone | bigram | best mix | lam | gain over bigram |
|---|---|---|---|---|---|---|
| 100 | 285 | 341.2 | 256.3 | 255.3 | 0.1 | **0.39%** |
| 250 | 712 | 335.2 | 256.3 | 255.2 | 0.1 | 0.43% |
| 860 | 2520 | 324.8 | 256.3 | 253.9 | 0.2 | 0.94% |

285 parameters do carry information a 34,021-count bigram lacks: the sign replicates at
every size and the optimal weight stays positive.

**Correction to an expectation E19.4 set.** That entry measured the mixture gain as ~4.7%
(263.1 vs 276.1). Here, at the same rule count, it is 0.94%. The difference is the fuzzy
side's training budget (5000 positions here). So the complementarity result replicates in
*sign* at every size but its *effect size* is budget-dependent and is well under one percent
at this setting -- E19.4's number should not be quoted as what to expect at a small budget.
This also reinforces E16: data, not model size, is the binding constraint.

### E22.5 What survives from E21

The negative result is about the *feature space*, not everything in `fuzzytok/`. Three
things stand on their own and are worth keeping:

* **The rules are more readable.** `IF ctx:prev1:AUX AND cand:Polarity=Neg` and
  `IF ctx:prev1:DET AND cand:Sem=Entity THEN P(next) ~ 0.262 (support=892, lift=+4.49)` are
  grammar, stated as grammar. The WordNet space cannot say `Polarity=Neg` at all.
* **It fits 9x faster** (6.8s vs 61.7s at equal rule count), because the seed pool has far
  fewer live features. Real, but speed is not the binding cost at this scale.
* **The hybrid vocabulary sizing and the tokenizer's graded lexical access** are independent
  of the parameter space and were validated in E21.1--E21.3.

The honest verdict: **the linguistic parameter space is not the way to close the bigram gap.**
It is a better *reporting* language and a worse *predictive* one. If it earns a place later
it will be as a readability layer over a representation that discriminates, or on a task
where morphology and proper nouns matter more than lexical semantics -- not as a replacement
for the WordNet block on next-token prediction.

61 tests pass (7 new, pinning the two interface constraints that would silently corrupt the
ranker: identity dims must stay last for the candidate mask, and `""` must map to an
explicit `BOUNDARY` dimension rather than an all-zero vector).

---

## Standing summary (SUPERSEDED — see the final one)

**Best results.** As a language model: perplexity **324.8** at 860 rules / 2520 learned
parameters. It beats a unigram (448.4) and loses head-to-head to both a bigram (256.3) and
a tuned trigram (269.6) on this data. **Mixing with a bigram beats the bigram alone**
(253.9 vs 256.3, gain 0.94%), replicated in sign down to 285 parameters (0.39%).
Representation: coverage 96.7%, exact multi-resolution rollup asserted in CI, L2 similarity
gap +0.278, explainable typo robustness.

**Smallness.** 285 learned parameters get within 5% of the full model, against 34,021 stored
counts for a bigram. The model is small structurally, because memberships are given rather
than fitted (E10). Training is seconds-to-a-minute on one CPU core, no GPU. Rule count
drives inference cost, not training cost.

**Honest framing.** Fuzzy rules do not replace an n-gram LM at this scale. They capture
something n-grams miss, in a readable form, worth a small but positive share of a mixture --
under 1% at the current training budget, more at larger budgets (E19.4).

**Ruled out:** context width (E12), rule order for ranking (E12, E15), Gaussian antecedents
(E9/E10), symmetric decode similarity (E11), representation coarseness as the blocker
(E13.1), Cython (E14 -- BLAS-bound), raw score normalisation (E18.2), candidate-side
lexicalisation (E19.1/E19.2), **the linguistic parameter space as a replacement feature
space (E22 -- strictly dominated), and the fuzzy tokenizer's graded readings as an accuracy
win (E22.1)**.

**Confirmed binding:** corpus size (E16, replicated on Brown; reinforced by E22.4);
within-category discrimination, partly relieved by context lexicalisation (E19).

**Corrected along the way:** the 0.569 balanced accuracy (E12.1, bug-inflated); all pre-E17
ranking numbers (train/test sentence overlap); the trigram baseline (E20.2, mis-weighted);
E19.4's mixture gain quoted without its budget dependence (E22.4).

**Open:** generation is still not grammatical; hits@1 caps near 0.14; GPT-2 untested
(blocked, and a data-gap comparison anyway); no neural-embedding comparison; Experiment B's
real encoder and SST paths unrun.

---

## E23 — Corpus scale and parallel training — **the profile overturned the premise**

Two requests: push corpus size (the constraint E16 and E22.4 both identified as binding, and
the only one never actually pushed), and evaluate multi-threaded training, motivated by
wanting a model trainable on local compute without servers.

### E23.1 Profiling first — and E22.3's attribution was wrong

E22.3 observed fit time flat in `max_rules` (65s at 100 rules, 62s at 2500) and attributed it
to "featurisation plus the candidate-growth GEMM". That was inference from a timing pattern,
not a measurement. Measured:

| stage | time | share |
|---|---|---|
| `build()` | 58.1s | **88%** |
| rule search | 7.8s | 12% |

Then `cProfile` on `build()` reported **0.2s**, because it was the *second* call and
`_token_vector`'s cache was warm. 2.5x fewer positions cannot explain 58s -> 0.2s, so the cost
is not per position at all: it is **per token type**, 19.6 ms each over 3,000 types. That also
resolves E22.3's puzzle -- fit time is flat in `max_rules` because the dominant cost is a
one-time per-type featurisation that has nothing to do with rules.

**Lesson worth keeping: "parallelise the training" would have optimised a cost that was 12% of
the total.** Profiling before parallelising changed the entire answer.

### E23.2 The actual hot spot: 8.7 million `project` calls for 400 token types

Inside cold featurisation, `hierarchy.rollup` and `hierarchy.project` accounted for 13s of
16.3s. `rollup` walked every key at the source level and called `project` per key, per call --
a pure-Python loop over up to 9,864 nodes, repeated for every level of every token. 21,700
`project` calls **per token type**.

The parent of each coordinate is structural, fixed by the hierarchy, so the mapping is
computable once. Sorting source indices by destination makes each destination's sources
contiguous, and the aggregation becomes a single `ufunc.reduceat`:

**cold featurisation of 3,000 types: 58.84s -> 1.36s, a 43x speedup.**

`build()` got the same treatment -- featurise each type once into a context table and a
candidate table, then assemble rows by fancy indexing, with the Python loop reduced to
collecting integer ids in the same RNG draw order. 1.31x warm, and bit-identical.

Both optimisations touch load-bearing claims, so neither is trusted on inspection. The
original implementations are retained as oracles: `_rollup_reference` compared across every
level pair, every op, and random vectors (exact multi-resolution readout is *the* claim
separating this from Matryoshka truncation), and `_build_reference` asserted **bit-identical**
(a changed negative sample would silently alter the training set and every number downstream,
in a way no perplexity comparison would reveal as a bug).

### E23.3 Parallelism, measured — threads lose, processes win, vectorising wins by 10x

| approach | featurisation speedup |
|---|---|
| 2 threads over token types | 0.88x |
| 4 threads over token types | **0.47x** (actively worse) |
| 2 processes (fork, COW-inherited embedder) | 1.47x |
| 4 processes (fork) | 1.68x post-fix / **3.85x pre-fix**, bit-exact |
| **vectorising the rollup** | **43x** |

* **Threads fail because the work is pure-Python WordNet traversal.** The GIL serialises it and
  contention makes 4 threads worse than 1. This is not a tuning problem; it is the wrong tool.
* **Processes work.** `fork` lets children inherit the built embedder through copy-on-write, so
  there is no pickling cost inbound, and the result is bit-exact against serial. 3.85x on 4
  cores measured *before* the rollup fix, i.e. near-linear.
* **The GEMMs were already threaded** and it is worth nothing at these sizes: rule search takes
  5.15s on 1 BLAS thread and 5.05s on 4. E14's "BLAS-bound, not a Cython candidate" conclusion
  stands, but the corollary -- that BLAS threading is doing useful work -- does not.
* **The algorithmic fix beat the best parallelism by 11x**, and afterwards featurisation is 4%
  of fit, so parallelising it is no longer worth doing at all. 4 cores cap any parallel
  speedup at 4x; a bad inner loop has no such ceiling.

New breakdown after both fixes (total 65.9s -> 31.9s): featurise 1.26s (4%), `build()` 22.5s
(70%, and see E23.5 -- almost all of it OOV lexical access), rule search 8.2s (26%).

### E23.4 Corpus scale — **it works, for the bigram, not for the fuzzy model**

New `NARRATIVE` corpus: all narrative prose in the nltk Gutenberg sample, **1,014,404 tokens**
against the old 86,559 (11.7x), 27,044 types. Verse, the KJV, and the Shakespeare plays are
excluded deliberately -- they would grow the token count fastest and confound the measurement
worst, since a perplexity change could then be "different language" rather than "more data".

Design: nested subsets of the training side, all evaluated on the **same held-out test set
with the same 3,000-word candidate vocabulary**. Comparing "small corpus vs big corpus"
directly would be invalid -- different test sets, so a difference could be "easier text".
Counts come from each subset, since the frequency prior q(w) is part of the model and handing
a small-data condition the full corpus's unigram statistics would leak.

| train tokens | fuzzy ppl | 2-gram | best mix | lambda |
|---|---|---|---|---|
| 64,575 | 413.1 | 389.3 | 374.4 | **0.4** |
| 180,120 | 383.4 | 315.4 | 314.1 | 0.2 |
| 404,909 | 368.5 | 288.5 | 288.5 | **0.0** |
| 811,254 | 322.1 | 253.3 | 253.3 | **0.0** |

(20,000 training positions; the 5,000-position rows show the same pattern.)

**The fuzzy model does improve with data** -- 413.1 -> 322.1, a 22% reduction. **But the bigram
improves faster** (389.3 -> 253.3, 35%), and **the complementarity result dies**: the optimal
mixture weight on the fuzzy model falls 0.4 -> 0.2 -> 0 -> 0.

This is a direct negative against my own recommendation. E16 concluded corpus size was the
binding constraint and the biggest available lever, and E22.4 reinforced it; on that basis I
recommended pushing corpus size as the promising direction. Pushed properly, it makes the
fuzzy model's *relative* contribution worse, not better. **The mixture win -- the strongest
result in the project (E19.4) -- is a small-data phenomenon.** Its lambda was already visibly
budget-dependent in E22.4 (4.7% at a large budget, 0.94% at a small one); extending the data
axis shows it going to zero.

Also worth noting: more training *pairs* barely helps at high data (332.1 -> 322.1 for 5,000
-> 20,000 positions at frac 1.0), so the fuzzy model is not merely starved of samples. Both
knobs are pushed and it still loses ground.

### E23.5 A mechanism, and it is not flattering to the featuriser

Out-of-vocabulary types cost **12.17 ms/type against 0.53 in-vocab** -- 23x -- and the profile
is unambiguous: 355,171 `_dl_simple` (Damerau-Levenshtein) calls for 400 types, i.e. the
BK-tree spelling search, ~890 edit-distance computations per word.

The bigger problem is what it *returns*. OOV tokens do not get a sense; they get a **blend of
several fuzzy-matched neighbours**:

```
'wooden'    noun.substance=0.57, noun.group=0.56, noun.person=0.51, adj.all=0.49,
            verb.cognition=0.46          <- four unrelated supersenses, all ~0.5
'arriving'  adj.all=0.55, noun.act=0.11, verb.communication=0.11, ...
'doorway'   (empty)
```

On the 1M-token corpus, **24,044 of 27,044 types are outside the 3,000-word sense vocabulary**,
so the larger the corpus, the larger the fraction of context that is diffuse noise rather than
named memberships. That is a coherent mechanism for E23.4: the bigram sees more *distinct
words* as data grows, while the fuzzy featuriser sees more *blur*.

It is also a correctness observation, not just a performance one. Running spelling correction
over the rare tail of a clean corpus is semantically wrong: `wooden` is not a misspelling of
anything, and answering as though it were injects noise the model then has to fit.

**And it is now the dominant training cost, which the E23.4 table understates.** That table
reports fit times of 60-70s on the 1M-token corpus, but `scale.py` builds one embedder and
reuses it across all eight cells, so every cell after the first runs with the embedder's caches
already warm. Building cold, the same configuration takes **324.9s**, of which 24,044 OOV types
at 12.17 ms each accounts for ~293s. So the honest cold-start cost on a 1M-token corpus is ~5.4
minutes, and ~90% of it is spelling-correcting words that are not misspelled. Fixing the OOV
path would cut training by roughly an order of magnitude *and* remove the feature noise --
the same change serves both the local-compute goal and the quality goal, which makes it the
clear next thing to do.

### E23.6 What this changes

**Confirmed:** the vectorisations (43x, and both proven equivalent), and that corpus size does
improve the fuzzy model in absolute terms.

**Refuted:** that corpus size is the lever that closes the gap (E16's framing, and my own
recommendation last turn). It closes the gap the wrong way.

**Refuted:** that training needs parallelising. It needed profiling. Threads are the wrong
tool (GIL), processes work but now address 4% of fit, and BLAS threading buys nothing here.

**Open, and now the leading candidate:** whether the ceiling is featuriser capacity (E23.5's
OOV blending) rather than the method. Testable by pinning the candidate set and raising only
the sense vocabulary and context lexicalisation -- `experiments/capacity.py`.

64 tests pass (2 new equivalence oracles for the vectorised paths).

---

## E24 — The OOV fix: **correct and 5.1x faster, but it did not improve the model**

Two changes, both from E23.5: sense and lexicon coverage over the whole vocabulary instead of
a truncated prefix, and relative pruning of out-of-vocabulary lexeme matches so a weak tail is
not merged into a reading.

### E24.1 Both stated goals achieved

**Resolution is now correct.** The same words that E23.5 showed as blends:

| token | before (E23.5) | after |
|---|---|---|
| `wooden` | noun.substance 0.57, noun.group 0.56, noun.person 0.51, verb.cognition 0.46 | **adj.all 1.00, OPEN_ADJ 1.00** |
| `arriving` | adj.all 0.55, noun.act 0.11, verb.communication 0.11 | **verb.motion 1.00, OPEN_VERB 1.00**, verb.social 0.50 |
| `doorway` | (empty) | **noun.artifact 1.00, OPEN_NOUN 1.00** |
| `harpooneer` | — | **noun.person 1.00, OPEN_NOUN 1.00** |
| `wodden` (real typo) | — | adj.all 0.69 — still graded, so robustness survives |

That last row is the one that matters for not having over-corrected: a genuine misspelling
still gets partial membership through fuzzy lexical access. The fix distinguishes "rare word"
from "misspelled word" instead of treating every unknown string as a typo.

**Cold fit: 324.9s -> 63.1s (5.1x)** on the 1M-token corpus, and it is nearly free --
full-vocabulary hierarchy build is 3.7s vs 5.6s for 3,000 types, and level 2 stays at exactly
45 dimensions (widths `[1, 4, 45, 12630, ...]` vs `[1, 4, 45, 4983, ...]`), so no
dimensionality or comparability changes. Combined with E23's 43x, cold training on 1M tokens
went from ~5.4 minutes to ~1 minute on one core.

### E24.2 And it did **not** improve perplexity — the capacity hypothesis is refuted

| | before | after |
|---|---|---|
| fuzzy ppl | 320.6 | **317.5** (1.0%) |
| bigram | 253.3 | 253.3 |
| best mixture | 253.3 @ lambda=0.0 | 253.3 @ **lambda=0.0** |

E23.5's hypothesis was that the ceiling was featuriser capacity: 24,044 of 27,044 types
getting blended noise, growing with corpus size. The blending was real, it was expensive, and
it was semantically wrong -- and fixing it bought **1%**, with the mixture weight still zero.

So the ceiling is not OOV representation quality. **Two of my hypotheses have now been
refuted in a row** (corpus scale in E23.4, featuriser capacity here), which is worth stating
plainly rather than moving quietly to a third.

### E24.3 Why generation fails, read directly off the rules

This is the first place where the interpretability claim actually paid a diagnostic dividend.
Sample output (14 tokens, prompt in bold):

```
the little | in with the time and it not of to the all his of she
she was    | not and the as and of to and you will his but you to
the old    | of his man the said of he would no said he was no the
```

Against the bigram on identical data:

```
the little | of he kept going to one who he woman of all an about time
she was    | true hull so young gentleman she the skeleton they him nay together like sun
the old    | mind jew theresa seated and him and he you situation to had mr tied
```

The fuzzy output is **function-word soup**; the bigram's is visibly more word-like. The rule
trace says exactly why:

```
IF ctx:prev1:PREPOSITION  AND cand:DETERMINER   THEN P ~ 0.298  (support=2956, lift=+10.09)
IF ctx:prev2:=the         AND cand:PREPOSITION  THEN P ~ 0.203  (support=1554, lift=+3.59)
IF cand:CONJUNCTION                             THEN P ~ 0.112  (support=17250)
```

**Every high-lift rule is about closed-class categories.** The model has genuinely learned
English function-word syntax -- "after a preposition, expect a determiner" is correct and
carries lift +10 -- and it has essentially nothing that *selects a content word*. So it
generates the things it can score, which are function words. The top alternatives at each
step confirm it: `the(0.829), and(0.083), his(0.022), that(0.012)`.

The cause is support. Closed-class categories are the highest-support features available
(`cand:CONJUNCTION` alone has support 17,250), so a greedy, support-gated rule search spends
its budget where the evidence is densest, and function words are where the evidence is
densest by an order of magnitude. Content-word selection needs discrimination *within*
`OPEN_NOUN`, which 45 supersenses at word granularity do not provide.

Also: `hedge=3.0` (Zadeh concentration, `mu**3`) makes generation *worse*, not sharper --
`and the all of the and to the`. Concentration amplifies the already-dominant function words,
so it sharpens toward exactly the wrong thing.

### E24.4 What this points at

Not more data (E23.4), not OOV quality (E24.2). The rule base is **budget-misallocated**: it
is spending itself on the densest signal rather than the most useful one. Testable directions,
in order of how directly they attack that:

1. **Stratify the rule budget** so closed-class antecedents cannot consume it -- an explicit
   quota for rules whose candidate side is open-class, analogous to the existing `order_quota`.
2. **Reweight training positions** toward content-word targets, so support stops being
   dominated by function-word co-occurrence.
3. **Discriminate within open classes** -- the E22 verdict said the parameter space was a
   better reporting language and a worse predictive one, but *within* `OPEN_NOUN` is precisely
   where the WordNet space is also thin.

66 tests pass (2 new: OOV pruning keeps real ambiguity and drops the weak tail; full coverage
is the default).

---

## Standing summary (SUPERSEDED — see the final one)

**Best results.** On the 1M-token narrative corpus: perplexity **317.5** against a bigram's
253.3, with optimal mixture weight **zero**. On the old 87K-token children's corpus:
perplexity 324.8, and mixing with a bigram beat the bigram alone (253.9 vs 256.3). The
representation claims all hold: coverage 96.7%, exact multi-resolution rollup asserted in CI,
L2 similarity gap +0.278, explainable typo robustness, correct sense assignment for rare words.

**Smallness and speed.** 285 learned parameters reach within 5% of the full 2,520-parameter
model, against 34,021 stored counts for a bigram. Cold training on 1M tokens is ~1 minute on
one CPU core (from ~5.4 minutes), no GPU. Rule count drives inference cost, not training cost.

**Honest framing.** As a language model this loses to a bigram trained on the same text, and
at 1M tokens it contributes nothing to a mixture with one. What it does deliver is a *readable
diagnosis of its own failure* (E24.3), which no dense model of comparable quality offers.

**Ruled out:** context width (E12), rule order for ranking (E12, E15), Gaussian antecedents
(E9/E10), symmetric decode similarity (E11), Cython (E14 -- and BLAS threading is worth
nothing here either, E23.3), raw score normalisation (E18.2), candidate-side lexicalisation
(E19), the linguistic parameter space as a replacement feature space (E22), **corpus scale as
the lever that closes the gap (E23.4 -- it widens it), thread-based parallelism (E23.3 --
0.47x, GIL), and OOV representation quality as the ceiling (E24.2 -- worth 1%)**.

**Confirmed:** vectorising beat parallelising by 11x (E23.2); within-category discrimination is
the live constraint, now with a mechanism -- **the rule budget goes to the highest-support
features, which are closed-class** (E24.3).

**Corrected along the way:** the 0.569 balanced accuracy (E12.1); all pre-E17 ranking numbers;
the trigram baseline (E20.2); E19.4's mixture gain quoted without its budget dependence
(E22.4); E22.3's attribution of fit cost (E23.1); my own recommendation to push corpus size
(E23.4).

**Open:** generation is function-word soup, and E24.4 says why and what to try; hits@1 caps
near 0.14; GPT-2 untested (blocked, and a data-gap comparison anyway); no neural-embedding
comparison; Experiment B's real encoder and SST paths unrun.

---

## E25 — Open-class rule quota — **FAILED (no effect)**, and it exposed a wrong diagnosis

E24.4 proposed stratifying the rule budget so closed-class antecedents could not consume it.
Implemented as `reserved_features` / `reserved_quota` in `MembershipRuleRegressor`, wired to
`JointNextTokenRanker(open_class_quota=...)`, swept at 0.0 / 0.25 / 0.50 / 0.75.

### E25.1 The quota changed literally nothing

| quota | ppl | 2gram | mix | lambda | rules | open-class rules | content% |
|---|---|---|---|---|---|---|---|
| 0.00 | 317.5 | 253.3 | 253.3 | 0.0 | 860 | 386 | 7.1% |
| 0.25 | 317.5 | 253.3 | 253.3 | 0.0 | 860 | 386 | 7.1% |
| 0.50 | 317.5 | 253.3 | 253.3 | 0.0 | 860 | 386 | 7.1% |
| 0.75 | 317.5 | 253.3 | 253.3 | 0.0 | 860 | 386 | 7.1% |

Every figure identical. **The rule base saturates at 860 rules against a 2,500 budget**, so the
budget was never scarce and a quota had nothing to reallocate. This was already documented in
`joint.py` -- the beam comment records that "the candidate supply runs out at ~835 admissible
order-2 rules, not because the beam binds" -- and I proposed a budget fix anyway without
checking whether the budget bound.

**And the rules I claimed were missing were already there.** 386 of 860 rules are open-class,
with high lift:

```
IF ctx:prev1:DETERMINER    AND cand:OPEN_NOUN  THEN P ~ 0.265  (support=3014, lift=+8.411)
IF ctx:prev1:INFINITIVE_TO AND cand:OPEN_VERB  THEN P ~ 0.272  (support=865,  lift=+4.700)
IF ctx:prev1:=his          AND cand:noun.body  THEN P ~ 0.663  (support=65,   lift=+4.440)
IF ctx:prev1:OPEN_ADJ      AND cand:OPEN_NOUN  THEN P ~ 0.211  (support=2672, lift=+5.098)
```

So **E24.3's diagnosis was wrong.** I read "every high-lift rule is closed-class" off the six
rules that happened to fire in one generation trace and generalised it to the rule base. The
base is 45% open-class. Reading a trace is not measuring a distribution.

### E25.2 The real cause was a decoding default: `top_k=20`

If the content-selection rules exist, the failure has to be downstream. Measured aggregate
probability mass on content words per step, and how much survives truncation:

| prompt | full distribution | top-20 | top-100 | top-500 |
|---|---|---|---|---|
| `the little` | **50.6%** | 1.9% | 15.8% | 36.3% |
| `she was` | **42.8%** | 2.7% | 12.4% | 29.7% |
| `he did` | **38.8%** | 2.1% | 9.3% | 26.1% |

Real held-out text is **46.2%** content words. **The model's distribution was already right.**
Content rate in free generation:

| `top_k` | content rate |
|---|---|
| 20 (old default) | 7.1% |
| 100 | 8.3% |
| 500 | 33.3% |
| 0 = all 2,871 | **48.8%** |

The mechanism: a category rule says "a noun comes next" but not *which* noun, so its consequent
spreads over ~2,699 content words, while a function word gets identity-level rules *and* a large
q(w) in the NCE inversion. Per token, any one function word outscores any one noun -- so the 20
highest-probability words are always function words, and `top_k=20` over 2,871 candidates kept
~2% of the content mass. Real text is 46% content because thousands of individually-rare words
have large *aggregate* mass, which is exactly what the truncation discarded.

`hedge` was wrong for the same reason and is now 1.0: concentration (`mu**h`) sharpens toward
whatever already dominates, so `hedge=3.0` amplified the function words.

Samples at the corrected defaults:

```
the little | though information to dislike that do top not great to party little was became
she was    | but be to on his what ship that are ashore acknowledge can well he
he did     | to emperor time of father daughter should reasonable to give my and soon road
```

Still not grammatical, but it is now real vocabulary at roughly the right content density, and
comparable in character to the bigram's output rather than degenerate.

**Perplexity is unchanged (317.5).** `top_k` is a decoding parameter and never entered scoring,
so this fixes generation quality without touching any reported perplexity -- which also means
none of the perplexity conclusions in E18-E24 were affected by it.

### E25.3 Three wrong diagnoses in a row, and what that pattern says

E23.4 (corpus scale), E24.2 (OOV capacity), E24.3/E25.1 (rule budget) were all wrong, and the
actual bug was a default argument. The common error: I inferred mechanism from aggregate
symptoms and from single traces instead of measuring the intermediate quantity. The measurement
that finally worked -- "how much mass is on content words, before and after each processing
step" -- is the obvious one and cost one short script.

The quota code is kept (defaulting to 0.0, i.e. off) because it is correct, tested, and cheap,
and it will matter if the candidate supply ever exceeds the budget. But it is not a fix for
anything currently observed, and it is recorded as a **negative result**, not as a feature.

71 tests pass (5 new: quota mechanics on a constructed support imbalance, the reserved set
excluding closed-class and identity dims, and the generation defaults, which are now
load-bearing).

---

## Standing summary (SUPERSEDED — see the final one)

**Best results.** On the 1M-token narrative corpus: perplexity **317.5** against a bigram's
253.3, optimal mixture weight zero. On the 87K-token children's corpus: 324.8, and mixing with
a bigram beat the bigram alone (253.9 vs 256.3). Generation now produces real vocabulary at
**48.8% content words against 46.2% in real text** (E25.2), though it is not grammatical.
Representation claims all hold: coverage 96.7%, exact multi-resolution rollup asserted in CI,
L2 similarity gap +0.278, explainable typo robustness, correct senses for rare words.

**Smallness and speed.** 285 learned parameters reach within 5% of the full 2,520-parameter
model, against 34,021 stored counts for a bigram. Cold training on 1M tokens is ~1 minute on
one CPU core, down from ~5.4. Rule count drives inference cost, not training cost.

**Honest framing.** As a language model this loses to a bigram on the same text, and at 1M
tokens contributes nothing to a mixture with one. The distribution is better calibrated than
the samples suggested -- content-word mass was right all along and a decoding default was
throwing it away.

**Ruled out:** context width (E12), rule order for ranking (E12, E15), Gaussian antecedents
(E9/E10), symmetric decode similarity (E11), Cython and BLAS threading (E14, E23.3), raw score
normalisation (E18.2), candidate-side lexicalisation (E19), the linguistic parameter space as a
feature space (E22), corpus scale as the closing lever (E23.4 -- it widens the gap),
thread-based parallelism (E23.3 -- 0.47x, GIL), OOV representation quality as the ceiling
(E24.2 -- worth 1%), **and the open-class rule quota (E25.1 -- no effect; the budget was never
scarce and the base is already 45% open-class)**.

**Confirmed:** vectorising beat parallelising by 11x (E23.2); the rule base does contain
high-lift content-selection rules (E25.1); the model's content/function mass split matches real
text (E25.2).

**Corrected along the way:** the 0.569 balanced accuracy (E12.1); all pre-E17 ranking numbers;
the trigram baseline (E20.2); E19.4's mixture gain quoted without its budget dependence (E22.4);
E22.3's attribution of fit cost (E23.1); my recommendation to push corpus size (E23.4);
**E24.3's "the rules are all closed-class" diagnosis, which came from reading one trace rather
than measuring the base (E25.1)**.

**Open:** generation is not grammatical -- word choice is now plausible but there is no
agreement, no clause structure, and a 2-token window cannot supply either. hits@1 caps near
0.14. GPT-2 untested (blocked, and a data-gap comparison anyway). No neural-embedding
comparison. Experiment B's real encoder and SST paths unrun.

**Methodological note (E25.3).** Four hypotheses about the quality ceiling have now been
refuted, and the real bug was a default argument. The recurring error was inferring mechanism
from aggregate symptoms or single traces instead of measuring the intermediate quantity. The
measurement that worked -- mass on content words before and after each processing step -- was
cheap and should have come first.

---

## E26 — Wider context windows (2, 4, 8, 16, 32) — **FAILED, monotonically**

E12 ruled out context width on the marginal formulation at 87K tokens. Re-asked on the current
stack -- joint ranker, 1M tokens, NCE inversion, corrected decoding -- because generation is now
plausible word-by-word but has no agreement or clause structure, and a 2-token window cannot
supply either. Rule budget raised (`max_rules=20000`, `beam=6000`) per the instruction not to
worry about saturating the base.

| window | dims | ppl | 2gram | mix | lambda | rules | order-2 | content% | fit | peak |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 783 | **355.5** | 286.4 | 285.3 | 0.1 | 2668 | 2608 | 41.1% | 37s | 2.3 GB |
| 4 | 1305 | 368.1 | 286.4 | 286.0 | 0.1 | 3899 | 3839 | 44.6% | 32s | 3.3 GB |
| 8 | 2349 | 374.2 | 286.4 | 286.4 | 0.0 | 6060 | 6000 | 48.2% | 60s | 5.2 GB |
| 16 | 4437 | 384.0 | 286.4 | 286.4 | 0.0 | 6060 | 6000 | 41.1% | 97s | 7.5 GB |
| 32 | 8613 | 395.2 | 286.4 | 286.4 | 0.0 | 6060 | 6000 | 41.1% | 186s | 12.7 GB |

**Monotonically worse with width**, and the mixture weight falls from 0.1 to 0. Content rate does
not improve and the samples do not become more grammatical -- window 32 gives
`the little | could sailed the necessity that so greater from had the wife without my length`.
This replicates E12's conclusion on a completely different formulation, which is worth more than
the original finding was.

### E26.1 Why: the search finds far more spurious long-range structure than real

**5,111 of 6,060 rules reference a lag greater than 2**, and they are weak:

```
IF ctx:prev31:QUANTIFIER AND cand:INFINITIVE_TO  THEN P ~ 0.458  (support=24,  lift=+1.695)
IF ctx:prev13:=whale     AND cand:DETERMINER     THEN P ~ 0.476  (support=21,  lift=+1.668)
IF ctx:prev31:=more      AND cand:INFINITIVE_TO  THEN P ~ 0.600  (support=10,  lift=+1.542)
IF ctx:prev29:DETERMINER AND cand:PREPOSITION    THEN P ~ 0.167  (support=846, lift=+1.582)
```

Against the window-2 rules, which reach lift +8 to +10 (`IF prev1:DETERMINER AND
cand:OPEN_NOUN`, lift +8.411). So the distant-lag rules are 4-5x weaker in lift, and "a
quantifier appeared 31 tokens ago, so expect *to*" on 24 samples is plainly noise.

The damage is not displacement -- `_apply_quota` sorts by |lift|, so the strong rules survive.
It is **dilution**: `predict` blends rules by firing weight, so 5,111 low-lift rules that fire
frequently (`prev29:DETERMINER` fires on 846 samples) pull every prediction toward the base rate.
Widening the window multiplies the pool of candidate pairs -- 32 lags x 261 features is ~2.2M
possible order-2 rules against ~0.2M at window 2 -- and spurious correlations scale with the
pool while real ones do not. More context is more opportunity to be wrong.

This is the multiple-comparisons problem in a rule learner. `min_support` and `min_interaction`
were tuned at window 2 and do not tighten as the search space grows, so the false-discovery rate
rises with width. A width-aware threshold (Bonferroni-style, or a held-out validation gate on
each rule) is the obvious remedy and is untested.

### E26.2 Two measurement bugs in my own sweep, both caught by controls

**A moving control.** The first run scored each condition at positions `i >= window`, giving
every window a *different* test set. It showed non-monotonic perplexity (324 -> 381 -> 318 -> 365)
which I would have had to explain. The tell was the **bigram column moving** (253.3 / 277.1 /
240.8 / 265.5) when the same bigram trained on the same data must score identically everywhere.
Fixed by scoring every condition at `i >= 32`; the bigram is now constant at 286.4 and the fuzzy
column is cleanly monotonic. **Include a control that cannot legitimately vary, then check that
it does not.**

**An uncatchable OOM.** window=32 in float64 needed ~22 GB and the OOM killer SIGKILLed the
process, so the `except MemoryError` handler never ran and the run simply ended with no message.
Fixed by adding a `dtype` option to `MembershipRuleRegressor` (default float64, so the GEMM
exactness test at rel=1e-9 still holds) and running the whole sweep in float32 so precision is
not a variable across rows.

### E26.3 What is left

Width is not the missing ingredient for grammaticality, and neither were corpus scale (E23.4),
OOV quality (E24.2), or rule budget (E25.1). What the samples still lack -- agreement, clause
structure -- is *structural* rather than a matter of how much left context is visible, and a
flat conjunction of lag-indexed memberships has no way to represent "the subject of this clause"
regardless of window size. The honest next step is not another knob on this architecture.

72 tests pass (1 new: `dtype` defaults to float64 so the exactness guarantee is not silently
weakened).

---

## Standing summary (SUPERSEDED — see the final one)

**Best results.** 1M-token narrative corpus, window 2: perplexity **317.5** at 20,000 training
positions against a bigram's 253.3, optimal mixture weight zero. On the 87K-token children's
corpus: 324.8, and mixing with a bigram beat the bigram alone (253.9 vs 256.3). Generation
produces real vocabulary at 48.8% content words against 46.2% in real text, but is not
grammatical. Representation claims hold: coverage 96.7%, exact rollup asserted in CI, L2
similarity gap +0.278, explainable typo robustness, correct senses for rare words.

**Smallness and speed.** 285 learned parameters reach within 5% of the full 2,520-parameter
model, against 34,021 stored counts for a bigram. Cold training on 1M tokens is ~1 minute on one
CPU core. Rule count drives inference cost, not training cost. A `dtype=float32` option halves
the design matrix when width demands it.

**Honest framing.** This loses to a bigram on the same text and adds nothing to a mixture with
one at 1M tokens. Its distribution is better calibrated than its samples suggested. Its real
deliverable so far is that every failure has been *legible* -- each negative below was diagnosed
by reading rules or intermediate quantities, not inferred from a loss curve.

**Ruled out:** context width (E12, and **replicated far more strongly in E26** -- monotonically
worse out to 32 tokens), rule order for ranking (E12, E15), Gaussian antecedents (E9/E10),
symmetric decode similarity (E11), Cython and BLAS threading (E14, E23.3), raw score
normalisation (E18.2), candidate-side lexicalisation (E19), the linguistic parameter space as a
feature space (E22), corpus scale as the closing lever (E23.4), thread-based parallelism (E23.3),
OOV representation quality as the ceiling (E24.2), the open-class rule quota (E25.1).

**Confirmed:** vectorising beat parallelising by 11x (E23.2); the base already contains high-lift
content-selection rules (E25.1); content/function mass matches real text (E25.2); **the rule
learner's false-discovery rate rises with the search space and dilutes predictions (E26.1)**.

**Corrected along the way:** the 0.569 balanced accuracy (E12.1); all pre-E17 ranking numbers;
the trigram baseline (E20.2); E19.4's mixture gain quoted without budget dependence (E22.4);
E22.3's fit-cost attribution (E23.1); my recommendation to push corpus size (E23.4); E24.3's
"all rules are closed-class" diagnosis (E25.1); **a per-condition test set in the first window
sweep, caught by a moving bigram control (E26.2)**.

**Open, and now the central question.** Five hypotheses about the quality ceiling have been
refuted (scale, OOV quality, rule budget twice, context width). What generation lacks --
agreement, clause structure -- is structural: a flat conjunction of lag-indexed memberships
cannot represent "the subject of this clause" at any window size. Candidates: a width-aware
significance threshold so wide contexts stop diluting (E26.1); rules over *relations* rather
than lag positions; or accepting that this architecture's contribution is interpretable
*scoring* rather than generation. hits@1 caps near 0.14. GPT-2 untested (blocked). No
neural-embedding comparison. Experiment B's real encoder and SST paths unrun.

---

## E27 — Both candidate fixes for E26 — **both FAILED, and they refuted E26.1's mechanism**

Two options from E26.3, planned and evaluated on the E26 protocol (1M-token corpus, same split,
TRAINPOS=6000, positions >= 32, float32, bigram control constant at 286.4).

**Option A -- width-aware false-discovery control.** Two gates, because they test the same claim
with and without a distributional assumption:
* **A1 analytic (Bonferroni).** |lift| is already a z-statistic up to a constant: under the null
  a rule's consequent is a mean of `support` draws with SE = sqrt(p(1-p)/support), so
  `z = (consequent - default)/SE = lift / sqrt(p(1-p))`. Gate on `alpha / m` with `m` the
  candidates *actually tested per level*, so the threshold tightens automatically with width --
  the width-awareness E26.1 said was missing. Costs nothing to compute.
* **A2 empirical (held-out replication).** Mine on 70% of rows, then require each rule's effect
  to reproduce in *direction* on the held-out 30%. No distributional assumption, and the survival
  rate is a **direct measurement of the false-discovery rate** rather than an estimate.

**Option B -- relational context slots** (`fuzzyembed/relations.py`). Replace lag-indexed slots
with 8 functionally-defined ones -- nearest preceding verb, subject, head noun, preposition,
determiner, sentence start, plus lags 1 and 2 which E26 showed do carry signal. Reach is
unbounded; dimension is fixed at `n_slots * base`. Not a parse: no parser is installable here
(no spaCy/stanza/benepar, and nltk ships no tagger data), so slots are filled by scanning back
for the nearest token whose fuzzy syntactic category matches -- a shallow approximation, stated
as such in the module docstring.

### E27.1 Results

| condition | dims | ppl | mix | lam | rules | gated | repl% | far% | fit |
|---|---|---|---|---|---|---|---|---|---|
| A0 lag w2 (E26 baseline) | 783 | **355.5** | 285.3 | 0.1 | 2668 | — | — | 0% | 29s |
| A0 lag w32 (E26 baseline) | 8613 | 395.1 | 286.4 | 0.0 | 6060 | — | — | 84% | 195s |
| A1 lag w2 + signif | 783 | 371.5 | 285.9 | 0.1 | **81** | 2587 | — | 0% | 23s |
| A1 lag w8 + signif | 2349 | 371.6 | 285.1 | 0.1 | **72** | 5988 | — | 1% | 52s |
| A1 lag w32 + signif | 8613 | 377.1 | 285.2 | 0.1 | **50** | 6010 | — | 0% | 214s |
| A2 lag w8 + holdout | 2349 | 371.7 | 286.2 | 0.1 | 4306 | — | **71%** | 63% | 55s |
| A2 lag w32 + holdout | 8613 | 394.5 | 286.4 | 0.0 | 4794 | — | **79%** | 83% | 179s |
| B1 relational | 2349 | 374.8 | 286.4 | 0.0 | 5629 | — | — | 53% | 42s |
| B2 relational + signif | 2349 | 374.0 | 286.2 | 0.1 | 80 | 5549 | — | 11% | 36s |
| B3 relational + holdout | 2349 | 373.5 | 286.4 | 0.0 | 4081 | — | **72%** | 52% | 39s |

**Neither option beats plain window-2 (355.5).** Every variant lands in 371-395.

### E27.2 The important result: E26.1's mechanism was wrong

E26.1 claimed the distant-lag rules were **spurious** -- false discoveries from a search space
grown to ~2.2M candidate pairs. The holdout gate tests that directly, and refutes it:

**79% of mined rules replicate on held-out rows at window 32, versus 71% at window 8.**

If wide windows were mining noise, replication should be *low* at window 32 and *fall* with
width. It is high and it *rises*. And `far%` stays at 83% after filtering, so the long-lag rules
are among those that reproduce. They are **statistically real and predictively useless**.

That makes sense in hindsight and I should have predicted it: long-range lexical co-occurrence in
a novel is genuinely present -- `IF prev13:=whale AND cand:DETERMINER` is a real regularity in
Moby-Dick, a topic/register effect. It is just not *next-token* information beyond what adjacency
already supplies. The problem is **redundancy, not overfitting**, and no amount of significance
control can fix redundancy. Both of Option A's gates were aimed at the wrong target, which is why
the analytic one succeeded mechanically (`far%` 84% -> 0-1%, exactly as designed) and still lost
perplexity: it removed rules that were real, and their removal cost more than their dilution did.

This is the sixth refuted hypothesis about the ceiling, and the second time a diagnosis of mine
was refuted by the experiment built to act on it (E24.3 -> E25.1, now E26.1 -> E27.2).

### E27.3 Option B: the structure is learnable, and it does not help

Relational slots do what they were designed to do. They reach window-32 distances at
window-8 dimension (2,349 vs 8,613 columns) and score 374.8 against lag-32's 395.1 -- **5%
better at 3.7x fewer features**. And the grammar they learn is real:

```
IF ctx:verb:AUXILIARY  AND cand:NEGATOR           THEN 0.304  (support=237,  lift=+2.949)
IF ctx:verb:=did       AND cand:NEGATOR           THEN 0.727  (support=11,   lift=+2.040)
IF ctx:verb:OPEN_VERB  AND cand:NEGATOR           THEN 0.029  (support=554,  lift=-1.949)
IF ctx:det:=the        AND cand:OPEN_NOUN         THEN 0.144  (support=2988, lift=+1.737)
IF ctx:prep:=of        AND cand:noun.phenomenon   THEN 0.397  (support=35,   lift=+1.678)
```

Those are correct English generalisations at unbounded distance -- "after an auxiliary expect
*not*", and the *negative* rule "after a main verb do **not** expect *not*", which is the
do-support constraint stated as a fuzzy rule. The representation can express structure. It just
does not improve next-token prediction, because adjacency already implies most of it: if `did` is
the governing verb, `did` is usually also `prev1`.

The heuristic's limits are visible where predicted:

```
after 'the little rabbit that lived in the old wood was very happy'
    verb='was', subj='wood', noun='wood', det='the'
```

`subj` should be `rabbit`; `wood` is inside the relative clause. So a real parser would help the
slot filling -- but since exact slots at window 2 already beat approximate slots with unbounded
reach, a parser would be improving the input to a mechanism that is not the bottleneck.

### E27.4 One genuine win, in the other direction

The significance gate is a **model-compression** tool even though it is not a quality fix:

| | rules | learned params | ppl |
|---|---|---|---|
| lag w2, ungated | 2668 | ~8,000 | 355.5 |
| lag w2, gated | **81** | **~240** | 371.5 |
| lag w32, gated | **50** | **~150** | 377.1 |

**33x fewer parameters for 4.5% worse perplexity**, and it also makes perplexity nearly
width-independent (371.5 / 371.6 / 377.1 across windows 2, 8, 32) where ungated it degraded
monotonically. It also restores a non-zero mixture weight at every width (lambda=0.1 where
ungated window-32 was 0.0). Consistent with E22.3, where 285 parameters reached within 5% of
2,520. This architecture is extraordinarily compressible, and that -- not perplexity -- is where
its numbers are competitive.

### E27.5 Where this leaves the project

Six refuted ceiling hypotheses: corpus scale (E23.4), OOV quality (E24.2), rule budget (E25.1),
context width (E26), false-discovery control and relational structure (both here). The evidence
now points somewhere fairly specific:

* Long-range context is **real but redundant** for next-token prediction (E27.2).
* The rule base **already contains** the right categorical grammar, at both adjacency (E25.1) and
  unbounded distance (E27.3).
* The distribution's content/function split **already matches real text** (E25.2).

So the missing ingredient is not more context, more data, better features, or better rule
selection. What generation lacks is *within-category choice* -- which noun, not that a noun --
and a zero-order TSK rule cannot express it, because its consequent is a constant shared by every
word matching the antecedent. That is an architectural limit, not a tuning one: the model is a
well-calibrated categorical predictor, and categorical prediction has a perplexity floor set by
category size.

The honest options are (a) accept interpretable *scoring* as the contribution and stop pursuing
generation, or (b) give rules word-level consequents -- a first-order TSK system where the
consequent is a function over the candidate rather than a scalar, which is a different model.
Neither is another knob on this one.

77 tests pass (6 new, including one that started out asserting the wrong thing: it expected a
12-row rule to be rejected outright, when at m=2 that rule genuinely is significant -- the gate's
property is width-awareness, not absolute strictness, and the test now checks that the same rule
survives a narrow search and fails a wide one).

---

## Standing summary (SUPERSEDED — see the final one)

**Best results.** 1M-token narrative corpus, window 2, 20,000 training positions: perplexity
**317.5** against a bigram's 253.3, optimal mixture weight zero. On the 87K-token children's
corpus: 324.8, and mixing with a bigram beat the bigram alone (253.9 vs 256.3). Generation
produces real vocabulary at 48.8% content words against 46.2% in real text, and is not
grammatical. Representation claims hold: coverage 96.7%, exact rollup asserted in CI, L2
similarity gap +0.278, explainable typo robustness, correct senses for rare words.

**Smallness -- the strongest numbers in the project.** With the E27 significance gate: **81 rules,
~240 learned parameters, ppl 371.5**, against 2,668 rules for 355.5 -- 33x fewer parameters for
4.5% worse perplexity, and perplexity becomes width-independent. A bigram needs 34,021 stored
counts for 253.3. Cold training on 1M tokens is ~1 minute on one CPU core, no GPU.

**Honest framing.** This loses to a bigram on the same text and adds nothing to a mixture with
one at 1M tokens. Its competitive axis is parameter count, not quality. Its distinctive
deliverable is that every failure has been *legible*: each negative below was diagnosed by
reading rules or measuring an intermediate quantity, and two of my own diagnoses were then
refuted by the experiments built to act on them.

**Ruled out:** context width (E12, E26 -- monotonic to 32 tokens), rule order (E12, E15), Gaussian
antecedents (E9/E10), symmetric decode similarity (E11), Cython and BLAS threading (E14, E23.3),
raw score normalisation (E18.2), candidate-side lexicalisation (E19), the linguistic parameter
space (E22), corpus scale as the closing lever (E23.4), thread parallelism (E23.3), OOV quality
as the ceiling (E24.2), open-class rule quota (E25.1), **width-aware false-discovery control and
relational context slots (E27)**.

**Confirmed:** vectorising beat parallelising by 11x (E23.2); the base already holds the right
categorical grammar at adjacency (E25.1) and at unbounded distance (E27.3); content/function mass
matches real text (E25.2); **long-range context is statistically real -- 79% of rules replicate on
held-out rows -- and predictively redundant (E27.2)**.

**Corrected along the way:** 0.569 balanced accuracy (E12.1); all pre-E17 ranking numbers; the
trigram baseline (E20.2); E19.4's mixture gain quoted without budget dependence (E22.4); E22.3's
fit-cost attribution (E23.1); my recommendation to push corpus size (E23.4); E24.3's "all rules
are closed-class" (E25.1); a per-condition test set caught by a moving control (E26.2);
**E26.1's dilution/false-discovery mechanism (E27.2)**.

**The diagnosis, now specific.** Six refuted ceiling hypotheses converge on one limit: the model
is a **well-calibrated categorical predictor**, and a zero-order TSK rule's consequent is a
scalar shared by every word matching its antecedent. It can say "a noun follows" but never "which
noun", so its perplexity floor is set by category size. That is architectural. The two honest
paths: accept interpretable *scoring* as the contribution, or move to word-level consequents
(first-order TSK, consequent a function over the candidate) -- a different model, not a knob on
this one.

**Open:** hits@1 caps near 0.14. GPT-2 untested (blocked, and a data-gap comparison anyway). No
neural-embedding comparison. Experiment B's real encoder and SST paths unrun.

---

## E28 — Word-level consequents (first-order TSK) — **the first architecture to beat the bigram**

E27.5's conclusion was that the ceiling is the *shape* of the consequent: a zero-order rule
assigns one scalar to every word matching its antecedent, so it can say "a noun follows" and
never "which noun". Six refuted hypotheses had all left that untouched. This replaces the scalar
with a word distribution per rule.

### E28.1 Formulation, and why the obvious first-order is the wrong one

Classic Takagi-Sugeno makes the consequent linear in the inputs. Here that means linear in the
*candidate's* features -- and it would not help, because those are the same coarse categories the
antecedent already uses. "Prefer noun.person over noun.substance" is already expressible in the
zero-order system as two rules; that route buys parameter efficiency, not capability.

What is missing is **word identity**, so:

    p(w | ctx)  =  sum_r  omega_r(ctx) . p_r(w)  /  sum_r omega_r(ctx)

with ``omega_r`` the rule's context-side firing and ``p_r(w)`` the word distribution observed
when it fires. Each rule becomes a soft, *named*, overlapping context class with its own
next-word distribution. This is a **fuzzy class-based language model** -- the graded analogue of
class-based n-grams (Brown, Della Pietra, deSouza, Lai & Mercer, *Class-Based n-gram Models of
Natural Language*, Computational Linguistics 18(4), 1992), where classes were a hard clustering.

Two structural consequences. **The NCE inversion disappears** -- ``p_r(w)`` are already
distributions over words, so there is no ``q(w).s/(1-s)`` step and no double-counting of the
unigram prior, which was the E19 failure. And **``must_include`` inverts**: it was correct for
scalar consequents (a context-only rule shifts every candidate equally and cannot change a
ranking) and is *wrong* here, because a context-only rule is exactly a class.

That mattered immediately. Reusing the zero-order rule base gives classes that are all **single
features** -- ``must_include`` plus ``max_order=2`` leaves exactly one context term per rule --
which is why the ``firing`` and ``specific`` weightings scored *identically*: ``n_ctx`` was
constant at 1. So a ``ContextClassMiner`` was added that drops the constraint and selects
context-only conjunctions by **firing-weighted information gain** about the next word, the
decision-tree criterion applied to fuzzy memberships. Lift cannot be used, being zero by
construction for context-only antecedents under the NCE target.

### E28.2 Results

Same protocol as E26/E27: 1M-token corpus, same split, positions >= 32, bigram control.

| model | ppl | params | classes |
|---|---|---|---|
| zero-order TSK (scalar consequents) | 355.5 | 7,944 | 2,668 |
| 2-gram, same data, tuned | **286.4** | 166,970 | — |
| reuse rules, firing weights | 607.8 | 54,369 | 2,589 |
| reuse rules, infogain weights | 621.1 | 54,369 | 2,589 |
| reuse rules, no identity ctx (control) | 791.1 | 23,814 | 1,134 |
| mined classes, order 1 | 641.1 | 7,812 | 372 |
| **mined classes, order 2** | **601.2** | 35,091 | 1,671 |
| mined classes, order 2, no identity | 646.2 | 24,675 | 1,175 |

Standalone this looked like a clear failure -- worse than the zero-order model it replaced. But
the mixture told a different story, and `explain` showed why.

### E28.3 Smoothing is the dominant parameter, and its optimum depends on the use

`explain` on the first version: ``IF prev2:=the AND prev1:QUANTIFIER -> jackal(0.105)`` -- "jackal"
as the top prediction after "the little", from a handful of firing observations. Two causes, and
the second is the E26.1 trap in a new place: ``alpha=0.5`` pseudo-counts is far too little
smoothing, **and information gain rewards low entropy, so class selection is actively biased
toward the under-observed classes that look sharp by accident**.

| alpha | classes | standalone ppl | best mixture | lambda |
|---|---|---|---|---|
| 0.5 | 1670 | 601.1 | **256.2** | 0.4 |
| 5 | 1671 | 437.0 | 257.2 | 0.4 |
| 50 | 1720 | 352.9 | 264.5 | 0.4 |
| 200 | 1652 | **343.5** | 272.9 | 0.4 |

**Standalone goes 601.1 -> 343.5**, which beats the zero-order model's 355.5 -- the first time a
change to this architecture has improved standalone perplexity in ten experiments. It still loses
to the bigram's 286.4.

**And the two optima are at opposite ends of the smoothing axis, for a principled reason.** Heavy
smoothing makes each class a reliable predictor on its own; a mixture partner already supplies
reliable mass and wants the fuzzy model's *sharp* class-conditional peaks instead. So "how much to
smooth" is not a tuning detail, it is a question about what the model is for.

### E28.4 The headline: complementarity, and it is large

| | ppl | vs bigram |
|---|---|---|
| bigram alone | 286.4 | — |
| **mixture at lambda=0.4** | **256.2** | **-10.5%** |

For comparison, the zero-order model at 1M tokens had optimal mixture weight **zero** (E23.4) --
it contributed nothing. The best complementarity previously measured anywhere in this project was
0.94% at 87K tokens (E22.4), and E19.4's 4.7% was at a small budget and did not survive scale.
This is **10.5% at 1M tokens with lambda=0.4**, a substantial weight rather than a token one.

The controls matter, and they hold. A class whose context side is a lexeme identity has
``p_r(w) = p(w | prev=w')``, which is literally a bigram row, so "beats the bigram" could have
meant "contains the bigram". Dropping every identity-context class costs 601.2 -> 646.2 standalone
-- real but not decisive -- so the categorical classes are carrying most of it, and the mixture
gain is not the model rediscovering its partner.

### E28.5 What the classes look like

```
IF ctx:prev1:=did                          -> not(0.783), at(0.058), the(0.030)     [5.02 nats]
IF ctx:prev1:AUXILIARY AND ctx:prev2:=he   -> not(0.183), to(0.044), have(0.037)    [2.04 nats]
IF ctx:prev1:=very                         -> much(0.095), good(0.059), well(0.036) [2.09 nats]
IF ctx:prev2:=was AND ctx:prev1:INTENSIFIER-> on(0.066), well(0.066), good(0.066)   [2.69 nats]
```

Each is a named class with a readable word distribution and a measured information gain in nats.
This is the interpretability claim finally doing work at the level the project was about: not just
"why this score" but "which words this context prefers, and how much knowing the context is
worth". Samples are still not grammatical, but content selection is visibly context-driven --
``the little | black royal of gingerbread in her hopes as now the instant which had entirely``.

### E28.6 Where this leaves it

**Confirmed:** E27.5's diagnosis was right. The consequent's shape *was* the binding constraint,
and it is the first thing in this project that, when changed, moved both standalone perplexity
(355.5 -> 343.5) and complementarity (0% -> 10.5% mixture weight 0.4).

**Still true:** standalone it loses to a bigram trained on the same text (343.5 vs 286.4), and
parameter count is now 35K sparse rather than 8K, so the extreme-smallness result belongs to the
zero-order system with significance gating (81 rules / ~240 params / 371.5, E27.4).

**Honest read.** The contribution is an interpretable model that *adds* ~10% to a conventional LM
while explaining its own predictions as named graded classes over words. It is not a replacement
for an n-gram at this scale, and the write-up should say so.

80 tests pass (3 new: conjunctive classes are actually mined, the model puts mass on the specific
word rather than the category, and smoothing monotonically flattens).

---

## Standing summary (SUPERSEDED — see the final one)

**Best result.** First-order TSK (word-level consequents, `firstorder.py`) mixed with a bigram:
**perplexity 256.2 against the bigram's 286.4, a 10.5% improvement at lambda=0.4** on the
1M-token corpus. Standalone 343.5, which beats the zero-order model's 355.5 and loses to the
bigram. This is the first change in the project to move both numbers, and it confirms E27.5's
diagnosis that the consequent's *shape* was the binding constraint.

**Smallest useful model.** Zero-order with E27's significance gate: **81 rules, ~240 learned
parameters, ppl 371.5** (against 2,668 rules for 355.5, and 34,021 stored counts for a bigram at
253.3 on the E27 protocol). Cold training on 1M tokens is ~1 minute on one CPU core, no GPU.

**Representation.** Coverage 96.7%, exact multi-resolution rollup asserted in CI, L2 similarity
gap +0.278, explainable typo robustness, correct sense assignment for rare words.

**Interpretability, doing actual work.** Named graded classes with readable word distributions and
information gain in nats (`IF prev1:=did -> not(0.783)`, 5.02 nats). Two of the project's failures
were diagnosed by reading rules, and three of my own diagnoses were refuted by the experiments
built to act on them.

**Honest framing.** No configuration beats a bigram trained on the same text standalone. The
contribution is a model that *adds* ~10% to a conventional LM while explaining itself. Generation
is not grammatical; content selection is context-driven but syntax is not.

**Ruled out:** context width (E12, E26), rule order (E12, E15), Gaussian antecedents (E9/E10),
symmetric decode similarity (E11), Cython and BLAS threading (E14, E23.3), raw score normalisation
(E18.2), candidate-side lexicalisation (E19), the linguistic parameter space (E22), corpus scale as
the closing lever (E23.4), thread parallelism (E23.3), OOV quality as the ceiling (E24.2),
open-class rule quota (E25.1), width-aware false-discovery control and relational slots (E27),
candidate-linear first-order consequents (E28.1, on argument).

**Confirmed:** vectorising beat parallelising 11x (E23.2); the base holds the right categorical
grammar at adjacency (E25.1) and unbounded distance (E27.3); content/function mass matches real
text (E25.2); long-range context is real but redundant (E27.2); **the consequent's shape was the
ceiling (E28)**.

**Corrected along the way:** 0.569 balanced accuracy (E12.1); all pre-E17 ranking numbers; the
trigram baseline (E20.2); E19.4's mixture gain quoted without budget dependence (E22.4); E22.3's
fit-cost attribution (E23.1); the recommendation to push corpus size (E23.4); E24.3's "all rules
are closed-class" (E25.1); a per-condition test set caught by a moving control (E26.2); E26.1's
dilution mechanism (E27.2); **E28's own first standalone numbers, which were an undersmoothing
artefact (E28.3)**.

**Open.** Generation is not grammatical. hits@1 caps near 0.14. Class selection by information
gain is biased toward under-observed classes -- a held-out gain criterion is the obvious fix and is
untested. GPT-2 untested (blocked, and a data-gap comparison anyway). No neural-embedding
comparison. Experiment B's real encoder and SST paths unrun.

---

## E29 — Plan: consolidate the first-order win

Written **before** running, so the hypotheses are on record and a negative result cannot be
retro-fitted into a success. Ordered by how directly each attacks a defect E28 actually measured.

**E29.1 Held-out gain for class selection.** *Defect:* E28.3 showed information gain is computed
on the same rows that estimated the class, so gain rewards low entropy and selection is biased
toward classes that look sharp by accident (`jackal` after "the little"). That bias is why low
alpha -- the setting the mixture wants -- is unusable standalone. *Change:* split positions;
estimate ``p_r`` on part A, score candidate classes by held-out cross-entropy reduction on part B.
*Hypothesis:* removes the bias, so sharp classes become reliable and the standalone/mixture
smoothing tradeoff (E28.3) narrows or disappears. *Success:* standalone at low alpha improves
substantially, mixture gain does not regress.

**E29.2 Hierarchical backoff instead of unigram smoothing.** *Defect:* every class currently
smooths toward the **unigram**, but a conjunction like ``prev2:=the AND prev1:QUANTIFIER`` should
back off to its *parent* class ``prev1:QUANTIFIER``, which is far more informative than the
unigram. Smoothing to the unigram throws away the class hierarchy the miner just built. *Change:*
back off an order-2 class to a mass-weighted blend of its order-1 parents. *Hypothesis:* strictly
better than unigram backoff, and reduces sensitivity to alpha. Standard practice for exactly this
problem (Katz backoff; Kneser-Ney). *Success:* beats E28's best standalone 343.5.

**E29.3 Class-estimation data.** *Rationale:* E28 estimated distributions from 20,000 positions
while the corpus has ~800K. Estimating a *distribution* per class needs more data than estimating
a scalar, and E16 found this stack data-bound. *Hypothesis:* more positions helps monotonically and
more than it did for the zero-order model. *Success:* monotone improvement.

**E29.4 A stronger mixture partner (honesty check).** *Rationale:* the 10.5% gain is measured
against a **bigram**. If it evaporates against a trigram, the contribution is much smaller than
E28 implies, and that must be reported. *Hypothesis:* the gain shrinks but survives, because
class-based backoff is complementary to sparse higher-order counts for a different reason than a
bigram is. *Success criterion is honesty, not a number.*

**E29.5 Parameter frontier.** How few classes retain the mixture gain? Ties to the standing
smallness result (81 rules / ~240 params for the zero-order model).

**E29.6 A number for "not grammatical".** Every generation claim so far has been eyeballed.
Measure syntactic plausibility of generated text against real text on the same tagger -- the rate
at which adjacent category pairs are ones the corpus actually produces. Gives a metric that can
move, instead of an impression.

---

## E29 — Results against the plan: 2 failed, 1 succeeded enormously, 1 metric was broken

Executed against the plan recorded above. Protocol unchanged (1M-token corpus, same split,
positions >= 32, bigram and trigram controls).

### E29.1 Held-out class selection — **FAILED**

| sel | backoff | alpha | classes | standalone | best mix | lambda |
|---|---|---|---|---|---|---|
| 0.0 | unigram | 0.5 | 1671 | 601.2 | **256.2** | 0.4 |
| 0.0 | unigram | 50 | 1721 | **352.9** | 264.5 | 0.4 |
| 0.3 | unigram | 0.5 | 397 | 475.8 | 286.4 | **0.0** |
| 0.3 | unigram | 50 | 397 | 359.1 | 257.1 | 0.4 |

At alpha=0.5 held-out selection *destroys* the mixture contribution (lambda falls to 0). At
alpha=50 it improves the mixture slightly (264.5 -> 257.1) but not past what plain low-alpha
already achieves. **Caveat on my own design:** with 30% of rows held out, the `min_mass=20` floor
applies to held-out mass, so the effective threshold is ~3.3x stricter and class count drops
1671 -> 397. The experiment therefore conflates *criterion* with *threshold*, and the clean
version would rescale the floor. The prediction (that removing the sharp-by-accident bias would
let low alpha work standalone) is not supported either way: standalone at alpha=0.5 stayed poor.

### E29.2 Hierarchical parent backoff — **FAILED, and the reason is instructive**

Worse standalone everywhere (735.2 vs 601.2 at alpha=0.5; 404.6 vs 352.9 at alpha=50), mixture
identical or worse.

**Why:** in a mixture-of-experts the parents are **already in the mixture**. Katz/Kneser-Ney
backoff exists for a model that consults *one* distribution and needs a fallback when the specific
context is unseen; here every class contributes simultaneously, so the averaging already performs
the backoff. Making a child resemble its parent adds redundancy, not robustness, and dilutes the
one thing the conjunction contributed. Right idiom, wrong architecture -- worth recording because
it is a natural thing to reach for and it is wrong here for a structural reason.

### E29.3 Class-estimation data — **WORKED, and it changes the project's conclusion**

| positions | classes | standalone | vs bigram | lambda | vs trigram | lambda | fit |
|---|---|---|---|---|---|---|---|
| 20,000 | 1671 | 601.2 | 256.2 | 0.4 | — | — | 17s |
| 60,000 | 2944 | 456.6 | 239.8 | 0.5 | — | — | 37s |
| 150,000 | 3000 | 308.2 | 228.7 | 0.6 | — | — | 84s |
| 300,000 | 3000 | 244.2 | 215.6 | 0.8 | 212.2 | 0.7 | 177s |
| 600,000 | 3000 | 221.0 | 204.8 | 0.8 | 202.0 | 0.8 | 705s |
| **624,325 (corpus exhausted)** | 3000 | **219.9** | **204.5** | 0.8 | **201.7** | 0.8 | 940s |

**Standalone 219.9 against a bigram's 286.4 and a tuned trigram's 284.4 — the first time any
configuration in this project beats an n-gram head-to-head**, by 23%. Mixed, 201.7, a 29% gain
over the trigram. The mixture weight on the fuzzy model rose from 0.4 to **0.8**: it is now the
dominant partner rather than a minor correction.

The reason E28 looked so much weaker is simply that it estimated distributions from 20,000
positions. Estimating a *distribution* per class needs far more data than estimating a scalar --
which is obvious in hindsight and is exactly the E16 pattern reasserting itself in a new place.
E23.4's conclusion ("corpus scale is the lever that makes the fuzzy model *relatively worse*")
was a fact about the **zero-order** model, and it inverts here: with word-level consequents,
scale is the lever that wins.

**A scaling defect found on the way.** The 150,000-position run was SIGKILLed with no traceback:
``np.column_stack`` materialised all 9,730 order-2 candidate columns at once, 150,000 x 9,730 x 8
bytes = 11.7 GB. Pair generation is now chunked, which is what made the 624K run possible at all.
Same uncatchable-OOM lesson as E26.2.

### E29.4 A stronger partner (honesty check) — **the gain survives**

| partner | alone | best mix | lambda | gain |
|---|---|---|---|---|
| bigram | 286.4 | 204.5 | 0.8 | 28.6% |
| trigram | 284.4 | **201.7** | 0.8 | 29.1% |

3-way mixture: **ppl 201.7 at fuzzy=0.8, bigram=0.0, trigram=0.2.** The bigram receives **zero
weight** once the fuzzy model is present -- it is subsumed. This was the check that could have
deflated E28's headline, and it does the opposite.

### E29.5 Parameter frontier

At 300,000 positions:

| classes | params (top-20 sparse) | standalone | best mix |
|---|---|---|---|
| 50 | ~1,050 | 312.6 | 257.2 |
| 200 | ~4,200 | 296.5 | 247.3 |
| 800 | ~16,800 | 254.6 | 228.6 |
| 3000 | ~63,000 | 244.2 | 215.6 |

**50 classes / ~1,050 parameters reach 312.6**, better than the zero-order model's best 355.5 at
8x the parameters. Quality keeps improving with class count, so the smallness and quality optima
now differ -- unlike the zero-order system, where 285 parameters were within 5% of the best.

### E29.6 The grammaticality metric — **broken as designed, then fixed**

First attempt: "share of adjacent category pairs the corpus produces". Real text, fuzzy
generation, and a bigram all scored **100.0%**. With ~17 categories there are only ~289 possible
pairs and essentially all occur somewhere in 20,000 sentences, so the metric cannot discriminate
anything. Recording it as a failed measurement rather than quietly dropping it.

Replaced with **category-sequence perplexity** under a category bigram fitted on training text --
graded, so a degenerate sequence cannot saturate it:

| | category ppl (lower = more plausible) |
|---|---|
| real held-out text | **8.18** |
| first-order fuzzy | **10.10** |
| bigram, same data | 12.84 |
| unigram (floor) | 14.76 |

So generation is measurably more syntactically plausible than a bigram's and still clearly short
of real text. "Not grammatical" now has a number, and the fuzzy model sits about 30% of the way
from bigram to real on this scale. Samples remain locally plausible and globally incoherent:
``he did | not possession of marrying you could hardly very voice and certainly man his seems``.

### E29.8 One inconsistency in the E29.3 table, found while plotting

The scaling table above was assembled from two runs with **different `backoff` settings**: 60K/150K
/300K came from the parent-backoff config (it was `best_mixed` in that script) and 20K/600K/624K
from unigram backoff. The two agree by 300K (244.6 vs 244.2) but differ a lot at 20K (735.2 vs
601.2), so plotting them as one curve would have been misleading. `experiments/scaling_curve.py`
refits the whole curve under one configuration; the corrected points are 60K **422.7** (was 456.6)
and 150K **303.7** (was 308.2), and `experiments/e29_curve.json` is the consistent series the
write-up plots. The conclusion is unchanged and slightly stronger.

### E29.7 What the classes look like at full data

```
IF ctx:prev2:=he AND ctx:prev1:=did  -> not(0.701), so(0.045), it(0.033)   [4.37 nats]
IF ctx:prev2:PRONOUN AND ctx:prev1:=did -> not(0.674), it(0.032), so(0.029) [4.03 nats]
IF ctx:prev1:=little  -> jackal(0.035), boy(0.033), girl(0.025), man(0.020) [0.52 nats]
```

The `little` class is now sensible -- `boy`, `girl`, `man` -- where at 20,000 positions it was
`jackal` at 0.105 off a handful of observations. `jackal` survives at 0.035 because Bryant's
stories genuinely contain "the little jackal" often; at full data it is ranked among plausible
company instead of dominating.

---

## Standing summary — CURRENT (as of E29)

**Headline.** First-order TSK -- word-level consequents, a fuzzy class-based LM -- **beats
n-gram baselines head-to-head on the same corpus and split**: perplexity **219.9** against a
bigram's 286.4 and a tuned trigram's 284.4, a 23% improvement. Mixed with a trigram, **201.7**
(29% better than the trigram alone) at fuzzy weight 0.8, with the bigram receiving zero weight in
a 3-way mixture. This is the first configuration in the project to win outright rather than
contribute to a mixture.

**How it gets there.** Rules keep the antecedents the zero-order system mined, but each rule
carries a **word distribution** rather than a scalar, estimated as the firing-weighted next-word
distribution when the rule's context side fires. Classes are mined context-only by
firing-weighted information gain. Estimation is strongly data-bound: 20,000 -> 624,325 positions
moves standalone perplexity 601.2 -> 219.9, and the curve only flattened when the corpus ran out.

**Generation.** Category-sequence perplexity **10.10** against real text's 8.18 and a bigram's
12.84 -- measurably more syntactically plausible than a bigram, clearly short of real text. Text
is locally plausible, globally incoherent.

**Smallness.** 50 classes / ~1,050 parameters reach 312.6 (better than the zero-order model's best
355.5 at 8x the parameters). Unlike the zero-order system, quality keeps improving with size, so
the small and best configurations now differ. Training on 1M tokens is ~16 minutes on one CPU
core at full data, ~3 minutes at 300K positions, no GPU.

**Representation.** Coverage 96.7%, exact multi-resolution rollup asserted in CI, L2 similarity
gap +0.278, explainable typo robustness, correct senses for rare words.

**Interpretability.** Named graded classes with readable word distributions and information gain
in nats: `IF prev2:=he AND prev1:=did -> not(0.701)` [4.37 nats]. This is what the project set out
to build, and it is now also the best-performing configuration -- interpretability is not being
paid for with quality here.

**Ruled out:** context width (E12, E26), rule order (E12, E15), Gaussian antecedents (E9/E10),
symmetric decode similarity (E11), Cython and BLAS threading (E14, E23.3), raw score normalisation
(E18.2), candidate-side lexicalisation (E19), the linguistic parameter space (E22), thread
parallelism (E23.3), OOV quality as the ceiling (E24.2), open-class rule quota (E25.1),
width-aware false-discovery control and relational slots (E27), candidate-linear first-order
consequents (E28.1, on argument), **held-out class selection (E29.1) and hierarchical parent
backoff (E29.2 -- the parents are already in the mixture, so backoff adds redundancy)**.

**Confirmed:** vectorising beat parallelising 11x (E23.2); the base holds the right categorical
grammar (E25.1, E27.3); long-range context is real but redundant (E27.2); the consequent's shape
was the ceiling (E28); **class estimation is data-bound and scale now favours the fuzzy model
(E29.3), inverting E23.4's conclusion, which was a fact about the zero-order model only**.

**Corrected along the way:** 0.569 balanced accuracy (E12.1); all pre-E17 ranking numbers; the
trigram baseline (E20.2); E19.4's mixture gain quoted without budget dependence (E22.4); E22.3's
fit-cost attribution (E23.1); the recommendation to push corpus size (E23.4, now inverted for
first-order); E24.3's "all rules are closed-class" (E25.1); a per-condition test set caught by a
moving control (E26.2); E26.1's dilution mechanism (E27.2); E28's standalone numbers, an
undersmoothing artefact (E28.3); **E29.6's first grammaticality metric, which could not
discriminate real text from a unigram**.

**Open.** Global coherence: samples are locally plausible and globally incoherent, and a 2-token
antecedent cannot carry discourse state. Class count is capped at 3,000 by `max_classes` and
quality was still improving there. Order-3 context classes untested. GPT-2 untested (blocked, and
a data-gap comparison anyway). No neural-embedding comparison. Experiment B's real encoder and SST
paths unrun.

---

## E30 — Plan: training speed. **Not Cython, not GPU** — and item 1 is already done (4.0x)

E29's full-data run took ~940s (16 min). The question is what to do about it. Profiled first,
because this project's record on cost attribution is bad (E22.3 blamed the wrong stage, E23.1
found it) and because the two previous speed fixes were **algorithmic, not parallel** — 43x from
vectorising a rollup against 3.85x from four processes.

### E30.1 Where the time actually went — and the fix, measured

Profile of `ContextClassMiner.fit` at 60,000 positions (total 25.7s):

| stage | time | share |
|---|---|---|
| per-pair column products `Fa[:,a] * Fa[:,b]` | 10.29s | 40% |
| `column_stack` of those columns | 8.81s | 34% |
| `np.add.at` scatter | 5.20s | 20% |
| collecting `F` (the Python loop) | **0.22s** | **1%** |

**94% in one operation**, and 1% in the Python loop I would have blamed. The counting problem is a
three-way contraction:

```
C[(a,b), w] = sum_i  F[i,a] . F[i,b] . 1[w_i = w]
```

Forming one column per pair is the wrong *shape*. Hold one seed `a` fixed and the whole row of
pairs `(a, b)` for every later `b` becomes one scaled block times a sparse one-hot matrix:

```
G = F[:, later_seeds] * F[:, a]      one block multiply
C = (Y.T @ G).T                      one sparse GEMM
```

Identical arithmetic, ~140 block operations instead of ~9,730 slice multiplies plus 17 giant
`column_stack` allocations, and the scatter-add disappears because the counts fall out of the GEMM.
This is the E14 reformulation applied to the other half of the pipeline.

**Measured, with the model asserted identical:**

| positions | before | after | speedup |
|---|---|---|---|
| 60,000 | 26.9s | **6.7s** | 4.0x |
| 300,000 | 179.7s | **40.8s** | 4.4x |
| 624,325 (projected) | ~940s | **~200s** | ~4.5x |

`_pair_blocks_reference` is kept and a test asserts the two produce the same classes and the same
distributions — this step produces every class's counts, so an error would silently change every
rule in the model.

### E30.2 The remaining plan, in priority order

Re-profiled after the fix (60,000 positions, total 6.7s): `ravel`/copy 35%, the block multiply 31%,
the sparse GEMM itself 18%, order-1 `add.at` 4%, collecting `F` 5%.

**2. Row-restrict each pair block to the rows where the fixed seed fires.** `F` is **1.5% dense**,
so `G = F[:, later] * F[:, a]` is a dense array that is almost entirely zero, and the GEMM
multiplies those zeros. For a fixed `a`, only rows where `F[i,a] > 0` can contribute:

```
idx = nonzero(F[:, a]);  G = F[ix_(idx, later)] * F[idx, a][:, None];  C = (Y.T[:, idx] @ G).T
```

Exact, not approximate. Expected the largest remaining win — proportional to how sparse the fixed
seed's column is, which for most context features is a few percent. This also subsumes "store `F`
sparse", and it is the memory fix: `F` at 624K positions is 2.6 GB in float64 and is the reason
the run needs care at all.

**3. float32 for `F` and the count matrices.** Halves bandwidth and memory, and the profile is now
dominated by copies rather than arithmetic. Precedent: E26 added `dtype` to the rule learner for
exactly this reason, with float64 kept as the default so the exactness test still holds.

**4. Process-parallel over row shards** — *only after 2 and 3*. Counting is a sum over rows, so
sharding rows and adding the partial count matrices is **exact**, not an approximation. Measured
ceiling on this box: **3.85x on 4 cores, bit-exact** (E23.3, on featurisation). Worth doing last
because it multiplies whatever the serial cost is, and items 2–3 shrink that first.

### E30.3 Why not Cython

**The profile shows essentially no time in scalar Python loops.** Collecting `F` — the only real
Python loop left — is 5% of 6.7s, and most of that is numpy call overhead, not interpretation.
Cython would compile a loop that should be *deleted* instead. This is the same conclusion as E14
("the work is already dense linear algebra dispatched to an optimised BLAS; hand-writing the loop
in C would at best match one thread of it"), now re-derived for the first-order pipeline rather
than assumed from the old one.

### E30.4 Why not a GPU

Four reasons, in order of weight:

1. **It contradicts the project's goal.** The stated aim is a model that trains on local compute
   without expensive servers. A GPU requirement is a worse outcome than a slower model.
2. **The work is memory-bound, not compute-bound.** BLAS threading on these matrices was measured
   worth *nothing* (5.15s at 1 thread vs 5.05s at 4, E23.3), and the post-fix profile is 35%
   copies. GPUs win on compute-bound dense math; they do not fix a bandwidth problem, they move it
   across a PCIe bus.
3. **There would be nothing left to accelerate.** Item 1 already takes full-data training to ~200s.
   Items 2–4 plausibly reach ~30–60s. A GPU cannot improve a one-minute job that a user runs a
   handful of times.
4. **It cannot be tested here**, so any claim about it would be unverified — and this log has
   enough of those already.

**When a GPU would become the right answer:** if the vocabulary grew such that the
`(classes x vocab)` count matrix dominated — that *is* a large dense GEMM and would map well — or
if the corpus grew ~100x, past where a single machine's memory holds `F` in any layout. Neither is
true at 1M tokens and a 3,000-word candidate set.

### E30.5 Expected end state

| | now | after items 2–4 (estimated) |
|---|---|---|
| 60,000 positions | 6.7s | ~1–2s |
| 624,325 positions (full corpus) | ~200s | ~30–60s |
| peak memory at full data | ~2.6 GB | ~0.3 GB |

Estimates, flagged as such. Item 1 is the only measured number in this section.

81 tests pass (1 new: the fast pair blocks are compared against the per-pair reference on both the
counts and the held-out columns).
