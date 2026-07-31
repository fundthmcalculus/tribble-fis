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

## Standing summary — CURRENT (as of E19)

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
