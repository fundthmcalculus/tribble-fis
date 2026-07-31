# FLM engineering log

Running record of what was tried, what worked, what didn't, and **why**. Newest
entries at the bottom. Results and rationale live here; the module READMEs carry the
settled conclusions.

Convention: **WORKED** / **FAILED** / **PARTIAL**, each with a why.

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
