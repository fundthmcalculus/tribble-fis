# `fuzzyembed` — Experiment A, built

A text embedding whose coordinates are **membership degrees in named nodes of a
lexical hierarchy**, plus a first fuzzy sequence model and fuzzy decoder on top.

Design rationale: [`../FUZZY_EMBEDDING_PLAN.md`](../FUZZY_EMBEDDING_PLAN.md).
Prior art: [`../literature/FLM_LITERATURE_REVIEW.md`](../literature/FLM_LITERATURE_REVIEW.md).

```bash
# one-time data fetch (from the nltk_data GitHub mirror)
uv run --with nltk python -c "import nltk; [nltk.download(p) for p in \
  ['wordnet','gutenberg','brown','opinion_lexicon']]"

uv run --with nltk python -m flm.fuzzyembed.run_flm --stage all
uv run --with nltk python -W ignore -m pytest flm/tests -q
```

---

## Two substitutions forced by the environment

| Planned | Used | Why |
|---|---|---|
| Roget's Thesaurus | **WordNet** | Open Roget's is on sites.google.com, the 1911 edition on gutenberg.org — both outside the egress allowlist. WordNet is reachable via the `nltk_data` GitHub mirror. |
| TinyStories | **children's narrative prose** — `bryant-stories` + `burgess-busterbrown` + `carroll-alice` (~478K chars) | TinyStories is Hugging Face only, and HF is blocked. This is the closest reachable register. `corpus.load_local()` takes a real TinyStories JSONL when you have one. |

Both substitutions are pluggable: write another `build_*_hierarchy` returning a
`FuzzyHierarchy`, or point `--corpus` at a file.

WordNet turned out closer to Roget's than expected in one respect and worse in two.
Its **45 lexicographer files** play the role of Roget's 39 Sections almost exactly.
But it has no antonymous opposed pairs (so polarity must be attached externally), and
— the sharp one — **no adjective hypernyms at all**, which breaks the ladder (below).

## Results

Every number below was produced by `run_flm.py` in this repo.

### M0 — coverage gate: **PASS**

| corpus | content-token coverage | verdict |
|---|---|---|
| children's narrative | **96.7%** | PASS (gate: ≥85%) |
| Brown (news+fiction+romance) | **94.7%** | PASS |

Two rounds of fixes got here from 92.6%/91.4%: contraction expansion in the tokenizer
(`don't`, `i'll`, `brown's` were the top "uncovered" types — a tokenization artifact,
not a lexicon gap) and adding reflexives/indefinites to the closed-class list. The
residual misses are now **almost entirely proper names** (`alice`, `joe`, `margery`,
`william`), which is precisely the named-entity gap the plan predicted and the reason
the Wikipedia-category graft is mandatory rather than optional.

### The resolution ladder — and its measured flaw

```
L0:     1     *
L1:     4     NOUN VERB ADJ ADV
L2:    45     noun.animal, verb.motion, adj.all, ...   <- the workhorse
L3:  4527
L4:  7261
L5:  9864
```

**The ladder is badly unbalanced: L2→L3 is a ~100× jump.** Two causes, both
structural: same-supersense hypernym chains are short, and WordNet organises
adjectives by antonymy rather than subsumption so **every adjective and adverb clamps
at depth 3**. A uniform ladder over all parts of speech is therefore impossible here.
Roget's designed 5-level tree (6/39/79/596/990) would not have this problem — this is
the sharpest measured cost of the substitution.

### Exact multi-resolution readout — **PASS**

The claim that separates this from Matryoshka truncation: every coarse level is an
*exact* t-conorm rollup of every finer level. Asserted in CI
([`../tests/test_fuzzyembed.py`](../tests/test_fuzzyembed.py), 18 tests) on both a
hand-built tree and the real WordNet hierarchy, and re-checked by `run_flm`.

The invariant it rests on is **prefix consistency**: every prefix of a terminal's path
must be the registered path of that ancestor. This is why the hypernym chain is
restricted to same-supersense ancestors — not for elegance. Dropping the restriction
was tried and it broke exactness, because `dog.n.01` is `lex:noun.animal` while its
ancestor `entity.n.01` is `lex:noun.Tops`, so `entity`'s canonical path is not a
prefix of `dog`'s. The unrestricted chain gives a better-balanced ladder
(`[1,4,45,3245,4582,5804]`) and a *void* central claim. Correctness won.

### Semantic similarity — L2 is the discriminative level

Per-level fuzzy Jaccard, similar vs unrelated pairs:

| pair | | L1 | **L2** | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| happy child / joyful boy | sim | .333 | **.333** | .000 | .000 | .000 |
| dog barked / wolf howled | sim | .581 | **.235** | .016 | .004 | .000 |
| girl ate bread / boy ate food | sim | .974 | **.407** | .300 | .270 | .267 |
| dog barked / king spoke | dif | .453 | **.062** | .033 | .018 | .001 |
| happy child / stone was cold | dif | .697 | **.187** | .000 | .000 | .000 |
| girl ate bread / mountain tall | dif | .421 | **.000** | .000 | .000 | .000 |

L1 (4 dimensions) is too coarse to separate anything; L3+ is so sparse that
near-synonyms share no coordinate at all. **L2 is the only level that orders the pairs
correctly** — confirming the plan's prediction that the supersense width is the design
centre, and it was a prediction, not a post-hoc rationalisation.

At L2: **mean similar = 0.344, mean unrelated = 0.066, gap +0.278**, with complete
separation. This also corrected `hierarchy_jaccard`, which originally decayed weights
from the *finest* level and so put full weight on the least informative resolution.

### Typo robustness

Clean vs perturbed, fuzzy Jaccard at the finest level:

| clean | perturbed | similarity |
|---|---|---|
| the rabbit ran home | the ra**b**it ran home | **0.967** |
| the rabbit ran home | the rab**bb**it ran **hoem** | **0.582** |
| a happy child laughed | a happy **chidl** laughed | **0.992** |

And every match is explainable, which is the part no shaped-subword-loss approach can
offer:

```
'rabit'      -> rabbit @ 0.99 (phonetic key agrees; high trigram overlap)
'littel'     -> little @ 0.58 (phonetic key agrees; frequent word)
'beautifull' -> beautiful @ 1.00 | beautifully @ 0.98 (honest ambiguity)
'mothre'     -> mother @ 0.67 (phonetic key agrees; keyboard-adjacent (fat-finger))
'xyzzyqq'    -> no lexeme match (out of vocabulary)
```

The aggregator is itself a fuzzy classifier over five similarity channels. Held-out
**separation +0.70** (positive-class degree minus negative-class degree). The
generalisation diagnostic — train on substitute/delete/double, test on
transpose/insert — drops to **+0.37**: transposition lowers trigram overlap while
keeping edit distance at 1, so channels learned on other error classes transfer
imperfectly. The shipped model trains on all five ops; the split exists to *measure*
that gap, not to hobble the model.

## Modules

| file | role |
|---|---|
| `hierarchy.py` | WordNet → named node tree, level projection, exact rollup |
| `corpus.py` | children's-narrative loader, tokenizer (contraction-aware), local/JSONL loader |
| `coverage.py` | **M0 gate** — content-token coverage with a go/no-go verdict |
| `lexical.py` | Stage 1: BK-tree candidates + 5 channels + fuzzy-classifier aggregator |
| `senses.py` | Stage 2: lexeme → node degrees; SemCor priors, antonyms, context relaxation |
| `embedder.py` | Stage 3: OWA aggregation, hedges, upward closure, explanations |
| `similarity.py` | fuzzy Jaccard / Dice, hierarchy-aware variant |
| `syntax.py` | **fuzzy syntax** — named closed-class categories + graded open-class markers |
| `rules.py` | **membership rule learner** — TSK for inputs that are already memberships |
| `sequence.py` | fuzzy sequence model — next-token prediction over the joint space |
| `decode.py` | fuzzy decoder — Zadeh linguistic approximation |
| `run_flm.py` | driver: `--stage coverage|embed|sequence|generate|all` |

## The rule learner, and why TRIBBLE's estimators do not fit here

Measured on identical features and splits, predicting "does category C come next?":

| target | logreg AUC | FIS/gaussian | FIS/trapezoid | **rules** |
|---|---|---|---|---|
| OPEN_NOUN | 0.664 | 0.488 | 0.500 (const) | 0.643 |
| OPEN_VERB | 0.745 | 0.511 | 0.500 (const) | 0.692 |
| DETERMINER | 0.742 | 0.515 | 0.500 (const) | **0.747** |
| PREPOSITION | 0.776 | — | — | **0.784** |

**Why the FIS fails.** The feature matrix is **93% zeros**. TRIBBLE fits a Gaussian per
`(feature, class)`, which presumes a continuous, unimodal, well-spread variable — true
of its benchmark data (concrete strength, turbine power, wine chemistry), false of a
sparse membership vector. Fit a Gaussian to `{0 w.p. .95, 1 w.p. .05}` and both classes
get the same narrow near-zero curve; the t-norm product then collapses to 0.5. This is a
finding about TRIBBLE, not just this experiment.

**The insight.** When the inputs are already membership degrees, **there is no membership
function to fit — the input value *is* the membership degree.** The antecedent-fitting
layer is redundant. `rules.py` uses the input values directly as firing strengths and
learns only rule structure and consequents.

**And the rules are linguistically correct**, recovered from data:

```
IF prev1:DETERMINER                    THEN next[OPEN_NOUN] ~ 0.675   (default 0.364)
IF prev1:adj.all AND prev2:DETERMINER  THEN next[OPEN_NOUN] ~ 0.788
IF prev1:POSSESSIVE                    THEN next[OPEN_NOUN] ~ 0.754
IF prev1:OPEN_VERB                     THEN next[OPEN_NOUN] ~ 0.243
```

Determiners precede nouns; "the red ___" is a noun; "my ___" is a noun; after a verb a
noun is less likely. English noun-phrase syntax, readable, at parity with a linear
model. Interpretability is *better* than the Gaussian version — an antecedent is
literally "the previous token is a determiner, to degree 0.9", with no fitted centre or
width to explain.

## Decoder: symmetric similarity was the wrong asymmetry

With `t[OPEN_NOUN]=0.44` the top candidates were `jolly`, `fourth`, `in` — not nouns.
Symmetric fuzzy Jaccard rewards words whose pattern resembles the *marginal
distribution*, i.e. bland words with membership spread thinly everywhere, and penalises
the pure noun the prediction asked for. Replaced with an asymmetric **coverage** score,
`Σ min(t,w) / Σ w`: how much of *this word's own* mass sits in the predicted categories.
(`similarity.py` still uses symmetric Jaccard for sentence-to-sentence comparison, where
the symmetry is correct.)

Pure-category retrieval after the fix — all correct:

```
OPEN_NOUN   -> mile, chimney, fame, paris, street, hen, city
OPEN_VERB   -> declare, argued, doubted, remarked, remarking, petted
OPEN_ADJ    -> alive, headstrong, hateful, harsh, famous, hotter
DETERMINER  -> those, these, the, another, this, an, no
PREPOSITION -> from, of, upon, through, in, since, during
AUXILIARY   -> must, can, were, did, am, shall, are
```

## Design corrections found by running it

Recorded because each was a wrong idea in the plan, not a coding slip.

**1. Zadeh's complement is wrong for negation here.** The plan said `not X → 1 - μ`,
one line, no training. In practice a lexeme's sense vector is sparse and peaked —
`happy` is `happy.a.01@1.0, felicitous@0.07, glad@0.01` — so the complement zeroes the
intended sense and promotes its weak siblings to ~0.95. *"not happy"* came out
asserting **strongly felicitous and glad**. The complement of a sparse membership
vector is a dense vector of near-ones, which is meaningless.

Negation is now **suppression + antonym transfer**: zero the concept, move its mass to
a WordNet antonym. `not very happy` → `unhappy = 1.00`. This reintroduces exactly the
structure Roget's opposed pairs would have supplied natively. The *hedges*
(concentration/dilation) were fine as specified — it is only the complement that
breaks, and the docs now say so precisely.

**2. Frequency is a prior, not a similarity channel.** Candidate corpus frequency
started as a sixth FIS input. Inside a t-norm product an atypical frequency value can
*veto* an obviously-correct match: `littel → little` scored **0.003** with features
otherwise indistinguishable from `freind → friend` at **0.605**, because `little` sits
at the sparse top of the frequency distribution and fell in the tail of that channel's
Gaussian. Frequency now modulates the FIS output from outside
(`degree = sim · (1 − w + w·freq)`, w=0.3), so it can re-rank but never gate. A
related fix: training words are now sampled proportional to log frequency, because
uniform sampling of a Zipfian vocabulary draws almost only rare words.

**3. Closed-class filters must agree across modules.** After the decoder metric fix,
`OPEN_NOUN` still retrieved `somebody`, `o`, `t`. `SyntaxTagger` read `lemma_synsets`
directly without the function-word filter `SenseAssigner` applies, so closed-class words
got a pure `OPEN_NOUN=1.0` and nothing else — and because the coverage metric normalises
by a word's own mass, those single-coordinate vectors scored a perfect 1.0 and topped
every retrieval. An inconsistency between two modules, surfaced only by the new metric.

**4. Interior nodes must not be dimensions.** Levels were originally derived from
every registered node. Since short paths clamp (which is what makes rollup exact),
every ancestor clamped into all deeper levels as an extra near-always-on coordinate —
the root worst of all, appearing at every resolution carrying the max of everything
and inflating similarity denominators for nothing. Levels now derive from *terminal*
paths only.

Also worth noting: `wn.synsets("was")` returns **WAS = Washington**, and it takes the
top sense prior, so *"the rabbit was happy"* acquired a strong `administrative
district` dimension. Function words are now filtered.

## Fuzzy sequence model and decoder

**Sequence** (`sequence.py`): windowed level-2 memberships of the previous *k* tokens →
membership vector of the next. Every input and output dimension is named, so a rule
reads `IF prev1[noun.animal] is High AND prev1[verb.motion] is Low THEN
next[verb.motion] ≈ 0.7` — a checkable statement about language.

One fuzzy **classifier** per output dimension, not `MimoGaussianPredictor`. The
regressor path quantile-bins its target via `partition_output`, and a membership
coordinate is mostly zero, so `pd.qcut` raises on duplicate bin edges. Binarising and
reading `predict_proba` is also the better semantic fit: "does the next token belong to
supersense S, and to what degree?" *is* a graded membership question.

### It does not work yet — a negative result

```
windows=2500  features=90  outputs=8/45
binarised task : separation=+0.015  balanced-acc=0.528   (chance=0.500)
continuous MAE : test=0.4295        mean-baseline=0.1493  -> LOSES TO baseline
```

**Barely above chance.** Stated plainly because it is the most informative result in
this directory. Both metrics are reported because they disagree and each alone would
mislead: a probability calibrated near 0.5 cannot beat a mean baseline on MAE when
targets are sparse, so MAE understates the model — but separation of +0.015 confirms
there is genuinely almost no skill to understate.

The likely cause is not a hyperparameter. **The representation that works for
similarity is structurally insufficient for sequence prediction**, because what
predicts the next word is mostly syntax and function words — exactly what this
embedding discards. A two-token window of supersense memberships says "an animal was
just mentioned"; it cannot say "a determiner was just used, so a noun is due".

That makes this the sharpest available evidence for the scope claim in
[`../FUZZY_EMBEDDING_PLAN.md`](../FUZZY_EMBEDDING_PLAN.md) §8: a fuzzy language model
needs a fuzzy model of **syntax** beside its fuzzy model of **semantics**. The decoder
below is fully working machinery; it is currently being fed a near-uninformative
signal. Before tuning this, the productive next step is adding syntactic features
(POS-tag memberships, position, function-word identity) as additional named inputs —
they are cheap, and they are what is missing.

**Decoder** (`decode.py`): Zadeh's **linguistic approximation** — the retranslation
step that closes the computing-with-words loop. Score every vocabulary lexeme by fuzzy
Jaccard against the predicted vector and sample. Two consequences worth the
architecture:

- **Temperature is a Zadeh hedge.** Sharpening is concentration (`μ^e`, e>1),
  flattening is dilation (e<1). No separate temperature parameter — `e = 2` is
  literally "*very* like the prediction".
- **Every decode step is auditable**: which named dimensions drove it, which lexemes
  competed, at what degree. No vocabulary-sized opaque distribution.

### What this is not

**This is a semantic-class sequence model, not a language model.** It predicts what
*kind* of thing comes next. Function words carry no hierarchy membership, so they can
never be emitted and the output is not grammatical text. That is a property of the
representation, not a tuning problem: a real FLM needs a fuzzy model of *syntax*
alongside this fuzzy model of *semantics*, and that does not exist yet. See
[`../FUZZY_EMBEDDING_PLAN.md`](../FUZZY_EMBEDDING_PLAN.md) §8.

## Honest limitations

- **Proper names are invisible** — no named entities in WordNet. The dominant residual
  coverage gap; needs the Wikipedia-category graft.
- **The ladder has no usable middle** (45 → 4527). Structural to WordNet; Roget's fixes it.
- **Word-sense disambiguation is shallow** — SemCor priors plus one relaxation round.
  Misspelled tokens inject competing senses (`rabit` pulls in `habit`, `wait`), and
  though the correct sense stays top-ranked, the noise is real.
- **The corpus is ~50× smaller** than a TinyStories subset and early-20th-century.
- **Typo evaluation partly circular** — the aggregator trains on `exp_b/perturb.py`
  noise and is evaluated on it. The held-out *error class* split mitigates this and
  quantifies the gap; published numbers should use AdvGLUE-style generators instead.
- **No comparison to a neural baseline yet.** The plan's distillation and MTEB
  evaluation (§4.1, §6) are not built. Until then there is no evidence about where
  this sits against EmbeddingGemma — and the expectation on record is that it *loses*
  on clean STS and wins on typo robustness, interpretability, and cost.
