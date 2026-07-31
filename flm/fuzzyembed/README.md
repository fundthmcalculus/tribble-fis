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
([`../tests/test_fuzzyembed.py`](../tests/test_fuzzyembed.py), 24 tests) on both a
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
| `sequence.py` | fuzzy sequence model — per-dimension marginal prediction |
| `joint.py` | **joint next-token ranker** — scores (context, candidate) pairs; the version that works |
| `decode.py` | fuzzy decoder — Zadeh linguistic approximation (marginal pipeline) |
| `generate.py` | **fuzzy language model** — generation + perplexity from the joint ranker |
| `baselines.py` | n-gram LMs on identical data, and a GPT-2 path for machines with HF access |
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

### The marginal sequence model does not work — and why that mattered

```
binarised task : separation=+0.015  balanced-acc=0.528   (chance=0.500)
```

Essentially no skill. Two independent changes fixed part of it — named **syntactic**
features (`syntax.py`), since what predicts the next word is mostly syntax and function
words that the semantic embedding discards; and the right **antecedent representation**
(`rules.py`), since a control showed logistic regression reaching AUC 0.66–0.78 on
identical features where the Gaussian FIS sat at chance. Neither worked alone
(0.528 → 0.534 with syntax only), which is why testing them in isolation would have given
the wrong conclusion.

But the deeper flaw was the *target*: predicting each dimension independently never
enforces that the prediction describes **one actual word**. That is what `joint.py` fixes.

## Joint next-token ranking (`joint.py`)

Scores `(context, candidate)` pairs — "is *w* the next word here?" — so a rule can express
the context×candidate interaction the marginal form cannot. Evaluated by ranking the true
next token against frequency-sampled distractors.

| model | MRR | hits@1 | hits@10 |
|---|---|---|---|
| unigram frequency | 0.189 | 0.068 | 0.486 |
| joint, 3000 train positions | 0.279 | 0.126 | 0.660 |
| **joint, 12000 positions** | **0.319** | **0.142** | **0.784** |
| oracle ceiling | 0.822 | 0.698 | — |

> ⚠ **These ranking numbers are optimistic.** Measured before `Corpus.split` existed:
> `build` and `evaluate` both iterated the full sentence list under different shuffles, so
> held-out *positions* came from training sentences. "Different seed" is not a split. Fixed
> with a test. The perplexity numbers below use a proper sentence-level split.

Two non-obvious fixes were required — the first run scored MRR 0.191, *below* unigram, with
every rule candidate-only:

1. **`must_include`** — every rule must touch a candidate feature. A context-only rule fires
   identically for all candidates, shifting every score by a constant, so it cannot change a
   ranking.
2. **`seed_features`** — context features have **exactly zero marginal lift** here (a
   position's positive and all its negatives share one context vector), so lift-based greedy
   seeding could never generate a ctx×cand rule. The XOR problem for feature selection:
   informative jointly, invisible marginally.

The **oracle ceiling was measured first** (2133 distinct signatures among 2897 words), which
disproved the standing hypothesis that L2 was too coarse to rank words and saved an expensive
detour into re-decoding at finer levels.

### Order-3 rules and beam

| config | MRR | hits@1 | rules |
|---|---|---|---|
| order 2, beam 200 | 0.265 | 0.116 | 259 |
| **order 2, beam 800** | **0.279** | **0.126** | 859 |
| order 3, beam 800 | 0.275 | 0.118 | 1659 |

**More interactions help; order-3 does not** — double the rules for no MRR gain. Beam
saturates at 800 because the *candidate supply* runs out (~835 admissible order-2 rules).
Order-3 does buy better *explanations* at no ranking cost, so use it when the rule base is
the deliverable: `"and he ___" → AUXILIARY (0.473)`, `"began ___" → INFINITIVE_TO (0.456)`.

### Corpus size is the binding constraint

| training positions | MRR | hits@1 | hits@10 |
|---|---|---|---|
| 750 | 0.260 | 0.112 | 0.614 |
| 6000 | 0.293 | 0.140 | 0.704 |
| 12000 | **0.319** | **0.142** | **0.784** |

Monotonic and still rising when the ~90K-token corpus runs out. Rule count saturates at ~860
throughout, so the gain is better-estimated **consequents**, not more rules. Replicated on
Brown (191K tokens), which goes twice as far: 0.253 → 0.273 → 0.285 at 25000 positions.

### Efficiency: GEMMs, not Cython

Under the product t-norm a whole growth level factors into matrix products over the frontier
firing matrix `F` and seed columns `S`:

```
support = F.T @ S ;   weighted = F.T @ (S * y)   →   consequent = weighted / support
```

The old code made one numpy call per candidate — ~30k per level at beam 200, where *call
overhead* dominated. Cython would at best match one BLAS thread. Order-3 at beam 800 went
from "cannot finish a sweep" to **0.95s**. `min` is not bilinear so it keeps the loop;
`test_batched_growth_matches_bruteforce` asserts exact agreement.

## As a language model (`generate.py`, `baselines.py`)

Normalising the joint scores over the vocabulary gives `p(w | context)`, which makes
**perplexity** available and therefore makes this comparable to any conventional LM on
identical held-out text.

| model | perplexity |
|---|---|
| uniform (floor) | 2897.0 |
| 1-gram (same data) | 472.9 |
| fuzzy, raw score normalised | 2477.8 |
| fuzzy, NCE-corrected | 385.8 |
| 3-gram (same data) | 370.2 |
| **fuzzy + context lexicalisation (K=500)** | **363.4** |
| 2-gram (same data) | 279.2 |

Same sentence split, same vocabulary, same restriction to positions whose gold token the
fuzzy model can represent (94.5% coverage; the rest skipped, not floored).

### Raw score normalisation was badly wrong — and it reframes the ranking results

Normalising raw scores gave **2477.8** against a 2897 floor: almost no information. Scores are
bounded in [0,1] and span under one order of magnitude, while an LM needs ratios of ~10³. The
fix is the noise-contrastive inversion — the ranker was trained contrastively, so its output
is `s = p/(p + k·q)`, giving `p ∝ q·s/(1−s)`. The odds ratio is unbounded, and multiplying by
`q` restores the frequency information the contrastive objective factored out. **6.4×.**

The gain over unigram is *provably* the rules' contribution: a constant score makes the
inversion reduce **identically** to the frequency prior, so 473 → 386 is exactly what context
adds. There is a test for that identity.

This also reframes the ranking table: **MRR flattered the model.** Ranking against 19
distractors mostly rewards getting the *category* right; perplexity over 2897 candidates
demands the *word*.

### Closing the bigram gap: lexicalise the context, not the candidate

The model was effectively a *category* bigram — `p(cat(w) | cat(prev))` where a real bigram
has `p(w | prev)`. English is Zipfian, so the head of the distribution is a few hundred words
belonging to no useful category (`the`, `of`, `said`): **lexicalise the head, generalise over
the tail** (`lexeme_top_k`).

Adding identity to *both* halves made it monotonically **worse** (385.7 → 394.6 → 399.0). Not
because the features are useless but because they are **redundant with a transform applied
downstream**: the NCE inversion already multiplies by `q(w)`, so `cand:=the → high` re-learns
frequency and the two compound. Context-side identity has no such overlap.

| config | ppl (1500/12000) | ppl (600/8000) |
|---|---|---|
| no lexeme features | 385.7 | — |
| top-200, **candidate** side | 399.0 | 365.1 |
| top-200, **context** side | 364.5 | 339.1 |
| **top-500, context side** | **363.4** | **338.5** |

Same features, same budget — only which half carries them. Saturates after ~200 dimensions.
**425 of 860** rules end up lexicalised, chosen on lift.

### The fuzzy model is complementary to a bigram, not strictly worse

The most important result here. Interpolating `p = λ·p_fuzzy + (1−λ)·p_bigram`:

| λ | ppl (1500/12000) | ppl (600/8000) |
|---|---|---|
| 0.0 (pure 2-gram) | 276.1 | 236.9 |
| **0.30–0.35 (best)** | **263.1** | **228.1** |
| 0.50 | 266.8 | 233.6 |
| 1.0 (pure fuzzy) | 364.0 | 338.5 |

**The mixture beats the bigram alone in both settings (−4.7%, −3.7%) even though the fuzzy
model loses head-to-head by ~30%.** An interior optimum near λ=0.3 in two independent
settings is not a fluke, and losing individually while improving the ensemble is the
signature of complementary information rather than a strictly-dominated model.

**This is the honest framing of the whole line of work:** fuzzy rules do not replace an
n-gram LM at this scale — they capture something n-grams miss, in a form a human can read,
worth about a third of a mixture's weight.

Probably why, from the learned rules:

```
IF ctx:prev1:=the    AND cand:OPEN_NOUN     THEN P(next) ~ 0.313   (support=1731)
IF ctx:prev1:=did    AND cand:NEGATOR       THEN P(next) ~ 1.000
IF ctx:prev1:=came   AND cand:INFINITIVE_TO THEN P(next) ~ 1.000
IF ctx:prev1:=said   AND cand:DETERMINER    THEN P(next) ~ 0.554
IF ctx:prev1:=of     AND cand:DETERMINER    THEN P(next) ~ 0.325
```

A bigram encodes "did → not" as a table entry. Here it is a word paired with a *category*, so
it generalises to negators never seen after "did" — information a bigram structurally cannot
hold.

### On GPT-2

Not runnable here: weights are on Hugging Face, outside the egress allowlist.
`baselines.gpt2_perplexity` implements it under the same restriction. Note the comparison is
weak on its own — GPT-2 saw ~40GB against this model's ~90K tokens, so it wins by a margin
measuring the *data gap*. The controlled baseline at this scale is an n-gram LM on the same
corpus, which is what the tables report, and n-grams on 90K tokens are strong rather than a
straw man.

### Generation is still not grammatical

Function words now appear at realistic rates (`the` 0.165, `and` 0.118, `to` 0.094) instead
of the earlier category salad, but output is not fluent. **Temperature is a Zadeh hedge** —
sharpening is concentration (`μ^e`), so `e=2` is literally "*very* like the prediction", and
every step reports the named rules that drove it.

## What this is not

An embedding model plus a classifier head is **not a language model.** A fuzzy LM needs a
fuzzy representation (built), a fuzzy sequence model (built, weak), and a fuzzy decoder
(built). Note `MixtureOfGaussiansFuzzySequenceClassifier` in this repo is a confusion-driven
cascade, not a temporal sequence model, and is not a seed for the sequence stage.

## Honest limitations

- **Proper names are invisible** — no named entities in WordNet. The dominant residual
  coverage gap; needs the Wikipedia-category graft.
- **The ladder has no usable middle** (45 → 4527). Structural to WordNet; Roget's fixes it.
- **Loses to a bigram by ~30%** head-to-head, and only complements it.
- **hits@1 caps near 0.14** — supersense rules cannot separate `bread` from `cake`.
- **Corpus is ~50× smaller** than a TinyStories subset, and data is the binding constraint.
- **Word-sense disambiguation is shallow** — SemCor priors plus one relaxation round.
- **Typo evaluation partly circular** — trained and evaluated on `exp_b/perturb.py` noise;
  the held-out *error class* split quantifies the gap but published numbers should use
  AdvGLUE-style generators.
- **No neural-embedding comparison** — the plan's distillation/MTEB work is not built.

Full experiment record with what worked, what didn't, and why: [`../LOG.md`](../LOG.md).
