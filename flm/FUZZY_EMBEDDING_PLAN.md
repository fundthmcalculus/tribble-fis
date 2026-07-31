# Experiment A — A Fuzzy Embedding Model

**Goal.** A text embedding whose coordinates are **membership degrees in named nodes of a
curated lexical hierarchy**, so that (i) every dimension has a name before training
begins, (ii) dimensional resolution can be raised or lowered by an *exact* aggregation
rather than a lossy truncation, and (iii) subtle misspellings are absorbed at a *fuzzy
lexical access* layer instead of shattering the tokenization.

Prior art and novelty positioning: [`literature/FLM_LITERATURE_REVIEW.md`](literature/FLM_LITERATURE_REVIEW.md).
The companion, much cheaper experiment is [`FIS_ON_EMBEDDINGS_PLAN.md`](FIS_ON_EMBEDDINGS_PLAN.md);
run that one first — it is the control that tells you whether A is worth building.

---

## 1. Definition

For a text span `s` and hierarchy node set `C`, the embedding is

```
E(s) = ( μ_c(s) )_{c ∈ C},        μ_c(s) ∈ [0, 1]
```

a vector of membership degrees, one per node. This is not a vector in a linear space and
must not be treated as one — see §6.2 on why cosine similarity is the wrong metric here.

Two structural constraints make this a hierarchy rather than a flat sparse vector:

**(C1) Fuzzy subsumption.** For every node `p` with children `ch(p)`:

```
μ_p(s)  ≥  max_{c ∈ ch(p)} μ_c(s)
```

A concept cannot be *less* present than its most present specialization. This is the
graded generalization of a box-containment or entailment-cone partial order
(§3.4 of the review) — crisp order embeddings are the special case where memberships are
0/1.

**(C2) Sibling partition (per level, optional but strongly preferred).** For every `p`:

```
Σ_{c ∈ ch(p)} μ_c(s)  =  μ_p(s)
```

i.e. children form a Ruspini partition of the parent's mass. This repo already builds and
refines exactly this object: `src/tribblefis/ruspini.py` constructs triangular
partition-of-unity families from shared apex knots, and `refine.refine_ruspini_partition`
moves the knots while preserving the property for free. The hierarchy's per-level partition
is that same construction, one level up.

### 1.1 The resolution ladder — why this beats Matryoshka truncation

Under **(C1)** with a t-conorm rollup, or under **(C2)** with a sum rollup, the level-ℓ
readout is a *closed-form exact aggregation* of the level-(ℓ+1) readout:

```
E_ℓ(s) = rollup( E_{ℓ+1}(s) )        with rollup ∈ { max, ⊕_s-norm, Σ }
```

Reuse `gauss_math.t_conorm` for the s-norm variant so the operator matches the rest of the
inference stack.

Contrast with MRL (§4 of the review): truncating a Matryoshka embedding to 128 dims is
*empirically* graceful, but nothing states what the discarded 640 coordinates meant, and
the coarse view is not a proposition about the fine view. Here, dropping from 990 to 39
dimensions is a **named disjunction**: "Section 12 fires at 0.7" *because* "Head 648 fires
at 0.7 and its siblings are quiet". That is a stronger, checkable guarantee — and §6.3
specifies the check.

The ladder itself (verify cardinalities against the actual dump — see §3.1):

| Level | Roget's unit | Approx. dim | Intended use |
|---|---|---|---|
| 0 | Thesaurus | 1 | (degenerate) |
| 1 | Class | 6 | coarse routing / HME top gate |
| 2 | Section | 39 | **the workhorse FIS input width** |
| 3 | Sub-Section | 79 | mid-resolution |
| 4 | Head Group | 596 | fine |
| 5 | Head | 990 | full resolution |

**Level 2 is the design centre.** 990 inputs is far beyond what a flat FIS can take without
the rule count exploding — this repo already fights that with `take_top_features` and the
HME. 39 named inputs is comfortable. Drilling to level 4/5 happens only inside the subtree
a gate routed to. That is precisely `fuzzytree.HierarchicalFuzzyExpertsRegressor`, which
means **the embedding hierarchy and the inference hierarchy can be the same tree** (§7.3).

## 2. Architecture — three stages

```
surface tokens ──[1] fuzzy lexical access──▶ lexeme memberships λ_t(w)
               ──[2] sense assignment ─────▶ node memberships   μ_c(t)
               ──[3] aggregation + hedges ─▶ span embedding     E(s)
```

### 2.1 Stage 1 — fuzzy lexical access (misspelling resilience lives here)

For each surface token `t`, produce a fuzzy set over the vocabulary:

```
λ_t(w) = degree to which surface form t denotes lexeme w
```

`recieve` should come out ≈0.95 `receive` and ≈0 everything else. This is the single most
natural application of fuzzy sets in the whole pipeline — the *is-this-that-word* judgment
is genuinely graded, and existing approaches force it to be crisp (spell-correct then
tokenize) or implicit (shaped subword loss, MOE §6 of the review).

**Candidate generation** (must precede scoring — you cannot edit-distance 100k vocab per
token): a **SymSpell deletion-neighborhood index** or a **BK-tree** gets you to ~top-50
candidates in microseconds. Without this the layer is a toy.

**Scoring channels** — five cheap, interpretable, deliberately non-redundant features:

| Channel | Catches | Note |
|---|---|---|
| Normalized Damerau–Levenshtein similarity | transpositions (`teh`→`the`) | plain Levenshtein misses these |
| Character 3-gram Dice / Jaccard | internal scrambles | fastText-style subword signal, untrained |
| Keyboard-adjacency-weighted edit cost | fat-finger errors (`wrold`→`world` cheap, `brold`→`world` dear) | **the interpretability win — you can say *why*** |
| Double Metaphone key agreement | phonetic errors (`fone`→`phone`, `definately`→`definitely`) | will not catch `their`/`there` |
| Length-normalized common prefix | typos are empirically rarer word-initially | cheap prior |
| Identity / in-vocabulary flag | *guard* — see below | prevents over-correction |

**Aggregate these with a trained FIS, not a hand-tuned formula.** This is the recursive
punchline of the whole project: stage 1 is itself a 6-input TSK system, and
`MixtureOfGaussiansFuzzyRegressor` fits it directly. Target: 1 for (misspelling, correct
word) pairs, 0 for hard negatives drawn from the candidate set. The fitted rules are the
*explanation* of the model's own robustness — no prior work in §6 of the review can produce
one, because a shaped subword loss has nothing to report.

**Training data:** the public misspelling dataset released with MOE (Edizel et al. 2019),
plus Wikipedia's commonly-misspelled-words list, plus synthetic keyboard noise. Take the
synthetic generators from the robustness literature (AdvGLUE / TextBugger-style, §6 of the
review) rather than hand-rolling them — a home-made typo generator is the easiest way to
accidentally report a win against your own noise model.

**The over-correction failure mode, stated up front.** A fuzzy lexical layer will cheerfully
"fix" a rare-but-real word into a common one: `Wattpad` → `watt pad`, `Kubernetes` →
something regrettable. Mitigations, all required: (a) hard identity channel — if `t` is in
vocabulary at all, `λ_t(t) = 1`; (b) gate correction strength on corpus frequency of `t`;
(c) cap total mass reassigned per token; (d) never correct capitalized non-sentence-initial
tokens (crude NER proxy). Measure over-correction explicitly as a **false-correction rate on
clean text** — a model that scores well on noisy text and mangles clean text is a
regression, and this metric is the only thing that catches it.

### 2.2 Stage 2 — lexeme → hierarchy node

Roget's already gives this as a crisp multiset: a word appears in `k` Heads, and that `k`
*is* its polysemy. Fuzzify with three signals:

1. **Structural position** — Roget's orders words within a semicolon group and paragraph
   roughly by centrality to the Head. Reciprocal rank of the word's position gives a prior.
2. **Corpus sense priors** — relative frequency of each sense, estimated by
   distributional agreement between the word's contexts and the Head's other members.
3. **Roget-distance similarity** — Jarmasz & Szpakowicz's measure, for smoothing mass onto
   near-miss Heads instead of leaving a hard zero.

Then **context disambiguation**: the other tokens in the span vote softly for which Heads
are live, one or two rounds of fuzzy relaxation over the hierarchy. Keep it to two rounds;
this is where compute quietly goes.

### 2.3 Stage 3 — aggregation, hedges, upward closure

Token-level node memberships → span embedding:

1. **Aggregate across tokens** with an OWA operator or t-conorm (`gauss_math.t_conorm`).
   OWA is worth the extra parameter: it interpolates between "any token mentioning this
   concept" (max) and "most tokens do" (mean), and the weights are learnable and readable.
2. **Apply modifier operators** — this is where fuzzy logic earns its keep:

   | Surface | Operator | Code |
   |---|---|---|
   | `not X` | complement | `t_complement(μ)` |
   | `very X`, `extremely X` | concentration | `μ ** 2` |
   | `somewhat X`, `fairly X` | dilation | `μ ** 0.5` |
   | `X and Y` | t-norm | `t_norm(μ_X, μ_Y)` |
   | `X or Y` | t-conorm | `t_conorm(μ_X, μ_Y)` |

   Zadeh's hedges, one line each, no training. Scope comes from a dependency parse or a
   fixed window; window is fine to start.
3. **Enforce (C1)** by upward closure: propagate `μ_p ← max(μ_p, max_child)` bottom-up.
   Renormalize for **(C2)** if the sum variant is used.

**The hedge layer is falsifiable, and that is the point.** SST labels *every parse-tree
node* (§8 of the review). So: given annotated children's sentiment memberships, does
complement-for-`not` and concentration-for-`very` predict the annotated parent's label
better than chance, and better than a learned composition function? That converts "fuzzy
logic composes interpretably" from a slogan into a measured result. It is the single most
under-exploited asset in SST for this purpose, and it is cheap — no embedding model
required, so **it can be run before Stage 1 or 2 exists** (Milestone 2).

## 3. The hierarchy

### 3.1 Roget's backbone

Use **Open Roget's** (CC BY-SA 4.0) for a modern machine-readable structure, or the
Gutenberg 1911 edition (public domain) if the reciprocal licence is unacceptable —
see §9.

The brief proposed the Wikipedia first-link "everything leads to Philosophy" graph.
§3.1–3.2 of the review argues against it in detail; the short version is three
disqualifying properties: **degenerate root mass** (Philosophy absorbs paths two orders of
magnitude more than any other article, so the coarse levels are near-constant across all
inputs and carry no information), **depth without balance** (mean ~23 hops, so there is no
principled level at which to read out), and **edit instability** (the graph is a function of
lead sentences, so one edit silently redefines dimensions between model versions). Roget's
supplies what the brief actually wanted from it: a designed, balanced, named resolution
ladder.

**Do re-derive the level cardinalities from the dump.** The 6/39/79/596/990 figures are
edition-dependent and were taken from a secondary source.

**The antonym gift.** Roget's arranges most categories in **opposed pairs** — 27 Equality /
28 Inequality, 648 Goodness / 649 Badness. Collapse each pair into **one signed bipolar
dimension** with a negative/neutral/positive fuzzy partition. Consequences:

- ~495 signed axes instead of 990 unsigned ones — halves the width.
- Each axis is *already* a linguistic variable in exactly the form
  `MixtureOfGaussiansFuzzyClassifier` consumes.
- Each already carries polarity, so a sentiment rule over the 648/649 axis needs **no
  additional machinery** to be interpretable. This is what makes the downstream sentiment
  FIS nearly free.

### 3.2 Wikipedia category graft (mandatory, not optional)

Roget's 1911 has no `Kubernetes`, no `CRISPR`, and no named entities at all. For any modern
corpus, graft a cleaned Wikipedia-category is-a DAG under specific Heads, following
Ponzetto & Strube: category cleaning → category-pair hypernymy classification → taxonomy
construction, with cycles broken by DFS from top-level categories refusing cycle-closing
arcs (§3.3 of the review). Graft *below* level 5 so the ladder's upper levels stay fixed
and stable.

## 4. What is actually trained

Worth being explicit, because "the hierarchy is given" is the whole point:

| Component | Learned? | How |
|---|---|---|
| Hierarchy structure & node names | **No — given** | Roget's. This is the contribution, not a limitation. |
| Stage-1 lexical-access FIS | Yes | TSK regressor on misspelling pairs (§2.1) |
| Stage-2 sense priors `μ_c(w)` | Yes | structural prior + corpus estimation (§2.2) |
| Stage-3 OWA / aggregation weights | Yes | end-to-end objectives below |
| Hedge exponents | Optionally | start at Zadeh's 2 / 0.5; tune only if §2.3's test says to |
| Membership widths for linguistic terms | Yes, **elicited** | Enhanced Interval Approach (§A2 of sources) rather than guessed |

### 4.1 Objectives

1. **Distillation (fastest path to a legitimate number).** Match a strong teacher's
   *pairwise similarities*, not its vectors — the spaces are incomparable:
   `L = Σ_{i,j} ( sim_fuzzy(E(s_i), E(s_j)) − sim_cos(T(s_i), T(s_j)) )²` with `T` =
   EmbeddingGemma. Cheap, needs no labels, and gets you onto STS quickly.
2. **Contrastive** on NLI / paraphrase pairs, using the fuzzy similarity of §6.2.
3. **Structural regularizer** penalizing violations of (C1)/(C2).
4. **Typo-invariance regularizer** — the direct optimization of the headline requirement:
   `L_typo = d( E(s), E(noise(s)) )` over synthetic character noise.
5. **Interpretability regularizer** (optional) — sparsity on `E(s)`, since a span should
   fire few Heads. Borrow the SPINE formulation.

## 5. Milestones — with go/no-go gates

Ordered so the cheapest falsification comes first. **Do not build stage 2 before
Milestone 0 passes.**

- **M0 — Coverage go/no-go (≈1 day). The gate that decides whether Experiment A is viable
  at all.** Parse the Roget's dump; measure what fraction of tokens in a modern corpus
  (Wikipedia sample + SST + a ticket-data sample from the existing
  `tests/test_textclassifier.py` corpus) land in *any* Head, by token and by type, with and
  without the Wikipedia graft. Report the level cardinalities from the dump.
  **Gate: ≥85% token coverage → proceed. 60–85% → proceed only with the graft built first.
  <60% → the Roget's backbone is the wrong scaffold; stop and reconsider.**
- **M1 — Fuzzy lexical access, standalone (≈3 days).** SymSpell/BK-tree index + five
  channels + TSK aggregator. Metrics: top-1 and top-5 correction accuracy on held-out MOE
  pairs; **false-correction rate on clean text**; median latency per token. Deliverable: a
  per-decision explanation string.
- **M2 — Hedge composition test on SST (≈2 days). Independent of M0/M1 — runnable now.**
  Do Zadeh hedges predict SST parent-node labels from child labels better than chance and
  better than a learned composition? Needs no embedding model. Cheap, publishable on its
  own, and de-risks §2.3.
- **M3 — Stage 2 + 3, first end-to-end embedding (≈1 week).** Emit level-2 (39-d) and
  level-5 (990-d) vectors. Verify rollup exactness (§6.3) as a unit test.
- **M4 — Benchmark (≈1 week).** §6.
- **M5 — Wire into the FIS stack (≈3 days).** Roget's Classes as HME top-level gates
  (§7.3); compare against Experiment B's numbers on identical splits.

## 6. Evaluation

### 6.1 What to measure

| Axis | Instrument | Expectation |
|---|---|---|
| Clean semantic quality | MTEB subset: STS, classification, a little retrieval | **Expect to lose to EmbeddingGemma.** Say so before running. |
| **Typo-perturbed quality** | same tasks under AdvGLUE-style character noise, swept by noise rate | **Expect to win** — the differentiating axis |
| Interpretability | word-intrusion test, per SPINE/NNSE protocol | expect a large win; it is nearly definitional |
| Rollup exactness | §6.3 | must be exact, else (C1)/(C2) are not enforced |
| Hedge composition | §2.3 / M2 | the falsifiable claim |
| Cost | latency, memory, params | expect a large win — no transformer forward pass |
| Over-correction | false-correction rate on clean text | must not regress |

### 6.2 Similarity metric — do not use cosine

Membership vectors are not points in a linear space; cosine on them is a category error
that will quietly cost accuracy. Use **fuzzy Jaccard**:

```
sim(A, B) = Σ_c min(μ_c^A, μ_c^B) / Σ_c max(μ_c^A, μ_c^B)
```

Optionally hierarchy-aware: give partial credit when mass lands on a *sibling* rather than
the same node, weighted by tree distance (Jarmasz–Szpakowicz Roget-distance is the natural
weighting). Report both — plain fuzzy Jaccard is the honest headline, the hierarchy-aware
variant is the interesting one.

### 6.3 The rollup-exactness unit test

For random spans, assert `E_ℓ(s) == rollup(E_{ℓ+1}(s))` to floating-point tolerance at
every level. This is the property that distinguishes the approach from MRL, so it belongs
in CI, not in a notebook. If it fails, (C1)/(C2) are not actually being enforced and the
central claim evaporates.

### 6.4 Frame the comparison honestly

A 990-dimensional lexically-grounded fuzzy embedding **will lose to EmbeddingGemma on clean
STS.** Committing to that in advance is what makes the wins credible. The defensible claims
are: typo-perturbed performance, interpretability, inference cost, auditability, and
steerability. Judging this model on clean-benchmark parity is judging it on the wrong axis —
and deciding that *after* seeing results is how projects talk themselves into bad papers.

## 7. Implementation

### 7.1 Module layout

```
flm/
  fuzzyembed/
    __init__.py
    hierarchy.py     # Roget's parse -> node tree; level readouts; rollup ops
    lexical.py       # Stage 1: candidate index + 5 channels + TSK aggregator
    senses.py        # Stage 2: lexeme -> node membership, context relaxation
    compose.py       # Stage 3: OWA aggregation, Zadeh hedges, upward closure
    similarity.py    # fuzzy Jaccard, hierarchy-aware variant
    explain.py       # per-decision explanation strings
  exp_b/             # Experiment B (see FIS_ON_EMBEDDINGS_PLAN.md)
  literature/
```

Follow the `tribble-tree/` precedent: self-contained under `flm/`, importing TRIBBLE
primitives (`t_norm`, `t_conorm`, `t_complement`, the Ruspini partition builders,
`MixtureOfGaussiansFuzzyRegressor`) without modifying `src/tribblefis/`.

### 7.2 New dependencies

`symspellpy` or a hand-rolled BK-tree; `jellyfish` or `rapidfuzz` (Damerau–Levenshtein,
Double Metaphone); `datasets` for SST. All optional-extras, not core deps — `pyproject.toml`
already carries a TODO about dependency count.

### 7.3 The payoff: one tree for representation and inference

Because the embedding hierarchy is a tree of named fuzzy variables and
`HierarchicalFuzzyExpertsClassifier` gates on named fuzzy variables, they can be **the same
tree**: pin Roget's Classes as the level-0 gates via `VariablePlan.level_order`, Sections as
level-1, and let each leaf expert consume only the level-4/5 Heads inside its own subtree.
The result is a single hierarchical fuzzy system from characters to prediction, with a
readable root-to-leaf path for every decision — and the `VariablePlan` / `NodePin` machinery
for expressing it already exists in `tribble-tree/fuzzytree/plan.py`.

That, not the embedding itself, is the actual argument that this leads somewhere.

## 8. What this is *not*

An embedding model plus a classifier head is **not a language model.** A fuzzy LM needs:

1. a fuzzy representation ← this document,
2. a fuzzy **sequence** model ← does not exist yet,
3. a fuzzy **decoder** ← does not exist yet.

Note specifically that `MixtureOfGaussiansFuzzySequenceClassifier` in this repo is a
*confusion-driven cascade of specialists*, not a temporal sequence model. It is **not** a
seed for stage 2 and should not be described as one in any writeup.

## 9. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| **Roget's coverage of modern vocabulary** | **Critical — kills the experiment** | M0 go/no-go before any other work; Wikipedia graft |
| Clean-benchmark performance below baseline | High, but expected | Pre-commit to the axes in §6.4 |
| Over-correction in stage 1 | High | Identity channel, frequency gating, clean-text FCR metric |
| Sense-assignment quality (stage 2) | High | It is WSD, an unsolved problem; keep to 2 relaxation rounds and measure the ceiling with gold senses |
| CC BY-SA reciprocity on Open Roget's | Medium | Gutenberg 1911 fallback; decide before writing code against a schema |
| Pre-emption by arXiv:2509.13357 | Medium | **Read it first** — it is the closest published relative |
| English-only, static hierarchy | Medium | Acknowledge as scope; Roget-style thesauri exist for other languages |
| Scope creep toward "we built an LLM" | Medium | §8, restated in every writeup |
