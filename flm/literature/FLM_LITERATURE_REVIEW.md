# Toward a Fuzzy Language Model — Literature Review and Novelty Map

**Scope.** Prior art bearing on two experiments: (**A**) a *fuzzy embedding model* whose
dimensions are named nodes of a lexical hierarchy, and (**B**) a TSK FIS head trained on a
frozen neural embedding model, culminating in textual sentiment analysis. Companion
planning docs: [`../FUZZY_EMBEDDING_PLAN.md`](../FUZZY_EMBEDDING_PLAN.md) and
[`../FIS_ON_EMBEDDINGS_PLAN.md`](../FIS_ON_EMBEDDINGS_PLAN.md). Source index:
[`SOURCES.md`](SOURCES.md); BibTeX: [`flm_review.bib`](flm_review.bib).

> **Verification status — read this first.** This session's network egress was restricted to
> an allowlist that excluded arXiv, ACL Anthology, Wikipedia, Springer, IEEE, and Hugging
> Face. Every claim below is sourced from **search-result snippets and abstracts only** —
> no PDF was opened, no DOI was resolved, no author list was checked against a galley.
> Entries are therefore marked ✎ (search-confirmed) throughout, never ★, in the sense of
> the existing convention in [`SOURCES.md`](../../tribble-tree/literature/SOURCES.md).
> **Spot-check every citation before it goes in a paper.** Two specific numbers to
> re-derive rather than trust: the Roget's level cardinalities (§3.2) and the
> Wikipedia-to-Philosophy convergence rate (§3.1).

---

## 1. The gap this work targets

Interpretability of language models is currently pursued almost entirely *post hoc*: train
a dense model, then try to recover human-legible structure from its activations. The
dominant instrument is the **sparse autoencoder (SAE)**, which decomposes residual-stream
activations into an overcomplete sparse basis in the hope that individual latents are
monosemantic ✎. The approach has real traction — SAEs measurably raise monosemanticity
over raw neurons, including in vision-language models ✎ — but its structural weakness is
that *the features arrive unnamed*. Naming is a separate, unreliable, downstream step:
recent work reports that auto-generated SAE feature explanations "tend to be overly broad
and fail to take polysemanticity into consideration" ✎, has had to build agentic
explainer frameworks to do the naming ✎, and has produced a literature specifically on
*falsifying* SAE feature explanations ✎ and on whether SAEs are useful at all under sparse
probing ✎.

This is the opening. A fuzzy system does not discover its variables — **it is given
them**, and it is given them *with names*, before any training happens. If the axes of an
embedding space are the nodes of a curated lexical hierarchy, there is no naming problem
to solve, no monosemanticity to hope for, and no explanation to falsify: dimension 648 *is*
Roget's Head 648 "Goodness" by construction. That inverts the entire post-hoc program.

The cost is equally structural, and the plan must be honest about it: a hierarchy you did
not learn cannot cover vocabulary its author never saw, and it cannot discover a
distinction its author did not draw. §6 treats this as the primary risk, not a footnote.

## 2. Fuzzy sets meeting natural language — the foundational layer

The idea that word meaning is a matter of degree is not a modern one; it is Zadeh's
original motivation. **Linguistic variables** — variables whose values are words rather
than numbers — and the **fuzzy hedge** calculus are the direct ancestors of anything called
a fuzzy language model ✎. Zadeh's **PRUF** (Possibilistic Relational Universal Fuzzy) was
explicitly a meaning-representation language for natural language ✎, and **computing with
words** formalized the three-step loop this project reproduces: translate propositions into
possibility distributions, reason with fuzzy inference, then retranslate to natural
language via linguistic approximation ✎.

Two threads from that lineage matter operationally.

**Hedges as composition operators.** Zadeh's concentration (`μ²` for *very*) and dilation
(`μ^0.5` for *somewhat*), together with the complement for *not*, give a compositional
semantics for modifiers that is one line of code each and needs no training. This repo
already has `t_norm`, `t_conorm`, and `t_complement` in `src/tribblefis/gauss_math.py`.
§4.3 of the embedding plan argues this is the single highest-leverage place fuzzy logic
beats a neural composition function — and, critically, that it is *testable* rather than
merely assertable (§5.3).

**Words as type-2 fuzzy sets.** How wide should the membership function for "good" be?
Mendel and co-workers answered this empirically with the **Interval Approach** and
**Enhanced Interval Approach**, which encode a word into an interval type-2 fuzzy set from
survey data on where people place its endpoints; the EIA is shown to converge in
mean-square, with ~30 data intervals a reasonable cost/accuracy compromise ✎. Successors
include the HM approach ✎, a general interval approach using a normal distribution with a
free parameter ✎, and a retained-region approach ✎; a three-way comparison of synthesis
methods exists ✎. This is directly reusable: **the antecedent widths in a sentiment FIS
over linguistic terms should be elicited this way rather than guessed**, and the resulting
footprint-of-uncertainty is itself a reportable interpretability artifact. It is also the
principled answer to "robust membership criteria" in the original brief.

## 3. Hierarchy scaffolds — and why the Wikipedia game is the wrong one

The brief proposes the Wikipedia first-link hierarchy ("everything goes back to
philosophy") as the embedding scaffold. The literature supports the *phenomenon* but
argues against the *application*.

### 3.1 What the first-link graph actually is

Following the first lowercase main-text link repeatedly does converge: **>94% of English
articles reach "Philosophy", with a mean chain length of ~23 clicks**, and one study puts
**97.0%** in the strongly connected component containing Philosophy, the residual 3% dying
in dead links ✎. The mechanism is a Manual of Style artifact — leads open with a
definitional sentence, so first links generalize from specific to broad ✎. The structure
has been studied as a network over all ~4.7M articles ✎, including cross-language
variation ✎.

Three properties make it unusable as a dimension hierarchy:

1. **Degenerate root mass.** "Philosophy directs more paths than any other article by two
   orders of magnitude" ✎. A hierarchy whose top node absorbs ~94% of probability mass
   carries almost no information in its upper levels — precisely the levels a
   coarse-resolution readout would use. The 6-dimensional view would be nearly constant
   across all inputs.
2. **Depth without balance.** A mean of 23 hops to the root means the tree is deep and
   wildly unbalanced. There is no natural "level 2" at which to read out; depth-*k*
   truncation slices semantically unrelated concepts into the same bucket.
3. **Edit instability.** The graph is a function of lead sentences. One editor rewording
   one lead rewires an entire subtree — and therefore silently redefines embedding
   dimensions between model versions. Dimension semantics must be stable to be citable.

The first-link graph is a beautiful result about encyclopedic *definitional* structure. It
is not a taxonomy of what words mean.

### 3.2 Roget's Thesaurus is the scaffold the brief actually wants

Roget's is a hierarchically organized semantic ontology mapping the English lexicon into
Classes → Sections → Sub-Sections → Head Groups → Heads; the reported cardinalities are
**6 Classes → 39 Sections → 79 Sub-Sections → 596 Head Groups → 990 Heads** ✎ (verify
these numbers against the edition you use — they are edition-dependent). It has been used
in NLP for word-sense disambiguation, information retrieval, and lexical cohesion ✎;
Jarmasz and Szpakowicz treated it as a lexical resource and built a Roget-distance
semantic similarity measure competitive with WordNet-based measures ✎, and Kennedy's thesis
covers automatic thesaurus construction against it ✎. The cross-reference structure has
been analysed with formal concept analysis ✎, and the design has been ported to other
languages ✎.

Four properties make it the right choice:

1. **Designed branching.** A fixed, curated, roughly balanced 5-level ladder with known
   cardinality per level *is* a resolution dial: 6 / 39 / 79 / 596 / 990. This is the
   structural requirement for "increase and decrease the dimensional resolution as needed"
   and the first-link graph cannot supply it.
2. **Lexical, not encyclopedic.** Roget classifies *senses of words*, which is what an
   embedding dimension should mean. Wikipedia classifies *articles*.
3. **Antonymous opposed pairs.** Most categories are arranged in opposed pairs — 27
   Equality vs 28 Inequality, 648 Goodness vs 649 Badness ✎. This is the single most
   valuable structural fact for this project. Each pair collapses to **one signed bipolar
   dimension** with a natural negative/neutral/positive fuzzy partition — i.e. ~495 signed
   axes that are *already* linguistic variables in exactly the form
   `MixtureOfGaussiansFuzzyClassifier` consumes, and each already carrying the polarity a
   sentiment FIS needs. A sentiment rule over the 648/649 axis is interpretable with zero
   additional machinery.
4. **Open licensing.** **Open Roget's** is distributed under CC BY-SA 4.0 as a downloadable
   tarball ✎, and Project Gutenberg carries the 1911 edition ✎ (public domain). Note the
   CC BY-SA reciprocity obligation if a permissive release is wanted — see §6.

### 3.3 Wikipedia's *category* graph as the modern-vocabulary graft

The category graph — not the first-link graph — is the right Wikipedia contribution, and
the taxonomy-induction literature says exactly how to use it. Ponzetto and Strube's
**WikiTaxonomy** treats all categories as noisy type candidates and subcategory pairs as
hypernymy candidates, in three steps: category cleaning, category-pair classification to
discard non-hypernymy pairs, and taxonomy graph construction ✎. The raw graph is *not* a
well-formed is-a hierarchy — categories have multiple parents, cycles exist on ancestor
paths, and assignments are neither consistent nor complete ✎. Cycles are handled by
depth-first exploration from top-level categories, refusing arcs that would close a cycle;
one such pass reduced >1.16M categories to a >423K-category taxonomy ✎. Related work
covers is-a derivation from the category graph ✎, unsupervised DBpedia taxonomy learning
✎, category semantics ✎, mapping categories and lists to DBpedia ✎, and taxonomy induction
in narrow domains ✎. DBpedia's own ontology is a smaller manually curated alternative ✎.

**Division of labour:** Roget's supplies the stable, balanced, named backbone; a cleaned
Wikipedia-category DAG is grafted underneath specific Heads to supply named entities and
post-1911 technical vocabulary, which Roget's structurally cannot have.

### 3.4 Geometric alternatives, and why they are the wrong tool here

Hierarchy-aware embedding geometry is a mature field. **Poincaré embeddings** (Nickel &
Kiela) place symbolic hierarchies in hyperbolic space, which behaves as a continuous
version of a tree ✎. **Hyperbolic entailment cones** (Ganea et al.) replace entailment
regions with geodesically convex cones inducing a partial order ✎, extended by
hierarchy-aware attention ✎. **Box embeddings** model concepts as axis-aligned boxes whose
volumes give probabilities and granularity ✎, with work on local identifiability ✎ and
joint box/vector knowledge-graph models ✎.

These learn *geometry that respects* a hierarchy. This project needs something different:
**coordinates that are membership degrees in named nodes of a hierarchy**. The relevant
borrowing is conceptual — box containment and cone subsumption are crisp special cases of
the fuzzy subsumption constraint `μ_parent ≥ max_child μ_child` in §4.2 of the embedding
plan. A fuzzy membership vector under that constraint is a *graded* order embedding, which
is strictly more expressive than a box and, unlike a box, has named axes.

## 4. Multi-resolution embeddings — the honest comparison

**Matryoshka Representation Learning** (Kusupati et al., NeurIPS 2022) is the incumbent and
the benchmark to beat on this axis. MRL learns nested coarse-to-fine prefixes within a
single embedding using O(log d) supervision points and no extra forward passes, so a
truncated prefix works standalone ✎. It is now standard practice: **EmbeddingGemma-300M**
(768-d, MRL-truncatable to 512/256/128, ~622MB, MTEB English v2 ≈ 69.67, best open
multilingual model under 500M params) ships it ✎, as does much of the current small-model
field ✎. Extensions include unsupervised/supervised adaptors ✎, retrieval-oriented
recompression ✎, nested product retrieval ✎, multi-sparsity-budget training ✎, and nested
visual clustering ✎.

The distinction to claim — and it is a real one, not a rhetorical one — is this. **MRL
truncation is empirically graceful but semantically opaque**: nothing tells you what the
discarded 640 coordinates meant, and the coarse view is not a *statement* about the fine
view. Under a hierarchical fuzzy embedding with a t-conorm rollup, the level-ℓ readout is
an **exact aggregation** of the level-(ℓ+1) readout: dropping to 39 dimensions loses
resolution in a fully specified way, because each coarse coordinate is a named disjunction
of its children. That is a stronger guarantee. It is also a *testable* one — §5.2 of the
embedding plan specifies the rollup-consistency check.

Note the repo already contains the machinery for the sibling constraint that makes rollup
exact: `src/tribblefis/ruspini.py` builds **Ruspini partitions** — families of fuzzy sets
whose memberships sum to exactly 1 at every point — from shared triangular apex knots, and
`refine_ruspini_partition` moves the knots while preserving partition-of-unity for free.
The hierarchy's per-level sibling partition is the same object, one level up.

## 5. Fuzzy representations of text — direct prior art on Experiment A

**Fuzzy bag-of-words.** The closest existing notion of a fuzzy text embedding. Unlike
classical BoW, fuzzy BoW contains *all* vocabulary words simultaneously at differing
membership degrees; each word is treated as a singleton set and membership of any other
word is the similarity between the two ✎. **Static Fuzzy Bag-of-Words** made this a
lightweight, fast sentence-embedding algorithm ✎ (two versions exist — ICNLSP 2021 and an
arXiv revision ✎). This is genuine prior art for "embedding coordinates are membership
degrees" and must be cited as such. The differences to claim: FBoW's axes are *vocabulary
items* with membership defined by *embedding similarity* (so the axes are as unnamed as the
embedding they came from, and there are |V| of them); the proposal here uses *hierarchy
nodes* as axes with membership defined by *lexical-semantic subsumption*, giving 990 named
axes and a resolution ladder.

**Fuzzy clustering of embeddings.** Applying fuzzy c-means and relatives to GloVe-style
vectors yields words holding graded membership in several clusters at once ✎, with
follow-up work on validity indices ✎. This establishes that graded multi-cluster
membership is a natural description of lexical semantics — supporting evidence for the
premise, not a competing method, since the clusters are discovered and unnamed.

**Fuzzy-membership features fused into a language model.** The nearest published relative
of Experiment A: each token is represented by a vector of interpretable features whose
values are graded degrees from *differentiable* membership functions, and these per-token
vectors form a sentence-level semantic matrix fused into the LM, aimed at controllable
generation ✎. This is close enough that it must be read in full and positioned against
before any claim of novelty is made. From the abstract alone the differences appear to be:
no curated hierarchy (hence no named multi-resolution ladder), no lexical-access layer for
misspelling robustness, and the fuzzy features augment a neural LM rather than constituting
the representation. **Treat this as the paper most likely to pre-empt part of the claim.**

**Set-word embeddings** offer another contextual set-theoretic model with semantic indices
✎, and there is work transforming between dense and sparse text representations ✎.

**Interpretable-by-dimension embeddings without fuzzy sets.** The non-fuzzy analogue of the
goal. **SPINE** is a denoising *k*-sparse autoencoder producing sparse non-negative word
vectors in a higher-dimensional space, preserving pairwise inner products, and is
substantially more interpretable than GloVe, word2vec, and Sparse Overcomplete Word Vectors
under human evaluation ✎; **NNSE** (non-negative sparse embedding) preceded it ✎, and
sparse self-representation variants followed ✎. Their evaluation methodology — word
intrusion tests — is the right instrument for §5.4 of the embedding plan, and should be
borrowed wholesale. Their limitation is the SAE limitation of §1: dimensions are
*discovered*, then named by hand, and "these dimensions are discovered naturally during
the course of training" ✎ is precisely the property the hierarchy scaffold discards on
purpose.

## 6. Misspelling robustness — direct prior art on requirement (1)

**Misspelling Oblivious Embeddings (MOE)**, Edizel, Piktus et al., NAACL 2019, is the
reference point: a fastText variant whose loss gains a term pulling subword embeddings of
misspelt terms toward the embedding of the correct term, trained on a purpose-built
misspelling dataset released publicly, with gains on intrinsic and extrinsic tasks ✎.
Related: robust embeddings via distributions ✎ and noise-aware training for sequence
labeling ✎.

The mechanism-level reason this matters more for modern models than for fastText: **"
character-level edits that break tokens alter tokenization and push embeddings far from
regions seen in training, making robustness to character-level noise more challenging than
word-level robustness"** ✎. A subword-tokenized transformer has no graceful degradation
path for `recieve` — it becomes a different token sequence. This is the mechanistic
argument that a *fuzzy lexical access* layer is not a gimmick: membership in the lexeme is
computed before tokenization can shatter.

Evaluation practice is well established. Robustness studies span character, word, and
sentence perturbations ✎; **AdvGLUE** includes TextBugger-style typo insertion at important
words ✎; there is recent work specifically on multilingual typographical robustness of
LLMs ✎ and on dense retrieval/embedding robustness to typos ✎, plus cheap character noise
for OCR-robust training ✎, noisy slot filling ✎, POS-level adversarial text classification
✎, restricted-output indexing for noisy input ✎, and multi-objective adversarial text
generation ✎. **Take the perturbation generators from this literature rather than inventing
them** — a hand-rolled typo generator is the easiest way to accidentally report a win.

## 7. FIS on top of neural embeddings — direct prior art on Experiment B

This is the part of the program that is **already published**, and the plan should treat
Experiment B as a *replication-plus-comparison* rather than a novel contribution.

**Fuzzy Fingerprints (FFP)** is the closest match, and it is the same architecture
Experiment B proposes. Utterances go to a pre-trained RoBERTa; the contextual embeddings
feed an adapted Fuzzy Fingerprint classification module ✎. Class prototypes are built by
**ranking and fuzzifying the activations** of pooled embeddings across training instances
per class; at inference an input is fingerprinted the same way and matched to prototypes by
a fuzzy similarity aggregating fuzzy-set intersections ✎. The authors report that prototype
use plus FFP sparsity buys interpretability normally unavailable in neural models, that the
method generalizes to any LLM, and that it "bridges the gap between Fuzzy Systems and
LLMs" ✎ — with a line of papers from the 2023 arXiv version through a Fuzzy Logic and
Technology chapter to a 2026 human-assessment and validity study ✎, plus a variant for
limited discrete feature spaces ✎. **Experiment B must cite this as prior art and, ideally,
include it as a baseline.**

**Fuzzy + BERT for sentiment** is likewise an established line: a three-step fuzzy-based
BERT model ✎; **FDiBD** (Fuzzification–DistilBERT–Defuzzification) targeting ambiguity and
long-range dependencies ✎; scalable fuzzy-inference ensembles for sentiment ✎; fuzzy
rule-based systems for interpretable sentiment analysis ✎; word-embedding + emotion-lexicon
+ fuzzy-inference hybrids where a BiLSTM learns word relations and a fuzzy rule mechanism
makes the final decision ✎; Sentence-BERT similarity with augmented lexicons and rule-based
modifier detection ✎; and fuzzy-weighted sentiment recognition in education ✎. On the
architecture side, TSK ensembles have been shown to cut rule count while improving
interpretability ✎, and there is a comprehensive review of interpretable fuzzy rules for
XAI ✎ plus a broader fuzzy-logic-for-interpretability review ✎. **CogniFNN** applies a
fuzzy neural framework to *evaluating* word embeddings against cognitive data ✎.

Adjacent LLM+fuzzy work is mostly orthogonal but worth knowing: fuzzy reasoning chains ✎,
LLM-as-a-fuzzy-judge ✎, fuzzy-logic prompting frameworks ✎, fuzzy-logic-augmented
educational QA ✎, LLMs combined with explainable FIS for industrial defect detection ✎, and
fuzzy graph-database search ✎. Non-fuzzy interpretable-rule text classification is also
represented, e.g. the **Tsetlin machine** ✎ and textual-interpretable-feature mining ✎.

**Nobody appears to have claimed an end-to-end fuzzy language model.** Searching for
"fuzzy large language model" / "fuzzy transformer" returns only augmentation and
post-processing work — fuzzy control of decoding parameters, fuzzy judging, fuzzy prompting,
fuzzy classification heads. The gap is real. But note what that implies about scope: the
unclaimed territory is the *representation and generation* stack, not the classifier head.

## 8. Where sentiment labels are genuinely fuzzy

A detail that materially shapes Experiment B. The **Stanford Sentiment Treebank** was
annotated with a slider allowing **up to 25 levels of sentiment**, over **215,154 uniquely
labeled phrases** from 11,855 Rotten Tomatoes sentences, with every parse-tree node labeled
by at least three annotators; the familiar SST-5 / SST-2 tasks are *discretizations* of
those continuous scores ✎.

Two consequences:

1. **Regress the continuous score, don't only classify the bucket.** SST ships the graded
   values; `MixtureOfGaussiansFuzzyRegressor` consumes a continuous target directly. Fuzzy
   systems are built for graded truth, and collapsing to 5 classes throws away the one
   property of the dataset that plays to the architecture's strength. Reporting MAE against
   the continuous score is a fairer and more distinctive evaluation than SST-5 accuracy.
2. **Phrase-level labels make hedge composition falsifiable.** Because SST labels internal
   parse nodes, the Zadeh-hedge composition rules of §2 can be tested directly: given
   children's memberships, does `not`-as-complement / `very`-as-concentration predict the
   annotated parent label better than chance, and better than a learned composition? That
   converts "fuzzy logic composes interpretably" from a claim into an experiment. It is the
   most under-exploited asset in the dataset for this purpose.

## 9. Novelty ledger

Stated plainly, so nothing gets oversold.

**Already done — cite, don't claim.**
- FIS / fuzzy-prototype classification heads on frozen pre-trained embeddings — Fuzzy
  Fingerprints ✎, and the fuzzy-BERT sentiment line ✎.
- Text representations whose coordinates are membership degrees — fuzzy BoW, static FBoW ✎.
- Graded multi-cluster membership of words — fuzzy clustering of embeddings ✎.
- Interpretable-by-dimension sparse embeddings — SPINE, NNSE ✎.
- Nested variable-dimension embeddings — MRL and its descendants ✎.
- Misspelling-robust embeddings via subword loss shaping — MOE ✎.
- Fuzzy membership features fused into an LM for controllable generation ✎ — **read this
  one before claiming anything.**

**Plausibly novel — the actual contribution surface.**
1. **A priori named axes with an exact resolution ladder.** An embedding whose coordinates
   are memberships in curated hierarchy nodes, where the coarse readout is an exact
   t-conorm aggregation of the fine readout. Distinct from MRL (nested but unnamed and only
   empirically graceful) and from SPINE/SAE (interpretable but discovered, then named
   post hoc).
2. **Fuzzy lexical access as a trained FIS.** Misspelling robustness that is not merely
   *achieved* but *explained* — "matched `recieve` → *receive* at 0.94, driven by
   transposition distance and phonetic key agreement". No prior work in §6 produces an
   attribution for its own robustness, because a shaped subword loss has nothing to report.
3. **Zadeh hedges as the composition operator, validated on SST phrase labels.** §8.2.
4. **Aligning the embedding hierarchy with the FIS gating hierarchy.** Roget's Classes as
   the top-level gates of a `HierarchicalFuzzyExpertsClassifier`, so the representation and
   the inference engine share one tree. This is the piece that makes the whole stack a
   single hierarchical fuzzy system rather than a fuzzy head bolted to a neural encoder —
   and it is what the existing `fuzzytree` HME code was already built for.

**The load-bearing risks.**
- **Coverage.** Roget's 1911 has no `Kubernetes`, no `CRISPR`, no named entities at all.
  If token coverage on a modern corpus is low, Experiment A does not work as specified.
  This is cheap to measure and must be measured *first* — it is Milestone 0 of the
  embedding plan, framed as a go/no-go.
- **Licensing.** Open Roget's is CC BY-SA 4.0 ✎ — reciprocal. The Gutenberg 1911 edition is
  public domain but lexically dated. Choose deliberately; the choice constrains release.
- **Ceiling.** A 990-dimensional lexically-grounded fuzzy embedding will very likely lose to
  EmbeddingGemma on clean STS. The defensible claims are typo-perturbed performance,
  interpretability, inference cost, and auditability — not clean-benchmark parity. Say so
  before running, not after.
- **Scope discipline.** An embedding model plus a classification head is not a language
  model. A fuzzy LM needs a fuzzy sequence model and a fuzzy decoder too, and neither
  exists here. Note that `MixtureOfGaussiansFuzzySequenceClassifier` in this repo is a
  *confusion-driven cascade*, not a temporal sequence model — it is not a seed for the
  sequence stage, and should not be described as one.

## Sources

Search-result URLs backing the above, grouped as in the text. Nothing here was opened
(§0 verification note) — the structured index with themes is in [`SOURCES.md`](SOURCES.md).

- [Semantic Fusion with Fuzzy-Membership Features for Controllable Language Modelling](https://arxiv.org/html/2509.13357)
- [Fuzzy Fingerprinting Transformer Language-Models for Emotion Recognition in Conversations](https://arxiv.org/abs/2309.04292) · [Fuzzy Fingerprinting Encoder PLMs: human assessment and validity](https://arxiv.org/pdf/2605.02665) · [Fuzzy Fingerprinting Large Pre-trained Models](https://link.springer.com/chapter/10.1007/978-3-031-39965-7_20) · [Fuzzy Fingerprints in Limited Discrete Feature Spaces](https://link.springer.com/chapter/10.1007/978-3-032-29000-7_21)
- [Matryoshka Representation Learning (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/c32319f4868da7613d78af9993100e42-Paper-Conference.pdf) · [Matryoshka-Adaptor](https://arxiv.org/pdf/2407.20243) · [SMEC](https://arxiv.org/pdf/2510.12474) · [NEAR²](https://arxiv.org/pdf/2506.19743) · [Franca](https://arxiv.org/pdf/2507.14137) · [Train one SAE across sparsity budgets](https://arxiv.org/pdf/2505.24473)
- [EmbeddingGemma announcement](https://developers.googleblog.com/en/introducing-embeddinggemma/) · [EmbeddingGemma model overview](https://ai.google.dev/gemma/docs/embeddinggemma) · [Open-source embedding models 2026](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models) · [Ollama embedding models benchmarked](https://www.morphllm.com/ollama-embedding-models)
- [SPINE: Sparse Interpretable Neural Embeddings](https://arxiv.org/pdf/1711.08792) · [NNSE](https://www.researchgate.net/publication/270878338_Learning_Effective_and_Interpretable_Semantic_Models_using_Non-Negative_Sparse_Embedding) · [Interpretable neural embeddings with sparse self-representation](https://arxiv.org/pdf/2306.14135) · [Transformation of dense and sparse text representations](https://arxiv.org/pdf/1911.02914)
- [Evaluating SAEs for monosemantic representation](https://arxiv.org/pdf/2508.15094) · [Revising and falsifying SAE feature explanations](https://openreview.net/forum?id=OJAW2mHVND) · [SAGE agentic SAE explainer](https://arxiv.org/pdf/2511.20820) · [SAEs learn monosemantic features in VLMs](https://openreview.net/forum?id=DaNnkQJSQf)
- [Misspelling Oblivious Word Embeddings (NAACL 2019)](https://aclanthology.org/N19-1326/) · [Semantic Scholar entry](https://www.semanticscholar.org/paper/Misspelling-Oblivious-Word-Embeddings-Edizel-Piktus/8fa7d1f3f82526935ba122c20c8d0648506301b3) · [Robust embeddings via distributions](https://arxiv.org/pdf/2104.08420) · [NAT: noise-aware training](https://arxiv.org/pdf/2005.07162)
- [AdvGLUE](https://arxiv.org/pdf/2111.02840) · [Multilingual typographical robustness of LLMs](https://arxiv.org/html/2510.09536v1) · [Cheap character noise for OCR-robust models](https://aclanthology.org/2025.findings-acl.609.pdf) · [Towards optimal adversarial texts](https://link.springer.com/article/10.1186/s42400-025-00500-3) · [Noisy slot filling](https://arxiv.org/pdf/2310.03518) · [Finite-context indexing for noisy input](https://arxiv.org/pdf/2310.14110) · [MRC robustness to noisy inputs](https://arxiv.org/pdf/2005.00190) · [POS adversarial text classification](https://arxiv.org/html/2408.08374v1)
- [Wikipedia:Getting to Philosophy](https://en.wikipedia.org/wiki/Wikipedia:Getting_to_Philosophy) · [Structure of Wikipedia's First Link Network](https://arxiv.org/pdf/1605.00309) · [Cultural structures of knowledge from first links](https://arxiv.org/pdf/1708.05368)
- [WikiTaxonomy](https://www.researchgate.net/publication/220836746_WikiTaxonomy_A_Large_Scale_Knowledge_Resource) · [Deriving a large scale taxonomy from Wikipedia](https://www.researchgate.net/publication/200773229_Deriving_a_large_scale_taxonomy_from_Wikipedia) · [Derivation of is-a taxonomy from the category graph](https://www.researchgate.net/publication/295075220_Derivation_of_is_a_taxonomy_from_Wikipedia_Category_Graph) · [Uncovering the semantics of Wikipedia categories](https://link.springer.com/chapter/10.1007/978-3-030-30793-6_13) · [Unsupervised DBpedia taxonomy](https://www.researchgate.net/publication/301377099_Unsupervised_learning_of_an_extensive_and_usable_taxonomy_for_DBpedia) · [SLHCat](https://arxiv.org/abs/2309.11791) · [TiFi](https://arxiv.org/pdf/1901.10263) · [Machine Knowledge (KB curation survey)](https://arxiv.org/pdf/2009.11564)
- [Open Roget's](https://sites.google.com/site/openrogets/) · [Roget's Thesaurus as a Lexical Resource (Jarmasz)](https://arxiv.org/pdf/1204.0140) · [Roget's Thesaurus and Semantic Similarity](https://arxiv.org/pdf/1204.0245) · [Kennedy, automatic thesaurus construction](https://www.cs.toronto.edu/~akennedy/publications/phd_thesis.pdf) · [Semantic structure of Roget's cross-references](https://ceur-ws.org/Vol-476/paper9.pdf) · [Swedish Roget-style thesaurus](https://aclanthology.org/2020.globalex-1.9.pdf)
- [Hyperbolic entailment cones (PMLR)](http://proceedings.mlr.press/v80/ganea18a/ganea18a.pdf) · [arXiv version](https://arxiv.org/pdf/1804.01882) · [Coneheads: hierarchy-aware attention](https://arxiv.org/pdf/2306.00392) · [Improving local identifiability in probabilistic box embeddings](https://arxiv.org/pdf/2010.04831) · [Concept2Box](https://arxiv.org/pdf/2307.01933)
- [Static Fuzzy Bag-of-Words (ICNLSP 2021)](https://aclanthology.org/2021.icnlsp-1.9.pdf) · [arXiv version](https://arxiv.org/pdf/2304.03098) · [Analysis of word embeddings using fuzzy clustering](https://arxiv.org/abs/1907.07672) · [Word embeddings and validity indexes in fuzzy clustering](https://arxiv.org/pdf/2205.06802) · [Set-word embeddings and semantic indices](https://www.mdpi.com/2073-431X/14/1/30)
- [Enhanced Interval Approach (IEEE)](https://ieeexplore.ieee.org/document/6086759/) · [Encoding words into normal IT2 FSs: HM approach](https://www.semanticscholar.org/paper/Encoding-Words-Into-Normal-Interval-Type-2-Fuzzy-HM-Hao-Mendel/17f8fc8228d953db00c9fbc85a9b5e17f0107539) · [General interval approach](https://link.springer.com/article/10.1007/s00500-018-3454-9) · [Retained region approach](https://www.sciencedirect.com/science/article/abs/pii/S0020025523001949) · [Comparison of three synthesis approaches](https://link.springer.com/article/10.1007/s41066-015-0009-7) · [The Perceptual Computer: past to future](https://link.springer.com/article/10.1007/s00287-018-1088-z) · [IT2 FSs for linguistic label perception](https://link.springer.com/article/10.1007/s00500-014-1246-4)
- [Fuzzy logic = computing with words](https://www.semanticscholar.org/paper/Fuzzy-logic-=-computing-with-words-Zadeh/f26cbe40db22c9b99fe95d368c3aff94beaef488) · [Genesis of fuzzy sets and systems](https://www.eolss.net/sample-chapters/c15/E6-44-40-07.pdf) · [Fuzzy sets in approximate reasoning: a personal view](https://link.springer.com/chapter/10.1007/978-3-322-88955-3_1) · [On quantified linguistic approximation](https://arxiv.org/pdf/1301.6712) · [Modeling vagueness in data-to-text via fuzzy sets](https://arxiv.org/pdf/1710.10093) · [Empirical study of CWW approaches](https://arxiv.org/pdf/2004.14892)
- [Stanford Sentiment Treebank (Zenodo)](https://zenodo.org/records/5256915) · [SST overview](https://medium.com/data-science/the-stanford-sentiment-treebank-sst-studying-sentiment-analysis-using-nlp-e1a4cad03065) · [Fine-grained sentiment classification using BERT](https://arxiv.org/pdf/1910.03474)
- [Scalable fuzzy inference-based ensemble for sentiment analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC9534613/) · [Three-step fuzzy-based BERT](https://www.researchgate.net/publication/359547225_A_Three-Step_Fuzzy-Based_BERT_Model_for_Sentiment_Analysis) · [FDiBD: fuzzy logic + DistilBERT](https://link.springer.com/article/10.1007/s42452-025-08015-9) · [Fuzzy rule based systems for interpretable sentiment analysis](https://ieeexplore.ieee.org/document/7974497/) · [Fuzzy-weighted sentiment recognition](https://www.scitepress.org/PublishedPapers/2025/137942/) · [CogniFNN](https://arxiv.org/pdf/2009.11485)
- [Fuzzy IS with interpretable rules for XAI (review)](https://www.sciencedirect.com/science/article/pii/S0020025524001257) · [Fuzzy Reasoning Chain](https://aclanthology.org/2025.findings-emnlp.541.pdf) · [LLM-as-a-Fuzzy-Judge](https://arxiv.org/pdf/2506.11221) · [Fuzzy logic prompting framework](https://arxiv.org/pdf/2508.06754) · [Chaotic fuzzy-logic-augmented LLM QA](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1404940/full) · [LLMs + explainable FIS for defect detection](https://www.sciencedirect.com/science/article/abs/pii/S0167865525001096)
- [Tsetlin machine for interpretable text categorization](https://arxiv.org/pdf/1809.04547) · [Mining textual interpretable features](https://arxiv.org/pdf/2106.06697)
