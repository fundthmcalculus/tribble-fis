# `flm` — toward a fuzzy language model

Language models are not interpretable. Fuzzy systems are. This directory holds the
literature groundwork and experiment plans for closing that gap from the fuzzy side:
build the representation out of **named fuzzy memberships** rather than trying to
recover names from a dense model after the fact.

## The two experiments

| | Experiment | Status | Doc |
|---|---|---|---|
| **A** | **Fuzzy embedding model** — coordinates are membership degrees in named nodes of a curated lexical hierarchy | **built & measured** (WordNet backend; M0 coverage 96.7%, exact rollup PASS, L2 similarity gap +0.278) | [`FUZZY_EMBEDDING_PLAN.md`](FUZZY_EMBEDDING_PLAN.md) · [`fuzzyembed/`](fuzzyembed/) |
| **A2** | **Fuzzy sequence + fuzzy syntax + fuzzy decoder** — next-token prediction over named semantic *and* syntactic dimensions, then Zadeh linguistic approximation | built; rules at logreg parity (bal-acc 0.528 → **0.569**), category-correct decoding; **aggregate skill still marginal (bal-acc 0.527±0.010) and generation not grammatical** | [`fuzzyembed/README.md`](fuzzyembed/README.md) |
| **B** | **FIS head on a frozen neural embedding model** — sentiment analysis, TSK heads vs. linear probe | harness built & smoke-tested; not run on real data | [`FIS_ON_EMBEDDINGS_PLAN.md`](FIS_ON_EMBEDDINGS_PLAN.md) · [`exp_b/`](exp_b/) |

**Experiment A is implemented** in [`fuzzyembed/`](fuzzyembed/) — see that README for
results, the design corrections found by running it, and honest limitations. A full
what-worked/what-didn't-and-why record is in [`LOG.md`](LOG.md).

**Run B first.** It is cheap, it reuses existing code, and it is the control that
determines whether A is worth building — see §5 of its plan for why its *expected
negative result* is the strongest available argument for A.

## The one-page argument

Post-hoc interpretability (sparse autoencoders being the dominant instrument) trains
a dense model and then tries to name the features it discovered. The naming step is
the weak link: SAE explanations come out too broad, dimensions turn out
polysemantic, and there is now a literature on *falsifying* those explanations.

A fuzzy system doesn't discover its variables — it is handed them, with names,
before training. If the axes of an embedding are the nodes of a curated lexical
hierarchy, there is no naming problem: dimension 648 *is* Roget's Head 648
"Goodness". Three consequences fall out:

1. **Exact multi-resolution readout.** Enforce fuzzy subsumption
   (`μ_parent ≥ max μ_child`) and a per-level sibling partition, and the coarse view
   becomes an *exact* t-conorm aggregation of the fine view. Matryoshka truncation is
   empirically graceful; this is semantically exact — a stronger and checkable claim.
   The repo's `ruspini.py` already builds and refines partition-of-unity families.
2. **Explainable misspelling robustness.** Put the fuzziness at *lexical access*,
   before tokenization can shatter `recieve`. Because that layer is itself a small
   TSK FIS over interpretable channels (edit distance, keyboard adjacency, phonetic
   key), it can *report why* it matched — which no shaped-subword-loss approach can.
3. **One tree for representation and inference.** The embedding hierarchy is a tree
   of named fuzzy variables; `HierarchicalFuzzyExpertsClassifier` gates on named
   fuzzy variables. They can be the same tree — Roget's Classes as top-level gates,
   Heads as leaf-expert inputs — giving a single hierarchical fuzzy system from
   characters to prediction. The `VariablePlan` / `NodePin` machinery for expressing
   that already exists in `tribble-tree/fuzzytree/plan.py`.

**On the Wikipedia game.** The "everything leads to Philosophy" structure is real
(>94% of articles, ~23 mean hops) but it is the wrong scaffold, for three reasons
developed in §3.1 of the review: Philosophy absorbs paths two orders of magnitude
more than any other article, so the coarse levels are near-constant and carry no
information; a mean depth of 23 with no balance means there is no principled level to
read out at; and the graph is a function of lead sentences, so a single edit silently
redefines dimensions between model versions. **Roget's Thesaurus** supplies what the
idea was reaching for — a designed, balanced, named ladder (6 Classes → 39 Sections →
79 Sub-Sections → 596 Head Groups → 990 Heads), openly licensed, and with categories
arranged in *antonymous opposed pairs* (648 Goodness / 649 Badness) that collapse
directly into signed bipolar linguistic variables. That last property is what makes a
sentiment FIS on top of it nearly free. Wikipedia's *category* graph still earns a
place — cleaned into an is-a DAG à la Ponzetto & Strube — as the graft supplying
named entities and post-1911 vocabulary that Roget's structurally cannot have.

## Literature

| Doc | Contents |
|---|---|
| [`literature/FLM_LITERATURE_REVIEW.md`](literature/FLM_LITERATURE_REVIEW.md) | narrative review + **novelty ledger** (§9): what is already published, what is plausibly novel, what the load-bearing risks are |
| [`literature/SOURCES.md`](literature/SOURCES.md) | themed source index (A–K), ◆-marked priority reads, verification debt |
| [`literature/flm_review.bib`](literature/flm_review.bib) | BibTeX skeleton, with `% VERIFY` markers on every uncertain field |

### ⚠ Verification caveat

The authoring session's network egress was allowlisted and excluded arXiv, ACL
Anthology, Springer, IEEE, Wikipedia, and Hugging Face. **Every citation comes from
search-result snippets and abstracts — no PDF was opened and no DOI resolved.** All
entries are marked ✎, none ★. Two numbers to re-derive before quoting: the Roget's
level cardinalities (edition-dependent; get them from the dump) and the
Wikipedia→Philosophy convergence rate (two studies, two methodologies).

### The honest novelty picture

Already published — cite, don't claim: FIS heads on frozen embeddings (**Fuzzy
Fingerprints**, and the fuzzy-BERT sentiment line); membership-valued text
representations (**fuzzy bag-of-words**); interpretable-by-dimension sparse
embeddings (**SPINE**, NNSE); nested variable-dimension embeddings (**MRL**);
misspelling-robust embeddings (**MOE**); and — closest of all, read it first —
**fuzzy-membership features fused into an LM** (arXiv:2509.13357).

Nobody appears to have claimed an end-to-end fuzzy language model. But note what
that implies about scope: the unclaimed territory is the representation and
generation stack, not the classifier head.

## Scope discipline

An embedding model plus a classification head is **not a language model.** A fuzzy
LM needs (1) a fuzzy representation ← Experiment A, **built**; (2) a fuzzy *sequence*
model ← built, with real but insufficient skill; (3) a fuzzy *decoder* ← built and
category-correct. The sequence stage is the bottleneck. Its first version was at chance
because next-word prediction is driven mostly by **syntax and function words**, which a
semantic representation discards by construction; adding a **fuzzy model of syntax**
(named closed-class categories) plus an antecedent representation suited to
membership-valued inputs raised separation ~6x (+0.015 → +0.094), though balanced
accuracy stays marginal at 0.527 ± 0.010. Widening the context and allowing order-3
rules were both predicted to help and measurably **do not**. In particular,
`MixtureOfGaussiansFuzzySequenceClassifier` in this repo is a confusion-driven
cascade of specialists, not a temporal sequence model, and is not a seed for (2).

## Relationship to the rest of the repo

Self-contained under `flm/`, following the `tribble-tree/` precedent: TRIBBLE
primitives are imported, never modified. Direct reuse — `t_norm` / `t_conorm` /
`t_complement` and `take_top_features` from `gauss_math`; the Ruspini
partition-of-unity builders from `ruspini.py` and `refine.py`; the estimator families
from `gaussian_classifier` / `gaussian_regressor`; `fuzzytree`'s trees, HME, and
`VariablePlan`.
