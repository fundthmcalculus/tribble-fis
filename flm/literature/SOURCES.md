# FLM — Structured Source Index

Prior-art index for the fuzzy language model work, grouped by theme. Companion to the
narrative review [`FLM_LITERATURE_REVIEW.md`](FLM_LITERATURE_REVIEW.md). BibTeX:
[`flm_review.bib`](flm_review.bib). Sibling index for the hierarchical-FIS work:
[`../../tribble-tree/literature/SOURCES.md`](../../tribble-tree/literature/SOURCES.md).

**Legend**
- ✎ — citation found via web search; title/venue taken from search snippets.
- ⚠ — **nothing in this file was PDF-verified.** Session egress was allowlisted and
  excluded arXiv, ACL Anthology, Springer, IEEE, Wikipedia, and Hugging Face, so no galley
  was opened and no DOI resolved. Verify authors, year, venue, and page numbers before
  citing. There are deliberately **no ★ (verified) entries** in this index yet.
- ◆ — **highest-priority reads**: closest prior art, most likely to pre-empt a claim.

Themes: **A** Foundational fuzzy semantics & computing with words · **B** Fuzzy text
representations · **C** Interpretable-by-dimension embeddings & SAE critique ·
**D** Multi-resolution / nested embeddings · **E** Hierarchy scaffolds (Roget's,
Wikipedia) · **F** Hierarchy-aware embedding geometry · **G** Misspelling &
character-noise robustness · **H** FIS heads on neural embeddings · **I** Fuzzy sentiment
analysis · **J** Datasets & models · **K** Adjacent LLM+fuzzy (orthogonal but relevant).

---

## A. Foundational fuzzy semantics & computing with words

| Mark | Ref | Venue | Link |
|---|---|---|---|
| ✎ | Zadeh, *Fuzzy logic = computing with words* | IEEE T-FS 4(2), 1996 | [Semantic Scholar](https://www.semanticscholar.org/paper/Fuzzy-logic-=-computing-with-words-Zadeh/f26cbe40db22c9b99fe95d368c3aff94beaef488) |
| ✎ | Zadeh, *PRUF — a meaning representation language for natural languages* | Int. J. Man-Machine Studies, 1978 | (cited via CWW literature; find galley) |
| ✎ | Zadeh, *The concept of a linguistic variable and its application to approximate reasoning* | Inf. Sci., 1975 | (foundational; hedges/concentration/dilation) |
| ✎ | Dubois & Prade, *Fuzzy sets in approximate reasoning: a personal view* | Springer chapter | [link](https://link.springer.com/chapter/10.1007/978-3-322-88955-3_1) |
| ✎ | *The genesis of fuzzy sets and systems* | EOLSS sample chapter | [PDF](https://www.eolss.net/sample-chapters/c15/E6-44-40-07.pdf) |
| ✎ | *On quantified linguistic approximation* | arXiv:1301.6712 | [PDF](https://arxiv.org/pdf/1301.6712) |
| ✎ | *On modeling vagueness and uncertainty in data-to-text systems through fuzzy sets* | arXiv:1710.10093 | [PDF](https://arxiv.org/pdf/1710.10093) |
| ✎ | *An empirical study of computing with words approaches* | arXiv:2004.14892 | [PDF](https://arxiv.org/pdf/2004.14892) |

### A2. Encoding words as (interval type-2) fuzzy sets — membership elicitation

| Mark | Ref | Venue | Link |
|---|---|---|---|
| ✎◆ | Wu, Mendel & Coupland, *Enhanced Interval Approach for encoding words into IT2 FSs + convergence analysis* | IEEE T-FS, 2012 | [IEEE](https://ieeexplore.ieee.org/document/6086759/) |
| ✎ | Hao & Mendel, *Encoding words into normal IT2 FSs: HM approach* | IEEE T-FS | [Semantic Scholar](https://www.semanticscholar.org/paper/Encoding-Words-Into-Normal-Interval-Type-2-Fuzzy-HM-Hao-Mendel/17f8fc8228d953db00c9fbc85a9b5e17f0107539) |
| ✎ | *General interval approach … normal distribution and free parameter* | Soft Comput., 2018 | [10.1007/s00500-018-3454-9](https://link.springer.com/article/10.1007/s00500-018-3454-9) |
| ✎ | *Encoding words into IT2 FSs: the retained region approach* | Inf. Sci., 2023 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025523001949) |
| ✎ | *A comparison of three approaches for estimating an IT2 FS model of a linguistic term* | Granular Comput., 2015 | [10.1007/s41066-015-0009-7](https://link.springer.com/article/10.1007/s41066-015-0009-7) |
| ✎ | Mendel, *The Perceptual Computer: the past, up to the present, and into the future* | Informatik Spektrum, 2018 | [link](https://link.springer.com/article/10.1007/s00287-018-1088-z) |
| ✎ | *IT2 FSs to model linguistic label perception in online services satisfaction* | Soft Comput., 2014 | [link](https://link.springer.com/article/10.1007/s00500-014-1246-4) |

## B. Fuzzy text representations

| Mark | Ref | Venue | Link |
|---|---|---|---|
| ✎◆ | *Semantic Fusion with Fuzzy-Membership Features for Controllable Language Modelling* | arXiv:2509.13357 | [HTML](https://arxiv.org/html/2509.13357) |
| ✎◆ | *Static Fuzzy Bag-of-Words: a lightweight and fast sentence embedding algorithm* | ICNLSP 2021 | [ACL](https://aclanthology.org/2021.icnlsp-1.9.pdf) · [arXiv:2304.03098](https://arxiv.org/pdf/2304.03098) |
| ✎ | Zhao & Mao, *Fuzzy bag-of-words model for document representation* (original FBoW — confirm exact cite) | IEEE T-FS, 2018 | (find galley) |
| ✎ | *Analysis of word embeddings using fuzzy clustering* | arXiv:1907.07672 | [abs](https://arxiv.org/abs/1907.07672) |
| ✎ | *Word embeddings and validity indexes in fuzzy clustering* | arXiv:2205.06802 | [PDF](https://arxiv.org/pdf/2205.06802) |
| ✎ | *Set-word embeddings and semantic indices* | Computers 14(1):30, 2025 | [MDPI](https://www.mdpi.com/2073-431X/14/1/30) |
| ✎ | *CogniFNN: a fuzzy neural network framework for cognitive word embedding evaluation* | arXiv:2009.11485 | [PDF](https://arxiv.org/pdf/2009.11485) |

## C. Interpretable-by-dimension embeddings & the SAE critique

| Mark | Ref | Venue | Link |
|---|---|---|---|
| ✎◆ | Subramanian et al., *SPINE: SParse Interpretable Neural Embeddings* | AAAI 2018 | [arXiv:1711.08792](https://arxiv.org/pdf/1711.08792) · [CMU copy](https://www.cs.cmu.edu/~hovy/papers/18AAAI-SPINE.pdf) |
| ✎ | Murphy, Talukdar & Mitchell, *Learning effective and interpretable semantic models using NNSE* | COLING 2012 | [RG](https://www.researchgate.net/publication/270878338_Learning_Effective_and_Interpretable_Semantic_Models_using_Non-Negative_Sparse_Embedding) |
| ✎ | *Interpretable neural embeddings with sparse self-representation* | arXiv:2306.14135 | [PDF](https://arxiv.org/pdf/2306.14135) |
| ✎ | *Transformation of dense and sparse text representations* | arXiv:1911.02914 | [PDF](https://arxiv.org/pdf/1911.02914) |
| ✎ | *Evaluating sparse autoencoders for monosemantic representation* | arXiv:2508.15094 | [PDF](https://arxiv.org/pdf/2508.15094) |
| ✎◆ | *Revising and falsifying sparse autoencoder feature explanations* | OpenReview | [forum](https://openreview.net/forum?id=OJAW2mHVND) |
| ✎ | *SAGE: an agentic explainer framework for interpreting SAE features* | arXiv:2511.20820 | [PDF](https://arxiv.org/pdf/2511.20820) |
| ✎ | *Sparse autoencoders learn monosemantic features in VLMs* | NeurIPS 2025 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/file/89e83382abeee53b932a6df62edbf9cc-Paper-Conference.pdf) |
| ✎ | *Are sparse autoencoders useful? A case study in sparse probing* | ICML 2025 | (referenced in above; find galley) |
| ✎ | *SoftSAE: dynamic top-k selection for adaptive SAEs* | arXiv:2605.06610 | [PDF](https://arxiv.org/pdf/2605.06610) |

## D. Multi-resolution / nested embeddings

| Mark | Ref | Venue | Link |
|---|---|---|---|
| ✎◆ | Kusupati et al., *Matryoshka Representation Learning* | NeurIPS 2022 | [proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/file/c32319f4868da7613d78af9993100e42-Paper-Conference.pdf) |
| ✎ | *Matryoshka-Adaptor: unsupervised and supervised tuning for smaller dims* | arXiv:2407.20243 | [PDF](https://arxiv.org/pdf/2407.20243) |
| ✎ | *SMEC: rethinking MRL for retrieval embedding compression* | arXiv:2510.12474 | [PDF](https://arxiv.org/pdf/2510.12474) |
| ✎ | *NEAR²: a nested embedding approach to product retrieval and ranking* | arXiv:2506.19743 | [PDF](https://arxiv.org/pdf/2506.19743) |
| ✎ | *Franca: nested Matryoshka clustering for visual representation* | arXiv:2507.14137 | [PDF](https://arxiv.org/pdf/2507.14137) |
| ✎ | *Enhancing semantic similarity in Arabic NLP with nested embedding learning* | arXiv:2407.21139 | [PDF](https://arxiv.org/pdf/2407.21139) |

## E. Hierarchy scaffolds

### E1. Roget's Thesaurus

| Mark | Ref | Venue | Link |
|---|---|---|---|
| ✎◆ | **Open Roget's** (CC BY-SA 4.0, downloadable tarball) | resource | [site](https://sites.google.com/site/openrogets/) |
| ✎◆ | Jarmasz, *Roget's Thesaurus as a lexical resource* (MSc thesis) | arXiv:1204.0140 | [PDF](https://arxiv.org/pdf/1204.0140) |
| ✎◆ | Jarmasz & Szpakowicz, *Roget's Thesaurus and semantic similarity* | RANLP 2003 | [arXiv:1204.0245](https://arxiv.org/pdf/1204.0245) |
| ✎ | Kennedy, *Automatic supervised thesauri construction with Roget's Thesaurus* (PhD thesis) | U. Ottawa | [PDF](https://www.cs.toronto.edu/~akennedy/publications/phd_thesis.pdf) |
| ✎ | Old, *The semantic structure of Roget's Thesaurus cross-references* | CEUR-WS Vol-476 | [PDF](https://ceur-ws.org/Vol-476/paper9.pdf) |
| ✎ | *Towards a Swedish Roget-style thesaurus for NLP* | GlobaLex 2020 | [ACL](https://aclanthology.org/2020.globalex-1.9.pdf) |
| ✎ | Roget's Thesaurus, 1911 edition (public domain) | Project Gutenberg | (gutenberg.org, ebook #10681 — confirm ID) |

### E2. Wikipedia first-link graph (studied, then rejected as scaffold — §3.1 of review)

| Mark | Ref | Venue | Link |
|---|---|---|---|
| ✎ | *Wikipedia:Getting to Philosophy* (>94%, ~23 mean clicks) | Wikipedia project page | [link](https://en.wikipedia.org/wiki/Wikipedia:Getting_to_Philosophy) |
| ✎ | *Connecting every bit of knowledge: the structure of Wikipedia's first link network* | arXiv:1605.00309 | [PDF](https://arxiv.org/pdf/1605.00309) |
| ✎ | *Cultural structures of knowledge from Wikipedia networks of first links* | arXiv:1708.05368 | [PDF](https://arxiv.org/pdf/1708.05368) |

### E3. Wikipedia category graph → is-a taxonomy (the usable graft)

| Mark | Ref | Venue | Link |
|---|---|---|---|
| ✎◆ | Ponzetto & Strube, *WikiTaxonomy: a large scale knowledge resource* | ECAI 2008 | [RG](https://www.researchgate.net/publication/220836746_WikiTaxonomy_A_Large_Scale_Knowledge_Resource) |
| ✎◆ | Ponzetto & Strube, *Deriving a large scale taxonomy from Wikipedia* | AAAI 2007 | [RG](https://www.researchgate.net/publication/200773229_Deriving_a_large_scale_taxonomy_from_Wikipedia) |
| ✎ | *Derivation of "is a" taxonomy from the Wikipedia Category Graph* | — | [RG](https://www.researchgate.net/publication/295075220_Derivation_of_is_a_taxonomy_from_Wikipedia_Category_Graph) |
| ✎ | *Uncovering the semantics of Wikipedia categories* | ISWC 2019 | [Springer](https://link.springer.com/chapter/10.1007/978-3-030-30793-6_13) |
| ✎ | *Unsupervised learning of an extensive and usable taxonomy for DBpedia* | SEMANTiCS 2016 | [RG](https://www.researchgate.net/publication/301377099_Unsupervised_learning_of_an_extensive_and_usable_taxonomy_for_DBpedia) |
| ✎ | *SLHCat: mapping Wikipedia categories and lists to DBpedia* | arXiv:2309.11791 | [abs](https://arxiv.org/abs/2309.11791) |
| ✎ | *TiFi: taxonomy induction for fictional domains* | arXiv:1901.10263 | [PDF](https://arxiv.org/pdf/1901.10263) |
| ✎ | Weikum et al., *Machine Knowledge: creation and curation of comprehensive KBs* | arXiv:2009.11564 | [PDF](https://arxiv.org/pdf/2009.11564) |

## F. Hierarchy-aware embedding geometry (conceptual borrowing only)

| Mark | Ref | Venue | Link |
|---|---|---|---|
| ✎ | Nickel & Kiela, *Poincaré embeddings for learning hierarchical representations* | NeurIPS 2017 | (find galley) |
| ✎ | Ganea, Bécigneul & Hofmann, *Hyperbolic entailment cones for learning hierarchical embeddings* | ICML 2018 | [PMLR](http://proceedings.mlr.press/v80/ganea18a/ganea18a.pdf) · [arXiv:1804.01882](https://arxiv.org/pdf/1804.01882) |
| ✎ | *Coneheads: hierarchy-aware attention* | arXiv:2306.00392 | [PDF](https://arxiv.org/pdf/2306.00392) |
| ✎ | *Improving local identifiability in probabilistic box embeddings* | arXiv:2010.04831 | [PDF](https://arxiv.org/pdf/2010.04831) |
| ✎ | *Concept2Box: joint geometric embeddings for two-view KGs* | arXiv:2307.01933 | [PDF](https://arxiv.org/pdf/2307.01933) |

## G. Misspelling & character-noise robustness

| Mark | Ref | Venue | Link |
|---|---|---|---|
| ✎◆ | Edizel, Piktus et al., *Misspelling Oblivious Word Embeddings* | NAACL 2019 | [ACL](https://aclanthology.org/N19-1326/) · [S2](https://www.semanticscholar.org/paper/Misspelling-Oblivious-Word-Embeddings-Edizel-Piktus/8fa7d1f3f82526935ba122c20c8d0648506301b3) |
| ✎ | *Robust embeddings via distributions* | arXiv:2104.08420 | [PDF](https://arxiv.org/pdf/2104.08420) |
| ✎ | *NAT: noise-aware training for robust neural sequence labeling* | arXiv:2005.07162 | [PDF](https://arxiv.org/pdf/2005.07162) |
| ✎◆ | Wang et al., *Adversarial GLUE: a multi-task benchmark for robustness evaluation* | NeurIPS 2021 D&B | [arXiv:2111.02840](https://arxiv.org/pdf/2111.02840) |
| ✎ | *Evaluating robustness of LLMs against multilingual typographical errors* | arXiv:2510.09536 | [HTML](https://arxiv.org/html/2510.09536v1) |
| ✎ | *Cheap character noise for OCR-robust multilingual embeddings* | Findings ACL 2025 | [ACL](https://aclanthology.org/2025.findings-acl.609.pdf) |
| ✎ | *Towards optimal adversarial texts: character, word, and sentence* (MOATG) | Cybersecurity, 2025 | [Springer](https://link.springer.com/article/10.1186/s42400-025-00500-3) |
| ✎ | *Evaluating NMC model robustness to noisy inputs and adversarial attacks* | arXiv:2005.00190 | [PDF](https://arxiv.org/pdf/2005.00190) |
| ✎ | *Towards robust and generalizable training: noisy slot filling* | arXiv:2310.03518 | [PDF](https://arxiv.org/pdf/2310.03518) |
| ✎ | *Finite-context indexing of restricted output space for NLP facing noisy input* | arXiv:2310.14110 | [PDF](https://arxiv.org/pdf/2310.14110) |
| ✎ | *Evaluating text classification robustness to POS adversarial examples* | arXiv:2408.08374 | [HTML](https://arxiv.org/html/2408.08374v1) |

*Also needed for implementation (tooling, not prior art): SymSpell deletion-neighborhood
index, BK-trees, Damerau–Levenshtein, Double Metaphone. Verify licences at integration.*

## H. FIS / fuzzy-prototype heads on neural embeddings — **Experiment B's prior art**

| Mark | Ref | Venue | Link |
|---|---|---|---|
| ✎◆ | *Fuzzy Fingerprinting Transformer Language-Models for Emotion Recognition in Conversations* | arXiv:2309.04292 | [abs](https://arxiv.org/abs/2309.04292) · [PDF](https://arxiv.org/pdf/2309.04292v1) |
| ✎◆ | *Fuzzy Fingerprinting Large Pre-trained Models* | EUSFLAT/AGOP 2023, Springer | [chapter](https://link.springer.com/chapter/10.1007/978-3-031-39965-7_20) · [ACM](https://dl.acm.org/doi/10.1007/978-3-031-39965-7_20) |
| ✎◆ | *Fuzzy Fingerprinting Encoder PLMs for ERC: human assessment and validity study* | arXiv:2605.02665 | [HTML](https://arxiv.org/html/2605.02665v1) |
| ✎ | *Fuzzy Fingerprints in Limited Discrete Feature Spaces* | Springer, 2025 | [chapter](https://link.springer.com/chapter/10.1007/978-3-032-29000-7_21) · [ULisboa](https://researchportal.ulisboa.pt/en/publications/fuzzy-fingerprints-inlimited-discrete-feature-spaces/) |

## I. Fuzzy sentiment analysis

| Mark | Ref | Venue | Link |
|---|---|---|---|
| ✎◆ | *The scalable fuzzy inference-based ensemble method for sentiment analysis* | 2022 | [PMC9534613](https://pmc.ncbi.nlm.nih.gov/articles/PMC9534613/) |
| ✎ | *A three-step fuzzy-based BERT model for sentiment analysis* | 2022 | [RG](https://www.researchgate.net/publication/359547225_A_Three-Step_Fuzzy-Based_BERT_Model_for_Sentiment_Analysis) |
| ✎ | *FDiBD: hybrid fuzzy logic + DistilBERT for ambiguity and long-range dependencies* | Discover Appl. Sci., 2025 | [Springer](https://link.springer.com/article/10.1007/s42452-025-08015-9) |
| ✎◆ | *Fuzzy rule based systems for interpretable sentiment analysis* | IEEE conf., 2016 | [IEEE](https://ieeexplore.ieee.org/document/7974497/) · [RG](https://www.researchgate.net/publication/309762308_Fuzzy_Rule_Based_Systems_for_Interpretable_Sentiment_Analysis) |
| ✎ | *Fuzzy-weighted sentiment recognition for educational feedback* | SciTePress, 2025 | [link](https://www.scitepress.org/PublishedPapers/2025/137942/) |
| ✎ | *Fuzzy IS with interpretable fuzzy rules for XAI in disease diagnosis* (review) | Inf. Sci., 2024 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0020025524001257) |
| ✎ | *Fuzzy-logic-based interpretability and explainability of ML: a comprehensive review* | Neurocomputing, 2025 | [10.1016/j.neucom.2025.130602](https://doi.org/10.1016/j.neucom.2025.130602) |
| ✎ | *Using the Tsetlin machine to learn human-interpretable rules for text categorization* | arXiv:1809.04547 | [PDF](https://arxiv.org/pdf/1809.04547) |
| ✎ | *Explaining deep NLP by mining textual interpretable features* | arXiv:2106.06697 | [PDF](https://arxiv.org/pdf/2106.06697) |

## J. Datasets & candidate embedding models

| Mark | Ref | Note | Link |
|---|---|---|---|
| ✎◆ | Socher et al., *Stanford Sentiment Treebank* | 11,855 sentences; 215,154 labeled phrases; 25-level slider; SST-2/SST-5 are discretizations | [Zenodo](https://zenodo.org/records/5256915) · [overview](https://medium.com/data-science/the-stanford-sentiment-treebank-sst-studying-sentiment-analysis-using-nlp-e1a4cad03065) |
| ✎ | *Fine-grained sentiment classification using BERT* | SST-5 baseline numbers | [arXiv:1910.03474](https://arxiv.org/pdf/1910.03474) |
| ✎◆ | **EmbeddingGemma-300M** | 768-d, MRL → 512/256/128; ~622MB; MTEB Eng v2 ≈ 69.67 | [blog](https://developers.googleblog.com/en/introducing-embeddinggemma/) · [docs](https://ai.google.dev/gemma/docs/embeddinggemma) |
| ✎ | Open-source embedding model surveys 2026 | model shortlist for Experiment B | [BentoML](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models) · [Ollama/MTEB table](https://www.morphllm.com/ollama-embedding-models) |
| ✎ | *Granite Embedding Multilingual R2* | alternative small model | [arXiv:2605.13521](https://arxiv.org/pdf/2605.13521) |
| — | `thenlper/gte-small` | already used by `tests/test_textclassifier.py` — zero-setup starting point | — |

## K. Adjacent LLM + fuzzy work (orthogonal; know it exists)

| Mark | Ref | Link |
|---|---|---|
| ✎ | *Fuzzy Reasoning Chain (FRC)* — Findings EMNLP 2025 | [ACL](https://aclanthology.org/2025.findings-emnlp.541.pdf) |
| ✎ | *LLM-as-a-Fuzzy-Judge* | [arXiv:2506.11221](https://arxiv.org/pdf/2506.11221) |
| ✎ | *A fuzzy logic prompting framework for LLMs in adaptive tutoring* | [arXiv:2508.06754](https://arxiv.org/pdf/2508.06754) |
| ✎ | *Chaotic fuzzy-logic-augmented LLM for educational Q&A (CHAQS)* | [Frontiers](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1404940/full) |
| ✎ | *Integrating LLMs with explainable FIS for steel defect detection* | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167865525001096) |

---

## Reading order for the ◆ set

1. **Semantic Fusion with Fuzzy-Membership Features** (B) — closest to Experiment A; read
   first, because it determines what is left to claim.
2. **Fuzzy Fingerprints** trio (H) — Experiment B is a re-run of this architecture; needed
   as baseline and as citation.
3. **Static Fuzzy Bag-of-Words** + original FBoW (B) — prior art for
   "coordinates are memberships".
4. **Open Roget's** + **Jarmasz & Szpakowicz** (E1) — the scaffold and its similarity
   measure; also the source for real level cardinalities.
5. **MRL** (D) — the multi-resolution incumbent to differentiate from.
6. **SPINE** (C) — borrow the word-intrusion interpretability protocol verbatim.
7. **MOE** (G) + **AdvGLUE** (G) — robustness baseline and perturbation generators.
8. **EIA** (A2) — how to elicit membership widths instead of guessing them.

## Verification debt

⚠ Every entry. In priority order, before anything is published:
1. Roget's level cardinalities (6/39/79/596/990) — edition-dependent; re-derive from the
   Open Roget's dump itself, not from a secondary source.
2. Wikipedia→Philosophy rate (>94% / 97.0%) and mean chain length (~23) — two different
   studies, different methodologies, cited loosely above.
3. Author lists for the Fuzzy Fingerprints line (Batista is referenced but the full list
   was not confirmed) and for FBoW (Zhao & Mao attribution is from memory, not search).
4. Every arXiv number above came from a search-result URL; confirm each resolves to the
   stated title.
