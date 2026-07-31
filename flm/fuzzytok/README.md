# `fuzzytok` — a fuzzy tokenizer and a linguistic parameter-space encoder

Two components, aimed at replacing the two weakest joints in
[`../fuzzyembed/`](../fuzzyembed/): the word-level vocabulary (no subword generalisation,
proper names invisible) and the WordNet-derived feature space (unbalanced ladder, English
only, no morphology).

1. **`tokenizer.py`** — a *fuzzy* tokenizer: a surface string maps to a **graded set over
   units**, not one segmentation.
2. **`params.py`** — an encoder onto a small, named **linguistic parameter space**:
   morphosyntax + orthographic shape + coarse semantics + affect, ~70 dimensions, every one
   named a priori.

---

## Why this design, quantitatively

The vocabulary split is not a guess. Measuring the corpus (`../LOG.md` E20):

| token mass covered | types needed | % of vocabulary |
|---|---|---|
| 50% | **54** | 1.8% |
| 80% | 415 | 13.8% |
| 90% | 943 | 31.4% |

**54 word types carry half of all tokens.** So the head of the distribution is tiny and
idiosyncratic — no category or subword decomposition helps with `the`, `of`, `said` — while
the tail is large, regular, and morphologically decomposable. That argues for a **hybrid
vocabulary**: whole-word units for the head, subword units for the tail. It also predicts
the saturation point we already measured independently: lexicalising ~200 words captured
essentially all the available benefit, and ~200 types is ~72% of token mass.

The same measurement explains why context sparsity, not the model, is the binding
constraint: a trigram context is unseen 75% of the time at 67K training tokens. Subword
units attack exactly that — they raise the effective count per unit.

## The fuzzy part

Standard tokenizers commit to one segmentation. BPE (Sennrich et al., 2016) merges greedily;
WordPiece (Schuster & Nakajima, 2012) maximises likelihood; the unigram-LM tokenizer (Kudo,
2018) *can* produce many segmentations with probabilities, and subword regularisation
exploits that by **sampling** one per training step.

This tokenizer keeps all of them at once, with **membership degrees**, and never samples:

```
"unhappily"  ->  un(0.9) + happy(0.8) + ly(0.9)   [morphological]
                 unhapp(0.4) + ily(0.4)           [competing, lower degree]
"recieve"    ->  receive(0.87)                    [fuzzy lexical access]
```

Two things follow that a hard tokenizer cannot offer:

- **Misspelling robustness is native, not bolted on.** `recieve` gets partial membership in
  `receive` because membership is graded to begin with — the mechanism already built in
  `../fuzzyembed/lexical.py`, now applied at the tokenizer rather than after it.
- **The ambiguity is *reported*.** A segmentation's degree is inspectable, so "the model saw
  this as `un + happy`, degree 0.72" is an auditable statement.

Compared to going tokenizer-free at the byte or character level (CANINE, Clark et al. 2022;
ByT5, Xue et al. 2022), which also avoids segmentation commitment, this keeps units
*nameable* — a rule can say `un-` or `-ly`, which a byte model cannot.

## The linguistic parameter space

Rather than WordNet's 45 supersenses (which gave an unbalanced ladder, no morphology, and no
proper nouns), the encoder uses four small named blocks:

| block | dims | basis |
|---|---|---|
| **UPOS** | 17 | Universal Dependencies universal POS tags (de Marneffe et al., 2021) |
| **FEATS** | ~20 | UD morphological features — Number, Tense, Degree, Person, Polarity |
| **Shape** | ~12 | orthographic: capitalisation, length band, suffix/prefix class, digits |
| **Semantics** | ~14 | coarse supersense groups (entity / act / state / quality / relation) |
| **Affect** | 3 | Osgood's semantic differential — evaluation, potency, activity (Osgood et al., 1957) |

Every dimension is graded and named before training, which is the whole premise of the
project. Three specific reasons for these choices:

- **UD is a designed, balanced, cross-lingual inventory** — exactly what WordNet's hypernym
  DAG was not (`../fuzzyembed/hierarchy.py` documents that failure: adjectives have no
  hypernyms, so the ladder jumps 45 → 4527).
- **Shape features cover what the lexicon cannot.** Capitalisation alone recovers most proper
  nouns, which were the dominant residual coverage gap at 96.7% (`../LOG.md` E1).
- **Osgood gives affect in 3 dimensions.** Evaluation/potency/activity explain most affective
  variance across cultures, and evaluation is the axis a sentiment task needs — the thing
  Roget's antonymous pairs would have supplied structurally and WordNet does not.

## Status

Implemented and unit-tested in isolation: the tokenizer's hybrid vocabulary induction, fuzzy
segmentation, and the parameter encoder. **Not yet wired into the joint ranker**, so there is
no perplexity number for it — that is the next step and the only honest thing to say about
its value so far. The comparison to make is against the current 61-dimensional
semantic+syntax space at matched rule budget.

## References

- R. Sennrich, B. Haddow, A. Birch. *Neural Machine Translation of Rare Words with Subword
  Units.* ACL 2016. (BPE.)
- M. Schuster, K. Nakajima. *Japanese and Korean Voice Search.* ICASSP 2012. (WordPiece.)
- T. Kudo. *Subword Regularization: Improving Neural Network Translation Models with Multiple
  Subword Candidates.* ACL 2018. (Unigram-LM tokenizer; multiple segmentations with
  probabilities — the closest prior art to the fuzzy segmentation here.)
- T. Kudo, J. Richardson. *SentencePiece: A simple and language independent subword tokenizer
  and detokenizer.* EMNLP 2018 (demo). [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)
- J. H. Clark, D. Garrette, I. Turc, J. Wieting. *CANINE: Pre-training an Efficient
  Tokenization-Free Encoder for Language Representation.* TACL 2022.
- L. Xue et al. *ByT5: Towards a token-free future with pre-trained byte-to-byte models.*
  TACL 2022.
- M.-C. de Marneffe, C. D. Manning, J. Nivre, D. Zeman. *Universal Dependencies.*
  Computational Linguistics 47(2), 2021. [ACL](https://aclanthology.org/2021.cl-2.11/) ·
  [universaldependencies.org](https://universaldependencies.org/)
- C. E. Osgood, G. J. Suci, P. H. Tannenbaum. *The Measurement of Meaning.* University of
  Illinois Press, 1957. (Semantic differential: evaluation, potency, activity.)
- N. Chomsky, M. Halle. *The Sound Pattern of English.* Harper & Row, 1968. (Distinctive
  features — the precedent for representing a linguistic unit as a bundle of graded named
  features.)
- L. A. Zadeh. *The concept of a linguistic variable and its application to approximate
  reasoning.* Information Sciences, 1975. (Linguistic variables and hedges.)

⚠ Citation caveat consistent with [`../literature/SOURCES.md`](../literature/SOURCES.md):
this session could not reach arXiv, ACL Anthology, or publisher sites, so author lists,
years, and venues above are from memory and search snippets. **Verify before publishing.**
