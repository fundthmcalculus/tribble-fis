# TRIBBLE / hFIS — Structured Source Index

A sortable prior-art index for the literature review, grouped by theme. Companion to
the novelty map [`../HFIS_NOVELTY_REVIEW.md`](../HFIS_NOVELTY_REVIEW.md). BibTeX:
the reviewed core is [`../hfis_review.bib`](../hfis_review.bib); the expansion set
added here is [`sources_extended.bib`](sources_extended.bib).

**Legend**
- ⬇ `file` — open-access PDF downloaded to [`pdfs/`](pdfs/) (filename given).
- 🔗 — link only (paywalled or landing page); no redistributable PDF pulled.
- ★ — in the reviewed core bib (`hfis_review.bib`), verified during the novelty pass.
- ✎ — expansion find (this round); citation confirmed by search, **spot-check DOI/authors before citing**.

Themes: **A** Foundational TSK & consequents · **B** Antecedent generation from data ·
**C** Parameter tuning & interpretability–accuracy · **D** Fuzzy/soft/model trees ·
**E** Mixture of experts · **F** Hierarchical/stacked/deep/wide TSK · **G** Fuzzy
partitions & explainability · **H** Multiclass decomposition & cascades ·
**I** Surveys (cross-cutting) · **J** Tools.

---

## A. Foundational TSK & consequent identification

| ★/✎ | Ref | Venue | Link | PDF |
|---|---|---|---|---|
| ★ | Takagi & Sugeno 1985, *Fuzzy identification of systems…* | IEEE T-SMC SMC-15 | [10.1109/TSMC.1985.6313399](https://doi.org/10.1109/TSMC.1985.6313399) | 🔗 |
| ★ | Jang 1993, *ANFIS: adaptive-network-based FIS* | IEEE T-SMC 23(3) | [10.1109/21.256541](https://doi.org/10.1109/21.256541) | 🔗 |
| ★ | Jang & Sun 1993, *Functional equivalence of RBF networks & FIS* | IEEE T-NN 4(1) | [10.1109/72.182710](https://doi.org/10.1109/72.182710) | 🔗 |
| ★ | Wu, Yuan, Huang & Tan 2020, *MBGD-RDA: optimize TSK for regression* | IEEE T-FS 28(5) | [arXiv:1903.10951](https://arxiv.org/abs/1903.10951) | ⬇ `wu2020optimize_mbgd.pdf` |

## B. Antecedent generation from data (output-first, clustering, rules-from-examples)

| ★/✎ | Ref | Venue | Link | PDF |
|---|---|---|---|---|
| ★ | Sugeno & Yasukawa 1993, *Fuzzy-logic-based approach to qualitative modeling* (cluster output → project to inputs) | IEEE T-FS 1(1) | [10.1109/TFUZZ.1993.390281](https://doi.org/10.1109/TFUZZ.1993.390281) | 🔗 |
| ★ | Wang & Mendel 1992, *Generating fuzzy rules by learning from examples* | IEEE T-SMC 22(6) | [10.1109/21.199466](https://doi.org/10.1109/21.199466) | 🔗 |
| ★ | Chiu 1994, *Fuzzy model identification based on cluster estimation* (subtractive) | J. Intell. Fuzzy Syst. 2(3) | [10.3233/IFS-1994-2306](https://doi.org/10.3233/IFS-1994-2306) | 🔗 |
| ★ | Abe & Lan 1995, *Fuzzy rules extraction… pattern classification* (class regions) | IEEE T-FS 3(1) | [10.1109/91.366565](https://doi.org/10.1109/91.366565) | 🔗 |
| ✎ | Babuška 1998, *Fuzzy Modeling for Control* (GK/Gath-Geva clustering TS id) | Kluwer (book) | [10.1007/978-94-011-4868-9](https://doi.org/10.1007/978-94-011-4868-9) | 🔗 |

## C. Parameter tuning (genetic/PSO/metaheuristic) & interpretability–accuracy

| ★/✎ | Ref | Venue | Link | PDF |
|---|---|---|---|---|
| ★ | Cordón, Herrera, Hoffmann & Magdalena 2001, *Genetic Fuzzy Systems* | World Scientific (book) | [10.1142/4177](https://doi.org/10.1142/4177) | 🔗 |
| ★ | Herrera 2008, *Genetic fuzzy systems: taxonomy…* | Evol. Intell. 1(1) | [10.1007/s12065-007-0001-5](https://doi.org/10.1007/s12065-007-0001-5) | 🔗 |
| ★ | Alcalá, Alcalá-Fdez, Gacto & Herrera 2007, *Rule reduction + 3-tuple genetic tuning* | Soft Comput. 11(5) | [10.1007/s00500-006-0106-2](https://doi.org/10.1007/s00500-006-0106-2) | 🔗 |
| ✎ | Fazzolari, Alcalá, Nojima, Ishibuchi & Herrera 2013, *Review of multiobjective evolutionary fuzzy systems* | IEEE T-FS 21(1) | [10.1109/TFUZZ.2012.2201338](https://doi.org/10.1109/TFUZZ.2012.2201338) | ⬇ `fazzolari2013_moefs_review.pdf` |
| ✎ | Gacto, Alcalá & Herrera 2011, *Interpretability measures of linguistic FRBS* | Inf. Sci. 181(20) | [10.1016/j.ins.2011.02.021](https://doi.org/10.1016/j.ins.2011.02.021) | 🔗 |
| ✎ | Ishibuchi & Nojima 2007, *Interpretability–accuracy tradeoff via MO fuzzy GBML* | Int. J. Approx. Reason. 44(1) | [10.1016/j.ijar.2006.01.004](https://doi.org/10.1016/j.ijar.2006.01.004) | 🔗 |
| ✎ | *EMOFS interpretability–accuracy tradeoff review* 2012 (confirm authors) | Information 3(3):256 | [10.3390/info3030256](https://doi.org/10.3390/info3030256) | 🔗 |
| ✎ | *Zero-order TSK learning via two-phase swarm intelligence* (PSO) | Fuzzy Sets Syst. (2009) | [S0165011408000894](https://www.sciencedirect.com/science/article/abs/pii/S0165011408000894) | 🔗 |
| ✎ | *Training high-order TSK using batch LS + PSO* | Int. J. Fuzzy Syst. (2019) | [10.1007/s40815-019-00747-2](https://doi.org/10.1007/s40815-019-00747-2) | 🔗 |

## D. Fuzzy / soft / model trees

| ★/✎ | Ref | Venue | Link | PDF |
|---|---|---|---|---|
| ★ | Yuan & Shaw 1995, *Induction of fuzzy decision trees* (ambiguity) | Fuzzy Sets Syst. 69(2) | [10.1016/0165-0114(94)00229-Z](https://doi.org/10.1016/0165-0114%2894%2900229-Z) | 🔗 |
| ★ | Janikow 1998, *Fuzzy decision trees: issues & methods* (fuzzy ID3) | IEEE T-SMC-B 28(1) | [10.1109/3477.658573](https://doi.org/10.1109/3477.658573) | 🔗 |
| ★ | Suárez & Lutsko 1999, *Globally optimal fuzzy decision trees* (soft, constant leaves) | IEEE TPAMI 21(12) | [10.1109/34.817409](https://doi.org/10.1109/34.817409) | ⬇ `suarez1999_globally_optimal.pdf` |
| ★ | Medina-Chico, Suárez & Lutsko 2001, *Backpropagation in decision trees for regression* (soft, **linear** leaves) | ECML, LNCS 2167 | [10.1007/3-540-44795-4_30](https://doi.org/10.1007/3-540-44795-4_30) | ⬇ `medina2001_backprop_trees.pdf` |
| ★ | Olaru & Wehenkel 2003, *A complete fuzzy decision tree technique* | Fuzzy Sets Syst. 138(2) | [10.1016/S0165-0114(03)00089-7](https://doi.org/10.1016/S0165-0114%2803%2900089-7) | ⬇ `olaru2003_complete_fuzzy_tree.pdf` |
| ★ | Fumanal-Idocin, Fernandez-Peralta & Andreu-Perez 2025, *A Fast Interpretable Fuzzy Tree Learner* | arXiv | [arXiv:2512.11616](https://arxiv.org/abs/2512.11616) | 🔗 |
| ✎ | Quinlan 1992, *Learning with continuous classes* (M5 model trees) | AI'92 | [Semantic Scholar](https://www.semanticscholar.org/paper/ead572634c6f7253bf187a3e9a7dc87ae2e34258) | 🔗 |
| ✎ | Wang & Witten 1997, *Induction of model trees for continuous classes* (M5′) | Univ. Waikato | [handle/10289/1183](https://researchcommons.waikato.ac.nz/handle/10289/1183) | 🔗 |
| ✎ | Frosst & Hinton 2017, *Distilling a NN into a soft decision tree* | CEUR-WS / arXiv | [arXiv:1711.09784](https://arxiv.org/abs/1711.09784) | ⬇ `frosst_hinton2017_soft_tree.pdf` |
| ✎ | Cózar, Marcelloni, Gámez et al. 2018, *Efficient fuzzy regression trees, large-scale/high-dim* | J. Big Data 5 | [10.1186/s40537-018-0159-y](https://doi.org/10.1186/s40537-018-0159-y) | ⬇ `cozar2018_fuzzy_regression_trees.pdf` |
| ✎ | Costa & Pedreira 2023, *Recent advances in decision trees: an updated survey* | Artif. Intell. Rev. 56 | [10.1007/s10462-022-10275-5](https://doi.org/10.1007/s10462-022-10275-5) | 🔗 |

## E. Mixture of experts

| ★/✎ | Ref | Venue | Link | PDF |
|---|---|---|---|---|
| ★ | Jordan & Jacobs 1994, *Hierarchical mixtures of experts & the EM algorithm* | Neural Comput. 6(2) | [10.1162/neco.1994.6.2.181](https://doi.org/10.1162/neco.1994.6.2.181) | 🔗 ([Hinton copy](https://www.cs.toronto.edu/~hinton/absps/hme.pdf)) |
| ★ | Wu, Lin, Huang & Zeng 2020, *Functional equivalence of TSK to NN, MoE, CART, stacking* | IEEE T-FS 28(10) | [10.1109/TFUZZ.2019.2941697](https://doi.org/10.1109/TFUZZ.2019.2941697) · [arXiv:1903.10572](https://arxiv.org/abs/1903.10572) | ⬇ `wu2020functional.pdf` |
| ✎ | *A Closer Look into Mixture-of-Experts in LLMs* 2024 (modern MoE context) | arXiv | [arXiv:2406.18219](https://arxiv.org/abs/2406.18219) | 🔗 |
| ✎ | *A Self-Constructing Multi-Expert Fuzzy System for high-dim classification* 2024 | arXiv | [arXiv:2410.13390](https://arxiv.org/abs/2410.13390) | 🔗 |

## F. Hierarchical / stacked / deep / wide TSK (dimensionality)

| ★/✎ | Ref | Venue | Link | PDF |
|---|---|---|---|---|
| ★ | Raju, Zhou & Kisner 1991, *Hierarchical fuzzy control* (linear vs exp. rule growth) | Int. J. Control 54(5) | [10.1080/00207179108934205](https://doi.org/10.1080/00207179108934205) | 🔗 |
| ★ | Wang 1998, *Universal approximation by hierarchical fuzzy systems* | Fuzzy Sets Syst. 93(2) | [10.1016/S0165-0114(96)00197-2](https://doi.org/10.1016/S0165-0114%2896%2900197-2) | 🔗 |
| ★ | Wang 1999, *Analysis and design of hierarchical fuzzy systems* | IEEE T-FS 7(5) | [10.1109/91.797984](https://doi.org/10.1109/91.797984) | 🔗 |
| ★ | Joo & Lee 2002, *Universal approximation by hierarchical fuzzy system w/ constraints* | Fuzzy Sets Syst. 130(2) | [10.1016/S0165-0114(01)00176-2](https://doi.org/10.1016/S0165-0114%2801%2900176-2) | 🔗 |
| ★ | Zhou, Chung & Wang 2017, *Deep TSK fuzzy classifier (D-TSK-FC), stacked generalization* | IEEE T-FS 25(5) | [10.1109/TFUZZ.2016.2604003](https://doi.org/10.1109/TFUZZ.2016.2604003) | 🔗 |
| ★ | Zhang, Wang, Zhou et al. 2023, *TSK fuzzy system fusion survey (hierarchical/wide/stacked)* | Inf. Fusion 101 | [10.1016/j.inffus.2023.101977](https://doi.org/10.1016/j.inffus.2023.101977) | 🔗 |
| ✎ | Xie et al. 2022, *Wide interpretable Gaussian TSK fuzzy classifier + incremental learning* | Knowl.-Based Syst. 241 | [10.1016/j.knosys.2022.108203](https://doi.org/10.1016/j.knosys.2022.108203) | 🔗 |
| ✎ | *Deep TSK fuzzy classifier with shared linguistic fuzzy rules* 2018 | IEEE T-FS | [10.1109/TFUZZ.2017.2729507](https://doi.org/10.1109/TFUZZ.2017.2729507) | 🔗 |
| ✎ | Razak, Abd Halim, Jamaludin, Ismail & Mohd Fauzi 2020, *An exploratory study of hierarchical fuzzy systems approach in recommendation system* | arXiv | [arXiv:2005.14026](https://arxiv.org/abs/2005.14026) | ⬇ `razak2020_hfs_exploratory.pdf` |

## G. Fuzzy partitions & explainability

| ★/✎ | Ref | Venue | Link | PDF |
|---|---|---|---|---|
| ★ | Ruspini 1969, *A new approach to clustering* (fuzzy partition) | Inf. Control 15(1) | [10.1016/S0019-9958(69)90591-9](https://doi.org/10.1016/S0019-9958%2869%2990591-9) | 🔗 |
| ★ | de Oliveira 1999, *Semantic constraints for membership function optimization* | IEEE T-SMC-A 29(1) | [10.1109/3468.736369](https://doi.org/10.1109/3468.736369) | 🔗 |
| ★ | Guillaume & Charnomordic 2004, *Generating an interpretable family of fuzzy partitions (HFP)* | IEEE T-FS 12(3) | [10.1109/TFUZZ.2004.825979](https://doi.org/10.1109/TFUZZ.2004.825979) | 🔗 ([HAL landing](https://hal.science/hal-01318299)) |
| ★ | Guillaume & Charnomordic 2006, *Expert guided integration of induced knowledge into a fuzzy KB* | Soft Comput. 10(9) | [10.1007/s00500-005-0007-9](https://doi.org/10.1007/s00500-005-0007-9) | 🔗 |
| ★ | Guillaume & Charnomordic 2011, *Learning interpretable FIS with FisPro* | Inf. Sci. 181(20) | [10.1016/j.ins.2011.03.025](https://doi.org/10.1016/j.ins.2011.03.025) | 🔗 |
| ★ | Higashi & Klir 1983, *Measures of uncertainty & information (nonspecificity)* | Int. J. Gen. Syst. 9(1) | [10.1080/03081078208960799](https://doi.org/10.1080/03081078208960799) | 🔗 |
| ★ | Magdalena 2018, *Do hierarchical fuzzy systems really improve interpretability?* | IPMU, CCIS 853 | [10.1007/978-3-319-91473-2_2](https://doi.org/10.1007/978-3-319-91473-2_2) | 🔗 |
| ✎ | Magdalena 2019, *Semantic interpretability in HFS: semantically decouplable hierarchies* | Inf. Sci. 496 | [10.1016/j.ins.2019.05.016](https://doi.org/10.1016/j.ins.2019.05.016) | 🔗 |
| ✎ | Alonso Moral, Castiello, Magdalena & Mencar 2021, *Explainable Fuzzy Systems* (book) | Springer | [10.1007/978-3-030-71098-9](https://doi.org/10.1007/978-3-030-71098-9) | ⬇ `alonso2021_explainable_fuzzy_book.pdf` |

## H. Multiclass decomposition & cascades

| ★/✎ | Ref | Venue | Link | PDF |
|---|---|---|---|---|
| ★ | Viola & Jones 2001, *Rapid object detection using a boosted cascade* | CVPR 2001 | [10.1109/CVPR.2001.990517](https://doi.org/10.1109/CVPR.2001.990517) | 🔗 |
| ★ | Cavalin & Oliveira 2019, *Confusion matrix-based building of hierarchical classification* | CIARP, LNCS 11401 | [10.1007/978-3-030-13469-3_32](https://doi.org/10.1007/978-3-030-13469-3_32) | 🔗 |
| ✎ | Dietterich & Bakiri 1995, *Solving multiclass problems via error-correcting output codes* | JAIR 2 | [10.1613/jair.105](https://doi.org/10.1613/jair.105) | ⬇ `dietterich1995_ecoc.pdf` |
| ✎ | Galar, Fernández, Barrenechea, Bustince & Herrera 2011, *OVO/OVA ensemble methods for multiclass* | Pattern Recognit. 44(8) | [10.1016/j.patcog.2011.01.017](https://doi.org/10.1016/j.patcog.2011.01.017) | 🔗 |

## I. Surveys (cross-cutting)

| ★/✎ | Ref | Venue | Link | PDF |
|---|---|---|---|---|
| ✎ | Nanfack, Temple & Frénay 2022, *Constraint enforcement on decision trees: a survey* | ACM Comput. Surv. 54(10s) | [10.1145/3506734](https://doi.org/10.1145/3506734) | 🔗 |
| ✎ | *The fusion of deep learning and fuzzy systems: a state-of-the-art survey* 2021 | IEEE T-FS | [10.1109/TFUZZ.2021.3062899](https://doi.org/10.1109/TFUZZ.2021.3062899) | 🔗 |
| ✎ | Talpur et al. 2023, *Deep neuro-fuzzy systems: a systematic survey* | Artif. Intell. Rev. | [PMC9005344](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9005344/) | 🔗 |
| ✎ | *Fuzzy-logic-based interpretability & explainability of ML: a comprehensive review* 2025 | Neurocomputing | [10.1016/j.neucom.2025.130602](https://doi.org/10.1016/j.neucom.2025.130602) | 🔗 |

## J. Tools

| ★/✎ | Ref | Venue | Link | PDF |
|---|---|---|---|---|
| ✎ | Cui, Wu et al. 2022, *PyTSK: a Python toolbox for TSK fuzzy systems* | arXiv | [arXiv:2206.03310](https://arxiv.org/abs/2206.03310) | ⬇ `pytsk_toolbox.pdf` |
| ✎ | Guillaume & Charnomordic, *FisPro* (interpretable FIS software) | — | [fispro.org](https://www.fispro.org/) | 🔗 |

---

### Downloaded PDFs (12, in [`pdfs/`](pdfs/))

`alonso2021_explainable_fuzzy_book.pdf` · `cozar2018_fuzzy_regression_trees.pdf` ·
`dietterich1995_ecoc.pdf` · `fazzolari2013_moefs_review.pdf` ·
`frosst_hinton2017_soft_tree.pdf` · `medina2001_backprop_trees.pdf` ·
`olaru2003_complete_fuzzy_tree.pdf` · `pytsk_toolbox.pdf` ·
`razak2020_hfs_exploratory.pdf` · `suarez1999_globally_optimal.pdf` ·
`wu2020functional.pdf` · `wu2020optimize_mbgd.pdf`

### Not downloaded (paywalled / no open galley) — links above
IEEE Xplore, ScienceDirect, Springer, and World Scientific items are 🔗 link-only.
Author/HAL copies exist for a few (Jordan–Jacobs via Hinton's page is linked; Guillaume
HFP has a HAL landing page but no public PDF galley).

### To confirm before citing (✎ items)
Authors/DOIs for the ✎ expansion entries were confirmed by search but not adversarially
verified like the ★ core. Spot-check especially: the *EMOFS review* (Information 3(3),
authors), *Deep TSK w/ shared rules* (exact DOI), and the two 2021/2025 fusion/XAI
surveys (author lists).
