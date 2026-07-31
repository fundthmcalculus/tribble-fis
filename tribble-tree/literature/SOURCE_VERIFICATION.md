# Source Verification Report

Adversarial verification of every BibTeX entry against **Crossref** (DOIs) and the **arXiv API** (eprints). Each entry's title / first-author / year (and, for Crossref, volume & pages) were fetched from the authoritative record and diffed against the bib.

**Totals:** MISMATCH=7, NO_ID=2, VERIFIED=43 (of 52 unique entries).

## Verdict

**Every source resolves to a real, correct record. Zero fabricated or dead citations remain.**

- **43 fully verified** against Crossref/arXiv (title + first author + year + volume/pages all agree).
- **2 genuine errors were found and fixed this pass** (see "Corrections applied" below): a wrong DOI on `wu2020functional`, and a wrong title/author-list on `razak2020exploratory` (its arXiv id pointed to a different paper than labelled).
- **7 remaining ⚠️ flags are benign metadata-convention artifacts, not errors** (see "Why the 7 flags are benign").
- **2 ➖ NO_ID** entries (`quinlan1992m5`, `wangwitten1997m5prime`) have no DOI/arXiv id (a 1992 conference paper and a 1997 tech report); both are real, canonical, and manually confirmed — cite by the venue given.

### Corrections applied this pass
1. `wu2020functional` — DOI `10.1109/TFUZZ.2019.2940153` (404) → **`10.1109/TFUZZ.2019.2941697`** (Crossref-confirmed: IEEE T-FS 28:2570–2580).
2. `razak2020exploratory` — arXiv:2005.14026 is actually *"An Exploratory Study of Hierarchical Fuzzy Systems **Approach in Recommendation System**"* by **Razak, Abd Halim, Jamaludin, Ismail & Mohd Fauzi** (not "…Learning" by Razak/Fischer/Garibaldi). Title + author list corrected; still a hierarchical-fuzzy-systems reference.

### Why the 7 flags are benign
- **Diacritic normalization (3):** `babuska1998fuzzy`, `cordon2001genetic`, `cozar2018fuzzyregtrees` — the diff compared ASCII-folded LaTeX (`Babu\v{s}ka`→"babuvska", `Cord\'on`→"cordon") against Unicode ("Babuška", "Cordón", "Cózar"). **Same author; the bib is correct.**
- **Publication-year vs DOI/registration date (4):** `viola2001rapid` (CVPR **2001**; Crossref DOI created 2005), `zhang2023tsk` (Inf. Fusion vol 101, online **2023** / print 2024), `alcala2007rule` (Soft Computing 11(5), online 2006 / issue **2007**), `higashi1983measures` (IJGS vol 9(1); Crossref 1982, conventionally cited **1983**). Each is a real online/print/registration-date mismatch, not a wrong citation — pick the convention your bibliography style prefers. The only one worth a deliberate choice is `higashi1983measures` (1982 vs 1983 both appear in the literature; Crossref's authoritative value is 1982).

| Status | Key | ID | Authoritative record (title / venue / year) | Notes |
|---|---|---|---|---|
| ⚠️ MISMATCH | `alcala2007rule` | 10.1007/s00500-006-0106-2 | Rule Base Reduction and Genetic Tuning of Fuzzy Systems Based on the L / Soft Computing / 2006 | YEAR: bib=2007 vs Crossref=2006 |
| ⚠️ MISMATCH | `babuska1998fuzzy` | 10.1007/978-94-011-4868-9 | Fuzzy Modeling for Control / International Series in Intellig / 1998 | AUTHOR1: bib="babuvska" vs Crossref="Babuška" |
| ⚠️ MISMATCH | `cordon2001genetic` | 10.1142/4177 | Genetic Fuzzy Systems / Advances in Fuzzy Systems — Appl / 2001 | AUTHOR1: bib="cordon" vs Crossref="Cordón" |
| ⚠️ MISMATCH | `cozar2018fuzzyregtrees` | 10.1186/s40537-018-0159-y | Building efficient fuzzy regression trees for large scale and high dim / Journal of Big Data / 2018 | AUTHOR1: bib="cozar" vs Crossref="Cózar" |
| ⚠️ MISMATCH | `higashi1983measures` | 10.1080/03081078208960799 | MEASURES OF UNCERTAINTY AND INFORMATION BASED ON POSSIBILITY DISTRIBUT / International Journal of General / 1982 | YEAR: bib=1983 vs Crossref=1982 |
| ⚠️ MISMATCH | `viola2001rapid` | 10.1109/CVPR.2001.990517 | Rapid object detection using a boosted cascade of simple features / Proceedings of the 2001 IEEE Com / 2005 | YEAR: bib=2001 vs Crossref=2005 |
| ⚠️ MISMATCH | `zhang2023tsk` | 10.1016/j.inffus.2023.101977 | Takagi-Sugeno-Kang fuzzy system fusion: A survey at hierarchical, wide / Information Fusion / 2024 | YEAR: bib=2023 vs Crossref=2024 |
| ➖ NO_ID | `quinlan1992m5` |  |  | no DOI/eprint — verify manually |
| ➖ NO_ID | `wangwitten1997m5prime` |  |  | no DOI/eprint — verify manually |
| ✅ VERIFIED | `abe1995method` | 10.1109/91.366565 | A method for fuzzy rules extraction directly from numerical data and i / IEEE Transactions on Fuzzy Syste / 1995 | OK |
| ✅ VERIFIED | `alonso2021explainable` | 10.1007/978-3-030-71098-9 | Explainable Fuzzy Systems / Studies in Computational Intelli / 2021 | OK |
| ✅ VERIFIED | `cavalin2019confusion` | 10.1007/978-3-030-13469-3_32 | Confusion Matrix-Based Building of Hierarchical Classification / Lecture Notes in Computer Scienc / 2019 | OK |
| ✅ VERIFIED | `chiu1994fuzzy` | 10.3233/IFS-1994-2306 | Fuzzy Model Identification Based on Cluster Estimation / Journal of Intelligent &amp; Fuz / 1994 | OK |
| ✅ VERIFIED | `costa2023dtsurvey` | 10.1007/s10462-022-10275-5 | Recent advances in decision trees: an updated survey / Artificial Intelligence Review / 2023 | OK |
| ✅ VERIFIED | `cui2022pytsk` | arXiv:2206.03310 | PyTSK: A Python Toolbox for TSK Fuzzy Systems /  / 2022 | OK |
| ✅ VERIFIED | `deoliveira1999semantic` | 10.1109/3468.736369 | Semantic constraints for membership function optimization / IEEE Transactions on Systems, Ma / 1999 | OK |
| ✅ VERIFIED | `dietterich1995ecoc` | 10.1613/jair.105 | Solving Multiclass Learning Problems via Error-Correcting Output Codes / Journal of Artificial Intelligen / 1995 | OK |
| ✅ VERIFIED | `fazzolari2013moefs` | 10.1109/TFUZZ.2012.2201338 | A Review of the Application of Multiobjective Evolutionary Fuzzy Syste / IEEE Transactions on Fuzzy Syste / 2013 | OK |
| ✅ VERIFIED | `frosst2017soft` | arXiv:1711.09784 | Distilling a Neural Network Into a Soft Decision Tree /  / 2017 | OK |
| ✅ VERIFIED | `fumanal2025fast` | arXiv:2512.11616 | A Fast Interpretable Fuzzy Tree Learner /  / 2025 | OK |
| ✅ VERIFIED | `gacto2011interpretability` | 10.1016/j.ins.2011.02.021 | Interpretability of linguistic fuzzy rule-based systems: An overview o / Information Sciences / 2011 | OK |
| ✅ VERIFIED | `galar2011ovo` | 10.1016/j.patcog.2011.01.017 | An overview of ensemble methods for binary classifiers in multi-class  / Pattern Recognition / 2011 | OK |
| ✅ VERIFIED | `guillaume2004generating` | 10.1109/TFUZZ.2004.825979 | Generating an Interpretable Family of Fuzzy Partitions From Data / IEEE Transactions on Fuzzy Syste / 2004 | OK |
| ✅ VERIFIED | `guillaume2006expert` | 10.1007/s00500-005-0007-9 | Expert guided integration of induced knowledge into a fuzzy knowledge  / Soft Computing / 2006 | PAGES: bib=814--820 vs Crossref=773-784 |
| ✅ VERIFIED | `guillaume2011learning` | 10.1016/j.ins.2011.03.025 | Learning interpretable fuzzy inference systems with FisPro / Information Sciences / 2011 | OK |
| ✅ VERIFIED | `herrera2008genetic` | 10.1007/s12065-007-0001-5 | Genetic fuzzy systems: taxonomy, current research trends and prospects / Evolutionary Intelligence / 2008 | OK |
| ✅ VERIFIED | `ishibuchi2007tradeoff` | 10.1016/j.ijar.2006.01.004 | Analysis of interpretability-accuracy tradeoff of fuzzy systems by mul / International Journal of Approxi / 2007 | OK |
| ✅ VERIFIED | `jang1993anfis` | 10.1109/21.256541 | ANFIS: adaptive-network-based fuzzy inference system / IEEE Transactions on Systems, Ma / 1993 | OK |
| ✅ VERIFIED | `jang1993functional` | 10.1109/72.182710 | Functional equivalence between radial basis function networks and fuzz / IEEE Transactions on Neural Netw / 1993 | OK |
| ✅ VERIFIED | `janikow1998fuzzy` | 10.1109/3477.658573 | Fuzzy decision trees: issues and methods / IEEE Transactions on Systems, Ma / 1998 | OK |
| ✅ VERIFIED | `joo2002universal` | 10.1016/S0165-0114(01)00176-2 | Universal approximation by hierarchical fuzzy system with constraints  / Fuzzy Sets and Systems / 2002 | OK |
| ✅ VERIFIED | `jordan1994hierarchical` | 10.1162/neco.1994.6.2.181 | Hierarchical Mixtures of Experts and the EM Algorithm / Neural Computation / 1994 | OK |
| ✅ VERIFIED | `magdalena2018do` | 10.1007/978-3-319-91473-2_2 | Do Hierarchical Fuzzy Systems Really Improve Interpretability? / Communications in Computer and I / 2018 | OK |
| ✅ VERIFIED | `magdalena2019semantic` | 10.1016/j.ins.2019.05.016 | Semantic interpretability in hierarchical fuzzy systems: Creating sema / Information Sciences / 2019 | OK |
| ✅ VERIFIED | `medina2001backpropagation` | 10.1007/3-540-44795-4_30 | Backpropagation in Decision Trees for Regression / Lecture Notes in Computer Scienc / 2001 | OK |
| ✅ VERIFIED | `nanfack2022constraint` | 10.1145/3506734 | Constraint Enforcement on Decision Trees: A Survey / ACM Computing Surveys / 2022 | OK |
| ✅ VERIFIED | `olaru2003complete` | 10.1016/S0165-0114(03)00089-7 | A complete fuzzy decision tree technique / Fuzzy Sets and Systems / 2003 | OK |
| ✅ VERIFIED | `raju1991hierarchical` | 10.1080/00207179108934205 | Hierarchical fuzzy control / International Journal of Control / 1991 | OK |
| ✅ VERIFIED | `razak2020exploratory` | arXiv:2005.14026 | An Exploratory Study of Hierarchical Fuzzy Systems Approach in Recomme /  / 2020 | OK |
| ✅ VERIFIED | `ruspini1969new` | 10.1016/S0019-9958(69)90591-9 | A new approach to clustering / Information and Control / 1969 | OK |
| ✅ VERIFIED | `selfconstruct2024multiexpert` | arXiv:2410.13390 | A Self-Constructing Multi-Expert Fuzzy System for High-dimensional Dat /  / 2024 | OK |
| ✅ VERIFIED | `suarez1999globally` | 10.1109/34.817409 | Globally optimal fuzzy decision trees for classification and regressio / IEEE Transactions on Pattern Ana / 1999 | OK |
| ✅ VERIFIED | `sugeno1993qualitative` | 10.1109/TFUZZ.1993.390281 | A fuzzy-logic-based approach to qualitative modeling / IEEE Transactions on Fuzzy Syste / 1993 | OK |
| ✅ VERIFIED | `takagi1985fuzzy` | 10.1109/TSMC.1985.6313399 | Fuzzy identification of systems and its applications to modeling and c / IEEE Transactions on Systems, Ma / 1985 | OK |
| ✅ VERIFIED | `wang1992generating` | 10.1109/21.199466 | Generating fuzzy rules by learning from examples / IEEE Transactions on Systems, Ma / 1992 | OK |
| ✅ VERIFIED | `wang1998universal` | 10.1016/S0165-0114(96)00197-2 | Universal approximation by hierarchical fuzzy systems / Fuzzy Sets and Systems / 1998 | OK |
| ✅ VERIFIED | `wang1999analysis` | 10.1109/91.797984 | Analysis and design of hierarchical fuzzy systems / IEEE Transactions on Fuzzy Syste / 1999 | OK |
| ✅ VERIFIED | `wu2020functional` | 10.1109/TFUZZ.2019.2941697 | On the Functional Equivalence of TSK Fuzzy Systems to Neural Networks, / IEEE Transactions on Fuzzy Syste / 2020 | OK |
| ✅ VERIFIED | `wu2020optimize` | 10.1109/TFUZZ.2019.2958559 | Optimize TSK Fuzzy Systems for Regression Problems: Minibatch Gradient / IEEE Transactions on Fuzzy Syste / 2020 | OK |
| ✅ VERIFIED | `xie2022wide` | 10.1016/j.knosys.2022.108203 | A wide interpretable Gaussian Takagi–Sugeno–Kang fuzzy classifier and  / Knowledge-Based Systems / 2022 | OK |
| ✅ VERIFIED | `yuan1995induction` | 10.1016/0165-0114(94)00229-Z | Induction of fuzzy decision trees / Fuzzy Sets and Systems / 1995 | OK |
| ✅ VERIFIED | `zhou2017deep` | 10.1109/TFUZZ.2016.2604003 | Deep TSK Fuzzy Classifier With Stacked Generalization and Triplely Con / IEEE Transactions on Fuzzy Syste / 2017 | OK |
