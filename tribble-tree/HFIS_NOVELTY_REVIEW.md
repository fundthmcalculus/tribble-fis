# Hierarchical FIS — Prior-Art & Novelty-Defense Literature Map for TRIBBLE

**Purpose.** A claim-by-claim map of the closest prior art to each mechanism in the
TRIBBLE fuzzy-inference approach, with a novelty assessment per mechanism, so the
PhD literature review / novelty defense can be written on top of it. This is a
*literature scaffold*, not the prose review. Companion bibliography:
[`hfis_review.bib`](hfis_review.bib).

**How this was produced.** A fan-out deep-research pass (6 search angles → 29
fetched sources → 101 extracted claims → 25 adversarially verified, 24 confirmed /
1 refuted), then a targeted second pass to fill the four mechanisms the first pass
left under-covered (VariablePlan, Ruspini, cascade, universal-approximation). Every
citation carries a verification status.

**Verification legend.**
`[V]` = adversarially verified this session (≥2/3 confirming votes) or confirmed
directly from the publisher/index page. `[S]` = standard, well-known DOI for a
seminal paper, machine-confirmed by search but spot-check the DOI at proof stage.

**The one-line thesis for the defense.** *Every individual building block of
TRIBBLE is established prior art. TRIBBLE's defensible novelty is the specific
composition* — an exact closed-form firing-weighted **ridge** consequent solve used
as the single inner solver across a **soft fuzzy tree with first-order TSK sub-FIS
leaves**, wrapped in an **automatic-but-overridable declarative structure layer
(`VariablePlan`)**, unified with an **HME-style gated blend**, and exportable to an
explicit **Ruspini triangular partition-of-unity** rule base tuned by
population/line-search metaheuristics. No single prior work combines these; the
novelty argument must be made at the level of the *integrated architecture and the
consequent-solver-as-shared-primitive*, not any one mechanism.

---

## Mechanism 1 — Flat TSK base + closed-form firing-weighted ridge consequents

**Closest prior art.**
- Takagi & Sugeno 1985 `[V]` — the TSK model itself (fuzzy antecedents, input-linear
  consequents). [DOI 10.1109/TSMC.1985.6313399](https://doi.org/10.1109/TSMC.1985.6313399)
- Jang 1993 (ANFIS) `[V]` — the neuro-fuzzy hybrid that identifies consequents by
  least-squares and antecedents by gradient descent; the canonical thing TRIBBLE's
  solver is contrasted against. [DOI 10.1109/21.256541](https://doi.org/10.1109/21.256541)
- Wu et al. 2020 (MBGD-RDA) `[V]` — gives the canonical taxonomy of TSK optimization
  (evolutionary; gradient/neuro-fuzzy; GD+LSE as in ANFIS) and situates
  ridge/regularized consequent learning. [arXiv:1903.10951](https://arxiv.org/abs/1903.10951)

**Novelty assessment.** TSK, Tikhonov/ridge least squares, and *firing-strength-
weighted* least-squares consequent identification are each individually standard.
What is defensible as new is the **packaging**: treating the exact closed-form
firing-weighted ridge solve as *the* optimum for fixed antecedents, selected by
k-fold CV, and reused as the identical inner solver everywhere (flat FIS, tree
leaves, HME experts). ⚠️ **Do not** motivate this by "ANFIS LSE overfits" — that
specific premise was **refuted** in verification (Wu et al. do not support it).
Motivate it instead by exactness + regularization + reuse-as-primitive.

**Closest competitor to distinguish against:** ANFIS (Jang 1993).

---

## Mechanism 1b — Output-first segmentation → per-feature conditional Gaussian-mixture antecedents

*(The antecedent-**generation** recipe, distinct from §1's consequent solver: `partition_output`
runs `pd.qcut` to split the **output** `y` into equal-frequency buckets — bins = classes for
classification, output quantiles for regression — then `fit_gaussians` fits, per (feature, bucket)
pair, a **1-D Gaussian mixture** by KMeans + per-cluster normal fit. Inputs are therefore modeled
**independently per feature, conditioned on the output segment**; the bucket mean is kept as the
segment's output prototype. One implicit rule per output segment.)*

**Closest prior art.**
- Sugeno & Yasukawa 1993 `[V]` — **the direct conceptual ancestor**: cluster the
  **output** (fuzzy c-means on `y`), then **project each output cluster back onto the
  input space** to induce input membership functions. "Segment the output first, then
  derive input MFs" is exactly their program.
  [DOI 10.1109/TFUZZ.1993.390281](https://doi.org/10.1109/TFUZZ.1993.390281)
- Wang & Mendel 1992 `[V]` — generate one rule per example after partitioning **both**
  input and output spaces into fuzzy regions; the foundational rules-from-data method
  in which the output *is* partitioned. [DOI 10.1109/21.199466](https://doi.org/10.1109/21.199466)
- Chiu 1994 `[V]` — subtractive clustering identifies TSK rules from clusters in the
  **joint input–output** space (one cluster → one rule).
  [DOI 10.3233/IFS-1994-2306](https://doi.org/10.3233/IFS-1994-2306)
- Jang & Sun 1993 `[V]` — functional equivalence of Gaussian RBF networks and a class
  of TSK systems: the theoretical license for using per-feature **Gaussian mixtures**
  as TSK antecedents. [DOI 10.1109/72.182710](https://doi.org/10.1109/72.182710)

**Novelty assessment.** Output-driven fuzzy modeling is **not novel** — Sugeno–Yasukawa
1993 already segments the output and derives input MFs from it, and cite it as such.
TRIBBLE's distinguishing choices are: (a) **equal-frequency quantile output bins**
(`qcut`) with kept bucket means as output prototypes, rather than FCM clusters on `y`;
(b) inputs fit as **independent, per-feature 1-D Gaussian mixtures conditioned on the
bin** — a *naive-Bayes-like factorization* — rather than Sugeno–Yasukawa's
cluster-projection or Chiu's joint-space clustering (this is the literal "GMM applied
to individual inputs"); (c) as a consequence, rule/parameter count grows **linearly**
in (#buckets × features) with **no joint input-space grid or clustering**, sidestepping
the curse of dimensionality at the *generation* stage (ties to §6); (d) one unified
construction covers classification (bins = classes) and regression (bins = output
quantiles), feeding directly into the §1 closed-form ridge TSK. ⚠️ **Honesty caveat:**
for the *classifier* case, per-class per-feature Gaussian fitting is essentially
**class-conditional density estimation** — the generative / Gaussian-naïve-Bayes view
(the repo's own `iris-v2` note already observed product t-norm + priors + 1/σ ≈
GaussianNB). So frame the novelty as the *quantile-conditioned, per-feature-factorized
mixture generator wired to a ridge-TSK / hierarchical stack*, **not** as output-first
modeling per se, and cite Sugeno–Yasukawa up front to preempt the obvious objection.

**Closest competitor:** Sugeno–Yasukawa 1993 (output-clustering → input-projection);
for the classifier, Abe & Lan 1995 `[V]` — fuzzy rules extracted directly per **class
region** from numerical data ([DOI 10.1109/91.366565](https://doi.org/10.1109/91.366565))
— and the class-conditional-Gaussian / naïve-Bayes lineage generally.

---

## Mechanism 2 — Antecedent refinement (local descent + population metaheuristics)

**Closest prior art.**
- Cordón, Herrera, Hoffmann & Magdalena 2001 `[V]` — the canonical Genetic Fuzzy
  Systems monograph; evolutionary tuning/learning of fuzzy knowledge bases is mature
  prior art. [DOI 10.1142/4177](https://doi.org/10.1142/4177)
- Herrera 2008 `[V]` — GFS taxonomy; explicitly separates *genetic tuning* (adjust an
  existing KB's MF parameters a posteriori, rule base unchanged) from *genetic
  learning*. TRIBBLE's (μ,σ) refinement is squarely "genetic tuning."
  [DOI 10.1007/s12065-007-0001-5](https://doi.org/10.1007/s12065-007-0001-5)
- Alcalá, Alcalá-Fdez, Gacto & Herrera 2007 `[V]` — 3-tuple MF tuning that shrinks
  the search space and co-optimizes rule selection + tuning ("positive synergy":
  accuracy up, rule base smaller). [DOI 10.1007/s00500-006-0106-2](https://doi.org/10.1007/s00500-006-0106-2)

**Novelty assessment.** GA/PSO/DE/ACO tuning of MF parameters under a CV fitness is
prior art; even tuning-plus-reduction synergy is anticipated (Alcalá 2007). The
genuinely distinguishing pieces are narrow and worth stating precisely: (a) an
explicit **never-worse acceptance guard** against a held-out fold; (b) **local
block/coordinate descent and population metaheuristics as interchangeable refiners
under one CV fitness**; and (c) the project's empirical finding that **local L-BFGS-B
tuning beats GA/DE here because the population methods overfit the CV estimate** —
an effectiveness result, not an architectural claim.

**Closest competitor:** Alcalá et al. 2007 (genetic tuning + rule reduction).

---

## Mechanism 3 — Soft hierarchical fuzzy tree with TSK leaves

**Closest prior art.**
- Janikow 1998 `[V]` — fuzzy-ID3 induction; source of TRIBBLE's information-gain split
  option. [DOI 10.1109/3477.658573](https://doi.org/10.1109/3477.658573)
- Yuan & Shaw 1995 `[V]` — the classification-**ambiguity** split criterion (built on
  Higashi–Klir nonspecificity, below). [DOI 10.1016/0165-0114(94)00229-Z](https://doi.org/10.1016/0165-0114(94)00229-Z)
- Suárez & Lutsko 1999 `[V]` — fuzzifies a CART skeleton with sigmoidal soft splits
  optimized globally on the tree; **crucially, its leaves are constants**, not
  first-order TSK. [DOI 10.1109/34.817409](https://doi.org/10.1109/34.817409)
- Olaru & Wehenkel 2003 `[V]` — single-attribute soft splits with cut-point α and
  overlap width β; overlap samples propagate down **both** successors; leaves
  aggregated by defuzzification — i.e. the soft multi-path mechanism itself.
  [PDF](https://services.montefiore.uliege.be//stochastic/pubs/2003/OW03/OW03.pdf)
- Fumanal-Idocin, Fernandez-Peralta & Andreu-Perez 2025 `[V]` — closest *recent*
  competitor: a fast greedy fuzzy tree targeting sensible linguistic partitions +
  small rule bases. **The suspect arXiv id 2512.11616 is CONFIRMED correct**, not
  fabricated. [arXiv:2512.11616](https://arxiv.org/abs/2512.11616)

**Novelty assessment.** The soft-split mechanism (single variable, partial
membership, multi-path defuzzified aggregation) is **fully anticipated** by
Olaru–Wehenkel and Suárez–Lutsko — do not claim it as novel. TRIBBLE's distinguishing
features inside the tree are: (a) **first-order TSK linear leaves solved by
firing-weighted ridge**, whereas the classic globally-optimal soft-tree line uses
*constant* leaves; (b) **multi-term linguistic splits** (not just binary sigmoids);
(c) **open-shouldered trapezoids** so out-of-range points still fire a leaf (no
silent predict-0 collapse); (d) a **firing-weighted CART-style variance-reduction
criterion offered alongside** Yuan–Shaw and Janikow criteria as selectable options.

- Medina-Chico, Suárez & Lutsko 2001 `[V]` — **the single most important competitor,
  now verified.** A soft CART tree with **linear predictors at the leaves**, made
  continuous by soft internal decisions and optimized globally by backpropagation. This
  **directly anticipates TRIBBLE's linear-leaf soft tree.**
  [DOI 10.1007/3-540-44795-4_30](https://doi.org/10.1007/3-540-44795-4_30) ·
  [PDF](https://jimlutsko.github.io/files/Medina-Chico2001_Chapter_BackpropagationInDecisionTrees.pdf)

**Closest competitor:** **Medina-Chico et al. 2001** (soft tree, linear leaves) —
Olaru–Wehenkel 2003 / Suárez–Lutsko 1999 for the soft-split mechanism.
⚠️ **Claim (a) narrows:** linear leaves on a soft tree are *not* new (Medina-Chico
2001). What survives as distinct is **which** linear leaf — a *firing-weighted ridge
TSK sub-model solved in closed form* (Medina-Chico fits leaves jointly by backprop, not
a per-leaf exact ridge solve) — plus multi-term linguistic splits and open shoulders.
Distinguish on the **closed-form ridge TSK leaf**, not on "linear leaves."

---

## Mechanism 4 — Automatic-but-overridable structure (`VariablePlan`)

**Closest prior art.**
- Nanfack, Temple & Frénay 2022 `[V]` — *Constraint Enforcement on Decision Trees: A
  Survey*. Establishes **constrained tree induction as a whole subfield**:
  attribute-level constraints (which feature may split where), monotonicity, hierarchy,
  cost, and instance-level must-link/cannot-link, enforced by constraint-aware
  recursive partitioning. TRIBBLE's per-node variable pinning is an *attribute-level
  structural constraint* in this taxonomy. [DOI 10.1145/3506734](https://doi.org/10.1145/3506734)
- Guillaume & Charnomordic 2006 / 2011 (FisPro) `[S]` — expert-guided integration of
  induced knowledge into a fuzzy KB; toolchain mixing automatic induction with expert
  control. [2006 DOI 10.1007/s00500-005-0007-9](https://doi.org/10.1007/s00500-005-0007-9) ·
  [2011 DOI 10.1016/j.ins.2011.03.025](https://doi.org/10.1016/j.ins.2011.03.025)

**Novelty assessment.** ⚠️ **This claim narrows.** Injecting expert/structural
constraints into tree induction is an established subfield (Nanfack et al. 2022), and
expert-guided fuzzy KB construction exists (Guillaume–Charnomordic). So "user can steer
the structure" is *not* novel. What has **no verified direct precedent** is the specific
artifact: a **declarative, JSON-serializable `VariablePlan` with a total precedence
lattice (path-pin > level-order > auto-by-criterion), per-node path pins, per-level
variable order, and exclusions, applied to *fuzzy-tree / HME* structure** — i.e. a
reproducible, serializable *specification* of the constraint set rather than
interactive editing or a single constraint type. Claim the **declarative-plan artifact
+ precedence semantics for fuzzy hierarchical structure**, and cite Nanfack et al. as
the general frame you specialize.

**Closest competitor:** constrained decision-tree induction (Nanfack et al. 2022) /
FisPro expert-in-the-loop design — distinguish on *declarative serializable precedence
plan over fuzzy tree/HME structure*.

---

## Mechanism 5 — Hierarchical mixture of fuzzy experts (HME)

**Closest prior art.**
- Jordan & Jacobs 1994 `[V]` — the HME architecture: a tree of soft (softmax GLIM)
  gates over GLIM leaf experts, fit by maximum likelihood via EM. This is exactly the
  base TRIBBLE names its "Jordan–Jacobs EM loop."
  [DOI 10.1162/neco.1994.6.2.181](https://doi.org/10.1162/neco.1994.6.2.181)
- Wu, Lin, Huang & Zeng 2020 `[V]` — **proves the functional equivalence of TSK fuzzy
  systems to (among others) the mixture of experts.** This is the paper that narrows
  the HME claim: a normalized-firing TSK system *is already* a mixture of experts with
  fuzzy-membership gates, so "TSK experts under fuzzy gates" is, at the equivalence
  level, established. [DOI 10.1109/TFUZZ.2019.2940153](https://doi.org/10.1109/TFUZZ.2019.2940153)

**Novelty assessment.** ⚠️ **This claim narrows the most.** Gated soft routing, greedy
growth, and EM are HME prior art (Jordan–Jacobs), *and* Wu et al. 2020 show TSK ≡ MoE —
so the fuzzy-gate + TSK-expert substitution is not, by itself, novel. What survives:
(a) the **hierarchical, tree-structured** composition (a *depth->1* tree of fuzzy gates
whose leaves are **full multi-rule TSK sub-FIS**, not single normalized-firing units —
the equivalence is at the single-layer level, not this nesting); (b) **soft-inclusion**
overlapping training sets at gate boundaries as the fitting rule; (c) the shared
closed-form ridge solve for experts. The EM loop is currently *design-only*
(`EM_REFINEMENT.md`), so do **not** rest novelty on the estimator. Frame it as
*hierarchical composition of TSK sub-FIS as HME experts*, explicitly conceding the
single-layer TSK≡MoE equivalence.

**Closest competitor:** Jordan–Jacobs 1994 (architecture) + Wu et al. 2020 (TSK≡MoE
equivalence). *A dedicated "fuzzy mixture of experts / hybrid fuzzy-neural HME" (e.g.
HHFNN) search could narrow it further; treat the nesting claim as promising-not-proven.*

---

## Mechanism 6 — Hierarchical / stacked TSK for dimensionality

**Closest prior art.**
- Raju, Zhou & Kisner 1991 `[V]` — foundational hierarchical fuzzy control: structure
  rules hierarchically so rule count grows **linearly, not exponentially**, in the
  number of variables. [DOI 10.1080/00207179108934205](https://doi.org/10.1080/00207179108934205)
  *(bibliographic fields verified via index; article body paywalled.)*
- Zhou, Chung & Wang 2017 (D-TSK-FC) `[V]` — deep/stacked TSK via Wolpert stacked
  generalization; controls per-layer rule count with **random** feature selection +
  a fixed 5-partition grid. [DOI 10.1109/TFUZZ.2016.2604003](https://doi.org/10.1109/TFUZZ.2016.2604003)
- Zhang, Wang, Zhou et al. 2023 `[V]` — the Information Fusion survey of TSK fusion at
  hierarchical/wide/stacked levels. [DOI 10.1016/j.inffus.2023.101977](https://doi.org/10.1016/j.inffus.2023.101977)
  ⚠️ **Correction for your notes:** the project reference list attributes this to
  "H. Wang et al."; the actual first author is **Y. Zhang** (Wang is a co-author).

**Novelty assessment.** Routing few variables per sub-model to dodge exponential
blow-up is the explicit Raju-1991 contribution; deep/stacked TSK complexity control is
D-TSK-FC prior art. TRIBBLE differs by **data-driven per-sub-FIS variable routing**
(chosen by split criterion / differentiation prefilter) rather than random feature
selection or a fixed manual hierarchy, and by keeping **interpretable TSK experts**
rather than random rule combinations. But note the README's own honest caveat: vs the
already-compact flat TRIBBLE base the tree does *not* reduce rule count — so frame the
dimensionality claim about the **HME/sub-FIS routing**, not the single tree.

**Closest competitor:** Zhou–Chung–Wang 2017 (D-TSK-FC).

---

## Mechanism 7 — Ruspini triangular partition-of-unity export

**Closest prior art.**
- Ruspini 1969 `[S]` — origin of the fuzzy partition (memberships summing to 1).
  [DOI 10.1016/S0019-9958(69)90591-9](https://doi.org/10.1016/S0019-9958(69)90591-9)
- de Oliveira 1999 `[S]` — semantic constraints for MF optimization; the canonical
  "keep it a valid interpretable (strong) partition while you tune it" argument.
  [DOI 10.1109/3468.736369](https://doi.org/10.1109/3468.736369)
- Guillaume & Charnomordic 2004 (HFP) `[S]` — generating an interpretable family of
  (strong) fuzzy partitions from data, self-selecting granularity per feature.
  [DOI 10.1109/TFUZZ.2004.825979](https://doi.org/10.1109/TFUZZ.2004.825979)

**Novelty assessment.** Strong/Ruspini triangular partitions of unity and their use
for interpretability are **well-established** (Ruspini; de Oliveira; Guillaume–
Charnomordic) — the partition idea is *not* novel and should be cited, not claimed.
TRIBBLE's distinguishing move is the **pipeline**: converting an *implicit per-class
Gaussian mixture* into an *explicit shared triangular Ruspini rule base* via
data-driven landmark-merging + term-to-rule matching, then refining the **apex knots
directly** (a low-dimensional, monotone, partition-preserving search) with a
**line-search / GA** because the knot objective is piecewise-linear (gradient methods
stall). The "refine knots, not MF params, to keep partition-of-unity for free" framing
is the crisp, defensible contribution. ⚠️ Knot/breakpoint optimization of triangular
MFs likely has precedent — search "breakpoint/knot optimization strong fuzzy partition"
before claiming the refinement itself is new.

**Closest competitor:** Guillaume–Charnomordic 2004 (HFP) — distinguish on
*export-from-a-trained-Gaussian-model + knot-only metaheuristic refinement*.

---

## Mechanism 8 — Confusion-matrix-driven cascade of fuzzy specialists

**Closest prior art.**
- Cavalin & Oliveira 2019 `[V]` — **the direct precedent, now verified.** Automatically
  **builds a hierarchical classifier from a flat classifier's confusion matrix**:
  transform the confusion matrix, compute class-similarity, group confused classes, and
  train per-group base classifiers. The "specialize where classes are confused" idea is
  exactly this. [DOI 10.1007/978-3-030-13469-3_32](https://doi.org/10.1007/978-3-030-13469-3_32)
- Viola & Jones 2001 `[S]` — the canonical boosted **cascade** (stagewise filtering /
  later-stage specialization). [DOI 10.1109/CVPR.2001.990517](https://doi.org/10.1109/CVPR.2001.990517)

**Novelty assessment.** ⚠️ **This claim narrows.** Confusion-matrix-driven construction
of hierarchical/layered classifiers is established (Cavalin–Oliveira 2019, and a broader
line of confusion-matrix class-grouping methods), so "specialize on confused classes" is
not novel. TRIBBLE's residual, distinctive choices are: (a) **binary specialists keyed
to the single largest off-diagonal *cell*** of the *running* confusion matrix (a
sequential, greedy, one-pair-at-a-time refinement) rather than a one-shot grouping into
disjoint blocks; (b) **fuzzy specialists with per-expert anomaly thresholds** that can
abstain; (c) integration into the same fuzzy-TSK model family. Claim the
*sequential single-cell fuzzy specialist with abstention*, and cite Cavalin–Oliveira as
the confusion-matrix-hierarchy frame.

**Closest competitor:** Cavalin & Oliveira 2019 (confusion-matrix hierarchy) — distinguish
on *sequential single-cell binary fuzzy specialists with anomaly abstention*.

---

## Mechanism 9 — Universal approximation, interpretability-vs-accuracy, XAI

**Closest prior art.**
- Wang 1998 `[S]` — hierarchical fuzzy systems are universal approximators.
  [DOI 10.1016/S0165-0114(96)00197-2](https://doi.org/10.1016/S0165-0114(96)00197-2)
- Wang 1999 `[S]` — analysis & design of hierarchical fuzzy systems (approximation +
  gradient tuning). [DOI 10.1109/91.797984](https://doi.org/10.1109/91.797984)
- Joo & Lee 2002 `[S]` — universal approximation for a constrained hierarchical fuzzy
  system (previous-layer outputs only in THEN-parts).
  [DOI 10.1016/S0165-0114(01)00176-2](https://doi.org/10.1016/S0165-0114(01)00176-2)
- Magdalena 2018 `[S]` — *"Do Hierarchical Fuzzy Systems Really Improve
  Interpretability?"* — the essential skeptical counterpoint: hierarchy only helps
  interpretability when intermediate variables are *meaningful*.
  [DOI 10.1007/978-3-319-91473-2_2](https://doi.org/10.1007/978-3-319-91473-2_2)
- Higashi & Klir 1983 `[V]` — nonspecificity / U-uncertainty, underpinning the
  Yuan–Shaw ambiguity criterion. [DOI 10.1080/03081078208960799](https://doi.org/10.1080/03081078208960799)

**Novelty assessment.** Universal approximation of hierarchical fuzzy systems is
settled theory — cite it to justify expressiveness, never claim it. Magdalena 2018 is
the paper your defense must engage head-on: it argues hierarchy ≠ interpretability
unless intermediate variables carry meaning. TRIBBLE's answer is that its hierarchy
splits on **named original input variables** (not synthetic intermediates), which is
exactly the condition Magdalena says is required — a strong, citable rebuttal to the
obvious reviewer objection.

**Closest competitor / must-engage:** Magdalena 2018.

---

## Empirical effectiveness — how to benchmark TRIBBLE credibly

- **Datasets & baselines** standard in this literature: UCI **Concrete Compressive
  Strength** (regression, report R²/RMSE), **wine**, **iris**, and **phishing**
  (classification accuracy/F1); usual baselines are **ANFIS, CART/C4.5 (crisp),
  M5 model trees, flat TSK**, and — for the recent line — the fast interpretable fuzzy
  tree (Fumanal-Idocin 2025) and D-TSK-FC.
- **The canonical "soft beats crisp" result** `[V]`: Olaru & Wehenkel 2003 evaluated
  soft decision trees on **11 UCI datasets** with repeated 10-fold CV + significance
  tests and found SDTs significantly more accurate than crisp trees (lower variance)
  *while keeping tree interpretability*. This is the template for your accuracy-vs-
  interpretability argument.
- **Quantified tradeoff** `[V]`: Alcalá et al. 2007 show rule-selection + tuning
  improving accuracy *while shrinking* the rule base — the shape of result reviewers
  expect.
- **TRIBBLE's own current numbers** (from the repo, external-lit-unverified, for
  orientation): Concrete R² ≈ 0.658 flat → 0.746 (1st-order tree) → **0.791 (HME)**;
  PhiUSIIL phishing acc ≈ 0.998 flat → 0.968–0.969 (tree) → 0.996 (HME); antecedent
  refinement lifts Concrete R² ~0.88→0.92; Ruspini classifier reaches/exceeds the
  Gaussian base on breast-cancer. **Recommendation:** report the tree/HME as an
  *interpretability-for-accuracy trade* (it does not beat the flat base on accuracy),
  mirroring Olaru–Wehenkel's framing, and add ANFIS + CART/M5 baselines on the same
  splits for a defensible comparison.
- ⚠️ Numeric figures for competing methods were **not** independently re-verified —
  pull them from each source paper before quoting.

---

## Citation verification status & to-do

**Verified this session (`[V]`, safe to cite as-is):** Takagi–Sugeno 1985; Jang 1993;
Wu et al. 2020 (MBGD-RDA); Sugeno–Yasukawa 1993; Wang–Mendel 1992; Chiu 1994;
Jang–Sun 1993; Abe–Lan 1995; Cordón et al. 2001; Herrera 2008; Alcalá et al. 2007;
Janikow 1998; Yuan–Shaw 1995; Suárez–Lutsko 1999; Olaru–Wehenkel 2003;
**Medina-Chico et al. 2001**; Fumanal-Idocin et al. 2025 (**arXiv:2512.11616 confirmed
correct**); Jordan–Jacobs 1994; **Wu et al. 2020 (TSK≡MoE equivalence)**; Raju et al.
1991 (fields via index); Zhou–Chung–Wang 2017; Zhang et al. 2023;
**Nanfack et al. 2022**; **Cavalin–Oliveira 2019**; Higashi–Klir 1983.

**Standard DOIs, machine-confirmed but spot-check at proof (`[S]`):** Ruspini 1969;
de Oliveira 1999; Guillaume–Charnomordic 2004 & 2006 & 2011; Viola–Jones 2001;
Wang 1998 & 1999; Joo–Lee 2002; Magdalena 2018.

**Corrections / cautions for the defense:**
1. The survey your reference list calls *"H. Wang et al. 2023"* is **Zhang et al.
   2023** (Information Fusion 101:101977) — fix the author.
2. **Do not** use "ANFIS LSE overfits regression" to motivate the ridge solver — that
   premise was refuted in verification.

**Previously-open searches — now RESOLVED (all four gaps closed, and three *narrow*
TRIBBLE's claims — better found here than by a reviewer):**
- **§3 linear-leaf soft tree** → **Medina-Chico et al. 2001** verified: soft CART with
  *linear leaf predictors* + backprop. The linear-leaf idea is **not new**; claim the
  *closed-form ridge TSK leaf* instead.
- **§5 fuzzy MoE / TSK experts** → **Wu et al. 2020** verified: TSK ≡ Mixture of Experts
  (functional equivalence). The fuzzy-gate+TSK-expert substitution is **not new** at the
  single-layer level; claim the *hierarchical nesting of multi-rule TSK sub-FIS +
  soft-inclusion*.
- **§4 constrained structure** → **Nanfack et al. 2022** verified: constrained tree
  induction is a whole subfield. "User steers structure" is **not new**; claim the
  *declarative, serializable `VariablePlan` precedence artifact for fuzzy hierarchies*.
- **§8 confusion-matrix cascade** → **Cavalin–Oliveira 2019** verified: hierarchies are
  routinely built from confusion matrices. "Specialize on confused classes" is **not
  new**; claim the *sequential single-cell binary fuzzy specialist with anomaly
  abstention*.
- **§7 Ruspini knot tuning** → covered by **de Oliveira 1999** (tune-while-keeping-a-
  valid-partition) already cited; the residual is the *knot-only, partition-preserving
  metaheuristic refinement of an exported partition*.

---

## Where the strongest novelty claims live

After closing all searches, **no single mechanism is unqualifiedly novel** — each has a
verified nearest precedent. The defense must therefore be made at the architecture level,
with each component claim *scoped* to its verified residual. Ranked by how defensible the
scoped claim is:

1. **Integrated architecture (the thesis-level claim)** — the closed-form firing-weighted
   ridge solve as a *single shared primitive* across flat FIS, an output-quantile-
   conditioned per-feature-GMM antecedent generator, a soft fuzzy tree with TSK leaves,
   an HME of TSK sub-FIS, and a Ruspini export. No prior work unifies these under one
   solver; this is the safest and strongest claim.
2. **Closed-form ridge TSK leaf in a soft tree** (§3) — *scoped:* not "linear leaves"
   (Medina-Chico 2001), but the *exact per-leaf firing-weighted ridge* leaf vs
   backprop-fit leaves.
3. **Quantile-conditioned, per-feature-factorized GMM antecedents** (§1b) — *scoped:* not
   output-first modeling (Sugeno–Yasukawa 1993) and not GaussianNB, but the *factorized
   quantile-conditioned generator wired to ridge-TSK with linear parameter growth*.
4. **Declarative `VariablePlan` precedence for fuzzy hierarchies** (§4) — *scoped:* not
   constrained induction in general (Nanfack 2022), but the *serializable precedence
   artifact*.
5. **Sequential single-cell fuzzy cascade specialists with abstention** (§8) — *scoped:*
   not confusion-matrix hierarchies (Cavalin–Oliveira 2019), but the *one-pair-at-a-time
   fuzzy specialist that can abstain*.
6. **Hierarchical nesting of multi-rule TSK sub-FIS as HME experts** (§5) — *weakest:*
   TSK≡MoE (Wu 2020) concedes the single-layer case; only the nesting + soft-inclusion
   survive.
7. **Ruspini export + knot-only partition-preserving refinement** (§7) — the *pipeline*
   is the contribution; partitions (Ruspini 1969) and constrained MF tuning
   (de Oliveira 1999) individually are not.

**Bottom line for the defense:** lead with the *integrated architecture + shared ridge
primitive*; present every component claim in its scoped "not X, but Y" form above, each
with its nearest precedent cited. That is a novelty story that survives an adversarial
reviewer, because the sharpest possible precedents are already conceded.

Full bibliography with keys: [`hfis_review.bib`](hfis_review.bib).
