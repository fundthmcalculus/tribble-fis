# fuzzytree — a hierarchical fuzzy tree for TRIBBLE

`fuzzytree` turns TRIBBLE's TSK fuzzy inference into a **hierarchical, human-readable
tree**. Where the flat `MixtureOfGaussiansFuzzyRegressor` produces a few *long* rules
(each an AND over every selected variable, keyed to output quantile buckets), a fuzzy
tree produces *short* rules — one per root→leaf path — that mention only the variables
on that path, in order of importance:

```
IF Cement is Low (≤ 275):
  IF Age is Low (≤ 28):
    IF Cement is Low (≤ 212): => y ≈ 44.2 − 0.069·Cement + 0.58·Age + ...
```

Each internal node splits **one** input variable into fuzzy linguistic terms
(Low/Med/High); a sample flows down **all** branches with a partial membership degree
(a *soft* tree), so predictions stay smooth. Leaves hold local TSK consequents (constant
for readability, linear/polynomial for accuracy). Variable-to-node assignment is chosen
**automatically** by a split criterion but is fully **user-overridable**.

## Design at a glance

- **Isolated**: everything lives here; nothing under `src/tribblefis/` is modified. We
  import TRIBBLE primitives (`GaussianMembership`/`TrapezoidMembership`, `t_norm`,
  `build_consequent_features`, `_normalize_firing_strengths`,
  `calculate_gaussian_correlation`, `partition_output`) and replicate only the ~20-line
  ridge consequent solve locally (the upstream solver derives firing internally and
  can't take an external leaf matrix).
- **Leaves are rules**: the leaf firing matrix `(n_samples, n_leaves)` is the exact
  analogue of the flat model's rule-firing matrix, so consequents are solved by the same
  firing-weighted ridge least squares, and prediction is the same weighted defuzzification.
- **Open-shouldered trapezoids**: split terms extend to ±∞ at the extremes, so
  out-of-range points always fire a leaf (no silent predict-0 collapse).
- **Honest scope**: vs. TRIBBLE's already-compact flat base the tree does *not* reduce
  rule count (it can increase it, up to `n_terms^max_depth` leaves). The payoff is
  **readability and an explicit variable hierarchy**, and it holds only at shallow depth
  / few terms — hence the hard `max_depth` / `max_leaves` / `n_terms` caps.

## Quick start

```python
import pandas as pd
from fuzzytree import FuzzyRegressionTree, render_tree_text, plot_fuzzy_tree

# Automatic structure (variance-reduction splits), constant leaves:
model = FuzzyRegressionTree(tsk_order="0th", max_depth=3, n_terms=2).fit(X, y)
print(render_tree_text(model))          # indented IF-THEN rules
plot_fuzzy_tree(model).savefig("tree.png")

# More accuracy: linear leaves.
model = FuzzyRegressionTree(tsk_order="1st", max_depth=3).fit(X, y)
```

### Controlling structure — `VariablePlan`

```python
from fuzzytree import VariablePlan, NodePin

plan = VariablePlan(
    level_order=("Cement", "Age"),           # depth 0 splits Cement, depth 1 Age
    pins=(NodePin(path=("Low",), variable="Water"),),   # override one node by path
    criterion="variance",                    # auto criterion for unspecified nodes
    exclude=frozenset({"CoarseAgg"}),        # never split on these
    max_depth=3, default_n_terms=2, max_terms_per_var=2,
)
model = FuzzyRegressionTree(variable_plan=plan, tsk_order="1st").fit(X, y)
```

Precedence at each node: **path pin > level order > auto-by-criterion**. Plans are
immutable and JSON-serialisable (`plan.to_dict()` / `VariablePlan.from_dict(...)`).

### Classification

```python
from fuzzytree import FuzzyClassificationTree
clf = FuzzyClassificationTree(criterion="ambiguity", max_depth=3, n_terms=2).fit(X, y)
clf.predict(X); clf.predict_proba(X)
```

## Hierarchical mixture of fuzzy experts (composing sub-FIS)

The models above are a single fuzzy tree. To instead **compose multiple sub-FIS into
one tree**, use the hierarchical mixture of experts (HME): internal nodes are fuzzy
**gates** that softly route a sample, and each leaf is a full TSK **expert sub-FIS**.
The output is the gated blend over leaves — `ŷ(x) = Σ_leaf g_leaf(x)·expert_leaf(x)`,
where `g_leaf` is the product of the partition-of-unity gate weights on the root→leaf
path (leaf gates sum to 1, so the blend is convex). This is the classic Jordan-Jacobs
HME with fuzzy gates and fuzzy experts.

```python
from fuzzytree import HierarchicalFuzzyExpertsRegressor, render_hme_text, plot_hme

hme = HierarchicalFuzzyExpertsRegressor(
    criterion="variance", max_depth=2, n_gate_terms=2,
    expert_kwargs={"n_output_buckets": 3, "tsk_order": "1st"},  # per-leaf sub-FIS
).fit(X, y)
print(render_hme_text(hme))          # ROUTE ... / expert sub-FIS ...
plot_hme(hme).savefig("hme.png")

from fuzzytree import HierarchicalFuzzyExpertsClassifier   # classification variant
```

**How the structure is inferred and built** (`hme.py`):

1. Reuse the criterion-driven recursive partitioning (`build_tree`) to infer the gate
   topology — *which* variable routes at each node and into how many fuzzy regimes —
   fully overridable via `VariablePlan` (pin a routing variable, fix a per-level order).
2. Turn each node's linguistic terms into a partition-of-unity gate.
3. Compute each training sample's leaf **responsibilities** (path gate product).
4. Fit each leaf's expert sub-FIS on the samples whose responsibility for that leaf
   exceeds `responsibility_threshold` (**soft-inclusion**: overlapping training sets
   cover the gate boundaries — a hard argmax partition starves them and hurts the blend).
5. Blend experts by the soft gates at predict time.

The plain `FuzzyRegressionTree` is exactly the special case where each expert is a single
TSK consequent instead of a full sub-FIS.

The current build is a one-shot greedy fit. A design note for refining it with a true
EM loop (posterior responsibilities that fold in expert fit quality, weighted expert +
gate updates) is in [`EM_REFINEMENT.md`](EM_REFINEMENT.md).

### MIMO regression

```python
from fuzzytree import MimoFuzzyTreeRegressor    # one tree per output column
MimoFuzzyTreeRegressor(tsk_order="1st").fit(X, Y_df).predict(X)   # -> DataFrame
```

## Split criteria

| criterion | use | basis |
|---|---|---|
| `variance` (default, regression) | firing-weighted variance reduction; aligned with the leaf MSE objective | CART-style |
| `ambiguity` (default, classifier) | Yuan-Shaw classification-ambiguity reduction | possibility/nonspecificity |
| `info_gain` | Janikow fuzzy ID3 fuzzy information gain | fuzzy entropy |
| `differentiation` | cheap relevance prefilter for wide inputs | reuses TRIBBLE's differentiation score, weight-aware |

## Files

`plan.py` (variable plan + precedence) · `terms.py` (linguistic terms) · `splitter.py`
(criteria) · `node.py` (tree node) · `firing.py` (leaf firing) · `solve.py` (consequent
solve/predict) · `builder.py` (recursive build) · `regressor.py` / `classifier.py`
(single-tree estimators) · `hme.py` (hierarchical mixture of fuzzy experts) ·
`render.py` (text + matplotlib for both) · `tests/` · `demo_concrete.py` (regression) ·
`demo_phishing.py` (classification).

## Running

```bash
uv run --extra dev python -m pytest tribble-tree/tests -v
uv run python tribble-tree/demo_concrete.py     # regression
uv run python tribble-tree/demo_phishing.py     # classification
```

**Regression — UCI Concrete Compressive Strength:**

```
Flat TRIBBLE (MixtureOfGaussiansFuzzyRegressor)   R2=0.658   RMSE=9.381 MPa
Fuzzy tree (0th-order, constant leaves)           R2=0.460   RMSE=11.799 MPa
Fuzzy tree (1st-order, linear leaves)             R2=0.746   RMSE=8.091 MPa
Hierarchical fuzzy experts (gated sub-FIS)        R2=0.791   RMSE=7.342 MPa
```

The auto-built tree splits first on **Cement**, then **Age** — with a threshold at the
standard 28-day curing mark — recovering domain knowledge directly from the data. The
HME goes further, routing to a full sub-FIS per region, and here is the most accurate.

**Classification — PhiUSIIL Phishing URL dataset (subsampled):**

```
Flat TRIBBLE (MixtureOfGaussiansFuzzyClassifier)  acc=0.998   F1(phish)=0.998
Fuzzy tree (ambiguity splits)                     acc=0.968   F1(phish)=0.972
Fuzzy tree (fuzzy info-gain splits)               acc=0.969   F1(phish)=0.972
Hierarchical fuzzy experts (gated sub-FIS)        acc=0.996   F1(phish)=0.997
```

The tree splits on interpretable signals — **HasSocialNet**, **HasCopyrightInfo**,
**URLSimilarityIndex** — trading a little accuracy for a rule set a human can read; the
HME recovers nearly all of the flat model's accuracy while keeping that gated structure.

## References

- C. Z. Janikow. *Fuzzy decision trees: issues and methods.* IEEE Trans. Systems, Man,
  and Cybernetics — Part B, 28(1):1–14, 1998. (Fuzzy ID3 / fuzzy information gain.)
- Y. Yuan, M. J. Shaw. *Induction of fuzzy decision trees.* Fuzzy Sets and Systems,
  69(2):125–139, 1995. (Classification-ambiguity split criterion.)
- M. Higashi, G. J. Klir. *Measures of uncertainty and information based on possibility
  distributions.* Int. J. General Systems, 9:43–58, 1983. (Nonspecificity / U-uncertainty.)
- J. Fumanal-Idocin et al. *A Fast Interpretable Fuzzy Tree Learner.* arXiv:2512.11616,
  2025. (Greedy fuzzy-term splits, fuzzy impurity, membership propagation.)
- J. J. Suárez, J. F. Lutsko. *Globally Optimal Fuzzy Decision Trees for Classification
  and Regression.* IEEE TPAMI, 21(12):1297–1311, 1999. (Soft trees with linear leaves.)
- H. Wang et al. *Takagi–Sugeno–Kang fuzzy system fusion: a survey at hierarchical, wide
  and stacked levels.* Information Fusion, 101, 2023. (Hierarchical TSK for dimensionality.)
- M. I. Jordan, R. A. Jacobs. *Hierarchical mixtures of experts and the EM algorithm.*
  Neural Computation, 6(2):181–214, 1994. (Gated tree of experts — the HME architecture.)
- G. V. S. Raju, J. Zhou, R. A. Kisner. *Hierarchical fuzzy control.* Int. J. Control,
  54(5):1201–1216, 1991. (Routing a few variables per sub-FIS to tame dimensionality.)
```
