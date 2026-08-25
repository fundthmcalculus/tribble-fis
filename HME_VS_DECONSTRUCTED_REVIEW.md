# Review: should HME be dropped in favor of the deconstructed tree?

> Review note, not a findings note for a new mechanism — no code changed here.
> Branch: `review-hme-vs-deconstructed`, off `deconstruct-hierarchical-fis`.

## Short answer: no, don't drop it — they solve different problems, and the
## evidence for "clearly better" doesn't hold up once HME gets a fair shot.

## Why this looked plausible

On the one real dataset both were run against (N-CMAPSS turbofan RUL, 18 real
channels), the numbers in `DECONSTRUCTED_TREE_FINDINGS.md` do look lopsided:

| Model | Test R² |
|---|---|
| Flat `TribbleRegressor` | 0.370 |
| HME (`max_depth=2`, under its own default of 3) | 0.405 |
| Deconstructed tree | **0.593** |

A ~0.19 R² gap looks like a clear win. But that HME run used a shallower
tree than its own default and no EM refinement — not the tool used as
intended.

## What a fairer shot at HME actually does

Two follow-ups, same data/split:

| HME configuration | Test R² |
|---|---|
| `max_depth=2` (the original eval) | 0.405 |
| `max_depth=3` (HME's own default), `n_gate_terms=3`, `top_n=8` (27 leaves) | **-0.109** |
| same, + `refine_em(max_iter=10)` | **-0.549** |

Giving HME *more* capacity on this dataset makes it worse, not better, and
EM refinement makes it worse again — despite the EM step's own
monotonicity guarantee holding exactly as designed (training log-likelihood
rose every iteration, -185950 → -174168). That's the textbook signature of
overfitting: EM is maximizing training likelihood, and with 27 leaves each
fitting their own multi-parameter sub-FIS on this dataset's collinear real
sensors, it fits the training distribution better while generalizing worse.
This is a genuinely useful finding in its own right — **HME's capacity
knobs are actively dangerous on this kind of high-dimensional, correlated
real-sensor problem** — but it's evidence HME struggles on *this specific
dataset shape*, not evidence the deconstructed tree is a strict upgrade.

## The deconstructed tree can't actually replace HME's job

The comparison was never quite apples-to-apples: `DeconstructedHierarchicalRegressor`
*requires* the caller to already know the correct topology (it deconstructs
a flat fit into a tree the user supplies). `HierarchicalFuzzyExpertsRegressor`
solves the opposite problem — *discovering* structure when nobody knows it
up front. On N-CMAPSS we had the topology from turbofan station numbers, so
the comparison was possible; on most datasets nobody has that.

And on the two datasets where structure genuinely isn't known in advance —
the repo's own README benchmarks — HME is not just competitive, it's the
best result on record:

| Dataset | Flat | Tree (1st-order) | **HME** |
|---|---|---|---|
| Concrete compressive strength (R²) | 0.658 | 0.746 | **0.791** |
| PhiUSIIL phishing (accuracy) | 0.998 | 0.968–0.969 | 0.996 |

`README.md`'s own framing: *"HME goes further, routing to a full sub-FIS
per region, and here is the most accurate."* Dropping HME would drop the
package's current best result on its flagship regression benchmark.

## It's also load-bearing for the PhD novelty claims

`HFIS_NOVELTY_REVIEW.md` frames HME as **Mechanism 5** (hierarchical
composition of TSK sub-FIS as HME experts, with soft-inclusion training and
the shared ridge solve as what survives the TSK≡MoE equivalence), and
**Mechanism 6**'s dimensionality argument explicitly rests on
"the HME/sub-FIS routing, not the single tree." Removing HME doesn't just
delete an estimator, it removes two of the document's claimed contributions.
[[hfis-novelty-review]]

## What removal would actually cost

- `fuzzytree/hme.py` (402 lines) + `fuzzytree/em.py` (371 lines) +
  `EM_REFINEMENT.md` (350 lines of design/validation writeup).
- `tests/test_fuzzy_tree.py::TestHME` and all of `tests/test_em_refinement.py`
  (log-likelihood monotonicity, sharp-boundary synthetic headline case,
  starved-leaf safeguard, classification smoke test) — real, passing
  guarantees, not placeholder tests.
- `demo_concrete.py`, `demo_phishing.py` both use it as the headline result.
- `README.md`'s "Hierarchical mixture of fuzzy experts" section and its
  worked examples.
- Two mechanisms in the novelty review would need to be struck or
  substantially rewritten.

## Recommendation

Keep both; they're complementary, not competing:

- **Known topology** (you can name the right feature groups, like turbofan
  components) → `DeconstructedHierarchicalRegressor`.
- **Unknown topology** (let the data suggest structure) → HME, still the
  best result on this repo's own regression benchmark.

Concretely:
1. Don't remove `hme.py`/`em.py`/their tests/docs.
2. Do write up the capacity/EM-overfitting result on N-CMAPSS somewhere
   (a short addendum to `DECONSTRUCTED_TREE_FINDINGS.md`, or its own note)
   — it's a real, non-obvious finding about when HME's own knobs hurt it,
   independent of the "should we drop it" question.
3. If anything, add a short "which one do I use" pointer between the two
   READMEs/docstrings so the next reader doesn't have to rediscover this.
