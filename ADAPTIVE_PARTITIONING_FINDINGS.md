# Findings: Error-Driven Adaptive Rule Partitioning for TSK Regression

> Evaluation note. Status: **not merged into the fit path** — see "Disposition" at
> the end. Implementation lives in this working tree (uncommitted) for reference:
> `src/tribblefis/adaptive_partition.py`, `partition_output_by_edges`/`_bucket_r2`
> in `src/tribblefis/regression.py`, `bucket_strategy="adaptive"` on
> `MixtureOfGaussiansFuzzyRegressor`, `gaussian_mixture/concrete_adaptive.py`,
> `tests/test_regression.py::TestAdaptivePartition`.

## Motivation

`MixtureOfGaussiansFuzzyRegressor` partitions the target `y` into a fixed
`n_output_buckets` via equal-frequency `qcut` (`partition_output`,
`regression.py`). Each bucket is one TSK rule — every feature gets a Gaussian
membership function per bucket, and consequents are solved jointly in closed
form. Because the partition is equal-frequency in *y*, it adds the same
resolution everywhere regardless of where the model is actually wrong.

The question evaluated here: can a partition that starts from a single rule
and adaptively splits the worst-fitting region beat the fixed uniform
partition at the same rule budget?

## What was built

`grow_adaptive_partition` (new module `adaptive_partition.py`): starts from
one rule covering all of `y`, and each iteration:
1. Fits memberships + closed-form consequents for the current partition.
2. Scores every rule's local R² (against its own y-mean).
3. Splits the lowest-scoring eligible rule into two, by `split_method`:
   - `"median"` — bisect at the bucket's median y (equal-frequency children).
   - `"sse"` — CART-style scan for the y-threshold minimizing the two
     children's combined SSE against their own means.
4. Stops when every rule clears `r2_threshold`, or `max_rules` is hit.

Exposed on the estimator as `bucket_strategy="adaptive"`, with
`max_rules`, `bucket_r2_threshold`, `min_bucket_samples`,
`adaptive_split_method`, and `guard_stalled_splits`.

## Findings, in the order they surfaced

**1. `pin_extremes=True` corrupts growth-time evaluation.** Pinning forces the
first/last bucket's constant to the *global* min/max of `y` — a sensible
constraint on a finished model's wide outer buckets, but as soon as growth
creates 2 buckets, it forces each half's prediction to the single most
extreme value in the whole dataset, which most of that half's rows aren't
near. This alone produced catastrophically negative results at the first
split. **Fix:** growth always fits unpinned; if the caller wants pinning, one
final pinned solve runs after the edge set is fixed.

**2. Naive "always split the globally worst bucket" fixates on one region.**
On the Concrete dataset, the loop kept re-bisecting the same narrow y-slice
(`edges` converging to `[0.266, 0.333, 0.371, 0.386, 0.391, 0.396]`) instead
of spreading rules across the error surface. Narrower buckets give the
per-feature Gaussian antecedents fewer points to fit, so firing-strength
routing degrades — the split doesn't help, but nothing told the loop that.

**3. Two guard designs were tried and rejected before finding one that works:**
   - *Compare a child's local R² to its parent's local R²*: rejected — a
     narrower sub-population has less of its own variance to explain, so its
     local R² is a strictly harder bar even for an equally good fit. This
     blocked a split that had genuinely improved the global fit.
   - *Compare global train R² before/after the split*: rejected — the joint
     closed-form solve is weakly monotonic in free parameters, so global
     train R² is nearly always non-decreasing when a rule is added,
     regardless of whether *that particular* split helped. Never triggered.
   - **Working version:** compare total squared error over exactly the
     parent bucket's rows, before vs. after the split (same fixed population,
     so it's apples-to-apples). A split that doesn't reduce it excludes its
     children from further splitting. This fixed the fixation pattern.

**4. SSE-minimizing splits: a clean win on clean data, a loss on messy data.**
On a synthetic case built to isolate the question (one feature, a clean
near-noiseless population plus a separate noisy cluster with a non-overlapping
y-range), `"sse"` correctly bracketed the noisy cluster and beat `"median"`
(train R² 0.818 vs 0.778). On the real 8-feature Concrete dataset, it did
**worse** than `"median"` at test time (0.763 vs 0.780 R²) despite fitting
training data *better* (0.798 vs 0.778 train R²) — a classic overfitting
signature: SSE's greedy training-optimal thresholds carve narrower, more
fragile buckets on messy multi-feature data, and this architecture's
antecedent fit degrades sharply on narrow buckets.

**5. Bottom line, matched 8-rule budget, Concrete dataset (test R²):**

| Method | Test R² |
|---|---|
| uniform (qcut) | **0.811** |
| adaptive, median split, guarded | 0.780 |
| adaptive, SSE split, guarded | 0.763 |

Both guarded adaptive variants beat their own unguarded/naive predecessor
(which was worse still, ~0.75), but neither closes the gap to plain uniform
partitioning.

## Root cause

This architecture ties rule labels to *two* things at once: the output
partition **and** the antecedent-fitting groups (each feature's Gaussian
membership is fit per label). Concentrating resolution by narrowing a
bucket simultaneously starves that label's antecedent fit of data, which
degrades firing-strength routing for that rule. On real, noisy,
multi-feature data this cost consistently outweighs the benefit of added
local resolution — so "add resolution where the error is" backfires unless
the antecedent-fitting cost of narrower buckets is also addressed, which
none of the variants tried here do.

## Disposition

Per direction: **not merged into the fit path.** The code stays in the
working tree, uncommitted, as a reference implementation and a record of what
was tried — only this findings note is committed. If this is picked up again,
the highest-leverage next step is a validation-based (not training-R²)
stopping/eligibility criterion, since training R² is exactly what let the SSE
split overfit here; a bigger lever would be decoupling the antecedent-fitting
groups from the output partition so narrowing a rule's y-range no longer
starves its own membership fit.
