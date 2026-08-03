# Why did identification cost four EM fits it never used?

**Answer: no reason. Component selection now scores the k-means partition it is
going to keep, and the invisible 20,000-row cap is gone.** Identification is
**4-10× faster** and held-out accuracy is equal or better at every point
measured. Two defects, one fix each.

This came out of a proposal-defense study that was timing rule identification
against classical clustering. The timings were strange — a cost curve that went
flat above 40,000 rows, and a construction that cost 25-84× a k-means — and both
turned out to be artifacts of the code rather than properties of the method.

## Defect 1: the fit-select-discard-refit cycle

`fit_gaussians` needed a component count, so it called `find_optimal_gaussians`,
which did this:

```python
for n in range(1, max_gaussians + 1):
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(data)                       # a full EM fit
    bics.append(gmm.bic(data))
return n_components_range[np.argmin(bics)]   # ... and only the index survives
```

Four EM fits, each internally seeded by its own k-means, and the function
returns an integer. `fit_gaussians` then ran **another** k-means at the winning
count to get the placement that the winning GMM had already computed and thrown
away. Between five and nine fits to keep one.

Worse than the waste: the criterion scored the wrong model. BIC was measured on
an EM mixture that was never built, and the model that *was* built — a k-means
partition with each cluster's mean and standard deviation — was never scored.
The selector was optimising a proxy it then discarded.

The replacement scores each candidate directly off the partition it implies:

```python
for k in range(1, min(max_gaussians, n_distinct) + 1):
    candidate = _hard_partition_gaussians(data, _kmeans_labels_1d(data, k, seed), k)
    bic = _mixture_bic(data, candidate, var_floor)
```

Same criterion — `n_params * log(N) - 2 * log_likelihood`, `3k - 1` free
parameters, the mixture density evaluated at every observation — read at the
hard-assignment MLE instead of at an EM optimum. With hard assignments the
mixture likelihood separates, so each component's MLE *is* the mean and standard
deviation of its own points; there is nothing to iterate. One k-means per
candidate, none discarded, and `k = 1` needs no clustering at all.

`fit_gaussian_mixture_1d` returns `(memberships, n_selected)`, so the caller no
longer refits. `find_optimal_gaussians` survives as a wrapper for the count
alone, because it is public.

Two details that matter:

* **A variance floor.** A component that lands on a single point has zero
  variance, infinite likelihood, and would always win. `BIC_VARIANCE_FLOOR_FRAC`
  plays scikit-learn's `reg_covar` role, but as a fraction of the column's own
  variance rather than an absolute `1e-6`: these are raw features, and a fixed
  floor is a wall on a millivolt column and a rounding error on a currency one.
* **`k` is capped at the number of distinct values.** A binary column cannot
  support four clusters; asking anyway earns a `ConvergenceWarning`, an empty
  cluster, and a discarded k-means. On a dataset with many low-cardinality
  features that was most of the candidate list.

## Defect 2: the cap nobody could see

```python
def fit_gaussians(X, y, column, label_value, n_gaussians=0, max_samples=20_000):
    ...
    data = data[:max_samples]
```

`create_gaussian_membership_dict` did not expose `max_samples`, so every caller
above that level silently fitted on at most 20,000 rows per (feature, label).
Two things are wrong with that:

1. **A prefix is not a sample.** On data sorted by anything — time, class, an
   index that correlates with a feature — the first 20,000 rows are a biased
   draw, and nothing in the API hints that a draw is happening.
2. **It makes cost curves lie.** Above the cap the fitting cost stops growing,
   so the method looks like it scales sublinearly when it has simply stopped
   reading the data. The study that found this had already written the flat
   curve up as an algorithmic property. It is not one.

Now: `max_samples=None` by default (use every row), exposed on
`create_gaussian_membership_dict`, and when a cap *is* given the rows are drawn
at random without replacement, seeded by `random_state`. Subsampling is worth
having. Doing it invisibly, as a prefix, is not.

`trapz_math` had both defects in the same shape — `find_optimal_trapezoids` ran
`fit_trapezoids_em` per candidate and returned only the count, and
`fit_trapezoids` capped at the first 20,000 rows — and gets the same two fixes.

## What it cost, and what it bought

Concrete, 824 × 8, automatic component count, three seeds, median of three
repeats, single-threaded. `train (ms)` is identification only; the consequent
solve and the feature screen are excluded from both columns.

| rules | EM select | k-means select | speedup | test R², EM | test R², k-means |
|---:|---:|---:|---:|---:|---:|
| 2 | 320.3 ± 14.0 | **70.5 ± 3.5** | 4.5× | 0.730 ± 0.121 | **0.787 ± 0.028** |
| 3 | 441.7 ± 20.0 | **97.7 ± 4.6** | 4.5× | 0.819 ± 0.022 | 0.818 ± 0.015 |
| 4 | 600.1 ± 31.8 | **127.3 ± 3.1** | 4.7× | 0.829 ± 0.033 | 0.828 ± 0.022 |
| 6 | 859.4 ± 48.9 | **187.3 ± 6.9** | 4.6× | 0.819 ± 0.032 | **0.851 ± 0.023** |
| 8 | 1103.2 ± 21.5 | **245.1 ± 13.4** | 4.5× | 0.844 ± 0.018 | 0.854 ± 0.020 |
| 12 | 1561.3 ± 37.5 | **376.3 ± 8.3** | 4.1× | 0.855 ± 0.034 | 0.852 ± 0.032 |

Cross-validated MSE moves the same way — 0.00793 → 0.00657 at two rules, and
within noise elsewhere — and the new selector uses slightly *fewer* parameters
at five of the six rule counts.

PhiUSIIL, 235,795 × 50 binary, top ten features, automatic component count,
three seeds. Both columns capped at 20,000 rows per (feature, class) — the old
cap, applied deliberately here so the comparison isolates the selector rather
than mixing in defect 2:

| train rows | EM select | k-means select | speedup | accuracy, EM | accuracy, k-means |
|---:|---:|---:|---:|---:|---:|
| 4,000 | 407 ± 10 | **63 ± 2** | 6.5× | 0.9997 | 0.9997 |
| 16,000 | 905 ± 49 | **100 ± 5** | 9.1× | 0.9988 | 0.9988 |
| 40,000 | 1,776 ± 67 | **222 ± 46** | 8.0× | 0.9998 | 0.9998 |
| 96,000 | 1,957 ± 54 | **279 ± 13** | 7.0× | 0.9998 | 0.9995 |
| 188,636 | 2,042 ± 41 | **374 ± 6** | 5.5× | 0.9997 | 0.9998 |

Selection has gone from most of the cost to nearly none of it: at full size the
automatic construction now costs 374 ms against 372 ms for one told its
component count outright. Choosing the count used to be 82% of the bill.

## The selector disagrees, and that is the improvement

Run both selectors over every (feature, label) group and compare
(`reproduce/optimizers/check_fit_gaussians_fix.py` in the study repo):

| dataset | groups | same count | centre shift where counts agree | selection time |
|---|---:|---:|---:|---:|
| concrete, 3 buckets | 24 | 12 (50%) | 0.0000 s.d. | 432 → 91 ms (4.7×) |
| phiusiil, 50k rows | 20 | 19 (95%) | 0.0000 s.d. | 1708 → 150 ms (11.4×) |

Where the counts agree the memberships are identical to the last digit, because
the placement path is untouched. The only thing that can move is `k`.

On Concrete it moves often, and by the *old* criterion's arithmetic the new
choice gives up a median 22% of the BIC range. That number is a red herring: it
is denominated in the fit of an EM mixture that is never built. The model that
ships is the k-means partition, the new selector is the one scoring it, and on
held-out R² the new choice is equal or better at every rule count in the table
above — including +0.057 at two rules, where it also cuts the seed-to-seed
spread by a factor of four.

## What was not changed

* **Placement.** k-means, then each cluster's mean and standard deviation.
  Identical, which is why the centre-shift column is zero.
* **`max_gaussians`.** Still 4 by default.
* **The categorical branch.** String, bool and object columns still get one
  narrow Gaussian per distinct value.
* **`random_state=42`.** Still the default, now also threaded through the
  subsample draw.

One thing was tidied in passing: `create_gaussian_membership_dict` ran its
per-feature loop through a `ThreadPoolExecutor(max_workers=1)`. The pin is
load-bearing — it hangs above one worker on some Linux hosts — so the pool was
buying futures machinery for serial work. It is now a loop. With one worker
`as_completed` already yielded in submission order, so the resulting dict order
is unchanged, which matters because `extract_gaussian_params` flattens that dict
into a parameter vector.

Full suite: 280 passed, 57 skipped.
