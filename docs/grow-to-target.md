# Growing the model to a target score (`fit_growing`)

`TribbleClassifier.fit_growing(X, y, target=0.9)` finds the number of features instead of making you
guess it. It ranks the features **once**, starts with the best one, and adds the next-ranked feature to
the existing model until the score on held-out rows reaches `target`. The size at which it does is the
"right" `top_n` (tribble-fis #242).

```python
clf = TribbleClassifier(scoring_method="wasserstein")
clf.fit_growing(X, y, target=0.90, metric="f1_macro")      # held-out 20 %, stratified, seeded
clf.n_features_selected_     # smallest size whose held-out macro F1 >= 0.90 (or the best size seen)
clf.target_reached_          # False => the target was never met; a warning says so
clf.growth_history_          # per size: feature added, held-out score, best so far, step/cumulative seconds
clf.predict(X_new)           # an ordinary fitted classifier
```

## Why it is cheaper than a sweep
Refitting at `top_n = 1, 2, 3, ...` re-scores every feature and refits every membership at every size.
Two properties of the model make that unnecessary, both verified:

* a feature's memberships are fitted per `(feature, label)` and **do not depend on which other features
  are present**: a feature fitted alone is bit-identical to the same feature in a joint fit;
* the firing strength folds over features, per label `t_norm(running, conorm-fold of this feature's
  memberships)` starting from 1.0, so adding a feature is **one more fold step** (`growth.RunningFiringStrengths`
  is `tsk_firing_strengths` unrolled over features and is tested against it).

So the ranking happens once, each feature's memberships are fitted once, and one running firing-strength
matrix on the held-out rows is extended per step. The held-out score after adding the k-th feature is
**exactly** the score of a model refit from scratch with `top_n=k` on the same rows (tested for every k).

Measured with `python -m benchmarks.grow_to_target` (synthetic 3-class, 30 features, held-out macro F1;
the script asserts both arms agree on every score and on the stopping size before it reports a time):

| comparison | size | growing | refit sweep | ratio |
|---|---|---|---|---|
| to target F1 >= 0.80 (sweep stops at the same size) | 6 | 0.38 s | 1.42 s | 3.7x |
| walking all 30 sizes | 30 | 1.50 s | 23.71 s | 15.9x |

The sweep is quadratic in the number of features tried and growing is linear, so the gap widens with
size. Seconds are host-specific; the ratios transfer. An earlier ad-hoc run of the second row read 11x.

## Decisions (the open questions in #242)
| Question | Decision |
|---|---|
| What is the target evaluated on? | **Held-out rows only.** Everything that decides the selection -- the ranking, the crossover-pair search, the memberships -- uses the training part, so the held-out labels never influence it (tested by flipping every held-out label: the order does not move, the scores do). `validation` is a fraction (stratified when possible, seeded by `random_state`) or an explicit `(X_val, y_val)`. |
| Stop rule | The **smallest** size whose held-out score meets the target. `patience` (optional) ends growth early, short of the target, after that many sizes with no new best. |
| Never reached | The best size seen is returned (earliest on ties), `target_reached_ = False`, and a `UserWarning`. Never a silent best effort. |
| Non-monotone curves | Held-out F1 can fall when a feature is added (it does on the benchmark data). Growth stops at the first size that meets the target even if a later size would score lower; with an unreachable target the best size is returned, not the last. |
| F1 definition | `metric="f1_macro"` by default; also `f1_weighted`, `f1_micro`, `accuracy`, `balanced_accuracy`, `f1` (binary; `pos_label` required unless the classes are {0, 1}) or any `metric(y_true, y_pred)`. |
| Redundancy filter | Same as `top_n`: a candidate correlated with an already-kept feature is skipped and growth continues down the ranking. The skipped features are in `growth_skipped_`. |
| User ranking (#239) | Pareto subsets of different sizes are not nested, so they have no growth order. For a **fixed weight** the weighted-sum ranking is a single order whose top-k is a prefix: `user_weight` w gives `(1 - w) * data score + w * user importance` (0 ignores the user, 1 follows them). |
| Crossover columns | `detect_interactions=True` works here: the pair search runs on the training part, the product columns join the ranking, and a pair that only helps jointly can be the *first* feature grown (XOR reaches F1 0.95 at size 1). |
| Refinement | Not applied per step (expensive). With `refit=True` (default) the final model is refit on all rows with the chosen features, and `refine` then applies as in `fit` (Gaussian only). |

## Limits
* **All membership builders work** (`gaussian`, `trap` with `trapz_method="fast"` or `"em"`,
  `triangular`; #247). A feature fitted alone is identical to the same feature in a joint call for every
  one (checked on a grid of evaluations), and the growth oracle and the unrolled-fold equivalence are
  tested for each. Growth is only as cheap as the builder it calls: the EM trapezoid and triangular
  builders cost seconds per fit, so fitting each feature once saves the most there (27 s and 15 s over a
  four-size sweep; see "Basic measurements"). `refine` is Gaussian-only in any case.
* The growth loop folds with NumPy while `predict` may use the compiled kernel, so in principle a held-out
  row could flip on a 1-ULP `exp` difference. Measured: no held-out prediction differed (70 comparisons,
  compiled kernel verified active). The tests keep a few-row tolerance on the compiled leg as a defensive
  measure only; the NumPy path is exact.
* The held-out score is one split: an estimate with its own noise (about 3 F1 points per run in the
  measurement below). Selecting "the smallest size meeting the target" on it showed no detectable
  optimism (+0.006 to +0.010 held-out minus untouched test, standard error 0.007, 20 seeds), but one run can
  still be off by several points. For a tighter claim pass a larger `validation` set, and report the final
  model on data that took no part in the choice.
* `top_n` and `top_p` are ignored here; `correlation_threshold` still drives the redundancy skip. A
  result of one feature means the target was easy on this split, not that one feature is enough in general.

## Regression: `TribbleRegressor.fit_growing` (#248)
```python
reg = TribbleRegressor(tsk_order="1st")          # a concrete order; "auto" is refused (see below)
reg.fit_growing(X, y, target=0.90, metric="r2")  # or metric="rmse"/"mae" (a CEILING) or a callable
```
Same shape as the classifier (ranked once, held-out target, smallest size wins, best size + warning if never
met, `growth_history_`, `user_weight`, `refit`), with these differences:

* **The prediction is not an argmax.** `y_hat = sum_r w_r (mean_r + basis(X) @ corr_r)`, with the
  coefficients solved by ridge normal equations. Adding a feature rescales every row's firing strength and widens
  every rule's design block, so the consequents are **re-solved at every step**, with the same `_solve` and
  `predict` a normal `fit` uses. Memberships (fitted once per feature, on the output buckets) and the
  ranking are what is reused. The held-out score after k features is exactly that of a model refit with
  `top_n=k` (tested for every k, with and without `pin_extremes`).
* **Metric direction.** `r2` is a floor (`score >= target`); `rmse` and `mae` are ceilings (`score <= target`)
  and "best" means lowest. A callable must say which with `higher_is_better`.
* **A concrete `tsk_order` is required.** `"auto"` picks the order by cross-validation inside every fit, far too
  costly at each step. `detect_interactions=True` is refused (the feature order changes every step, and
  `cross_pairs_` index into it).
* The output partition (`partition_output`), the ranking, the memberships and the consequents all use the
  training part only.

Measured with `python -m benchmarks.grow_regressor` (synthetic, 24 features, 1st-order consequents; the script
asserts both arms agree on every held-out score and on the stopping size first):

| comparison | size | growing | refit sweep | ratio |
|---|---|---|---|---|
| to target R2 >= 0.90 (sweep stops at the same size) | 4 | 0.21 s | 0.59 s | 2.8x |
| walking all 24 sizes | 24 | 1.07 s | 11.84 s | 11.1x |

The per-step cost is flat (0.040 s first, 0.044 s last), so growth is linear and the sweep is not. #248
guessed the solve would dominate; at 1st order it does not. At `full-2nd` the cross terms grow quadratically
and it does: per-step cost is flat to about 6 features, then climbs to 10-12x the 1st-order step at 24
features (0.47 s vs 0.047 s at 1,875 rows; 0.95 s vs 0.08 s at 7,500). `solve_tsk_consequents` delegates to
`solve_tsk_consequents_from_firing`, which takes precomputed firing strengths, so the running fold could in
principle feed the solve and skip the per-step fold; with a flat 0.04 s per step there is little to gain at 1st
order, and it is not done. At `full-2nd` the cost is the solve itself, which this would not remove.

## Basic measurements (not runs of record)
`python -m benchmarks.grow_basic_measurements` (about 3 minutes). One 4-core Xeon, synthetic data, a few seeds:
ratios transfer, seconds do not. Growing scores matched the refit sweep exactly in every case.

| item | result |
|---|---|
| Regressor, 10 sizes, 1,875 rows | 1st: 0.44 s vs 2.39 s sweep (5.4x); full-2nd: 0.53 s vs 2.37 s (4.5x) |
| Regressor growth to 24 features, 7,500 rows | 1st 1.7 s; full-2nd 7.0 s (last step 0.95 s) |
| Classifier, 14 sizes, 750 held-out rows, 5 seeds | growth vs sweep 7.4x (NumPy), 7.6x (compiled); max score diff 0 on both |
| Builders, 4 features, 300 rows (growth vs sweep) | gaussian 2.3-3.1x; trap fast 2.1x; triangular 3.7x (saves 15 s); trap EM 3.7-4.0x (saves 26-27 s) |
| Selection optimism, 20 seeds | reachable target: +0.006 +/- 0.007 (SE); unreachable: +0.010 +/- 0.007; per-run sd 0.031 |

