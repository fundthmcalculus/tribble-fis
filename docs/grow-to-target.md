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
| Refinement | Not applied per step (expensive). With `refit=True` (default) the final model is refit on all rows with the chosen features, and `refine` then applies as in `fit`. |

## Limits
* **`member_function="gaussian"` only.** The trapezoid and triangular builders are not wired in
  (`NotImplementedError`). Classification only; the regression analogue (a target R2/RMSE) is not done.
* The growth loop folds with NumPy. On the compiled-kernel path a held-out row can flip on a 1-ULP `exp`
  difference, so a score can differ from `predict` by a few rows; the NumPy path is exact.
* The held-out score is one split. It is an estimate with its own noise, and "the smallest size
  meeting the target" on it is optimistic by selection. For a tighter claim pass a larger `validation`
  set, and report the final model on data that took no part in the choice.
* `top_n` and `top_p` are ignored here; `correlation_threshold` still drives the redundancy skip. A
  result of one feature means the target was easy on this split, not that one feature is enough in general.
