# Finding candidate cross terms before feature selection drops them

**Answer, up front:** `calculate_gaussian_correlation` (`gauss_math.py`) scores
every feature independently, and `take_top_features` filters on that score
alone. A feature that is only informative *jointly* with another can score
near the bottom individually and be dropped before anything downstream --
including `full-2nd`'s cross terms -- ever gets a chance to use it.
`gauss_math.calculate_interaction_scores` scores candidate *pairs* for
interaction "lift" during the same feature-selection pass, so a jointly
useful feature can be rescued into the model instead of silently discarded.

## The gap, precisely

`TribbleRegressor.fit` ran these steps in this order, before
this change:

1. `calculate_gaussian_correlation` -- one score per feature, computed from
   that feature's own distribution alone.
2. `take_top_features` -- keeps features above a score threshold.
3. `create_gaussian_membership_dict` -- builds antecedents *only for the
   survivors*.
4. `solve_tsk_consequents` -- if `tsk_order='full-2nd'`, cross terms are
   built from every pair *among the survivors of step 2*.

A feature dropped at step 2 is invisible to every step after it. Since step
2 never looks at pairs, a feature whose only value is in combination with
another can be dropped even though the pair together would be highly
informative. Separately: `regression.select_interaction_terms` already
implements a LassoCV screen to sparsify `full-2nd`'s dense all-pairs cross
terms -- but grep across the whole repo turns up zero call sites for it. It
was never wired into the estimator; `consequent-plan.md`'s own "Phase 1
integration" section planned to and didn't. Both gaps are closed together
here, since the new early-detection method is designed to feed the existing
late-stage screen a shortlist instead of nothing.

## The method

`calculate_interaction_scores(X, y, feature_differentiators)` scores every
candidate pair `(i, j)` as:

```
lift(i, j) = score(z_i * z_j) - max(score(i), score(j))
```

where `score(·)` is `calculate_gaussian_correlation`'s own per-label-pair
distance metric (`wasserstein`/`bhattacharyya`/`composite`), `z_i`/`z_j` are
z-scored columns, and `score(i)`/`score(j)` are the same features' own
individual scores from the caller's existing ranking. A positive lift means
the *joint* value separates labels/output-buckets better than either input
alone -- the same idea as Friedman's H-statistic or interaction information,
phrased in the metric this module already speaks rather than a new one.

**Why this metric and not mutual information or an H-statistic.** Both of
those are more standard tools for exactly this question, and were
considered. This module already has a validated, in-house distributional
distance with three interchangeable estimators
(`docs/norm-family-evaluation.md`'s sibling analyses did the same kind of
work for the t-norm choice); reusing it costs zero new dependencies, stays
on the same scale as the univariate scores the lift is compared against by
construction, and avoids a second, differently-calibrated notion of
"informative" living next to the first. The trade is real: it is not a
peer-reviewed interaction statistic, and `_differentiation_score` on a
product column is a specific, deliberate choice rather than the only
reasonable one. Treat it as "this project's own metric, extended
pairwise," not a claim that it dominates the standard alternatives.

`take_top_interactions` applies `take_top_features`'s exact threshold
semantics to lift instead of score (and additionally requires positive
lift -- a pair that doesn't beat its better half is not a candidate at any
threshold). `rescue_interacting_features` unions any feature in a kept pair
into the selected set. Both are pure post-processing, same as
`take_top_features` itself.

## Wiring, and the combinatorial guard

`TribbleRegressor` gets three opt-in constructor parameters
(default `False`/`0.95`, reproducing prior behavior exactly when unset):
`detect_interactions`, `interaction_top_p`, `select_interactions`. When
`detect_interactions=True`, `fit` runs the detection pass between
`take_top_features` and model construction, rescues qualifying features, and
-- if `tsk_order='full-2nd'` -- builds `cross_pairs_` from the kept pairs.
`select_interactions=True` additionally routes that shortlist through the
existing `select_interaction_terms` LassoCV screen (now accepting an
optional `candidate_pairs` argument, so it screens the shortlist instead of
every pair among the final features) for a final sparsity pass.
`select_interactions=True` with any other `tsk_order` raises a `RuntimeWarning`
-- there is nowhere for `cross_pairs_` to go.

Pairs grow as `n_choose_2`; `calculate_interaction_scores` raises a
`ValueError` naming the fix (narrow `candidate_pool`, or pass `max_pairs`
explicitly) rather than silently grinding through an accidentally huge
candidate pool, the same "explicit failure over silent slow behavior"
convention `anfis.RuleExplosionError` uses on the ANFIS branch for the same
kind of combinatorial growth.

## Measured comparison

Two synthetic problems, three seeds each, `TribbleRegressor(tsk_order="full-2nd")` at a deliberately strict `top_n` -- the setting where the
gap actually bites -- with vs. without `detect_interactions=True,
select_interactions=True`. Reproduce with
`python -m benchmarks.interaction_detection_comparison`.

| problem | top_n | R² without detection | R² with detection |
|---|---|---|---|
| `pure_interaction` (`y = x0·x1`, 3 noise cols) | 1 | 0.012 | **0.996** |
| `interaction_plus_dominant` (`y = 2·x2 + 0.8·x0·x1`, 3 noise cols) | 2 | 0.951 | **1.000** |

`pure_interaction` is the sharpest illustration: at `top_n=1`, univariate
ranking keeps exactly one of `x0`/`x1` (whichever scores fractionally
higher) and the model has no way to reconstruct a product of two things when
it only has one of them -- R² collapses to noise-floor. Detection rescues
the dropped half, `select_interactions` correctly identifies `(x0, x1)` as
the only real cross term, and the fit is near-exact.

`interaction_plus_dominant` is the more realistic shape: a dominant additive
feature already gets most of the R² without any interaction awareness at
all (0.951), and detection recovers the remaining gap the interaction term
was responsible for. This is the case worth internalizing: the technique
does not need a "nothing works without it" scenario to be worth using --
even when the univariate ranking mostly gets it right, an interaction is a
distinct kind of signal that individually-ranked features cannot express on
their own, dominant or not.

## What this does not show

Both scenarios were constructed to *have* a real interaction and a strict
enough `top_n` for it to matter -- worst case for the univariate baseline,
by design. A dataset with no real interactions pays a small, bounded extra
cost (the `O(n_choose_2)` scoring pass -- see the `interaction-score*`
benchmark workloads in `benchmarks/workloads.py`) for a shortlist that
correctly comes back near-empty; it does not need to be enabled by default,
which is why every new parameter defaults to off.
