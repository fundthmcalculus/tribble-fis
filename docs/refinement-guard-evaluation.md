# Should refinement be allowed to reject itself?

**Answer: no. The guard is now `"none"` by default.** Every rejection rule
tested destroyed expected accuracy, and removing the guard also returns a
quarter of the training data to the search. Combined, that is **+1.5 points**
over the old behaviour.

This started as issue #65 — the guard looked underpowered and the plan was to
strengthen it. Four routes were implemented and measured. All four made things
worse.

## How the guard was scored

The guard answers "did this refinement beat its starting point?", and that has a
ground truth: run both models on a test set neither the search nor the guard
ever sees. So the guard is scored as a binary classifier, via
`benchmarks/guard_bench.py`.

108 cases: six datasets (iris, wine, breast_cancer, digits, two overlapping
`make_classification` problems) × three splits × six refinement configurations.
The configurations deliberately span sensible and reckless — no shrinkage, six
sweeps, a starved validation split — because a guard benchmark needs failures to
detect, and the honest way to produce them is settings that genuinely overfit.

## How the stats moved, in order

| guard | accept rate | false accepts | false rejects | mean kept accuracy |
|---|---|---|---|---|
| `legacy` (accuracy >, CE tiebreak) | 0.676 | 6 / 12 | 24 / 85 | 0.8145 |
| `ce` (score on the loss actually minimised) | 0.676 | 7 / 12 | 25 / 85 | 0.8139 |
| `effect-size` (gain > 1 paired SE) | 0.426 | 2 / 12 | 44 / 85 | 0.8084 |
| `mcnemar` (paired exact test, α = 0.10) | 0.324 | 2 / 12 | 53 / 85 | 0.8044 |
| **`none` (keep always, train on all data)** | 1.000 | 12 / 12 | 0 / 85 | **0.8296** |

Each stricter rule does exactly what it was designed to do — `mcnemar` cuts
false accepts from 6 to 2 — and each one is a net loss, because it buys that
with far more false rejects.

## Why every guard loses

The base rate is lopsided and the magnitudes point the same way:

- refinement **beats** its start in 85 of 108 cases, and **loses** in 12
- when it wins, it gains ~4.0 points
- when it loses, it sheds ~1.9 points on average, ~4.4 at worst

So the expected value of accepting blindly is strongly positive, and a rule that
rejects one bad refinement while rejecting three good ones is losing money on
every trade. No threshold fixes this; it is the base rate, not the test.

The safety argument does not survive either. `none` has the **better** worst
case (0.4583 against 0.4028 for every guard) — the run all guards rejected was
one where the heuristic was itself terrible and the refinement was an
improvement.

Nor was the guard rescued by restricting to the reckless configurations, which
is where I expected it to earn its keep. `none` wins in **all six**
configurations individually, including `no-shrink-deep` and `tiny-val`. The
ridge shrinkage toward `x0`, the box bounds and the `sub_maxfun` budget already
bound how far the search can wander; `l2_shrink=0` does not produce the
catastrophe the guard was defending against.

## The half that mattered more

Dropping the guard also removes the reason to withhold `val_fraction` of the
training data. With the search using all of it:

| | mean kept accuracy | worst | vs never refining |
|---|---|---|---|
| `none`, still holding out 25% | 0.8184 | 0.4250 | +0.0417 |
| `none`, training on everything | **0.8296** | **0.4583** | **+0.0529** |

Reclaiming the data (+0.0112) was worth nearly three times as much as removing
the rejection rule (+0.0039). The guard's real cost was never its false
rejects — it was the quarter of the dataset it consumed to make them.

**It is not free.** Training on 100% of the data instead of 75% makes every
fitness evaluation proportionally more expensive: `refine-classifier` goes
129 ms → 152 ms (0.85x), `refine-classifier-wide` is unchanged within noise.
Roughly 17% more training time for 1.5 points of accuracy, which is the right
side of that trade for a step users opt into — but it is a trade, and the two
`refine-*` benchmark checksums move because the search now sees different data.

## What this corrects

`MixtureOfGaussiansFuzzyClassifier.refine` used to be documented as: *"The
refinement is accepted only if it does not worsen a held-out validation split,
so it can never hurt."* The mechanism existed; the guarantee did not. Refinement
is worse than not refining about one time in nine, typically by ~2 points, and
the guard caught half of those at best while discarding good refinements at
three times the rate. The docstring now says what is measured.

## The Ruspini partition: a different answer

`refine_ruspini_partition` carried a copy of the same guard. It would have been
easy to delete it by analogy. Measured on its own benchmark
(`benchmarks/ruspini_guard_bench.py`), the analogy does not hold — **its guard
stays**.

48 paired cases, differences against `none`:

| guard | mean diff | se | t | verdict |
|---|---|---|---|---|
| `legacy` | +0.0049 | 0.0058 | 0.84 | not significant |
| `ce` | +0.0021 | 0.0071 | 0.29 | not significant |
| `effect-size` | −0.0045 | 0.0071 | −0.64 | not significant |
| `mcnemar` | −0.0240 | 0.0094 | −2.56 | **significantly worse** |

No evidence for a change, so the existing `"legacy"` default stands. The two
refiners now have different guard defaults, and the reason is that they were
measured separately and gave different readings.

**Why they differ.** The base rate is much less lopsided here — refinement helps
2.2x more often than it hurts, against 7.1x on the classifier — so a guard has
more to catch. The Ruspini search is also lower-dimensional (one apex knot per
term, against two parameters per membership function) and optimises a *shared*
partition every class rule reads, so there is less to overfit and the
data-reclaiming half of the classifier's win does not materialise: `none` trains
on a third more rows and still only ties.

Accepting blindly remains positive expected value here too (+0.0946 when
refinement helps against −0.0308 when it hurts). The guard is simply not costing
anything on this search, which is a different situation from earning its keep.

**Power caveat, stated plainly.** This benchmark is 48 cases, not 108: Ruspini's
coordinate step is a *grid line search* (25 evaluations per knot per sweep), so
the full matrix did not finish a single dataset in fifteen minutes. Datasets and
configurations were cut to make it tractable. With `se ≈ 0.006` this can only
resolve differences above roughly 0.012 — so "not significant" here means "no
detectable difference at this scale", not "proven equal". A genuine effect
around half a point would be invisible.

**One finding is consistent across both searches:** `mcnemar` is significantly
worse. Rejecting on a strict significance test throws away far more good
refinements than bad ones, wherever it was tried.

## What remains

- The alternatives are still selectable — `guard="mcnemar"` if a bounded worst
  case is worth ~2.5 points of mean accuracy to you.
- Routes C (per-sweep validation curve) and D (cross-validated refereeing) are
  **dropped**, not deferred. C never fitted the accept/reject frame — it changes
  which model is produced rather than filtering the finished one — and with the
  classifier's holdout reclaimed there is no validation split left to draw a
  curve on; reintroducing one would give back more than the whole rejection-rule
  question was worth. D is the most expensive route and makes a decision the
  evidence says should not be made.
- Einstein remains nominally ahead of `probability` as a norm family
  (+0.0294 against +0.0254) but statistically tied. Left at `probability`.

## Reproducing

```bash
python -m benchmarks.guard_bench                  # all guards
python -m benchmarks.guard_bench --guards legacy mcnemar
python -m benchmarks.guard_bench --margin 0.01    # what counts as a tie
```
