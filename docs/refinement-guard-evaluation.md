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

## What remains

- The alternatives are still selectable — `guard="mcnemar"` if a bounded worst
  case is worth ~2.5 points of mean accuracy to you.
- Route C from #65 (track the validation curve per sweep and keep the argmin)
  is **not** implemented. It is the one route that does not fit the
  accept/reject frame: it changes which model is produced rather than filtering
  the final one, and it is not obviously in conflict with `guard="none"` — with
  the data reclaimed there is no validation split to draw the curve on, so it
  would need its own design. Left open on #65.
- Route D (cross-validated refereeing) is also unimplemented, and now looks
  unattractive: it is the most expensive route and it makes a decision the
  evidence says should not be made.
- `refine_ruspini_partition` still has the original copied guard. Worth the same
  treatment, but it needs its own measurement — the Ruspini partition is a much
  lower-dimensional search and the base rate there may differ.

## Reproducing

```bash
python -m benchmarks.guard_bench                  # all guards
python -m benchmarks.guard_bench --guards legacy mcnemar
python -m benchmarks.guard_bench --margin 0.01    # what counts as a tie
```
