# Layered sub-dominant rules

**Answer: it works, and on the six real datasets tested it almost never fires.**
On problems with the structure it targets — a class hidden inside another,
separated by evidence invisible at global scale — it is **+12.1 points**, and it
chains through layers to reach classes nested two deep. On iris, wine,
breast_cancer, digits and two `make_classification` problems it mines
essentially nothing and moves accuracy by +0.0002 over 36 held-out cases, never
downward. It is off by default, and the honest reason is in the section on the
gain floor: at a permissive setting it *did* fire on those datasets, and every
firing was a coin flip.

## The defect

A rule per label is one account of that class, fit from that class's own
marginals. Where two classes overlap, one rule wins the whole overlap and the
other loses it, and nothing in the rule base records that the region was
contested. The confusion matrix knows — it is exactly the off-diagonal mass —
but the rules do not.

The sharpest form: class `B` is a small, dense pocket sitting inside class `A`'s
cloud, separated from it only by a feature that is uninformative everywhere else.
`B` is a minority, so that feature never survives the global ranking, and `B`'s
Gaussians land inside `A`'s. `A`'s rule — fit on far more rows — takes the whole
pocket. Re-fitting `B` cannot help: at the scale of the whole dataset the
separating evidence is not there. Restricted to the rows `A` gets wrong, it is
the only thing there.

## The correction

Run the training data through the fitted model, read the top-n confusions off the
resulting matrix, and for each `(predicted P, true T)` pair add a more specific
rule *underneath* the one that gets it wrong:

```
IF   rule P fires
AND  x is [...] AND y is [...]        <- fit on the rows P gets wrong
THEN T   (instead of P)
```

```
LAYER 0:
  IF   rule A fires
  AND  f2 is [~4.45+/-0.385]
  AND  f1 is [~-0.0186+/-0.775]
  AND  f0 is [~-0.0766+/-0.764]
  THEN B   -- instead of A; activation >= 0.020, n=74, 17% of region A really B
```

`f2` is the hidden feature. It appears in the sub-rule and nowhere else in the
rule base, which is the mechanism working exactly as intended.

Three things distinguish this from re-fitting `T`'s own rule:

- **It is gated.** `w_sub = T(w_P, a_sub)` is silent everywhere rule `P` is
  silent, so the sub-rule is an *exception to `P`* rather than a competing
  account of `T`.
- **It is fit on the confusion.** The antecedent is fit on the rows `P` claims
  but that are truly `T`, so it describes where `P` is wrong, not where `T`
  lives. That restriction is what lets it key on evidence too weak to survive
  global feature ranking.
- **It is layered.** A sub-rule's consequent may be another sub-rule's parent, so
  corrections chain. Each layer is mined against the labels the previous layer
  produced, so layer 1 addresses the confusions layer 0 *left*, not the ones it
  fixed.

## Why the decision is by precedence, not by argmax

Every t-norm obeys `T(a, b) ≤ min(a, b)`. So `w_sub ≤ w_P` identically: **the
gate that makes a sub-rule subordinate also caps its activation below the very
rule it exists to correct.** A flat argmax over firing strengths would render
every sub-rule inert, no matter how good its antecedent.

Resolution is therefore by *specificity*. Where the gated activation clears the
rule's threshold, the more specific rule takes the label, and the parent's own
firing strength is left untouched — the ordinary reading of an exception in a
rule base. This is why `apply_subdominant` acts on labels rather than strengths,
and it is the one structural decision in this design that could not have been
made differently without abandoning either the gate or the pure `T` consequent.

**Termination.** A row may never return to a label it has already held. Without
that, `P → T` and `T → P` rules would trade a row back and forth and the result
would depend on iteration order. With it, every row's label sequence is strictly
non-repeating and the cascade ends in at most as many layers as there are
classes. Within a layer, a contested row goes to the higher activation, then to
the earlier rule, so nothing depends on dictionary ordering.

**Probabilities.** Precedence supplies no strength for a sub-rule to contribute,
so `predict_proba` would otherwise disagree with `predict` — which quietly breaks
calibration, ROC curves and `cross_val_predict(method="predict_proba")`. Where an
override fires, the parent's and the corrected class's entries are exchanged:
the smallest edit consistent with the decision, still summing to one, and the
corrected class becomes the argmax exactly because the parent was.

## Existence proof: the hidden pocket

Six splits, 30 % held out.

| problem | base | with cascade | delta | rules | layers |
|---|---|---|---|---|---|
| 1 pocket (2 classes) | 0.8436 | 0.9641 | **+0.1205** | 1.0 | 1.0 |
| 2 nested (3 classes) | 0.8050 | 0.9113 | **+0.1063** | 2.2 | 1.5 |

On a single held-out run of the first problem, class `B`'s recall goes from
**0.34 to 0.88** while precision stays at 0.92 — the cascade is recovering the
pocket, not trading `A` away for it.

## Real data

Six problems × six splits, 30 % held out, out-of-fold confusion matrix.

| n_gaussians | mean delta | better | worse | unchanged | worst case |
|---|---|---|---|---|---|
| 1 | +0.0000 ± 0.0000 | 0 | 0 | 36 | +0.0000 |
| 2 | +0.0009 ± 0.0005 | 3 | **0** | 33 | +0.0000 |

Almost entirely inert. That is the result, and it is worth being precise about
what it does and does not say: these six datasets do not have much of the
structure this mechanism repairs. The control below supports that reading rather
than the alternative that the mechanism is simply weak.

## Against the other confusion repairs

Same six datasets and splits, `n_gaussians=2`, everything measured against the
same base model.

| variant | mean acc | vs base | better | worse |
|---|---|---|---|---|
| base | 0.8569 | +0.0000 | 0 | 0 |
| `exclude_cross_terms` | 0.8758 | **+0.0188** | 14 | 0 |
| `subdominant` | 0.8571 | +0.0002 | 1 | 0 |
| both | 0.8758 | +0.0188 | 14 | 0 |
| sequence-classifier experts | 0.8569 | +0.0000 | 0 | 0 |

The last row is the control. `MixtureOfGaussiansFuzzySequenceClassifier` attacks
the *same* confusions by a completely different route — whole binary models
consulted in `predict` — and it too changes nothing on these datasets, in either
direction, on any split. Two independent mechanisms aimed at between-class
confusion both finding nothing is evidence about the datasets, not about either
mechanism.

Meanwhile `exclude_cross_terms`, which repairs a different defect (a rule
over-claiming cells of its own outer product), finds plenty on the same data.
The two are not substitutes, and `both` costs nothing relative to exclusions
alone.

## The gain floor is where the safety comes from

A sub-rule is kept only when its tuned threshold improves accuracy over the
parent's region by more than `min_region_gain`. The threshold search can always
choose to fire on nothing, so even `0.0` rejects a rule that cannot help *at
all*. It does not reject a rule that helps *slightly* — and that is where the
losses were:

| `min_region_gain` | real: mean | worst | better | worse | rules | nested synthetic | layers |
|---|---|---|---|---|---|---|---|
| 0.00 | +0.0006 | −0.0175 | 7 | **5** | 0.9 | +0.1078 | 2.2 |
| 0.01 | +0.0011 | −0.0074 | 6 | 2 | 0.5 | +0.1078 | 2.0 |
| 0.02 | +0.0010 | −0.0074 | 6 | 2 | 0.3 | +0.1078 | 1.7 |
| **0.03** | **+0.0009** | **+0.0000** | 3 | **0** | 0.1 | **+0.1063** | **1.5** |
| 0.05 | +0.0002 | +0.0000 | 1 | 0 | 0.0 | +0.0573 | 1.0 |

At `0.0` the stage fires on real data and is a **coin flip**: seven wins, five
losses, worst case −1.75 points. A rule whose region gain is 1–2 points on
training data is measuring noise, and the held-out result says so.

`0.03` removes every loss. `0.05` also removes every loss but is strictly worse
than `0.03`: same zero-loss column, lower mean, and it **halves the nested
synthetic win and collapses the cascade to one layer**. That is not a
coincidence — a fixed absolute floor is harsher on deeper layers by
construction, because layer 0 has already fixed the rows that made the region
gain large. `0.03` is the knee: the largest floor that still admits a chained
correction.

### Honest thresholds

Each rule's threshold is bisected against activations from folds that never saw
the row: the region is split, the antecedent re-fit on the confused rows of each
training part, and the held-out part scored. A sub-rule fit on dozens of confused
rows fits them closely, so in-sample activation is far higher on those rows than
on new ones and the threshold comes out too permissive.

This is the right thing to do and it is worth noting that on this data it was
*not* what rescued the real-data result — switching it on moved the mean from
+0.0005 to +0.0006 and the loss count from 6 to 5. The gain floor did the work.
Both are kept: the honest threshold is correct on its own terms and costs one
extra fit per pair, and its effect will be larger on smaller confusion regions
than these datasets happen to produce.

## What this is not

Not a replacement for the sequence classifier's experts. Those re-select
features and re-fit a whole binary model per pair, and can therefore find
structure a single gated rule cannot. A sub-dominant rule is a *rule in the rule
base* instead: it is inspectable next to every other rule, it carries its own
antecedents, `describe_subdominant` prints it as the exception it is, and it
participates in `predict`/`predict_proba` through one consistent path rather than
a post-processing cascade. The two are alternative treatments of the same
confusion and there is no reason to run both, which is why `subdominant` is not
passed through the sequence classifier's layers the way `exclude_cross_terms` is.

## Reproducing

```bash
python -m benchmarks.subdominant_bench                 # everything
python -m benchmarks.subdominant_bench --only real     # the held-out table
python -m benchmarks.subdominant_bench --only compare  # against the alternatives
```
