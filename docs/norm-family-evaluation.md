# Which t-norm family should be the default?

**Answer: `probability`, not the textbook `min/max`.** Measured across 18
dataset x split combinations, min/max was the *worst* of the four De Morgan
families for classification accuracy — by 2.5 points.

## The comparison

Every cell is one heuristic model (KMeans + `norm.fit`, identical across
families) evaluated and refined through each operator pair, scored on a 30%
holdout. Six datasets — iris, wine, breast_cancer, digits (64 features, 10
classes), and two `make_classification` problems with deliberately overlapping
classes — times three splits.

| family | heuristic | refined | worst | vs min/max (refined) |
|---|---|---|---|---|
| min/max | 0.7418 | 0.7881 | 0.3722 | — |
| hamacher | 0.7523 | 0.8029 | 0.3833 | +0.0148 ± 0.0078 |
| **probability** | **0.7767** | **0.8135** | **0.4028** | **+0.0254 ± 0.0063** |
| einstein | 0.7782 | 0.8175 | 0.4083 | +0.0294 ± 0.0061 |

probability wins 13 of 18 and loses 2; einstein wins 15 and loses 1. Both are
four-plus standard errors clear of min/max. The gap *between* those two is
0.004 ± 0.009 — not separable.

**Why probability over einstein**, given einstein's nose is marginally ahead:
they are statistically tied, probability is the cheapest to evaluate (a multiply
where einstein needs two divisions), it is the one family whose objective is
smooth everywhere — which is what makes an exact analytic gradient possible at
all — and product / probabilistic sum is the more familiar pair. If einstein
later separates on more evidence, the constant is one line.

## Avoid `luk`

Not a close call. Łukasiewicz's bounded sum saturates and its bounded product
underflows, and with more than a handful of features it leaves most rows with no
membership at all — which the classifier then answers with a uniform fallback:

| dataset | features | accuracy | rows with zero firing |
|---|---|---|---|
| iris | 4 | 0.6991 | 41% |
| wine | 13 | 0.3427 | 99% |
| breast_cancer | 30 | 0.3743 | 100% |
| synthetic | 40 | 0.2675 | 100% |

Mean accuracy 0.4458 against ~0.97 for every other family. It is kept because it
is a textbook t-norm and someone may want it on two or three features, but it
should never be a default and is now called out in the estimator docstring.

## "Training got slower" — no, min/max was stalling

Switching the default makes the benchmark's `refine-*` rows 1.6–1.9x *slower*,
and that number is misleading enough to be worth spelling out.

Those workloads start refinement from a **randomly parameterised** model, which
is the right fixed point for measuring a kernel and a poor proxy for training,
where the search starts from the heuristic fit. From each start, on the wide
shape:

| start | min/max | probability |
|---|---|---|
| heuristic (how you actually train) | 269.9 ms → accuracy 1.0000 | **237.8 ms** → 1.0000 |
| random (what the benchmark uses) | 609.3 ms → accuracy **0.9427** | 1247.8 ms → **0.9992** |

From a realistic start, probability is **1.13–1.16x faster**. From a random
start it uses 7 444 evaluations against min/max's 3 199 — because on a smooth
surface L-BFGS-B keeps making progress, where on min/max's piecewise-constant
one it hits a flat region and gives up at 0.9427. That is not a regression;
those two runs are not doing the same amount of work, and only one of them
finished.

Inference is unambiguously faster: `predict-large` 30.64 ms → 27.36 ms (1.12x).

## Consequences

Three benchmark checksums move — `predict-large`, `refine-classifier`,
`refine-classifier-wide`. That is correct and expected: they exercise the
default pair, and the default changed. Every other checksum, including all the
`forward-*` rows that pass an explicit pair, is untouched. `results/` is
re-baselined in the same commit.

Wall-clock on the two `refine-*` rows is **not comparable across this change**,
for the reason above. Use the heuristic-start table.

## Reproducing

The scripts behind the tables sweep
`sklearn.datasets.load_{iris,wine,breast_cancer,digits}` plus
`make_classification`, and `benchmarks.workloads.make_dataset/make_model` for
the synthetic shapes. `tests/test_classifier_norm_passthrough.py` locks in the
plumbing this depends on — before it, refinement always optimised under min/max
regardless of what the estimator predicted with, so none of these numbers would
have meant anything.
