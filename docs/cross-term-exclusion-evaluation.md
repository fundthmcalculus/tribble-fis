# Cross-class confusion inside a single rule

**Answer: mine the cells, negate the bad ones, and attach the clause to the one
rule that over-claims.** Across 36 held-out cases it is **+1.9 points** at
`n_gaussians=2` and **+3.4** at `n_gaussians=3`, and it was **never worse than
the baseline in any case measured**. It is off by default anyway, because on
half the datasets tested it correctly finds nothing at all — the defect it
repairs is real but not universal, and a stage that usually no-ops does not
belong in the default path.

## The defect

A zeroth-order TSK classifier gives each label one rule, and that rule is a
conjunction *of disjunctions* — one disjunction per feature, folded by the
t-conorm over that feature-label's membership functions:

```
IF (x is X1 OR X2 OR X4) AND (y is Y1 OR Y3) THEN A
```

The conorm is where the information goes. By the time the t-norm ANDs the
features together, the rule no longer knows *which* `x` term carried the
disjunction, so it cannot condition on the pairing. It admits the entire outer
product — `X1&Y1, X2&Y1, X4&Y1, … X4&Y3` — with no way to prefer one cell over
another.

That is correct when the label occupies every cell, and wrong when it does not.
If `X1&Y3` is in truth class `B`, rule `A` claims it at full strength.

**Re-fitting the antecedents cannot fix this.** `X1` and `Y3` are individually
right for `A`; both earn their place in the rule from the cells `A` legitimately
owns. Moving either one to dodge `X1&Y3` costs `A` those cells. The defect is in
the combination, so the repair has to name the combination — which is what a
per-feature membership function structurally cannot do.

## The correction

Mine the cells from training data and append the negation of the bad ones to the
offending rule:

```
IF (x is X1 OR X2 OR X4) AND (y is Y1 OR Y3)
   AND NOT (x is X1 AND y is Y3)
THEN A
```

At inference the parent's firing becomes `T(w_A, 1 - strength·cell)`, where
`cell` is the t-norm of the *individual* memberships the clause names — not the
conorm the parent folds. Naming individual memberships is the whole mechanism:
it is the only way to point at one cell of the product.

Three properties make the reduction narrow enough to trust:

- **It is per parent rule.** Only `A`'s column moves. `B` is untouched, and the
  blamed class is never boosted — it takes the argmax by default once `A` stops
  over-claiming. Boosting `B` would be a second, unrelated edit to a rule whose
  own data never asked for it.
- **It is per cell.** `X1` still fires for `A` alongside `Y1`; `Y3` still fires
  for `A` alongside `X2` and `X4`. Only the confused block is withdrawn.
- **It is inert where it finds nothing.** A model with no mined clauses fires
  bit-identically to one built before the feature existed.

## Existence proof: the checkerboard

Two features, four blobs, class `A` on the diagonal and `B` on the
anti-diagonal. Both classes have **identical marginals** — `A` is at `X1` and
`X2`, and at `Y1` and `Y2`; so is `B`. A rule per label is therefore at chance by
construction, whatever its Gaussians do, because its firing strength is a
function of marginals that do not differ.

| problem | base | with clauses | delta | clauses |
|---|---|---|---|---|
| checkerboard 2×2 | 0.5023 | 0.8889 | **+0.3866** | 4.0 |
| checkerboard 3×3 | 0.7546 | 0.8009 | +0.0463 | 1.0 |
| striped 3×3, 3 classes | 0.3539 | 0.4431 | +0.0892 | 3.0 |

`n_gaussians=2`, three splits. The 2×2 row is the clean case; the 3×3 rows are
under-resourced at two memberships per feature (a 3×3 grid needs three) and are
reported as-is rather than tuned into a better number.

## Real data

Six problems × six splits, 30 % held out, mining on the training split only.
Cases are counted individually rather than averaged per dataset, so a method
that wins big twice and quietly loses four times cannot hide behind a mean.

| n_gaussians | mean delta | better | worse | unchanged | worst case |
|---|---|---|---|---|---|
| 1 | +0.0000 ± 0.0000 | 0 | 0 | 36 | +0.0000 |
| 2 | **+0.0186 ± 0.0057** | 14 | **0** | 22 | +0.0000 |
| 3 | **+0.0344 ± 0.0085** | 16 | **0** | 20 | +0.0000 |

Per dataset, at `n_gaussians=2`:

| dataset | base | with clauses | delta | clauses |
|---|---|---|---|---|
| iris | 0.9593 | 0.9593 | +0.0000 | 0.0 |
| wine | 0.9753 | 0.9753 | +0.0000 | 0.0 |
| breast_cancer | 0.9425 | 0.9425 | +0.0000 | 0.0 |
| digits | 0.8565 | 0.8571 | +0.0006 | 0.8 |
| synth_easy | 0.8537 | 0.8833 | +0.0296 | 2.5 |
| synth_hard | 0.5543 | 0.6358 | +0.0815 | 11.2 |

Two things in that table matter more than the mean.

**`n_gaussians=1` is exactly zero, not approximately zero.** With one membership
per feature-label the rule *is* a single cell — there is no outer product to
reduce. Mining returns nothing and says so in its diagnostics
(`no_multi_mf_features`), rather than leaving an empty result to be guessed at.

**The gains are concentrated, not spread.** Three of six datasets mine zero
clauses and move not at all. The wins are on the two `make_classification`
problems with several informative features and low class separation — where
classes genuinely are multi-modal and interleaved, which is precisely the
structure the outer product mishandles. This is a repair for a specific defect,
and it declines to act when that defect is absent. That is the reason it is
opt-in: a default-on stage whose modal effect is "nothing" costs a mining pass
on every fit to buy nothing on most of them.

## Where the safety comes from

The zero-loss column is the mining thresholds, not luck. Sweeping them at
`n_gaussians=2`, 36 cases each:

| config | mean | worst | better | worse | clauses |
|---|---|---|---|---|---|
| **default** | **+0.0186** | +0.0000 | 14 | **0** | 2.4 |
| `min_support=3` | +0.0195 | +0.0000 | 17 | 0 | 5.0 |
| `min_support=5` | +0.0194 | +0.0000 | 17 | 0 | 3.4 |
| `min_support=25` | +0.0080 | +0.0000 | 7 | 0 | 1.4 |
| `cross_margin=-1.0` (off) | +0.0183 | +0.0000 | 14 | 0 | 2.4 |
| `cross_margin=0.0` | +0.0183 | +0.0000 | 14 | 0 | 2.4 |
| `cross_margin=0.2` | +0.0097 | **−0.0222** | 10 | **2** | 1.0 |
| `max_purity=0.7` | +0.0186 | +0.0000 | 14 | 0 | 2.4 |
| `max_purity=0.9` | +0.0186 | +0.0000 | 14 | 0 | 2.4 |
| `max_clauses=1` | +0.0120 | +0.0000 | 12 | 0 | 0.9 |
| `max_clauses=16` | +0.0175 | +0.0000 | 14 | 0 | 3.9 |

`min_support` is the load-bearing threshold: at 25 it starves the stage of more
than half its gain, and loosening it below the default buys very little. The
default of 10 sits at the knee.

`max_purity` does nothing at all here — every cell that clears `min_support` and
has a blamed class is already far past the 0.5 default, so raising the ceiling
admits no new cells. It is kept as an explicit knob rather than removed because
it is the parameter that *states* what a bad cell is, and a reader of a mined
rule base should be able to see the criterion rather than infer it.

### The cross test does not pay for itself in accuracy

This is the honest result in the table. `cross_margin` is the test that a cell
must be worse than *each* of its single-feature terms — the thing that separates
genuine cross-confusion from one badly placed membership function. Turning it
off entirely (`-1.0`) changes the mean by 0.0003. On this data it earns nothing.

It is kept on, at a low default, for a reason that is not accuracy: without it
the stage will happily describe a marginal defect as a cross-term. A rule that
says `AND NOT (x is X1 AND y is Y1)` when the real problem is that `X1` is
misplaced has withdrawn half the affected region and given a false account of
why. That matters for a model whose selling point is that its rules can be read.
`tests/test_exclusion.py` builds both cases and pins the discrimination.

The `cross_margin=0.2` row is worth reading carefully, because it looks
backwards: a *stricter* filter produces two losses the default does not. A
stricter margin accepts a strict subset of the same cells, so this is not new
bad clauses being admitted — it is helpful companions being removed, leaving a
marginally harmful clause without the ones that compensated for it. Clauses on
one rule are not independent, which is an argument against tuning `cross_margin`
upward and for leaving `max_clauses` generous.

### Strength

`strength` scales the negation to `1 - strength·cell`; `1.0` is the hard veto.

| strength | mean | worst | better | worse |
|---|---|---|---|---|
| 0.25 | +0.0080 | +0.0000 | 12 | 0 |
| 0.50 | +0.0145 | +0.0000 | 13 | 0 |
| 0.75 | +0.0164 | +0.0000 | 13 | 0 |
| **1.00** | **+0.0186** | +0.0000 | 14 | 0 |

Monotone, with no sign that softening buys safety — the worst case is 0.0000 at
every setting. A cell that clears the mining thresholds is worth vetoing
outright, so the hard clause is the default. The knob remains for callers who
have loosened `min_support` and want the confidence discount back.

## Why not a rule per cell instead

The alternative repair is to stop folding the conorm: give each label a rule per
cell of its outer product, so the pairing is never lost. That works, and it
multiplies the rule base by the size of the product — which is the cost the
conorm existed to avoid — and it re-fits the cells that were already right.

Mining exclusions keeps one rule per label, adds a clause only where the data
shows a specific cell is wrong, and leaves the rule base readable: the clauses
are exceptions, and `describe_exclusions` prints them as exceptions, with the
support and the blamed class that justified each one.

## Relationship to the sequence classifier's experts

`MixtureOfGaussiansFuzzySequenceClassifier` also targets cross-class confusion,
from the other side. The two are independent halves:

- an **expert** is a *between*-rule repair — it re-decides a whole confused class
  pair with freshly selected features, and moves a decision boundary;
- a **clause** is a *within*-rule repair — it withdraws one rule from a cell it
  should never have claimed, and moves no boundary the experts arbitrate.

Neither substitutes for the other. An expert cannot repair a rule that
over-claims a cell (it re-decides the pair everywhere, not in the cell), and a
clause cannot supply evidence a rule does not have. The sequence classifier
passes `exclude_cross_terms` to every layer, so the base model's rules are
already narrowed before the confusion matrix that selects the experts is
measured.

## Reproducing

```bash
python -m benchmarks.exclusion_bench                 # everything
python -m benchmarks.exclusion_bench --only real     # the held-out table
```

The tables above use six splits; the script's default is three. Accuracy numbers
move slightly with the split count, the sign and the zero-loss column do not.
