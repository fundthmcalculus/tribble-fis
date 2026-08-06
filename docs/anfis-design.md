# Why an ANFIS engine, and why it looks the way it does

**Answer, up front:** `tribblefis.anfis` implements Jang's (1993) canonical
five-layer ANFIS network directly -- grid-Cartesian rules, product t-norm,
closed-form consequent LSE alternated with a batch gradient step on every
premise parameter at once. It is not a variant of
`gaussian_regressor.MixtureOfGaussiansFuzzyRegressor`; it is the textbook
algorithm this package's own design has always been compared against
(`tribble-tree/HFIS_NOVELTY_REVIEW.md` names Jang 1993 explicitly), made
literal and runnable rather than only cited.

## Why grid-Cartesian rules, when the rest of the package avoids them

Every other TSK model here (`GaussianMixtureModel` and friends) uses an
*implicit* rule base: exactly one rule per output label/bucket, with no
per-input combinatorics. That design choice exists specifically to dodge what
ANFIS embraces -- a rule for every *combination* of one term per input,
`R = prod(K_f)` rules for `F` features with `K_f` terms each. Grid rules buy
something the implicit layout cannot express at all: a linear consequent
*local to a specific region of the joint input space*, not just local to one
input's range independent of the others. Section "Measured comparison" below
shows exactly where that matters and where it doesn't.

The combinatorial cost is real and the module does not hide it:
`init_anfis_model` raises `RuleExplosionError` above 5000 rules rather than
silently building something slow and overfit, and its message points at the
mixture regressor for problems with more than a handful of features.

## Why the product t-norm only, and why gradients are exact

The rest of the package supports five De Morgan t-norm families and defaults
to `"probability"` (`gauss_data.py`) because it is the one family that is
smooth everywhere, which is what makes an exact analytic gradient possible at
all (`docs/norm-family-evaluation.md`; `kernel.IncrementalFIS.supports_gradient`
gates the same way). ANFIS was defined with the product t-norm from the
start, and this module leans on it for something the rest of the package
doesn't need: the *grid* firing strength factors as a literal product across
features, `w[n,r] = prod_f g_f[n, k_f(r)]`. That factorization is what turns
"gradient with respect to every premise parameter" into a reshape + one
elementwise divide + one sum per feature, instead of a loop over rules or
membership functions -- see `anfis._premise_gradients`'s docstring for the
full derivation. A different t-norm family would break that factorization,
so unlike the rest of the package, ANFIS does not expose a norm choice.

## Why the hybrid rule is implemented as two *alternating* solves, not one joint gradient

Per epoch: solve every rule's consequent in closed form for the current
premises (exact, since the output is linear in the consequents for fixed
firing strengths -- reusing `regression.build_consequent_features` and the
same ridge-normal-equations shape as `regression.solve_tsk_consequents`, just
duplicated onto a grid firing matrix rather than imported, the same pattern
`fuzzytree/solve.py` uses for the same reason); then, with those consequents
now held fixed, take one Adam step on every premise parameter against the
training MSE.

Holding the consequents fixed during the gradient step is not an
approximation of Jang's algorithm -- it *is* the algorithm. "Hybrid" refers
to exactly this alternation, each half treating the other's parameters as a
constant. This is worth being precise about because `refine.py` has a
superficially similar-looking gradient (`_fold_mse_and_grad`) that
*re-solves* the consequents inside the derivative, because it is
differentiating a nested optimum and the envelope theorem doesn't apply
there. ANFIS's premise gradient never re-solves anything, so that subtlety
simply does not arise here -- a different problem, not a shortcut taken on
the same one.

## Why full-batch vectorized descent, not the package's block coordinate descent

`refine.py` / `kernel.IncrementalFIS` deliberately move *one* membership
function's `(mu, sigma)` at a time, caching per-cell folds to make that
cheap. That is the right shape for a non-smooth global search (GA/DE) over
`min/max`, where nothing is differentiable and there is no reason to update
every parameter in lock-step. ANFIS's premises are exactly the numbers a
smooth loss can be differentiated through *simultaneously* -- so
`_premise_gradients` computes the gradient for every `(mu, sigma)` on every
feature in one pass, `O(n * R * F)` like the forward pass itself, and
`_adam_step` updates all of them at once. No per-parameter cache, no
per-slot Python loop -- a handful of vectorized array operations per epoch.

Adam, not Jang's original fixed/decayed step size: it converges reliably
without a hand-tuned schedule, which is the same "pragmatic default over
literal textbook fidelity" reasoning behind this package's own choice of the
`probability` norm and the closed-form consequent solve over iterative
L-BFGS.

## Measured comparison

Three synthetic regression problems, three seeds each, 70/30 train/test
split, `ANFISRegressor(n_terms=3)` (mean R² over the 3 seeds) against
`MixtureOfGaussiansFuzzyRegressor` -- first at its own defaults (also meaned
over the 3 seeds), then the *best single result* found by sweeping
`n_output_buckets in {6,10,15}` x `tsk_order in {1st,2nd,full-2nd}` x the same
3 seeds (9x more fits than ANFIS gets), to give the comparison every benefit
of the doubt. Reproduce with `python -m benchmarks.anfis_vs_mixture`.

| problem | shape | ANFIS R² (mean) | mixture R² (defaults, mean) | mixture R² (best of 27 configs) | ANFIS rules | mixture rules |
|---|---|---|---|---|---|---|
| `sinc2d` (radial, 2 features) | 1200×2 | **0.996** | 0.800 | 0.982 (10 buckets, full-2nd) | 9 | 6–15 |
| `additive3d` (separable, 3 features) | 1200×3 | **0.999** | 0.932 | 0.991 (15 buckets, 2nd) | 27 | 6–15 |
| `interaction2d` (`x0·cos(x1)`, 2 features) | 1200×2 | **0.999** | −0.07 | 0.862 (15 buckets, full-2nd) | 9 | 6–15 |

(Mean ANFIS fit time 0.07–0.45s per fit at its one configuration; the
mixture regressor's own fits are each 0.18–0.22s, but the "best of 27" column
cost 27 of them. ANFIS is not claiming to be faster to *train* at equal
tuning effort -- only that its per-epoch gradient step stays cheap as rules
grow, which the benchmark suite's `anfis-fit*` workloads track going forward.)

**Reading `interaction2d` is the point of this table.** The mixture
regressor's implicit rule base is one rule per output bucket, each an AND
across features of that bucket's own Gaussians -- there is no rule that says
"in *this* region of `(x0, x1)` jointly, behave like *this* line," only "for
*this* range of `y`, `x0` looks like this and `x1` looks like that." A target
that changes sign depending on the *combination* of two inputs is exactly
what that structure cannot represent without help, and it shows: R² is
negative at defaults. Raising `tsk_order` to `full-2nd` recovers most of the
gap (0.862, its best result of the 27 configs tried) because it adds explicit
`x0*x1` cross terms *to the consequent*
-- a different, and here less direct, way of encoding the same interaction
that ANFIS's grid rules encode structurally in the *antecedent*. On the
separable and radial problems, where no such interaction exists, tuning
closes nearly all of the gap, and either model is a reasonable choice.

**What this does and does not show.** It is not evidence that ANFIS's
closed-form consequent solve is more "correct" than the mixture regressor's
-- they are the same solve (`docs/analytic-gradient-evaluation.md` and this
module both reuse `build_consequent_features`/ridge normal equations), and
`tribble-tree/HFIS_NOVELTY_REVIEW.md` is explicit that "ANFIS LSE overfits"
was checked and refuted as a rationale, so it is not repeated here. What
differs is the rule *structure* the two premises can express, and that is a
property of grid-Cartesian antecedents versus implicit per-label ones, fully
separable from which solver fits the linear part. Pick ANFIS when the
interaction structure is unknown and the feature count is small enough for
the grid; pick the mixture regressor otherwise, or add explicit interaction
terms to its consequent when you suspect the same problem.

## What is deliberately not implemented (yet)

- **Only Gaussian premises**, one `n_terms` per feature. No trapezoid/
  triangular option, unlike the rest of the package -- ANFIS's premises are
  conventionally Gaussian or generalized bell, and the gradient in
  `_premise_gradients` is specific to the Gaussian's closed-form partials.
- **No Cython/GPU backend.** `kernel.py`/`gpu.py` exist because the mixture
  model's forward pass is the dominant cost across tens of thousands of
  fitness evaluations; ANFIS's cost profile (a few hundred gradient epochs,
  not tens of thousands of independent candidate evaluations) has not yet
  been measured against that bar. `benchmarks/workloads.py`'s `anfis-fit*`
  rows exist so that if it ever does need one, there is already a checksum
  to hold it to.
- **Ragged per-feature `n_terms`** works (pass a list), but there is no
  automatic term-count selection -- unlike `ruspini.py`'s landmark-merging
  heuristic, every feature's term count is a hyperparameter you choose.
