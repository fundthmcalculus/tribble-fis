# Should classifier refinement use an analytic gradient?

**Answer: it is implemented, tested, and off by default. Leave it off unless you
have measured your own case.** It is ~1.4x faster and lands on a different, not
reliably better, model.

Reproduce with `python -m benchmarks.bench -k refine` and the paired-holdout
harness described at the bottom. Hardware as in `benchmarks/README.md`.

## Why the question came up

After the fuzzy arithmetic was made ~5x faster, training stopped being dominated
by it. Profiling `refine-classifier-wide` (4k x 20 features x 6 labels x 3 MF,
720 free parameters) showed the objective itself was 62% of the run and SciPy's
L-BFGS-B machinery the other 38% — and of the objective's 3 198 evaluations,
**two thirds existed only so L-BFGS-B could finite-difference a two-parameter
block.** Three evaluations per gradient: one value, two differences.

An instrumented run put the ceiling at **1.71x** (wide) and **1.50x** (small) if
the gradient were entirely free.

## What was built

`d(CE)/d(mu, sigma)` for one membership function, carried through the same two
folds that compute the value, so it costs no extra pass over the data:

- `_fis_kernel.refold_label_with_grad` — the value fold plus the derivative
  chain, in one threaded pass.
- `IncrementalFIS.evaluate_slot_with_grad` — column value and `d(column)`.
- `_CrossEntropy.with_column_grad` — turns the column derivative into the loss
  gradient. Only one column moves, so
  `d(-log p_i)/dθ = -([y_i == c]/fs[i,c] - 1/S_i) · dfs[i,c]/dθ`.

Correct to ~1e-8 relative against a central finite difference of the same
objective, on every slot, for both `probability` (exact) and `min/max`
(subgradient) — see `tests/test_analytic_classifier_grad.py`.

## Result: 1.43x faster, and an accuracy lottery

Twelve paired runs. Both arms see identical data and an identical starting
model; accuracy is measured on a 30% holdout neither arm trained on.

| case | finite-diff | analytic | Δ accuracy | speed |
|---|---|---|---|---|
| small/s1 | 0.9667 | 0.9900 | **+0.0233** | 1.15x |
| small/s2 | 0.9367 | 0.8933 | −0.0433 | 1.24x |
| small/s3 | 0.9900 | 1.0000 | +0.0100 | 1.34x |
| small/s4 | 1.0000 | 1.0000 | 0.0000 | 1.19x |
| medium/s1 | 0.9933 | 0.9883 | −0.0050 | 1.44x |
| medium/s2 | 0.9917 | 0.9900 | −0.0017 | 1.51x |
| medium/s3 | 0.9933 | 0.9933 | 0.0000 | 1.42x |
| medium/s4 | 0.9983 | 0.9983 | 0.0000 | 1.55x |
| wide/s1 | 0.9383 | 0.9167 | −0.0217 | 1.71x |
| wide/s2 | 0.9650 | 0.9750 | +0.0100 | 1.38x |
| wide/s3 | 0.9775 | 0.8808 | **−0.0967** | 1.61x |
| wide/s4 | 0.9350 | 0.9508 | +0.0158 | 1.46x |

**Speed** mean 1.43x (1.15–1.71x), evaluations roughly halved — close to the
1.71x ceiling once the gradient's own cost is counted.

**Accuracy** mean −0.0091, worse on 5, better on 4, tied on 3. With a standard
error of ~0.009 that mean is not distinguishable from zero: this is not a
systematic bias, it is **added variance**, and the left tail is long — one run
lost 9.7 points.

That is the case against defaulting it on. A 1.4x speedup is not worth a
one-in-twelve chance of losing ten points of accuracy silently.

## Two more things that turned out not to be true

**"The zero subgradient is the problem."** Under `min/max`, a membership
function that is not the arg-min/arg-max anywhere has a gradient of exactly
zero, so L-BFGS-B stops instantly on that block — 11–22% of calls. The obvious
fix is to finite-difference exactly those calls, and it was implemented and
measured: evaluation counts moved, **every final accuracy was byte-identical**,
and it cost 13% of the speedup. So the stall is not the mechanism; the code was
removed rather than kept as plausible-looking dead weight.

What is actually going on is duller: coordinate descent on a non-convex
objective is path-dependent. A different — equally valid — sequence of descent
directions reaches a different local optimum. Nothing is broken.

**"The acceptance guard will catch a bad refinement."** It will not. The guard
compares the refined model against the *heuristic* starting point, which scored
0.005–0.51 on these problems. Every run of both arms was accepted, including the
one that lost 9.7 points. The guard protects against refinement being worse than
not refining; it says nothing about one refinement being worse than another.

## A third thing that turned out not to be true: cProfile lied

Profiling for this evaluation appeared to show two easy wins.
`_cross_entropy_from_strengths` looked like **twice** the cost of the forward
pass it consumes (0.352 s against 0.176 s cumulative), rebuilding `np.arange(n)`
for a fancy index on each of 3 199 calls; and `IncrementalFIS.evaluate_slot`
copied a strided feature column every candidate. Both were fixed — the first
hoisted into a `_CrossEntropy` object, the second by keeping a feature-major
copy of the samples.

A first measurement said 1.12x. **A paired, isolated A/B says it is a wash**:
three runs each way, medians 648 ms before and 652 ms after on
`refine-classifier-wide`, and no change on `refine-classifier`.

The first number was noise, and the profile that motivated the change was
misleading in a specific, repeatable way: cProfile adds a fixed overhead per
*call*, so it systematically overstates small functions invoked thousands of
times, which is exactly the shape of everything in this loop. The lesson for
this repo is that a profile is only a hypothesis; the benchmark's wall clock is
the evidence.

Both changes are kept, but not as optimizations — they are what the gradient
path needs (`with_column_grad` lives on `_CrossEntropy`, and the gradient kernel
wants a contiguous feature column), and they remove per-call allocations that
would matter at a sample count where 32 KB per call is not lost in the noise.
No speedup is claimed for them.

## If you want to turn it on

```python
refine_classifier_antecedents(model, X, y, analytic_gradient=True)
```

It silently does nothing without the compiled kernel, or under `luk`,
`hamacher` or `einstein` (no partials implemented — `supports_gradient()`
reports this). The case where it is unambiguously sound is `probability` norms,
where the objective is genuinely smooth and the gradient exact rather than a
subgradient; the classifier objective currently hard-codes `min/max`, so
reaching that would mean threading a norm choice through
`_make_classifier_fitness` first.

## Harness

`tests/test_analytic_classifier_grad.py` covers correctness. The timing and
paired-accuracy numbers come from the scripts described here; the shapes are
`benchmarks.workloads.make_dataset/make_model` with seeds 1–4 at
`(1000, 8, 3, 2)`, `(2000, 12, 4, 3)` and `(4000, 20, 6, 3)`.
