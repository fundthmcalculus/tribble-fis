# How should IT2/GT2 be accelerated: GPU forward pass, or something else?

> **Status: design decided, implementation not started.** Issue #146.
> Backed by `benchmarks/it2_incremental_probe.py`.

**Answer: not a GPU forward pass first.** Port IT2/GT2 *refinement* onto the
existing `kernel.IncrementalFIS`, the same CPU path Type-1 coordinate
refinement already uses. Measured: **16-56x per refinement candidate**, no
new dependency, no device transfer, and it attacks the cost #143/#213
actually found (refinement is 85-1388x the unrefined fit; the forward pass
is sub-second). GPU stays a later, opt-in `backend=` knob that rides on the
same `CompiledFIS` objects.

## Two premises in #146 were stale

Checked against `main`:

| #146 said | Actually |
|---|---|
| IT2/GT2 call "plain numpy" `tsk_firing_strengths`, bypassing `kernel.py` | `gauss_math.tsk_firing_strengths` already dispatches to `kernel.firing_strengths` whenever the Cython kernel is built and the model compiles (`gauss_math.py`, "Fast path" block). IT2's upper/lower models are ordinary all-Gaussian `GaussianMixtureModel`s, so IT2/GT2 forward passes **already get the compiled CPU kernel** on the same terms as Type-1. |
| `TribbleClassifier`/`TribbleRegressor` use the Torch backend | No estimator uses it. `backend="torch"` is reachable only by calling `kernel.firing_strengths` directly; `auto` never selects it (CUDA `exp` drift). There is no Type-1 estimator GPU pattern to mirror. |

So "IT2 GPU parity with Type-1" is already true at the estimator level --
neither has one. The real gap is elsewhere.

## Where the time goes

`it2_refine` / `gt2_refine` evaluate each candidate by rebuilding the
upper/lower (per alpha-plane, for GT2) `GaussianMixtureModel` and running
two **full** forward passes (`it2_refine._it2_rule_firing`). Type-1
refinement stopped doing that long ago: it moves one slot and asks
`IncrementalFIS.evaluate_slot` for only the affected cell and label fold,
`O(n·(K+F))` instead of `O(n·F·K·L)`, bit-identical.

Per-candidate cost of moving one IT2 slot (upper + lower bounds):

| shape | Cython | rebuild (today) | full pass on persistent `CompiledFIS` | `IncrementalFIS` ×2 | speedup |
|---|---|--:|--:|--:|--:|
| n=5k, F=10, L=6, K=3 | no | 7.96 ms | 11.22 ms | 0.50 ms | **15.8x** |
| n=5k, F=10, L=6, K=3 | yes | 3.71 ms | 2.97 ms | 0.10 ms | **36.4x** |
| n=20k, F=20, L=8, K=3 | no | 73.60 ms | 113.18 ms | 3.16 ms | **23.3x** |
| n=20k, F=20, L=8, K=3 | yes | 37.45 ms | 35.53 ms | 0.67 ms | **56.0x** |

Notes:

- The full-pass column shows the forward pass is already at its floor
  (`kernel.py`'s docstring: ~93% `np.exp`). Faster full passes -- which is
  all a GPU offers here -- buy ≤1.3x on CPU; the incremental path buys 16-56x.
- GT2 multiplies every column by `n_alpha_planes`; all planes share one
  layout, so the ratio is unchanged.
- This is the firing-strength share of a candidate. Regression candidates
  also pay a closed-form consequent re-solve and a Karnik-Mendel search;
  those become the new floor and should be profiled after the port.

## Design

1. **`IT2IncrementalFIS`** (new, small, in `it2_kernel.py` or `kernel.py`):
   holds two `IncrementalFIS` -- upper and lower -- compiled from
   `_extract_upper_model` / `_extract_lower_model` with one shared
   `feature_matrix`. An IT2 slot `(mu, sigma_lower, sigma_upper)` maps to the
   same slot index in both (identical layout, by construction).
   `evaluate_slot(slot, mu, s_lo, s_hi) -> (f_upper, f_lower)`, `commit()`.
   Non-Gaussian or non-compilable models fall back to today's rebuild path,
   exactly as `kernel.can_compile` already arranges for Type-1.
2. **`it2_refine`**: replace the per-candidate rebuild in the classifier and
   regressor fitness closures with `IT2IncrementalFIS.evaluate_slot`. The
   coordinate-descent loop, L2 anchor, never-worse guard and CV folds stay
   as they are (one `IT2IncrementalFIS` per fold).
3. **GT2**: `GT2IncrementalFIS` = `n_alpha_planes` `IT2IncrementalFIS`, one
   per plane. A GT2 slot's 4 parameters map to each plane's
   `alpha_cut` interval, so one candidate = `2K` `evaluate_slot` calls.
   No change to `gt2_kernel`'s combination math.
4. **Bit-exactness test**: `IT2IncrementalFIS.evaluate_slot` must equal the
   rebuild path to the same tolerance `test_incremental_kernel.py` already
   uses for Type-1, for every norm family.
5. **GPU, later and opt-in**: add a `kernel_backend="auto"` estimator
   parameter to T1/IT2/GT2 together (not IT2 alone), passed down to
   `kernel.firing_strengths`. GT2's `2K` same-layout planes are a natural
   `TorchFIS.firing_strengths_batch` call (one upload, `P = 2K` parameter
   sets). Gate it on a measured workload where a single forward pass is
   large enough for a device dispatch to beat the Cython kernel -- the
   evidence #146's own comment asked for and #143 did not produce.

## Recommendation

- Retitle/re-scope #146 to "IT2/GT2 refinement on `IncrementalFIS`" and do
  steps 1-4; that is the measured win.
- Track step 5 (GPU) separately and keep it deprioritised until a workload
  shows the forward pass, not refinement, dominating.

## Reproducing

```bash
python -m benchmarks.it2_incremental_probe
python -m benchmarks.it2_incremental_probe --n 20000 --features 20 --labels 8
```

The Cython rows need the optional kernel (`python setup_cython.py build_ext
--inplace`); without it the probe reports `cython=False` and measures the
NumPy path.
