# Does histogram binning cost the trapezoid/triangle EM real resolution?

**Answer: no, not for the case this code actually fits.** Histogram bin count
is not the bottleneck for well-separated, roughly-unimodal-per-label
components -- 50 bins (the existing default) already recovers them about as
well as 1000 does, at a fraction of the cost. The bottleneck, where one
exists, is the peak-detection *initialization*, and a naive unbinned ("raw
sample") EM does not fix that -- it makes it markedly worse, because the
existing peak-finder implicitly assumes evenly-spaced bins.

This came out of designing histogram-based EM for triangular membership
functions -- fitting the triangle directly as a trapezoid whose plateau has
collapsed to a single apex point, which is why the fitting code lives in
`trapz_math.py` itself (via a `shape="trapezoid"|"triangle"` argument) rather
than a separate module -- see that module's docstring. Before committing to
histogram-based EM for the triangle case at all, three questions needed
answers:

1. Does the 50-bin default under-resolve components once they get narrow?
2. Does increasing `n_bins` reliably fix that?
3. Would fitting directly against raw, unbinned samples ("the traditional
   approach") do better?

## Method

`trapz_math.py`'s actual EM internals (`_init_trapz_from_histogram`,
`_em_e_step`, `_em_m_step_weights`, `_em_m_step_params`,
`_trapz_log_likelihood`) were reused unchanged, swapping only what feeds them:
either a real histogram (`np.histogram` over `n_bins`), or "raw" mode --
`bin_centers` set to the sorted raw observations with `bin_counts` all ones.
Both paths run through the same E-step/M-step/log-likelihood code, so this
isolates the binning effect from everything else. Three synthetic cases:

1. **Wide + narrow, width ratio ~1:66.** A uniform component over `[-10, 10]`
   plus a narrow uniform bump over `[4.85, 5.15]` (width 0.3).
2. **Wide + narrow, width ratio ~1:400.** Same wide component, bump narrowed
   to `[4.975, 5.025]` (width 0.05).
3. **Three well-separated narrow clusters.** `Normal(0, 0.05)`,
   `Normal(0.5, 0.05)`, `Normal(1.0, 0.05)`, 800 points each -- the realistic
   shape for this code's actual use (per-label marginals that are genuinely
   separated, each roughly unimodal).

## Results

| case | 50 bins | 200 bins | 1000 bins | raw (unbinned) |
|---|---|---|---|---|
| wide + narrow (1:66) | 1 component found (collapsed) | 1 component found (collapsed) | 2 components, but split at the wrong point | 1 component found (collapsed) |
| wide + narrow (1:400) | 1 component found (collapsed) | 1 component found (collapsed) | 1 component found (collapsed) | 1 component found (collapsed) |
| three narrow clusters | 3 components, centered near 0/0.5/1.0 | same, ~6x slower | same, ~1x slower than 50 bins (non-monotonic cost) | **1** component -- collapsed onto the entire range, log-likelihood -640 vs +1000 for every binned setting |

Full timings for case 3 (`n_components=3`, three seeds' worth of noise not
separately tracked -- single run, but the qualitative pattern reproduces):

| bins | time | log-likelihood |
|---:|---:|---:|
| 50 | 142 ms | 1001.9 |
| 200 | 808 ms | 976.0 |
| 1000 | 130 ms | 978.8 |
| raw | 3.5 ms | **-639.8** |

## Reading the results

**For the case this code is meant to fit** (case 3: separated, roughly
unimodal components -- the realistic shape of a per-(feature, label)
marginal), 50 bins already resolves the structure correctly, and more bins
buy essentially nothing: log-likelihood *drops slightly* at 200 and 1000 bins
relative to 50 (more bins means finer per-bin noise for the SLSQP fit to chase
without more real signal), while cost rises 1-6x. There is no accuracy
argument for a bigger default here.

**For the pathological case** (a narrow component embedded inside a much
wider, nearly flat one), no bin count -- 50, 200, or 1000 -- reliably
recovers it. The failure is in `_init_trapz_from_histogram`'s peak detection:
`scipy.signal.find_peaks` with a fixed relative-prominence threshold
(`prominence=max(smoothed) * 0.05`) does not register a small bump sitting on
an approximately flat density as a peak, regardless of how finely that
density is binned. At 1000 bins the search does occasionally find "2 peaks",
but at an arbitrary split point unrelated to the true structure -- worse, not
better, than just not trying. This is an initialization-heuristic limitation,
orthogonal to binning resolution.

**Raw (unbinned) mode is not a fix -- it is worse everywhere it was tried,**
including on the *easy* case. The reason is mechanical, not conceptual:
`_init_trapz_from_histogram`'s smoothing (`gaussian_filter1d(sigma=1.0)`) and
half-power-width search walk the array by *index*, which is only meaningful
if consecutive entries are evenly spaced in x. Raw sorted samples are not
evenly spaced -- gaps between adjacent points vary with local density -- so
"one bin of smoothing" means a wildly different x-window at every point in
the array, and the half-power search silently measures the wrong widths. On
the three-cluster case (the case that binned methods solve cleanly at every
bin count tried) raw mode collapsed to a single component spanning the entire
range, with a log-likelihood roughly 1600 nats worse than any binned setting.
This is not "less resolution" -- it is a broken initialization, because
nothing here was written to assume irregular spacing.

A genuinely more "traditional" per-sample EM would need its own
initialization from scratch (e.g. k-means seeding on the raw values, the same
idea the Gaussian mixture code moved *to* in #72 -- see
`docs/identification-cost-evaluation.md`) rather than removing the histogram
and expecting the existing peak-finder to still work unbinned. That is a
larger, separate change, and there is no accuracy evidence from this
evaluation that it is even the right direction: nothing here shows that
sample-level resolution is the thing holding back real fits, only that the
current *toy* substitution of raw data into a bins-shaped algorithm breaks it.

## Decision

The triangle case uses the same histogram-based EM design as the trapezoid
case (`n_bins=50` default, unchanged) -- literally the same engine in
`trapz_math.py`, just with `shape="triangle"` optimizing one fewer parameter
per component in the M-step. No raw/unbinned mode was added. If a future
case turns up where separated components are genuinely too narrow relative
to the data range for the default to resolve, the fix indicated by this
evaluation is a better *initialization* (e.g. k-means seeding, matching the
Gaussian mixture's own precedent), not a larger or absent histogram -- and
that is future work, not something this change attempts.
