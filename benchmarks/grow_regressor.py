"""Cost of `TribbleRegressor.fit_growing` against the refit sweep it replaces (#248).

#248 guessed the consequent solve would dominate, because adding a feature rescales every row's firing strength
and widens every rule's design block, so the ridge normal equations are re-solved at every step. This measures it
instead: a synthetic regression (24 features, 8 informative), 1st-order consequents, two comparisons --

  1. to a reachable R2 target, the sweep stopping at the SAME size (the realistic comparison);
  2. walking all 24 sizes (an unreachable target), the worst case for both.

Both arms use the same rows and ``correlation_threshold=0`` so ``top_n=k`` is a prefix of the same ranking; the
script asserts they agree on every held-out score and on the stopping size before it reports a time. Timings are
host-specific: ratios transfer, seconds do not.

    python -m benchmarks.grow_regressor
"""

from __future__ import annotations

import contextlib
import io
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from tribblefis.gaussian_regressor import TribbleRegressor


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def main(target: float = 0.90) -> None:
    X, y = make_regression(n_samples=3000, n_features=24, n_informative=8, noise=10.0, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(24)])
    y = pd.Series(y)
    Xtr, Xva, ytr, yva = (a.reset_index(drop=True) for a in train_test_split(X, y, test_size=0.25, random_state=0))
    kw = dict(correlation_threshold=0.0, tsk_order="1st", n_output_buckets=3, random_state=42)

    def sweep(stop_at_target: bool):
        scores, t = [], time.perf_counter()
        for k in range(1, 25):
            m = TribbleRegressor(top_n=k, **kw)
            _quiet(m.fit, Xtr, ytr)
            scores.append(r2_score(yva, m.predict(Xva)))
            if stop_at_target and scores[-1] >= target:
                break
        return scores, time.perf_counter() - t

    def grow(t_target: float):
        g = TribbleRegressor(**kw)
        t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _quiet(g.fit_growing, Xtr, ytr, target=t_target, validation=(Xva, yva), refit=False)
        return g, time.perf_counter() - t

    print(f"{'comparison':32s} {'size':>5s} {'growing':>9s} {'refit sweep':>12s} {'ratio':>7s}")
    sweep_scores, sweep_s = sweep(True)
    g, grow_s = grow(target)
    assert g.target_reached_ and g.n_features_selected_ == len(sweep_scores), "arms disagree on the stopping size"
    assert np.allclose(g.growth_history_["score"], sweep_scores, atol=1e-9), "arms disagree on the held-out scores"
    print(f"{f'to target R2 >= {target}':32s} {g.n_features_selected_:>5d} {grow_s:8.2f}s {sweep_s:11.2f}s {sweep_s / grow_s:6.1f}x")

    full_scores, full_s = sweep(False)
    g, grow_s = grow(1.0)
    assert np.allclose(g.growth_history_["score"], full_scores, atol=1e-9)
    step = g.growth_history_["step_seconds"]
    print(f"{'walking all 24 sizes':32s} {24:>5d} {grow_s:8.2f}s {full_s:11.2f}s {full_s / grow_s:6.1f}x")
    print(f"per-step seconds, first / last: {step.iloc[0]:.3f} / {step.iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
