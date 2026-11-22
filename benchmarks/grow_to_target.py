"""Cost of `TribbleClassifier.fit_growing` against the refit sweep it replaces (#242).

Choosing ``top_n`` without growing means refitting at ``top_n = 1, 2, 3, ...`` and reading off where
quality is good enough; every refit re-scores every feature and refits every membership. Growing ranks
once and fits each feature's memberships once. Two comparisons on a synthetic 3-class problem
(30 features, 10 informative; held-out macro F1 is non-monotone in the number of features):

  1. to a reachable target -- the sweep is allowed the SAME stopping size (it also stops at the first
     size meeting the target), so this is the fair, realistic comparison;
  2. walking every size -- the whole curve (an unreachable target), the worst case for both.

Both arms use the same rows and ``correlation_threshold=0`` so ``top_n=k`` is a prefix of the same
ranking, which is also what makes their held-out scores comparable (the script asserts they agree).
Timings are host-specific: the ratios transfer, the seconds do not.

    python -m benchmarks.grow_to_target
"""

from __future__ import annotations

import contextlib
import io
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from tribblefis.gaussian_classifier import TribbleClassifier


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):  # the ranking prints to stdout
        return fn(*a, **k)


def main(target: float = 0.80) -> None:
    X, y = make_classification(
        n_samples=4000, n_features=30, n_informative=10, n_redundant=0, n_classes=3,
        n_clusters_per_class=1, class_sep=1.2, random_state=0,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(30)])
    y = pd.Series(y)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    Xtr, Xva, ytr, yva = (a.reset_index(drop=True) for a in (Xtr, Xva, ytr, yva))
    f1 = lambda a, b: f1_score(a, b, average="macro", zero_division=0)  # noqa: E731

    def sweep(stop_at_target: bool):
        scores, t = [], time.perf_counter()
        for k in range(1, 31):
            m = TribbleClassifier(top_n=k, correlation_threshold=0.0, random_state=42)
            _quiet(m.fit, Xtr, ytr)
            scores.append(f1(yva, m.predict(Xva)))
            if stop_at_target and scores[-1] >= target:
                break
        return scores, time.perf_counter() - t

    def grow(t_target: float):
        g = TribbleClassifier(correlation_threshold=0.0, random_state=42)
        t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _quiet(g.fit_growing, Xtr, ytr, target=t_target, validation=(Xva, yva), refit=False)
        return g, time.perf_counter() - t

    print(f"{'comparison':32s} {'size':>5s} {'growing':>9s} {'refit sweep':>12s} {'ratio':>7s}")
    sweep_scores, sweep_s = sweep(stop_at_target=True)
    g, grow_s = grow(target)
    assert g.n_features_selected_ == len(sweep_scores) and g.target_reached_, "arms disagree on the stopping size"
    assert np.allclose(g.growth_history_["score"], sweep_scores, atol=1e-9), "arms disagree on the held-out scores"
    print(f"{f'to target F1 >= {target}':32s} {g.n_features_selected_:>5d} {grow_s:8.2f}s {sweep_s:11.2f}s {sweep_s / grow_s:6.1f}x")

    full_scores, full_s = sweep(stop_at_target=False)
    g, grow_s = grow(1.0)
    assert np.allclose(g.growth_history_["score"], full_scores, atol=1e-9)
    print(f"{'walking all 30 sizes':32s} {30:>5d} {grow_s:8.2f}s {full_s:11.2f}s {full_s / grow_s:6.1f}x")


if __name__ == "__main__":
    main()
