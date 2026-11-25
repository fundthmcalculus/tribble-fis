"""Stage A coverage check for split-conformal tree intervals (#268).

Same generator and split as `demo_deconstruct_synthetic.py` (4000 rows, 25%
held-out test), repeated over several seeds. For each estimator and target
miscoverage ``alpha``, reports empirical test coverage of
`predict_intervals()`, mean interval width (absolute and relative to the test
target's range) and test R^2 of `predict`.

Acceptance for #268: mean coverage within +/-0.03 of ``1 - alpha``.

Run:
    uv run python tribble-tree/demo_conformal_intervals.py
"""

import os
import sys
import warnings

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from demo_deconstruct_synthetic import TOPOLOGY, make_data  # noqa: E402
from fuzzytree import DeconstructedHierarchicalRegressor, HierarchicalFuzzyExpertsRegressor  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.1, 0.2)
FLAT = {"n_output_buckets": 5, "tsk_order": "1st", "top_n": -1, "random_state": 42}

ARMS = {
    "Deconstructed (true topology)": lambda a, s: (
        DeconstructedHierarchicalRegressor(
            flat_regressor_kwargs=FLAT, conformal_calibration=True,
            conformal_alpha=a, conformal_random_state=s,
        ), TOPOLOGY),
    "Deconstructed (topology=\"auto\")": lambda a, s: (
        DeconstructedHierarchicalRegressor(
            flat_regressor_kwargs=FLAT, conformal_calibration=True,
            conformal_alpha=a, conformal_random_state=s,
        ), "auto"),
    "HME (gaussian gates)": lambda a, s: (
        HierarchicalFuzzyExpertsRegressor(
            gate_style="gaussian", conformal_calibration=True,
            conformal_alpha=a, random_state=s,
        ), None),
}


def main():
    print("| estimator | alpha | target | coverage (mean, min-max) | width | width / range | R^2 |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    for name, make in ARMS.items():
        for alpha in ALPHAS:
            cov, width, rel, r2 = [], [], [], []
            for seed in SEEDS:
                X, y, _ = make_data(seed=seed)
                i_tr, i_te = train_test_split(np.arange(len(X)), test_size=0.25, random_state=seed)
                X_tr, X_te = X.iloc[i_tr].reset_index(drop=True), X.iloc[i_te].reset_index(drop=True)
                y_tr, y_te = y[i_tr], y[i_te]
                model, topo = make(alpha, seed)
                model.fit(X_tr, y_tr, topo) if topo is not None else model.fit(X_tr, y_tr)
                lo, hi = model.predict_intervals(X_te)
                cov.append(np.mean((y_te >= lo) & (y_te <= hi)))
                width.append(np.mean(hi - lo))
                rel.append(np.mean(hi - lo) / np.ptp(y_te))
                r2.append(r2_score(y_te, model.predict(X_te)))
            print(f"| {name} | {alpha} | {1 - alpha:.2f} | {np.mean(cov):.3f} "
                  f"({min(cov):.3f}-{max(cov):.3f}) | {np.mean(width):.3f} | "
                  f"{np.mean(rel):.3f} | {np.mean(r2):.3f} |", flush=True)


if __name__ == "__main__":
    main()
