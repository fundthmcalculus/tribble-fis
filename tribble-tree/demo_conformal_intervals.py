"""Stage A coverage check for split-conformal tree intervals (#268).

Same generator and split as `demo_deconstruct_synthetic.py` (4000 rows, 25%
held-out test), repeated over several seeds. For each estimator and target
miscoverage ``alpha``, reports empirical test coverage of
`predict_intervals()`, mean interval width (absolute and relative to the test
target's range) and test R^2 of `predict`.

Acceptance for #268: mean coverage within +/-0.03 of ``1 - alpha``.

``--adaptive`` instead compares HME's ``conformal_mode="constant"`` and
``"adaptive"`` on the Stage A signal with heteroscedastic noise (sd grows
linearly with feature ``a``, 0.05 -> 2.55), reporting marginal coverage plus
coverage and width in the lowest/highest-noise thirds and the Spearman
correlation of width with |residual| -- the conditional behaviour a constant
margin cannot have.

Run:
    uv run python tribble-tree/demo_conformal_intervals.py
    uv run python tribble-tree/demo_conformal_intervals.py --adaptive
"""

import os
import sys
import warnings

import numpy as np
from scipy.stats import spearmanr
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


def make_heteroscedastic(n=4000, seed=0, noise_feature="a"):
    X, _, groups = make_data(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    signal = 1.0 * groups["G1"] + 0.7 * groups["G2"] + 1.4 * groups["G3"]
    noise_sd = 0.05 + 0.25 * X[noise_feature].to_numpy()
    return X, signal.to_numpy() + rng.normal(0, noise_sd), noise_sd


def adaptive_main(alpha=0.1):
    print(f"HME, heteroscedastic Stage A (noise sd 0.05 -> 2.55 along one feature), "
          f"target {1 - alpha:.2f}, {len(SEEDS)} seeds\n")
    print("| noise along | mode | coverage | cov. low-noise third | cov. high-noise third | width low | width high "
          "| mean width | rho(width, abs resid) | R^2 |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for noise_feature, mode in [(f, m) for f in ("a", "e") for m in ("constant", "adaptive")]:
        rows = []
        for seed in SEEDS:
            X, y, sd = make_heteroscedastic(seed=seed, noise_feature=noise_feature)
            i_tr, i_te = train_test_split(np.arange(len(X)), test_size=0.25, random_state=seed)
            X_tr, X_te = X.iloc[i_tr].reset_index(drop=True), X.iloc[i_te].reset_index(drop=True)
            y_tr, y_te, sd_te = y[i_tr], y[i_te], sd[i_te]
            model = HierarchicalFuzzyExpertsRegressor(
                gate_style="gaussian", conformal_calibration=True, conformal_alpha=alpha,
                conformal_mode=mode, random_state=seed,
            ).fit(X_tr, y_tr)
            pred = model.predict(X_te)
            lo, hi = model.predict_intervals(X_te)
            inside, width = (y_te >= lo) & (y_te <= hi), hi - lo
            low, high = sd_te <= np.quantile(sd_te, 1 / 3), sd_te >= np.quantile(sd_te, 2 / 3)
            rho = spearmanr(width, np.abs(y_te - pred)).statistic
            rows.append([inside.mean(), inside[low].mean(), inside[high].mean(),
                         width[low].mean(), width[high].mean(), width.mean(),
                         0.0 if np.isnan(rho) else rho, r2_score(y_te, pred)])
        m = np.mean(rows, axis=0)
        print(f"| `{noise_feature}` | {mode} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.2f} | {m[4]:.2f} | "
              f"{m[5]:.2f} | {m[6]:+.3f} | {m[7]:.3f} |", flush=True)


if __name__ == "__main__":
    if "--adaptive" in sys.argv:
        adaptive_main()
    else:
        main()
