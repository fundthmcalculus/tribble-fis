"""Antecedent membership-function shape: Gaussian vs trapezoid vs triangular.

Where `t1_it2_gt2_tradeoff` sweeps the *type-2* dimension, this isolates the
**antecedent membership-function method** on the plain Type-1 `TribbleRegressor`:

  gaussian | trap (fast histogram) | trap (EM) | triangular

for a couple of standard regression datasets. It backs the recommendation in
`docs/antecedent-membership-function-evaluation.md` and issues #163 / #165:

  * the **fast histogram** trapezoid is competitive with Gaussian and the
    cheapest to fit -- it builds wide, range-tiling MFs;
  * the **EM** trapezoid/triangular is markedly worse, because its objective
    (max-likelihood of a normalized density) collapses the support onto the
    data mode -- a poor antecedent partition (fixed, but only partially, by the
    `width_reg` knob added in #167).

Run:

    python -m benchmarks.membership_function_antecedent
    python -m benchmarks.membership_function_antecedent -o benchmarks/results/mine.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, make_friedman1
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.gauss_math import tsk_firing_strengths

CONFIGS = [
    ("gaussian", dict(member_function="gaussian")),
    ("trap-fast", dict(member_function="trap", trapz_method="fast")),
    ("trap-em", dict(member_function="trap", trapz_method="em")),
    ("triangular", dict(member_function="triangular")),
]


def datasets():
    Xf, yf = make_friedman1(n_samples=800, n_features=8, noise=1.0, random_state=0)
    Xd, yd = load_diabetes(return_X_y=True)
    return {"friedman1": (Xf, yf), "diabetes": (Xd, yd)}


def run_one(X, y, kw):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    reg = TribbleRegressor(tsk_order="1st", top_p=0.99, l2_reg=0.01,
                           random_state=42, **kw)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        reg.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    pred = reg.predict(Xte)
    # Fraction of test rows where at least one rule fires. Bounded-support MFs
    # (trapezoid/triangular) can drop to ~0 in higher dimensions -- every row
    # lands outside some feature's support -> zero firing -> constant fallback.
    Xdf = pd.DataFrame(Xte, columns=reg.feature_names_in_)
    fs, _ = tsk_firing_strengths(Xdf, reg.model_)
    coverage = float((fs.sum(axis=1) > 1e-9).mean())
    return {
        "coverage": round(coverage, 3),
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": float(r2_score(yte, pred)),
        "n_rules": int(reg.n_rules_),
        "fit_s": round(fit_s, 3),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args(argv)

    records = []
    for name, (X, y) in datasets().items():
        print(f"\n== {name} ({X.shape[0]} samples, {X.shape[1]} features) ==")
        print(f"  {'membership':12s} {'test RMSE':>10s} {'test R2':>9s} {'firing cov':>10s} {'fit s':>7s}")
        for label, kw in CONFIGS:
            r = run_one(X, y, kw)
            r.update(dataset=name, membership=label)
            records.append(r)
            print(f"  {label:12s} {r['rmse']:10.3f} {r['r2']:9.3f} {r['coverage']:10.0%} {r['fit_s']:7.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2))
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
