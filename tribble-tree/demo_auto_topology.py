"""Stage A, re-run against the auto-topology path (#226).

`DECONSTRUCTED_TREE_FINDINGS.md`'s Stage A compares a flat regressor, HME, and
the deconstruction given the **true** topology. #226 asks for the same protocol
applied to a derived topology, so this adds those arms to the same generator and
the same split.

It also answers a question Stage A cannot answer about itself, and the answer
changes how the whole comparison should be read -- see the "identifiability"
note printed at the end.

Run:
    uv run python tribble-tree/demo_auto_topology.py
    uv run python tribble-tree/demo_auto_topology.py --seeds 5
"""

import argparse
import contextlib
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from fuzzytree import (  # noqa: E402
    DeconstructedHierarchicalRegressor,
    affinity_topology,
    feature_affinity,
    per_feature_topology,
)
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402

# Verbatim from demo_deconstruct_synthetic.py, so these numbers sit beside the
# ones already in DECONSTRUCTED_TREE_FINDINGS.md rather than beside a variant.
TOPOLOGY = {"TOTAL": ["G1", "G2", "G3"], "G1": ["a", "b"], "G2": ["c", "d"], "G3": ["e", "f"]}
TRUE_WEIGHTS = {"G1": 1.0, "G2": 0.7, "G3": 1.4}
FLAT_KWARGS = {"n_gaussians": 3}


def make_data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({name: rng.uniform(0, 10, n) for name in list("abcdef")})
    g1 = np.sin(X["a"] / 2) * 5 + 0.5 * X["b"]
    g2 = np.cos(X["c"] / 3) * 4 - 0.3 * X["d"]
    g3 = np.tanh((X["e"] - 5) / 2) * 6 + 0.2 * X["f"]
    y = sum(TRUE_WEIGHTS[k] * v for k, v in {"G1": g1, "G2": g2, "G3": g3}.items())
    return X, (y + rng.normal(0, 0.3, n)).to_numpy()


def _quiet():
    """`TribbleRegressor.fit` prints a feature-ranking table with no off switch,
    and this fits one per arm per seed."""
    return contextlib.redirect_stdout(io.StringIO())


def _score(model_factory, X_tr, y_tr, X_te, y_te, **fit_kwargs):
    with _quiet():
        model = model_factory()
        model.fit(X_tr, y_tr, **fit_kwargs)
        predictions = model.predict(X_te)
    return float(r2_score(y_te, predictions)), model


def run_seed(seed: int, n: int):
    X, y = make_data(n=n, seed=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    X_tr = X_tr.reset_index(drop=True)
    X_te = X_te.reset_index(drop=True)

    def deconstructed():
        return DeconstructedHierarchicalRegressor(flat_regressor_kwargs=FLAT_KWARGS)

    results = {}

    with _quiet():
        flat = TribbleRegressor(**FLAT_KWARGS)
        flat.fit(X_tr, pd.Series(y_tr))
        results["flat TribbleRegressor"] = float(r2_score(y_te, flat.predict(X_te)))

    results["deconstructed, TRUE topology"] = _score(
        deconstructed, X_tr, y_tr, X_te, y_te, topology=TOPOLOGY
    )[0]
    results["deconstructed, per_feature (floor)"] = _score(
        deconstructed, X_tr, y_tr, X_te, y_te, topology=per_feature_topology(X_tr)
    )[0]
    for k in (2, 3, 4):
        results[f"deconstructed, affinity_k{k}"] = _score(
            deconstructed, X_tr, y_tr, X_te, y_te, topology=affinity_topology(X_tr, k)
        )[0]

    score, model = _score(deconstructed, X_tr, y_tr, X_te, y_te, topology="auto")
    results["deconstructed, auto (selected)"] = score
    return results, model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=4000)
    args = parser.parse_args(argv)

    per_seed, chosen = [], []
    for seed in range(args.seeds):
        results, model = run_seed(seed, args.samples)
        per_seed.append(results)
        chosen.append(model.topology_name_)

    names = list(per_seed[0])
    width = max(len(n) for n in names)
    print(f"Stage A generator, N={args.samples}, 75/25 split, {args.seeds} seeds, "
          f"TribbleRegressor(n_gaussians=3)\n")
    print(f"{'arm'.ljust(width)}   test R^2")
    print("-" * (width + 20))
    for name in names:
        values = [r[name] for r in per_seed]
        print(f"{name.ljust(width)}   {np.mean(values):.3f} +/- {np.std(values):.3f}")

    print(f"\nauto picked: {chosen}")

    X, _ = make_data(n=args.samples, seed=0)
    off_diagonal = feature_affinity(X).to_numpy()[~np.eye(X.shape[1], dtype=bool)]
    print(f"\nmax off-diagonal |corr| among the six features: {off_diagonal.max():.4f}")
    print(
        "\nIdentifiability note. This generator draws every feature as an\n"
        "independent uniform, and its group functions are additively separable\n"
        "in their own members (g1 = 5*sin(a/2) + 0.5*b, and so on). So the whole\n"
        "of y is additively separable across all six features, and the 'true'\n"
        "topology G1=[a,b] G2=[c,d] G3=[e,f] is a labelling convention from how\n"
        "the generator was written -- it leaves no trace in the data. No\n"
        "topology-derivation strategy can recover it, and none should be judged\n"
        "on whether it does. Judge the arms on held-out R^2 alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
