"""Score the *Ruspini* refinement guard the same way the classifier one was.

`refine_ruspini_partition` carries a copy of the acceptance guard that
`refine_classifier_antecedents` just shed (issue #65, PR #66). Whether the same
answer applies is genuinely open rather than obvious: the Ruspini search is much
lower-dimensional -- one apex knot per term, against two parameters per
membership function -- and it optimises a *shared* partition where every class
rule reads the same knots. Fewer parameters and a coupled representation could
plausibly mean less to overfit, a lower base rate of harmful refinements, and a
guard that earns its keep. Or the opposite: coupling means one bad knot moves
every rule at once.

So: measure, do not extrapolate. Same protocol as `guard_bench.py` -- ground
truth from a test set neither the search nor the guard ever sees.

Run with ``python -m benchmarks.ruspini_guard_bench``.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import (load_breast_cancer, load_iris, load_wine,
                              make_classification)
from sklearn.model_selection import train_test_split

from tribblefis import refine as R
from tribblefis.gaussian_classifier import TribbleClassifier
from tribblefis.ruspini import ruspinize_model

# A narrower spread than the classifier benchmark, on purpose. Ruspini's
# coordinate step is a *grid line search* -- `max(5, sub_maxfun)` = 25 objective
# evaluations per knot per sweep -- so a single refinement costs orders of
# magnitude more than the classifier's L-BFGS-B block. The full 6x6x5 matrix did
# not finish a single dataset in 15 minutes. These four configurations still
# span sensible to reckless, which is what the benchmark needs; the dropped two
# were near-duplicates on the same axis.
CONFIGS = [
    ("default",      dict()),
    ("no-shrink",    dict(l2_shrink=0.0)),
    ("tiny-val",     dict(val_fraction=0.10, l2_shrink=0.0)),
    ("heavy-shrink", dict(l2_shrink=0.3)),
]


def _accuracy(rmodel, X, y) -> float:
    proba, labels = rmodel.class_proba(X)
    pred = np.array([labels[i] for i in np.argmax(proba, axis=1)], dtype=object)
    return float(np.mean(pred == np.asarray(y, dtype=object)))


def datasets():
    # digits (64 features x 10 classes) and breast_cancer (30 features) are left
    # out for cost: the knot vector scales with features and the grid line
    # search scales with the knot vector, so those two alone dominated the run
    # without adding a regime the `hard*` sets do not already cover.
    for name, loader in (("iris", load_iris), ("wine", load_wine)):
        X, y = loader(return_X_y=True)
        yield name, pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), y
    for nf, ninf, sep in ((8, 5, 1.0), (12, 7, 0.8)):
        X, y = make_classification(n_samples=600, n_features=nf, n_informative=ninf,
                                   n_classes=3, class_sep=sep, random_state=0)
        yield f"hard{nf}f", pd.DataFrame(X, columns=[f"f{i}" for i in range(nf)]), y


def _starting_partition(X_pool, y_pool):
    """The Ruspini model a `RuspiniFuzzyClassifier` would start refinement from."""
    base = TribbleClassifier(n_gaussians=2, random_state=42)
    base.fit(X_pool, y_pool)
    feats = [f for f in base.top_features_ if f in X_pool.columns]
    return ruspinize_model(base.model_, X_pool[feats]), feats


def collect(guards, splits=(0, 1, 2)):
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, X, y in datasets():
            for split in splits:
                pool, test = train_test_split(np.arange(len(y)), test_size=0.3,
                                              random_state=split, stratify=y)
                X_pool, y_pool = X.iloc[pool].reset_index(drop=True), y[pool]
                X_test, y_test = X.iloc[test].reset_index(drop=True), y[test]
                start, feats = _starting_partition(X_pool, y_pool)
                X_pool_f = X_pool[feats]
                X_test_f = X_test[feats]
                start_acc = _accuracy(start, X_test_f, y_test)

                for cfg_name, cfg in CONFIGS:
                    for guard in guards:
                        out, info = R.refine_ruspini_partition(
                            start, X_pool_f, y_pool, guard=guard, seed=42,
                            verbose=False, **cfg)
                        rows.append({
                            "dataset": name, "split": split, "config": cfg_name,
                            "guard": guard, "accepted": bool(info["refined"]),
                            "kept_acc": _accuracy(out, X_test_f, y_test),
                            "start_acc": start_acc,
                        })
                print(f"  done {name}/split{split}", flush=True)
    return pd.DataFrame(rows)


def score(df: pd.DataFrame, margin: float = 0.005) -> pd.DataFrame:
    truth = df[df.guard == "none"].set_index(["dataset", "split", "config"])["kept_acc"]
    out = []
    for guard, g in df.groupby("guard"):
        key = list(zip(g.dataset, g.split, g.config))
        refined_acc = truth.loc[key].to_numpy()
        start_acc = g.start_acc.to_numpy()
        accepted = g.accepted.to_numpy()
        kept = g.kept_acc.to_numpy()
        worse = refined_acc < start_acc - margin
        better = refined_acc > start_acc + margin
        out.append({
            "guard": guard,
            "n": len(g),
            "accept rate": accepted.mean(),
            "false accepts": int(np.sum(accepted & worse)),
            "of possible": int(np.sum(worse)),
            "false rejects": int(np.sum(~accepted & better)),
            "of possible.": int(np.sum(better)),
            "mean kept acc": kept.mean(),
            "worst kept acc": kept.min(),
            "vs never refining": kept.mean() - start_acc.mean(),
        })
    return pd.DataFrame(out).set_index("guard")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--guards", nargs="+", default=list(R.GUARDS))
    p.add_argument("--margin", type=float, default=0.005)
    args = p.parse_args(argv)

    guards = list(dict.fromkeys(["none", *args.guards]))
    df = collect(guards)
    print(f"\n{len(df) // len(guards)} (dataset x split x config) cases per guard, "
          f"tie margin {args.margin}")
    print(score(df, margin=args.margin).to_string(float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
