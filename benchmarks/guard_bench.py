"""Score the refinement acceptance guard as a classifier of its own.

The guard answers "did this refinement beat its starting point?". That question
has a ground truth: run both models on a large test set that neither the search
nor the guard ever sees. So the guard can be scored the way any binary decision
is -- how often does it keep a refinement that is actually worse, and how often
does it throw away one that is actually better?

Run with ``python -m benchmarks.guard_bench``. See issue #65.

A guard that accepts everything has perfect recall and is useless; the current
`legacy` guard is close to that. The interesting column is therefore false
accepts, and the interesting trade is how many genuine improvements a stricter
guard throws away to avoid them.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import (load_breast_cancer, load_digits, load_iris,
                              load_wine, make_classification)
from sklearn.model_selection import train_test_split

from tribblefis import refine as R
from tribblefis.gauss_data import DefaultNormCornorm, NormPair
from tribblefis.gauss_math import create_gaussian_membership_dict, tsk_firing_strengths

NORMS = NormPair(DefaultNormCornorm, DefaultNormCornorm)

# Configurations chosen to span good and bad refinements rather than to be
# sensible. A guard benchmark needs failures to detect, and the honest way to
# get them is settings that genuinely overfit -- no shrinkage with many sweeps,
# a starved validation split -- not synthetic sabotage of the result.
CONFIGS = [
    ("default",        dict()),
    ("no-shrink",      dict(l2_shrink=0.0)),
    ("no-shrink-deep", dict(l2_shrink=0.0, n_sweeps=6)),
    ("tiny-val",       dict(val_fraction=0.10, l2_shrink=0.0, n_sweeps=4)),
    ("heavy-shrink",   dict(l2_shrink=0.5)),
    ("deep",           dict(n_sweeps=6)),
]


def _test_accuracy(model, X, y) -> float:
    fs, labels = tsk_firing_strengths(X, model, norms=NORMS)
    pred = np.array([labels[i] for i in np.argmax(fs, axis=1)], dtype=object)
    return float(np.mean(pred == np.asarray(y, dtype=object)))


def datasets():
    for name, loader in (("iris", load_iris), ("wine", load_wine),
                         ("breast_cancer", load_breast_cancer),
                         ("digits", load_digits)):
        X, y = loader(return_X_y=True)
        yield name, pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), y
    for nf, ninf, sep in ((10, 6, 1.0), (25, 10, 0.8)):
        X, y = make_classification(n_samples=1200, n_features=nf, n_informative=ninf,
                                   n_classes=4, class_sep=sep, random_state=0)
        yield f"hard{nf}f", pd.DataFrame(X, columns=[f"f{i}" for i in range(nf)]), y


def collect(guards, splits=(0, 1, 2), margin=0.0):
    """For every (dataset, split, config), the truth and each guard's verdict.

    `margin` is the accuracy difference below which a refinement is treated as
    a tie rather than a genuine change -- accepting something 0.1 points worse
    is not the failure the guard exists to prevent.
    """
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, X, y in datasets():
            for split in splits:
                pool, test = train_test_split(np.arange(len(y)), test_size=0.3,
                                              random_state=split, stratify=y)
                X_pool, y_pool = X.iloc[pool].reset_index(drop=True), y[pool]
                X_test, y_test = X.iloc[test].reset_index(drop=True), y[test]
                start = create_gaussian_membership_dict(
                    X_pool, pd.Series(y_pool).reset_index(drop=True),
                    top_n_var_names=list(X.columns), n_gaussians=2)
                start_acc = _test_accuracy(start, X_test, y_test)

                for cfg_name, cfg in CONFIGS:
                    for guard in guards:
                        # `guard="none"` always keeps the refinement, so the
                        # model it returns *is* the raw refinement -- which is
                        # also how the ground truth is obtained, for free.
                        out, info = R.refine_classifier_antecedents(
                            start, X_pool, y_pool, norms=NORMS, guard=guard,
                            seed=42, verbose=False, **cfg)
                        rows.append({
                            "dataset": name, "split": split, "config": cfg_name,
                            "guard": guard, "accepted": bool(info["refined"]),
                            "kept_acc": _test_accuracy(out, X_test, y_test),
                            "start_acc": start_acc,
                        })
    return pd.DataFrame(rows)


def score(df: pd.DataFrame, margin: float = 0.0) -> pd.DataFrame:
    """Confusion of each guard against the test-set truth."""
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
    p.add_argument("--margin", type=float, default=0.005,
                   help="accuracy change below this counts as a tie")
    args = p.parse_args(argv)

    guards = list(dict.fromkeys(["none", *args.guards]))  # `none` supplies truth
    df = collect(guards)
    table = score(df, margin=args.margin)
    print(f"\n{len(df) // len(guards)} (dataset x split x config) cases per guard, "
          f"tie margin {args.margin}")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
