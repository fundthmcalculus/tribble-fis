"""Measure the second-stage admissibility reduction (negated cross-terms).

A label's rule ANDs one per-feature disjunction per feature, so it admits the
whole outer product of its terms and cannot condition on *which* term fired.
:mod:`tribblefis.exclusion` mines the cells of that product the data assigns to
another class and appends their negation to the offending rule. The question
this benchmark answers is whether that helps on real data, and by how much.

Run with ``python -m benchmarks.exclusion_bench``.

Two things are reported, because they are different claims:

``synthetic``
    Problems built so that the outer product is provably the binding
    constraint -- checkerboards, where both classes have identical marginals.
    This is an existence proof: it shows the correction does what it claims when
    the defect it targets is what is actually wrong.

``real``
    Stock scikit-learn datasets, held-out accuracy, several splits. This is the
    honest question: does a defect the representation *can* have show up often
    enough, and cleanly enough, to be worth mining for by default?

Only models with more than one membership function per feature-label have an
outer product at all, so every configuration here sets ``n_gaussians >= 2``.
With one Gaussian per feature-label the rule *is* a single cell and there is
nothing to reduce -- mining correctly returns nothing, which is why
``n_gaussians=1`` is reported separately rather than folded into the mean.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.model_selection import train_test_split

from tribblefis.exclusion import describe_rules
from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier

SPLITS = (0, 1, 2)


def checkerboard(n_per_cell=120, seed=0, spread=0.55, blobs=2):
    """``blobs x blobs`` alternating cells. Both classes share every marginal,
    so no per-feature rule can separate them and the cross-term is the whole
    problem."""
    rng = np.random.RandomState(seed)
    centres = np.linspace(-3.0, 3.0, blobs)
    rows, labels = [], []
    for i, cx in enumerate(centres):
        for j, cy in enumerate(centres):
            rows.append(rng.normal([cx, cy], spread, size=(n_per_cell, 2)))
            labels += ["A" if (i + j) % 2 == 0 else "B"] * n_per_cell
    return pd.DataFrame(np.vstack(rows), columns=["x", "y"]), pd.Series(labels)


def striped(n_per_cell=90, seed=0, spread=0.5):
    """Three classes over a 3x3 grid, each owning three scattered cells.

    Less degenerate than the checkerboard -- the marginals differ slightly -- so
    the base rules are not at chance, and the clause has to earn its improvement
    rather than supply the entire signal.
    """
    rng = np.random.RandomState(seed)
    assignment = [
        ["A", "B", "C"],
        ["C", "A", "B"],
        ["B", "C", "A"],
    ]
    centres = (-3.0, 0.0, 3.0)
    rows, labels = [], []
    for i, cx in enumerate(centres):
        for j, cy in enumerate(centres):
            rows.append(rng.normal([cx, cy], spread, size=(n_per_cell, 2)))
            labels += [assignment[i][j]] * n_per_cell
    return pd.DataFrame(np.vstack(rows), columns=["x", "y"]), pd.Series(labels)


def blocky(n_per_cell=70, seed=0, spread=0.5):
    """3x3 grid where class B owns a contiguous 2x2 rectangle of cells.

    The case merging exists for: four confused cells that abut, and so collapse
    into one ``AND NOT (x is [X2, X3] AND y is [Y2, Y3])`` rather than four
    separate lines saying the same thing.
    """
    rng = np.random.RandomState(seed)
    centres = (-4.0, 0.0, 4.0)
    rows, labels = [], []
    for i, cx in enumerate(centres):
        for j, cy in enumerate(centres):
            rows.append(rng.normal([cx, cy], spread, size=(n_per_cell, 2)))
            labels += ["A" if i == 0 or j == 0 else "B"] * n_per_cell
    return pd.DataFrame(np.vstack(rows), columns=["x", "y"]), pd.Series(labels)


def real_datasets():
    for name, loader in (
        ("iris", load_iris),
        ("wine", load_wine),
        ("breast_cancer", load_breast_cancer),
        ("digits", load_digits),
    ):
        data = loader()
        columns = [f"f{i}" for i in range(data.data.shape[1])]
        yield name, pd.DataFrame(data.data, columns=columns), pd.Series(data.target)

    for name, sep in (("synth_easy", 1.5), ("synth_hard", 0.7)):
        X, y = make_classification(
            n_samples=900, n_features=8, n_informative=5, n_redundant=1,
            n_classes=3, class_sep=sep, random_state=0,
        )
        yield name, pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), pd.Series(y)


def score_pair(X, y, seed, n_gaussians, **exclusion_kwargs):
    """Held-out accuracy without and with mining, plus the clause count."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    common = dict(n_gaussians=n_gaussians, top_p=1.0, random_state=seed)

    with contextlib.redirect_stdout(io.StringIO()):
        base = MixtureOfGaussiansFuzzyClassifier(**common).fit(X_train, y_train)
        corrected = MixtureOfGaussiansFuzzyClassifier(
            exclude_cross_terms=True, **common, **exclusion_kwargs
        ).fit(X_train, y_train)

    return (
        base.score(X_test, y_test),
        corrected.score(X_test, y_test),
        len(corrected.exclusions_),
        corrected,
    )


def run_synthetic():
    """``n_gaussians`` is matched to each grid, since a 3x3 problem needs three
    membership functions per feature before its cells are even expressible."""
    print("\n== synthetic: the outer product is the binding constraint ==\n")
    print(f"{'problem':<24}{'base':>8}{'excl':>8}{'delta':>9}{'clauses':>9}{'cells':>7}")
    print("-" * 65)
    for name, builder, n_gaussians in (
        ("checkerboard 2x2", lambda s: checkerboard(seed=s), 2),
        ("checkerboard 3x3", lambda s: checkerboard(seed=s, blobs=3, n_per_cell=80), 3),
        ("striped 3x3 (3 class)", lambda s: striped(seed=s), 3),
        ("block 3x3 (2 class)", lambda s: blocky(seed=s), 3),
    ):
        rows = [score_pair(*builder(seed), seed, n_gaussians) for seed in SPLITS]
        base = float(np.mean([r[0] for r in rows]))
        excl = float(np.mean([r[1] for r in rows]))
        clauses = float(np.mean([r[2] for r in rows]))
        cells = float(np.mean([sum(c.n_cells for c in r[3].exclusions_) for r in rows]))
        print(
            f"{name:<24}{base:>8.4f}{excl:>8.4f}{excl - base:>+9.4f}"
            f"{clauses:>9.1f}{cells:>7.1f}"
        )

    print("\nExample rule base (block 3x3), showing what is admitted and discarded:")
    *_, model = score_pair(*blocky(seed=0), 0, 3, exclusion_max_clauses=8)
    print(describe_rules(model.model_))


def run_real(n_gaussians_values=(1, 2, 3), splits=SPLITS):
    """Per (dataset, split) case, so the tally below is over cases rather than
    dataset means. A mean can hide a method that wins big on two problems and
    quietly loses on four; the win/loss/worst columns cannot."""
    print(f"\n== real datasets: held-out accuracy, {len(splits)} splits ==\n")
    for n_gaussians in n_gaussians_values:
        print(f"-- n_gaussians={n_gaussians} " + "-" * 44)
        print(f"{'dataset':<18}{'base':>8}{'excl':>8}{'delta':>9}{'clauses':>9}")
        every_delta = []
        for name, X, y in real_datasets():
            rows = [score_pair(X, y, seed, n_gaussians)[:3] for seed in splits]
            base = float(np.mean([r[0] for r in rows]))
            excl = float(np.mean([r[1] for r in rows]))
            clauses = float(np.mean([r[2] for r in rows]))
            every_delta += [r[1] - r[0] for r in rows]
            print(f"{name:<18}{base:>8.4f}{excl:>8.4f}{excl - base:>+9.4f}{clauses:>9.1f}")

        every_delta = np.asarray(every_delta)
        mean = float(every_delta.mean())
        stderr = float(every_delta.std(ddof=1) / np.sqrt(every_delta.size))
        wins = int((every_delta > 1e-12).sum())
        losses = int((every_delta < -1e-12).sum())
        print(
            f"{'ALL CASES':<18}{'':>8}{'':>8}{mean:>+9.4f}  +/- {stderr:.4f}"
            f"   {wins} better / {losses} worse / {every_delta.size - wins - losses} unchanged"
            f"   (worst {every_delta.min():+.4f})\n"
        )


def run_strength_sweep(n_gaussians=2):
    """How much of a cell to withdraw. ``1.0`` is the hard veto."""
    print("\n== exclusion_strength sweep (mean delta over real datasets) ==\n")
    print(f"{'strength':<12}{'mean delta':>12}{'clauses':>10}")
    print("-" * 34)
    for strength in (0.25, 0.5, 0.75, 1.0):
        deltas, clauses = [], []
        for name, X, y in real_datasets():
            for seed in SPLITS:
                base, excl, n, _ = score_pair(
                    X, y, seed, n_gaussians, exclusion_strength=strength
                )
                deltas.append(excl - base)
                clauses.append(n)
        print(f"{strength:<12}{np.mean(deltas):>+12.4f}{np.mean(clauses):>10.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("synthetic", "real", "strength"), default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    if args.only in (None, "synthetic"):
        run_synthetic()
    if args.only in (None, "real"):
        run_real()
    if args.only in (None, "strength"):
        run_strength_sweep()


if __name__ == "__main__":
    main()
