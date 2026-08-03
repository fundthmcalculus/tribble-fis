"""Measure the layered sub-dominant rule cascade.

A rule per label is one account of that class, fit from its own marginals. Where
two classes overlap, one rule wins the whole overlap and nothing in the rule base
records that the region was contested. :mod:`tribblefis.subdominant` reads the
model's own confusion matrix and adds a more specific rule *underneath* the one
that gets each pair wrong -- gated on that rule firing, fit on the rows it gets
wrong, firing explicitly for the corrected class.

Run with ``python -m benchmarks.subdominant_bench``.

Three questions, three sections:

``synthetic``
    A class hidden inside another, separated only by a feature the global ranking
    cannot see. An existence proof: the defect the cascade targets, isolated.

``real``
    Stock scikit-learn datasets, held out, several splits. Does the defect show
    up often enough to be worth mining for?

``compare``
    Against :class:`MixtureOfGaussiansFuzzySequenceClassifier`, which reaches the
    same confusions with whole binary *models* consulted in ``predict``, and
    against ``exclude_cross_terms``, which repairs rules that over-claim rather
    than regions that lack evidence. The three are different mechanisms and the
    interesting question is whether they are redundant.
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

from tribblefis.gaussian_classifier import (
    MixtureOfGaussiansFuzzyClassifier,
    MixtureOfGaussiansFuzzySequenceClassifier,
)
from tribblefis.subdominant import describe_subdominant

SPLITS = (0, 1, 2, 3, 4, 5)


def hidden_pocket(n_a=500, n_b=150, seed=0, depth=1):
    """``depth`` nested pockets, each separated from its host only by one feature
    that is uninformative at the scale of the whole dataset."""
    rng = np.random.RandomState(seed)
    columns = [f"f{i}" for i in range(2 + depth)]
    rows = [np.column_stack(
        [rng.normal(0.0, 3.0, n_a), rng.normal(0.0, 3.0, n_a)]
        + [rng.uniform(-6.0, 6.0, n_a) for _ in range(depth)]
    )]
    labels = ["A"] * n_a
    for level in range(depth):
        n = max(20, n_b // (level + 1))
        block = [rng.normal(0.0, 0.7, n), rng.normal(0.0, 0.7, n)]
        for k in range(depth):
            block.append(
                rng.normal(4.5, 0.35, n) if k <= level else rng.uniform(-6.0, 6.0, n)
            )
        rows.append(np.column_stack(block))
        labels += [chr(ord("B") + level)] * n
    return pd.DataFrame(np.vstack(rows), columns=columns), pd.Series(labels)


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


def fit_score(factory, X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model = factory(seed).fit(X_train, y_train)
        score = model.score(X_test, y_test)
    return score, model


def base_factory(n_gaussians=1, **kwargs):
    def make(seed):
        return MixtureOfGaussiansFuzzyClassifier(
            n_gaussians=n_gaussians, top_p=1.0, random_state=seed, **kwargs
        )
    return make


def run_synthetic():
    print("\n== synthetic: a class hidden inside another ==\n")
    print(f"{'problem':<26}{'base':>8}{'sub':>8}{'delta':>9}{'rules':>7}{'layers':>8}")
    print("-" * 66)
    for label, depth in (("1 pocket (2 class)", 1), ("2 nested (3 class)", 2)):
        bases, subs, rules, layers = [], [], [], []
        for seed in SPLITS:
            X, y = hidden_pocket(seed=seed, depth=depth)
            base, _ = fit_score(base_factory(), X, y, seed)
            sub, model = fit_score(
                base_factory(subdominant=True, subdominant_max_layers=3), X, y, seed
            )
            bases.append(base)
            subs.append(sub)
            rules.append(len(model.subdominant_))
            layers.append(len({r.layer for r in model.subdominant_}))
        # Means across splits, not the last split's pair beside a mean delta --
        # those do not add up and invite reading one split as the result.
        print(
            f"{label:<26}{np.mean(bases):>8.4f}{np.mean(subs):>8.4f}"
            f"{np.mean(subs) - np.mean(bases):>+9.4f}"
            f"{np.mean(rules):>7.1f}{np.mean(layers):>8.1f}"
        )

    print("\nExample cascade (1 pocket):")
    X, y = hidden_pocket(seed=0, depth=1)
    _, model = fit_score(base_factory(subdominant=True), X, y, 0)
    print(describe_subdominant(model.model_))


def run_real(n_gaussians_values=(1, 2)):
    print(f"\n== real datasets: held-out accuracy, {len(SPLITS)} splits ==\n")
    for n_gaussians in n_gaussians_values:
        print(f"-- n_gaussians={n_gaussians} " + "-" * 46)
        print(f"{'dataset':<18}{'base':>8}{'sub':>8}{'delta':>9}{'rules':>7}")
        every = []
        for name, X, y in real_datasets():
            pairs, counts = [], []
            for seed in SPLITS:
                base, _ = fit_score(base_factory(n_gaussians), X, y, seed)
                sub, model = fit_score(
                    base_factory(n_gaussians, subdominant=True), X, y, seed
                )
                pairs.append((base, sub))
                counts.append(len(model.subdominant_))
            deltas = [s - b for b, s in pairs]
            every += deltas
            print(
                f"{name:<18}{np.mean([b for b, _ in pairs]):>8.4f}"
                f"{np.mean([s for _, s in pairs]):>8.4f}"
                f"{np.mean(deltas):>+9.4f}{np.mean(counts):>7.1f}"
            )
        every = np.asarray(every)
        wins = int((every > 1e-12).sum())
        losses = int((every < -1e-12).sum())
        print(
            f"{'ALL CASES':<18}{'':>8}{'':>8}{every.mean():>+9.4f}"
            f"  +/- {every.std(ddof=1) / np.sqrt(every.size):.4f}"
            f"   {wins} better / {losses} worse / {every.size - wins - losses} unchanged"
            f"   (worst {every.min():+.4f})\n"
        )


def run_compare(n_gaussians=2):
    """Sub-dominant rules against the two mechanisms nearest to them."""
    print("\n== against the other confusion repairs ==\n")

    def sequence(seed):
        return MixtureOfGaussiansFuzzySequenceClassifier(
            n_gaussians=n_gaussians, top_p=1.0, random_state=seed
        )

    variants = {
        "base": base_factory(n_gaussians),
        "exclusions": base_factory(n_gaussians, exclude_cross_terms=True),
        "subdominant": base_factory(n_gaussians, subdominant=True),
        "both": base_factory(n_gaussians, exclude_cross_terms=True, subdominant=True),
        "sequence experts": sequence,
    }

    scores = {name: [] for name in variants}
    for dataset, X, y in real_datasets():
        for seed in SPLITS:
            for name, factory in variants.items():
                score, _ = fit_score(factory, X, y, seed)
                scores[name].append(score)

    reference = np.asarray(scores["base"])
    print(f"{'variant':<20}{'mean acc':>10}{'vs base':>10}{'better':>8}{'worse':>7}")
    print("-" * 55)
    for name, values in scores.items():
        values = np.asarray(values)
        delta = values - reference
        print(
            f"{name:<20}{values.mean():>10.4f}{delta.mean():>+10.4f}"
            f"{int((delta > 1e-12).sum()):>8}{int((delta < -1e-12).sum()):>7}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("synthetic", "real", "compare"), default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    if args.only in (None, "synthetic"):
        run_synthetic()
    if args.only in (None, "real"):
        run_real()
    if args.only in (None, "compare"):
        run_compare()


if __name__ == "__main__":
    main()
