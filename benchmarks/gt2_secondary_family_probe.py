"""GT2 secondary-membership family probe: triangular vs Gaussian-on-Gaussian.

Answers issue #145's step 1 empirically: does replacing GT2's triangular
secondary grade (the shipped ``GT2GaussianMembership.alpha_cut``) with a
truncated Gaussian-on-Gaussian one change anything a caller can observe --
accuracy/R^2, raw interval coverage/width, conformal coverage/width, or the
*sign* of the width-vs-error relationship that #149 found running backwards?

No production code changes: the Gaussian family is swapped in by patching
``GT2GaussianMembership.alpha_cut`` for the duration of one fit/predict.

Gaussian-on-Gaussian, truncated to the footprint of uncertainty: the
secondary grade over sigma is a Gaussian centred on ``principal_mf.sigma``
whose per-side spread is ``(edge - principal) / c``, clipped at the FOU edge.
Its alpha-cut half-width, as a fraction of the per-side FOU span, is
``min(1, sqrt(-2 ln alpha) / c)`` -- versus triangular's ``1 - alpha``. Both
are nested, both collapse to the principal at alpha=1; ``c`` is the
truncation point in secondary standard deviations (``c=3``: FOU edge at
3 sd, alpha ~= 0.011).

Backs docs/gt2-secondary-family-evaluation.md. Run with:

    python -m benchmarks.gt2_secondary_family_probe
    python -m benchmarks.gt2_secondary_family_probe -o benchmarks/results/gt2_secondary_family.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine, make_friedman1
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from tribblefis.gauss_data import GaussianMembership, GT2GaussianMembership, IT2GaussianMembership
from tribblefis.gt2_classifier import GT2TribbleClassifier
from tribblefis.gt2_regressor import GT2TribbleRegressor
from tribblefis.it2_classifier import IT2TribbleClassifier
from tribblefis.it2_regressor import IT2TribbleRegressor

SEEDS = (42, 7, 123)
N_GAUSSIANS = 3
N_ALPHA_PLANES = 5

CLASSIFICATION = {
    "wine": (lambda: load_wine(return_X_y=True), 8),
    "breast_cancer": (lambda: load_breast_cancer(return_X_y=True), 10),
}
REGRESSION = {
    "friedman1": (lambda: make_friedman1(n_samples=500, n_features=10, noise=1.0, random_state=42), 5),
    "diabetes": (lambda: load_diabetes(return_X_y=True), 6),
}

# family name -> fraction-of-FOU-span schedule f(alpha) in [0, 1]
FAMILIES = {
    "triangular": lambda a: 1.0 - a,
    "gauss_c2": lambda a: min(1.0, np.sqrt(-2.0 * np.log(a)) / 2.0) if a > 0 else 1.0,
    "gauss_c3": lambda a: min(1.0, np.sqrt(-2.0 * np.log(a)) / 3.0) if a > 0 else 1.0,
}


def _make_alpha_cut(schedule):
    def alpha_cut(self, alpha: float) -> IT2GaussianMembership:
        f = schedule(float(alpha))
        p = self.principal_mf.sigma
        sigma_lo = p - f * (p - self.lower_mf.sigma)
        sigma_hi = p + f * (self.upper_mf.sigma - p)
        mu = self.principal_mf.mu
        return IT2GaussianMembership(
            upper_mf=GaussianMembership(mu=mu, sigma=sigma_hi, id=self.upper_mf.id),
            lower_mf=GaussianMembership(mu=mu, sigma=sigma_lo, id=self.lower_mf.id),
            id=self.id,
        )

    return alpha_cut


@contextlib.contextmanager
def secondary_family(name: str):
    original = GT2GaussianMembership.alpha_cut
    GT2GaussianMembership.alpha_cut = _make_alpha_cut(FAMILIES[name])
    try:
        yield
    finally:
        GT2GaussianMembership.alpha_cut = original


def _regression_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    lo, hi = model.predict_intervals(X_test)
    y_test = np.asarray(y_test)
    rng = float(y_test.max() - y_test.min()) or 1.0
    width = hi - lo
    rho = spearmanr(width, np.abs(y_test - y_pred)).statistic
    return dict(
        r2=float(r2_score(y_test, y_pred)),
        coverage=float(np.mean((y_test >= lo) & (y_test <= hi))),
        rel_width=float(np.mean(width)) / rng,
        width_abs_resid_spearman=float(rho) if np.isfinite(rho) else None,
    )


def _classification_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    upper, lower = model.predict_intervals(X_test)
    classes = list(model.classes_)
    idx = np.array([classes.index(p) for p in y_pred])
    width = upper[np.arange(len(y_pred)), idx] - lower[np.arange(len(y_pred)), idx]
    correct = y_pred == np.asarray(y_test)
    return dict(
        accuracy=float(accuracy_score(y_test, y_pred)),
        width_correct=float(width[correct].mean()) if correct.any() else None,
        width_incorrect=float(width[~correct].mean()) if (~correct).any() else None,
    )


def run() -> list[dict]:
    records = []
    for ds, (loader, top_n) in REGRESSION.items():
        X, y = loader()
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
            for conformal in (False, True):
                common = dict(top_n=top_n, n_gaussians=N_GAUSSIANS, conformal_calibration=conformal)
                m = IT2TribbleRegressor(**common).fit(Xtr, ytr)
                records.append(dict(task="regression", dataset=ds, seed=seed, conformal=conformal,
                                    family="IT2", **_regression_metrics(m, Xte, yte)))
                for fam in FAMILIES:
                    with secondary_family(fam):
                        m = GT2TribbleRegressor(n_alpha_planes=N_ALPHA_PLANES, **common).fit(Xtr, ytr)
                        rec = _regression_metrics(m, Xte, yte)
                    records.append(dict(task="regression", dataset=ds, seed=seed, conformal=conformal,
                                        family=fam, **rec))
    for ds, (loader, top_n) in CLASSIFICATION.items():
        X, y = loader()
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            common = dict(top_n=top_n, n_gaussians=N_GAUSSIANS)
            m = IT2TribbleClassifier(**common).fit(Xtr, ytr)
            records.append(dict(task="classification", dataset=ds, seed=seed, family="IT2",
                                **_classification_metrics(m, Xte, yte)))
            for fam in FAMILIES:
                with secondary_family(fam):
                    m = GT2TribbleClassifier(n_alpha_planes=N_ALPHA_PLANES, **common).fit(Xtr, ytr)
                    rec = _classification_metrics(m, Xte, yte)
                records.append(dict(task="classification", dataset=ds, seed=seed, family=fam, **rec))
    return records


def _summarize(records: list[dict]) -> None:
    def mean(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print("\nregression (mean over seeds)")
    print("| dataset | conformal | family | R^2 | coverage | rel width | spearman(width,|resid|) |")
    print("|---|---|---|--:|--:|--:|--:|")
    for ds in REGRESSION:
        for conformal in (False, True):
            for fam in ("IT2", *FAMILIES):
                rows = [r for r in records if r["task"] == "regression" and r["dataset"] == ds
                        and r["conformal"] == conformal and r["family"] == fam]
                print(f"| {ds} | {conformal} | {fam} | {mean(rows, 'r2'):.3f} | {mean(rows, 'coverage'):.3f} "
                      f"| {mean(rows, 'rel_width'):.3f} | {mean(rows, 'width_abs_resid_spearman'):+.3f} |")
    print("\nclassification (mean over seeds)")
    print("| dataset | family | accuracy | width correct | width incorrect |")
    print("|---|---|--:|--:|--:|")
    for ds in CLASSIFICATION:
        for fam in ("IT2", *FAMILIES):
            rows = [r for r in records if r["task"] == "classification" and r["dataset"] == ds
                    and r["family"] == fam]
            print(f"| {ds} | {fam} | {mean(rows, 'accuracy'):.3f} | {mean(rows, 'width_correct'):.3f} "
                  f"| {mean(rows, 'width_incorrect'):.3f} |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = run()
    _summarize(records)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
