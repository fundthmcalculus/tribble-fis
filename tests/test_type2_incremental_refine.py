"""IT2/GT2 refinement on `IncrementalFIS` (#146) matches the rebuild path.

`it2_refine`/`gt2_refine` now score each candidate through
`type2_incremental` instead of rebuilding the upper/lower (per alpha-plane)
models and running full forward passes. The search must be unchanged: same
initial loss, same accepted moves, same refined model, to floating-point
tolerance. Each test runs the refiner twice -- once as shipped, once with
`_try_incremental` forced to return ``None`` (the pre-#146 path, which is
also the fallback for non-Gaussian models) -- and compares.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from tribblefis import gt2_refine, it2_refine, type2_incremental
from tribblefis.gt2_classifier import GT2TribbleClassifier
from tribblefis.gt2_regressor import GT2TribbleRegressor
from tribblefis.it2_classifier import IT2TribbleClassifier
from tribblefis.it2_regressor import IT2TribbleRegressor
from tribblefis.kernel import NotCompilable


@pytest.fixture
def clf_data():
    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=3, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=7,
    )
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(4)]), y


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(3)
    X = rng.uniform(-2, 2, size=(120, 2))
    y = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] + 0.05 * rng.standard_normal(120)
    return pd.DataFrame(X, columns=["x0", "x1"]), y


def _params(model):
    out = []
    for fmodel in model.feature_models.values():
        for lmodel in fmodel.label_models.values():
            for mf in lmodel.memberships:
                for half in ("upper_mf", "lower_mf", "principal_mf"):
                    g = getattr(mf, half, None)
                    if g is not None:
                        out.extend([g.mu, g.sigma])
    return np.asarray(out)


def _run_both(monkeypatch, module, call):
    """(incremental_result, rebuild_result, n_incremental_builds)."""
    built = []
    real = module._try_incremental

    def spy(build):
        result = real(build)
        built.append(result is not None)
        return result

    monkeypatch.setattr(module, "_try_incremental", spy)
    fast = call()
    assert built and all(built), "incremental path was not taken"
    monkeypatch.setattr(module, "_try_incremental", lambda build: None)
    slow = call()
    return fast, slow


def test_it2_classifier_refine_matches_rebuild(monkeypatch, clf_data):
    X, y = clf_data
    clf = IT2TribbleClassifier(top_n=3, uncertainty_width=0.5, random_state=0).fit(X, y)
    y_idx = np.searchsorted(clf.classes_, y)

    def call():
        return it2_refine.refine_it2_antecedents(
            X, y_idx, clf.model_, clf.norms_, n_sweeps=1, km_iterations=10, verbose=False,
        )

    fast, slow = _run_both(monkeypatch, it2_refine, call)
    assert not np.array_equal(_params(fast), _params(clf.model_)), "refinement moved nothing"
    np.testing.assert_allclose(_params(fast), _params(slow), rtol=1e-9, atol=1e-12)


def test_gt2_classifier_refine_matches_rebuild(monkeypatch, clf_data):
    X, y = clf_data
    clf = GT2TribbleClassifier(top_n=3, uncertainty_width=0.5, n_alpha_planes=3,
                               random_state=0).fit(X, y)
    y_idx = np.searchsorted(clf.classes_, y)

    def call():
        return gt2_refine.refine_gt2_antecedents(
            X, y_idx, clf.model_, clf.norms_, n_alpha_planes=3, n_sweeps=1,
            km_iterations=None, verbose=False,
        )

    fast, slow = _run_both(monkeypatch, gt2_refine, call)
    assert not np.array_equal(_params(fast), _params(clf.model_))
    np.testing.assert_allclose(_params(fast), _params(slow), rtol=1e-9, atol=1e-12)


def _reg_kwargs(base):
    return dict(order=base.tsk_order, l2_reg=base.l2_reg, basis=base.consequent_basis,
                cross_pairs=base.cross_pairs_)


def test_it2_regressor_refine_matches_rebuild(monkeypatch, reg_data):
    X, y = reg_data
    reg = IT2TribbleRegressor(top_n=2, n_gaussians=2, n_output_buckets=3, random_state=42).fit(X, y)
    base = reg._base_regressor

    def call():
        return it2_refine.refine_it2_regressor_antecedents(
            X, y, reg.model_, reg.norms_, base.top_features_, **_reg_kwargs(base),
            km_iterations=10, n_sweeps=1, n_folds=2, verbose=False,
        )

    (m_f, ct_f, yb_f, info_f), (m_s, ct_s, yb_s, info_s) = _run_both(monkeypatch, it2_refine, call)
    assert info_f["init_val_mse"] == pytest.approx(info_s["init_val_mse"], rel=1e-9)
    assert info_f["val_mse"] == pytest.approx(info_s["val_mse"], rel=1e-9)
    assert info_f["val_mse"] <= info_f["init_val_mse"]
    np.testing.assert_allclose(_params(m_f), _params(m_s), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ct_f, ct_s, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(yb_f, yb_s, rtol=1e-7, atol=1e-9)


def test_gt2_regressor_refine_matches_rebuild(monkeypatch, reg_data):
    X, y = reg_data
    reg = GT2TribbleRegressor(top_n=2, n_gaussians=2, n_output_buckets=3, n_alpha_planes=3,
                              random_state=42).fit(X, y)
    base = reg._base_regressor

    def call():
        return gt2_refine.refine_gt2_regressor_antecedents(
            X, y, reg.model_, reg.norms_, base.top_features_, **_reg_kwargs(base),
            n_alpha_planes=3, km_iterations=10, n_sweeps=1, n_folds=2, verbose=False,
        )

    (m_f, ct_f, _, info_f), (m_s, ct_s, _, info_s) = _run_both(monkeypatch, gt2_refine, call)
    assert info_f["val_mse"] == pytest.approx(info_s["val_mse"], rel=1e-9)
    np.testing.assert_allclose(_params(m_f), _params(m_s), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ct_f, ct_s, rtol=1e-7, atol=1e-9)


def test_non_gaussian_models_fall_back(clf_data):
    X, y = clf_data
    clf = IT2TribbleClassifier(top_n=3, member_function="trap", random_state=0).fit(X, y)
    arrays = {c: X[c].to_numpy() for c in X.columns}
    with pytest.raises(NotCompilable):
        type2_incremental.IT2IncrementalFiring(clf.model_, arrays, clf.norms_)
    assert it2_refine._try_incremental(
        lambda: type2_incremental.IT2IncrementalFiring(clf.model_, arrays, clf.norms_)
    ) is None
