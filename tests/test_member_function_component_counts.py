"""How many memberships each `member_function` actually builds, and why.

Written for tribble-fis#213, which reported that
`test_refine_gt2_regressor_antecedents_never_increases_cv_loss[triangular]`
costs 4x its `[trap]` sibling "on identical settings", and called the result
"the part that doesn't add up" -- a trapezoid has more free parameters per slot
than a triangle, so the naive expectation runs the other way.

The settings are identical. **The models are not.** Measured on that test's own
fixture (`make_regression(n_samples=200, n_features=4, n_informative=3)`,
`top_n=3`, `n_gaussians=2`):

    member_function   fit      memberships   refine
    trap              0.02s     6            98.2s
    triangular        0.97s    12           372.9s

The two legs take entirely different fitting paths:

* `member_function="trap"` with the default `trapz_method="fast"` routes to
  `trapz_math_fast.create_trapz_membership_dict_fast`, a histogram fitter that
  takes **no component-count argument at all**. It emits one trapezoid per
  merged contiguous non-empty histogram region, so the count is a property of
  the data, and `n_gaussians` has no effect whatsoever.
* `member_function="triangular"` has no fast path. Its branch in
  `gaussian_regressor.fit` hardcodes the EM fitter
  (`trapz_math.create_trapz_membership_dict(..., shape="triangle")`), which
  *does* honour `n_gaussians` -- hence 2 per feature/label, and hence the 48x
  fit-time gap as well.

The 4x then follows from the refinement loop's shape rather than from anything
about triangles. `gt2_refine.refine_gt2_regressor_antecedents` is
``for sweep: for slot in slots: <sub_maxfun evaluations>``, and every evaluation
runs the whole model across every alpha plane. So cost is O(slots x model size),
both factors scale with the membership count, and doubling the memberships
quadruples the work: 12/6 = 2, and 2^2 = 4 against a measured 3.80x.

Nothing here is a bug in the triangular path. What *is* worth pinning is the
structural asymmetry itself, so the next person profiling that test reads this
instead of re-deriving it -- and so a future change that quietly makes the two
paths comparable (or makes them diverge further) has to say so.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from tribblefis.gaussian_regressor import TribbleRegressor


def _membership_counts(model):
    """Memberships per (feature, label), flattened."""
    return [
        len(label_model.memberships)
        for feature_model in model.feature_models.values()
        for label_model in feature_model.label_models.values()
    ]


@pytest.fixture
def regression_data():
    """#213's own fixture, verbatim, so the counts here are that test's counts."""
    X, y = make_regression(
        n_samples=200, n_features=4, n_informative=3, noise=5.0, random_state=3
    )
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]), y


def _fit(X, y, **kwargs):
    kwargs.setdefault("top_n", 3)
    return TribbleRegressor(random_state=0, **kwargs).fit(X, y).model_


def test_fast_trapezoids_ignore_n_gaussians(regression_data, capsys):
    """`n_gaussians` has no effect on the default trap path. Not a typo -- measured.

    `create_trapz_membership_dict_fast` has no component-count parameter to pass
    it to. The count comes from the histogram, so asking for 1, 2 or 8
    components yields the same model.

    This is the half of #213 that surprises people: the parameter is accepted,
    is documented as "per feature per label", and silently does nothing here.
    """
    X, y = regression_data
    counts = {
        n: _membership_counts(_fit(X, y, member_function="trap", n_gaussians=n))
        for n in (1, 2, 8)
    }
    capsys.readouterr()

    assert counts[1] == counts[2] == counts[8], counts
    # And what it settles on is one per feature/label, because these features are
    # unimodal: a single contiguous run of non-empty histogram bins.
    assert set(counts[2]) == {1}, counts[2]


def test_triangular_honours_n_gaussians(regression_data, capsys):
    """The triangular path routes through EM, which does take a component count."""
    X, y = regression_data
    counts = {
        n: _membership_counts(_fit(X, y, member_function="triangular", n_gaussians=n))
        for n in (1, 2)
    }
    capsys.readouterr()

    assert set(counts[1]) == {1}, counts[1]
    assert set(counts[2]) == {2}, counts[2]


def test_trap_and_triangular_build_different_sized_models(regression_data, capsys):
    """The claim at the centre of #213: "identical settings" are not identical work.

    This is the assertion that makes the issue's headline resolvable. At the
    same `n_gaussians`, the triangular model carries twice the memberships, and
    GT2 antecedent refinement costs O(slots x model size) -- so the leg with 2x
    the memberships costs 4x, which is what was measured (3.80x).
    """
    X, y = regression_data
    trap = _membership_counts(_fit(X, y, member_function="trap", n_gaussians=2))
    triangular = _membership_counts(
        _fit(X, y, member_function="triangular", n_gaussians=2)
    )
    capsys.readouterr()

    assert sum(triangular) == 2 * sum(trap), (sum(trap), sum(triangular))


def test_fast_trapezoid_count_follows_the_data_not_the_parameter(capsys):
    """The fast fitter is data-adaptive, which is a design choice, not an oversight.

    One trapezoid per merged contiguous non-empty histogram region. A unimodal
    column gives one; a column with a genuine gap gives two. That is a defensible
    way to choose a component count -- it just is not `n_gaussians`, and nothing
    in the parameter's documentation said so before #213.

    The bimodal column is deliberately *independent of the target*. Memberships
    are fitted per (feature, output bucket), so a bimodal feature that drives
    the target is unimodal inside each bucket -- the partition separates the
    modes before the fitter ever sees them, and the first version of this test
    measured 1 region for exactly that reason. Both modes have to survive into
    every bucket for this to be testing the fitter rather than the partition.
    """
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(
        {
            "unimodal": rng.normal(0.0, 1.0, n),
            # Two tight clusters far apart, so the bins between them are empty by
            # more than `merge_width_ratio` and the regions do not merge.
            "bimodal": np.concatenate(
                [rng.normal(-20.0, 0.3, n // 2), rng.normal(20.0, 0.3, n - n // 2)]
            ),
        }
    )
    y = X["unimodal"] + rng.normal(0, 0.1, n)

    model = _fit(X, y, member_function="trap", n_gaussians=1, top_n=2)
    capsys.readouterr()

    per_feature = {
        name: max(len(lm.memberships) for lm in fm.label_models.values())
        for name, fm in model.feature_models.items()
    }
    assert per_feature["unimodal"] == 1, per_feature
    assert per_feature["bimodal"] >= 2, per_feature


def test_triangular_ignores_trapz_method(regression_data, capsys):
    """`trapz_method` is accepted and does nothing when `member_function` is triangular.

    The branch hardcodes the EM fitter; there is no histogram-based triangle
    implementation to select. `TribbleClassifier`'s docstring says so;
    `TribbleRegressor`'s did not until #213.

    Worth recording *why* there is no fast triangle rather than treating it as a
    gap to fill. `fit_trapezoids_fast`'s geometry is built around the plateau:
    `[b, c]` spans the data region so every observed value has membership
    exactly 1, and the ramps sit outside it. That is not a stylistic choice --
    its docstring records that the inset version gave the smallest observed
    value (and everything tied with it) zero membership from the very term
    fitted to describe it, leaving 78.6% of held-out Concrete rows covered by no
    rule at all. A triangle has no plateau to spread over the region, so the
    same construction would put the region's edges near zero membership again.
    Porting the fast path to triangles means solving that first, not just
    collapsing `b` and `c`.
    """
    X, y = regression_data
    fast = _membership_counts(
        _fit(X, y, member_function="triangular", n_gaussians=2, trapz_method="fast")
    )
    em = _membership_counts(
        _fit(X, y, member_function="triangular", n_gaussians=2, trapz_method="em")
    )
    capsys.readouterr()

    assert fast == em, (fast, em)
