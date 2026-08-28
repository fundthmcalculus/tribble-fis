"""A constant feature must not produce a zero-width box bound.

`optimizers` validates its variables as of #121:

    if not lower_bound < upper_bound:
        raise ValueError("lower_bound (...) must be less than upper_bound (...)")

Every refinement path here builds mu bounds straight from a feature's observed
min and max, so a constant column made `refine_method="optimizers"` raise before
doing any work. The bug was quiet for a long time because the code *did* guard
the degenerate case one line earlier, for sigma:

    rng = hi - lo if hi > lo else 1.0     # guarded
    bounds.append((lo, hi))               # not guarded

so it reads as handled at a glance. Five call sites had the identical shape --
`refine.py`, and two each in `it2_refine.py` and `gt2_refine.py` -- which is why
the fix is one shared `feature_span()` rather than five patches.

Constant columns are ordinary, not pathological: RT-IOT2022 ships one
(`bwd_URG_flag_count`) among 82 numeric features, and a train split of a
low-cardinality column can produce one from data that is not globally constant.
Nothing upstream drops them.
"""

import numpy as np
import pandas as pd
import pytest

from tribblefis.refine import build_param_bounds, feature_span


def test_feature_span_widens_a_constant_column():
    lo, hi, rng = feature_span(np.full(50, 3.0))
    assert lo < hi, "a constant column must still give a usable interval"
    assert rng == 1.0, "degenerate range falls back to the unit convention"
    assert lo < 3.0 < hi, "the interval must contain the observed value"


def test_feature_span_is_unchanged_for_an_ordinary_column():
    col = np.array([1.0, 2.0, 5.0, 4.0])
    assert feature_span(col) == (1.0, 5.0, 4.0)


def test_feature_span_handles_a_single_row():
    lo, hi, _ = feature_span(np.array([7.5]))
    assert lo < 7.5 < hi


@pytest.mark.parametrize("value", [0.0, -2.5, 1e6])
def test_constant_bounds_are_accepted_by_the_optimizer(value):
    """The bounds must satisfy the constraint `optimizers` actually enforces.

    Asserting `lo < hi` ourselves would only re-state the fix. This asserts it
    against the validator that rejected the old bounds, so the test fails again
    if that contract changes rather than merely if this helper does.
    """
    variables = pytest.importorskip("optimizers.continuous.variables")
    lo, hi, _ = feature_span(np.full(10, value))
    variables.InputContinuousVariable("mu", float(lo), float(hi))


def test_build_param_bounds_survives_a_constant_feature():
    """End to end through the builder that raised, on a frame with one constant
    column beside ordinary ones -- the RT-IOT2022 shape, in miniature."""
    from tribblefis.gauss_math import create_gaussian_membership_dict
    from tribblefis.regression import partition_output

    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(
        {
            "varies": rng.uniform(-3, 3, size=n),
            "also_varies": rng.uniform(-3, 3, size=n),
            "constant": np.zeros(n),
        }
    )
    y = pd.Series(np.sin(X["varies"]) + 0.3 * X["also_varies"], name="y_value")
    y_partitioned, _ = partition_output(3, y)

    model = create_gaussian_membership_dict(
        X,
        y_partitioned["y_bucket"],
        top_n_var_names=list(X.columns),
        n_gaussians=1,
    )
    bounds = build_param_bounds(model, X)

    assert bounds, "the model produced no parameter slots to bound"
    for lo, hi in bounds:
        assert lo < hi, f"zero-width bound {(lo, hi)} would be rejected upstream"

    variables = pytest.importorskip("optimizers.continuous.variables")
    for i, (lo, hi) in enumerate(bounds):
        variables.InputContinuousVariable(f"p{i}", float(lo), float(hi))
