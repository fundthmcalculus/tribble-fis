"""Properties the uniformity-preserving scalers must have.

Issue #220 motivates these three classes with an accuracy table from experiments
in a separate repository. Those numbers cannot be re-run here -- the datasets
and the sweep harness live in `grad-school` -- so this file deliberately does
not try to restate them as assertions. Asserting a downstream R^2 the suite
cannot reproduce would be a test that pins a claim rather than a behaviour, and
the first environment change that moved it by 0.001 would get it deleted.

What is pinned here instead is everything that *is* mechanical:

- the transforms do what their names say (uniformity, monotonicity, bounds);
- the relationships between them (``n_pieces=1`` **is** min-max; log1p cannot
  move an empirical CDF);
- the edge cases that quietly produce wrong numbers rather than errors
  (constant features, NaN, ties, out-of-range inputs at transform time).

`benchmarks/uniformity_scaling.py` carries the empirical half, on synthetic
distributions with known pathology, where the numbers are reproducible.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from tribblefis.gaussian_regressor import TribbleRegressor
from tribblefis.scaling import (
    EmpiricalCDFScaler,
    MinMaxScaler,
    PiecewiseLinearCDFScaler,
    QuantileUniformScaler,
)

UNIFORMITY_SCALERS = [EmpiricalCDFScaler, PiecewiseLinearCDFScaler, QuantileUniformScaler]


@pytest.fixture
def pathological():
    """Four marginals, each non-uniform in a different way.

    A lognormal column alone would let a transform pass by being good at right
    skew and nothing else. Each column here breaks a different naive
    implementation: the bimodal one has a gap with no data in it, the
    zero-inflated one has a heavy atom at the minimum (which is what makes
    "rescale from 1/n" wrong), and the discrete one is nothing but ties.
    """
    rng = np.random.default_rng(20220)
    n = 400
    return pd.DataFrame(
        {
            "lognormal": rng.lognormal(0.0, 2.0, n),
            "bimodal": np.concatenate(
                [rng.normal(-5, 0.5, n // 2), rng.normal(5, 0.5, n - n // 2)]
            ),
            "zero_inflated": np.where(
                rng.random(n) < 0.4, 0.0, rng.uniform(1.0, 10.0, n)
            ),
            "discrete": rng.integers(0, 5, n).astype(float),
        }
    )


def _ks_against_uniform(u):
    """Kolmogorov-Smirnov distance between ``u`` and Uniform(0, 1).

    Written out rather than pulled from scipy: scipy is not a dependency of this
    package, and the one-sample statistic against a uniform is four lines.
    """
    x = np.sort(np.asarray(u, dtype=float))
    n = x.size
    edf_hi = np.arange(1, n + 1) / n
    edf_lo = np.arange(0, n) / n
    return float(max(np.max(edf_hi - x), np.max(x - edf_lo)))


# --------------------------------------------------------------------------
# The property the family is named for
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", UNIFORMITY_SCALERS)
def test_output_is_far_more_uniform_than_the_input(cls, pathological):
    """Each scaler must actually flatten the marginal it is given.

    Measured as a KS distance against Uniform(0, 1), against the same statistic
    for a plain min-max of the raw column. Min-max is the honest control: it
    also bounds to [0, 1], so any difference here is the *shape* change and not
    the bounding.

    The threshold is deliberately loose (half the min-max distance). The claim
    being pinned is "these transforms make the marginal substantially more
    uniform", which is the mechanism issue #220 rests on; a tight bound would
    be pinning the sample rather than the property.

    On a column with a heavy atom the bound is the atom's mass instead, and
    there it is *tight* rather than loose -- see the comment on `bound` below.
    """
    baseline = MinMaxScaler(log_dynamic_range=None).fit_transform(pathological)
    transformed = cls().fit_transform(pathological)

    for j, col in enumerate(pathological.columns):
        raw_ks = _ks_against_uniform(baseline[:, j])
        new_ks = _ks_against_uniform(transformed[:, j])

        # No monotone map can split an atom. If a fraction `a` of a column is
        # one value, that fraction of the output is one value too, and the KS
        # distance against a uniform is bounded below by exactly `a`. Demanding
        # "half the min-max distance" on such a column would be demanding the
        # impossible, so the bound becomes whichever of the two is achievable.
        #
        # Note this is not a let-off. Measured on this fixture all three
        # scalers land *on* the floor -- zero_inflated (atom 0.405) at 0.405,
        # discrete (atom 0.230) at 0.210 -- so on those columns the assertion
        # says the transform is optimal, which is a stronger statement than the
        # halving it replaces. The continuous columns keep the halving bound
        # and clear it by two orders of magnitude (0.960 -> 0.003).
        atom = float(pathological[col].value_counts().max()) / len(pathological)
        bound = max(raw_ks / 2.0, atom)
        assert new_ks <= bound + 1e-9, (
            f"{col}: minmax {raw_ks:.4f} -> {new_ks:.4f} (atom floor {atom:.3f})"
        )


def test_empirical_cdf_attains_the_atom_floor_exactly():
    """On an atom-heavy column the empirical CDF is optimal, not merely better.

    A monotone map cannot split an atom, so a column that is 40% zeros has a KS
    distance against uniform of at least 0.40 whatever is done to it. Asserting
    the transform *reaches* that floor -- rather than just beating min-max -- is
    the strongest claim available on such a column, and it is the one that
    catches an off-by-one in the rescale: a version that mapped the atom to
    0.4 instead of 0.0 also has 40% of its mass in one place, but at the wrong
    place, and scores worse.
    """
    rng = np.random.default_rng(20220)
    values = np.where(rng.random(2000) < 0.4, 0.0, rng.uniform(1.0, 10.0, 2000))
    atom = float((values == 0.0).mean())

    u = EmpiricalCDFScaler().fit_transform(pd.DataFrame({"x": values})).ravel()
    assert _ks_against_uniform(u) == pytest.approx(atom, abs=1e-3)


def test_empirical_cdf_is_exactly_uniform_on_distinct_training_values(pathological):
    """With no ties the empirical CDF hits the uniform grid exactly.

    This is the strong form of the property, and it is available on the
    continuous columns because ranks there are a permutation of 1..n.
    """
    column = pathological[["lognormal"]]
    u = EmpiricalCDFScaler().fit_transform(column).ravel()
    assert np.allclose(np.sort(u), np.linspace(0.0, 1.0, len(u)))


# --------------------------------------------------------------------------
# Relationships between the classes -- the claims made in the docstrings
# --------------------------------------------------------------------------


def test_one_piece_is_exactly_min_max(pathological):
    """``PiecewiseLinearCDFScaler(n_pieces=1)`` **is** min-max, to the bit.

    The class docstring calls itself a strict generalization of
    :class:`MinMaxScaler` rather than an alternative to it. That is a factual
    claim about the code and this is what makes it one: two breakpoints at the
    minimum and maximum, one affine map. `log_dynamic_range=None` on the
    control because the piecewise scaler has no log step to match.
    """
    expected = MinMaxScaler(log_dynamic_range=None).fit_transform(pathological)
    actual = PiecewiseLinearCDFScaler(n_pieces=1).fit_transform(pathological)
    assert np.allclose(actual, expected, rtol=0, atol=1e-12)


def test_log1p_cannot_change_the_empirical_cdf(pathological):
    """Rank is monotone-invariant, so log1p before an ECDF is a no-op.

    This is why :class:`EmpiricalCDFScaler` takes no ``log_features`` argument.
    The claim is exact, not approximate -- the ranks are identical integers, so
    the outputs are identical floats -- and asserting it exactly is the point:
    an implementation that interpolated in value space instead of ranking would
    pass an `allclose` and fail this.
    """
    positive = pathological[["lognormal", "zero_inflated", "discrete"]]
    direct = EmpiricalCDFScaler().fit_transform(positive)
    logged = EmpiricalCDFScaler().fit_transform(np.log1p(positive))
    assert np.array_equal(direct, logged)


def test_more_pieces_approach_the_empirical_cdf(pathological):
    """More pieces move monotonically toward the empirical CDF.

    Toward, not to. The two never coincide: the piecewise scaler interpolates
    linearly between order statistics while the empirical CDF is a step
    function, so they differ by up to one step's height at any `n_pieces`. What
    is testable, and what this asserts, is that the error is monotone
    decreasing and falls by at least an order of magnitude across the range --
    a real dial, not a convergence proof.
    """
    column = pathological[["lognormal"]]
    target = EmpiricalCDFScaler().fit_transform(column)
    errors = [
        float(
            np.abs(
                PiecewiseLinearCDFScaler(n_pieces=k).fit_transform(column) - target
            ).mean()
        )
        for k in (1, 2, 5, 10, 50, 200)
    ]
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < errors[0] / 10


# --------------------------------------------------------------------------
# Invertibility
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", UNIFORMITY_SCALERS)
def test_round_trip_recovers_the_training_data(cls, pathological):
    """`inverse_transform(transform(X))` must return X on the training set.

    Exact for the empirical CDF, because the inverse looks its answer up in the
    same table the forward map wrote. The first implementation recomputed a
    rank as `ceil(f * n) - 1` and was wrong by a whole training value wherever
    `f * n` landed a few ulps above an integer -- max round-trip error 1.03 on
    the lognormal column, and silent. Hence a tolerance tight enough to catch
    an off-by-one-order-statistic rather than merely a float wobble.

    An independent implementation of the same class in #224 had the stronger
    form of this: `(cdf * n).astype(int)` is off by one for *every* input, so
    `[1, 2, 3, 4, 5]` round-tripped to `[2, 3, 4, 5, 6]`. It shipped green past
    a test named for the inverse transform, because that test compared shapes
    and bounds rather than values. `TransformedTargetRegressor` inverts through
    this path, so a target scaled that way returns shifted by one order
    statistic on every prediction.
    """
    scaler = cls().fit(pathological)
    recovered = scaler.inverse_transform(scaler.transform(pathological))
    assert np.allclose(recovered, pathological.to_numpy(), rtol=1e-9, atol=1e-9)


def test_piecewise_inverse_is_exact_off_the_training_grid():
    """The piecewise scaler is a genuine bijection, not just a round-tripper.

    The empirical CDF can only snap an unseen value to a training order
    statistic; this one interpolates, so it must recover unseen values too. That
    difference is the reason both classes exist.
    """
    rng = np.random.default_rng(7)
    train = pd.DataFrame({"x": rng.lognormal(0, 1.5, 500)})
    unseen = pd.DataFrame({"x": rng.uniform(train["x"].min(), train["x"].max(), 100)})

    scaler = PiecewiseLinearCDFScaler(n_pieces=25).fit(train)
    recovered = scaler.inverse_transform(scaler.transform(unseen)).ravel()
    assert np.allclose(recovered, unseen["x"].to_numpy(), rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------
# Monotonicity and bounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", UNIFORMITY_SCALERS)
def test_transform_is_monotone_non_decreasing(cls, pathological):
    """A scaler that reorders its input has destroyed the feature.

    Every claim in the module docstring -- that these subsume log1p, that they
    preserve rank, that the controls in #220's table move by <= 0.002 because
    both transforms are monotone -- rests on this and nothing else.
    """
    scaler = cls().fit(pathological)
    grid = pd.DataFrame(
        {
            col: np.linspace(pathological[col].min(), pathological[col].max(), 200)
            for col in pathological.columns
        }
    )
    out = scaler.transform(grid)
    assert np.all(np.diff(out, axis=0) >= -1e-12)


@pytest.mark.parametrize("cls", UNIFORMITY_SCALERS)
@pytest.mark.parametrize("feature_range", [(0.0, 1.0), (-1.0, 1.0), (-0.5, 1.5)])
def test_output_lands_in_feature_range_including_out_of_range_inputs(
    cls, feature_range, pathological
):
    """Bounded output is the contract, and it must survive unseen extremes.

    A FIS placed on `[0, 1]` and then handed 1.4 at predict time is the failure
    this prevents. Values far outside the training range clamp to the endpoints
    -- which discards the magnitude of the excursion, exactly as the module's
    pre-log flooring does, and is documented on each class.
    """
    lo, hi = feature_range
    scaler = cls(feature_range=feature_range).fit(pathological)

    on_train = scaler.transform(pathological)
    assert on_train.min() >= lo - 1e-12 and on_train.max() <= hi + 1e-12
    # The training extremes must *attain* the endpoints, not merely stay inside
    # them: MF placement on a domain the data never reaches wastes the tails.
    # Every column, not a prefix of them -- the atom-heavy and discrete columns
    # are exactly the ones where this fails. #224's `EmpiricalCDFScaler` used
    # `rank / n` with no rescale, which put the minimum at `F(min)` -- 1/n on a
    # continuous column, but 0.373 on a column that is 40% zeros, leaving the
    # bottom 37% of feature_range unreachable by any input.
    assert np.allclose(on_train.min(axis=0), lo)
    assert np.allclose(on_train.max(axis=0), hi)

    wild = pathological * 100.0 - 500.0
    out = scaler.transform(wild)
    assert out.min() >= lo - 1e-12 and out.max() <= hi + 1e-12


# --------------------------------------------------------------------------
# Degenerate input
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", UNIFORMITY_SCALERS)
def test_constant_feature_maps_to_the_low_end(cls):
    """Matches `MinMaxScaler`'s existing convention rather than inventing one.

    A constant column has no spread to normalize by. `MinMaxScaler` maps it to
    the low end of the range; if these disagreed, swapping the scaler in a
    pipeline would change behaviour on a degenerate column for reasons nobody
    would think to look for. #204 is the reminder that a constant feature is a
    real thing that reaches this code.
    """
    X = pd.DataFrame({"const": np.full(50, 7.0), "varying": np.arange(50.0)})
    out = cls(feature_range=(-1.0, 1.0)).fit_transform(X)
    assert np.allclose(out[:, 0], -1.0)


@pytest.mark.parametrize("cls", UNIFORMITY_SCALERS)
def test_constant_feature_round_trips_to_its_value(cls):
    """A constant column is degenerate, not unknown.

    Returning NaN from the inverse would lose a perfectly well-defined value
    and turn one awkward column into a NaN-poisoned frame downstream.

    All three classes, including the sklearn wrapper. An earlier version of
    this test excluded `QuantileUniformScaler` on the assumption that
    delegating to `QuantileTransformer` would not preserve this -- it does, via
    its own quantile table. Excluding a class from a property test because it
    is assumed to fail leaves the file asserting something for "the uniformity
    scalers" that one of them was never checked for.
    """
    X = pd.DataFrame({"const": np.full(20, 7.0), "varying": np.arange(20.0)})
    scaler = cls().fit(X)
    recovered = scaler.inverse_transform(scaler.transform(X))
    assert np.allclose(recovered[:, 0], 7.0)


@pytest.mark.parametrize("cls", [EmpiricalCDFScaler, PiecewiseLinearCDFScaler])
def test_nan_passes_through_rather_than_becoming_the_maximum(cls):
    """The specific wrong answer this guards is "NaN is the largest value".

    `np.searchsorted` places NaN at the far right of a sorted array, so the
    naive empirical CDF reports quantile 1.0 for a missing value -- a
    fabricated number rather than a missing one, and one that looks entirely
    plausible in the output. Confirmed as a live defect in #224's independent
    implementation, which returned exactly 1.0 for a NaN input.
    """
    rng = np.random.default_rng(3)
    values = rng.lognormal(0, 1, 100)
    values[[5, 50, 99]] = np.nan
    X = pd.DataFrame({"x": values})

    out = cls().fit_transform(X).ravel()
    assert np.isnan(out[[5, 50, 99]]).all()
    assert np.isfinite(np.delete(out, [5, 50, 99])).all()
    assert out[np.isfinite(out)].max() <= 1.0 + 1e-12


def test_zero_inflated_column_still_pins_its_minimum_to_the_low_end():
    """The atom at the minimum is why the rescale is not simply `(f - 1/n)`.

    On a column that is 40% zeros the empirical CDF of the minimum is 0.4, not
    1/n. Rescaling by an assumed `1/n` would leave every zero at ~0.4 and put
    the bottom 40% of the domain permanently out of reach of any membership
    function.
    """
    rng = np.random.default_rng(11)
    values = np.where(rng.random(500) < 0.4, 0.0, rng.uniform(1.0, 10.0, 500))
    out = EmpiricalCDFScaler().fit_transform(pd.DataFrame({"x": values})).ravel()
    assert np.allclose(out[values == 0.0], 0.0)
    assert out.max() == pytest.approx(1.0)


def test_piecewise_collapses_tied_breakpoints_and_reports_the_effective_count():
    """Ten pieces cannot survive an atom that swallows five of the quantiles.

    On a column that is 40% zeros the quantiles at p = 0.0, 0.1, 0.2, 0.3 and
    0.4 all land on zero. `np.interp` requires a strictly increasing `xp`, so
    tied breakpoints would leave the map ill-defined at exactly the value the
    column is mostly made of. They collapse to one, and `n_pieces_` reports the
    effective count rather than letting the caller believe the requested one.

    The exact count is not asserted -- it follows from `np.quantile`'s
    interpolation and is not a property of this class.
    """
    rng = np.random.default_rng(5)
    values = np.where(rng.random(400) < 0.4, 0.0, rng.uniform(1.0, 10.0, 400))
    X = pd.DataFrame({"zero_inflated": values})

    scaler = PiecewiseLinearCDFScaler(n_pieces=10).fit(X)
    assert 1 <= scaler.n_pieces_["zero_inflated"] < 10

    mapping = scaler.mappings_["zero_inflated"]
    assert np.all(np.diff(mapping["xs"]) > 0), "breakpoints must be strictly increasing"
    assert np.all(np.diff(mapping["ys"]) > 0), "targets must be strictly increasing"

    # The collapse keeps the highest target of each run, which left ys[0] at
    # 0.4 in the first implementation -- every zero mapped to 0.4 and the bottom
    # 40% of feature_range was unreachable by any input. The targets are
    # rescaled back onto the full span, so the atom sits at the low end where a
    # membership function can find it.
    out = scaler.transform(X).ravel()
    assert np.all(np.isfinite(out))
    assert np.allclose(out[values == 0.0], 0.0)
    assert out.max() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Parameter validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, 2.5, "10", True])
def test_bad_n_pieces_is_rejected_at_fit(bad):
    """`True` is in the list on purpose: `bool` is an `int` subclass, so
    `n_pieces=True` would otherwise run as `n_pieces=1` and quietly give plain
    min-max to someone who asked for something else."""
    with pytest.raises(ValueError):
        PiecewiseLinearCDFScaler(n_pieces=bad).fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))


@pytest.mark.parametrize("cls", UNIFORMITY_SCALERS)
@pytest.mark.parametrize("bad", [(1.0, 0.0), (1.0, 1.0), 0.5, (1.0, 2.0, 3.0)])
def test_bad_feature_range_is_rejected_at_fit(cls, bad):
    """An inverted range would silently flip every feature; a degenerate one
    would divide by zero in `inverse_transform` and return infinities."""
    with pytest.raises(ValueError):
        cls(feature_range=bad).fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))


# --------------------------------------------------------------------------
# sklearn contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", UNIFORMITY_SCALERS)
def test_clone_and_get_params_round_trip(cls):
    """`__init__` must only store its arguments, or `GridSearchCV` breaks."""
    scaler = cls(feature_range=(-1.0, 2.0))
    twin = clone(scaler)
    assert twin.get_params() == scaler.get_params()


@pytest.mark.parametrize("cls", UNIFORMITY_SCALERS)
def test_pandas_output_preserves_column_names(cls, pathological):
    """FIS consumers key membership dictionaries by feature name, so losing the
    columns on the way through a scaler is the failure that matters here."""
    scaler = cls().set_output(transform="pandas").fit(pathological)
    out = scaler.transform(pathological)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(pathological.columns)


@pytest.mark.parametrize("cls", UNIFORMITY_SCALERS)
def test_ndarray_input_is_accepted_and_named_positionally(cls, pathological):
    """Same surface as the existing scalers for un-named input."""
    scaler = cls().fit(pathological.to_numpy())
    assert list(scaler.get_feature_names_out()) == [
        f"feature_{i}" for i in range(pathological.shape[1])
    ]
    assert scaler.transform(pathological.to_numpy()).shape == pathological.shape


@pytest.mark.parametrize("cls", UNIFORMITY_SCALERS)
def test_fit_on_train_only_then_transform_test(cls, pathological):
    """The transform is fitted on the training fold and applied to the test one.

    Every number on #220 was measured that way ("scaler fit on training fold
    only"), and it is the only usage that is not leakage. Unseen values outside
    the training range must not raise.
    """
    train, test = pathological.iloc[:300], pathological.iloc[300:]
    scaler = cls().fit(train)
    out = scaler.transform(test)
    assert out.shape == test.shape
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("cls", UNIFORMITY_SCALERS)
def test_composes_into_a_pipeline_in_front_of_a_fis(cls):
    """The documented usage, end to end.

    Small and fast on purpose -- this asserts the plumbing holds (a FIS can be
    fitted and can predict through the scaler), not that accuracy improved.
    The accuracy question is `benchmarks/uniformity_scaling.py`'s, where it can
    be measured over seeds instead of asserted once.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"a": rng.lognormal(0, 1.5, 120), "b": rng.uniform(0, 1, 120)}
    )
    y = pd.Series(np.log1p(X["a"]) + 2.0 * X["b"] + rng.normal(0, 0.05, 120))

    pipe = make_pipeline(cls(), TribbleRegressor(n_gaussians=2))
    pipe.fit(X, y)
    predictions = pipe.predict(X)
    assert predictions.shape == (len(X),)
    assert np.all(np.isfinite(predictions))
