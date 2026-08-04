import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
from tribblefis import scaling
from tribblefis.scaling import StandardFuzzyScalar, UnitFuzzyScalar


def _wide_range_column(rng, n):
    """Multi-scale content: some rows near 1e-2, some near 1e5 -- a robust
    trigger for the dynamic-range log-transform heuristic regardless of which
    particular samples land near the extremes."""
    low = rng.uniform(0.01, 1.0, n // 2)
    high = rng.uniform(1e4, 1e5, n - n // 2)
    return rng.permutation(np.concatenate([low, high]))


class _SharedScalarTests:
    """Behavior every fuzzy scalar must satisfy, regardless of final
    normalization. Mixed into concrete per-class TestCase subclasses below,
    which set ``scalar_cls``."""

    scalar_cls = None

    def setUp(self):
        rng = np.random.default_rng(0)
        n = 200
        self.X = pd.DataFrame(
            {
                "wide": _wide_range_column(rng, n),
                "narrow": rng.uniform(10.0, 20.0, n),
            }
        )
        self.rng = rng

    def test_detects_wide_dynamic_range_feature(self):
        scaler = self.scalar_cls().fit(self.X)
        self.assertEqual(scaler.log_features_, ["wide"])

    def test_no_log_detection_when_disabled(self):
        scaler = self.scalar_cls(log_dynamic_range=None).fit(self.X)
        self.assertEqual(scaler.log_features_, [])

    def test_inverse_transform_round_trips(self):
        scaler = self.scalar_cls()
        Xt = scaler.fit_transform(self.X)
        Xinv = scaler.inverse_transform(Xt)
        np.testing.assert_allclose(Xinv, self.X.to_numpy(), atol=1e-6)

    def test_ndarray_input(self):
        scaler = self.scalar_cls()
        scaler.fit_transform(self.X.to_numpy())
        self.assertEqual(list(scaler.feature_names_in_), ["feature_0", "feature_1"])

    def test_get_feature_names_out(self):
        scaler = self.scalar_cls().fit(self.X)
        np.testing.assert_array_equal(
            scaler.get_feature_names_out(), np.array(["wide", "narrow"], dtype=object)
        )

    def test_sklearn_clone_and_get_params(self):
        scaler = self.scalar_cls(log_dynamic_range=2.5)
        cloned = clone(scaler)
        self.assertEqual(cloned.get_params(), scaler.get_params())
        self.assertEqual(cloned.get_params()["log_dynamic_range"], 2.5)

    # -- log_features: naming the transformed columns explicitly -------------

    def test_explicit_log_features_logs_exactly_those_columns(self):
        """The whole point of the parameter: the logged set is what was asked
        for, even when it is the column auto-detection would *not* have
        picked."""
        scaler = self.scalar_cls(log_features=["narrow"]).fit(self.X)
        self.assertEqual(scaler.log_features_, ["narrow"])
        self.assertEqual(list(scaler.log_shift_), ["narrow"])

    def test_explicit_log_features_actually_applies_the_transform(self):
        """Guards against the list being recorded but never used: the logged
        column's transformed values must differ from the un-logged fit."""
        logged = self.scalar_cls(log_features=["narrow"]).fit_transform(self.X)
        plain = self.scalar_cls(log_features=[]).fit_transform(self.X)
        narrow = self.X.columns.get_loc("narrow")
        self.assertFalse(np.allclose(logged[:, narrow], plain[:, narrow]))
        # ...and the column that was *not* named must be untouched by the change.
        wide = self.X.columns.get_loc("wide")
        np.testing.assert_allclose(logged[:, wide], plain[:, wide], atol=1e-12)

    def test_empty_log_features_logs_nothing_and_differs_from_none(self):
        """``[]`` is falsy in Python, so the obvious implementation bug is
        treating it as "unset" and silently auto-detecting instead."""
        explicit_none = self.scalar_cls(log_features=[]).fit(self.X)
        self.assertEqual(explicit_none.log_features_, [])
        self.assertEqual(explicit_none.log_shift_, {})

        auto = self.scalar_cls(log_features=None).fit(self.X)
        self.assertEqual(auto.log_features_, ["wide"])

    def test_explicit_log_features_beats_auto_detection(self):
        """Both set is not an error -- the explicit list wins outright, and
        the auto-detected column is *not* unioned in."""
        scaler = self.scalar_cls(log_dynamic_range=1.0, log_features=["narrow"]).fit(self.X)
        self.assertEqual(scaler.log_features_, ["narrow"])
        self.assertNotIn("wide", scaler.log_features_)

    def test_empty_log_features_beats_permissive_auto_threshold(self):
        """The "log nothing" case has to win over a threshold that would
        otherwise log every column."""
        scaler = self.scalar_cls(log_dynamic_range=0.0, log_features=[]).fit(self.X)
        self.assertEqual(scaler.log_features_, [])

    def test_unknown_log_feature_raises_naming_the_offender(self):
        scaler = self.scalar_cls(log_features=["wide", "nonexistent"])
        with self.assertRaises(ValueError) as ctx:
            scaler.fit(self.X)
        message = str(ctx.exception)
        self.assertIn("nonexistent", message)
        # The message should also help the caller fix the typo.
        self.assertIn("wide", message)
        self.assertIn("narrow", message)

    def test_bare_string_log_features_raises(self):
        """A string is iterable, so accepting one would silently ask for
        columns named 'n', 'a', 'r', ... -- refuse it clearly instead."""
        with self.assertRaises(TypeError) as ctx:
            self.scalar_cls(log_features="narrow").fit(self.X)
        self.assertIn("narrow", str(ctx.exception))

    def test_log_features_validated_at_fit_not_init(self):
        """sklearn requires ``__init__`` to only store parameters."""
        scaler = self.scalar_cls(log_features=["nope"])  # must not raise
        self.assertEqual(scaler.log_features, ["nope"])
        with self.assertRaises(ValueError):
            scaler.fit(self.X)

    def test_inverse_transform_round_trips_with_explicit_log_features(self):
        scaler = self.scalar_cls(log_features=["narrow"])
        Xt = scaler.fit_transform(self.X)
        Xinv = scaler.inverse_transform(Xt)
        np.testing.assert_allclose(Xinv, self.X.to_numpy(), atol=1e-6)

    def test_inverse_transform_round_trips_with_all_columns_logged(self):
        scaler = self.scalar_cls(log_features=["wide", "narrow"])
        Xt = scaler.fit_transform(self.X)
        Xinv = scaler.inverse_transform(Xt)
        np.testing.assert_allclose(Xinv, self.X.to_numpy(), atol=1e-6)

    def test_ndarray_log_features_by_position_and_by_synthetic_name(self):
        """For ndarray input, positional indices and the ``feature_N`` names
        ``_as_dataframe`` synthesises must select the same columns."""
        X = self.X.to_numpy()
        by_index = self.scalar_cls(log_features=[1]).fit(X)
        by_name = self.scalar_cls(log_features=["feature_1"]).fit(X)
        self.assertEqual(by_index.log_features_, ["feature_1"])
        self.assertEqual(by_name.log_features_, ["feature_1"])
        np.testing.assert_allclose(by_index.transform(X), by_name.transform(X))

    def test_negative_and_out_of_range_indices(self):
        X = self.X.to_numpy()
        self.assertEqual(self.scalar_cls(log_features=[-1]).fit(X).log_features_, ["feature_1"])
        with self.assertRaises(ValueError) as ctx:
            self.scalar_cls(log_features=[5]).fit(X)
        self.assertIn("5", str(ctx.exception))

    def test_duplicate_log_features_are_not_applied_twice(self):
        """log1p applied twice would silently corrupt the column, so
        duplicates collapse rather than compounding."""
        once = self.scalar_cls(log_features=["narrow"]).fit_transform(self.X)
        twice = self.scalar_cls(log_features=["narrow", "narrow"]).fit_transform(self.X)
        np.testing.assert_allclose(once, twice, atol=1e-12)

    def test_clone_and_get_params_with_log_features(self):
        scaler = self.scalar_cls(log_features=["wide"])
        cloned = clone(scaler)
        self.assertEqual(cloned.get_params(), scaler.get_params())
        self.assertEqual(cloned.get_params()["log_features"], ["wide"])
        # And the clone must actually behave the same way once fitted.
        np.testing.assert_allclose(
            cloned.fit_transform(self.X), scaler.fit_transform(self.X), atol=1e-12
        )

    def test_log_features_column_order_is_independent_of_list_order(self):
        """Naming the same set in a different order must not change results."""
        forward = self.scalar_cls(log_features=["wide", "narrow"]).fit_transform(self.X)
        reverse = self.scalar_cls(log_features=["narrow", "wide"]).fit_transform(self.X)
        np.testing.assert_allclose(forward, reverse, atol=1e-12)

    def test_log_features_survives_set_output_pandas(self):
        scaler = self.scalar_cls(log_features=["narrow"]).set_output(transform="pandas")
        Xt = scaler.fit_transform(self.X)
        self.assertIsInstance(Xt, pd.DataFrame)
        self.assertEqual(list(Xt.columns), ["wide", "narrow"])

    def test_selects_a_subset_no_threshold_can_reach(self):
        """The reason ``log_features`` has to exist at all.

        ``log_dynamic_range`` can only ever select a *prefix* of the columns
        ordered by dynamic range. This mirrors the real UCI Concrete case,
        where the wanted set ['Slag', 'FlyAsh', 'Age'] straddles an unwanted
        column ('Superplasticizer') in that ordering, so no scalar threshold
        reproduces it. Here 'mid' plays the unwanted straddler.
        """
        rng = np.random.default_rng(1)
        n = 100
        X = pd.DataFrame(
            {
                "big": rng.uniform(1.0, 1e4, n),  # ~4 decades  -- wanted
                "mid": rng.uniform(1.0, 1e2, n),  # ~2 decades  -- NOT wanted
                "small": rng.uniform(1.0, 1e1, n),  # ~1 decade -- wanted
            }
        )
        wanted = ["big", "small"]

        # No threshold reproduces `wanted`: every threshold low enough to
        # admit 'small' also admits 'mid'.
        for threshold in [0.0, 0.5, 0.9, 1.5, 2.5, 3.5, 5.0]:
            detected = self.scalar_cls(log_dynamic_range=threshold).fit(X).log_features_
            self.assertNotEqual(sorted(detected), sorted(wanted), f"threshold={threshold}")

        # The explicit list does reproduce it exactly, and round-trips.
        scaler = self.scalar_cls(log_features=wanted)
        Xt = scaler.fit_transform(X)
        self.assertEqual(sorted(scaler.log_features_), sorted(wanted))
        np.testing.assert_allclose(scaler.inverse_transform(Xt), X.to_numpy(), atol=1e-6)

    def test_log_features_in_pipeline(self):
        y = self.X["wide"] * 2 + self.X["narrow"]
        pipe = make_pipeline(
            self.scalar_cls(log_features=["wide"]), MixtureOfGaussiansFuzzyRegressor()
        )
        pipe.fit(self.X, y)
        self.assertEqual(len(pipe.predict(self.X)), len(y))

    def test_pipeline_with_classifier(self):
        y = (self.X["wide"] > np.median(self.X["wide"])).astype(int)
        pipe = make_pipeline(self.scalar_cls(), MixtureOfGaussiansFuzzyClassifier())
        pipe.fit(self.X, y)
        preds = pipe.predict(self.X)
        self.assertEqual(len(preds), len(y))

    def test_pipeline_with_regressor(self):
        y = self.X["wide"] * 2 + self.X["narrow"]
        pipe = make_pipeline(self.scalar_cls(), MixtureOfGaussiansFuzzyRegressor())
        pipe.fit(self.X, y)
        preds = pipe.predict(self.X)
        self.assertEqual(len(preds), len(y))

    def test_constant_feature_does_not_divide_by_zero(self):
        X = self.X.copy()
        X["constant"] = 5.0
        scaler = self.scalar_cls().fit(X)
        Xt = scaler.transform(X)
        self.assertTrue(np.all(np.isfinite(Xt)))


class TestUnitFuzzyScalar(_SharedScalarTests, unittest.TestCase):
    scalar_cls = UnitFuzzyScalar

    def test_output_bounded_to_unit_interval(self):
        scaler = UnitFuzzyScalar()
        Xt = scaler.fit_transform(self.X)
        self.assertGreaterEqual(Xt.min(), 0.0)
        self.assertLessEqual(Xt.max(), 1.0)
        np.testing.assert_allclose(Xt.min(axis=0), [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(Xt.max(axis=0), [1.0, 1.0], atol=1e-10)

    def test_custom_feature_range(self):
        scaler = UnitFuzzyScalar(feature_range=(-1.0, 1.0))
        Xt = scaler.fit_transform(self.X)
        self.assertGreaterEqual(Xt.min(), -1.0)
        self.assertLessEqual(Xt.max(), 1.0)

    def test_clips_out_of_range_values_at_transform_time(self):
        scaler = UnitFuzzyScalar().fit(self.X)
        X_test = self.X.copy()
        X_test.iloc[0, X_test.columns.get_loc("narrow")] = 1e9
        Xt = scaler.transform(X_test)
        self.assertLessEqual(Xt.max(), 1.0)

    def test_inverse_transform_round_trip_warns_with_clipped_out_of_range_data(self):
        """When clip=True (the default), out-of-range values are silently
        clipped at transform time, breaking the round-trip. inverse_transform
        should warn when it detects potential clipped artifacts (values sitting
        exactly on the bounds)."""
        scaler = UnitFuzzyScalar(clip=True).fit(self.X)
        X_test = self.X.copy()
        # Set one value way outside the fitted range.
        X_test.iloc[0, X_test.columns.get_loc("narrow")] = 1e9
        Xt = scaler.transform(X_test)
        # The transformed value should be clipped to 1.0 (the upper bound).
        self.assertEqual(Xt[0, X_test.columns.get_loc("narrow")], 1.0)

        # inverse_transform should warn about potential clipped artifacts.
        with self.assertWarns(UserWarning):
            Xinv = scaler.inverse_transform(Xt)

        # The round-trip will not recover the original value; it will be wrong.
        # This is the core issue: inverse_transform(transform(X)) != X
        self.assertNotAlmostEqual(Xinv[0, X_test.columns.get_loc("narrow")], 1e9)


class TestStandardFuzzyScalar(_SharedScalarTests, unittest.TestCase):
    scalar_cls = StandardFuzzyScalar

    def test_output_has_zero_mean_unit_variance(self):
        scaler = StandardFuzzyScalar()
        Xt = scaler.fit_transform(self.X)
        np.testing.assert_allclose(Xt.mean(axis=0), [0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(Xt.std(axis=0), [1.0, 1.0], atol=1e-8)

    def test_not_bounded_to_unit_interval(self):
        # Sanity check that this is genuinely z-score, not min-max in disguise:
        # a value several sigma out should transform well outside [0, 1].
        scaler = StandardFuzzyScalar().fit(self.X)
        X_test = self.X.copy()
        X_test.iloc[0, X_test.columns.get_loc("narrow")] = (
            self.X["narrow"].mean() + 10 * self.X["narrow"].std()
        )
        Xt = scaler.transform(X_test)
        self.assertGreater(Xt[0, X_test.columns.get_loc("narrow")], 1.0)


class TestBackwardsCompatibleAliases(unittest.TestCase):
    """The ``*FuzzyScalar`` names are canonical, but the shorter names shipped
    first and are imported across the ``grad-school`` workspace
    (``reproduce/tables/_fuzzy_models.py``,
    ``reproduce/tables/table_hyperparam_normalization.py``, and nine
    ``FuzzySystemsExperiments/*.py`` scripts). Breaking them is the concrete
    regression this test exists to catch."""

    def test_aliases_are_the_same_class_objects(self):
        self.assertIs(scaling.UnitScalar, scaling.UnitFuzzyScalar)
        self.assertIs(scaling.StandardScalar, scaling.StandardFuzzyScalar)

    def test_old_import_form_still_works(self):
        from tribblefis.scaling import StandardScalar, UnitScalar

        self.assertIs(UnitScalar, UnitFuzzyScalar)
        self.assertIs(StandardScalar, StandardFuzzyScalar)

    def test_instances_of_alias_are_instances_of_canonical(self):
        """Aliases are bindings, not subclasses, so ``isinstance`` must agree
        in both directions -- downstream code type-checks on these."""
        self.assertIsInstance(scaling.UnitScalar(), UnitFuzzyScalar)
        self.assertIsInstance(UnitFuzzyScalar(), scaling.UnitScalar)
        self.assertIsInstance(scaling.StandardScalar(), StandardFuzzyScalar)
        self.assertIsInstance(StandardFuzzyScalar(), scaling.StandardScalar)

    def test_alias_and_canonical_behave_identically(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            {"wide": _wide_range_column(rng, 100), "narrow": rng.uniform(10.0, 20.0, 100)}
        )
        for alias, canonical in [
            (scaling.UnitScalar, UnitFuzzyScalar),
            (scaling.StandardScalar, StandardFuzzyScalar),
        ]:
            with self.subTest(canonical=canonical.__name__):
                np.testing.assert_allclose(
                    alias(log_features=["wide"]).fit_transform(X),
                    canonical(log_features=["wide"]).fit_transform(X),
                    atol=1e-12,
                )

    def test_no_alias_denotes_the_wrong_transform(self):
        """The defect this naming exists to prevent: a symbol whose name says
        "standard" must not compute min-max, and vice versa. Checked
        behaviourally, through every public name the module exports."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.uniform(1.0, 50.0, 200), "b": rng.uniform(-30.0, 5.0, 200)})

        for name in ["UnitScalar", "UnitFuzzyScalar"]:
            with self.subTest(name=name):
                Xt = getattr(scaling, name)(log_features=[]).fit_transform(X)
                # Min-max: every column lands exactly on [0, 1].
                np.testing.assert_allclose(Xt.min(axis=0), [0.0, 0.0], atol=1e-10)
                np.testing.assert_allclose(Xt.max(axis=0), [1.0, 1.0], atol=1e-10)

        for name in ["StandardScalar", "StandardFuzzyScalar"]:
            with self.subTest(name=name):
                Xt = getattr(scaling, name)(log_features=[]).fit_transform(X)
                # z-score: zero mean, unit sigma -- and NOT bounded to [0, 1].
                np.testing.assert_allclose(Xt.mean(axis=0), [0.0, 0.0], atol=1e-8)
                np.testing.assert_allclose(Xt.std(axis=0), [1.0, 1.0], atol=1e-8)
                self.assertLess(Xt.min(), 0.0)

    def test_standard_scalar_docstring_warns_against_fis_use(self):
        """The honest name is still the one reached for from memory, so the
        docstring carries the guardrail. Assert it is actually there."""
        doc = StandardFuzzyScalar.__doc__
        self.assertIn("not the recommended default", doc)
        self.assertIn("UnitFuzzyScalar", doc)
        self.assertIn("0.646", doc)  # the raw-features baseline it falls below


if __name__ == "__main__":
    unittest.main()
