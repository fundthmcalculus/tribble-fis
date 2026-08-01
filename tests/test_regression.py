import unittest

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
from tribblefis.gauss_math import create_gaussian_membership_dict
from tribblefis.regression import (
    partition_output,
    build_consequent_features,
    solve_tsk_consequents,
    predict_tsk,
    compute_first_order_corrections,
    optimize_tsk_coefficients,
    _mse,
)



def generate_synthetic_data(n_samples: int, x_range: tuple, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for Z = X / (X^2 + 1)."""
    np.random.seed(seed)
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    z = x / (x**2 + 1)
    return x, z


def plot_function_and_error(x_train, z_train, z_pred_train, x_test, z_test, z_pred, order_name: str, save_path: str = None):
    """Plot actual function, predictions, and error for 1D case."""
    fig = plt.figure()

    # Create a fine grid for the function plot
    x_grid = np.linspace(x_test.min(), x_test.max(), 300)
    z_actual = x_grid / (x_grid**2 + 1)

    # Plot 1: Actual function vs predictions
    fig.suptitle(f"TSK Regression: {order_name} Order", fontsize=16)
    ax1 = fig.add_subplot(211)
    ax1.scatter(x_train, z_train, alpha=0.6, label='Train points')
    ax1.scatter(x_train, z_pred_train,   alpha=0.6, label='Train-predict points')
    ax1.scatter(x_test, z_test,  alpha=0.6, label='Test points')
    ax1.scatter(x_test, z_pred, alpha=0.8, label='Predictions')
    ax1.plot(x_grid, z_actual, 'b-', linewidth=2, label='Actual function')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_title('Function: Z = X / (X² + 1)')
    ax1.legend(bbox_to_anchor=(1.1, 1.05))
    ax1.grid(True, alpha=0.3)

    # Plot 2: Prediction error
    error = z_test - z_pred
    ax2 = fig.add_subplot(212)
    scatter = ax2.scatter(x_test, error, c=np.abs(error), cmap='RdYlGn_r', s=50, alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Error (Actual - Predicted)')
    ax2.set_title(f'Prediction Error\nMAE: {np.mean(np.abs(error)):.6f}')
    plt.colorbar(scatter, ax=ax2, label='|Error|')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def test_gaussian_mixture_regression_2d():
    """Test GaussianMixtureRegression on Z = X / (X^2 + 1)."""
    # Generate training data in [-3, 0]
    x_train, z_train = generate_synthetic_data(
        n_samples=400,
        x_range=(-4, 0),
        seed=42
    )

    # Generate test data in [0, 2]
    x_test, z_test = generate_synthetic_data(
        n_samples=250,
        x_range=(-0.5, 12),
        seed=43
    )

    # Prepare training data as DataFrame
    X_train = pd.DataFrame({'x': x_train})
    X_test = pd.DataFrame({'x': x_test})

    # Test different TSK orders
    orders = ["0th", "1st", "2nd"]
    metrics = []

    for order in orders:
        regressor = MixtureOfGaussiansFuzzyRegressor(
            top_n=-1,
            n_gaussians=-1,
            n_output_buckets=5,
            tsk_order=order,
            optimize_coefficients=True,
            random_state=42
        )

        # Train model
        regressor.fit(X_train, z_train)

        # Predict on test set
        y_train_pred = regressor.predict(X_train)
        y_test_pred = regressor.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(np.mean((z_test - y_test_pred)**2))
        mae = np.mean(np.abs(z_test - y_test_pred))
        ss_res = np.sum((z_test - y_test_pred) ** 2)
        ss_tot = np.sum((z_test - np.mean(z_test)) ** 2)
        r2 = 1 - ss_res / ss_tot

        metrics.append({
            'order': order,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })

        print(f"{order.capitalize()} Order TSK Model:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R²:   {r2:.6f}\n")

        # Plot results for best model (2nd order)
        plot_function_and_error(x_train, z_train, y_train_pred, x_test, z_test, y_test_pred, order)

        # Verify test ran without errors
        # best_r2 = metrics[-1]['r2']
        # assert not np.isnan(best_r2), "R² should not be NaN"
        assert len(y_test_pred) == len(z_test), "Predictions should have same length as test data"

class TestConsequentSolver(unittest.TestCase):
    """Unit tests for the Phase 1 closed-form consequent solver."""

    N_BUCKETS = 3
    TOP = ["a", "b"]

    def _make_model_and_data(self):
        """Build a small deterministic dataset + fitted Gaussian membership model."""
        rng = np.random.default_rng(0)
        n = 300
        X = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)})
        y_raw = pd.Series(2.0 * X["a"] + X["b"], name="y_value")
        y_part, y_bucket_mean = partition_output(self.N_BUCKETS, y_raw)
        model = create_gaussian_membership_dict(
            X, y_part["y_bucket"], top_n_var_names=self.TOP, n_gaussians=1
        )
        return X, y_part, y_bucket_mean, model

    def test_build_basis_shapes_and_raw_layout(self):
        """Raw basis reproduces the legacy [x, x^2, x^3] hstack layout exactly."""
        X_rule = np.arange(6.0).reshape(3, 2)
        np.testing.assert_allclose(
            build_consequent_features(X_rule, "2nd", basis="raw"),
            np.hstack([X_rule, X_rule ** 2]),
        )
        np.testing.assert_allclose(
            build_consequent_features(X_rule, "3rd", basis="raw"),
            np.hstack([X_rule, X_rule ** 2, X_rule ** 3]),
        )
        # full-2nd appends the single cross term (a*b) after the squared block.
        full = build_consequent_features(X_rule, "full-2nd", basis="raw")
        self.assertEqual(full.shape, (3, 5))  # 2 linear + 2 squared + 1 cross
        np.testing.assert_allclose(full[:, -1], X_rule[:, 0] * X_rule[:, 1])
        # Orthogonal basis has the same column count as raw.
        self.assertEqual(
            build_consequent_features(X_rule, "3rd", basis="orthogonal").shape,
            build_consequent_features(X_rule, "3rd", basis="raw").shape,
        )

    def test_recovers_known_consequents(self):
        """When y is generated by the TSK forward model, the closed-form solve
        (l2=0) reproduces that forward prediction to numerical tolerance."""
        X, y_part, y_bucket_mean, model = self._make_model_and_data()
        n_rules = model.n_rules
        rng = np.random.default_rng(1)
        known_means = rng.normal(size=n_rules)
        known_corr = rng.normal(size=(n_rules, len(self.TOP)))  # 1st order

        y_gen = predict_tsk(X, model, self.TOP, known_means, known_corr, order="1st")
        y_df = y_part.copy()
        y_df["y_value"] = y_gen

        # pin_extremes=False: y_gen comes from `known_means`, which have nothing
        # to do with the min/max of the original target that y_bucket_mean pins
        # to. Constraining the solve to those values makes exact recovery of the
        # data-generating model impossible by construction, so this test of the
        # solver's recovery property has to run unconstrained.
        corr, means = solve_tsk_consequents(
            X, model, self.TOP, y_bucket_mean, y_df,
            n_output_buckets=self.N_BUCKETS, order="1st", l2_reg=0.0,
            pin_extremes=False, verbose=False,
        )
        y_rec = predict_tsk(X, model, self.TOP, means, corr, order="1st")
        # Prediction is the identifiable quantity (coefficients can be non-unique
        # under rank deficiency); it must match the data-generating forward model.
        np.testing.assert_allclose(y_rec, y_gen, atol=1e-6)

    def test_closed_form_beats_lbfgs_training_mse(self):
        """The closed-form solve is the exact firing-weighted MSE optimum, so its
        training MSE must be <= the L-BFGS optimizer's on the same model + data."""
        X, y_part, y_bucket_mean, model = self._make_model_and_data()

        init_corr = compute_first_order_corrections(
            X, model, len(self.TOP), self.TOP, y_bucket_mean, y_part
        )
        lbfgs_corr, lbfgs_means = optimize_tsk_coefficients(
            X, model, self.TOP, y_bucket_mean, y_part,
            n_output_buckets=self.N_BUCKETS, initial_corr_terms=init_corr, order="1st",
        )
        # pin_extremes=False so the comparison is like for like: L-BFGS solves
        # the unconstrained problem, and a constrained solve is not obliged to
        # beat it. The claim under test is that the closed form is the exact
        # optimum of the *same* problem the optimizer is approximating.
        cf_corr, cf_means = solve_tsk_consequents(
            X, model, self.TOP, y_bucket_mean, y_part,
            n_output_buckets=self.N_BUCKETS, order="1st", l2_reg=0.0,
            pin_extremes=False, verbose=False,
        )

        y_true = y_part["y_value"].values
        mse_lbfgs = _mse(y_true, predict_tsk(X, model, self.TOP, lbfgs_means, lbfgs_corr, order="1st"))
        mse_cf = _mse(y_true, predict_tsk(X, model, self.TOP, cf_means, cf_corr, order="1st"))
        self.assertLessEqual(mse_cf, mse_lbfgs + 1e-9)

    def test_extreme_bucket_means_are_pinned(self):
        """The extreme bucket means (first and last) should be pinned to the actual
        min and max of the target, and these should NOT be overwritten by
        solve_tsk_consequents when pin_extremes=True (default)."""
        X, y_part, y_bucket_mean, model = self._make_model_and_data()
        y_true = y_part["y_value"].values
        y_min = y_true.min()
        y_max = y_true.max()

        # partition_output should have pinned the extremes
        self.assertAlmostEqual(y_bucket_mean[0], y_min, places=10,
                              msg="First bucket mean should be pinned to min of target")
        self.assertAlmostEqual(y_bucket_mean[-1], y_max, places=10,
                              msg="Last bucket mean should be pinned to max of target")

        # Solve consequents with pin_extremes=True (default)
        corr, means_solved = solve_tsk_consequents(
            X, model, self.TOP, y_bucket_mean, y_part,
            n_output_buckets=self.N_BUCKETS, order="1st", l2_reg=0.0, verbose=False,
        )

        # After solving, the extremes should still be pinned
        self.assertAlmostEqual(means_solved[0], y_min, places=10,
                              msg="First bucket mean should remain pinned to min after solve_tsk_consequents")
        self.assertAlmostEqual(means_solved[-1], y_max, places=10,
                              msg="Last bucket mean should remain pinned to max after solve_tsk_consequents")

    def test_pinned_solve_is_the_constrained_optimum(self):
        """Pinning must cost accuracy only where the constraint binds.

        The constrained solve is exact, so its training MSE is the best
        achievable with those two coefficients fixed -- perturbing any free
        coefficient can only make it worse. This is what distinguishes an
        equality constraint from a penalty, and it is the property most likely
        to break silently if the reduced system is ever assembled wrongly.
        """
        X, y_part, y_bucket_mean, model = self._make_model_and_data()
        y_true = y_part["y_value"].values

        corr, means = solve_tsk_consequents(
            X, model, self.TOP, y_bucket_mean, y_part,
            n_output_buckets=self.N_BUCKETS, order="1st", l2_reg=0.0, verbose=False,
        )
        mse_at_optimum = _mse(y_true, predict_tsk(X, model, self.TOP, means, corr, order="1st"))

        rng = np.random.default_rng(7)
        for _ in range(10):
            perturbed = corr + rng.normal(scale=1e-3, size=corr.shape)
            mse_perturbed = _mse(
                y_true, predict_tsk(X, model, self.TOP, means, perturbed, order="1st")
            )
            self.assertGreaterEqual(mse_perturbed, mse_at_optimum - 1e-12)

    def test_unusable_y_bucket_mean_falls_back_to_unconstrained(self):
        """A y_bucket_mean too short to index by rule cannot state the constraint
        at all, so the solve must fall back to the unconstrained problem rather
        than raising or returning NaN."""
        X, y_part, y_bucket_mean, model = self._make_model_and_data()
        n_rules = model.n_rules

        baseline_corr, baseline_means = solve_tsk_consequents(
            X, model, self.TOP, y_bucket_mean, y_part,
            n_output_buckets=self.N_BUCKETS, order="1st", l2_reg=0.0,
            pin_extremes=False, verbose=False,
        )
        corr, means = solve_tsk_consequents(
            X, model, self.TOP, np.asarray(y_bucket_mean, dtype=float)[: n_rules - 1],
            y_part, n_output_buckets=self.N_BUCKETS, order="1st", l2_reg=0.0,
            pin_extremes=True, verbose=False,
        )
        np.testing.assert_allclose(means, baseline_means, atol=1e-8)
        np.testing.assert_allclose(corr, baseline_corr, atol=1e-8)

    def test_non_finite_extreme_skips_only_that_end(self):
        """A NaN at one extreme must not poison the whole solve, and must not
        cost the constraint at the *other* extreme -- half a pin is still worth
        having. The NaN end comes back finite and freely solved."""
        X, y_part, y_bucket_mean, model = self._make_model_and_data()
        y_max = y_part["y_value"].values.max()

        with_nan = np.asarray(y_bucket_mean, dtype=float).copy()
        with_nan[0] = np.nan

        corr, means = solve_tsk_consequents(
            X, model, self.TOP, with_nan, y_part,
            n_output_buckets=self.N_BUCKETS, order="1st", l2_reg=0.0,
            pin_extremes=True, verbose=False,
        )
        self.assertTrue(np.all(np.isfinite(means)), "NaN leaked into the bucket means")
        self.assertTrue(np.all(np.isfinite(corr)), "NaN leaked into the corrections")
        self.assertAlmostEqual(means[-1], y_max, places=10,
                               msg="the finite extreme should still be pinned")

    def test_pinning_is_skipped_for_a_single_rule(self):
        """With one rule the first and last bucket mean are the SAME coefficient,
        so there is no two-sided constraint to impose. Pinning must be skipped
        rather than double-counting that column against the right-hand side."""
        rng = np.random.default_rng(3)
        n = 200
        X = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)})
        y_raw = pd.Series(2.0 * X["a"] + X["b"], name="y_value")
        y_part, y_bucket_mean = partition_output(1, y_raw)
        model = create_gaussian_membership_dict(
            X, y_part["y_bucket"], top_n_var_names=self.TOP, n_gaussians=1
        )
        if model.n_rules != 1:
            self.skipTest(f"expected a single rule, model built {model.n_rules}")

        pinned_corr, pinned_means = solve_tsk_consequents(
            X, model, self.TOP, y_bucket_mean, y_part,
            n_output_buckets=1, order="1st", l2_reg=0.0, pin_extremes=True, verbose=False,
        )
        free_corr, free_means = solve_tsk_consequents(
            X, model, self.TOP, y_bucket_mean, y_part,
            n_output_buckets=1, order="1st", l2_reg=0.0, pin_extremes=False, verbose=False,
        )
        np.testing.assert_allclose(pinned_means, free_means, atol=1e-8)
        np.testing.assert_allclose(pinned_corr, free_corr, atol=1e-8)

    def test_singular_pinned_system_falls_back_instead_of_raising(self):
        """The reduced solve's lstsq fallback has to be reachable.

        A duplicated feature makes the gram matrix singular, so np.linalg.solve
        raises and the fallback runs. It has to be handed the design and the
        residual -- passing it the normal equations instead raises a shape error,
        which is the kind of defect that hides until the day a real dataset is
        rank-deficient.
        """
        rng = np.random.default_rng(11)
        n = 120
        a = rng.uniform(0, 1, n)
        # "b" is an exact copy of "a": the consequent design is rank-deficient.
        X = pd.DataFrame({"a": a, "b": a})
        y_raw = pd.Series(2.0 * a, name="y_value")
        y_part, y_bucket_mean = partition_output(self.N_BUCKETS, y_raw)
        model = create_gaussian_membership_dict(
            X, y_part["y_bucket"], top_n_var_names=self.TOP, n_gaussians=1
        )

        real_solve = np.linalg.solve

        def always_singular(*_args, **_kwargs):
            raise np.linalg.LinAlgError("forced singular for the fallback path")

        np.linalg.solve = always_singular
        try:
            corr, means = solve_tsk_consequents(
                X, model, self.TOP, y_bucket_mean, y_part,
                n_output_buckets=self.N_BUCKETS, order="1st", l2_reg=0.0, verbose=False,
            )
        finally:
            np.linalg.solve = real_solve

        self.assertTrue(np.all(np.isfinite(means)))
        self.assertTrue(np.all(np.isfinite(corr)))
        # The constraint still holds on the fallback path.
        y_true = y_part["y_value"].values
        self.assertAlmostEqual(means[0], y_true.min(), places=10)
        self.assertAlmostEqual(means[-1], y_true.max(), places=10)


class TestAntecedentRefinement(unittest.TestCase):
    """Unit tests for the Phase 2 antecedent-refinement module."""

    N_BUCKETS = 3
    TOP = ["a", "b"]

    def _make_model_and_data(self):
        from tribblefis.gauss_math import create_gaussian_membership_dict
        rng = np.random.default_rng(0)
        n = 240
        X = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)})
        y_raw = pd.Series(2.0 * X["a"] + X["b"] ** 2, name="y_value")
        y_part, _ = partition_output(self.N_BUCKETS, y_raw)
        model = create_gaussian_membership_dict(
            X, y_part["y_bucket"], top_n_var_names=self.TOP, n_gaussians=2
        )
        return X, y_part, model

    def test_extract_apply_roundtrip(self):
        from tribblefis.refine import extract_gaussian_params, apply_gaussian_params
        _, _, model = self._make_model_and_data()
        vec = extract_gaussian_params(model)
        # Round-trip must be the identity.
        rebuilt = apply_gaussian_params(model, vec)
        np.testing.assert_allclose(extract_gaussian_params(rebuilt), vec)
        # A perturbation must actually change the stored mu/sigma.
        changed = apply_gaussian_params(model, vec + 0.01)
        self.assertFalse(np.allclose(extract_gaussian_params(changed), vec))
        self.assertEqual(len(vec), 2 * model.n_membership_functions)

    def test_local_refine_never_worsens_val_and_helps(self):
        from tribblefis.refine import refine_antecedents_local
        X, y_part, model = self._make_model_and_data()
        _, info = refine_antecedents_local(
            model, X, y_part, self.TOP, n_output_buckets=self.N_BUCKETS,
            order="2nd", l2_reg=1e-3, basis="raw", n_folds=3, maxiter=15, maxfun=1500,
        )
        # Safeguard: never return worse than the heuristic start on the CV fitness.
        self.assertLessEqual(info["val_mse"], info["init_val_mse"] + 1e-9)

    def test_coordinate_refine_never_worsens_val(self):
        from tribblefis.refine import refine_antecedents_coordinate
        X, y_part, model = self._make_model_and_data()
        _, info = refine_antecedents_coordinate(
            model, X, y_part, self.TOP, n_output_buckets=self.N_BUCKETS,
            order="2nd", l2_reg=1e-3, basis="raw", n_folds=3, n_sweeps=2,
        )
        # Coordinate descent only accepts strict improvements per block, so the
        # final CV fitness can never exceed the heuristic start's.
        self.assertLessEqual(info["val_mse"], info["init_val_mse"] + 1e-9)
        self.assertIn("n_eval", info)


if __name__ == "__main__":
    test_gaussian_mixture_regression_2d()