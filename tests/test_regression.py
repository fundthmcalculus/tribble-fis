import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    create_gaussian_membership_dict,
    take_top_features,
    tsk_firing_strengths,
)
from tribblefis.regression import (
    partition_output,
    optimize_tsk_coefficients,
    _rsquared,
)



def generate_synthetic_data(n_samples: int, x_range: tuple, y_range: tuple, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data for Z = X / (Y^2 + 1)."""
    np.random.seed(seed)
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    y = np.random.uniform(y_range[0], y_range[1], n_samples)
    z = x / (y**2 + 1)
    return x, y, z


def plot_function_and_error(x_test, y_test, z_test, z_pred, save_path: str = None):
    """Plot actual function, predictions, and error."""
    fig = plt.figure(figsize=(5, 15))

    # Create a fine grid for the surface plot
    x_grid = np.linspace(x_test.min(), x_test.max(), 30)
    y_grid = np.linspace(y_test.min(), y_test.max(), 30)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    Z_actual = X_grid / (Y_grid**2 + 1)

    # Plot 1: Actual function
    ax1 = fig.add_subplot(311, projection='3d')
    ax1.scatter(x_test, y_test, z_test, c='blue', marker='o', s=20, alpha=0.7, label='Actual')
    ax1.plot_surface(X_grid, Y_grid, Z_actual, alpha=0.4, cmap='viridis')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Actual Function: Z = X / (Y² + 1)')
    ax1.legend()

    # Plot 2: Model predictions
    ax2 = fig.add_subplot(312, projection='3d')
    ax2.scatter(x_test, y_test, z_pred, c='red', marker='o', s=20, alpha=0.7, label='Predicted')
    ax2.plot_surface(X_grid, Y_grid, Z_actual, alpha=0.4, cmap='viridis')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Model Predictions')
    ax2.legend()

    # Plot 3: Prediction error
    error = z_test - z_pred
    ax3 = fig.add_subplot(313)
    scatter = ax3.scatter(range(len(error)), error, c=np.abs(error), cmap='RdYlGn_r', s=20, alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax3.set_xlabel('Test Sample Index')
    ax3.set_ylabel('Error (Actual - Predicted)')
    ax3.set_title(f'Prediction Error\nMAE: {np.mean(np.abs(error)):.6f}')
    plt.colorbar(scatter, ax=ax3, label='|Error|')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def test_gaussian_mixture_regression_2d():
    """Test GaussianMixtureRegression on Z = X / (Y^2 + 1)."""
    # Generate training data in [0,1]x[0,1]
    x_train, y_train, z_train = generate_synthetic_data(
        n_samples=200,
        x_range=(0, 1),
        y_range=(0, 1),
        seed=42
    )

    # Generate test data in [1,2]x[3,4]
    x_test, y_test, z_test = generate_synthetic_data(
        n_samples=100,
        x_range=(1, 2),
        y_range=(3, 4),
        seed=43
    )

    # Prepare training data as DataFrame
    X_train = pd.DataFrame({'x': x_train, 'y': y_train})
    y_train_series = pd.Series(z_train, name='y_value')

    X_test = pd.DataFrame({'x': x_test, 'y': y_test})
    y_test_series = pd.Series(z_test, name='y_value')

    # Partition output into buckets
    n_output_buckets = 5
    y_train_bucketed, y_bucket_mean = partition_output(n_output_buckets, y_train_series)

    # Calculate Gaussian correlations
    feature_differentiators = calculate_gaussian_correlation(X_train, y_train_bucketed["y_bucket"])

    # Take top features
    top_n, top_n_todo = take_top_features(feature_differentiators, top_n=-1)

    # Create Gaussian membership model
    gaussian_memberships = create_gaussian_membership_dict(
        X_train,
        y_train_bucketed["y_bucket"],
        top_n_var_names=top_n_todo,
        n_gaussians=-1
    )

    # Optimize TSK coefficients for different orders
    orders = ["0th", "1st", "2nd"]
    y_test_preds = []
    metrics = []

    for order in orders:
        corr_terms, y_bucket_mean_opt = optimize_tsk_coefficients(
            X_train,
            gaussian_memberships,
            top_n_todo,
            y_bucket_mean,
            y_train_bucketed,
            n_output_buckets,
            order=order
        )

        # Predict on test set
        firing_strengths_test, _ = tsk_firing_strengths(X_test[top_n_todo], gaussian_memberships)
        norm_firing_strength_test = firing_strengths_test / firing_strengths_test.sum(axis=1)[:, np.newaxis]

        y_test_pred = np.zeros(len(X_test))
        X_test_array = X_test[top_n_todo].to_numpy()

        n_terms = corr_terms.shape[1]

        for i in range(len(X_test)):
            pred = 0
            for rule_id in gaussian_memberships.rule_ids:
                pred += norm_firing_strength_test[i, rule_id] * y_bucket_mean_opt[rule_id]

                if order == "1st":
                    pred += norm_firing_strength_test[i, rule_id] * np.dot(corr_terms[rule_id], X_test_array[i])
                elif order == "2nd":
                    feat = np.concatenate([X_test_array[i], X_test_array[i]**2])
                    pred += norm_firing_strength_test[i, rule_id] * np.dot(corr_terms[rule_id], feat)

            y_test_pred[i] = pred

        y_test_preds.append(y_test_pred)

        # Calculate metrics
        rmse = np.sqrt(np.mean((y_test_series - y_test_pred)**2))
        mae = np.mean(np.abs(y_test_series - y_test_pred))
        r2 = _rsquared(y_test_series, y_test_pred)

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
    best_idx = 2  # 2nd order
    plot_function_and_error(x_test, y_test, z_test, y_test_preds[best_idx])

    # Verify test ran without errors
    best_r2 = metrics[best_idx]['r2']
    assert not np.isnan(best_r2), "R² should not be NaN"
    assert len(y_test_preds[best_idx]) == len(z_test), "Predictions should have same length as test data"

if __name__ == "__main__":
    test_gaussian_mixture_regression_2d()