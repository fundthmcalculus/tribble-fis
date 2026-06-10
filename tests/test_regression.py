import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor



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

if __name__ == "__main__":
    test_gaussian_mixture_regression_2d()