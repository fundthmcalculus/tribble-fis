import numpy as np
import matplotlib.pyplot as plt
from tribblefis.optimizer_utils import optimizers_sub_solve


def gauss(x):
    return np.exp(-(x**2) / 2)


def triangular(x, x1):
    x = abs(x)
    return max(0, 1 - (x / x1))


def trapezoid(x, x1, x2):
    x = abs(x)
    if x > x2:
        return 0
    if x > x1:
        return 1 - (x - x1) / (x2 - x1)
    return 1


def pentagonal(x, x1, x2, y):
    # Symmetric
    x = abs(x)
    if x > x2:
        return 0
    if x > x1:
        return y * (1 - (x - x1) / (x2 - x1))
    return 1 - (1 - y) * abs(x / x1)


def main():
    # We only consider the right-half gaussian for simplicity
    # Use standard-normal distribution for simplicity as well.
    x = np.linspace(0, 5, 5000)

    # Scalar error functions for optimization
    tri_err_scalar = lambda params: np.sum((gauss(x) - np.array([triangular(xi, params[0]) for xi in x])) ** 2)
    trap_err_scalar = lambda params: np.sum(
        (gauss(x) - np.array([trapezoid(xi, params[0], params[1]) for xi in x])) ** 2
    )
    pen_err_scalar = lambda params: np.sum(
        (gauss(x) - np.array([pentagonal(xi, params[0], params[1], params[2]) for xi in x])) ** 2
    )

    # Optimize triangular
    print("Optimizing triangular shape...")
    result_tri = optimizers_sub_solve(tri_err_scalar, [1.0], [(0.1, 10.0)])
    print(f"Triangular optimal params: z1={result_tri.x[0]:.4f}, error={result_tri.fun:.6f}")

    # Optimize trapezoid
    print("Optimizing trapezoid shape...")
    result_trap = optimizers_sub_solve(trap_err_scalar, [0.5, 2.0], [(0.1, 5.0), (0.1, 10.0)])
    print(
        f"Trapezoid optimal params: z1={result_trap.x[0]:.4f}, z2={result_trap.x[1]:.4f}, error={result_trap.fun:.6f}"
    )

    # Optimize pentagonal
    print("Optimizing pentagonal shape...")
    result_pen = optimizers_sub_solve(
        pen_err_scalar, [0.5, 2.0, 0.6], [(0.1, 5.0), (0.1, 10.0), (0.1, 1.0)]
    )
    print(
        f"Pentagonal optimal params: z1={result_pen.x[0]:.4f}, z2={result_pen.x[1]:.4f}, y={result_pen.x[2]:.4f}, error={result_pen.fun:.6f}"
    )

    # Plot error surfaces for triangular
    z1_range = np.linspace(0.1, 5, 100)
    tri_errors = [tri_err_scalar([z1]) for z1 in z1_range]

    plt.figure(figsize=(4, 12))
    plt.subplot(3, 1, 1)
    plt.plot(z1_range, tri_errors)
    plt.axvline(result_tri.x[0], color="r", linestyle="--", label=f"Optimal z1={result_tri.x[0]:.4f}")
    plt.xlabel("z1")
    plt.ylabel("Error")
    plt.title("Triangular Error vs z1")
    plt.legend()
    plt.grid(True)

    # Plot error surfaces for trapezoid
    z1_range_trap = np.linspace(0.1, 3, 50)
    z2_range_trap = np.linspace(0.1, 5, 50)
    Z1, Z2 = np.meshgrid(z1_range_trap, z2_range_trap)
    trap_errors = np.zeros_like(Z1)
    for i in range(len(z2_range_trap)):
        for j in range(len(z1_range_trap)):
            if Z1[i, j] < Z2[i, j]:
                trap_errors[i, j] = trap_err_scalar([Z1[i, j], Z2[i, j]])
            else:
                trap_errors[i, j] = np.nan

    plt.subplot(3, 1, 2)
    contour = plt.contourf(Z1, Z2, trap_errors, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Error")
    plt.plot(
        result_trap.x[0],
        result_trap.x[1],
        "r*",
        markersize=15,
        label=f"Optimal ({result_trap.x[0]:.2f}, {result_trap.x[1]:.2f})",
    )
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Trapezoid Error Surface")
    plt.legend()

    # Plot error surfaces for pentagonal (fixing y at optimal, varying z1 and z2)
    z1_range_pen = np.linspace(0.1, 3, 50)
    z2_range_pen = np.linspace(0.1, 5, 50)
    Z1_pen, Z2_pen = np.meshgrid(z1_range_pen, z2_range_pen)
    pen_errors = np.zeros_like(Z1_pen)
    for i in range(len(z2_range_pen)):
        for j in range(len(z1_range_pen)):
            if Z1_pen[i, j] < Z2_pen[i, j]:
                pen_errors[i, j] = pen_err_scalar([Z1_pen[i, j], Z2_pen[i, j], result_pen.x[2]])
            else:
                pen_errors[i, j] = np.nan

    plt.subplot(3, 1, 3)
    contour = plt.contourf(Z1_pen, Z2_pen, pen_errors, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Error")
    plt.plot(
        result_pen.x[0],
        result_pen.x[1],
        "r*",
        markersize=15,
        label=f"Optimal ({result_pen.x[0]:.2f}, {result_pen.x[1]:.2f})",
    )
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title(f"Pentagonal Error Surface (y={result_pen.x[2]:.2f})")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Create overlay plot comparing all shapes
    plt.figure()

    # Plot gaussian
    plt.plot(x, gauss(x), "k-", linewidth=2, label="Gaussian")

    # Plot optimized triangular
    tri_values = np.array([triangular(xi, result_tri.x[0]) for xi in x])
    plt.plot(x, tri_values, "b--", linewidth=2, label=f"Triangular (z1={result_tri.x[0]:.2f})")

    # Plot optimized trapezoidal
    trap_values = np.array([trapezoid(xi, result_trap.x[0], result_trap.x[1]) for xi in x])
    plt.plot(
        x, trap_values, "g--", linewidth=2, label=f"Trapezoid (z1={result_trap.x[0]:.2f}, z2={result_trap.x[1]:.2f})"
    )

    # Plot optimized pentagonal
    pen_values = np.array([pentagonal(xi, result_pen.x[0], result_pen.x[1], result_pen.x[2]) for xi in x])
    plt.plot(
        x,
        pen_values,
        "r--",
        linewidth=2,
        label=f"Pentagonal (z1={result_pen.x[0]:.2f}, z2={result_pen.x[1]:.2f}, y={result_pen.x[2]:.2f})",
    )

    plt.xlabel("x")
    plt.ylabel("Membership Value")
    plt.title("Comparison of Fuzzy Number Approximations to Gaussian")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 3)
    plt.ylim(0, 1.1)
    plt.show()


if __name__ == "__main__":
    main()
