"""
Benchmark: EM-based vs Fast histogram-based trapezoid fitting.

Compares the runtime performance and accuracy of:
1. trapz_math.TrapzMixtureModel (EM-based, iterative optimization)
2. trapz_math_fast.fit_trapezoids_fast (histogram-based, O(n) direct fitting)

Run this benchmark to understand the performance/accuracy trade-off.
"""

import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tribblefis.trapz_math import TrapzMixtureModel, fit_trapezoids_em, trapz_pdf
from tribblefis.trapz_math_fast import fit_trapezoids_fast, trapz_pdf_fast


def generate_synthetic_data(distribution_type: str, size: int = 1000) -> np.ndarray:
    """Generate test data with different distributions."""
    np.random.seed(42)

    if distribution_type == "unimodal":
        return np.random.normal(0, 1, size)
    elif distribution_type == "bimodal":
        return np.concatenate([
            np.random.normal(-3, 0.5, size // 2),
            np.random.normal(3, 0.5, size // 2)
        ])
    elif distribution_type == "trimodal":
        return np.concatenate([
            np.random.normal(-4, 0.4, size // 3),
            np.random.normal(0, 0.4, size // 3),
            np.random.normal(4, 0.4, size // 3)
        ])
    elif distribution_type == "exponential":
        return np.random.exponential(2.0, size)
    elif distribution_type == "heavy_tail":
        return np.concatenate([
            np.random.normal(0, 1, int(0.95 * size)),
            np.random.normal(0, 5, int(0.05 * size))
        ])
    else:
        raise ValueError(f"Unknown distribution: {distribution_type}")


def benchmark_em_method(data: np.ndarray, n_components: int, n_bins: int = 50, max_iter: int = 100) -> dict:
    """Benchmark EM-based fitting."""
    start_time = time.perf_counter()

    try:
        trapezoids, weights, ll = fit_trapezoids_em(
            data, n_components=n_components, n_bins=n_bins, max_iter=max_iter
        )
        elapsed = time.perf_counter() - start_time

        # Compute fit quality: coverage and reconstruction
        hist_counts, hist_edges = np.histogram(data, bins=50)
        hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

        # Compute PDF coverage
        coverage = 0.0
        for trapz, w in zip(trapezoids, weights):
            pdf_vals = trapz_pdf(hist_centers, trapz.a, trapz.b, trapz.c, trapz.d)
            coverage += np.sum(pdf_vals * hist_counts) * w

        return {
            'method': 'EM',
            'elapsed': elapsed,
            'n_trapezoids': len(trapezoids),
            'weights': weights,
            'trapezoids': trapezoids,
            'log_likelihood': ll,
            'coverage': coverage / np.sum(hist_counts) if np.sum(hist_counts) > 0 else 0.0,
            'success': True
        }
    except Exception as e:
        return {
            'method': 'EM',
            'elapsed': time.perf_counter() - start_time,
            'error': str(e),
            'success': False
        }


def benchmark_fast_method(data: np.ndarray, n_bins: int = 50) -> dict:
    """Benchmark fast histogram-based fitting."""
    start_time = time.perf_counter()

    try:
        trapezoids, weights = fit_trapezoids_fast(data, n_bins=n_bins)
        elapsed = time.perf_counter() - start_time

        # Compute fit quality: coverage and reconstruction
        hist_counts, hist_edges = np.histogram(data, bins=50)
        hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

        # Compute PDF coverage
        coverage = 0.0
        for trapz, w in zip(trapezoids, weights):
            pdf_vals = trapz_pdf_fast(hist_centers, trapz.a, trapz.b, trapz.c, trapz.d)
            coverage += np.sum(pdf_vals * hist_counts) * w

        return {
            'method': 'Fast',
            'elapsed': elapsed,
            'n_trapezoids': len(trapezoids),
            'weights': weights,
            'trapezoids': trapezoids,
            'coverage': coverage / np.sum(hist_counts) if np.sum(hist_counts) > 0 else 0.0,
            'success': True
        }
    except Exception as e:
        return {
            'method': 'Fast',
            'elapsed': time.perf_counter() - start_time,
            'error': str(e),
            'success': False
        }


def print_comparison(distribution: str, em_result: dict, fast_result: dict):
    """Pretty-print benchmark comparison."""
    print(f"\n{'='*70}")
    print(f"Distribution: {distribution.upper()}")
    print(f"{'='*70}")

    if em_result['success']:
        print(f"\nEM Method:")
        print(f"  Time: {em_result['elapsed']:.6f}s")
        print(f"  Trapezoids: {em_result['n_trapezoids']}")
        print(f"  Coverage: {em_result['coverage']:.4f}")
        print(f"  Log-Likelihood: {em_result['log_likelihood']:.2f}")
    else:
        print(f"\nEM Method: FAILED - {em_result.get('error', 'Unknown error')}")
        print(f"  Time: {em_result['elapsed']:.6f}s")

    if fast_result['success']:
        print(f"\nFast Method:")
        print(f"  Time: {fast_result['elapsed']:.6f}s")
        print(f"  Trapezoids: {fast_result['n_trapezoids']}")
        print(f"  Coverage: {fast_result['coverage']:.4f}")
    else:
        print(f"\nFast Method: FAILED - {fast_result.get('error', 'Unknown error')}")
        print(f"  Time: {fast_result['elapsed']:.6f}s")

    if em_result['success'] and fast_result['success']:
        speedup = em_result['elapsed'] / fast_result['elapsed']
        print(f"\nSpeedup: {speedup:.1f}x (Fast is {speedup:.1f}x faster)")
        coverage_diff = abs(em_result['coverage'] - fast_result['coverage'])
        print(f"Coverage Difference: {coverage_diff:.4f}")


def run_benchmark_suite():
    """Run comprehensive benchmark across different data distributions and sizes."""
    print("\n" + "="*70)
    print("TRAPZ FITTING BENCHMARK: EM vs Fast Histogram Method")
    print("="*70)

    distributions = ['unimodal', 'bimodal', 'trimodal', 'exponential', 'heavy_tail']
    data_sizes = [100, 500, 1000, 5000]

    # Summary table
    print("\n" + "-"*70)
    print("SUMMARY: Speedup of Fast Method (higher = better)")
    print("-"*70)
    print(f"{'Distribution':<15} {'Size':<8} {'EM Time':<12} {'Fast Time':<12} {'Speedup':<10}")
    print("-"*70)

    summary_data = []

    for distribution in distributions:
        for size in data_sizes:
            data = generate_synthetic_data(distribution, size)
            n_components = 1 if distribution == 'unimodal' else (2 if distribution == 'bimodal' else 3)

            em_result = benchmark_em_method(data, n_components=n_components, n_bins=50, max_iter=100)
            fast_result = benchmark_fast_method(data, n_bins=50)

            if em_result['success'] and fast_result['success']:
                speedup = em_result['elapsed'] / fast_result['elapsed']
                summary_data.append({
                    'distribution': distribution,
                    'size': size,
                    'em_time': em_result['elapsed'],
                    'fast_time': fast_result['elapsed'],
                    'speedup': speedup
                })
                print(f"{distribution:<15} {size:<8} {em_result['elapsed']:<12.6f} {fast_result['elapsed']:<12.6f} {speedup:<10.1f}x")

            # Detailed comparison for each distribution/size combo
            if size == 1000:  # Only show detailed output for 1000 samples
                print_comparison(f"{distribution} (n={size})", em_result, fast_result)

    # Print summary statistics
    print("\n" + "="*70)
    print("AGGREGATE STATISTICS")
    print("="*70)
    if summary_data:
        speedups = [d['speedup'] for d in summary_data]
        em_times = [d['em_time'] for d in summary_data]
        fast_times = [d['fast_time'] for d in summary_data]

        print(f"Average Speedup: {np.mean(speedups):.1f}x")
        print(f"Min Speedup: {np.min(speedups):.1f}x")
        print(f"Max Speedup: {np.max(speedups):.1f}x")
        print(f"Total EM Time: {np.sum(em_times):.4f}s")
        print(f"Total Fast Time: {np.sum(fast_times):.4f}s")
        print(f"Overall Speedup: {np.sum(em_times) / np.sum(fast_times):.1f}x")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
Fast Method is best for:
  - High-speed inference or real-time applications
  - Large datasets (>10,000 samples)
  - When you need reproducible, deterministic results
  - When histogram structure is meaningful to your problem

EM Method is best for:
  - When you need precise maximum-likelihood estimates
  - Small to medium datasets (<5,000 samples)
  - When you want automatic component selection via BIC
  - When fine-tuned membership functions matter
    """)


if __name__ == '__main__':
    run_benchmark_suite()
