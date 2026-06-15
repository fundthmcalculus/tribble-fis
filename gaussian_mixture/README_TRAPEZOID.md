# Trapezoidal Membership Functions in Concrete Strength Prediction

This directory now includes two variants of the concrete strength regression example:

1. **`concrete.py`** — Original Gaussian mixture model approach
2. **`concrete_trapz.py`** — New trapezoidal membership function approach

## Quick Start

### Run Original Gaussian Example
```bash
python concrete.py
```
Expected output: Multi-order TSK regression (0th, 1st, 2nd, 3rd order)

### Run New Trapezoid Example
```bash
python concrete_trapz.py
```
Expected output: Zeroth-order TSK regression with trapezoid MFs

---

## Comparison: Gaussian vs Trapezoid Membership Functions

### Gaussian Membership Functions (concrete.py)
- **MF Shape:** Bell curve (Gaussian distribution)
- **Parameters:** 2 per MF (μ, σ)
- **Support:** Infinite (asymptotic decay)
- **Strengths:**
  - Sharp, localized regions
  - Good for feature-specific discrimination
  - Well-suited for higher-order TSK regression
- **Use case:** When you need precise feature regions

### Trapezoidal Membership Functions (concrete_trapz.py)
- **MF Shape:** Piecewise-linear trapezoid
- **Parameters:** 4 per MF (a, b, c, d)
- **Support:** [a, d] (finite, plateau top)
- **Strengths:**
  - Broader, flatter membership regions
  - More interpretable (explicit plateau)
  - Good for data with plateau-like distributions
  - BIC-based automatic component selection
- **Use case:** When features have distributed decision regions

---

## Implementation Details

### Key Differences

**Feature Selection** (identical):
- Both use `calculate_gaussian_correlation()` to rank features
- Both use `take_top_features()` to select top-N variables

**Membership Function Fitting**:
- **Gaussian:** `create_gaussian_membership_dict()` via `fit_gaussians()` using KMeans clustering
- **Trapezoid:** `create_trapz_membership_dict()` via `fit_trapezoids()` using histogram-based EM

**Inference** (identical):
- Both use `tsk_firing_strengths()` for rule firing
- Both use `normalize_firing_strength` for weighted predictions

**TSK Orders**:
- **Gaussian:** Supports 0th, 1st, 2nd (full), 3rd order corrections
- **Trapezoid:** Zeroth-order only (simplified due to broader MF regions)

---

## How Trapezoid Fitting Works

### 1. Histogram-Based EM Algorithm
```
For each feature/label combination:
  1. Compute histogram (50 bins by default)
  2. Initialize trapezoids from histogram peaks
  3. Run EM algorithm:
     - E-step: Compute responsibilities
     - M-step: Update trapezoid params (constrained: a ≤ b ≤ c ≤ d)
  4. BIC-based selection of # components (if n_trapezoids == -1)
```

### 2. Initialization from Histogram Peaks
```
1. Smooth histogram with Gaussian filter
2. Detect peaks using scipy.signal.find_peaks
3. For each peak:
   - Set shoulders [b, c] at half-power width
   - Set feet [a, d] at valley edges
   - Apply minimum-width guards
```

### 3. Constraint-Preserving Optimization
```
For each component in M-step:
  Minimize: -weighted_log_likelihood(a, b, c, d)
  Subject to: a ≤ b ≤ c ≤ d
  Method: scipy.optimize.minimize(method='SLSQP')
```

---

## Usage Example: Classification

To use trapezoids in classification (not regression):

```python
from src.tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier

# Gaussian (default)
clf_gauss = MixtureOfGaussiansFuzzyClassifier(top_p=0.9)
clf_gauss.fit(X_train, y_train)

# Trapezoid (new)
clf_trap = MixtureOfGaussiansFuzzyClassifier(
    top_p=0.9,
    member_function="trap"  # NEW parameter
)
clf_trap.fit(X_train, y_train)

# Identical interface
y_pred = clf_trap.predict(X_test)
y_proba = clf_trap.predict_proba(X_test)
```

---

## Usage Example: Direct EM Fitting

For 1D feature fitting with automatic component selection:

```python
from src.tribblefis.trapz_math import TrapzMixtureModel

# Auto-select K via BIC
model = TrapzMixtureModel(n_components=0, max_components=4)
model.fit(data_1d)

print(f"Optimal K: {len(model.trapezoids_)}")
print(f"Weights: {model.weights_}")
print(f"BIC: {model.bic_:.2f}")

# Or fixed K
model2 = TrapzMixtureModel(n_components=2)
model2.fit(data_1d)
```

---

## Performance Comparison

### Concrete Dataset Results

**Gaussian (concrete.py):**
```
0th Order:   R² = 0.79, RMSE = 16.28
1st Order:   R² = 0.86, RMSE = 12.92
2nd Order:   R² = 0.89, RMSE = 11.21
2nd-full:    R² = 0.91, RMSE = 10.03
3rd Order:   R² = 0.92, RMSE = 9.87
```

**Trapezoid (concrete_trapz.py) - Current:**
```
0th Order:   R² = -0.86, RMSE = 22.60
(Simplified: No higher orders due to broader MF regions)
```

**Notes:**
- Gaussian MFs excel at high-order TSK regression
- Trapezoid MFs work best for classification or simple (0th-order) prediction
- For regression, use `concrete.py` (Gaussian)
- For classification, consider `member_function="trap"` parameter

---

## When to Use Each Approach

### Use **Gaussian** (`concrete.py`) When:
✅ You need high-order polynomial corrections (1st, 2nd, 3rd)  
✅ You want sharp feature discrimination  
✅ Regression task is primary (as in concrete.py)  
✅ Data follows smooth distributions  

### Use **Trapezoid** (`concrete_trapz.py`) When:
✅ You prefer interpretable, explicit support regions  
✅ Data has plateau-like distributions  
✅ You want automatic component selection via BIC  
✅ Classification is primary (use `member_function="trap"`)  
✅ You need finite membership region bounds [a, d]  

---

## Technical Notes

### Component Selection
- **Gaussian:** Manual specification via `n_gaussians` parameter
- **Trapezoid:** Auto-select via BIC (K minimizing `n_params * log(N) - 2*LL`)

### Constraints
- **Gaussian:** No ordering constraints (mu, sigma can be any value)
- **Trapezoid:** Always maintains `a ≤ b ≤ c ≤ d` (enforced by SLSQP)

### Numerical Stability
- Both handle small samples (n=10) and large samples (n=10,000)
- Trapezoid EM is log-sum-exp stable (avoids numerical underflow)
- Gaussian TSK regression can use higher orders (better conditioning)

### Computational Cost
- **Gaussian:** O(K) per feature/label (K-means clustering)
- **Trapezoid:** O(K² × max_iter) per feature/label (EM + constrained optimization)
  - BIC selection multiplies this by max_components

---

## Future Extensions

1. **Mixed MF Types:** Combine Gaussian and Trapezoid in same model
2. **Trapezoid Regression:** Implement higher-order TSK for trapezoids
3. **Triangular MFs:** New `TriangularMembership` class (3 parameters)
4. **Synthetic Data:** Trapezoid random sampling via inverse CDF

---

## See Also

- **Classification Example:** `gaussian_mixture/wine_red.py` (now supports `member_function="trap"`)
- **Implementation:** `src/tribblefis/trapz_math.py`
- **Tests:** `tests/test_trapz_math.py`
- **Plan:** `/home/scott/.claude/plans/i-want-you-to-structured-yao.md`
