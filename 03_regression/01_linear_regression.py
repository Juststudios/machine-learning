"""
Linear Regression: From Mathematics to Code
=============================================
Regression answers: "How much?" or "What value?"
Examples:
  - Predict CPU temperature from clock speed and load
  - Predict energy consumption from building features  
  - Predict project delivery time from team size and complexity

MATH FOUNDATION:
  y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
  y = Xw + b  (matrix form)
  
  Loss function: MSE = (1/n) Σ(y_pred - y_true)²
  Goal: find w and b that minimize MSE
  Method: gradient descent (or analytical solution for linear regression)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

np.random.seed(42)

# ===========================================================================
# PART 1: Simple Linear Regression (1 Feature)
# ===========================================================================
print("=" * 60)
print("PART 1: Simple Linear Regression (1 Feature)")
print("=" * 60)

print("""
Real problem: CPU thermal management
  - We need to predict CPU temperature from clock speed
  - If we know the temperature, we can throttle the CPU before damage
  - A regression model gives us real-time temperature estimates
""")

# Synthetic CPU data: temperature rises with clock speed (nonlinearly, but
# linear approximation is useful over a limited range)
n = 100
clock_ghz = np.random.uniform(1.0, 4.5, n)
# True relationship: temp = 15 * clock + 25 + noise
temperature_C = 15 * clock_ghz + 25 + np.random.normal(0, 4, n)

X = clock_ghz.reshape(-1, 1)
y = temperature_C

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Model learned:")
print(f"  Slope (coefficient): {model.coef_[0]:.4f}")
print(f"  Intercept:           {model.intercept_:.4f}")
print(f"\nInterpretation:")
print(f"  For every 1 GHz increase in clock speed,")
print(f"  temperature increases by {model.coef_[0]:.1f}°C")
print(f"  A CPU at 0 GHz would be {model.intercept_:.1f}°C (baseline/intercept)")

y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\nTest performance:")
print(f"  MAE:  {mae:.2f}°C  (average error)")
print(f"  RMSE: {rmse:.2f}°C (penalizes large errors more)")
print(f"  R²:   {r2:.4f}   ({r2*100:.1f}% of temperature variance explained)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Regression line
axes[0].scatter(X_train, y_train, alpha=0.5, label='Training data', color='steelblue')
axes[0].scatter(X_test, y_test, alpha=0.7, label='Test data', color='coral', marker='s')
x_line = np.linspace(1.0, 4.5, 100).reshape(-1, 1)
axes[0].plot(x_line, model.predict(x_line), 'k-', linewidth=2, label=f'y = {model.coef_[0]:.1f}x + {model.intercept_:.1f}')
axes[0].set_xlabel("Clock Speed (GHz)")
axes[0].set_ylabel("Temperature (°C)")
axes[0].set_title("CPU Temperature vs Clock Speed")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residual plot
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.7, color='steelblue')
axes[1].axhline(y=0, color='red', linestyle='--')
axes[1].set_xlabel("Predicted Temperature (°C)")
axes[1].set_ylabel("Residual (Actual - Predicted)")
axes[1].set_title("Residual Plot — Check for Patterns")
axes[1].grid(True, alpha=0.3)
# Good: residuals should be random around 0 (no pattern)
# Bad patterns: funnel shape (heteroscedasticity), curve (nonlinear relationship)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "linear_regression_simple.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: {OUTPUT_DIR}/linear_regression_simple.png")

# ===========================================================================
# PART 2: Multiple Linear Regression (Many Features)
# ===========================================================================
print("\n" + "=" * 60)
print("PART 2: Multiple Linear Regression (Many Features)")
print("=" * 60)

print("""
Real problem: Server energy prediction
  Features: CPU load, memory usage, disk I/O, network traffic, core count
  Target: power consumption (Watts)
  
  y = w₁(cpu_load) + w₂(memory) + w₃(disk_io) + w₄(network) + w₅(cores) + b
""")

n = 300
cpu_load    = np.random.uniform(0, 100, n)
memory_gb   = np.random.uniform(4, 128, n)
disk_io     = np.random.uniform(0, 500, n)
network_mb  = np.random.uniform(0, 1000, n)
cores       = np.random.choice([4, 8, 16, 32], n)

# True relationship (we pick the weights)
power_W = (
    1.5 * cpu_load       # each % CPU load costs 1.5W
    + 0.8 * memory_gb    # each GB RAM costs 0.8W
    + 0.02 * disk_io     # disk I/O is cheap
    + 0.05 * network_mb  # network is fairly cheap
    + 5.0 * cores        # each core has base power draw
    + 20                 # baseline (PSU overhead)
    + np.random.normal(0, 15, n)  # real-world noise
)

X_multi = np.column_stack([cpu_load, memory_gb, disk_io, network_mb, cores])
y_multi = power_W

feature_names = ['CPU Load (%)', 'Memory (GB)', 'Disk I/O (MB/s)', 'Network (MB/s)', 'Cores']

X_tr, X_te, y_tr, y_te = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_tr, y_tr)

y_pred_multi = model_multi.predict(X_te)
r2_multi  = r2_score(y_te, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_te, y_pred_multi))

print(f"\nModel coefficients (learned feature weights):")
for name, coef in zip(feature_names, model_multi.coef_):
    print(f"  {name:25s}: {coef:+.4f} W per unit")
print(f"  {'Intercept':25s}: {model_multi.intercept_:+.4f} W")

print(f"\nTrue weights used to generate data:")
true_weights = [1.5, 0.8, 0.02, 0.05, 5.0]
for name, true_w, learned_w in zip(feature_names, true_weights, model_multi.coef_):
    print(f"  {name:25s}: true={true_w:+.2f}, learned={learned_w:+.4f}")

print(f"\nTest R²:   {r2_multi:.4f}  (with noise, we expect < 1.0)")
print(f"Test RMSE: {rmse_multi:.2f} Watts")

# ===========================================================================
# PART 3: Analytical Understanding — The Normal Equation
# ===========================================================================
print("\n" + "=" * 60)
print("PART 3: What Does 'Fitting' Actually Compute?")
print("=" * 60)

print("""
For linear regression, sklearn doesn't use gradient descent by default.
It uses the NORMAL EQUATION (analytical solution):

  w* = (XᵀX)⁻¹ Xᵀy

This gives the EXACT minimum of MSE in one calculation.

WHY sklearn uses this:
  - Exact solution (no learning rate, no epochs needed)
  - Fast for moderate numbers of features (< ~10,000)
  - For HUGE datasets: gradient descent is more practical

Let's verify by computing it manually:
""")

# Small example
X_demo = np.array([[1, 2], [1, 4], [1, 6], [1, 8]])  # Note: first col is all 1s for intercept
y_demo = np.array([3.2, 5.8, 8.1, 10.4])

# Normal equation: w = (X'X)^-1 X'y
w_normal = np.linalg.pinv(X_demo.T @ X_demo) @ X_demo.T @ y_demo
print(f"Normal equation result:  intercept={w_normal[0]:.4f}, slope={w_normal[1]:.4f}")

# Sklearn
lr_demo = LinearRegression(fit_intercept=False)  # intercept already in X column
lr_demo.fit(X_demo, y_demo)
print(f"sklearn LinearRegression: intercept={lr_demo.coef_[0]:.4f}, slope={lr_demo.coef_[1]:.4f}")
print("  ↑ Same result! sklearn also uses the normal equation (via SVD)")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. Linear regression: y = Xw + b, minimizes MSE cost function
2. Coefficients = partial effect of each feature on target
3. Positive coef → feature increases target; Negative → decreases it
4. R² = 1 means perfect; R² = 0 means model is no better than predicting the mean
5. Always check residual plots — patterns reveal model problems
6. sklearn uses the normal equation (exact solution), not gradient descent

Next: 02_polynomial_regression.py — When the relationship isn't linear
""")
