"""
Exercises: Regression (Module 3)
==================================
4-tier exercises covering linear regression, polynomial regression,
regularization, and gradient descent from scratch.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

np.random.seed(42)

print("=" * 60)
print("REGRESSION — EXERCISES")
print("=" * 60)

# ===========================================================================
# LEVEL 1: RECALL
# ===========================================================================
print("\n--- LEVEL 1: RECALL ---")
print("""
Study this code and answer the questions:

1a. What does a coefficient of -2.3 for 'temperature' mean?
1b. What does R² = 0.72 mean in plain English?
1c. Why is RMSE preferred over MSE for reporting error?
1d. Why might Ridge perform better than LinearRegression on this data?
1e. What happens to Lasso coefficients as alpha increases?
""")

# Demo dataset: predict server power from CPU metrics
n = 150
cpu_load  = np.random.uniform(5, 95, n)
mem_usage = np.random.uniform(2, 64, n)
network   = np.random.uniform(0, 1000, n)

power_W = 0.8*cpu_load + 1.2*mem_usage + 0.01*network + 30 + np.random.randn(n)*8

X = np.column_stack([cpu_load, mem_usage, network])
y = power_W

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

for name, model in [('LinearRegression', LinearRegression()),
                     ('Ridge(α=1.0)',     Ridge(alpha=1.0)),
                     ('Lasso(α=0.1)',     Lasso(alpha=0.1))]:
    model.fit(X_tr_s, y_train)
    r2   = r2_score(y_test, model.predict(X_te_s))
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_te_s)))
    print(f"  {name:25s}: R²={r2:.4f}, RMSE={rmse:.2f}W, coef={model.coef_.round(3)}")

# ===========================================================================
# LEVEL 2: DEBUGGING
# ===========================================================================
print("\n--- LEVEL 2: DEBUGGING ---")
print("""
Find the 3 bugs in this polynomial regression pipeline.
""")

buggy = '''
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

np.random.seed(0)
x = np.linspace(0, 5, 100).reshape(-1, 1)
y = 3*x.ravel()**2 - 2*x.ravel() + 1 + np.random.randn(100)*0.5

# BUG 1: Polynomial features fitted BEFORE scaling
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_poly)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)

# BUG 2: Fitted poly on test data separately (should use already-fitted poly!)
x_test_poly = PolynomialFeatures(degree=2).fit_transform(x_test)
x_test_scaled = scaler.transform(x_test_poly)

# BUG 3: R² computed on y_train but labeled as "test"
print(f"Test R²: {r2_score(y_train, model.predict(x_train)):.4f}")
'''
print(buggy)
print("""
Bug 1: ________________________________________________
Bug 2: ________________________________________________
Bug 3: ________________________________________________

TODO: Write the correct version using a Pipeline
""")

# ===========================================================================
# LEVEL 3: APPLICATION — Energy Prediction with Regularization
# ===========================================================================
print("\n--- LEVEL 3: APPLICATION ---")
print("""
TASK: Build a regularized regression model to predict building energy.

Dataset:
  - 200 buildings with 15 features (many correlated/irrelevant)
  - True relationship uses only 4 features
  - Features include noise and multicollinearity

Requirements:
  1. Generate data (use the starter below)
  2. Split 80/20, scale features
  3. Compare: LinearRegression vs Ridge vs Lasso vs ElasticNet
  4. For each: 5-fold CV R² on train, and test R²
  5. Use RidgeCV and LassoCV for automatic alpha selection
  6. Report: which model has best test R²?
  7. Report: how many features does Lasso keep non-zero?
  8. Print the top 5 features by absolute Lasso coefficient
""")

n, p = 200, 15
X_app = np.random.randn(n, p)
# True: only features 0, 3, 7, 12 matter
w_true = np.zeros(p)
w_true[[0, 3, 7, 12]] = [4.0, -2.5, 1.8, -3.0]
# Add collinearity: feature 1 ≈ feature 0, feature 8 ≈ feature 7
X_app[:, 1] = X_app[:, 0] + np.random.randn(n) * 0.3
X_app[:, 8] = X_app[:, 7] + np.random.randn(n) * 0.3
y_app = X_app @ w_true + np.random.randn(n) * 2.0

print(f"Data generated: X.shape={X_app.shape}")
print(f"True feature weights: {dict(zip(range(p), w_true))}")
print("TODO: Complete the regularization comparison above")

# ===========================================================================
# LEVEL 4: CHALLENGE — Gradient Descent From Scratch
# ===========================================================================
print("\n--- LEVEL 4: CHALLENGE ---")
print("""
TASK: Implement mini-batch gradient descent for linear regression
      using only numpy (NO sklearn for the model).

Requirements:
  class LinearRegressionGD:
      def __init__(self, lr=0.01, n_epochs=100, batch_size=32):
      
      def fit(self, X, y):
          # Initialize weights (small random values)
          # For each epoch:
          #   Shuffle data
          #   For each mini-batch:
          #     Compute predictions: y_pred = X @ w + b
          #     Compute MSE gradient:
          #       dL/dw = (2/batch_size) * X.T @ (y_pred - y)
          #       dL/db = (2/batch_size) * sum(y_pred - y)
          #     Update: w -= lr * dL/dw, b -= lr * dL/db
          # Return self
      
      def predict(self, X):
          return X @ self.w + self.b
  
  Test your implementation:
  - Train on the Level 3 data
  - Compare R² with sklearn LinearRegression
  - Plot the training loss curve (MSE per epoch)
  - They should give similar R² on test data

Tip: Scale features FIRST (gradient descent is sensitive to scale)
""")

print("TODO: Implement LinearRegressionGD and compare with sklearn")
# Reference to beat:
scaler_ch = StandardScaler()
X_app_s = scaler_ch.fit_transform(X_app)
X_app_tr, X_app_te, y_app_tr, y_app_te = train_test_split(X_app_s, y_app, test_size=0.2, random_state=42)
sk_model = LinearRegression()
sk_model.fit(X_app_tr, y_app_tr)
print(f"sklearn R² (to beat): {r2_score(y_app_te, sk_model.predict(X_app_te)):.4f}")

print("\n" + "=" * 60)
print("EXERCISES COMPLETE")
print("Check solutions/regression_solutions.py for reference answers.")
print("=" * 60)
