"""
Regularization: Ridge, Lasso, and ElasticNet
=============================================
Overfitting is the enemy of generalization.

When a model has too many parameters relative to data:
  - It memorizes training noise
  - Coefficients become extremely large
  - Performance on new data collapses

Regularization adds a PENALTY to the cost function for large coefficients,
forcing the model to stay simple and generalize better.

MSE (standard):   L(w) = ||Xw - y||² / n
Ridge (L2):       L(w) = ||Xw - y||² / n  +  α · ||w||²
Lasso (L1):       L(w) = ||Xw - y||² / n  +  α · ||w||₁
ElasticNet:       L(w) = ||Xw - y||² / n  +  α(ρ·||w||₁ + (1-ρ)/2 · ||w||²)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet,
    RidgeCV, LassoCV, ElasticNetCV,
    LinearRegression
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

np.random.seed(42)

# ===========================================================================
# PART 1: The Overfitting Problem
# ===========================================================================
print("=" * 60)
print("PART 1: The Overfitting Problem")
print("=" * 60)

print("""
With many features relative to samples, linear regression
overfits by assigning huge coefficients to noise.

Example: 50 samples, 45 features → LinearRegression can
essentially memorize training data (R² ≈ 1.0 on train!)
but fails completely on test data.
""")

n, p = 80, 40
# Only first 5 features are truly relevant
X_raw = np.random.randn(n, p)
w_true = np.zeros(p)
w_true[:5] = [3.0, -2.0, 1.5, -1.0, 2.5]  # only 5 real features
y = X_raw @ w_true + np.random.randn(n) * 1.5

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Unregularized linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"LinearRegression  — Train R²: {r2_score(y_train, lr.predict(X_train)):.4f}, "
      f"Test R²: {r2_score(y_test, lr.predict(X_test)):.4f}")
print(f"  Max |coefficient|: {np.abs(lr.coef_).max():.2f}  (explodes with multicollinearity)")

# ===========================================================================
# PART 2: Ridge Regression (L2 Regularization)
# ===========================================================================
print("\n" + "=" * 60)
print("PART 2: Ridge Regression (L2)")
print("=" * 60)

print("""
Ridge adds: α · Σwᵢ²  (L2 norm = Euclidean distance of weights)

Effect:
  - All coefficients shrink toward 0
  - NO coefficient is driven exactly to 0
  - Works well when many features contribute moderately

WHY it works: The optimizer can't make any single weight large
without paying a growing penalty proportional to w².

α (alpha): the regularization strength
  α = 0  → pure LinearRegression (no penalty)
  α → ∞  → all weights → 0 (predict mean)
  α = 1  → default; use CV to find best
""")

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    train_r2 = r2_score(y_train, ridge.predict(X_train))
    test_r2  = r2_score(y_test,  ridge.predict(X_test))
    max_coef = np.abs(ridge.coef_).max()
    print(f"  α={alpha:7.2f}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, max|coef|={max_coef:.3f}")

# RidgeCV: automatic alpha selection via cross-validation
alphas_cv = np.logspace(-3, 4, 50)
ridge_cv = RidgeCV(alphas=alphas_cv, cv=5)
ridge_cv.fit(X_train, y_train)
print(f"\nRidgeCV best alpha: {ridge_cv.alpha_:.4f}")
print(f"  Train R²: {r2_score(y_train, ridge_cv.predict(X_train)):.4f}")
print(f"  Test R²:  {r2_score(y_test,  ridge_cv.predict(X_test)):.4f}")

# ===========================================================================
# PART 3: Lasso Regression (L1 Regularization)
# ===========================================================================
print("\n" + "=" * 60)
print("PART 3: Lasso Regression (L1) — Built-in Feature Selection")
print("=" * 60)

print("""
Lasso adds: α · Σ|wᵢ|  (L1 norm = sum of absolute weights)

Effect:
  - Some coefficients are driven EXACTLY to 0
  - This IS feature selection (zero weight = feature ignored)
  - Produces a sparse model (few non-zero weights)

WHY L1 gives sparsity: The L1 penalty has a "corner" at 0 —
gradient descent often lands exactly at 0 for irrelevant features.
(Technical: the subdifferential at 0 makes 0 a stable attractor)

Use Lasso when:
  - You believe only a few features truly matter
  - You want an interpretable model (automatic feature selection)
  - p >> n (more features than samples)
""")

# Remember: only first 5 of 40 features are real
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, max_iter=5000)
lasso_cv.fit(X_train, y_train)
print(f"LassoCV best alpha: {lasso_cv.alpha_:.6f}")
print(f"  Train R²: {r2_score(y_train, lasso_cv.predict(X_train)):.4f}")
print(f"  Test R²:  {r2_score(y_test,  lasso_cv.predict(X_test)):.4f}")

non_zero = np.sum(lasso_cv.coef_ != 0)
print(f"  Non-zero coefficients: {non_zero} / {p}  ← feature selection!")

# Which features survived?
print(f"\n  Features with non-zero coefficients (true are 0-4):")
for i, c in enumerate(lasso_cv.coef_):
    if c != 0:
        marker = "← TRUE FEATURE" if i < 5 else "← spurious"
        print(f"    Feature {i:2d}: {c:+.4f}  {marker}")

# ===========================================================================
# PART 4: ElasticNet — Best of Both
# ===========================================================================
print("\n" + "=" * 60)
print("PART 4: ElasticNet — Combining L1 and L2")
print("=" * 60)

print("""
ElasticNet: α·[ρ·||w||₁ + (1-ρ)/2·||w||²]
  ρ (l1_ratio): 0 → pure Ridge, 1 → pure Lasso, 0.5 → balanced

WHY ElasticNet:
  - Lasso struggles when features are correlated (picks one, drops others)
  - Ridge doesn't do feature selection
  - ElasticNet: sparse like Lasso + stable with correlated features

sklearn: ElasticNetCV finds best alpha AND l1_ratio
""")

en_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
    alphas=np.logspace(-4, 1, 50),
    cv=5,
    max_iter=5000
)
en_cv.fit(X_train, y_train)
print(f"ElasticNetCV: best alpha={en_cv.alpha_:.6f}, best l1_ratio={en_cv.l1_ratio_:.2f}")
print(f"  Train R²: {r2_score(y_train, en_cv.predict(X_train)):.4f}")
print(f"  Test R²:  {r2_score(y_test,  en_cv.predict(X_test)):.4f}")
print(f"  Non-zero: {np.sum(en_cv.coef_ != 0)} / {p}")

# ===========================================================================
# PART 5: Coefficient Path — How Coefficients Shrink With Alpha
# ===========================================================================
print("\n--- Plotting coefficient paths ---")

alphas_path = np.logspace(-3, 4, 100)
ridge_coefs = []
lasso_coefs = []

for a in alphas_path:
    r = Ridge(alpha=a).fit(X_train, y_train)
    ridge_coefs.append(r.coef_)
    l = Lasso(alpha=a, max_iter=5000).fit(X_train, y_train)
    lasso_coefs.append(l.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Regularization: Coefficient Paths", fontsize=13, fontweight='bold')

# Ridge path
for i in range(p):
    color = 'steelblue' if i < 5 else 'lightgray'
    lw    = 2 if i < 5 else 0.5
    axes[0].plot(alphas_path, ridge_coefs[:, i], color=color, lw=lw, alpha=0.8)
axes[0].set_xscale('log')
axes[0].set_xlabel("Alpha (regularization strength →)")
axes[0].set_ylabel("Coefficient value")
axes[0].set_title("Ridge: All coefficients shrink, NONE reach 0")
axes[0].axhline(0, color='black', lw=0.5)
axes[0].grid(True, alpha=0.3)

# Lasso path
for i in range(p):
    color = 'coral' if i < 5 else 'lightgray'
    lw    = 2 if i < 5 else 0.5
    axes[1].plot(alphas_path, lasso_coefs[:, i], color=color, lw=lw, alpha=0.8)
axes[1].set_xscale('log')
axes[1].set_xlabel("Alpha (regularization strength →)")
axes[1].set_ylabel("Coefficient value")
axes[1].set_title("Lasso: Coefficients hit exactly 0 (sparse!)")
axes[1].axhline(0, color='black', lw=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "regularization_paths.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Coefficient paths saved: {OUTPUT_DIR}/regularization_paths.png")

print("""
===========================================
WHEN TO USE WHICH REGULARIZATION
===========================================
LinearRegression: no regularization needed (few features, large n)
Ridge (L2):       many correlated features, you believe all matter
Lasso (L1):       many features, you believe only a few truly matter
ElasticNet:       large p, correlated features, want some sparsity

Choosing alpha:
  Always use RidgeCV / LassoCV / ElasticNetCV — don't guess!
  Or use GridSearchCV with a Pipeline.

RULE: Always scale features BEFORE applying regularization.
  Regularization penalizes large coefficients.
  If features are on different scales, unscaled large-range features
  get small coefficients anyway — regularization treats them unfairly.
""")
