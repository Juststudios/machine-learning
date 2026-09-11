"""
Polynomial Regression: When Linear Isn't Enough
================================================
Linear regression assumes: y = w₁x₁ + w₂x₂ + b

But nature is rarely linear:
  - Battery discharge follows an exponential curve
  - Airflow through a duct scales with velocity squared
  - Biological growth follows sigmoid patterns

Polynomial regression extends linear regression by adding
polynomial features: x, x², x³, x·z, x²·z, ...
Then fits a linear model on the expanded features.

Key insight: the model is still LINEAR in the coefficients —
we just engineered new nonlinear features from the original ones.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

np.random.seed(42)

# ===========================================================================
# PART 1: When Linear Fails
# ===========================================================================
print("=" * 60)
print("PART 1: When Linear Fails — A Real Engineering Problem")
print("=" * 60)

print("""
Real problem: Aerodynamic drag
  Drag force F = ½ρCdAv²
  
  Drag is proportional to velocity SQUARED.
  A linear model can't capture this — it needs a quadratic term.
  
  Similarly in ML: if the true relationship is nonlinear,
  a linear model will systematically underfit no matter
  how much data you give it.
""")

# True relationship: F_drag = 0.5 * v² (simplified)
n = 80
v = np.linspace(0, 10, n)  # velocity 0–10 m/s
F = 0.5 * v**2 + np.random.normal(0, 1.5, n)  # true: quadratic + noise

X_v = v.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X_v, F, test_size=0.2, random_state=42)

# Linear model (wrong assumption)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
r2_lin = r2_score(y_test, lin_model.predict(X_test))

# Degree-2 polynomial (correct for this physics)
poly2_pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('reg',  LinearRegression()),
])
poly2_pipe.fit(X_train, y_train)
r2_poly2 = r2_score(y_test, poly2_pipe.predict(X_test))

print(f"Linear model    test R²: {r2_lin:.4f}  ← systematically wrong")
print(f"Polynomial (d=2) test R²: {r2_poly2:.4f}  ← matches the physics")

# ===========================================================================
# PART 2: How PolynomialFeatures Works
# ===========================================================================
print("\n" + "=" * 60)
print("PART 2: How PolynomialFeatures Works")
print("=" * 60)

print("""
Input: [x₁, x₂]
PolynomialFeatures(degree=2, include_bias=False) produces:
  [x₁, x₂, x₁², x₁·x₂, x₂²]

PolynomialFeatures(degree=3) would produce:
  [x₁, x₂, x₁², x₁x₂, x₂², x₁³, x₁²x₂, x₁x₂², x₂³]
""")

# 1-feature example
from sklearn.preprocessing import PolynomialFeatures
pf1 = PolynomialFeatures(degree=3, include_bias=False)
X_demo1 = np.array([[2.0], [3.0]])
X_poly1 = pf1.fit_transform(X_demo1)
print("1 feature, degree=3:")
print(f"  Input:  {X_demo1.flatten()}")
print(f"  Output: {X_poly1}  (columns: x, x², x³)")

# 2-feature example
pf2 = PolynomialFeatures(degree=2, include_bias=False)
X_demo2 = np.array([[2.0, 3.0], [4.0, 5.0]])
X_poly2 = pf2.fit_transform(X_demo2)
print(f"\n2 features, degree=2:")
print(f"  Input:  {X_demo2}")
print(f"  Output:\n{X_poly2}")
print(f"  Feature names: {pf2.get_feature_names_out(['x1', 'x2'])}")

# Feature count explosion!
print("\nFeature count explosion with degree:")
for d in [1, 2, 3, 4, 5]:
    pf = PolynomialFeatures(degree=d, include_bias=False)
    n_feat = pf.fit_transform(np.zeros((1, 5))).shape[1]
    print(f"  degree={d}, 5 original features → {n_feat} polynomial features")

# ===========================================================================
# PART 3: Degree Selection — The Bias-Variance Tradeoff
# ===========================================================================
print("\n" + "=" * 60)
print("PART 3: Choosing the Right Degree")
print("=" * 60)

print("""
Too low a degree → underfitting (high bias: model too simple)
Too high a degree → overfitting (high variance: model memorizes noise)

The right degree → best generalization (balanced bias-variance)

Use cross-validation to pick the best degree.
""")

# True relationship: y = sin(x) + noise
n = 100
X_sin = np.sort(np.random.uniform(0, 2*np.pi, n)).reshape(-1, 1)
y_sin = np.sin(X_sin.ravel()) + np.random.normal(0, 0.2, n)

X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_sin, y_sin, test_size=0.2, random_state=42)

degrees = [1, 2, 3, 4, 5, 7, 10, 15]
results = {}

print(f"{'Degree':>7} | {'Train R²':>9} | {'Test R²':>8} | {'CV R² (mean±std)':>20}")
print("-" * 55)

for d in degrees:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(d, include_bias=False)),
        ('reg',  LinearRegression())
    ])
    pipe.fit(X_tr_s, y_tr_s)
    
    r2_train = r2_score(y_tr_s, pipe.predict(X_tr_s))
    r2_test  = r2_score(y_te_s, pipe.predict(X_te_s))
    cv_scores = cross_val_score(pipe, X_sin, y_sin, cv=5, scoring='r2')
    
    results[d] = {'train': r2_train, 'test': r2_test, 'cv': cv_scores}
    
    marker = " ← best" if d == 5 else ""  # roughly optimal for sin(x)
    print(f"  d={d:2d}  | {r2_train:9.4f} | {r2_test:8.4f} | {cv_scores.mean():.4f} ± {cv_scores.std():.4f}{marker}")

# ===========================================================================
# PART 4: Visualization — Different Degree Fits
# ===========================================================================
x_plot = np.linspace(0, 2*np.pi, 300).reshape(-1, 1)
x_true = np.linspace(0, 2*np.pi, 300)
y_true_plot = np.sin(x_true)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Polynomial Regression: Effect of Degree on Fit", fontsize=13, fontweight='bold')

for ax, d in zip(axes.flat, degrees):
    pipe = Pipeline([('poly', PolynomialFeatures(d)), ('reg', LinearRegression())])
    pipe.fit(X_tr_s, y_tr_s)
    
    ax.scatter(X_tr_s, y_tr_s, alpha=0.4, s=15, color='gray', label='Train')
    ax.plot(x_true, y_true_plot, 'g-', linewidth=1.5, alpha=0.7, label='True: sin(x)')
    ax.plot(x_plot, pipe.predict(x_plot), 'r-', linewidth=2, label=f'd={d}')
    
    test_r2 = results[d]['test']
    ax.set_title(f"Degree={d}  (Test R²={test_r2:.3f})")
    ax.set_ylim(-2.5, 2.5)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "polynomial_regression_degrees.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nDegree comparison plot saved: {OUTPUT_DIR}/polynomial_regression_degrees.png")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. Polynomial regression: add x², x³ as features, then fit linear regression
2. PolynomialFeatures transformer expands features automatically
3. Features explode exponentially with degree and original feature count
4. Low degree → underfitting (high bias)
5. High degree → overfitting (high variance: memorizes noise)
6. Use cross-validation to select the best degree
7. For high-degree polynomials → must use regularization (Ridge/Lasso)!

Next: 03_regularization.py — Controlling overfitting with Ridge and Lasso
""")
