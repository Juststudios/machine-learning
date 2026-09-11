"""
Lesson 4: Train/Test Split & Overfitting
=========================================
How do you honestly measure if your model is any good?

This lesson covers:
- WHY we can't test on training data (the 'studying the answer key' problem)
- Train/test split: honest evaluation
- Validation sets: tuning without peeking at test data
- Cross-validation: more reliable than a single split
- Overfitting vs underfitting: recognizing and fixing them
- Learning curves: your diagnostic tool

Real engineers don't just build models — they build models they can TRUST.
Trustworthy evaluation is what separates ML engineers from ML beginners.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, learning_curve
)
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

np.random.seed(42)

# ===========================================================================
# PART 1: The Problem With Testing on Training Data
# ===========================================================================
print("=" * 60)
print("PART 1: The Cheating Problem")
print("=" * 60)

print("""
Analogy: Imagine a student who memorizes the EXACT answers to 100 practice 
problems. On the practice test: 100/100. On the real exam (new questions): 
they fail completely.

This is OVERFITTING: the model memorizes training data instead of learning 
general patterns. It performs perfectly on data it has seen, but fails on 
new data.

WHY this matters: in production, your model ONLY sees new data.
A model with 99% training accuracy and 60% test accuracy is USELESS.
""")

# Demonstrate overfitting with polynomial regression
# True relationship: y = 2x + noise (linear)
n = 30
X_all = np.linspace(0, 10, n).reshape(-1, 1)
y_all = 2 * X_all.ravel() + np.random.normal(0, 2, n)

# Degree-15 polynomial — wildly overfits
from sklearn.pipeline import Pipeline
poly_15 = Pipeline([
    ('poly', PolynomialFeatures(degree=15)),
    ('reg',  LinearRegression())
])
poly_15.fit(X_all, y_all)
train_r2_15 = r2_score(y_all, poly_15.predict(X_all))

# Degree-1 polynomial — appropriate model
poly_1 = Pipeline([
    ('poly', PolynomialFeatures(degree=1)),
    ('reg',  LinearRegression())
])
poly_1.fit(X_all, y_all)
train_r2_1 = r2_score(y_all, poly_1.predict(X_all))

print(f"Degree-15 polynomial training R²: {train_r2_15:.4f}  (near perfect!)")
print(f"Degree-1  linear  model  training R²: {train_r2_1:.4f}  (lower — why?)")
print()
print("If we tested on NEW points from the same distribution:")

# Generate new test points
X_test_honest = np.linspace(0, 10, 50).reshape(-1, 1)
y_test_honest = 2 * X_test_honest.ravel() + np.random.normal(0, 2, 50)

test_r2_15 = r2_score(y_test_honest, poly_15.predict(X_test_honest))
test_r2_1  = r2_score(y_test_honest, poly_1.predict(X_test_honest))

print(f"Degree-15 polynomial TEST R²:  {test_r2_15:.4f}  ← terrible on new data!")
print(f"Degree-1  linear  model TEST R²: {test_r2_1:.4f}  ← generalizes well")

# ===========================================================================
# PART 2: Train/Test Split — The Solution
# ===========================================================================
print("\n" + "=" * 60)
print("PART 2: Train/Test Split")
print("=" * 60)

print("""
SOLUTION: Before training, hide some data from the model.

  All Data (100%)
  ├── Training Set (80%): model sees this, learns from it
  └── Test Set    (20%): model NEVER sees this during training
                          only used ONCE to report final performance

Key rules:
1. Split BEFORE any preprocessing (scaler fits on train only)
2. Use the test set ONLY ONCE at the very end
3. If you tune hyperparameters using test data, it's 'leaking'
   → use a VALIDATION set or cross-validation for tuning
""")

# Create a realistic dataset
n_samples = 200
study_hours = np.random.uniform(0, 10, n_samples)
attendance  = np.random.uniform(50, 100, n_samples)
prior_grade = np.random.uniform(50, 95, n_samples)

# True formula (with noise)
final_grade = (
    3.5 * study_hours + 0.4 * attendance + 0.3 * prior_grade
    + np.random.normal(0, 5, n_samples)
)
final_grade = np.clip(final_grade, 0, 100)

X = np.column_stack([study_hours, attendance, prior_grade])
y = final_grade

# The split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,     # 20% goes to test
    random_state=42    # reproducible split
)

print(f"Total samples: {len(X)}")
print(f"Training set:  {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"Test set:      {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")

# Train and evaluate
model = LinearRegression()
model.fit(X_train, y_train)

train_r2  = r2_score(y_train, model.predict(X_train))
test_r2   = r2_score(y_test,  model.predict(X_test))
train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
test_rmse  = np.sqrt(mean_squared_error(y_test,  model.predict(X_test)))

print(f"\nModel performance:")
print(f"  Training R²:  {train_r2:.4f}")
print(f"  Test R²:      {test_r2:.4f}  ← honest evaluation")
print(f"  Training RMSE: {train_rmse:.2f} grade points")
print(f"  Test RMSE:     {test_rmse:.2f} grade points")

gap = train_r2 - test_r2
if gap < 0.05:
    print(f"  Gap: {gap:.4f} → Small gap: model generalizes well ✓")
elif gap < 0.15:
    print(f"  Gap: {gap:.4f} → Moderate gap: slight overfitting")
else:
    print(f"  Gap: {gap:.4f} → Large gap: significant overfitting!")

# ===========================================================================
# PART 3: Train / Validation / Test — Three-Way Split
# ===========================================================================
print("\n" + "=" * 60)
print("PART 3: Validation Set (When Tuning Hyperparameters)")
print("=" * 60)

print("""
Problem: What if you try 100 different models, each time checking test
performance to pick the best? Now your test set is contaminated!

Solution: Three-way split:
  Training   (60%): fit the model
  Validation (20%): tune hyperparameters and select model
  Test       (20%): final honest evaluation — report this number

The test set should be looked at ONCE and ONLY ONCE.
""")

# Three-way split
X_temp, X_test2, y_temp, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)
X_train2, X_val2, y_train2, y_val2 = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Three-way split sizes:")
print(f"  Training:   {len(X_train2)} ({len(X_train2)/len(X)*100:.0f}%)")
print(f"  Validation: {len(X_val2)} ({len(X_val2)/len(X)*100:.0f}%)")
print(f"  Test:       {len(X_test2)} ({len(X_test2)/len(X)*100:.0f}%)")

# ===========================================================================
# PART 4: Cross-Validation — More Reliable than a Single Split
# ===========================================================================
print("\n" + "=" * 60)
print("PART 4: Cross-Validation")
print("=" * 60)

print("""
Problem with single train/test split:
  - If you get unlucky, your test set might be unusually easy or hard
  - Performance estimate has HIGH VARIANCE (it changes with random_state!)

Solution: K-Fold Cross-Validation
  1. Split data into K folds (e.g., K=5)
  2. For each fold:
     a. Use K-1 folds for training
     b. Use 1 fold for validation
  3. Average the K scores

This uses ALL data for both training and validation!
Result: lower variance estimate of true model performance.
""")

# Demonstrate variance of single splits
single_split_scores = []
for seed in range(20):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    m = LinearRegression()
    m.fit(X_tr, y_tr)
    single_split_scores.append(r2_score(y_te, m.predict(X_te)))

print(f"20 different random single splits:")
print(f"  R² scores: {[f'{s:.3f}' for s in single_split_scores[:10]]} ...")
print(f"  Mean: {np.mean(single_split_scores):.4f}")
print(f"  Std:  {np.std(single_split_scores):.4f}  ← high variance!")

# K-Fold Cross-Validation
cv_scores = cross_val_score(
    LinearRegression(), X, y,
    cv=5,              # 5-fold
    scoring='r2'
)

print(f"\n5-Fold Cross-Validation:")
print(f"  Fold scores: {[f'{s:.3f}' for s in cv_scores]}")
print(f"  Mean ± Std:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  More reliable estimate than any single split!")

# ===========================================================================
# PART 5: Overfitting vs Underfitting — Diagnosing With Learning Curves
# ===========================================================================
print("\n" + "=" * 60)
print("PART 5: Diagnosing With Learning Curves")
print("=" * 60)

print("""
Learning curve: plot train and validation score vs. number of training samples.

  HIGH BIAS (Underfitting):
    - Both train and validation scores are LOW
    - Adding more data doesn't help much
    - Fix: more complex model, more features

  HIGH VARIANCE (Overfitting):
    - Train score HIGH, validation score LOW
    - Large gap between the two curves
    - Fix: more data, regularization, simpler model, dropout
""")

def plot_learning_curve(estimator, X, y, title, filename, cv=5):
    """Plot learning curve showing train vs validation score."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        n_jobs=1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='steelblue')
    ax.plot(train_sizes, val_mean, 'o-', color='coral', label='Validation score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='coral')
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R² Score")
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_ylim(-0.2, 1.1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

# Good model (linear) — balanced bias/variance
plot_learning_curve(
    LinearRegression(), X, y,
    title="Learning Curve: Linear Regression (Well-fitted)",
    filename=OUTPUT_DIR / "learning_curve_linear.png"
)
print(f"Saved: {OUTPUT_DIR}/learning_curve_linear.png")

# Overfit model (degree-12 polynomial)
overfit_model = Pipeline([
    ('poly', PolynomialFeatures(degree=12)),
    ('reg',  LinearRegression())
])
plot_learning_curve(
    overfit_model, X, y,
    title="Learning Curve: Degree-12 Polynomial (Overfit)",
    filename=OUTPUT_DIR / "learning_curve_overfit.png"
)
print(f"Saved: {OUTPUT_DIR}/learning_curve_overfit.png")

# ===========================================================================
# PART 6: Summary — Rules of Honest Evaluation
# ===========================================================================
print("\n" + "=" * 60)
print("SUMMARY: Rules of Honest Model Evaluation")
print("=" * 60)

print("""
RULE 1: Always split data BEFORE touching it (before scaling, before anything)

RULE 2: Fit preprocessors (scalers, encoders) on TRAINING data only.
        Apply (transform) to test data using training statistics.
        Never fit on test data — that would leak test info into training.

RULE 3: Use cross-validation for model selection and hyperparameter tuning.
        Use the test set ONCE to report final performance.

RULE 4: A small train/test gap → good generalization
        A large gap → overfitting → need regularization or more data

RULE 5: Report both train AND test scores. Anyone reporting only training
        scores either doesn't know what they're doing or is hiding something.

Common train/test split sizes:
  - Small datasets (<1000 samples): 80/20 or use K-fold CV
  - Medium datasets (1000-100K):    80/20 split with 5-fold CV for tuning
  - Large datasets (>1M):           90/10 or even 99/1 split

Next: 02_scikit_learn/ — The sklearn API that makes all this easy
""")
