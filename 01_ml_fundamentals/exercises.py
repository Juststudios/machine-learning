"""
Exercises: ML Fundamentals
============================
Test your understanding of Modules 1-4.

Exercise Levels:
  Level 1 (Recall)       — Run the code, read the output, answer questions
  Level 2 (Understanding) — Debug broken code
  Level 3 (Application)  — Build something new
  Level 4 (Challenge)    — Implement core concepts from scratch

IMPORTANT: Try each exercise yourself before checking solutions/ml_fundamentals_solutions.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42)

print("=" * 60)
print("ML FUNDAMENTALS — EXERCISES")
print("=" * 60)

# ===========================================================================
# LEVEL 1: RECALL
# Questions you should be able to answer after reading the lessons.
# ===========================================================================
print("\n--- LEVEL 1: RECALL ---")
print("""
Run this code block and study the output. Then answer the questions in comments.

Questions:
  1a. What is the difference between supervised and unsupervised learning?
  1b. What are features (X) and what is the target (y)?
  1c. Why do we split data into train and test sets?
  1d. What does R² = 0.85 mean in plain English?
  1e. Name one problem caused by NOT scaling features.
""")

# Demo code to study:
X = np.array([[2, 70], [5, 90], [1, 60], [8, 95], [3, 75]])
y = np.array([65, 82, 55, 91, 68])

print("Features (X) — study_hours, attendance:")
print(X)
print("Target (y) — final_grade:")
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nModel coefficients: {model.coef_}")
print(f"Model intercept:    {model.intercept_:.2f}")
print(f"Test R²: {r2_score(y_test, model.predict(X_test)):.4f}")

# ===========================================================================
# LEVEL 2: UNDERSTANDING — Debug the broken code
# ===========================================================================
print("\n--- LEVEL 2: DEBUGGING ---")
print("""
The following ML pipeline has 3 bugs. Find and fix them.

HINT: Think about data leakage, correct evaluation, and scaling.
""")

# BUGGY CODE — DO NOT RUN AS-IS, find the bugs first
buggy_code = '''
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

np.random.seed(42)
X = np.random.randn(200, 5)
y = X[:, 0] * 3 + X[:, 1] * -2 + np.random.randn(200)

# BUG 1: Fit scaler on ALL data before splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ← What's wrong here?

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# BUG 2: Evaluate on training data and call it "test accuracy"
score = r2_score(y_train, model.predict(X_train))
print(f"Model accuracy: {score:.4f}")  # ← What's wrong here?

# BUG 3: Making a new prediction using unscaled input
new_sample = np.array([[1.5, -0.5, 0.3, 2.1, -1.2]])
prediction = model.predict(new_sample)  # ← What's wrong here?
print(f"Prediction: {prediction[0]:.2f}")
'''

print("Study the buggy code above (printed as string for safety):")
print(buggy_code)

print("""
TODO: Write the FIXED version below this line.
Fix all 3 bugs and explain what each bug caused.

Bug 1 fix: ________________________________________________
Bug 2 fix: ________________________________________________  
Bug 3 fix: ________________________________________________
""")

# YOUR FIXED CODE HERE:
# ---------------------------------------------------------------


# ===========================================================================
# LEVEL 3: APPLICATION
# Build a complete ML pipeline for sensor fault prediction
# ===========================================================================
print("\n--- LEVEL 3: APPLICATION ---")
print("""
TASK: Build an ML pipeline to predict power consumption from building data.

Dataset (generate it):
  - building_age (years): 1-50
  - num_floors: 1-30
  - avg_occupancy (people): 10-500
  - insulation_rating: 1-5 (ordinal, 1=poor, 5=excellent)
  - has_solar_panels: 0 or 1 (binary)
  - power_consumption_kWh (target): depends on above features

Requirements:
  1. Generate synthetic data (n=300 samples) with a realistic formula
  2. Split into train (80%) and test (20%)
  3. Scale numerical features (but NOT the binary or target)
  4. Train a LinearRegression model
  5. Evaluate with R² and RMSE on the TEST set
  6. Report: which features have the largest positive/negative coefficients?
  7. Cross-validate with 5-fold CV and report mean ± std R²

Expected test R² should be > 0.8 if your formula is correct.
""")

# YOUR CODE HERE:
# ---------------------------------------------------------------
print("TODO: Complete the Level 3 exercise above")
print("  Generate data, preprocess, train, evaluate, cross-validate")


# ===========================================================================
# LEVEL 4: CHALLENGE
# Implement cross-validation from scratch using numpy (NO sklearn CV)
# ===========================================================================
print("\n--- LEVEL 4: CHALLENGE ---")
print("""
TASK: Implement 5-fold cross-validation from scratch.

Requirements:
  - Use ONLY numpy and sklearn's LinearRegression + r2_score
  - NO cross_val_score, KFold, or any sklearn CV utility
  - Manually split data into 5 folds
  - For each fold: train on other 4 folds, evaluate on current fold
  - Report the R² for each fold and the mean ± std

Starter:
""")

# Starter data
n = 150
X_ch = np.random.randn(n, 3)
y_ch = X_ch[:, 0] * 2 - X_ch[:, 1] + 0.5 * X_ch[:, 2] + np.random.randn(n) * 0.5

print("Data for challenge:")
print(f"  X shape: {X_ch.shape}, y shape: {y_ch.shape}")
print()
print("YOUR TASK: Implement 5-fold CV from scratch below.")
print("Expected output should be similar to sklearn's cross_val_score result:")

# Reference answer using sklearn (for verification only)
ref_scores = cross_val_score(LinearRegression(), X_ch, y_ch, cv=5, scoring='r2')
print(f"  sklearn cross_val_score: {ref_scores.round(4)} | mean={ref_scores.mean():.4f}")

print("""
Now implement this manually:

k = 5
fold_size = n // k
scores = []

for fold_idx in range(k):
    # YOUR CODE: create train/val indices for this fold
    # YOUR CODE: fit model, compute R²
    pass

print(f"Manual CV scores: {scores}")
print(f"Mean ± Std: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
""")

# YOUR CODE HERE:
# ---------------------------------------------------------------


print("\n" + "=" * 60)
print("EXERCISES COMPLETE")
print("Check solutions/ml_fundamentals_solutions.py for reference answers.")
print("=" * 60)
