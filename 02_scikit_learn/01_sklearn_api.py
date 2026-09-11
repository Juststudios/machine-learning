"""
Scikit-learn Module 1: The Estimator API
==========================================
Why scikit-learn became the standard for classical ML in Python.

Key insight: Every sklearn model follows the SAME interface.
  - estimator.fit(X_train, y_train)   ← learn from data
  - estimator.predict(X_test)         ← make predictions
  - estimator.score(X_test, y_test)   ← evaluate performance
  - transformer.transform(X)          ← change data representation

This consistency means: learn the API once, use 100+ algorithms.
"""

import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import make_regression, make_classification
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

np.random.seed(42)

# ===========================================================================
# PART 1: The Consistent Estimator API
# ===========================================================================
print("=" * 60)
print("PART 1: The Consistent Estimator API")
print("=" * 60)

print("""
Every sklearn estimator follows this interface:
  fit(X, y)      → learns parameters from training data
  predict(X)     → returns predictions for new data  
  score(X, y)    → returns a default performance metric

This means you can swap algorithms with ONE LINE change!
""")

# Generate regression data
X_reg, y_reg = make_regression(n_samples=200, n_features=5, noise=15, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Generate classification data
X_cls, y_cls = make_classification(n_samples=200, n_features=5, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# 5 different REGRESSION estimators — all using the SAME API
print("Regression: 5 different algorithms, same API:")
regressors = [
    ("LinearRegression",    LinearRegression()),
    ("Ridge",               Ridge(alpha=1.0)),
    ("RandomForest",        RandomForestRegressor(n_estimators=50, random_state=42)),
    ("SVR",                 SVR(kernel='rbf')),
    ("PolynomialReg(deg=2)",Pipeline([('poly', PolynomialFeatures(2)), ('reg', LinearRegression())])),
]

for name, est in regressors:
    est.fit(X_train_r, y_train_r)            # SAME
    score = est.score(X_test_r, y_test_r)    # SAME
    print(f"  {name:30s}: R² = {score:.4f}")

# 3 different CLASSIFICATION estimators — all using the SAME API
print("\nClassification: 3 different algorithms, same API:")
classifiers = [
    ("LogisticRegression",  LogisticRegression(max_iter=200)),
    ("DecisionTree",        DecisionTreeClassifier(max_depth=5)),
    ("RandomForest",        __import__('sklearn.ensemble', fromlist=['RandomForestClassifier']).RandomForestClassifier(n_estimators=50, random_state=42)),
]

for name, est in classifiers:
    est.fit(X_train_c, y_train_c)
    score = est.score(X_test_c, y_test_c)
    print(f"  {name:30s}: Accuracy = {score:.4f}")

# ===========================================================================
# PART 2: Transformers — fit + transform
# ===========================================================================
print("\n" + "=" * 60)
print("PART 2: Transformers — fit() then transform()")
print("=" * 60)

print("""
Transformers CHANGE the data (rather than predicting).
They have two key methods:
  fit(X_train)        → compute statistics from training data
  transform(X)        → apply the transformation to any X

CRITICAL RULE: Always fit on TRAINING data, then transform BOTH train and test.
               Never fit on test data — that would 'cheat' by using test info.
""")

# Correct scaler usage
scaler = StandardScaler()

# FIT: compute mean and std from training data ONLY
scaler.fit(X_train_r)

print(f"Scaler learned from training data:")
print(f"  Mean of feature 0: {scaler.mean_[0]:.4f}")
print(f"  Std  of feature 0: {scaler.scale_[0]:.4f}")

# TRANSFORM: apply the same scaling to train AND test
X_train_scaled = scaler.transform(X_train_r)
X_test_scaled  = scaler.transform(X_test_r)  # ← uses training mean/std!

print(f"\nAfter scaling:")
print(f"  X_train_scaled mean (col 0): {X_train_scaled[:, 0].mean():.6f}  (≈ 0)")
print(f"  X_test_scaled  mean (col 0): {X_test_scaled[:, 0].mean():.4f}  (close but not exactly 0 — correct!)")

# fit_transform: convenience shortcut (only use on training data)
X_train_ft = scaler.fit_transform(X_train_r)  # same as fit then transform

# ===========================================================================
# PART 3: Inspecting a Model
# ===========================================================================
print("\n" + "=" * 60)
print("PART 3: Inspecting Estimator Parameters")
print("=" * 60)

lr = LinearRegression()
print("Default parameters of LinearRegression:")
print(f"  {lr.get_params()}")

# Set parameters
lr.set_params(fit_intercept=False)
print(f"\nAfter set_params(fit_intercept=False):")
print(f"  {lr.get_params()}")

# After fitting: access learned parameters (attributes end with _)
lr2 = LinearRegression()
lr2.fit(X_train_r, y_train_r)
print(f"\nAfter fitting, learned attributes (end with _):")
print(f"  coef_    shape: {lr2.coef_.shape}")
print(f"  intercept_:     {lr2.intercept_:.4f}")
print(f"  n_features_in_: {lr2.n_features_in_}")

# ===========================================================================
# PART 4: Pipelines — Chain Transformers + Estimator
# ===========================================================================
print("\n" + "=" * 60)
print("PART 4: Pipelines — The Right Way to Combine Steps")
print("=" * 60)

print("""
Problem without Pipeline:
  scaler.fit(X_train)
  X_train_s = scaler.transform(X_train)
  model.fit(X_train_s, y_train)
  X_test_s  = scaler.transform(X_test)
  model.predict(X_test_s)
  
  → Easy to forget to scale test data
  → Easy to accidentally fit scaler on test data
  → Messy code

Solution: Pipeline wraps everything into one object!
  pipeline = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
  pipeline.fit(X_train, y_train)    ← fits scaler then model (correctly)
  pipeline.predict(X_test)          ← scales then predicts (correctly)
""")

# Build a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LinearRegression()),
])

# fit: scaler.fit(X_train) → scaler.transform(X_train) → model.fit(scaled_X_train)
pipeline.fit(X_train_r, y_train_r)

# predict: scaler.transform(X_test) → model.predict(scaled_X_test)
score = pipeline.score(X_test_r, y_test_r)
print(f"Pipeline R²: {score:.4f}")

# Access steps
print(f"\nPipeline steps:")
for step_name, step_obj in pipeline.steps:
    print(f"  '{step_name}': {type(step_obj).__name__}")

# Access inner objects
print(f"\nAccess scaler mean from pipeline:")
print(f"  pipeline.named_steps['scaler'].mean_[:3] = {pipeline.named_steps['scaler'].mean_[:3].round(4)}")

# Pipeline with hyperparameter access: prefix with step name + __
ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge',  Ridge()),
])
print(f"\nRidge pipeline default params:")
print(f"  {ridge_pipe.get_params()}")
ridge_pipe.set_params(ridge__alpha=10.0)  # set alpha inside the pipeline
print(f"  After set_params(ridge__alpha=10.0):")
print(f"  alpha = {ridge_pipe.named_steps['ridge'].alpha}")

# ===========================================================================
# PART 5: Saving and Loading Models
# ===========================================================================
print("\n" + "=" * 60)
print("PART 5: Saving and Loading Models (joblib)")
print("=" * 60)

model_path = OUTPUT_DIR / "pipeline_model.joblib"

# Save the trained pipeline
joblib.dump(pipeline, model_path)
print(f"Model saved to: {model_path}")

# Load it back
loaded_pipeline = joblib.load(model_path)
loaded_score = loaded_pipeline.score(X_test_r, y_test_r)
print(f"Loaded model R²: {loaded_score:.4f} (matches original: {score:.4f})")

print("""
WHY joblib over pickle:
  - More efficient for large numpy arrays (used inside models)
  - Same simple interface: dump() / load()
  - Standard choice in the sklearn ecosystem

In production, you save the PIPELINE (including the scaler), not just the model.
Otherwise you'll need to manually scale inputs before prediction!
""")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. The sklearn API: fit() → predict() / transform() — consistent for all estimators
2. Transformers: fit on TRAINING only, transform on BOTH train and test
3. Pipeline: wraps preprocessing + model into one object (prevents data leakage)
4. get_params() / set_params() for inspecting and changing hyperparameters
5. Learned attributes end with _ (coef_, mean_, etc.)
6. Save full pipelines with joblib (not just the model)

Next: 02_preprocessing.py — All the data preparation tools
""")
