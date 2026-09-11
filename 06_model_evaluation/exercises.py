"""
Exercises: Model Evaluation (Module 6)
========================================
4-tier exercises covering metrics, cross-validation, hyperparameter
tuning, bias-variance tradeoff, and handling imbalanced data.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, learning_curve, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, r2_score, mean_squared_error
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
np.random.seed(42)

print("=" * 60)
print("MODEL EVALUATION — EXERCISES")
print("=" * 60)

# ===========================================================================
# LEVEL 1: RECALL
# ===========================================================================
print("\n--- LEVEL 1: RECALL ---")
print("""
Study the results and answer the questions:

1a. If precision=0.80, recall=0.60: what does this mean for a fraud detector?
1b. Which metric should you optimize for: a cancer screening test?
    Options: Accuracy / Precision / Recall / F1
    Explain why.
1c. What is overfitting? Give a diagnostic signature using train/val scores.
1d. What is the difference between cross_val_score() and hold-out test R²?
1e. Why should you use StratifiedKFold for imbalanced classification?
""")

X, y = make_classification(n_samples=500, n_features=10, weights=[0.8, 0.2], random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=300)
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)

print(f"Accuracy:  {accuracy_score(y_te, y_pred):.4f}")
print(f"Precision: {precision_score(y_te, y_pred):.4f}")
print(f"Recall:    {recall_score(y_te, y_pred):.4f}")
print(f"F1:        {f1_score(y_te, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_te, model.predict_proba(X_te)[:,1]):.4f}")
print(f"Confusion matrix:\n{confusion_matrix(y_te, y_pred)}")

# ===========================================================================
# LEVEL 2: DEBUGGING
# ===========================================================================
print("\n--- LEVEL 2: DEBUGGING ---")
print("""
This evaluation code makes 3 evaluation mistakes. Find them.
""")

buggy = '''
import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

np.random.seed(0)
X, y = make_classification(n_samples=1000, n_features=20,
                             weights=[0.95, 0.05], random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# BUG 1: Training an unlimited decision tree and reporting training accuracy as "model quality"
dt = DecisionTreeClassifier()   # no max_depth — will overfit
dt.fit(X_train, y_train)
print(f"Model quality: {accuracy_score(y_train, dt.predict(X_train)):.4f}")  # train accuracy!

# BUG 2: Using accuracy for severely imbalanced data (95%/5% split)
print(f"Test accuracy: {accuracy_score(y_test, dt.predict(X_test)):.4f}")

# BUG 3: Cross-validating with accuracy on imbalanced data
cv_scores = cross_val_score(dt, X, y, cv=5, scoring="accuracy")
print(f"CV accuracy: {cv_scores.mean():.4f}")
'''
print(buggy)
print("""
Bug 1: ________________________________________________
Bug 2: ________________________________________________
Bug 3: ________________________________________________
TODO: Write the corrected version using appropriate metrics
""")

# ===========================================================================
# LEVEL 3: APPLICATION — Fault Detection Evaluation Suite
# ===========================================================================
print("\n--- LEVEL 3: APPLICATION ---")
print("""
TASK: Build a comprehensive evaluation suite for fault detection.

Dataset:
  - 3-class imbalanced: OK=70%, WARNING=20%, FAULT=10%
  - 800 samples, 8 sensor features

Requirements:
  1. Generate the dataset (realistic class imbalance)
  2. Split: 75% train, 25% test (stratified)
  3. Train two models:
     a. LogisticRegression with class_weight='balanced'
     b. RandomForestClassifier with class_weight='balanced'
  4. Evaluate EACH model with:
     - Confusion matrix (plot heatmap, save to output/)
     - Full classification_report (per-class precision, recall, F1)
     - Macro-averaged F1
     - 5-fold stratified cross-validation (report macro-F1 mean ± std)
  5. For FAULT detection specifically:
     - Which model has higher recall for class 2 (FAULT)?
     - What is the cost of a missed FAULT vs a false alarm?
     - Which model would you choose and why?
  6. Plot learning curves for the best model:
     - Train vs validation macro-F1 vs training set size
     - Save to output/evaluation_learning_curve.png
""")

# Starter data generation
n = 800
n_ok    = int(0.70 * n)
n_warn  = int(0.20 * n)
n_fault = n - n_ok - n_warn

np.random.seed(42)
X_ok    = np.random.randn(n_ok, 8) * 0.5 + np.array([70, 1.0, 3000, 15, 5, 60, 220, 20])
X_warn  = np.random.randn(n_warn, 8) * 0.8 + np.array([82, 2.5, 2800, 19, 5.8, 62, 218, 35])
X_fault = np.random.randn(n_fault, 8) * 1.2 + np.array([95, 5.5, 2500, 27, 7, 65, 213, 60])
X_ev = np.vstack([X_ok, X_warn, X_fault])
y_ev = np.array([0]*n_ok + [1]*n_warn + [2]*n_fault)

print(f"Dataset: {X_ev.shape}, class counts: {np.bincount(y_ev)}")
print("TODO: Complete the evaluation suite above")

# ===========================================================================
# LEVEL 4: CHALLENGE — Implement Stratified K-Fold From Scratch
# ===========================================================================
print("\n--- LEVEL 4: CHALLENGE ---")
print("""
TASK: Implement Stratified K-Fold cross-validation from scratch.

def stratified_kfold_cv(model, X, y, k=5, scoring='accuracy'):
    '''
    Performs stratified k-fold cross-validation.
    
    Stratified: each fold has approximately the same class distribution
                as the full dataset (critical for imbalanced data!)
    
    Algorithm:
      1. For each class c:
         - Get indices where y == c
         - Split those indices into K roughly equal parts
      2. For each fold i (0..K-1):
         - val_idx = fold i from each class
         - train_idx = all other folds from each class
         - Fit model on train, evaluate on val
      3. Return list of K scores
    '''
    scores = []
    # YOUR IMPLEMENTATION HERE
    return scores

Verify:
  - Compare your scores with sklearn's StratifiedKFold cross_val_score
  - They should be identical (or very close) on the same model/data
  - Prove stratification: print class distribution in each fold
    (should be approximately equal to full dataset distribution)
""")

from sklearn.model_selection import StratifiedKFold

def stratified_kfold_cv(model, X, y, k=5):
    """TODO: Implement this function."""
    scores = []
    # HINT: 
    # class_indices = {c: np.where(y==c)[0] for c in np.unique(y)}
    # For each class, split its indices into K folds
    # Combine across classes for each fold
    return scores

# Reference
from sklearn.linear_model import LogisticRegression
sk_scores = cross_val_score(
    LogisticRegression(max_iter=300, class_weight='balanced'),
    X_ev, y_ev, cv=StratifiedKFold(5), scoring='f1_macro'
)
print(f"sklearn StratifiedKFold F1: {sk_scores.round(4)}")
print(f"  mean: {sk_scores.mean():.4f} ± {sk_scores.std():.4f}")
print("TODO: your stratified_kfold_cv should match this")

print("\n" + "=" * 60)
print("EXERCISES COMPLETE")
print("Check solutions/model_evaluation_solutions.py for reference answers.")
print("=" * 60)
