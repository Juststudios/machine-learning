"""
Solutions: Classification (Module 4)
=====================================
Reference solutions for Module 4 exercises:
  - Level 1: Recall (Conceptual & Theoretical Q&A)
  - Level 2: Debugging (Data Leakage & Train/Test Evaluation Bugs)
  - Level 3: Application (Fault Detection Pipeline: RF vs Logistic Regression)
  - Level 4: Challenge (K-Nearest Neighbors Classifier From Scratch)
"""

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True, parents=True)
np.random.seed(42)

print("=" * 65)
print("MODULE 4 SOLUTIONS: CLASSIFICATION")
print("=" * 65)

# ===========================================================================
# LEVEL 1: RECALL SOLUTIONS
# ===========================================================================
print("\n--- LEVEL 1: RECALL SOLUTIONS ---")

print("""
1a. How many classes and features in Iris?
    Answer: 3 classes (setosa, versicolor, virginica; 50 samples each),
            4 features (sepal length, sepal width, petal length, petal width).

1b. Baseline accuracy predicting the majority class:
    Answer: 0.3333 (1/3). Since all 3 classes have 50 samples, predicting
            any single class yields 50/150 = 33.33% accuracy.

1c. What does predict_proba() return that predict() doesn't?
    Answer: predict_proba() returns calibrated probability estimates [P(y=c|x)]
            for every class summing to 1.0, enabling threshold tuning,
            uncertainty estimation, and risk-weighted decision making.
            predict() only returns the argmax discrete class label.

1d. Advantage of Decision Tree over Logistic Regression:
    Answer: Decision trees naturally model non-linear boundaries and high-order
            feature interactions without requiring manual polynomial features,
            and they produce intuitive, human-interpretable if-then decision paths.

1e. Why scale features for KNN but NOT for Random Forest?
    Answer: KNN computes geometric Euclidean distances: ||x_a - x_b||_2.
            A feature with scale 0-3000 (e.g. RPM) completely dominates a feature
            with scale 0-5 (e.g. vibration). Random Forests make axis-aligned
            splits (x_j <= threshold), which are strictly invariant to any
            monotonic transformation of individual features.
""")

# ===========================================================================
# LEVEL 2: DEBUGGING SOLUTIONS
# ===========================================================================
print("=" * 65)
print("LEVEL 2: DEBUGGING SOLUTIONS")
print("=" * 65)

print("""
BUG 1 ANALYSIS:
  - Cause: `scaler.fit_transform(X)` is called BEFORE `train_test_split`.
    This causes DATA LEAKAGE: the mean and standard deviation of the test set
    leak into the training set, giving overly optimistic validation results.
  - Fix: First split into X_train and X_test. Then call `scaler.fit_transform(X_train)`
    and only `scaler.transform(X_test)`.

BUG 2 ANALYSIS:
  - Cause: `y_pred = model.predict(X_train)` evaluates performance on the training
    set, while the print statement reports it as "Test accuracy".
  - Fix: Predict on `X_test` using `y_pred = model.predict(X_test)` and score
    against `y_test`.
""")

print("Executing Corrected Code:")
X_dbg, y_dbg = make_classification(n_samples=500, n_features=10, random_state=42)

# Step 1: Split first!
X_tr_dbg, X_te_dbg, y_tr_dbg, y_te_dbg = train_test_split(
    X_dbg, y_dbg, test_size=0.2, random_state=42, stratify=y_dbg
)

# Step 2: Fit scaler ONLY on train data
scaler_dbg = StandardScaler()
X_tr_dbg_sc = scaler_dbg.fit_transform(X_tr_dbg)
X_te_dbg_sc = scaler_dbg.transform(X_te_dbg)

# Step 3: Train model
model_dbg = SVC(random_state=42)
model_dbg.fit(X_tr_dbg_sc, y_tr_dbg)

# Step 4: Evaluate on unseen test data
y_te_dbg_pred = model_dbg.predict(X_te_dbg_sc)
correct_test_acc = accuracy_score(y_te_dbg_pred, y_te_dbg)
print(f"Corrected Unbiased Test Accuracy: {correct_test_acc:.4f}")

# ===========================================================================
# LEVEL 3: APPLICATION SOLUTIONS — Fault Detection Pipeline
# ===========================================================================
print("\n" + "=" * 65)
print("LEVEL 3: APPLICATION SOLUTIONS — FAULT DETECTION")
print("=" * 65)

n = 600
n_ok, n_warn, n_fault = 360, 150, 90

np.random.seed(42)
# Realistic sensor telemetry:
# Temperature, Vibration, RPM, Current, Pressure, Humidity, Voltage, Acoustic
X_ok = np.random.randn(n_ok, 8) + np.array([70.0, 1.0, 3000.0, 15.0, 5.0, 60.0, 220.0, 20.0])
X_warn = np.random.randn(n_warn, 8) + np.array([82.0, 2.5, 2800.0, 19.0, 5.8, 62.0, 218.0, 35.0])
X_fault = np.random.randn(n_fault, 8) + np.array([96.0, 5.5, 2500.0, 27.0, 7.0, 65.0, 213.0, 60.0])

X_cls = np.vstack([X_ok, X_warn, X_fault])
y_cls = np.array([0] * n_ok + [1] * n_warn + [2] * n_fault)

# 1. Stratified train/test split (70% train, 30% test)
X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(
    X_cls, y_cls, test_size=0.30, random_state=42, stratify=y_cls
)

# 2. Standardize features
scaler_app = StandardScaler()
X_train_app_sc = scaler_app.fit_transform(X_train_app)
X_test_app_sc = scaler_app.transform(X_test_app)

# 3. Train models
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_app_sc, y_train_app)

lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train_app_sc, y_train_app)

# 4. Evaluate Random Forest
y_pred_rf = rf_clf.predict(X_test_app_sc)
acc_rf = accuracy_score(y_test_app, y_pred_rf)
f1_rf = f1_score(y_test_app, y_pred_rf, average='macro')
rec_fault_rf = recall_score(y_test_app, y_pred_rf, average=None)[2]
cv_rf = cross_val_score(rf_clf, X_train_app_sc, y_train_app, cv=5, scoring='f1_macro').mean()

# 5. Evaluate Logistic Regression
y_pred_lr = lr_clf.predict(X_test_app_sc)
acc_lr = accuracy_score(y_test_app, y_pred_lr)
f1_lr = f1_score(y_test_app, y_pred_lr, average='macro')
rec_fault_lr = recall_score(y_test_app, y_pred_lr, average=None)[2]
cv_lr = cross_val_score(lr_clf, X_train_app_sc, y_train_app, cv=5, scoring='f1_macro').mean()

print(f"{'Metric':<25} {'Random Forest':<15} {'Logistic Regression':<15}")
print("-" * 55)
print(f"{'Test Accuracy':<25} {acc_rf:<15.4f} {acc_lr:<15.4f}")
print(f"{'Macro F1':<25} {f1_rf:<15.4f} {f1_lr:<15.4f}")
print(f"{'5-Fold CV Macro F1':<25} {cv_rf:<15.4f} {cv_lr:<15.4f}")
print(f"{'Class 2 (FAULT) Recall':<25} {rec_fault_rf:<15.4f} {rec_fault_lr:<15.4f}")

print("\nRandom Forest Classification Report:")
print(classification_report(y_test_app, y_pred_rf, target_names=['OK', 'WARNING', 'FAULT']))

print("Engineering Decision:")
if rec_fault_rf >= rec_fault_lr:
    print("Random Forest provides optimal or equal recall on critical FAULT cases,")
    print("preventing catastrophic undetected machine failures.")
else:
    print("Logistic Regression provides superior or equal recall on critical FAULT cases.")

# ===========================================================================
# LEVEL 4: CHALLENGE SOLUTIONS — KNN From Scratch
# ===========================================================================
print("\n" + "=" * 65)
print("LEVEL 4: CHALLENGE SOLUTIONS — KNN FROM SCRATCH")
print("=" * 65)


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier from scratch using NumPy.
    
    Parameters
    ----------
    k : int, default=5
        Number of nearest neighbors to consider.
    """

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store training data (lazy learner).
        """
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        """
        Predict majority vote class for each query vector in X.
        
        Uses efficient vectorized pairwise Euclidean distance computation:
        ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
        """
        if self.X_train is None:
            raise RuntimeError("Classifier must be fitted before predict.")
        X = np.asarray(X, dtype=float)

        # Vectorized pairwise squared Euclidean distance:
        # dists[i, j] = ||X[i] - X_train[j]||^2
        x_sq = np.sum(X ** 2, axis=1, keepdims=True)
        tr_sq = np.sum(self.X_train ** 2, axis=1, keepdims=True).T
        cross = X @ self.X_train.T
        dists = x_sq + tr_sq - 2.0 * cross
        # Numerical safeguard against small negative numbers from floating point inaccuracies
        dists = np.maximum(dists, 0.0)

        # Find indices of top k smallest distances
        k_indices = np.argsort(dists, axis=1)[:, :self.k]

        # Gather labels for k nearest neighbors
        k_labels = self.y_train[k_indices]

        # Majority vote
        predictions = []
        for labels in k_labels:
            unique, counts = np.unique(labels, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            predictions.append(majority_class)

        return np.array(predictions)

    def score(self, X, y):
        """Return accuracy score on given test data and labels."""
        return accuracy_score(y, self.predict(X))


# Evaluate KNN on Iris dataset across various K values
iris = load_iris()
X_ir, y_ir = iris.data, iris.target
X_tr_ir, X_te_ir, y_tr_ir, y_te_ir = train_test_split(
    X_ir, y_ir, test_size=0.3, random_state=42, stratify=y_ir
)

scaler_ir = StandardScaler()
X_tr_ir_sc = scaler_ir.fit_transform(X_tr_ir)
X_te_ir_sc = scaler_ir.transform(X_te_ir)

print(f"{'K':<5} {'Scratch KNN Accuracy':<25} {'Sklearn KNN Accuracy':<25} {'Difference':<10}")
print("-" * 65)

for k_val in [3, 5, 7, 11]:
    knn_scratch = KNNClassifier(k=k_val)
    knn_scratch.fit(X_tr_ir_sc, y_tr_ir)
    acc_sc = knn_scratch.score(X_te_ir_sc, y_te_ir)

    knn_sk = KNeighborsClassifier(n_neighbors=k_val)
    knn_sk.fit(X_tr_ir_sc, y_tr_ir)
    acc_sk = knn_sk.score(X_te_ir_sc, y_te_ir)

    diff = abs(acc_sc - acc_sk)
    print(f"{k_val:<5} {acc_sc:<25.4f} {acc_sk:<25.4f} {diff:<10.4f}")
    assert diff <= 0.03, f"Scratch KNN deviates significantly from sklearn for K={k_val}"

print("\n" + "=" * 65)
print("ALL MODULE 4 EXERCISE SOLUTIONS VERIFIED SUCCESSFULLY!")
print("=" * 65)
