"""
Exercises: Classification (Module 4)
======================================
4-tier exercises covering logistic regression, decision trees,
random forests, SVM, and KNN.
"""

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
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
print("CLASSIFICATION — EXERCISES")
print("=" * 60)

# ===========================================================================
# LEVEL 1: RECALL
# ===========================================================================
print("\n--- LEVEL 1: RECALL ---")

print("""
Questions (run code below, read output, then answer):
1a. Load the Iris dataset. How many classes are there? How many features?
1b. What is the baseline accuracy if you predict the majority class?
1c. What does predict_proba() return that predict() doesn't?
1d. Name one advantage of a Decision Tree over Logistic Regression.
1e. Why do we need to scale features for KNN but NOT for Random Forest?
""")

# Demo: Load Iris, try different classifiers
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

print(f"Iris dataset: {X_iris.shape[0]} samples, {X_iris.shape[1]} features, {len(np.unique(y_iris))} classes")
print(f"Classes: {iris.target_names}")
print(f"Baseline accuracy (majority class): {np.bincount(y_iris).max() / len(y_iris):.4f}")

lr_iris = LogisticRegression(max_iter=200)
lr_iris.fit(X_tr, y_tr)
print(f"LogisticRegression accuracy: {lr_iris.score(X_te, y_te):.4f}")
print(f"predict_proba sample: {lr_iris.predict_proba(X_te[:2]).round(4)}")

# ===========================================================================
# LEVEL 2: DEBUGGING
# ===========================================================================
print("\n--- LEVEL 2: DEBUGGING ---")
print("""
The following code produces misleadingly good results. Find the 2 bugs.
""")

buggy_code = '''
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=500, n_features=10, random_state=42)

# BUG 1: Scaler fitted on ALL data before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = SVC()
model.fit(X_train, y_train)

# BUG 2: Evaluating on training data and calling it "test accuracy"
y_pred = model.predict(X_train)
print(f"Test accuracy: {accuracy_score(y_train, y_pred):.4f}")
'''
print(buggy_code)
print("""
Bug 1 cause and fix: ____________________________________
Bug 2 cause and fix: ____________________________________
""")

print("TODO: Write the corrected version below")

# ===========================================================================
# LEVEL 3: APPLICATION — Fault Detection Classifier
# ===========================================================================
print("\n--- LEVEL 3: APPLICATION ---")
print("""
TASK: Build a multi-class fault detection classifier.

Dataset (generate it):
  - 8 sensor features: temperature, vibration, rpm, current,
                        pressure, humidity, voltage, acoustic
  - 3 classes: 0=OK (60%), 1=WARNING (25%), 2=FAULT (15%)
  - n=600 samples

Requirements:
  1. Generate synthetic sensor data with realistic class separation
     (FAULT has higher temperature, vibration, current than OK)
  2. Split: 70% train, 30% test (stratified)
  3. Scale features
  4. Train: RandomForestClassifier and LogisticRegression
  5. Evaluate each with:
     - Overall accuracy
     - Classification report (precision, recall, F1 per class)
     - 5-fold cross-validation macro-F1
  6. Which model is better for detecting FAULT specifically?
     (Hint: look at per-class recall for class 2)
""")

# Starter data generation:
n = 600
n_ok, n_warn, n_fault = 360, 150, 90

# Normal sensor readings for OK machines
X_ok = np.random.randn(n_ok, 8) + np.array([70, 1.0, 3000, 15, 5, 60, 220, 20])
# Elevated readings for WARNING
X_warn = np.random.randn(n_warn, 8) + np.array([82, 2.5, 2800, 19, 5.8, 62, 218, 35])
# High readings for FAULT
X_fault = np.random.randn(n_fault, 8) + np.array([96, 5.5, 2500, 27, 7, 65, 213, 60])

X_cls = np.vstack([X_ok, X_warn, X_fault])
y_cls = np.array([0]*n_ok + [1]*n_warn + [2]*n_fault)

print(f"Dataset: X.shape={X_cls.shape}, classes={np.bincount(y_cls)}")
print("TODO: Complete the classification pipeline above")
print("  Train RandomForest + LogisticRegression")
print("  Report accuracy, F1, and cross-validation scores")


# ===========================================================================
# LEVEL 4: CHALLENGE — KNN From Scratch
# ===========================================================================
print("\n--- LEVEL 4: CHALLENGE ---")
print("""
TASK: Implement K-Nearest Neighbors from scratch using numpy.

Requirements:
  - class KNNClassifier:
      __init__(self, k=5): store k
      fit(self, X, y): store training data
      predict(self, X): return predicted class for each test sample
        - For each test sample:
          1. Compute Euclidean distance to all training samples
          2. Find the k nearest neighbors
          3. Return the majority class among those k neighbors
  
  - Verify: your KNN should match sklearn's KNeighborsClassifier
    on the same data (within ~1% due to tie-breaking)
  
  - Test on Iris dataset with K=3, 5, 7, 11 and report accuracy

Starter:
""")

class KNNClassifier:
    """TODO: Implement KNN from scratch."""
    
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        # TODO: Store training data
        pass
    
    def predict(self, X):
        # TODO: For each test sample, find k nearest neighbors
        # and return majority vote class
        predictions = []
        for x in X:
            # Step 1: Compute distances to all training points
            # distances = np.sqrt(((self.X_train - x) ** 2).sum(axis=1))
            
            # Step 2: Get indices of k nearest
            # k_nearest_idx = np.argsort(distances)[:self.k]
            
            # Step 3: Majority vote
            # k_labels = self.y_train[k_nearest_idx]
            # majority = np.bincount(k_labels).argmax()
            
            predictions.append(0)  # REPLACE with actual prediction
        return np.array(predictions)

print("TODO: Complete KNNClassifier and test it on Iris dataset")

# Reference: sklearn KNN
sk_knn = KNeighborsClassifier(n_neighbors=5)
sk_knn.fit(X_tr, y_tr)
sk_acc = sk_knn.score(X_te, y_te)
print(f"sklearn KNN accuracy: {sk_acc:.4f}")
print("Your implementation should get close to this value")

print("\n" + "=" * 60)
print("EXERCISES COMPLETE")
print("Check solutions/classification_solutions.py for reference answers.")
print("=" * 60)
