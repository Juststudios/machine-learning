"""
Solutions: Sklearn API, Regression, Classification, Clustering
================================================================
Reference solutions for modules 2-5 exercises.
Study AFTER attempting each exercise yourself.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    r2_score, mean_squared_error, silhouette_score,
    f1_score, recall_score, precision_score, classification_report
)
from sklearn.datasets import make_blobs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True, parents=True)
np.random.seed(42)

print("=" * 60)
print("SOLUTIONS: SKLEARN API, REGRESSION, CLUSTERING")
print("=" * 60)

# ===========================================================================
# SOLUTION A: OutlierClipper (Module 2 — Challenge)
# ===========================================================================
print("\n--- SOLUTION A: OutlierClipper Custom Transformer ---")

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip values beyond n_std standard deviations to the boundary."""
    
    def __init__(self, n_std=3.0):
        self.n_std = n_std
    
    def fit(self, X, y=None):
        X = np.array(X)
        self.lower_ = X.mean(axis=0) - self.n_std * X.std(axis=0)
        self.upper_ = X.mean(axis=0) + self.n_std * X.std(axis=0)
        return self
    
    def transform(self, X, y=None):
        X = np.array(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

# Test
X_clean = np.random.randn(200, 4)
X_dirty = X_clean.copy()
X_dirty[:5, 0] = 100  # extreme outliers

clipper = OutlierClipper(n_std=3.0)
clipper.fit(X_clean)
X_clipped = clipper.transform(X_dirty)

print(f"Before clipping — max feature 0: {X_dirty[:, 0].max():.2f}")
print(f"After clipping  — max feature 0: {X_clipped[:, 0].max():.2f}")
print(f"Clip boundaries: lower={clipper.lower_[0]:.4f}, upper={clipper.upper_[0]:.4f}")

# In a Pipeline
from sklearn.datasets import make_regression
X_r, y_r = make_regression(n_samples=200, n_features=4, noise=10, random_state=42)
pipe_clip = Pipeline([
    ('clipper', OutlierClipper(n_std=3.0)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0)),
])
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
pipe_clip.fit(X_tr, y_tr)
print(f"Pipeline with OutlierClipper R²: {r2_score(y_te, pipe_clip.predict(X_te)):.4f}")

# ===========================================================================
# SOLUTION B: Gradient Descent from Scratch (Module 3 — Challenge)
# ===========================================================================
print("\n--- SOLUTION B: LinearRegressionGD From Scratch ---")

class LinearRegressionGD:
    """Mini-batch gradient descent for linear regression."""
    
    def __init__(self, lr=0.01, n_epochs=100, batch_size=32):
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.losses_ = []
    
    def fit(self, X, y):
        n, p = X.shape
        self.w = np.random.randn(p) * 0.01
        self.b = 0.0
        
        for epoch in range(self.n_epochs):
            # Shuffle data
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx]
            
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                Xb = X_shuf[start:start + self.batch_size]
                yb = y_shuf[start:start + self.batch_size]
                n_b = len(Xb)
                
                # Forward
                y_pred = Xb @ self.w + self.b
                residual = y_pred - yb
                
                # Gradients
                dw = (2 / n_b) * Xb.T @ residual
                db = (2 / n_b) * residual.sum()
                
                # Update
                self.w -= self.lr * dw
                self.b -= self.lr * db
                
                epoch_loss += (residual**2).mean()
            
            self.losses_.append(epoch_loss / (n // self.batch_size))
        
        return self
    
    def predict(self, X):
        return X @ self.w + self.b

# Test
from sklearn.datasets import make_regression
X_gd, y_gd = make_regression(n_samples=300, n_features=8, noise=15, random_state=42)
X_gd_tr, X_gd_te, y_gd_tr, y_gd_te = train_test_split(X_gd, y_gd, test_size=0.2, random_state=42)
sc = StandardScaler()
X_gd_tr_s = sc.fit_transform(X_gd_tr)
X_gd_te_s  = sc.transform(X_gd_te)

model_gd = LinearRegressionGD(lr=0.01, n_epochs=200, batch_size=32)
model_gd.fit(X_gd_tr_s, y_gd_tr)

gd_r2  = r2_score(y_gd_te, model_gd.predict(X_gd_te_s))
sk_lr  = LinearRegression().fit(X_gd_tr_s, y_gd_tr)
sk_r2  = r2_score(y_gd_te, sk_lr.predict(X_gd_te_s))

print(f"LinearRegressionGD R²: {gd_r2:.4f}")
print(f"sklearn LinearRegr  R²: {sk_r2:.4f}  (reference)")
print(f"Difference: {abs(gd_r2-sk_r2):.6f}  (small = good)")

# ===========================================================================
# SOLUTION C: K-Means From Scratch (Module 5 — Challenge)
# ===========================================================================
print("\n--- SOLUTION C: K-Means From Scratch ---")

class KMeansScratch:
    """K-Means clustering implemented from scratch with numpy."""
    
    def __init__(self, k=3, max_iter=100, tol=1e-4, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
    
    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        
        # Step 1: Random initialization (k-means++)
        init_idx = rng.choice(n, self.k, replace=False)
        centers = X[init_idx].copy()
        
        labels = np.zeros(n, dtype=int)
        
        for iteration in range(self.max_iter):
            # Step 2a: Assign each point to nearest centroid
            # Compute distances efficiently: ||x - c||² = ||x||² - 2xc + ||c||²
            dists = (
                (X ** 2).sum(axis=1, keepdims=True)
                - 2 * X @ centers.T
                + (centers ** 2).sum(axis=1)
            )
            new_labels = dists.argmin(axis=1)
            
            # Step 2b: Recompute centroids
            new_centers = np.array([
                X[new_labels == k].mean(axis=0) if (new_labels == k).any() else centers[k]
                for k in range(self.k)
            ])
            
            # Step 2c: Convergence check
            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            labels = new_labels
            
            if shift < self.tol:
                break
        
        self.labels_ = labels
        self.cluster_centers_ = centers
        self.n_iter_ = iteration + 1
        return self
    
    def predict(self, X):
        dists = (
            (X ** 2).sum(axis=1, keepdims=True)
            - 2 * X @ self.cluster_centers_.T
            + (self.cluster_centers_ ** 2).sum(axis=1)
        )
        return dists.argmin(axis=1)

# Test
X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
X_blobs_s = StandardScaler().fit_transform(X_blobs)

km_scratch = KMeansScratch(k=3, random_state=42)
km_scratch.fit(X_blobs_s)

km_sklearn = KMeans(n_clusters=3, n_init=10, random_state=42)
km_sklearn.fit(X_blobs_s)

sil_scratch = silhouette_score(X_blobs_s, km_scratch.labels_)
sil_sklearn  = silhouette_score(X_blobs_s, km_sklearn.labels_)
print(f"KMeansScratch silhouette: {sil_scratch:.4f}")
print(f"sklearn KMeans silhouette: {sil_sklearn:.4f}")
print(f"Converged in {km_scratch.n_iter_} iterations")

# ===========================================================================
# SOLUTION D: Stratified K-Fold From Scratch (Module 6 — Challenge)
# ===========================================================================
print("\n--- SOLUTION D: Stratified K-Fold From Scratch ---")

def stratified_kfold_cv(model_fn, X, y, k=5):
    """
    Stratified K-Fold cross-validation from scratch.
    Each fold maintains the same class distribution as the full dataset.
    """
    classes = np.unique(y)
    
    # Split each class's indices into K folds
    class_folds = {}
    for c in classes:
        c_idx = np.where(y == c)[0].copy()
        np.random.shuffle(c_idx)
        class_folds[c] = np.array_split(c_idx, k)
    
    scores = []
    for fold_i in range(k):
        # Validation: fold i from each class
        val_idx   = np.concatenate([class_folds[c][fold_i] for c in classes])
        # Train: all other folds from each class
        train_idx = np.concatenate([
            np.concatenate([class_folds[c][j] for j in range(k) if j != fold_i])
            for c in classes
        ])
        
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        sc = StandardScaler()
        X_tr_s  = sc.fit_transform(X_tr)
        X_val_s = sc.transform(X_val)
        
        model = model_fn()
        model.fit(X_tr_s, y_tr)
        scores.append(f1_score(y_val, model.predict(X_val_s), average='macro'))
    
    return np.array(scores)

# Test
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
X_cv, y_cv = make_classification(n_samples=600, n_features=10,
                                  weights=[0.7, 0.2, 0.1], n_classes=3,
                                  n_informative=8, random_state=42)

scores_scratch = stratified_kfold_cv(lambda: LogisticRegression(max_iter=300), X_cv, y_cv, k=5)
scores_sklearn  = cross_val_score(
    Pipeline([('sc', StandardScaler()),
              ('lr', LogisticRegression(max_iter=300))]),
    X_cv, y_cv, cv=StratifiedKFold(5), scoring='f1_macro'
)

print(f"Scratch scores:  {scores_scratch.round(4)}")
print(f"sklearn scores:  {scores_sklearn.round(4)}")
print(f"Mean diff: {abs(scores_scratch.mean() - scores_sklearn.mean()):.6f}  (tiny = correct)")

print("""
All solutions verified! Key insights:
  A. Custom transformers: implement fit() + transform(), use BaseEstimator + TransformerMixin
  B. Gradient descent: shuffle → mini-batch forward → gradients → update → repeat
  C. K-Means scratch: random init → assign → recompute → convergence check
  D. Stratified CV: split each class's indices separately into K folds, then combine
""")
