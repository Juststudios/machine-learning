"""
Solutions: ML Fundamentals (Module 1)
=======================================
Reference solutions for exercises.py in 01_ml_fundamentals.
Study these AFTER attempting each exercise yourself.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True, parents=True)
np.random.seed(42)

print("=" * 60)
print("SOLUTIONS: ML FUNDAMENTALS")
print("=" * 60)

# ===========================================================================
# SOLUTION 1: Feature type identification
# ===========================================================================
print("\n--- SOLUTION 1: Feature Types ---")
feature_types = {
    'age': 'Continuous numerical',
    'city': 'Nominal categorical (needs OneHotEncoder)',
    'education_level': 'Ordinal categorical (High School < Bachelor < Master < PhD)',
    'has_wifi': 'Binary (already 0/1)',
    'num_purchases': 'Discrete numerical (count data)',
    'temperature_C': 'Continuous numerical',
    'satisfaction': 'Ordinal (1=Low, 2=Med, 3=High)',
    'product_category': 'Nominal categorical (needs OneHotEncoder)',
}
for feat, type_str in feature_types.items():
    print(f"  {feat:20s}: {type_str}")

# ===========================================================================
# SOLUTION 2: Encoding
# ===========================================================================
print("\n--- SOLUTION 2: Encoding Categoricals ---")
cities = ['London', 'Paris', 'Berlin', 'London', 'Tokyo']
edu    = ['High School', 'PhD', 'Bachelor', 'Master', 'PhD']

# Nominal: OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, drop='first')
city_encoded = ohe.fit_transform(np.array(cities).reshape(-1, 1))
print(f"OneHot cities:\n{city_encoded}")

# Ordinal: OrdinalEncoder with explicit order
from sklearn.preprocessing import OrdinalEncoder
order = [['High School', 'Bachelor', 'Master', 'PhD']]
oe = OrdinalEncoder(categories=order)
edu_encoded = oe.fit_transform(np.array(edu).reshape(-1, 1))
print(f"\nOrdinal education: {edu_encoded.ravel()}")

# ===========================================================================
# SOLUTION 3: Scaling
# ===========================================================================
print("\n--- SOLUTION 3: Scaling ---")
X_raw = np.array([[70, 1.0], [82, 2.5], [95, 5.5], [68, 0.9]])

print("StandardScaler (best for algorithms using distances/gradients):")
ss = StandardScaler()
print(ss.fit_transform(X_raw).round(4))

print("\nMinMaxScaler (best when algorithm expects [0,1] input):")
mms = MinMaxScaler()
print(mms.fit_transform(X_raw).round(4))

# ===========================================================================
# SOLUTION 4: Complete ML Pipeline (the power plant exercise)
# ===========================================================================
print("\n--- SOLUTION 4: Power Plant Pipeline ---")
n = 500
temp = np.random.uniform(5, 35, n)
pressure = np.random.uniform(990, 1020, n)
humidity = np.random.uniform(20, 90, n)
load = 0.3*temp + 0.1*pressure + 0.05*humidity + np.random.randn(n)*5 + 20

X = np.column_stack([temp, pressure, humidity])
y = load

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s  = sc.transform(X_te)

model = LinearRegression()
model.fit(X_tr_s, y_tr)
y_pred = model.predict(X_te_s)

print(f"  R²:   {r2_score(y_te, y_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_te, y_pred)):.4f} MW")
print(f"  Coefficients: temp={model.coef_[0]:.4f}, pressure={model.coef_[1]:.4f}, humidity={model.coef_[2]:.4f}")

# ===========================================================================
# SOLUTION 5: K-Fold Cross-Validation From Scratch (the challenge)
# ===========================================================================
print("\n--- SOLUTION 5: K-Fold CV From Scratch ---")

def manual_kfold_cv(model_class, X, y, k=5, random_state=42):
    """
    Implements K-Fold cross-validation from scratch.
    Returns: list of R² scores for each fold.
    """
    n = len(X)
    np.random.seed(random_state)
    indices = np.random.permutation(n)
    folds = np.array_split(indices, k)
    
    scores = []
    for fold_idx in range(k):
        # Val indices: this fold
        val_idx   = folds[fold_idx]
        # Train indices: all OTHER folds
        train_idx = np.concatenate([folds[i] for i in range(k) if i != fold_idx])
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale (fit ONLY on train fold!)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)
        
        # Train model
        model = model_class()
        model.fit(X_train_s, y_train)
        
        # Evaluate
        r2 = r2_score(y_val, model.predict(X_val_s))
        scores.append(r2)
    
    return scores

scores = manual_kfold_cv(LinearRegression, X, y, k=5)
print(f"Manual K-Fold scores: {[round(s, 4) for s in scores]}")
print(f"Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Compare with sklearn
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

pipe = Pipeline([('sc', StandardScaler()), ('lr', LinearRegression())])
sk_scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
print(f"sklearn K-Fold scores: {sk_scores.round(4).tolist()}")
print(f"Mean: {sk_scores.mean():.4f} ± {sk_scores.std():.4f}")
print(f"Difference (should be tiny): {abs(np.mean(scores) - sk_scores.mean()):.6f}")

print("""
Matching! The manual implementation correctly:
  - Splits data into K folds
  - For each fold: trains on K-1 folds, validates on held-out fold
  - Fits the scaler ONLY on training folds (no data leakage!)
""")
