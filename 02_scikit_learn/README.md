# Module 2: Scikit-Learn — The Standard ML Library

## Overview

Scikit-learn is the most widely used ML library in Python, and for good reason:
it gives you **100+ algorithms through a single, consistent interface**.
Once you understand the Estimator API pattern, you can swap algorithms,
combine preprocessing steps, and search hyperparameters — all with minimal code changes.

This module teaches the *engineering* of sklearn, not just the syntax.
You'll understand *why* the API is designed the way it is, and how to use
Pipelines to write production-quality, leakage-free ML code.

---

## Prerequisites

- **Module 1 completed**: you know what features, targets, train/test splits are
- Python: functions, classes (basic understanding), list comprehensions
- NumPy and Pandas at the level from Module 1
- You've seen `model.fit()` and `model.predict()` in Module 1 — now we go deep

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Use any sklearn estimator** confidently because they all follow the same API contract
2. **Build preprocessing Pipelines** that prevent data leakage
3. **Apply the right scaler** for each situation (StandardScaler vs MinMaxScaler vs RobustScaler)
4. **Handle categorical features** with LabelEncoder and OneHotEncoder
5. **Deal with missing data** using SimpleImputer
6. **Combine mixed-type preprocessing** with ColumnTransformer
7. **Save and reload trained models** for deployment using joblib
8. **Search hyperparameters** systematically with GridSearchCV inside a Pipeline

---

## The Estimator API — sklearn's Master Design Pattern

Every sklearn object (model, scaler, encoder, imputer) follows the same contract:

```
           ┌──────────────────────────────────────────────┐
           │            sklearn Estimator                  │
           │                                              │
  X_train ─┤─▶ .fit(X_train, y_train)   → learns params  │
  y_train ─┘                                              │
                                                          │
  X_new  ──┤─▶ .predict(X_new)           → returns y_hat │  (Models)
           │                                              │
  X_new  ──┤─▶ .transform(X_new)         → returns X_new'│  (Transformers)
           │                                              │
  X_new  ──┤─▶ .fit_transform(X, y)      → fit + transform│ (shortcut)
           │                                              │
           │   .score(X_test, y_test)    → evaluation     │
           └──────────────────────────────────────────────┘
```

**Why this matters**: You can write code that works with *any* estimator:
```python
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)           # Works for LinearRegression,
    return model.score(X_test, y_test)    # RandomForest, SVM, or anything!
```

---

## Pipeline — The Key to Production ML

A Pipeline chains preprocessing steps and a model into one object:

```
Raw Data
   │
   ▼  Step 1: Impute missing values  (fit on train, apply to test)
   │
   ▼  Step 2: Scale features         (fit on train, apply to test)
   │
   ▼  Step 3: Encode categoricals    (fit on train, apply to test)
   │
   ▼  Step 4: Model                  (fit on train, predict on test)
   │
   ▼
Predictions
```

**Why not just do these steps manually?**
→ **Data leakage**: if you fit the scaler on all data before splitting,
  you've secretly let test data influence training. The model then appears
  to perform better than it really does. Pipelines make this impossible.

---

## sklearn ↔ PyTorch: When to Use What

| Task | Tool | Why |
|------|------|-----|
| Tabular data, classical ML | **sklearn** | Fast, interpretable, no GPU needed |
| Images, sequences, text | **PyTorch** | Needs deep learning, GPU acceleration |
| Feature preprocessing | **sklearn** | Battle-tested, integrates everywhere |
| Custom neural architectures | **PyTorch** | Full flexibility |
| Quick prototype | **sklearn** | 3 lines to a working model |
| Production at scale | **PyTorch + sklearn pipelines** | Best of both worlds |

---

## Concept Map

```
scikit-learn
├── Estimators
│   ├── Supervised
│   │   ├── Regressors  (LinearRegression, Ridge, RandomForestRegressor)
│   │   └── Classifiers (LogisticRegression, SVM, RandomForestClassifier)
│   └── Unsupervised
│       ├── Clusterers  (KMeans, DBSCAN)
│       └── Decomposers (PCA, TruncatedSVD)
│
├── Transformers
│   ├── Scalers         (StandardScaler, MinMaxScaler, RobustScaler)
│   ├── Encoders        (OneHotEncoder, LabelEncoder, OrdinalEncoder)
│   ├── Imputers        (SimpleImputer, KNNImputer)
│   └── Feature Eng.   (PolynomialFeatures, SelectKBest)
│
├── Meta-estimators
│   ├── Pipeline        (chain steps together)
│   ├── ColumnTransformer (apply different transforms to different columns)
│   ├── GridSearchCV    (systematic hyperparameter tuning)
│   └── cross_val_score (robust evaluation)
│
└── Utilities
    ├── train_test_split
    ├── metrics          (mean_squared_error, accuracy_score, ...)
    └── datasets         (load_iris, make_regression, ...)
```

---

## File Guide

| File | What It Teaches | Run Time |
|------|-----------------|----------|
| `01_sklearn_api.py` | The Estimator contract; fit/predict/transform/score; joblib | ~10s |
| `02_preprocessing.py` | All scalers, encoders, imputers; ColumnTransformer | ~10s |
| `03_pipelines.py` | Pipeline construction; leakage prevention; GridSearchCV | ~30s |
| `exercises.py` | 4-tier practice problems | varies |

### Running Order
```bash
cd 02_scikit_learn/
python 01_sklearn_api.py
python 02_preprocessing.py
python 03_pipelines.py
python exercises.py
```

---

## Real-World Connections

- **Consistent API** → writing library code that works with any model (AutoML systems)
- **Pipelines** → mandatory in any production ML system to prevent subtle bugs
- **ColumnTransformer** → real datasets always have mixed types (sensor readings + equipment categories)
- **joblib serialization** → how trained models are deployed to production servers
- **GridSearchCV** → systematic search used in competitive ML and industrial tuning

---

## Next Steps → Module 3: Regression

You'll apply everything here to the most common supervised learning task:
predicting continuous values (temperatures, prices, load, lifetime).
