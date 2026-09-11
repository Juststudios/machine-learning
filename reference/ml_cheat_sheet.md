# Machine Learning Cheat Sheet

> Quick reference for algorithm selection, sklearn API, metrics, and hyperparameters.

---

## Algorithm Selection Guide

### Which algorithm should I use?

```
START
  │
  ├─► Do you have LABELS (supervised)?
  │     │
  │     ├─► YES
  │     │     │
  │     │     ├─► Is the target CONTINUOUS (number)?
  │     │     │     → REGRESSION
  │     │     │         < 1K samples, few features:  LinearRegression
  │     │     │         Noisy data, outliers:         Ridge / Lasso
  │     │     │         Nonlinear relationship:       RandomForest / GradientBoosting
  │     │     │         Large dataset + deep patterns: PyTorch MLP
  │     │     │
  │     │     └─► Is the target CATEGORICAL (class)?
  │     │           → CLASSIFICATION
  │     │               Need interpretability:        LogisticRegression / DecisionTree
  │     │               Best accuracy, tabular data:  RandomForest / GradientBoosting
  │     │               High-dimensional sparse:      SVM (RBF kernel)
  │     │               Large dataset + raw features: PyTorch MLP / CNN
  │     │
  │     └─► NO → UNSUPERVISED
  │           │
  │           ├─► Find groups:        KMeans (spherical) / DBSCAN (arbitrary shape)
  │           ├─► Reduce dimensions:  PCA → t-SNE (for visualization)
  │           └─► Detect anomalies:   IsolationForest / DBSCAN
  │
  └─► Raw images / text / audio → DEEP LEARNING (CNN / Transformer)
```

---

## Quick Cheat: When to Use What

| Algorithm | Best When | Avoid When |
|---|---|---|
| LinearRegression | Linear relationship, interpretability needed | Nonlinear data |
| Ridge / Lasso | Multicollinearity, many features | Very nonlinear |
| LogisticRegression | Binary/multi classification, interpretable | Complex boundaries |
| DecisionTree | Need human-readable rules | Large datasets (overfits) |
| RandomForest | Best general-purpose tabular | Need fast inference |
| SVM (RBF) | High-dimensional, medium dataset | Very large datasets (slow) |
| KNN | Simple, no training needed | Large datasets (slow predict) |
| KMeans | Clustering, balanced spherical groups | Arbitrary shapes, outliers |
| DBSCAN | Arbitrary shapes, outlier detection | Very high dimensions |
| PCA | Dimensionality reduction, noise removal | Nonlinear structure |
| MLP (PyTorch) | Large tabular data, complex patterns | Small datasets |
| CNN | Images, spatial patterns | Tabular data |
| Transformer | Sequences, text, attention patterns | Small datasets |

---

## Sklearn API Quick Reference

### Estimator Interface

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Instantiate with hyperparameters
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Fit (learn from training data)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)                 # class labels
y_prob = model.predict_proba(X_test)           # probabilities (classifiers)

# Score (accuracy for classifiers, R² for regressors)
score = model.score(X_test, y_test)

# Inspect parameters
print(model.get_params())
model.set_params(n_estimators=200)             # change params before refitting

# Learned attributes (always end with _)
model.feature_importances_                     # RandomForest
model.coef_                                    # LinearRegression, LogisticRegression
```

### Transformer Interface

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)              # learn mean + std from TRAINING data only
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)   # apply SAME stats to test data

# Shortcut (only on training data!):
X_train_s = scaler.fit_transform(X_train)
```

### Pipeline

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler',  StandardScaler()),
    ('model',   RandomForestClassifier(n_estimators=100))
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
pipe.score(X_test, y_test)

# Access steps
pipe.named_steps['model'].feature_importances_

# Cross-validation with pipeline (no leakage!)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5)
```

### Train/Test Split & Cross-Validation

```python
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold,
    GridSearchCV, RandomizedSearchCV
)

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify for classification
)

# K-Fold CV
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
print(f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Grid Search
param_grid = {'model__n_estimators': [50, 100, 200], 'model__max_depth': [5, 10, None]}
gs = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
```

---

## Metrics Quick Reference

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # or 'weighted', 'binary'
recall    = recall_score(y_test, y_pred, average='macro')
f1        = f1_score(y_test, y_pred, average='macro')
cm        = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
roc_auc   = roc_auc_score(y_test, y_prob[:, 1])              # binary only
```

| Metric | Formula | Use When |
|---|---|---|
| Accuracy | TP+TN / Total | Balanced classes |
| Precision | TP / (TP+FP) | FP is costly (spam filter) |
| Recall | TP / (TP+FN) | FN is costly (disease detection) |
| F1 | 2·P·R/(P+R) | Imbalanced classes |
| ROC-AUC | Area under ROC curve | Binary classification |

### Regression Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
```

| Metric | Unit | Interpretation |
|---|---|---|
| MAE | Same as y | Average absolute error. Easy to interpret. |
| MSE | y² | Penalizes large errors heavily |
| RMSE | Same as y | Like MAE but penalizes outliers more |
| R² | Unitless (0-1) | 0.85 = model explains 85% of variance |

---

## Key Hyperparameters

### RandomForestClassifier / RandomForestRegressor

```python
RandomForestClassifier(
    n_estimators=100,    # number of trees (more = better but slower)
    max_depth=None,      # None = grow until pure (can overfit)
    min_samples_split=2, # min samples to split a node
    max_features='sqrt', # features considered per split ('sqrt' for classification)
    class_weight='balanced',  # for imbalanced data
    n_jobs=-1,           # use all CPU cores
    random_state=42,
)
```

### LogisticRegression

```python
LogisticRegression(
    C=1.0,               # inverse regularization (smaller = more regularization)
    penalty='l2',        # 'l1', 'l2', 'elasticnet', None
    max_iter=1000,       # increase if ConvergenceWarning
    class_weight='balanced',
    solver='lbfgs',      # 'saga' for l1 on large datasets
)
```

### SVM

```python
from sklearn.svm import SVC, SVR
SVC(
    C=1.0,               # regularization (larger = harder margin)
    kernel='rbf',        # 'linear', 'poly', 'rbf', 'sigmoid'
    gamma='scale',       # RBF kernel bandwidth
    class_weight='balanced',
    probability=True,    # enable predict_proba (slower)
)
```

---

## Common Preprocessing

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Scaler selection:
# StandardScaler:  default, works for most algorithms
# MinMaxScaler:    when algorithm expects [0,1] (neural nets, image pixels)
# RobustScaler:    when data has many outliers

# For mixed numerical + categorical data:
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
])
```

---

## Overfitting Checklist

| Symptom | Diagnosis | Fix |
|---|---|---|
| Train ≫ Test accuracy | Overfitting | Regularization, more data, simpler model |
| Both low | Underfitting | More complex model, more features |
| Test varies wildly | High variance | More data, cross-validate |
| Training never improves | Bad LR or wrong loss | Check optimizer, loss function |

---

## Saving / Loading Models

```python
import joblib

# Save
joblib.dump(model, 'model.joblib')

# Load
model = joblib.load('model.joblib')

# Always save the FULL PIPELINE (including scaler)
joblib.dump(pipeline, 'pipeline.joblib')
```
