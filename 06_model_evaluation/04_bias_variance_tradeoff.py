"""
Bias-Variance Tradeoff and Imbalanced Data
============================================
Two of the most important concepts for building reliable ML systems.

BIAS-VARIANCE:
  Every ML model's error = Bias² + Variance + Irreducible Noise
  
  Bias     = error from wrong assumptions (model too simple)
  Variance = error from sensitivity to training noise (model too complex)
  
  You cannot reduce both simultaneously — there is always a tradeoff.
  Your job: find the sweet spot.

IMBALANCED DATA:
  99% of credit card transactions are legitimate.
  A model that always predicts "legitimate" gets 99% accuracy!
  But it catches 0% of frauds — completely useless.
  
  Accuracy is a lying metric for imbalanced classes.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, cross_val_score, learning_curve, validation_curve
)
from sklearn.datasets import make_classification
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

np.random.seed(42)

# ===========================================================================
# PART 1: Bias-Variance Decomposition
# ===========================================================================
print("=" * 60)
print("PART 1: Bias-Variance Tradeoff")
print("=" * 60)

print("""
Total Error = Bias² + Variance + Irreducible Noise

BIAS (underfitting):
  Model makes systematic errors — wrong assumptions about the data
  Symptoms: both train AND test accuracy are LOW
  Example: using linear model on data with curved relationship
  Fix: more complex model, add features, add interactions

VARIANCE (overfitting):  
  Model is too sensitive to training data — memorizes noise
  Symptoms: HIGH train accuracy, LOW test accuracy (big gap)
  Example: deep tree, high-degree polynomial
  Fix: regularization, more data, simpler model, cross-validation

IRREDUCIBLE NOISE:
  No model can eliminate this — it's inherent in the measurement process
  Example: sensor measurement error, human labeling disagreements
""")

# Demonstrate with polynomial regression
n = 60
X_raw = np.random.uniform(0, 1, n).reshape(-1, 1)
y = np.sin(2 * np.pi * X_raw.ravel()) + np.random.normal(0, 0.3, n)

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=42)

print("Polynomial degree vs Train/Test R²:")
print(f"{'Degree':>7} | {'Train R²':>9} | {'Test R²':>8} | {'Diagnosis':>20}")
print("-" * 55)

from sklearn.metrics import r2_score
for degree in [1, 2, 5, 10, 20]:
    pipe = Pipeline([('poly', PolynomialFeatures(degree)), ('lr', LogisticRegression.__mro__[0].__new__(type('LR', (), {})))])
    # Use a simpler approach
    from sklearn.linear_model import LinearRegression
    poly_feat = PolynomialFeatures(degree)
    X_tr_p = poly_feat.fit_transform(X_train)
    X_te_p = poly_feat.transform(X_test)
    m = LinearRegression()
    m.fit(X_tr_p, y_train)
    tr_r2 = r2_score(y_train, m.predict(X_tr_p))
    te_r2 = r2_score(y_test, m.predict(X_te_p))
    
    if degree <= 2:
        diag = "Underfitting (high bias)"
    elif degree <= 5:
        diag = "Good fit"
    else:
        diag = "Overfitting (high variance)"
    print(f"  d={degree:2d}  | {tr_r2:9.4f} | {te_r2:8.4f} | {diag}")

# Learning curves to diagnose
from sklearn.linear_model import LinearRegression

def plot_learning_curve_comparison(axes_row, degree, title):
    """Plot learning curve for a polynomial model."""
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree)),
        ('lr',   LinearRegression()),
    ])
    sizes, train_scores, val_scores = learning_curve(
        pipe, X_raw, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring='r2'
    )
    tr_mean = train_scores.mean(axis=1)
    va_mean = val_scores.mean(axis=1)
    
    axes_row.plot(sizes, tr_mean, 'o-', color='steelblue', label='Train')
    axes_row.plot(sizes, va_mean, 'o-', color='coral',     label='Validation')
    axes_row.fill_between(sizes, tr_mean - train_scores.std(axis=1),
                           tr_mean + train_scores.std(axis=1), alpha=0.15, color='steelblue')
    axes_row.fill_between(sizes, va_mean - val_scores.std(axis=1),
                           va_mean + val_scores.std(axis=1), alpha=0.15, color='coral')
    axes_row.set_title(title)
    axes_row.set_xlabel("Training Size")
    axes_row.set_ylabel("R²")
    axes_row.set_ylim(-0.5, 1.1)
    axes_row.legend()
    axes_row.grid(True, alpha=0.3)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Learning Curves: Diagnosing Bias vs Variance", fontsize=12, fontweight='bold')
plot_learning_curve_comparison(axes[0], 1,  "Degree 1 — High Bias (Underfitting)")
plot_learning_curve_comparison(axes[1], 4,  "Degree 4 — Good Fit")
plot_learning_curve_comparison(axes[2], 15, "Degree 15 — High Variance (Overfitting)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bias_variance_learning_curves.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nLearning curves saved: {OUTPUT_DIR}/bias_variance_learning_curves.png")

# ===========================================================================
# PART 2: Imbalanced Data
# ===========================================================================
print("\n" + "=" * 60)
print("PART 2: Imbalanced Data")
print("=" * 60)

print("""
A balanced dataset has roughly equal class sizes.
An imbalanced dataset: one class dominates (fraud=1%, ok=99%).

Why accuracy fails for imbalanced data:
  Predict ALL samples as majority class → 99% accuracy!
  But 0% of minority class detected → useless.

Better metrics:
  Precision = TP / (TP + FP)  → "Of those I flagged, how many are real?"
  Recall    = TP / (TP + FN)  → "Of all real positives, how many did I find?"
  F1        = harmonic mean of Precision and Recall
  ROC-AUC   = discrimination ability regardless of threshold
""")

# Create severely imbalanced dataset
X_imb, y_imb = make_classification(
    n_samples=2000,
    n_features=10,
    weights=[0.95, 0.05],  # 95% class 0, 5% class 1
    random_state=42
)

print(f"Class distribution: {np.bincount(y_imb)}  ({np.bincount(y_imb)[1]/len(y_imb)*100:.1f}% minority)")

X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(X_imb, y_imb, test_size=0.2, stratify=y_imb, random_state=42)

# Approach 1: Naive model (no balancing)
lr_naive = LogisticRegression(max_iter=500)
lr_naive.fit(X_tr_i, y_tr_i)

# Approach 2: class_weight='balanced'
lr_balanced = LogisticRegression(max_iter=500, class_weight='balanced')
lr_balanced.fit(X_tr_i, y_tr_i)

print("\nNaive Logistic Regression:")
y_pred_naive = lr_naive.predict(X_te_i)
print(f"  Accuracy:  {(y_pred_naive == y_te_i).mean():.4f}  ← high but misleading!")
print(f"  Recall(1): {recall_score(y_te_i, y_pred_naive):.4f}  ← very low!")
print(f"  F1(1):     {f1_score(y_te_i, y_pred_naive):.4f}")

print("\nBalanced Logistic Regression (class_weight='balanced'):")
y_pred_bal = lr_balanced.predict(X_te_i)
print(f"  Accuracy:  {(y_pred_bal == y_te_i).mean():.4f}  ← lower but model is MORE useful!")
print(f"  Recall(1): {recall_score(y_te_i, y_pred_bal):.4f}  ← much higher!")
print(f"  F1(1):     {f1_score(y_te_i, y_pred_bal):.4f}")

print("\nFull classification reports:")
print("Naive:")
print(classification_report(y_te_i, y_pred_naive, target_names=['OK', 'FRAUD']))
print("Balanced:")
print(classification_report(y_te_i, y_pred_bal, target_names=['OK', 'FRAUD']))

# Approach 3: Adjust decision threshold
print("\n--- Adjusting the Decision Threshold ---")
print("""
Default threshold: predict class 1 if P(y=1|X) ≥ 0.5
For imbalanced data: LOWER the threshold to detect more of the minority class.

Precision-Recall tradeoff:
  Lower threshold → higher recall (catch more), lower precision (more false alarms)
  Higher threshold → lower recall, higher precision
""")
from sklearn.metrics import precision_recall_curve

probs_naive  = lr_naive.predict_proba(X_te_i)[:, 1]
probs_bal    = lr_balanced.predict_proba(X_te_i)[:, 1]

# Show effect of different thresholds
for threshold in [0.3, 0.4, 0.5, 0.6]:
    y_th = (probs_bal >= threshold).astype(int)
    print(f"  Threshold={threshold}: Precision={precision_score(y_te_i, y_th):.3f}, "
          f"Recall={recall_score(y_te_i, y_th):.3f}, F1={f1_score(y_te_i, y_th):.3f}")

print("""
===========================================
KEY TAKEAWAYS
===========================================
BIAS-VARIANCE:
  1. Underfitting (high bias): both train + val low → more complex model
  2. Overfitting (high variance): train >> val → regularize, more data
  3. Learning curves are your diagnostic tool — plot them always!

IMBALANCED DATA:
  1. NEVER trust accuracy alone for imbalanced classes
  2. Use F1, Recall, Precision, or ROC-AUC instead
  3. class_weight='balanced': most models support this
  4. Adjust decision threshold to control precision/recall tradeoff
  5. For extreme imbalance: oversample minority (SMOTE) or undersample majority

Next: exercises.py — Test your understanding of evaluation techniques
""")
