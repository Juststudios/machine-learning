"""
Imbalanced Data: Advanced Techniques
======================================
Module 4 (bias-variance) introduced class_weight='balanced'.
This lesson goes deeper with practical tools for severe imbalance.

Typical severity levels:
  Mild:   90/10 split  → class_weight='balanced' is usually enough
  Severe: 95/5 split   → need sampling strategies
  Extreme: 99/1 split  → need SMOTE or ensemble resampling

Three main strategies:
  1. Algorithmic: class weights, threshold adjustment
  2. Oversampling: create synthetic minority samples (SMOTE)
  3. Undersampling: remove majority samples (random, Tomek links)
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
np.random.seed(42)

# ===========================================================================
# PART 1: The Problem at Scale
# ===========================================================================
print("=" * 60)
print("PART 1: Severe Imbalance (97%/3%)")
print("=" * 60)

print("""
Credit card fraud: 97% legitimate, 3% fraudulent.
Industrial fault: 98% OK, 2% fault.

A model that always predicts 'OK' gets 98% accuracy!
But it is completely useless in production.

What you actually care about:
  - Recall (Sensitivity): "Of all real faults, how many did I catch?"
  - Precision: "Of all my fault alerts, how many are real faults?"
  - F1: harmonic mean (especially useful when both matter)
  - PR-AUC: Area under Precision-Recall curve (better than ROC for imbalance)
""")

# Generate severely imbalanced dataset
X, y = make_classification(
    n_samples=5000,
    n_features=15,
    n_informative=10,
    weights=[0.97, 0.03],   # 97% class 0, 3% class 1
    random_state=42
)

print(f"Class distribution: {np.bincount(y)}")
print(f"  Class 0: {np.bincount(y)[0]} ({100*np.bincount(y)[0]/len(y):.1f}%)")
print(f"  Class 1: {np.bincount(y)[1]} ({100*np.bincount(y)[1]/len(y):.1f}%)")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

def evaluate(name, model, X_te, y_te):
    """Compute key metrics for imbalanced classification."""
    y_pred  = model.predict(X_te)
    y_prob  = model.predict_proba(X_te)[:, 1]
    acc     = (y_pred == y_te).mean()
    f1      = f1_score(y_te, y_pred)
    recall  = (y_pred[y_te == 1] == 1).mean() if (y_te == 1).any() else 0
    pr_auc  = average_precision_score(y_te, y_prob)
    roc_auc = roc_auc_score(y_te, y_prob)
    print(f"  {name:40s}: Acc={acc:.4f}, F1={f1:.4f}, Recall={recall:.4f}, PR-AUC={pr_auc:.4f}")
    return y_pred, y_prob

print("\nBaseline comparison:")

# Naive model
lr_naive = LogisticRegression(max_iter=500).fit(X_tr_s, y_tr)
evaluate("LogisticRegression (no balancing)", lr_naive, X_te_s, y_te)

# Class weight balanced
lr_bal = LogisticRegression(max_iter=500, class_weight='balanced').fit(X_tr_s, y_tr)
evaluate("LogisticRegression (class_weight='balanced')", lr_bal, X_te_s, y_te)

# RandomForest with class weight
rf_bal = RandomForestClassifier(100, class_weight='balanced', random_state=42).fit(X_tr_s, y_tr)
evaluate("RandomForest (class_weight='balanced')", rf_bal, X_te_s, y_te)

# ===========================================================================
# PART 2: Threshold Adjustment
# ===========================================================================
print("\n--- PART 2: Threshold Adjustment ---")

print("""
Default threshold: predict class 1 if P(y=1|X) ≥ 0.50
For imbalanced data: lower the threshold to increase recall.

Tradeoff:
  Lower threshold → more alarms → more true positives AND more false positives
                  → Recall ↑, Precision ↓

Choose threshold based on business cost:
  - Missing a fault costs $10,000 → prioritize recall (lower threshold)
  - False alarm costs $500 → you can afford more, keep threshold moderate
  
Use the Precision-Recall curve to pick the right threshold.
""")

probs_bal = lr_bal.predict_proba(X_te_s)[:, 1]

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>7} | {'F1':>6}")
print("-" * 42)
from sklearn.metrics import precision_score, recall_score
for t in thresholds:
    y_t = (probs_bal >= t).astype(int)
    p = precision_score(y_te, y_t, zero_division=0)
    r = recall_score(y_te, y_t, zero_division=0)
    f = f1_score(y_te, y_t, zero_division=0)
    print(f"  t={t:.2f}     | {p:10.4f} | {r:7.4f} | {f:6.4f}")

# ===========================================================================
# PART 3: SMOTE (Synthetic Minority Oversampling Technique)
# ===========================================================================
print("\n--- PART 3: SMOTE — Synthetic Oversampling ---")

print("""
SMOTE: create NEW synthetic minority samples by interpolating
between existing minority samples and their k nearest neighbors.

For each minority sample x:
  1. Find its k nearest minority neighbors (k=5 by default)
  2. Choose a random neighbor x_nn
  3. Create synthetic sample: x_new = x + λ·(x_nn - x), λ ∈ [0,1]

WHY this helps: more minority training examples → model better learns
                the minority class boundary.

NOTE: imbalanced-learn is NOT installed in this environment.
The code below is commented out — it shows the pattern for when available.
""")

print("""
# SMOTE example (requires: pip install imbalanced-learn)
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline

# smote = SMOTE(sampling_strategy=0.3, random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_tr_s, y_tr)
# print(f"After SMOTE: class 0={sum(y_resampled==0)}, class 1={sum(y_resampled==1)}")

# Pipeline with SMOTE:
# pipe = ImbPipeline([
#     ('scaler', StandardScaler()),
#     ('smote',  SMOTE(sampling_strategy=0.3)),
#     ('model',  LogisticRegression(max_iter=500))
# ])
# pipe.fit(X_tr, y_tr)
""")

# Demonstrate manual oversampling (without imbalanced-learn)
print("Manual random oversampling (no imbalanced-learn needed):")
minority_idx = np.where(y_tr == 1)[0]
oversample_n = int((y_tr == 0).sum() * 0.2) - (y_tr == 1).sum()

if oversample_n > 0:
    oversample_idx = np.random.choice(minority_idx, size=oversample_n, replace=True)
    X_over = np.vstack([X_tr_s, X_tr_s[oversample_idx]])
    y_over = np.concatenate([y_tr, y_tr[oversample_idx]])
    print(f"  Before: class 0={sum(y_tr==0)}, class 1={sum(y_tr==1)}")
    print(f"  After:  class 0={sum(y_over==0)}, class 1={sum(y_over==1)}")
    
    lr_over = LogisticRegression(max_iter=500).fit(X_over, y_over)
    evaluate("LogisticRegression (manual oversampling)", lr_over, X_te_s, y_te)

# ===========================================================================
# PART 4: Which Strategy to Choose?
# ===========================================================================
print("\n--- PART 4: Strategy Selection Guide ---")

print("""
╔══════════════════════════════════════════════════════════════════╗
║ IMBALANCED DATA STRATEGY SELECTION GUIDE                        ║
╠══════════════════════════════════════════════════════════════════╣
║ Imbalance Ratio | Recommended Strategy                          ║
╠══════════════════════════════════════════════════════════════════╣
║ 80/20 (mild)    | class_weight='balanced' is usually enough     ║
║ 90/10           | class_weight + threshold tuning               ║
║ 95/5            | class_weight + threshold + check PR-AUC       ║
║ 97/3            | class_weight + SMOTE (if imb-learn available) ║
║ 99/1 (extreme)  | SMOTE + specialized algorithms (IsolForest)   ║
╠══════════════════════════════════════════════════════════════════╣
║ ALWAYS:                                                          ║
║  - Use F1/Recall/PR-AUC, NOT accuracy alone                     ║
║  - Use StratifiedKFold for CV (preserves class ratio)           ║
║  - Pick threshold based on business cost of FP vs FN            ║
╚══════════════════════════════════════════════════════════════════╝
""")

# ===========================================================================
# PART 5: Precision-Recall Curve
# ===========================================================================
prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_te, probs_bal)
pr_auc = average_precision_score(y_te, probs_bal)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Imbalanced Data: Evaluation Curves", fontsize=12, fontweight='bold')

# PR curve
axes[0].plot(rec_curve, prec_curve, 'b-', linewidth=2, label=f'PR-AUC={pr_auc:.3f}')
axes[0].axhline(y=np.bincount(y_te)[1]/len(y_te), color='gray', linestyle='--',
                label=f'Random baseline ({np.bincount(y_te)[1]/len(y_te)*100:.1f}%)')
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("Precision-Recall Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# F1 vs threshold
f1_scores = [f1_score(y_te, (probs_bal >= t).astype(int), zero_division=0) for t in thresh_curve]
best_t = thresh_curve[np.argmax(f1_scores)]
axes[1].plot(thresh_curve, f1_scores, 'g-', linewidth=2)
axes[1].axvline(best_t, color='red', linestyle='--', label=f'Best threshold={best_t:.3f}')
axes[1].set_xlabel("Threshold")
axes[1].set_ylabel("F1 Score")
axes[1].set_title("F1 Score vs Decision Threshold")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "imbalanced_data_curves.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Evaluation curves saved: {OUTPUT_DIR}/imbalanced_data_curves.png")
print(f"Optimal threshold (max F1): {best_t:.4f}")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. Never trust accuracy for imbalanced data (a dumb model can get 97%)
2. Use: Recall, F1, PR-AUC — these reflect minority class performance
3. class_weight='balanced': easiest first fix, supported by most sklearn models
4. Threshold tuning: lower threshold → more recalls, more false positives
5. Use PR curve to choose threshold based on your precision/recall priorities
6. SMOTE (imbalanced-learn): generate synthetic minority samples
7. StratifiedKFold: ensures each CV fold preserves class ratio
8. Extreme imbalance: consider IsolationForest, OneClassSVM, or specialized approaches
""")
