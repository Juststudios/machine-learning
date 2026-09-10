"""
Module 6, Lesson 1: Classification and Regression Metrics
==========================================================
WHAT THIS LESSON TEACHES:
  The single biggest mistake beginners make in ML is defaulting to accuracy.
  This lesson shows every important metric, when to use each, and how to
  interpret the results — with real imbalanced-data examples.

WHY THIS MATTERS:
  In fraud detection, disease diagnosis, or equipment failure prediction,
  the cost of missing a positive case (false negative) vastly exceeds the
  cost of a false alarm. Accuracy hides this completely.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # WHY: non-interactive backend for saving PNGs without a display
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from pathlib import Path

# WHY: __file__-relative paths so script works from any directory
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("MODULE 6 | LESSON 1: Classification & Regression Metrics")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────
# SECTION 1: WHY ACCURACY FAILS ON IMBALANCED DATA
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 1: The Accuracy Trap on Imbalanced Data ---")

# WHY: 1% fraud rate mirrors real-world fraud detection datasets
np.random.seed(42)
X_imb, y_imb = make_classification(
    n_samples=10_000,
    n_features=10,
    weights=[0.99, 0.01],   # 99% legitimate, 1% fraud
    flip_y=0.01,
    random_state=42,
)
print(f"Dataset: {(y_imb == 0).sum():,} legitimate, {(y_imb == 1).sum():,} fraud")
print(f"Fraud rate: {y_imb.mean() * 100:.1f}%")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=42, stratify=y_imb
)

# A classifier that always predicts "not fraud" — illustrates the trap
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_tr, y_tr)
y_pred_dummy = dummy.predict(X_te)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_tr, y_tr)
y_pred_lr = lr.predict(X_te)

print("\n  Naive 'always predict not-fraud' classifier:")
print(f"    Accuracy:  {accuracy_score(y_te, y_pred_dummy) * 100:.1f}%  <- looks great!")
print(f"    Precision: {precision_score(y_te, y_pred_dummy, zero_division=0):.3f}  <- catches zero fraud")
print(f"    Recall:    {recall_score(y_te, y_pred_dummy, zero_division=0):.3f}  <- catches zero fraud")
print(f"    F1 Score:  {f1_score(y_te, y_pred_dummy, zero_division=0):.3f}  <- the truth")

print("\n  Logistic Regression:")
print(f"    Accuracy:  {accuracy_score(y_te, y_pred_lr) * 100:.1f}%")
print(f"    Precision: {precision_score(y_te, y_pred_lr, zero_division=0):.3f}")
print(f"    Recall:    {recall_score(y_te, y_pred_lr, zero_division=0):.3f}")
print(f"    F1 Score:  {f1_score(y_te, y_pred_lr, zero_division=0):.3f}")

# ─────────────────────────────────────────────────────────────────────
# SECTION 2: PRECISION vs RECALL — THE FUNDAMENTAL TRADEOFF
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 2: Precision vs Recall Tradeoff ---")
print("""
  PRECISION = TP / (TP + FP)
    "Of all cases I flagged as fraud, how many actually were?"
    High precision => few false alarms (fewer annoyed legitimate users)

  RECALL = TP / (TP + FN)
    "Of all actual fraud cases, how many did I catch?"
    High recall => few missed frauds (lower financial loss)

  F1 = 2 * (Precision * Recall) / (Precision + Recall)
    Harmonic mean — punishes extreme values more than arithmetic mean.
    Use when you need BOTH precision and recall to be good.
    Use F-beta to weight recall higher (beta > 1) or precision (beta < 1).
""")

# WHY: Adjusting decision threshold shifts the precision/recall balance
y_prob_lr = lr.predict_proba(X_te)[:, 1]

thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
print("  Threshold | Precision | Recall | F1")
print("  " + "-" * 40)
for thresh in thresholds:
    y_pred_t = (y_prob_lr >= thresh).astype(int)
    p = precision_score(y_te, y_pred_t, zero_division=0)
    r = recall_score(y_te, y_pred_t, zero_division=0)
    f = f1_score(y_te, y_pred_t, zero_division=0)
    print(f"     {thresh:.1f}     |   {p:.3f}   | {r:.3f}  | {f:.3f}")

print("\n  KEY INSIGHT: There is no universally 'best' threshold.")
print("  Your business problem determines where to set it.")

# ─────────────────────────────────────────────────────────────────────
# SECTION 3: CONFUSION MATRIX — SEE ALL 4 OUTCOMES
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 3: Confusion Matrix ---")
print("""
  The confusion matrix shows all 4 prediction outcomes:

           Predicted NEG    Predicted POS
  Actual NEG   TN (correct)     FP (false alarm!)
  Actual POS   FN (missed!)     TP (correct)

  For fraud detection:
    FN = missed fraud  (criminal goes free — financial loss)
    FP = false alarm   (legitimate customer annoyed — churn risk)
""")

cm = confusion_matrix(y_te, y_pred_lr)
print(f"  Confusion matrix:\n{cm}")
print(f"\n  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

# Plot confusion matrix heatmap
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Not Fraud", "Fraud"])
ax.set_yticklabels(["Not Fraud", "Fraud"])
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix — Fraud Detection")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14, fontweight="bold")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=120)
plt.close()
print("\n  Saved: output/confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────
# SECTION 4: ROC CURVE AND AUC
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 4: ROC Curve and AUC ---")
print("""
  ROC = Receiver Operating Characteristic
  Plots True Positive Rate vs False Positive Rate across ALL thresholds.

  AUC = Area Under the ROC Curve
    AUC = 1.0  -> perfect model (TPR=1, FPR=0 always)
    AUC = 0.5  -> random guessing (the diagonal reference line)
    AUC < 0.5  -> worse than random (but flip predictions for >0.5!)

  WHY AUC: measures the model's ability to RANK positives above negatives,
  independent of which decision threshold you choose later.
  AUC = probability that a random positive gets a higher score than a random negative.
""")

fpr_lr, tpr_lr, _ = roc_curve(y_te, y_prob_lr)
auc_lr = roc_auc_score(y_te, y_prob_lr)
print(f"  Logistic Regression ROC-AUC: {auc_lr:.4f}")

y_prob_rand = np.random.RandomState(99).rand(len(y_te))
fpr_rand, tpr_rand, _ = roc_curve(y_te, y_prob_rand)
auc_rand = roc_auc_score(y_te, y_prob_rand)

# Precision-Recall curve (preferred for imbalanced data)
precision_arr, recall_arr, _ = precision_recall_curve(y_te, y_prob_lr)
ap_lr = average_precision_score(y_te, y_prob_lr)
baseline_pr = y_te.mean()  # random PR baseline = class prevalence

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(fpr_lr, tpr_lr, "b-", lw=2, label=f"Logistic Reg (AUC={auc_lr:.3f})")
ax1.plot(fpr_rand, tpr_rand, "r--", lw=1.5, label=f"Random (AUC={auc_rand:.3f})")
ax1.plot([0, 1], [0, 1], "k:", lw=1, label="No skill (AUC=0.5)")
ax1.fill_between(fpr_lr, tpr_lr, alpha=0.1, color="blue")
ax1.set_xlabel("False Positive Rate (FPR)", fontsize=11)
ax1.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
ax1.set_title("ROC Curve — Fraud Detection", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# WHY PR curve: for heavy imbalance, ROC can look artificially good
ax2.plot(recall_arr, precision_arr, "b-", lw=2, label=f"Logistic Reg (AP={ap_lr:.3f})")
ax2.axhline(baseline_pr, color="r", linestyle="--",
            label=f"Random baseline ({baseline_pr:.3f})")
ax2.set_xlabel("Recall", fontsize=11)
ax2.set_ylabel("Precision", fontsize=11)
ax2.set_title("Precision-Recall Curve\n(Better for imbalanced data!)", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_pr_curves.png", dpi=120)
plt.close()
print("  Saved: output/roc_pr_curves.png")
print("\n  WHY PR over ROC for imbalance?")
print("  ROC is optimistic because it uses TN count in FPR denominator.")
print("  With many TN (legitimate transactions), FPR stays low even with many FP.")
print("  PR curve shows ONLY the positive class — it cannot hide imbalance.")

# ─────────────────────────────────────────────────────────────────────
# SECTION 5: REGRESSION METRICS
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 5: Regression Metrics ---")

# WHY: house prices make metric interpretation intuitive (everyone has intuition for $)
np.random.seed(0)
X_reg, y_reg = make_regression(n_samples=500, n_features=5, noise=20, random_state=0)
y_reg = y_reg * 1000 + 300_000  # scale to realistic house prices

X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=0)
reg = LinearRegression()
reg.fit(X_r_tr, y_r_tr)
y_r_pred = reg.predict(X_r_te)

mae  = mean_absolute_error(y_r_te, y_r_pred)
mse  = mean_squared_error(y_r_te, y_r_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_r_te, y_r_pred)

print(f"\n  Predicting house prices (~${y_reg.mean():,.0f} mean):")

print(f"\n  MAE  = ${mae:>12,.0f}")
print(f"    On average, predictions are off by ${mae:,.0f}")
print(f"    Same units as target — easy to explain to stakeholders.")
print(f"    Treats every error equally (robust to outliers).")

print(f"\n  MSE  = ${mse:>18,.0f}")
print(f"    Units are (dollars)^2 — hard to interpret directly.")
print(f"    WHY USE IT: differentiable everywhere, penalizes large errors heavily.")
print(f"    A $100k error contributes 4x more than a $50k error.")

print(f"\n  RMSE = ${rmse:>12,.0f}")
print(f"    Back to dollar units (square root of MSE).")
print(f"    Larger than MAE because it is dominated by big errors.")
print(f"    Use RMSE when large errors are especially costly.")
print(f"    Use MAE when all errors should be treated equally.")

print(f"\n  R2   = {r2:.4f}")
print(f"    The model explains {r2 * 100:.1f}% of variance in house prices.")
print(f"    R2 = 1.0: perfect predictions")
print(f"    R2 = 0.0: no better than always predicting the mean")
print(f"    R2 < 0.0: worse than predicting the mean (model is broken)")

# ─────────────────────────────────────────────────────────────────────
# SECTION 6: FULL CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 6: sklearn classification_report ---")
print("\n  classification_report gives all per-class metrics at once:")
print(classification_report(
    y_te, y_pred_lr,
    target_names=["Legitimate", "Fraud"],
    zero_division=0,
))
print("  WHY 'macro avg' vs 'weighted avg'?")
print("  - macro avg:    treats all classes equally regardless of size")
print("                  Use for balanced datasets")
print("  - weighted avg: weights by support (class size)")
print("                  For imbalanced data, dominated by the majority class!")
print("  - BEST for imbalanced: look at the minority class row directly.")

print("\n" + "=" * 65)
print("SUMMARY — WHICH METRIC TO CHOOSE:")
print("=" * 65)
print("  Balanced classification:          Accuracy, F1 macro")
print("  Imbalanced classification:        F1 minority class, PR-AUC")
print("  Ranking / probability quality:    ROC-AUC")
print("  Regression (interpretable):       MAE")
print("  Regression (penalize outliers):   RMSE")
print("  Regression (variance explained):  R2")
print("\nAll plots saved to: output/")
