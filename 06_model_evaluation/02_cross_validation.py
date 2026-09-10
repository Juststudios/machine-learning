"""
Module 6, Lesson 2: Cross-Validation
=====================================
WHAT THIS LESSON TEACHES:
  A single train/test split gives you ONE estimate of model performance.
  That estimate has high variance — you might get lucky or unlucky with the split.
  Cross-validation averages across MULTIPLE splits for a more reliable estimate.

WHY THIS MATTERS:
  If you train on split A and test on split B, you might report 95% accuracy.
  If you had used split C and D, you'd report 80%. Which is the truth?
  Cross-validation tells you BOTH the mean (expected performance) AND
  standard deviation (stability of performance).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut,
    TimeSeriesSplit, cross_val_score, cross_validate,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("MODULE 6 | LESSON 2: Cross-Validation")
print("=" * 65)

np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=10, random_state=42)

# ─────────────────────────────────────────────────────────────────────
# SECTION 1: K-FOLD CROSS-VALIDATION — HOW THE SPLITS WORK
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 1: K-Fold CV — How Data is Split ---")
print("""
  K-Fold splits data into K equal parts (folds).
  For each fold:
    - Use that fold as the validation set
    - Use ALL OTHER folds for training
    - Record the validation score

  Final score = mean of K validation scores
  Stability   = standard deviation of K validation scores

  For K=5 with 1000 samples:
    Each fold has 200 samples for validation, 800 for training.
    5 different models trained, 5 different evaluation scores.
""")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("  Fold | Train samples | Val samples | Train indices (first 5)...")
print("  " + "-" * 60)
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    print(f"    {fold_idx}  |    {len(train_idx):>5}      |    {len(val_idx):>5}    | "
          f"{list(train_idx[:5])} ...")

# WHY shuffle=True: without shuffling, adjacent samples may be correlated
# (e.g., data collected in time order). Shuffling ensures random assignment to folds.

# ─────────────────────────────────────────────────────────────────────
# SECTION 2: CROSS_VAL_SCORE — THE EASY SKLEARN INTERFACE
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 2: cross_val_score ---")

# WHY Pipeline: ensures StandardScaler is fit ONLY on training folds,
# preventing data leakage where test statistics influence training.
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(random_state=42, max_iter=500)),
])

scores_5fold = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
scores_10fold = cross_val_score(pipe, X, y, cv=10, scoring="accuracy")

print(f"\n  5-Fold  CV scores:  {scores_5fold}")
print(f"  5-Fold  mean={scores_5fold.mean():.4f}  std={scores_5fold.std():.4f}")

print(f"\n  10-Fold CV scores:  {scores_10fold}")
print(f"  10-Fold mean={scores_10fold.mean():.4f}  std={scores_10fold.std():.4f}")

print("""
  WHY 10-fold over 5-fold?
    - More folds => each training set is larger => less bias in estimate
    - More folds => more computation (K models trained instead of 5)
    - More folds => lower variance in the estimate
    - Common choice: 5 or 10. For small datasets (<1000), use 10.
    - For large datasets (>100k), even 3-fold is fine.
""")

# ─────────────────────────────────────────────────────────────────────
# SECTION 3: STRATIFIED K-FOLD — PRESERVE CLASS PROPORTIONS
# ─────────────────────────────────────────────────────────────────────
print("--- SECTION 3: Stratified K-Fold for Classification ---")

# WHY: With imbalanced classes, a random fold might contain 0 positive samples!
# Stratified K-Fold ensures each fold has the same class ratio as the full dataset.
X_iris, y_iris = load_iris(return_X_y=True)
# Only use classes 0 and 1, and heavily subsample class 1 to create imbalance
X_imb = X_iris[y_iris != 2]
y_imb = y_iris[y_iris != 2]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kf_plain = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n  Fold | Class 0 % (plain KFold) | Class 0 % (Stratified)")
print("  " + "-" * 55)
for fold_idx, ((_, val_kf), (_, val_skf)) in enumerate(
    zip(kf_plain.split(X_imb), skf.split(X_imb, y_imb)), start=1
):
    pct_kf = (y_imb[val_kf] == 0).mean() * 100
    pct_skf = (y_imb[val_skf] == 0).mean() * 100
    print(f"    {fold_idx}  |         {pct_kf:.1f}%              |        {pct_skf:.1f}%")

overall_pct = (y_imb == 0).mean() * 100
print(f"\n  Overall class 0 %: {overall_pct:.1f}%")
print("  Stratified preserves this in every fold; plain KFold can drift.")

# ─────────────────────────────────────────────────────────────────────
# SECTION 4: CROSS_VALIDATE — GET MULTIPLE METRICS AT ONCE
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 4: cross_validate — Multiple Metrics at Once ---")

cv_results = cross_validate(
    pipe, X, y,
    cv=5,
    scoring=["accuracy", "f1", "roc_auc"],
    return_train_score=True,  # WHY: lets us compare train vs val to detect overfitting
)

print("\n  Per-fold results:")
print(f"  {'Fold':<6} {'Train Acc':<12} {'Val Acc':<12} {'Val F1':<10} {'Val AUC'}")
print("  " + "-" * 52)
for i in range(5):
    print(f"    {i+1}    {cv_results['train_accuracy'][i]:.4f}       "
          f"{cv_results['test_accuracy'][i]:.4f}       "
          f"{cv_results['test_f1'][i]:.4f}     "
          f"{cv_results['test_roc_auc'][i]:.4f}")

print(f"\n  Mean val accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"  Mean val F1:       {cv_results['test_f1'].mean():.4f}")
print(f"  Mean val AUC:      {cv_results['test_roc_auc'].mean():.4f}")

gap = cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
print(f"\n  Train-val accuracy gap: {gap:.4f}")
print("  Small gap => model generalizes well. Large gap => overfitting.")

# ─────────────────────────────────────────────────────────────────────
# SECTION 5: LEAVE-ONE-OUT CROSS-VALIDATION (LOO)
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 5: Leave-One-Out CV (LOO) ---")
print("""
  LOO is the extreme case of K-Fold where K = N (number of samples).
  Each sample gets to be the validation set exactly once.

  WHEN TO USE:
    - Very small datasets (<50 samples) where every sample counts
    - When you need maximum training data per fold
    - Medical datasets where collecting data is expensive

  WHEN NOT TO USE:
    - Large datasets (LOO trains N models — very slow)
    - When you need a probability estimate (single binary 0/1 predictions)

  LOO has low bias but HIGH VARIANCE in the estimate.
  For most cases, 10-fold CV is preferred.
""")

# WHY: Only run LOO on a small subset to demonstrate — it's slow
X_small = X[:50]
y_small = y[:50]
loo = LeaveOneOut()
loo_scores = cross_val_score(pipe, X_small, y_small, cv=loo, scoring="accuracy")
print(f"  LOO on 50 samples: {loo.get_n_splits(X_small)} models trained")
print(f"  Mean accuracy: {loo_scores.mean():.4f}")
print(f"  Std deviation: {loo_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────
# SECTION 6: TIME SERIES SPLIT — PREVENT FUTURE LEAKAGE
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 6: TimeSeriesSplit — Critical for Sequential Data ---")
print("""
  For time series data (stock prices, sensor readings, user activity),
  standard K-Fold is WRONG. Here is why:

  Standard K-Fold can use FUTURE data to train and PAST data to validate.
  This is data leakage — the model "sees the future" during training,
  giving falsely optimistic performance estimates.

  TimeSeriesSplit always trains on the PAST and validates on the FUTURE.
  The training window grows over time.

  Example with 4 splits:
    Split 1: Train[1..25]       Val[26..50]
    Split 2: Train[1..50]       Val[51..75]
    Split 3: Train[1..75]       Val[76..100]
    Split 4: Train[1..100]      Val[101..125]
""")

tscv = TimeSeriesSplit(n_splits=5)
n = 200
X_ts = np.arange(n).reshape(-1, 1).astype(float)
y_ts = np.zeros(n)

print(f"  TimeSeriesSplit on {n} sequential samples:")
print(f"  {'Split':<7} {'Train range':<20} {'Val range':<20} {'Train size':<12} {'Val size'}")
print("  " + "-" * 70)
for split_idx, (train_idx, val_idx) in enumerate(tscv.split(X_ts), start=1):
    print(f"    {split_idx}     [{train_idx[0]}..{train_idx[-1]}]"
          f"{'':>10}  [{val_idx[0]}..{val_idx[-1]}]"
          f"{'':>10}  {len(train_idx):<12}  {len(val_idx)}")

# Demonstrate on synthetic time series
np.random.seed(7)
t = np.linspace(0, 4 * np.pi, 300)
y_sin = np.sin(t) + np.random.randn(300) * 0.1

X_features = np.column_stack([
    np.sin(t),
    np.cos(t),
    t / (4 * np.pi),
])

from sklearn.linear_model import Ridge
pipe_ts = Pipeline([("scaler", StandardScaler()), ("reg", Ridge())])
tscv_eval = TimeSeriesSplit(n_splits=5)

ts_scores = cross_val_score(pipe_ts, X_features, y_sin, cv=tscv_eval, scoring="r2")
print(f"\n  Time series CV R2 scores: {ts_scores}")
print(f"  Mean R2: {ts_scores.mean():.4f}")
print("\n  WHY THIS MATTERS: Using standard CV on time series gives falsely")
print("  optimistic results because models can 'cheat' by using future information.")

# ─────────────────────────────────────────────────────────────────────
# SECTION 7: VARIANCE OF CV SCORES — MODEL STABILITY
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 7: CV Score Variance as Stability Measure ---")

# WHY: A model with mean=0.85 std=0.02 is MORE reliable than mean=0.87 std=0.15
models = {
    "Logistic Regression": Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(random_state=0, max_iter=500)),
    ]),
    "Decision Tree (max_depth=2)": Pipeline([
        ("sc", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=2, random_state=0)),
    ]),
    "Decision Tree (no limit)": Pipeline([
        ("sc", StandardScaler()),
        ("clf", DecisionTreeClassifier(random_state=0)),
    ]),
}

fig, ax = plt.subplots(figsize=(9, 5))
all_scores = []
labels = []

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    all_scores.append(scores)
    labels.append(name)
    print(f"\n  {name}:")
    print(f"    Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")
    print(f"    Scores: {scores}")

ax.boxplot(all_scores, labels=labels, patch_artist=True,
           boxprops=dict(facecolor="lightblue", color="navy"),
           medianprops=dict(color="red", linewidth=2))
ax.set_ylabel("10-Fold CV Accuracy", fontsize=11)
ax.set_title("Model Stability Comparison\n(box = IQR, whiskers = full range, red = median)",
             fontsize=12)
ax.tick_params(axis="x", labelsize=9)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cv_stability.png", dpi=120)
plt.close()
print("\n  Saved: output/cv_stability.png")
print("\n  WHY std matters: high variance means unpredictable performance in production.")
print("  A model with mean=0.82, std=0.02 is SAFER than mean=0.87, std=0.12.")

print("\n" + "=" * 65)
print("CROSS-VALIDATION CHEAT SHEET:")
print("=" * 65)
print("  Dataset size   | Recommended CV")
print("  " + "-" * 40)
print("  < 50 samples   | LOO (Leave-One-Out)")
print("  50–1000        | 10-Fold Stratified")
print("  1000–100k      | 5-Fold Stratified")
print("  > 100k         | 3-Fold or held-out validation set")
print("  Time series    | TimeSeriesSplit (ALWAYS)")
print("  Imbalanced     | StratifiedKFold (ALWAYS for classification)")
