"""
Module 6, Lesson 3: Hyperparameter Tuning
==========================================
WHAT THIS LESSON TEACHES:
  Parameters are learned by the model during training (e.g., weights).
  Hyperparameters are set BY YOU before training and control HOW the model learns
  (e.g., learning rate, depth of a tree, C in SVM).
  This lesson teaches how to find the best hyperparameters without cheating.

WHY THIS MATTERS:
  The same algorithm with different hyperparameters can differ by 10-20%
  in performance. Finding good hyperparameters is often where practitioners
  spend most of their time.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    cross_val_score, learning_curve, validation_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("MODULE 6 | LESSON 3: Hyperparameter Tuning")
print("=" * 65)

np.random.seed(42)
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, random_state=42,
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# ─────────────────────────────────────────────────────────────────────
# SECTION 1: PARAMETERS VS HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 1: Parameters vs Hyperparameters ---")
print("""
  PARAMETERS (learned from data during training):
    - Linear regression: weights and bias (w, b)
    - Neural network: all the connection weights
    - SVM: the support vectors and their weights

  HYPERPARAMETERS (set by you before training):
    - Random Forest: n_estimators, max_depth, min_samples_leaf
    - SVM: C (regularization), kernel, gamma
    - Neural network: learning rate, number of layers, dropout rate
    - Ridge regression: alpha (regularization strength)

  KEY RULE: Hyperparameters must NEVER be tuned using the test set.
  If you do, the test set leaks into your model — it becomes a training signal.
  You would report falsely optimistic performance.

  CORRECT APPROACH: Tune on validation data (or via cross-validation),
  then evaluate ONCE on the held-out test set.
""")

# ─────────────────────────────────────────────────────────────────────
# SECTION 2: GRIDSEARCHCV — EXHAUSTIVE SEARCH
# ─────────────────────────────────────────────────────────────────────
print("--- SECTION 2: GridSearchCV — Exhaustive Search ---")
print("""
  GridSearchCV tries EVERY combination of specified hyperparameter values.
  For each combination, it runs K-fold cross-validation and records the score.
  Returns the combination with the best mean CV score.

  COST: If you have 3 hyperparameters with 4 values each and K=5:
        4 * 4 * 4 * 5 = 320 models trained.
""")

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, None],
    "min_samples_leaf": [1, 5],
}
n_combinations = (len(param_grid["n_estimators"]) *
                  len(param_grid["max_depth"]) *
                  len(param_grid["min_samples_leaf"]))
print(f"  Grid has {n_combinations} combinations x 5 folds = {n_combinations * 5} model fits")

t0 = time.time()
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,      # WHY: use all CPU cores in parallel
    verbose=0,
)
grid_search.fit(X_tr, y_tr)
grid_time = time.time() - t0

print(f"\n  GridSearchCV completed in {grid_time:.1f}s")
print(f"  Best parameters: {grid_search.best_params_}")
print(f"  Best CV score:   {grid_search.best_score_:.4f}")
print(f"  Test set score:  {grid_search.score(X_te, y_te):.4f}")

# Show top 5 results
print("\n  Top 5 parameter combinations:")
results = grid_search.cv_results_
sorted_idx = np.argsort(results["mean_test_score"])[::-1]
print(f"  {'Rank':<5} {'Mean CV':<10} {'Std':<8} Parameters")
print("  " + "-" * 60)
for rank, i in enumerate(sorted_idx[:5], start=1):
    params = results["params"][i]
    print(f"  {rank:<5} {results['mean_test_score'][i]:.4f}     "
          f"{results['std_test_score'][i]:.4f}  {params}")

# ─────────────────────────────────────────────────────────────────────
# SECTION 3: RANDOMIZEDSEARCHCV — MORE EFFICIENT
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 3: RandomizedSearchCV — Efficient Alternative ---")
print("""
  RandomizedSearchCV samples N random combinations from the parameter space.
  Works better than GridSearch when:
    - The search space is very large
    - Some hyperparameters matter much more than others
      (random search is efficient because it explores all dims independently)
    - You have a compute budget

  KEY INSIGHT: Adding an unimportant hyperparameter to GridSearch multiplies
  cost by the number of its values. RandomizedSearch cost stays the same.
""")

# WHY scipy distributions: allows continuous sampling, not just discrete grids
param_dist = {
    "n_estimators": randint(50, 300),        # sample from integers 50-299
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_leaf": randint(1, 20),
    "max_features": uniform(0.3, 0.7),       # continuous: 0.3 to 1.0
}

t0 = time.time()
rand_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=30,          # WHY 30: good exploration for this space; tune to your budget
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
rand_search.fit(X_tr, y_tr)
rand_time = time.time() - t0

print(f"  RandomizedSearchCV (30 iterations) completed in {rand_time:.1f}s")
print(f"  Best parameters: {rand_search.best_params_}")
print(f"  Best CV score:   {rand_search.best_score_:.4f}")
print(f"  Test set score:  {rand_search.score(X_te, y_te):.4f}")
print(f"\n  Time comparison: Grid={grid_time:.1f}s vs Random={rand_time:.1f}s")
print("  Random search often finds similarly good results in much less time.")

# ─────────────────────────────────────────────────────────────────────
# SECTION 4: NESTED CROSS-VALIDATION — UNBIASED EVALUATION
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 4: Nested Cross-Validation — Unbiased Evaluation ---")
print("""
  PROBLEM: If you use CV to tune hyperparameters, that CV score is optimistic.
  The tuning process has "peeked" at the outer validation data.

  SOLUTION: Nested CV uses TWO loops:
    Outer loop: estimate generalization performance (unseen test folds)
    Inner loop: tune hyperparameters (inner CV on training folds only)

    Outer fold 1: [val set 1 | inner CV on folds 2,3,4,5 -> best params -> eval on 1]
    Outer fold 2: [val set 2 | inner CV on folds 1,3,4,5 -> best params -> eval on 2]
    ...

  Result: multiple outer scores, each from a model tuned on its own subset.
  The mean outer score is an UNBIASED estimate of the algorithm's performance.

  WHY: In competition, if you report CV score from GridSearch, you're cheating.
  Nested CV gives you the honest number to report.
""")

param_grid_small = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, None],
}

from sklearn.model_selection import KFold
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)

nested_scores = cross_val_score(
    GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_small,
        cv=inner_cv,
        scoring="accuracy",
        n_jobs=-1,
    ),
    X, y,
    cv=outer_cv,
    scoring="accuracy",
)

print(f"  Nested CV outer scores: {nested_scores}")
print(f"  Nested CV mean: {nested_scores.mean():.4f} +/- {nested_scores.std():.4f}")
print("\n  This is your honest, unbiased estimate of algorithm performance.")
print("  Report THIS number, not the inner GridSearch best score.")

# ─────────────────────────────────────────────────────────────────────
# SECTION 5: LEARNING CURVES — DIAGNOSE BIAS vs VARIANCE
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 5: Learning Curves ---")
print("""
  Learning curves plot train and validation score vs. TRAINING SET SIZE.
  They diagnose whether you have:
    - High bias (underfitting): both scores low, close together
    - High variance (overfitting): large gap between train and val scores
    - Need more data: val score still rising as training size increases
""")

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_sizes, train_mean, "b-o", lw=2, label="Training score")
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.1, color="blue")
ax.plot(train_sizes, val_mean, "r-o", lw=2, label="Validation score")
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                alpha=0.1, color="red")
ax.set_xlabel("Training set size", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Learning Curve — Random Forest\n(shaded area = ±1 std across CV folds)",
             fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.05])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "learning_curve.png", dpi=120)
plt.close()
print("  Saved: output/learning_curve.png")
print(f"\n  At max training size:")
print(f"    Train score: {train_mean[-1]:.4f}")
print(f"    Val score:   {val_mean[-1]:.4f}")
print(f"    Gap:         {train_mean[-1] - val_mean[-1]:.4f}")

# ─────────────────────────────────────────────────────────────────────
# SECTION 6: VALIDATION CURVES — SCORE VS HYPERPARAMETER
# ─────────────────────────────────────────────────────────────────────
print("\n--- SECTION 6: Validation Curves — Score vs Hyperparameter ---")
print("""
  Validation curves show how train/val score changes as ONE hyperparameter varies.
  Use this to understand:
    - What values cause underfitting (too simple)?
    - What values cause overfitting (too complex)?
    - Where is the sweet spot?
""")

param_range = [1, 2, 3, 5, 7, 10, 15, 20]
train_scores_vc, val_scores_vc = validation_curve(
    RandomForestClassifier(n_estimators=50, random_state=42),
    X, y,
    param_name="max_depth",
    param_range=param_range,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

train_mean_vc = train_scores_vc.mean(axis=1)
val_mean_vc = val_scores_vc.mean(axis=1)
train_std_vc = train_scores_vc.std(axis=1)
val_std_vc = val_scores_vc.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(param_range, train_mean_vc, "b-o", lw=2, label="Training score")
ax.fill_between(param_range,
                train_mean_vc - train_std_vc,
                train_mean_vc + train_std_vc, alpha=0.1, color="blue")
ax.plot(param_range, val_mean_vc, "r-o", lw=2, label="Validation score")
ax.fill_between(param_range,
                val_mean_vc - val_std_vc,
                val_mean_vc + val_std_vc, alpha=0.1, color="red")

best_depth = param_range[np.argmax(val_mean_vc)]
ax.axvline(x=best_depth, color="green", linestyle="--", lw=1.5,
           label=f"Best depth = {best_depth}")
ax.set_xlabel("max_depth", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Validation Curve — Random Forest max_depth\n"
             "Left of peak: underfitting. Right of peak: overfitting.", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "validation_curve.png", dpi=120)
plt.close()
print(f"  Saved: output/validation_curve.png")
print(f"  Best max_depth: {best_depth} (val score = {val_mean_vc[np.argmax(val_mean_vc)]:.4f})")

print("\n" + "=" * 65)
print("HYPERPARAMETER TUNING DECISION GUIDE:")
print("=" * 65)
print("  Few hyperparameters (<= 3), small space   -> GridSearchCV")
print("  Many hyperparameters, large space          -> RandomizedSearchCV")
print("  Need unbiased algorithm performance        -> Nested CV")
print("  Diagnose overfitting/underfitting           -> Learning curves")
print("  Find optimal value for one hyperparameter  -> Validation curves")
print("\nPlots saved to: output/")
