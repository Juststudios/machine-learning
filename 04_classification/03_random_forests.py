"""
Lesson: Random Forests and Ensemble Methods
============================================
WHY THIS LESSON EXISTS:
    A single decision tree has high variance — small changes in training data
    produce very different trees. Random Forests solve this by training MANY
    trees on different random subsets of data and features, then combining
    their votes. This is the "wisdom of crowds" principle applied to ML.

    Random forests are among the most reliable algorithms for tabular data
    in real-world engineering applications — they require minimal preprocessing,
    handle mixed feature types, resist overfitting better than single trees,
    and provide feature importance scores automatically.

WHAT YOU WILL LEARN:
    1. Bootstrap aggregating (bagging): how randomness reduces variance
    2. Feature randomness: why we use random subsets of features per split
    3. RandomForestClassifier: n_estimators, max_features, max_depth
    4. Out-of-bag (OOB) score: free validation without a held-out set
    5. Feature importance (more stable than single tree)
    6. Comparison: single tree vs random forest vs gradient boosting

REAL ENGINEERING APPLICATION:
    NASA uses random forests for anomaly detection in spacecraft telemetry.
    Banks use them for fraud detection (millions of transactions/day).
    Manufacturing: multi-sensor fault classification with 95%+ accuracy.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# PART 1: The Problem — Why Single Trees Have High Variance
# =============================================================================
print("=" * 65)
print("PART 1: The High-Variance Problem With Single Trees")
print("=" * 65)
print("""
EXPERIMENT: Train many decision trees on different random subsets of
the same training data. See how different they are!

This demonstrates why a SINGLE tree is unreliable:
""")

np.random.seed(42)

def make_sensor_data(n=750, seed=42):
    """3-class machine health data: OK / WARNING / FAULT."""
    rng = np.random.RandomState(seed)
    n_ok, n_warn, n_fault = n//3 + n%3, n//3, n//3
    X = np.vstack([
        rng.randn(n_ok, 4)   * [3, 0.4, 3, 2]  + [65, 1.8, 50, 0.3],
        rng.randn(n_warn, 4) * [4, 0.7, 4, 2]  + [78, 4.0, 44, 0.5],
        rng.randn(n_fault, 4)* [5, 1.0, 4, 2]  + [91, 7.5, 38, 0.8],
    ])
    y = np.array([0]*n_ok + [1]*n_warn + [2]*n_fault)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

X, y = make_sensor_data(750)
feature_names = ['Temperature', 'Vibration', 'Pressure', 'Current']
class_names = ['OK', 'WARNING', 'FAULT']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train 10 trees on different bootstrap samples — observe variance
n_demo_trees = 10
demo_accuracies = []
demo_importances = []

for i in range(n_demo_trees):
    # Bootstrap sample: sample WITH replacement
    boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot, y_boot = X_train[boot_idx], y_train[boot_idx]
    t = DecisionTreeClassifier(max_depth=None, random_state=i)
    t.fit(X_boot, y_boot)
    demo_accuracies.append(accuracy_score(y_test, t.predict(X_test)))
    demo_importances.append(t.feature_importances_)

print(f"10 individual trees trained on bootstrap samples:")
print(f"  Accuracies: {[f'{a:.3f}' for a in demo_accuracies]}")
print(f"  Mean accuracy: {np.mean(demo_accuracies):.3f}")
print(f"  Std deviation: {np.std(demo_accuracies):.3f}  ← HIGH variance!")
print(f"\n  Majority vote of all 10: ", end="")

# Simulate majority vote manually
all_preds = []
for i in range(n_demo_trees):
    boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot, y_boot = X_train[boot_idx], y_train[boot_idx]
    t = DecisionTreeClassifier(max_depth=None, random_state=i+100)
    t.fit(X_boot, y_boot)
    all_preds.append(t.predict(X_test))

ensemble_preds = np.array(all_preds)  # shape: (10, n_test)
# Majority vote: most common prediction per sample
from scipy import stats
majority_preds = stats.mode(ensemble_preds, axis=0, keepdims=False).mode
ensemble_acc = accuracy_score(y_test, majority_preds)
print(f"{ensemble_acc:.3f}  ← Better than most individual trees!")

# =============================================================================
# PART 2: How Random Forest Reduces Variance (Bagging + Feature Randomness)
# =============================================================================
print("\n" + "=" * 65)
print("PART 2: How Random Forest Works")
print("=" * 65)
print("""
Random Forest adds TWO sources of randomness to bagging:

1. BOOTSTRAP SAMPLING (Bagging):
   Each tree trains on a different random sample (with replacement)
   from the training data. ≈ 63% of samples appear; 37% are "out-of-bag"
   
2. FEATURE RANDOMNESS:
   At each split, consider only √(n_features) random features
   (instead of all features). This de-correlates the trees.
   
   WHY this matters: If one feature is very strong (e.g. temperature),
   ALL trees would split on it → highly correlated trees → averaging
   doesn't help. Feature randomness forces trees to use other features.

3. AGGREGATION:
   For classification: majority vote across all trees
   For regression: average prediction across all trees

RESULT: High bias reduced by using many weak learners.
        High variance reduced by averaging independent (de-correlated) trees.
""")

# =============================================================================
# PART 3: RandomForestClassifier — Key Hyperparameters
# =============================================================================
print("=" * 65)
print("PART 3: RandomForestClassifier — Training and Tuning")
print("=" * 65)

# WHY n_estimators=200: more trees → better performance, but diminishing returns
# WHY oob_score=True: automatically uses out-of-bag samples as a free validation set
rf = RandomForestClassifier(
    n_estimators=200,    # number of trees
    max_features='sqrt', # WHY sqrt: standard recommendation for classification
    max_depth=None,      # let trees grow fully — averaging prevents overfitting
    oob_score=True,      # free validation using out-of-bag samples
    n_jobs=-1,           # use all CPU cores
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print(f"Random Forest (200 trees) Test Accuracy:  {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"Out-of-Bag Score (≈ validation accuracy): {rf.oob_score_:.3f}")
print()
print(classification_report(y_test, y_pred_rf, target_names=class_names))

# =============================================================================
# PART 4: Effect of n_estimators — When Is More Better?
# =============================================================================
print("=" * 65)
print("PART 4: How Many Trees Do You Need?")
print("=" * 65)

n_tree_range = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200]
rf_test_accs = []
rf_oob_accs = []

for n in n_tree_range:
    rf_n = RandomForestClassifier(
        n_estimators=n, max_features='sqrt', oob_score=True,
        n_jobs=-1, random_state=42
    )
    rf_n.fit(X_train, y_train)
    rf_test_accs.append(accuracy_score(y_test, rf_n.predict(X_test)))
    rf_oob_accs.append(rf_n.oob_score_)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(n_tree_range, rf_test_accs, 'b-o', linewidth=2, label='Test accuracy')
ax.plot(n_tree_range, rf_oob_accs, 'r--s', linewidth=2, label='OOB score')
ax.set_xlabel('Number of Trees (n_estimators)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Random Forest: Effect of Number of Trees', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.75, 1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_n_estimators_effect.png", dpi=120, bbox_inches='tight')
plt.close()
print("✓ Saved: output/03_n_estimators_effect.png")
print(f"\nAccuracy plateau: after ~50 trees, gains are minimal.")
print(f"  n=10: {rf_test_accs[2]:.3f}  n=50: {rf_test_accs[5]:.3f}  n=200: {rf_test_accs[-1]:.3f}")

# =============================================================================
# PART 5: Feature Importance (More Stable Than Single Tree)
# =============================================================================
print("\n" + "=" * 65)
print("PART 5: Feature Importance — Which Sensor Matters Most?")
print("=" * 65)

importances = rf.feature_importances_
std_importances = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
sorted_idx = np.argsort(importances)[::-1]

print("Feature Importances (averaged over 200 trees, ± std dev):")
for rank, i in enumerate(sorted_idx, 1):
    bar = "█" * int(importances[i] * 60)
    print(f"  {rank}. {feature_names[i]:14s}: {importances[i]:.3f} ± {std_importances[i]:.3f}  {bar}")

fig, ax = plt.subplots(figsize=(7, 4))
y_pos = np.arange(len(feature_names))
sorted_names = [feature_names[i] for i in sorted_idx]
sorted_imps = importances[sorted_idx]
sorted_std = std_importances[sorted_idx]

bars = ax.barh(y_pos, sorted_imps,
               xerr=sorted_std,
               color=['#F44336', '#FF9800', '#2196F3', '#4CAF50'],
               capsize=5, error_kw={'linewidth': 2})
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names, fontsize=11)
ax.set_xlabel('Feature Importance (mean ± std over 200 trees)', fontsize=10)
ax.set_title('Random Forest Feature Importance\nMore Stable Than Single Tree', fontsize=12)
for bar, val in zip(bars, sorted_imps):
    ax.text(val + 0.008, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_rf_feature_importance.png", dpi=120, bbox_inches='tight')
plt.close()
print("\n✓ Saved: output/03_rf_feature_importance.png")

# =============================================================================
# PART 6: Compare Single Tree vs Random Forest vs Gradient Boosting
# =============================================================================
print("\n" + "=" * 65)
print("PART 6: Algorithm Comparison")
print("=" * 65)

# Single decision tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
dt_cv = cross_val_score(dt, X, y, cv=5, scoring='accuracy').mean()

# Random forest
rf_cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy').mean()

# Gradient Boosting (brief: trees trained SEQUENTIALLY, each fixing errors of previous)
# WHY: GradientBoosting can often match RandomForest with fewer trees but is slower to train
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                 max_depth=3, random_state=42)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
gb_cv = cross_val_score(gb, X, y, cv=5, scoring='accuracy').mean()

print(f"{'Algorithm':<30} {'Test Acc':>10} {'5-Fold CV Acc':>14}")
print("-" * 56)
print(f"{'Decision Tree (depth=5)':<30} {dt_acc:>10.3f} {dt_cv:>14.3f}")
print(f"{'Random Forest (200 trees)':<30} {accuracy_score(y_test, y_pred_rf):>10.3f} {rf_cv:>14.3f}")
print(f"{'Gradient Boosting (100 trees)':<30} {gb_acc:>10.3f} {gb_cv:>14.3f}")

print("""
WHY Gradient Boosting is different from Random Forest:
  Random Forest: trees trained IN PARALLEL on random subsets (bagging)
  Gradient Boosting: trees trained SEQUENTIALLY, each correcting errors
                     of the previous tree (boosting)

  Random Forest  → reduces variance (overfitting)
  Gradient Boost → reduces bias (underfitting) iteratively

In practice: XGBoost / LightGBM (not shown here, need installation)
often outperform plain GradientBoostingClassifier on tabular data.
""")

print("""
KEY TAKEAWAYS — Random Forests:
  ✓ Almost always better than a single decision tree
  ✓ Minimal preprocessing needed (no scaling, handles missing values)
  ✓ Built-in OOB score = free validation
  ✓ Feature importances are more stable and reliable
  ✓ Very hard to overfit (averaging prevents it)
  ✗ Less interpretable than a single tree
  ✗ Slower to predict than single tree (must run through 200 trees)
  ✗ Not great for very high-dimensional sparse data (SVMs better)
""")
