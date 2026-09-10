"""
Lesson: K-Nearest Neighbors (KNN)
===================================
WHY THIS LESSON EXISTS:
    KNN is the simplest possible classifier based on a powerful intuition:
    "Things that are similar to each other tend to belong to the same class."
    
    It requires NO training step, NO assumptions about data distribution,
    and can capture arbitrarily complex decision boundaries. But it has
    real costs: slow prediction, sensitive to irrelevant features, and
    requires careful choice of K.
    
    Understanding KNN builds intuition about:
    - What it means for data points to be "similar" (distance metrics)
    - The bias-variance tradeoff (K=1 vs K=large)
    - Why feature scaling matters for distance-based methods
    - Instance-based vs. model-based learning

WHAT YOU WILL LEARN:
    1. Instance-based learning: no model, just memory
    2. How KNN makes predictions (majority vote of K neighbors)
    3. Effect of K: K=1 (overfit) vs. K=large (underfit)
    4. Choosing K with cross-validation
    5. Distance metrics: Euclidean vs. Manhattan
    6. When to use KNN (and when not to)

REAL ENGINEERING APPLICATION:
    Recommendation systems: "users similar to you also liked X"
    Materials science: predict material properties from similar known materials
    Medical: predict patient outcome based on similar historical cases
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# PART 1: How KNN Works — Instance-Based Learning
# =============================================================================
print("=" * 65)
print("PART 1: How KNN Works — No Training, Just Memory")
print("=" * 65)
print("""
KNN Algorithm:
    TRAINING (lazy — does nothing!):
        Just store all (X_train, y_train) pairs in memory.
    
    PREDICTION for a new point x_new:
        1. Compute distance from x_new to every training point
        2. Find the K closest training points (the "neighbors")
        3. Take a majority vote among their labels
        4. Return the most common label as the prediction

    WHY "lazy learner": no model is built during training.
                        All computation happens at prediction time!
    
    This has consequences:
        FAST training:     O(1) — just store data
        SLOW prediction:   O(n * d) for each query (n = training size, d = features)
        HIGH memory:       Must store entire training set

Example (K=3):
    Query: x_new = [78°C, 4.2 mm/s]
    
    Neighbors found:      Distance:  Label:
      [76°C, 4.0 mm/s]   2.06       WARNING
      [79°C, 4.5 mm/s]   0.54       WARNING  
      [80°C, 5.2 mm/s]   2.06       FAULT
    
    Majority vote (K=3): WARNING=2, FAULT=1
    Prediction: WARNING ✓
""")

# =============================================================================
# PART 2: Generate Data and Visualize KNN Decision Boundaries
# =============================================================================
np.random.seed(42)

def make_motor_data(n=400):
    """2D motor health data for visualization."""
    n_ok, n_warn, n_fault = n//2, n//4, n//4
    X = np.vstack([
        np.random.randn(n_ok, 2)   * [3, 0.5] + [65, 1.8],
        np.random.randn(n_warn, 2) * [4, 0.7] + [78, 4.0],
        np.random.randn(n_fault, 2)* [4, 0.9] + [90, 7.0],
    ])
    y = np.array([0]*n_ok + [1]*n_warn + [2]*n_fault)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

X, y = make_motor_data()
feature_names = ['Temperature (°C)', 'Vibration (mm/s)']
class_names = ['OK', 'WARNING', 'FAULT']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# WHY: KNN uses Euclidean distance — temperature is in [60-100]°C
# and vibration is in [0-10] mm/s. Without scaling, temperature
# completely dominates the distance calculation. Scale FIRST!
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# =============================================================================
# PART 3: Effect of K — The Bias-Variance Tradeoff
# =============================================================================
print("=" * 65)
print("PART 3: Effect of K on Decision Boundaries")
print("=" * 65)
print("""
K=1:   Overfit — decision boundary perfectly wraps around each point
        Every training point is its own neighborhood → 100% train accuracy
        But noisy, jagged boundary → poor generalization
        
K=large: Underfit — boundary becomes very smooth (approaches prior probability)
        With K = all training points → always predicts majority class
        
K=optimal: Found by cross-validation — usually 5 to 30 for moderate datasets
""")

k_values = [1, 3, 5, 7, 15, 30]
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.ravel()

x_min, x_max = X_train_sc[:, 0].min()-0.5, X_train_sc[:, 0].max()+0.5
y_min, y_max = X_train_sc[:, 1].min()-0.5, X_train_sc[:, 1].max()+0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                      np.linspace(y_min, y_max, 200))
colors_bg = ['#BBDEFB', '#C8E6C9', '#FFCDD2']
colors_pt = ['#1565C0', '#2E7D32', '#B71C1C']

for idx, K in enumerate(k_values):
    ax = axes[idx]
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train_sc, y_train)
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    train_acc = accuracy_score(y_train, knn.predict(X_train_sc))
    test_acc  = accuracy_score(y_test,  knn.predict(X_test_sc))
    
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=colors_bg, alpha=0.4)
    ax.contour(xx, yy, Z, levels=[0.5, 1.5], colors='k', linewidths=0.8, alpha=0.5)
    
    for cls, color in enumerate(colors_pt):
        mask = y_train == cls
        ax.scatter(X_train_sc[mask, 0], X_train_sc[mask, 1],
                   c=color, s=12, alpha=0.6, label=class_names[cls])
    
    overfit = " ← OVERFIT" if train_acc - test_acc > 0.05 else ""
    ax.set_title(f'K={K}\nTrain={train_acc:.2f}, Test={test_acc:.2f}{overfit}', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

handles = [plt.scatter([], [], c=c, s=30, label=n)
           for c, n in zip(colors_pt, class_names)]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10)
fig.suptitle('KNN Decision Boundaries for Different K Values\n'
             '(Motor Health: OK / WARNING / FAULT)', fontsize=13)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(OUTPUT_DIR / "05_knn_boundaries.png", dpi=110, bbox_inches='tight')
plt.close()
print("✓ Saved: output/05_knn_boundaries.png")

# =============================================================================
# PART 4: Choosing K with Cross-Validation
# =============================================================================
print("\n" + "=" * 65)
print("PART 4: Choosing K with Cross-Validation")
print("=" * 65)

k_range = range(1, 35)
train_accs = []
cv_accs = []

for K in k_range:
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train_sc, y_train)
    train_accs.append(accuracy_score(y_train, knn.predict(X_train_sc)))
    # WHY 5-fold CV: estimate generalization without touching test set
    cv_score = cross_val_score(knn, X_train_sc, y_train, cv=5, scoring='accuracy')
    cv_accs.append(cv_score.mean())

best_k = k_range.start + int(np.argmax(cv_accs))
print(f"Best K (by 5-fold CV): K = {best_k}  (CV accuracy = {max(cv_accs):.3f})")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_range), train_accs, 'b-o', markersize=4, linewidth=1.5, label='Training accuracy')
ax.plot(list(k_range), cv_accs, 'r-s', markersize=4, linewidth=1.5, label='CV accuracy (5-fold)')
ax.axvline(best_k, color='green', linestyle='--', linewidth=1.5, label=f'Best K = {best_k}')
ax.set_xlabel('K (number of neighbors)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Choosing K with Cross-Validation\nLow K = overfit, High K = underfit', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_knn_choosing_k.png", dpi=120, bbox_inches='tight')
plt.close()
print("✓ Saved: output/05_knn_choosing_k.png")

# Final model with best K
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_sc, y_train)
y_pred = knn_best.predict(X_test_sc)
print(f"\nBest KNN (K={best_k}) on test set:")
print(classification_report(y_test, y_pred, target_names=class_names))

# =============================================================================
# PART 5: Distance Metrics — Euclidean vs. Manhattan
# =============================================================================
print("=" * 65)
print("PART 5: Distance Metrics")
print("=" * 65)
print("""
Euclidean distance (p=2): √(Σ(xᵢ - yᵢ)²)
    - Standard "as the crow flies" distance
    - Good when all directions matter equally

Manhattan distance (p=1): Σ|xᵢ - yᵢ|
    - Sum of absolute differences
    - Good for high-dimensional data (less affected by curse of dimensionality)
    - Named after city-block walking distance

Chebyshev distance (p=∞): max(|xᵢ - yᵢ|)
    - Only the largest difference matters
    - Rarely used in practice
""")

for metric, p in [('Euclidean (p=2)', 2), ('Manhattan (p=1)', 1)]:
    knn_m = KNeighborsClassifier(n_neighbors=best_k, metric='minkowski', p=p)
    knn_m.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, knn_m.predict(X_test_sc))
    cv = cross_val_score(knn_m, X_train_sc, y_train, cv=5).mean()
    print(f"  K={best_k}, {metric}: test={acc:.3f}, CV={cv:.3f}")

# =============================================================================
# PART 6: KNN Without Feature Scaling — Don't Do This!
# =============================================================================
print("\n" + "=" * 65)
print("PART 6: Why Feature Scaling is Mandatory for KNN")
print("=" * 65)

knn_unscaled = KNeighborsClassifier(n_neighbors=best_k)
knn_unscaled.fit(X_train, y_train)  # raw features, no scaling
acc_unscaled = accuracy_score(y_test, knn_unscaled.predict(X_test))
acc_scaled   = accuracy_score(y_test, knn_best.predict(X_test_sc))

print(f"KNN (K={best_k}) WITHOUT scaling: {acc_unscaled:.3f}")
print(f"KNN (K={best_k}) WITH scaling:    {acc_scaled:.3f}  ← always better")
print(f"\nWHY: Temperature range ≈ 40°C, Vibration range ≈ 8 mm/s")
print(f"     Without scaling, distance is dominated by temperature.")
print(f"     A neighbor at [65°C, 7.0 mm/s] FAULT appears closer than")
print(f"     [66°C, 2.0 mm/s] OK — temperature overwhelms vibration!")

print("""
KEY TAKEAWAYS — K-Nearest Neighbors:
  ✓ No training step — immediately usable on any data
  ✓ Non-parametric: no assumptions about data distribution
  ✓ Naturally handles multi-class problems
  ✓ Works well with small datasets when features are meaningful
  ✗ Slow prediction: O(n*d) per query — bad for large n
  ✗ High memory: must store all training data
  ✗ Sensitive to irrelevant features and scaling
  ✗ Struggles in high dimensions (curse of dimensionality)
  
  WHEN TO USE KNN:
    ✓ Small dataset (< 10,000 samples)
    ✓ When similarity is the right notion (recommendation systems)
    ✓ As a baseline or sanity check before complex models
    ✗ Large datasets → use random forest or neural networks
    ✗ Real-time prediction → use logistic regression or SVM
""")
