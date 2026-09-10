"""
Lesson: Support Vector Machines (SVM)
======================================
WHY THIS LESSON EXISTS:
    Logistic regression finds ANY boundary that separates classes.
    SVM finds the BEST boundary — the one with maximum margin.
    This seemingly small difference produces excellent generalization,
    especially in high-dimensional spaces (text, genomics, signal processing).

    The kernel trick extends this to non-linear boundaries WITHOUT
    explicitly computing the transformed feature space — an elegant
    mathematical trick that makes SVMs extremely powerful.

WHAT YOU WILL LEARN:
    1. The maximum margin intuition: why bigger margin = better generalization
    2. Support vectors: only the hardest examples matter
    3. C parameter: hard vs. soft margin (handling noise)
    4. The kernel trick: non-linear boundaries with RBF kernel
    5. Linear vs RBF kernel — when to use which
    6. SVM vs logistic regression vs random forest decision guide

REAL ENGINEERING APPLICATION:
    Text classification (spam detection, sentiment analysis) often uses
    linear SVMs because text data is high-dimensional and sparse.
    Medical image classification uses RBF SVMs on extracted features.
    Signal processing: SVM on frequency-domain features for fault detection.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import make_moons, make_blobs

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# PART 1: The Maximum Margin Intuition
# =============================================================================
print("=" * 65)
print("PART 1: The Maximum Margin Intuition")
print("=" * 65)
print("""
PROBLEM: Many decision boundaries correctly separate training data.
         Which one should we choose?

         ──────────────────────────────────────
         Boundary A (barely separates): 
             ○ ○ ○  | ● ●
             ○ ○    | ● ● ●
                 ^── very close to ○ points — fragile!
         
         Boundary B (maximum margin):
             ○ ○ ○       ● ●
             ○ ○    |    ● ● ●
                    ^── maximum distance to both classes
         ──────────────────────────────────────

SVM INSIGHT: Maximize the margin (gap between classes).
             Wider margin → more "room" for new points → better generalization.

The boundary is defined by the SUPPORT VECTORS:
    - The training points closest to the boundary
    - All other training points are irrelevant to the boundary!
    - This makes SVM memory-efficient and robust to outliers (far from boundary)

Mathematical objective:
    Maximize:  2 / ||w||     (the margin width)
    Subject to: all points correctly classified with margin ≥ 1
""")

# =============================================================================
# PART 2: Visualize Linear SVM on 2D Synthetic Data
# =============================================================================
print("=" * 65)
print("PART 2: Linear SVM — Maximum Margin Visualization")
print("=" * 65)

np.random.seed(42)
X_lin, y_lin = make_blobs(n_samples=100, centers=2, cluster_std=0.8,
                           center_box=(-2, 2), random_state=42)

svm_lin = SVC(kernel='linear', C=1.0)
svm_lin.fit(X_lin, y_lin)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

def plot_svm_boundary(ax, X, y, clf, title, show_margin=True):
    """Plot decision boundary and margin for SVM."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    
    # Decision function gives signed distance to boundary
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # Background fill
    ax.contourf(xx, yy, Z, levels=[-100, 0, 100],
                colors=['#BBDEFB', '#FFCDD2'], alpha=0.4)
    # Decision boundary (Z = 0)
    ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
    
    if show_margin and hasattr(clf, 'support_vectors_'):
        # Margin boundaries (Z = ±1)
        ax.contour(xx, yy, Z, levels=[-1, 1], colors='k',
                   linewidths=1, linestyles='--')
    
    # Data points
    colors = ['#1565C0', '#B71C1C']
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[cls],
                   s=40, zorder=5, alpha=0.8, label=f'Class {cls}')
    
    # Support vectors
    if hasattr(clf, 'support_vectors_'):
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                   s=150, facecolors='none', edgecolors='gold',
                   linewidths=2.5, zorder=6, label='Support Vectors')
    
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

plot_svm_boundary(axes[0], X_lin, y_lin, svm_lin,
                  'Linear SVM: Decision Boundary + Margin\n(dashed lines = margin edges)')

# Effect of C parameter
svm_small_c = SVC(kernel='linear', C=0.05)
svm_small_c.fit(X_lin, y_lin)
plot_svm_boundary(axes[1], X_lin, y_lin, svm_small_c,
                  'Small C (C=0.05): Wider Margin, More Violations\n(soft margin — allows some misclassification)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_svm_linear_margin.png", dpi=120, bbox_inches='tight')
plt.close()
print("✓ Saved: output/04_svm_linear_margin.png")
print(f"\nLinear SVM (C=1.0) support vectors: {svm_lin.support_vectors_.shape[0]}")
print(f"Small C (C=0.05) support vectors:   {svm_small_c.support_vectors_.shape[0]}")
print("""
C parameter controls the bias-variance tradeoff:
  C → large: small margin, few violations, low bias, HIGH VARIANCE (overfit)
  C → small: large margin, more violations, higher bias, low variance (underfit)
""")

# =============================================================================
# PART 3: The Kernel Trick — Non-Linear Boundaries
# =============================================================================
print("=" * 65)
print("PART 3: The Kernel Trick — Non-Linear Boundaries")
print("=" * 65)
print("""
PROBLEM: What if classes are NOT linearly separable?

IDEA: Map data to a higher-dimensional space where it IS separable.
      Example: x → (x, x²) — a line in (x, x²) space is a parabola in x.

PROBLEM with mapping: Computing transformed features is expensive.
                      In infinite dimensions, it's impossible!

KERNEL TRICK: We never actually compute the transformation φ(x).
              Instead, we compute K(x, x') = φ(x)·φ(x') directly.
              For RBF kernel: K(x, x') = exp(-γ||x - x'||²)

This is mathematically equivalent to working in infinite dimensions
but only requires computing pairwise distances — O(n²) not O(∞)!

RBF = Radial Basis Function. It measures similarity:
  - K(x, x) = 1 (a point is perfectly similar to itself)
  - K(x, x') → 0 as distance increases
""")

# Non-linearly separable data (moons)
np.random.seed(0)
X_moon, y_moon = make_moons(n_samples=200, noise=0.18, random_state=42)

X_tr, X_te, y_tr, y_te = train_test_split(X_moon, y_moon, test_size=0.2, random_state=42)

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc = sc.transform(X_te)

# WHY: always scale before SVM! RBF kernel uses distances, so features
# on different scales will dominate the distance calculation.
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')  # gamma='scale' = 1/(n_features * var)
svm_lin2 = SVC(kernel='linear', C=1.0)
svm_rbf.fit(X_tr_sc, y_tr)
svm_lin2.fit(X_tr_sc, y_tr)

print(f"Moon-shaped data (non-linear):")
print(f"  Linear SVM accuracy: {accuracy_score(y_te, svm_lin2.predict(X_te_sc)):.3f}")
print(f"  RBF SVM accuracy:    {accuracy_score(y_te, svm_rbf.predict(X_te_sc)):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, clf, title in [
    (axes[0], svm_lin2, 'Linear SVM\nPoor fit for moon-shaped data'),
    (axes[1], svm_rbf, 'RBF SVM\nCaptures non-linear boundary'),
]:
    x_min, x_max = X_tr_sc[:, 0].min()-0.5, X_tr_sc[:, 0].max()+0.5
    y_min, y_max = X_tr_sc[:, 1].min()-0.5, X_tr_sc[:, 1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                          np.linspace(y_min, y_max, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    for cls, color in [(0, '#1565C0'), (1, '#B71C1C')]:
        mask = y_tr == cls
        ax.scatter(X_tr_sc[mask, 0], X_tr_sc[mask, 1],
                   c=color, s=30, alpha=0.7, label=f'Class {cls}')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_svm_kernel_comparison.png", dpi=120, bbox_inches='tight')
plt.close()
print("✓ Saved: output/04_svm_kernel_comparison.png")

# =============================================================================
# PART 4: Effect of C and gamma on RBF SVM
# =============================================================================
print("\n" + "=" * 65)
print("PART 4: Tuning C and gamma for RBF SVM")
print("=" * 65)
print("""
C:     controls margin softness (same as linear SVM)
gamma: controls "reach" of each training point
  
  gamma small → smooth, broad boundary (low variance, high bias)
  gamma large → wiggly, tight boundary (high variance, low bias)
""")

C_vals = [0.1, 1.0, 10.0]
gamma_vals = [0.1, 1.0, 10.0]

fig, axes = plt.subplots(len(C_vals), len(gamma_vals), figsize=(13, 11))

for i, C in enumerate(C_vals):
    for j, gamma in enumerate(gamma_vals):
        ax = axes[i][j]
        clf = SVC(kernel='rbf', C=C, gamma=gamma)
        clf.fit(X_tr_sc, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te_sc))
        
        x_min, x_max = X_tr_sc[:, 0].min()-0.3, X_tr_sc[:, 0].max()+0.3
        y_min_g, y_max_g = X_tr_sc[:, 1].min()-0.3, X_tr_sc[:, 1].max()+0.3
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                              np.linspace(y_min_g, y_max_g, 150))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
        for cls, color in [(0, '#1565C0'), (1, '#B71C1C')]:
            mask = y_tr == cls
            ax.scatter(X_tr_sc[mask, 0], X_tr_sc[mask, 1],
                       c=color, s=15, alpha=0.6)
        ax.set_title(f'C={C}, γ={gamma}\nacc={acc:.2f}', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

fig.suptitle('RBF SVM: Effect of C (rows) and gamma (columns)', fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_svm_C_gamma_grid.png", dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: output/04_svm_C_gamma_grid.png")

# =============================================================================
# PART 5: SVM on Machine Fault Data — with Engineering Features
# =============================================================================
print("\n" + "=" * 65)
print("PART 5: SVM for Machine Fault Detection (3-class, 4 features)")
print("=" * 65)

np.random.seed(42)
n = 750
X_eng = np.vstack([
    np.random.randn(250, 4) * [3, 0.4, 3, 2] + [65, 1.8, 50, 0.3],
    np.random.randn(250, 4) * [4, 0.7, 4, 2] + [78, 4.0, 44, 0.5],
    np.random.randn(250, 4) * [5, 1.0, 4, 2] + [91, 7.5, 38, 0.8],
])
y_eng = np.array([0]*250 + [1]*250 + [2]*250)
idx = np.random.permutation(n)
X_eng, y_eng = X_eng[idx], y_eng[idx]

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_eng, y_eng, test_size=0.2, stratify=y_eng, random_state=42)
sc2 = StandardScaler()
X_tr2_sc = sc2.fit_transform(X_tr2)
X_te2_sc = sc2.transform(X_te2)

# Compare kernels
for kernel, C, g in [('linear', 1.0, 'scale'), ('rbf', 10.0, 'scale')]:
    clf = SVC(kernel=kernel, C=C, gamma=g, probability=True)
    clf.fit(X_tr2_sc, y_tr2)
    acc = accuracy_score(y_te2, clf.predict(X_te2_sc))
    cv_acc = cross_val_score(clf, sc2.fit_transform(X_eng), y_eng, cv=5).mean()
    print(f"  SVM ({kernel:6s} kernel, C={C}): test={acc:.3f}, 5-fold CV={cv_acc:.3f}")

print("""
WHY FEATURE SCALING IS MANDATORY FOR SVM:
    SVM uses distances (Euclidean) to compute margins and kernel values.
    If temperature is in range 60-100 and pressure is 30-70, they have
    similar scales here. But if one feature were in [0, 10000], it would
    completely dominate the distance — unfair!
    
    ALWAYS call StandardScaler().fit_transform() before SVC.

WHEN TO USE SVM vs. OTHER CLASSIFIERS:
    ✓ SVM (linear):  High-dimensional, sparse data (text/NLP)
    ✓ SVM (RBF):     Medium-sized datasets, clear clusters, non-linear boundary
    ✗ SVM:           Very large datasets (n > 100k) — too slow (O(n²) to O(n³))
    ✗ SVM:           When probabilities/calibration are important (use LR instead)
    
    For tabular engineering data with n > 10k: Random Forest usually wins.
    For n < 5k with complex boundaries: RBF SVM is excellent.
""")
