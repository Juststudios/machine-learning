"""
PCA: Dimensionality Reduction
================================
The curse of dimensionality: as features grow, data becomes sparse,
distances lose meaning, and models overfit.

PCA (Principal Component Analysis):
  Find the directions of MAXIMUM VARIANCE in high-dimensional data
  and project the data onto those directions.

Result:
  - Fewer features (reduced dimensionality)
  - Features are uncorrelated (orthogonal principal components)
  - Maximum variance preserved in fewer dimensions

Math intuition:
  1. Compute the covariance matrix of X
  2. Find eigenvectors (principal components = directions of variance)
  3. Sort by eigenvalue (amount of variance explained)
  4. Keep top K eigenvectors → project X onto them

Engineers use PCA for:
  - Visualizing high-dimensional sensor data in 2D
  - Removing noisy/redundant sensor channels
  - Speeding up downstream models
  - Detecting multicollinearity (correlated sensors)
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

np.random.seed(42)

# ===========================================================================
# PART 1: PCA in Action — Industrial Sensors
# ===========================================================================
print("=" * 60)
print("PART 1: PCA for Sensor Data Visualization")
print("=" * 60)

print("""
Problem: 8 sensors monitor an industrial machine.
  - We want to visualize machine states, but 8D can't be plotted
  - PCA compresses 8D → 2D while keeping the most important variation
  - Points in the 2D plot that are far apart = different machine states
""")

# Generate 8-sensor data with 3 machine states
n_per_class = 100
# State 0: Normal operation
X0 = np.random.randn(n_per_class, 8) * 0.5 + np.array([70, 1.0, 3000, 15, 5, 60, 220, 20])
# State 1: High load
X1 = np.random.randn(n_per_class, 8) * 0.8 + np.array([82, 2.5, 3500, 20, 6, 62, 218, 35])
# State 2: Degraded
X2 = np.random.randn(n_per_class, 8) * 1.2 + np.array([95, 5.5, 2500, 27, 7, 65, 213, 60])

X_sensors = np.vstack([X0, X1, X2])
y_states  = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)

feature_names = ['Temperature', 'Vibration', 'RPM', 'Current', 'Pressure', 'Humidity', 'Voltage', 'Acoustic']
state_names   = ['Normal', 'High-Load', 'Degraded']

print(f"Sensor data: {X_sensors.shape}  (300 samples, 8 features)")

# ALWAYS scale before PCA!
# WHY: PCA finds directions of maximum VARIANCE.
#      If features have different scales, high-scale features dominate.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sensors)

# Fit PCA with all 8 components
pca_full = PCA(n_components=8)
pca_full.fit(X_scaled)

print(f"\nExplained variance ratio per component:")
for i, var in enumerate(pca_full.explained_variance_ratio_):
    bar = "█" * int(var * 50)
    cum = pca_full.explained_variance_ratio_[:i+1].sum()
    print(f"  PC{i+1}: {var:.4f}  {bar}  (cumulative: {cum:.4f})")

# ===========================================================================
# PART 2: Choosing the Number of Components
# ===========================================================================
print("\n--- PART 2: Choosing n_components ---")

print("""
How many components to keep?

Rule of thumb:
  - Keep components until cumulative explained variance ≥ 95%
  - Or: keep the first K components before the "elbow" in the scree plot

Scree plot: eigenvalue vs component number.
The elbow indicates where adding more components gives diminishing returns.
""")

# Find components needed for 95% variance
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
n_for_95 = np.searchsorted(cumulative_var, 0.95) + 1
print(f"Components needed for 95% explained variance: {n_for_95}")
print(f"  Cumulative variance at {n_for_95} components: {cumulative_var[n_for_95-1]:.4f}")

# ===========================================================================
# PART 3: 2D Visualization
# ===========================================================================
print("\n--- PART 3: 2D Visualization ---")

# Project to 2D for visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

print(f"Original: {X_scaled.shape}  →  PCA 2D: {X_2d.shape}")
print(f"Variance explained by 2 PCs: {pca_2d.explained_variance_ratio_.sum()*100:.1f}%")

# ===========================================================================
# PART 4: PCA as Preprocessing for ML
# ===========================================================================
print("\n--- PART 4: PCA as Preprocessing Step ---")

print("""
PCA in a Pipeline:
  Scale → PCA (reduce dimensions) → Classifier
  
WHY this helps:
  - Removes noisy/redundant features
  - Eliminates multicollinearity (PCs are orthogonal)
  - Speeds up model training
  - May improve generalization with noisy data
""")

# Load digits dataset (8x8 images = 64 features)
digits = load_digits()
X_digs, y_digs = digits.data, digits.target

print(f"Digits dataset: {X_digs.shape}  (64 pixel features, 10 classes)")

# Without PCA
pipe_no_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    LogisticRegression(max_iter=2000)),
])
score_no_pca = cross_val_score(pipe_no_pca, X_digs, y_digs, cv=5).mean()

# With PCA (keep 95% variance)
pipe_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_components=0.95)),  # 0.95 = keep 95% variance
    ('clf',    LogisticRegression(max_iter=2000)),
])
score_pca = cross_val_score(pipe_pca, X_digs, y_digs, cv=5).mean()

pipe_pca_temp = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95))])
pipe_pca_temp.fit(X_digs)
n_components_used = pipe_pca_temp.named_steps['pca'].n_components_

print(f"\nLogistic Regression accuracy:")
print(f"  Without PCA (64 features):  {score_no_pca:.4f}")
print(f"  With PCA ({n_components_used} features, 95% var): {score_pca:.4f}")
print(f"  Feature reduction: {64} → {n_components_used} ({100*(64-n_components_used)/64:.0f}% fewer features)")

# ===========================================================================
# PART 5: What PCA "Learned" — Feature Loadings
# ===========================================================================
print("\n--- PART 5: Interpreting Principal Components ---")

pca_sensor = PCA(n_components=3)
pca_sensor.fit(X_scaled)

print("PCA loadings — what each PC captures (which sensors contribute most):")
for pc_idx in range(3):
    loadings = pca_sensor.components_[pc_idx]
    top_idx  = np.argsort(np.abs(loadings))[::-1][:3]
    print(f"\n  PC{pc_idx+1} (explains {pca_sensor.explained_variance_ratio_[pc_idx]*100:.1f}% variance):")
    for i in top_idx:
        print(f"    {feature_names[i]:15s}: loading = {loadings[i]:+.4f}")

# ===========================================================================
# PART 6: Visualization
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("PCA: Sensor Data Dimensionality Reduction", fontsize=13, fontweight='bold')

# Scree plot
axes[0].bar(range(1, 9), pca_full.explained_variance_ratio_, color='steelblue', alpha=0.8, label='Individual')
axes[0].plot(range(1, 9), cumulative_var, 'o-', color='coral', linewidth=2, label='Cumulative')
axes[0].axhline(0.95, color='green', linestyle='--', alpha=0.7, label='95% threshold')
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Scree Plot")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2D PCA visualization
colors = ['steelblue', 'coral', 'green']
for state, color, name in zip([0, 1, 2], colors, state_names):
    mask = y_states == state
    axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, s=20, alpha=0.7, label=name)
axes[1].set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)")
axes[1].set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)")
axes[1].set_title("8D Sensor Data → 2D PCA")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Biplot (feature directions in PCA space)
for i, name in enumerate(feature_names):
    axes[2].arrow(0, 0, pca_2d.components_[0, i]*3, pca_2d.components_[1, i]*3,
                  head_width=0.1, color='red', alpha=0.7)
    axes[2].text(pca_2d.components_[0, i]*3.2, pca_2d.components_[1, i]*3.2,
                 name, fontsize=7, ha='center')
axes[2].set_xlim(-1.5, 1.5)
axes[2].set_ylim(-1.5, 1.5)
axes[2].set_xlabel("PC1")
axes[2].set_ylabel("PC2")
axes[2].set_title("PCA Biplot: Feature Directions")
axes[2].axhline(0, color='gray', linewidth=0.5)
axes[2].axvline(0, color='gray', linewidth=0.5)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_analysis.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nPCA visualization saved: {OUTPUT_DIR}/pca_analysis.png")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. ALWAYS scale before PCA (variance-sensitive!)
2. PCA finds directions of maximum variance in the data
3. Principal components are orthogonal (uncorrelated)
4. Scree plot: look for elbow to choose n_components
5. n_components=0.95: automatically keep 95% of variance
6. PCA in a Pipeline prevents data leakage during cross-validation
7. Loadings: show which original features contribute to each PC
8. Limitation: PCA is LINEAR — use t-SNE or UMAP for nonlinear structure

Next: exercises.py — Practice clustering and dimensionality reduction
""")
