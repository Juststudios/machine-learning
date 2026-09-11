"""
K-Means Clustering: Finding Structure Without Labels
=====================================================
Unsupervised learning: we have data but NO labels.

Why use clustering?
  - Customer segmentation (group customers by behavior)
  - Anomaly detection (find data points that don't fit any cluster)
  - Data compression (replace similar samples with their cluster center)
  - Exploratory analysis (discover natural groupings in data)

K-Means algorithm:
  1. Initialize K centroids randomly
  2. Assign each point to nearest centroid
  3. Move each centroid to the mean of its assigned points
  4. Repeat 2-3 until centroids stop moving

WHY "K-Means": K clusters, each defined by its MEAN (centroid)
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

np.random.seed(42)

# ===========================================================================
# PART 1: The K-Means Algorithm — Step by Step
# ===========================================================================
print("=" * 60)
print("PART 1: K-Means — How It Works")
print("=" * 60)

print("""
Real problem: Group manufacturing machines by operational profile.
  - We have sensor data but NO pre-assigned groups
  - K-Means finds natural clusters in the data
  - After clustering, we analyze each cluster to understand it
""")

# Generate synthetic machine sensor data
n_machines = 300
n_features = 2  # Use 2 features for easy visualization

# 3 true operational modes: normal, high-load, degraded
X_machines, y_true = make_blobs(
    n_samples=n_machines,
    centers=[
        [70, 1.0],   # Normal: temp=70°C, vibration=1.0 mm/s
        [85, 3.5],   # High-load: higher temp, higher vibration
        [92, 6.0],   # Degraded: very high temp and vibration
    ],
    cluster_std=[2.5, 2.0, 2.0],
    random_state=42
)

print(f"Machine data shape: {X_machines.shape}")
print(f"Features: [temperature_C, vibration_mm_s]")
print(f"True cluster counts: {np.bincount(y_true)}")

# Scale features (critical for distance-based algorithms!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_machines)

# ===========================================================================
# PART 2: Fitting K-Means
# ===========================================================================
print("\n--- PART 2: Fitting K-Means ---")

# Fit with K=3 (we know the true K for this example)
kmeans = KMeans(
    n_clusters=3,
    n_init=10,        # WHY 10: run 10 times with different random inits, keep best
    max_iter=300,     # maximum iterations per run
    random_state=42
)
kmeans.fit(X_scaled)

labels = kmeans.labels_          # cluster assignment for each point
centers = kmeans.cluster_centers_ # centroid coordinates (scaled)
inertia = kmeans.inertia_        # sum of squared distances to centroids (lower = better)

print(f"Cluster assignments (first 10): {labels[:10]}")
print(f"Cluster counts: {np.bincount(labels)}")
print(f"Inertia (sum of squared distances): {inertia:.4f}")

# Transform centroids back to original scale for interpretation
centers_original = scaler.inverse_transform(centers)
print(f"\nCluster centroids (original scale):")
for i, c in enumerate(centers_original):
    print(f"  Cluster {i}: temperature={c[0]:.1f}°C, vibration={c[1]:.2f} mm/s")

# ===========================================================================
# PART 3: Choosing K — The Elbow Method
# ===========================================================================
print("\n--- PART 3: Choosing K With the Elbow Method ---")

print("""
Problem: K-Means needs K as input, but we often don't know K!

Elbow method:
  - Try K = 1, 2, 3, ..., 10
  - For each K, compute inertia (sum of squared distances to centroids)
  - Plot K vs inertia
  - Look for the "elbow" — where inertia stops decreasing sharply
  
WHY: Adding more clusters always reduces inertia. The elbow shows
     where additional clusters give diminishing returns.
""")

k_values = range(1, 11)
inertias = []
sil_scores = []

for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    if k > 1:  # silhouette score requires at least 2 clusters
        sil = silhouette_score(X_scaled, km.labels_)
        sil_scores.append(sil)

print("K vs Inertia:")
for k, inertia in zip(k_values, inertias):
    print(f"  K={k}: inertia={inertia:.2f}")

# ===========================================================================
# PART 4: Silhouette Score — Better Than Elbow for Evaluation
# ===========================================================================
print("\n--- PART 4: Silhouette Score ---")

print("""
Silhouette score: measures how well each point fits its cluster.
  Range: -1 to +1
  +1: point is very well-matched to its cluster (far from other clusters)
   0: point is on the boundary between clusters
  -1: point is better matched to a neighboring cluster

For each point i:
  a(i) = avg distance to other points in SAME cluster
  b(i) = avg distance to points in NEAREST OTHER cluster
  s(i) = (b(i) - a(i)) / max(a(i), b(i))
  
  Silhouette score = mean of s(i) over all points
""")

print("\nSilhouette scores by K:")
for k, sil in zip(range(2, 11), sil_scores):
    marker = " ← best" if sil == max(sil_scores) else ""
    print(f"  K={k}: silhouette={sil:.4f}{marker}")

# Final evaluation with K=3
km3 = KMeans(n_clusters=3, n_init=10, random_state=42)
km3.fit(X_scaled)
final_sil = silhouette_score(X_scaled, km3.labels_)
print(f"\nK=3 silhouette score: {final_sil:.4f}")
print(f"Interpretation: {'Good' if final_sil > 0.5 else 'Moderate'} cluster separation")

# ===========================================================================
# PART 5: Visualize Everything
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("K-Means Clustering: Machine Sensor Analysis", fontsize=14, fontweight='bold')

COLORS = ['steelblue', 'coral', 'green']
LABELS_STR = ['Normal', 'High-Load', 'Degraded']

# 1. Raw data (no labels — this is what unsupervised learning sees)
axes[0, 0].scatter(X_machines[:, 0], X_machines[:, 1], alpha=0.5, color='gray', s=20)
axes[0, 0].set_xlabel("Temperature (°C)")
axes[0, 0].set_ylabel("Vibration (mm/s)")
axes[0, 0].set_title("Raw Data (No Labels)")
axes[0, 0].grid(True, alpha=0.3)

# 2. Discovered clusters
for i in range(3):
    mask = km3.labels_ == i
    axes[0, 1].scatter(X_machines[mask, 0], X_machines[mask, 1],
                       color=COLORS[i], alpha=0.6, s=20, label=f'Cluster {i}')

# Plot centroids (inverse-transformed)
c_orig = scaler.inverse_transform(km3.cluster_centers_)
axes[0, 1].scatter(c_orig[:, 0], c_orig[:, 1], c='black', marker='X',
                   s=200, zorder=5, label='Centroids')
axes[0, 1].set_xlabel("Temperature (°C)")
axes[0, 1].set_ylabel("Vibration (mm/s)")
axes[0, 1].set_title("K-Means Discovered Clusters (K=3)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Elbow plot
axes[1, 0].plot(list(k_values), inertias, 'o-', color='steelblue', linewidth=2)
axes[1, 0].axvline(x=3, color='red', linestyle='--', label='Optimal K=3')
axes[1, 0].set_xlabel("Number of Clusters (K)")
axes[1, 0].set_ylabel("Inertia")
axes[1, 0].set_title("Elbow Method for K Selection")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Silhouette scores
axes[1, 1].bar(range(2, 11), sil_scores, color='coral', alpha=0.8)
axes[1, 1].axvline(x=3, color='red', linestyle='--', label='K=3')
axes[1, 1].set_xlabel("Number of Clusters (K)")
axes[1, 1].set_ylabel("Silhouette Score")
axes[1, 1].set_title("Silhouette Score by K (higher = better)")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kmeans_analysis.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nVisualization saved: {OUTPUT_DIR}/kmeans_analysis.png")

# ===========================================================================
# PART 6: Predicting New Points
# ===========================================================================
print("\n--- PART 6: Predicting New Points ---")

new_machines = np.array([
    [68, 0.9],   # looks normal
    [94, 5.8],   # looks degraded
    [83, 3.2],   # looks high-load
])

new_scaled = scaler.transform(new_machines)
predictions = km3.predict(new_scaled)

print("New machine readings → Cluster assignment:")
for reading, cluster in zip(new_machines, predictions):
    print(f"  Temp={reading[0]}°C, Vib={reading[1]:.1f} mm/s → Cluster {cluster}")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. K-Means: iteratively assigns points to nearest centroid and updates centroids
2. ALWAYS scale features before K-Means (it uses distances!)
3. Choose K using: Elbow method + Silhouette score
4. Silhouette score: -1 to +1, higher = better defined clusters
5. K-Means limitation: assumes spherical clusters of similar size
   (DBSCAN and hierarchical clustering handle arbitrary shapes better)
6. After clustering: interpret each cluster by analyzing its centroid values

Next: 02_hierarchical_clustering.py — Bottom-up clustering with dendrograms
""")
