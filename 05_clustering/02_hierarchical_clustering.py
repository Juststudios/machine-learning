"""
Hierarchical Clustering: Building a Cluster Tree
==================================================
K-Means problem: you must choose K before seeing the data.

Hierarchical clustering builds a TREE (dendrogram) showing
ALL possible clusterings from n individual clusters to 1.
You choose the cut point AFTER seeing the full picture.

Two approaches:
  Agglomerative (bottom-up): start with n clusters (one per point),
    repeatedly merge the closest pair until one cluster remains.
  Divisive (top-down): start with 1 cluster, recursively split.
  (Agglomerative is far more common in practice)

Real-world uses:
  - Gene expression clustering in bioinformatics
  - Document grouping (topic hierarchies)
  - Customer segmentation with natural sub-groups
  - Product categorization in e-commerce
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
np.random.seed(42)

# ===========================================================================
# PART 1: How Agglomerative Clustering Works
# ===========================================================================
print("=" * 60)
print("PART 1: Agglomerative Hierarchical Clustering")
print("=" * 60)

print("""
Algorithm (single pass, agglomerative):
  1. Start: n clusters, one per data point
  2. Compute distance matrix between ALL pairs of clusters
  3. Merge the TWO closest clusters → now n-1 clusters
  4. Recompute distances involving the new merged cluster
  5. Repeat until only 1 cluster remains

Linkage criteria (how to measure distance between clusters):
  'single':   min distance between any two points across clusters
              → tends to create chain-like, elongated clusters
  'complete': max distance between any two points across clusters
              → tends to create compact, evenly sized clusters
  'average':  average distance between all pairs across clusters
              → compromise, generally good
  'ward':     minimize within-cluster variance at each merge
              → most similar to K-Means, usually best default
""")

# Small example to trace
X_small = np.array([[1, 1], [1.5, 1.5], [3, 3], [3.5, 3.5], [5, 5]])
labels_small = ['A', 'B', 'C', 'D', 'E']

print("Small 5-point example:")
for label, point in zip(labels_small, X_small):
    print(f"  {label}: {point}")

# Compute linkage
Z = linkage(X_small, method='ward')
print(f"\nWard linkage matrix (each row = one merge):")
print(f"  [cluster1, cluster2, distance, new_size]")
for row in Z:
    print(f"  {row}")

print("""
Reading the linkage matrix:
  Row 0: merged clusters 0 (A) and 1 (B) at distance 0.71 → new cluster 5
  Row 1: merged clusters 2 (C) and 3 (D) at distance 0.71 → new cluster 6
  Row 2: merged cluster 4 (E) with cluster 6 (C+D) → new cluster 7
  Row 3: merged cluster 5 (A+B) with cluster 7 (C+D+E) → final cluster 8
""")

# ===========================================================================
# PART 2: Dendrogram — Visualizing the Full Tree
# ===========================================================================
print("--- PART 2: The Dendrogram ---")
print("""
The dendrogram shows ALL merges at once.
  - X axis: individual samples (or cluster labels)
  - Y axis: distance at which merge occurred
  - Higher merge = more dissimilar clusters being joined

How to choose K from the dendrogram:
  Draw a horizontal line across the dendrogram.
  Count how many vertical lines it crosses → that's K.
  
  Rule: cut where there's the LARGEST vertical gap (biggest jump in distance).
  A big jump = the algorithm was forced to merge very different clusters.
""")

# Generate multi-cluster data for dendrogram
X_demo, y_demo = make_blobs(n_samples=30, centers=4, cluster_std=0.8, random_state=42)
scaler = StandardScaler()
X_demo_s = scaler.fit_transform(X_demo)

Z_demo = linkage(X_demo_s, method='ward')

# ===========================================================================
# PART 3: sklearn AgglomerativeClustering
# ===========================================================================
print("--- PART 3: sklearn AgglomerativeClustering ---")

# Generate sensor data
n_machines = 200
X_sensors, y_true = make_blobs(n_samples=n_machines, centers=4,
                                cluster_std=1.2, random_state=42)
X_sensors_s = StandardScaler().fit_transform(X_sensors)

# Compare linkage methods
methods = ['ward', 'complete', 'average', 'single']
print(f"{'Method':>10} | Silhouette Score")
print("-" * 35)

from sklearn.metrics import silhouette_score
results = {}
for method in methods:
    agg = AgglomerativeClustering(n_clusters=4, linkage=method)
    labels = agg.fit_predict(X_sensors_s)
    sil = silhouette_score(X_sensors_s, labels)
    results[method] = (labels, sil)
    print(f"  {method:>8}: {sil:.4f}")

best_method = max(results, key=lambda m: results[m][1])
print(f"\nBest linkage: '{best_method}'")

# ===========================================================================
# PART 4: Choosing K From the Dendrogram
# ===========================================================================
print("\n--- PART 4: Auto-select K from Distance Jumps ---")

# Examine merge distances in the linkage matrix
merge_distances = Z_demo[:, 2]
distance_gaps   = np.diff(merge_distances)

# Large gap → we just crossed a "natural break" between clusters
top_k_gaps_idx = np.argsort(distance_gaps)[-3:][::-1]

print("Largest distance jumps in the dendrogram:")
for i in top_k_gaps_idx:
    # K at this cut = total - (i+1) merges already done + 1
    k_at_cut = len(X_demo) - (i + 1)
    print(f"  After merge {i+1}: distance jumped {distance_gaps[i]:.3f} → suggests K={k_at_cut}")

# Use fcluster to cut the dendrogram at K=4
labels_cut = fcluster(Z_demo, t=4, criterion='maxclust')
sil_cut = silhouette_score(X_demo_s, labels_cut)
print(f"\nDendrogram cut at K=4: silhouette = {sil_cut:.4f}")

# ===========================================================================
# PART 5: Visualization
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Hierarchical Clustering Analysis", fontsize=13, fontweight='bold')

# Dendrogram
dendrogram(Z_demo, ax=axes[0, 0], color_threshold=2.5, leaf_font_size=6)
axes[0, 0].set_title("Dendrogram (Ward linkage)")
axes[0, 0].set_xlabel("Sample Index")
axes[0, 0].set_ylabel("Merge Distance")
axes[0, 0].axhline(y=2.5, color='red', linestyle='--', linewidth=1.5, label='Cut point (K=4)')
axes[0, 0].legend()

# Clusters (best method)
COLORS = ['steelblue', 'coral', 'green', 'purple']
best_labels = results[best_method][0]
for i in range(4):
    mask = best_labels == i
    axes[0, 1].scatter(X_sensors_s[mask, 0], X_sensors_s[mask, 1],
                       c=COLORS[i], s=25, alpha=0.7, label=f'Cluster {i}')
axes[0, 1].set_title(f"AgglomerativeClustering ({best_method} linkage, K=4)")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Compare linkage methods
for ax, method in zip([axes[1, 0], axes[1, 1]], ['single', 'ward']):
    labels_m = results[method][0]
    for i in range(4):
        mask = labels_m == i
        ax.scatter(X_sensors_s[mask, 0], X_sensors_s[mask, 1],
                   c=COLORS[i], s=25, alpha=0.7, label=f'C{i}')
    sil_m = results[method][1]
    ax.set_title(f"Linkage='{method}' (silhouette={sil_m:.3f})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hierarchical_clustering.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nVisualization saved: {OUTPUT_DIR}/hierarchical_clustering.png")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. Hierarchical clustering builds a TREE of all possible clusterings
2. No need to choose K in advance — choose after seeing the dendrogram
3. Read the dendrogram: cut where the LARGEST vertical gap appears
4. Ward linkage: best default (minimizes within-cluster variance)
5. Single linkage: creates chain-like clusters (usually not ideal)
6. scipy.hierarchy.linkage + dendrogram: full hierarchical analysis
7. sklearn.AgglomerativeClustering: for applying a specific K

K-Means vs Hierarchical:
  K-Means:       fast, scalable, works on millions of points
  Hierarchical:  slower (O(n² log n)), reveals structure, good for small data

Next: 03_dbscan.py — Density-based clustering for arbitrary shapes
""")
