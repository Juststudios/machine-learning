"""
DBSCAN: Density-Based Clustering
==================================
K-Means assumption: clusters are spherical and roughly equal size.
Real data is messier — clusters can be crescent-shaped, irregular,
surrounded by noise, or of vastly different densities.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
  - Finds clusters of ARBITRARY SHAPE
  - Automatically detects OUTLIERS (marks them as noise, label=-1)
  - Does NOT require specifying K in advance
  - Works by: "if there are enough nearby points, they form a cluster"

Key parameters:
  epsilon (eps): neighborhood radius — "how far to look for neighbors"
  min_samples:   minimum points to form a dense region (core point)
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_blobs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

np.random.seed(42)

# ===========================================================================
# PART 1: The Problem K-Means Can't Solve
# ===========================================================================
print("=" * 60)
print("PART 1: When K-Means Fails")
print("=" * 60)

print("""
K-Means assumes clusters are convex (roughly spherical blobs).
On crescent or ring-shaped data, K-Means fails completely.

Real examples where DBSCAN wins:
  - Geographic clustering (neighborhoods separated by rivers)
  - Anomaly detection in network traffic (outliers = attacks)
  - Clustering star clusters with irregular shapes
""")

# Crescent data — K-Means will fail here
X_moon, y_moon = make_moons(n_samples=300, noise=0.08, random_state=42)

scaler = StandardScaler()
X_moon_s = scaler.fit_transform(X_moon)

kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
labels_km = kmeans.fit_predict(X_moon_s)

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_db = dbscan.fit_predict(X_moon_s)

n_noise_db = (labels_db == -1).sum()
n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)

print(f"K-Means (K=2) — clusters: 2 (forced)")
print(f"DBSCAN        — clusters: {n_clusters_db}, noise points: {n_noise_db}")

# ===========================================================================
# PART 2: How DBSCAN Works
# ===========================================================================
print("\n--- PART 2: DBSCAN Algorithm ---")

print("""
DBSCAN classifies each point as:

1. CORE POINT: has at least min_samples neighbors within radius eps
   → Forms the dense interior of a cluster

2. BORDER POINT: not a core point, but within eps of a core point
   → Belongs to a cluster, but not its dense core

3. NOISE POINT (label = -1): neither core nor border
   → Outlier! Not assigned to any cluster

Algorithm:
  1. Pick any unvisited point
  2. If it has ≥ min_samples neighbors within eps → it's a core point
     → Start a new cluster, expand it by adding reachable core points
  3. If it has < min_samples neighbors → mark as noise (for now)
  4. Repeat until all points visited
  
  (Border points near a core point get absorbed into its cluster)
""")

# ===========================================================================
# PART 3: Epsilon and min_samples — Choosing Parameters
# ===========================================================================
print("--- PART 3: Choosing eps and min_samples ---")

print("""
min_samples: rule of thumb: ≥ dimensionality + 1
  More noise → increase min_samples (require denser cores)

epsilon: use the k-distance plot
  - Compute distance from each point to its k-th nearest neighbor (k = min_samples)
  - Sort distances and plot
  - "Elbow" in the plot → good epsilon value
""")

# Demonstrate k-distance plot
from sklearn.neighbors import NearestNeighbors

k = 5  # min_samples
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_moon_s)
distances, _ = nn.kneighbors(X_moon_s)
k_distances = np.sort(distances[:, k-1])  # distance to k-th neighbor

print(f"K-distance plot statistics:")
print(f"  Min:    {k_distances.min():.4f}")
print(f"  Median: {np.median(k_distances):.4f}")
print(f"  Max:    {k_distances.max():.4f}")
print(f"  Look for the elbow → suggests eps ≈ 0.3 for this dataset")

# ===========================================================================
# PART 4: Anomaly Detection With DBSCAN
# ===========================================================================
print("\n--- PART 4: Anomaly Detection ---")

print("""
DBSCAN's noise points (label=-1) ARE the anomalies.
This makes DBSCAN a natural anomaly detector.

Application: manufacturing process monitoring
  - Normal machines: operate in tight sensor clusters
  - Anomalous machines: sensors deviate from the cluster → noise points
""")

# Generate normal machine data + anomalies
n_normal  = 250
n_anomaly = 20

X_normal   = np.random.randn(n_normal, 2) * 0.5 + np.array([2, 3])   # tight cluster
X_anomaly  = np.random.uniform(-3, 8, (n_anomaly, 2))                 # scattered anomalies

X_all = np.vstack([X_normal, X_anomaly])
true_labels = np.array([0]*n_normal + [1]*n_anomaly)  # 0=normal, 1=anomaly

X_all_s = StandardScaler().fit_transform(X_all)

db_anomaly = DBSCAN(eps=0.4, min_samples=8)
detected = db_anomaly.fit_predict(X_all_s)

# Points labeled -1 by DBSCAN are detected anomalies
detected_anomalies = (detected == -1)

# Evaluation
true_positives  = (detected_anomalies & (true_labels == 1)).sum()
false_positives = (detected_anomalies & (true_labels == 0)).sum()
false_negatives = ((~detected_anomalies) & (true_labels == 1)).sum()

precision = true_positives / (true_positives + false_positives + 1e-8)
recall    = true_positives / (true_positives + false_negatives + 1e-8)

print(f"Anomaly detection results:")
print(f"  True anomalies:      {n_anomaly}")
print(f"  Detected as noise:   {detected_anomalies.sum()}")
print(f"  True positives:      {true_positives}")
print(f"  False positives:     {false_positives}")
print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")

# ===========================================================================
# PART 5: Comparing DBSCAN vs K-Means on Multiple Shapes
# ===========================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("DBSCAN vs K-Means on Different Data Shapes", fontsize=13, fontweight='bold')

COLORS = ['steelblue', 'coral', 'green', 'purple', 'orange', 'brown', 'pink']

datasets = [
    ("Crescent (moons)", X_moon_s, 2, (0.3, 5)),
    ("Blobs",            StandardScaler().fit_transform(make_blobs(300, centers=3, random_state=42)[0]), 3, (0.5, 5)),
    ("Anomaly Detection",X_all_s, 2, (0.4, 8)),
]

for col, (name, X_data, k_km, (eps_db, ms_db)) in enumerate(datasets):
    # K-Means
    km_labels = KMeans(n_clusters=k_km, n_init=10, random_state=42).fit_predict(X_data)
    unique_km = sorted(set(km_labels))
    for lbl in unique_km:
        mask = km_labels == lbl
        axes[0, col].scatter(X_data[mask, 0], X_data[mask, 1],
                             c=COLORS[lbl % len(COLORS)], s=20, alpha=0.7)
    axes[0, col].set_title(f"K-Means (K={k_km}): {name}")
    axes[0, col].grid(True, alpha=0.3)
    
    # DBSCAN
    db_labels = DBSCAN(eps=eps_db, min_samples=ms_db).fit_predict(X_data)
    unique_db = sorted(set(db_labels))
    n_cl = len(unique_db) - (1 if -1 in unique_db else 0)
    n_ns = (db_labels == -1).sum()
    for lbl in unique_db:
        mask = db_labels == lbl
        color = 'black' if lbl == -1 else COLORS[lbl % len(COLORS)]
        marker = 'x' if lbl == -1 else 'o'
        label = 'Noise' if lbl == -1 else f'Cluster {lbl}'
        axes[1, col].scatter(X_data[mask, 0], X_data[mask, 1],
                             c=color, s=20, alpha=0.7, marker=marker)
    axes[1, col].set_title(f"DBSCAN (eps={eps_db}): {n_cl} clusters, {n_ns} noise")
    axes[1, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dbscan_comparison.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nComparison plot saved: {OUTPUT_DIR}/dbscan_comparison.png")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. DBSCAN: finds clusters of arbitrary shape, no need to specify K
2. Noise points (label=-1) = natural anomaly detection
3. Two parameters: eps (neighborhood radius), min_samples (density threshold)
4. Choose eps using k-distance plot (look for the elbow)
5. rule of thumb for min_samples: ≥ n_features + 1

When to use DBSCAN:
  ✓ Clusters of unknown or irregular shape
  ✓ Presence of noise/outliers in data
  ✓ Anomaly detection
  
When to use K-Means instead:
  ✓ Large datasets (K-Means scales better)
  ✓ Roughly spherical, well-separated clusters
  ✓ You know K in advance

Next: 04_dimensionality_reduction.py — PCA for high-dimensional data
""")
