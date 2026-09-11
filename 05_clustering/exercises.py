"""
Exercises: Clustering and Dimensionality Reduction (Module 5)
==============================================================
4-tier exercises covering K-Means, DBSCAN, and PCA.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons, load_iris
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
np.random.seed(42)

print("=" * 60)
print("CLUSTERING & DIMENSIONALITY REDUCTION — EXERCISES")
print("=" * 60)

# ===========================================================================
# LEVEL 1: RECALL
# ===========================================================================
print("\n--- LEVEL 1: RECALL ---")
print("""
Study the code below. Then answer:

1a. K-Means uses labels_ and cluster_centers_. What does each store?
1b. What does the Silhouette Score measure? What is its range?
1c. Why must you scale features BEFORE K-Means?
1d. What does DBSCAN label=-1 mean?
1e. In PCA, what is an "explained variance ratio" of 0.35?
""")

# Demo
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

scaler = StandardScaler()
X_s = scaler.fit_transform(X_iris)

km = KMeans(n_clusters=3, n_init=10, random_state=42)
km.fit(X_s)

print(f"K-Means on Iris (K=3):")
print(f"  labels_ unique: {np.unique(km.labels_)}")
print(f"  cluster_centers_ shape: {km.cluster_centers_.shape}")
print(f"  Silhouette score: {silhouette_score(X_s, km.labels_):.4f}")

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_s)
print(f"\nPCA on Iris (2 components):")
print(f"  Explained variance: {pca.explained_variance_ratio_}")
print(f"  Total explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ===========================================================================
# LEVEL 2: DEBUGGING
# ===========================================================================
print("\n--- LEVEL 2: DEBUGGING ---")
print("""
Find the 3 bugs in this clustering pipeline.
""")

buggy = '''
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

np.random.seed(0)
X = np.random.randn(200, 4)
X[:100] += 3   # two natural clusters

# BUG 1: Fit K-Means on UNSCALED data, but compare with scaled silhouette
km = KMeans(n_clusters=2, n_init=10)
km.fit(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# BUG 2: Computing silhouette on scaled data but labels from unscaled fit
sil = silhouette_score(X_scaled, km.labels_)
print(f"Silhouette: {sil:.4f}")

# BUG 3: Using silhouette_score with K=1 (invalid, needs >= 2 clusters)
km1 = KMeans(n_clusters=1, n_init=10).fit(X_scaled)
sil1 = silhouette_score(X_scaled, km1.labels_)
print(f"K=1 silhouette: {sil1:.4f}")
'''
print(buggy)
print("""
Bug 1: ________________________________________________
Bug 2: ________________________________________________
Bug 3: ________________________________________________
TODO: Write the corrected version
""")

# ===========================================================================
# LEVEL 3: APPLICATION — Customer Segmentation
# ===========================================================================
print("\n--- LEVEL 3: APPLICATION ---")
print("""
TASK: Segment retail customers using K-Means and PCA.

Dataset (generate it):
  Customer features:
    - annual_spend_usd:   500 to 50,000
    - purchase_frequency: 1 to 52 (times per year)
    - avg_basket_size:    10 to 500 (USD)
    - loyalty_years:      0 to 15
    - online_pct:         0 to 100 (% of purchases online)
  
  n = 500 customers

Requirements:
  1. Generate synthetic customer data with 3-4 natural segments
     (e.g., budget shoppers, regular customers, VIP customers)
  2. Scale features
  3. Use elbow method (K=1..10) to identify the optimal K
  4. Compute Silhouette scores for K=2..8 and identify the best K
  5. Fit K-Means with the best K
  6. Use PCA to visualize customers in 2D, colored by cluster
     - Save to output/customer_segments.png
  7. Characterize each cluster:
     - Print the centroid (unscaled) for each cluster
     - Name each cluster (e.g., "Budget Shopper", "VIP Customer")
""")

# Starter data
n = 500
segment = np.random.choice(3, n, p=[0.5, 0.35, 0.15])

annual_spend = np.where(segment==0,
    np.random.uniform(500, 5000, n),
    np.where(segment==1,
        np.random.uniform(5000, 20000, n),
        np.random.uniform(20000, 50000, n)
    )
)
frequency = np.where(segment==0,
    np.random.uniform(1, 8, n),
    np.where(segment==1,
        np.random.uniform(8, 30, n),
        np.random.uniform(30, 52, n)
    )
)
basket    = annual_spend / (frequency * 52) * 52 + np.random.randn(n) * 20
loyalty   = np.random.uniform(0, 15, n)
online_pct = np.random.uniform(0, 100, n)

X_cust = np.column_stack([annual_spend, frequency, basket, loyalty, online_pct])
feature_names = ['annual_spend', 'frequency', 'basket_size', 'loyalty_years', 'online_pct']

print(f"Customer data generated: {X_cust.shape}")
print("TODO: Complete the customer segmentation analysis")

# ===========================================================================
# LEVEL 4: CHALLENGE — K-Means From Scratch
# ===========================================================================
print("\n--- LEVEL 4: CHALLENGE ---")
print("""
TASK: Implement K-Means clustering from scratch using only numpy.

class KMeansScratch:
    def __init__(self, k=3, max_iter=100, tol=1e-4):
    
    def fit(self, X):
        # 1. Initialize K centroids by randomly sampling K data points
        # 2. Repeat until convergence (or max_iter):
        #    a. Assign each point to nearest centroid
        #       distances = ||x - centroid||² for each centroid
        #       labels = argmin(distances, axis=1)
        #    b. Recompute centroids as mean of assigned points
        #    c. Check convergence: if centroids moved < tol, stop
        # 3. Store self.labels_ and self.cluster_centers_
        return self
    
    def predict(self, X):
        # Assign each new point to nearest centroid
        pass

Verify:
  - On make_blobs with 3 centers, your implementation should give
    silhouette_score ≥ 0.6 and match sklearn's KMeans result closely
  - Compare your cluster centers with sklearn's centers (should be similar)
""")

class KMeansScratch:
    """TODO: Implement K-Means from scratch."""
    
    def __init__(self, k=3, max_iter=100, tol=1e-4, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
    
    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        # TODO: Implement K-Means
        # Step 1: Random initialization
        # Step 2: Iterate (assign + update + check convergence)
        self.labels_ = np.zeros(len(X), dtype=int)  # REPLACE
        self.cluster_centers_ = X[:self.k]            # REPLACE
        return self
    
    def predict(self, X):
        # TODO: Assign each point to nearest center
        return np.zeros(len(X), dtype=int)  # REPLACE

# Test data
X_test, y_test = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
X_test_s = StandardScaler().fit_transform(X_test)

# Reference
km_ref = KMeans(n_clusters=3, n_init=10, random_state=42)
km_ref.fit(X_test_s)
sil_ref = silhouette_score(X_test_s, km_ref.labels_)
print(f"sklearn KMeans silhouette: {sil_ref:.4f}  ← match this!")

# Your implementation
km_scratch = KMeansScratch(k=3, random_state=42)
km_scratch.fit(X_test_s)
# Should give silhouette close to sil_ref
print("TODO: Implement KMeansScratch — should get silhouette ≥ 0.6")

print("\n" + "=" * 60)
print("EXERCISES COMPLETE")
print("Check solutions/clustering_solutions.py for reference answers.")
print("=" * 60)
