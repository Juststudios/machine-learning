# Module 5: Clustering — Unsupervised Learning

## Overview

Clustering is the task of **grouping similar data points together — without any labels**. You don't tell the algorithm what the groups should be; it discovers the structure on its own.

This is **unsupervised learning**: the data has no y (no target labels).

**Why does this matter for engineers?**
- Sometimes you *don't have* labels (labeling is expensive or impossible)
- Sometimes you want to *discover* structure you didn't know existed
- Anomaly detection: find data points that don't fit any cluster

---

## Prerequisites

- NumPy arrays and broadcasting
- Matplotlib subplots
- Module 4 (Classification) — understanding feature spaces and distance
- Basic linear algebra: vectors, dot products, covariance

---

## Learning Objectives

By the end of this module you will be able to:

1. Explain the difference between supervised (classification) and unsupervised (clustering) learning
2. Implement K-Means clustering and choose K using the Elbow method
3. Build and interpret hierarchical clustering dendrograms
4. Apply DBSCAN for density-based clustering and anomaly detection
5. Use PCA to reduce high-dimensional data to 2D for visualization
6. Evaluate cluster quality without ground-truth labels (silhouette score)

---

## Concept Map

```
                 ┌──────────────────────────────────────┐
                 │      UNSUPERVISED LEARNING            │
                 │  "Find structure in data — no labels" │
                 └───────────────┬──────────────────────┘
                                 │
         ┌───────────────────────┼──────────────────────────┐
         ▼                       ▼                          ▼
   Partition-Based          Hierarchical              Density-Based
   ┌─────────────┐         ┌───────────┐            ┌─────────────┐
   │   K-Means   │         │ Ward /    │            │   DBSCAN    │
   │             │         │ Complete  │            │             │
   │ Assigns each│         │ Average   │            │ Finds dense │
   │ point to    │         │ Linkage   │            │ regions;    │
   │ nearest     │         │           │            │ marks noise │
   │ centroid    │         │ Dendrogram│            │ as -1       │
   └─────────────┘         └───────────┘            └─────────────┘

         Dimensionality Reduction (for visualization + preprocessing)
         ┌─────────────────────────────────────────────────────────┐
         │  PCA (linear projection onto max-variance directions)   │
         │  t-SNE (non-linear, great for visualization only)       │
         └─────────────────────────────────────────────────────────┘
```

---

## The Core Challenge: Evaluating Without Labels

In supervised learning, you have ground truth → measure accuracy.
In clustering, you have NO labels → how do you know if clusters are good?

**Internal metrics** (no labels needed):
- **Inertia (WCSS):** Sum of squared distances from points to their cluster centroid. Lower is better — but always decreases as K increases (Elbow method needed).
- **Silhouette Score:** Measures how similar a point is to its own cluster vs. neighboring clusters. Range: [-1, 1]. Higher is better.

**External metrics** (if you happen to have labels for validation):
- **Adjusted Rand Index (ARI):** Compares cluster assignments to ground truth labels
- **Normalized Mutual Information (NMI):** Information-theoretic similarity

---

## File Guide

| File | What it teaches | Difficulty |
|------|----------------|------------|
| `01_kmeans.py` | K-Means algorithm, Elbow method, Silhouette score, cluster visualization | ⭐⭐ |
| `02_hierarchical_clustering.py` | Agglomerative clustering, dendrograms, linkage methods | ⭐⭐ |
| `03_dbscan.py` | Density-based clustering, noise handling, anomaly detection | ⭐⭐⭐ |
| `04_dimensionality_reduction.py` | PCA, explained variance, curse of dimensionality | ⭐⭐⭐ |
| `exercises.py` | 4-tier practice problems | ⭐ to ⭐⭐⭐⭐ |

---

## Running Instructions

```bash
cd /home/settings/Documents/pearl/machine-learning/05_clustering/
python 01_kmeans.py
python 02_hierarchical_clustering.py
python 03_dbscan.py
python 04_dimensionality_reduction.py
ls output/
```

---

## Real-World Connections

### Customer Segmentation (E-commerce / Marketing)
No one labels customers as "budget shopper" or "premium buyer." Clustering on purchase history + frequency + value discovers these segments automatically → targeted marketing.

### Anomaly Detection (Cybersecurity / Manufacturing)
DBSCAN marks points that don't belong to any dense cluster as noise (label = -1). These outliers are anomalies: a machine behaving unusually, a user account doing suspicious things.

### Image Compression (Computer Graphics)
K-Means on pixel colors: replace each pixel's color with the nearest centroid color. With K=16 centroids, you need only 4 bits per pixel instead of 24 → 6× compression.

### Genomics (Bioinformatics)
Cluster gene expression profiles across thousands of patients. Discover patient subgroups that respond differently to treatment — without having been told those subgroups exist.

### Sensor Network Analysis (IoT / Smart Buildings)
Cluster temperature + humidity + CO₂ readings from 100 room sensors. Find rooms that behave similarly without being told which rooms are offices vs. labs vs. conference rooms.

---

## K-Means vs. Hierarchical vs. DBSCAN: Quick Guide

| Question | K-Means | Hierarchical | DBSCAN |
|----------|---------|-------------|--------|
| Need to specify K upfront? | Yes | No (choose cut) | No |
| Handles arbitrary shapes? | No (spherical) | Partially | Yes |
| Handles noise/outliers? | No | No | Yes (marks as -1) |
| Scales to large datasets? | Yes | No (O(n²)) | Yes |
| Produces nested structure? | No | Yes | No |
| Best for | Large, spherical clusters | Exploring structure | Noisy data, arbitrary shapes |

---

## Next Steps

After completing this module:
- **Module 6:** Model Evaluation — how to properly evaluate and compare any ML model
- **Module 7:** Neural Networks — learn representations automatically from raw data
