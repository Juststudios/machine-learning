# Module 4: Classification

## Overview

Classification is the task of predicting **which category** an input belongs to — as opposed to regression, which predicts a continuous number.

**Examples by domain:**
- 🏭 **Manufacturing:** Is this sensor reading OK, WARNING, or FAULT?
- 📧 **Email:** Is this message spam or not spam?
- 🏥 **Medicine:** Does this patient have disease X?
- 🖼️ **Computer Vision:** Which object is in this image?

---

## Prerequisites

Before starting this module you should be comfortable with:
- NumPy arrays and basic statistics
- Pandas DataFrames and CSV loading
- Matplotlib plots
- Linear algebra basics (dot products, matrix multiplication)
- Module 3 (Linear & Polynomial Regression) concepts

---

## Learning Objectives

By the end of this module you will be able to:

1. Explain the difference between regression and classification
2. Implement and interpret Logistic Regression (binary and multi-class)
3. Build and visualize Decision Trees — and explain splits
4. Understand why Random Forests outperform single trees
5. Apply SVMs with linear and RBF kernels
6. Implement K-Nearest Neighbors and tune the K hyperparameter
7. Choose the right classifier for a given problem
8. Interpret key classification metrics (accuracy, precision, recall, F1, AUC)

---

## Concept Map

```
                    ┌─────────────────────────────────────┐
                    │         CLASSIFICATION               │
                    │   "Which category does this belong   │
                    │    to?" — output is a discrete label │
                    └──────────────┬──────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
   Linear Models           Tree Models               Instance-Based
   ┌───────────┐         ┌─────────────┐            ┌───────────────┐
   │ Logistic  │         │  Decision   │            │      KNN      │
   │Regression │         │    Tree     │            │  (K-Nearest   │
   │           │         │             │            │  Neighbors)   │
   │ Sigmoid   │         │ Random      │            │               │
   │ function  │         │ Forest      │            └───────────────┘
   │ 0→1 prob  │         │             │
   └───────────┘         │ Gradient    │           Kernel Methods
                         │ Boosting    │           ┌───────────────┐
                         └─────────────┘           │      SVM      │
                                                   │  (Support     │
                                                   │   Vector      │
                                                   │   Machine)    │
                                                   └───────────────┘
```

---

## Key Concepts Introduced

### Decision Boundaries
A **decision boundary** is the line (or curve, or hyperplane) that separates one class from another in feature space. Every classifier learns a different *shape* of boundary:
- **Logistic Regression:** straight line / hyperplane
- **Decision Tree:** axis-aligned rectangular regions
- **SVM (RBF):** curved, smooth boundary
- **KNN:** irregular, local boundary

### Classification Metrics — Quick Reference

| Metric | Formula | When to use |
|--------|---------|-------------|
| **Accuracy** | (TP+TN)/Total | Balanced classes |
| **Precision** | TP/(TP+FP) | When false positives are costly (spam filter) |
| **Recall** | TP/(TP+FN) | When false negatives are costly (fault detection!) |
| **F1 Score** | 2·P·R/(P+R) | Imbalanced data, balance P and R |
| **ROC-AUC** | Area under ROC curve | Ranking quality, independent of threshold |

> ⚠️ Engineering warning: In fault detection, a missed fault (false negative) is far more dangerous than a false alarm (false positive). Always choose your metric *before* training, based on the cost of each error type.

---

## File Guide

| File | What it teaches | Difficulty |
|------|----------------|------------|
| `01_logistic_regression.py` | Sigmoid function, binary/multi-class classification, probability outputs | ⭐⭐ |
| `02_decision_trees.py` | Gini impurity, tree visualization, overfitting via depth | ⭐⭐ |
| `03_random_forests.py` | Bagging, ensemble methods, feature importance | ⭐⭐⭐ |
| `04_svm.py` | Margins, kernel trick, C parameter | ⭐⭐⭐ |
| `05_knn.py` | Lazy learning, distance metrics, K selection | ⭐⭐ |
| `exercises.py` | 4-tier practice problems | ⭐ to ⭐⭐⭐⭐ |

---

## Running Instructions

```bash
cd /home/settings/Documents/pearl/machine-learning/04_classification/
python 01_logistic_regression.py
python 02_decision_trees.py
python 03_random_forests.py
python 04_svm.py
python 05_knn.py
ls output/
```

---

## Real-World Connections

**Predictive Maintenance (Manufacturing):** Vibration + temperature → OK / WARNING / FAULT. A model that catches 95% of faults before they happen saves millions in unplanned downtime.

**Medical Diagnostics:** Blood test results → disease present / absent. Missing a positive (false negative) can cost a life — recall matters more than precision.

**Network Intrusion Detection:** Packets → normal / attack. 0.1% false positive rate at millions of packets/second floods analysts with false alerts.

**Quality Control (Vision):** Camera image → pass / fail. Model must be interpretable so engineers understand *why* a part was rejected.

---

## Comparison Guide: Which Classifier to Choose?

| Situation | Recommended Classifier | Why |
|-----------|----------------------|-----|
| Need probability outputs + interpretability | Logistic Regression | Coefficients are interpretable |
| Need to explain decisions to stakeholders | Decision Tree | "If temp > 80°C AND vibration > 5mm/s → FAULT" |
| Best accuracy on tabular data | Random Forest / Gradient Boosting | Ensemble reduces variance |
| High-dimensional sparse data (text, NLP) | SVM (linear kernel) | Works well in high dimensions |
| Small dataset, distance makes sense | KNN | Simple, no assumptions about data shape |
| Non-linear boundary, medium dataset | SVM (RBF kernel) | Maximum margin + kernel trick |

---

## Next Steps

After completing this module:
- **Module 5:** Clustering — unsupervised learning (no labels required)
- **Module 6:** Model Evaluation — cross-validation, metrics deep dive, hyperparameter tuning
- **Module 7:** Neural Networks — learn non-linear representations automatically
