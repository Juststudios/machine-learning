# Module 6: Model Evaluation — The Most Critical Skill in ML

## Overview

Building a model is easy. **Knowing whether your model actually works** is hard.

This module covers everything you need to rigorously evaluate machine learning models — the skills that separate practitioners who ship reliable systems from those who ship quietly broken ones.

> **Core Idea**: A model with 99% accuracy can be completely useless. A model with 70% accuracy can save lives. The metric you choose changes everything.

---

## Prerequisites

Before starting, you should be comfortable with:
- Supervised learning concepts (Modules 1–5)
- Training a classifier and regressor with sklearn
- NumPy arrays and Pandas DataFrames
- Basic probability (conditional probability, distributions)

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Choose the right metric** for any ML problem (don't default to accuracy)
2. **Diagnose overfitting and underfitting** using learning curves
3. **Use cross-validation correctly** to get unbiased performance estimates
4. **Tune hyperparameters** without leaking test information
5. **Handle class imbalance** with resampling and weighted loss
6. **Decompose error** into bias and variance components
7. **Build evaluation pipelines** that hold up in production

---

## Why Evaluation is the Most Important Skill in ML

Consider this real scenario:

> You build a fraud detection model. It classifies 99.7% of transactions correctly. You deploy it. Fraud losses don't change. Why?
>
> Because 99.7% of transactions are legitimate. Your model learned to predict "not fraud" for everything — and it's 99.7% accurate.

**Accuracy on imbalanced data is a trap.** This module teaches you to avoid it and dozens of other evaluation mistakes.

---

## The Danger of Accuracy on Imbalanced Data

| Dataset | Positive % | "Always predict negative" accuracy |
|---------|-----------|-----------------------------------|
| Email spam | 20% | 80% |
| Credit card fraud | 0.1% | 99.9% |
| Disease diagnosis | 2% | 98% |
| Equipment failure | 0.5% | 99.5% |

In all these cases, a model that predicts the majority class 100% of the time achieves high accuracy while being completely useless. You need **precision, recall, F1, and AUC** to see through this.

---

## Overfitting vs Underfitting — Diagnosing with Learning Curves

```
High Bias (Underfitting)          High Variance (Overfitting)
─────────────────────             ─────────────────────────
Train score:  low                 Train score:  high
Val score:    low                 Val score:    low
Gap:          small               Gap:          large
Fix:          bigger model        Fix:          more data,
              more features                     regularization
              less regularization               dropout, etc.
```

**Learning curves** plot train and validation score vs. training set size. They reveal *why* your model fails — not just *that* it fails.

---

## Concept Map

```
Model Evaluation
├── Metrics
│   ├── Classification: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
│   └── Regression: MAE, MSE, RMSE, R²
│
├── Validation Strategy
│   ├── Train/Val/Test Split
│   ├── K-Fold Cross-Validation
│   ├── Stratified K-Fold
│   ├── Leave-One-Out (LOO)
│   └── Time Series Split (no data leakage!)
│
├── Hyperparameter Tuning
│   ├── GridSearchCV (exhaustive)
│   ├── RandomizedSearchCV (efficient)
│   └── Nested CV (unbiased estimate)
│
├── Bias-Variance Tradeoff
│   ├── Underfitting (high bias)
│   ├── Overfitting (high variance)
│   └── The decomposition: Error = Bias^2 + Variance + Noise
│
└── Imbalanced Data
    ├── class_weight='balanced'
    ├── Oversampling (SMOTE)
    ├── Undersampling
    └── Right metrics: F1, PR-AUC
```

---

## File Guide

| File | What You Learn | Key Concepts |
|------|---------------|--------------|
| `01_metrics.py` | Every important metric with real interpretation | Precision/Recall tradeoff, ROC vs PR curve |
| `02_cross_validation.py` | Why single train/test splits lie to you | K-Fold, Stratified, LOO, TimeSeriesSplit |
| `03_hyperparameter_tuning.py` | Finding the best model without cheating | GridSearch, RandomSearch, learning curves |
| `04_bias_variance_tradeoff.py` | The fundamental tension in ML | Polynomial regression as a diagnostic |
| `05_imbalanced_data.py` | Handling skewed datasets safely | Resampling, weighted loss, better metrics |
| `exercises.py` | Practice at 4 difficulty levels | — |

---

## Running Instructions

```bash
cd 06_model_evaluation/

python 01_metrics.py
python 02_cross_validation.py
python 03_hyperparameter_tuning.py
python 04_bias_variance_tradeoff.py
python 05_imbalanced_data.py
python exercises.py
```

Plots are saved to `output/` as PNG files.

---

## Real-World Connections

- **Medical diagnosis**: Recall is critical — missing a cancer case (false negative) is worse than a false alarm
- **Spam filtering**: Precision matters — incorrectly flagging important email is unacceptable
- **Fraud detection**: Both matter — catch fraud (recall) without annoying legitimate users (precision)
- **Weather forecasting**: Time series CV prevents you from "seeing the future" during training
- **Ad click prediction**: Heavy class imbalance (0.1% click rate) requires special handling

---

## Next Steps

After completing this module:
- **Module 7**: Deep Learning Introduction — when do we need neural networks?
- **Module 8**: Feature Engineering — garbage in, garbage out
- **Module 9**: Ensemble Methods — why combining models beats single models

---

*"All models are wrong, but some are useful. Evaluation tells you which ones."* — adapted from George Box
