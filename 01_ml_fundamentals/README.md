# Module 1: Machine Learning Fundamentals

## Overview

This module answers the foundational question every engineer asks before diving into ML:
**"What actually *is* machine learning, and why should I care?"**

You'll build a concrete mental model of what ML does differently from traditional programming,
walk through the complete ML workflow end-to-end, and learn the vocabulary that every
practitioner uses daily.

---

## Prerequisites

You already know (from your earlier coursework):

- **Python**: variables, loops, functions, list comprehensions, dicts
- **NumPy**: array creation, indexing, broadcasting, `np.mean`, `np.std`, `np.dot`
- **Pandas**: `read_csv`, `DataFrame`, `groupby`, `describe`, boolean filtering
- **Matplotlib**: `plt.plot`, `plt.scatter`, subplots, saving figures
- **Engineering Math**:
  - Linear Algebra: matrix multiplication, dot products, vector norms
  - Calculus: derivatives, chain rule, gradient (used during optimization)
  - Probability: mean, variance, conditional probability, Bayes' theorem

You do **not** need to know any ML yet — that's what this module teaches.

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Explain** the difference between traditional programming (rules → outputs) and ML (data + outputs → rules) with concrete examples
2. **Classify** a given problem as supervised, unsupervised, or reinforcement learning and justify your choice
3. **Execute** the complete ML workflow — from raw data to a trained, evaluated model — in Python
4. **Distinguish** features from targets, and explain why feature engineering often matters more than algorithm choice
5. **Apply** `train_test_split` correctly and explain *why* holding out test data is not optional
6. **Interpret** model evaluation metrics: MAE, MSE, RMSE, R² — what they mean in real units
7. **Identify** overfitting and underfitting from train/test error curves and suggest remedies

---

## Concept Map

```
Traditional Programming          Machine Learning
┌─────────────────────┐         ┌──────────────────────────────┐
│  Rules (hand-coded) │         │  Training Data               │
│  Input Data         │─────▶   │  (examples with answers)     │
│         │           │         │           │                  │
│         ▼           │         │           ▼                  │
│      Output         │         │       Algorithm              │
└─────────────────────┘         │           │                  │
                                │           ▼                  │
                                │     Learned Model            │
                                │     (the "rules")            │
                                │           │                  │
                                │    New Input Data ──▶ Output │
                                └──────────────────────────────┘
```

### The 3 Types of ML

```
Machine Learning
├── Supervised Learning     → We provide (X, y) pairs; model learns f(X) ≈ y
│   ├── Regression          → y is continuous (price, temperature, load)
│   └── Classification      → y is a category (spam/not-spam, fault/no-fault)
│
├── Unsupervised Learning   → We provide only X; model finds structure
│   ├── Clustering          → Group similar items (customer segments)
│   ├── Dimensionality Red. → Compress features (PCA, autoencoders)
│   └── Density Estimation  → Model data distribution
│
└── Reinforcement Learning  → Agent takes actions, receives rewards
    └── Applications: robotics, game AI, control systems
```

### The ML Workflow

```
1. Define Problem  →  2. Collect Data  →  3. Explore (EDA)
                                               │
6. Deploy          ←  5. Evaluate      ←  4. Preprocess & Train
      ↑_________________________↓  (iterate)
```

---

## Key Vocabulary

| Term | Plain English | Engineering Analogy |
|------|---------------|---------------------|
| **Feature** (X) | An input measurement used to make a prediction | A sensor reading (voltage, temperature) |
| **Target** (y) | The output we want to predict | The label on a measurement (pass/fail) |
| **Training set** | Data used to fit the model | Calibration data |
| **Test set** | Data held back to evaluate real-world performance | Validation measurement on new samples |
| **Overfitting** | Model memorizes training data but fails on new data | Curve-fit through noise, fails to extrapolate |
| **Underfitting** | Model is too simple to capture the pattern | Fitting a line to clearly nonlinear data |
| **Bias** | Systematic error from wrong model assumptions | Sensor offset |
| **Variance** | Error from sensitivity to small data fluctuations | Sensor noise amplification |
| **Bias-Variance Tradeoff** | Simple models have high bias; complex models have high variance | Filter bandwidth vs noise rejection |
| **Hyperparameter** | A setting you choose before training (not learned) | Bandwidth of a filter |
| **Parameter** | A value the model learns from data | Coefficients in a regression |

---

## Connection to Engineering Mathematics

### Why Linear Algebra?
- Your dataset is a **matrix** X of shape (n_samples, n_features)
- ML models compute `y = Xw + b` — matrix-vector multiplication
- PCA and SVD compress high-dimensional data
- Covariance matrices describe feature relationships

### Why Calculus?
- Training a model = minimizing a **loss function** L(w)
- We use **gradient descent**: `w ← w - α·∇L(w)`
- The gradient ∇L tells us which direction increases loss — we go the opposite way
- Chain rule enables backpropagation in neural networks

### Why Probability?
- Real data has noise — we model uncertainty
- Classification outputs are **probabilities**, not hard decisions
- Regularization has a Bayesian interpretation (prior beliefs on weights)
- Cross-entropy loss comes directly from maximum likelihood estimation

---

## File Guide

| File | What It Teaches | Run Time |
|------|-----------------|----------|
| `01_what_is_ml.py` | Traditional programming vs ML; first sklearn model | ~5s |
| `02_ml_workflow.py` | Complete 8-step ML workflow on student grade data | ~5s |
| `03_data_and_features.py` | Feature types, engineering, encoding, scaling | ~5s |
| `04_train_test_split.py` | Evaluation methodology, cross-validation, overfitting | ~15s |
| `exercises.py` | 4-tier practice problems | varies |

### Running Order

```bash
cd 01_ml_fundamentals/
python 01_what_is_ml.py        # Start here
python 02_ml_workflow.py
python 03_data_and_features.py
python 04_train_test_split.py  # Saves plots to output/
python exercises.py            # Attempt before looking at solutions
```

---

## Real-World Connections

Every concept in this module appears directly in industry:

- **Supervised regression** → predicting remaining useful life of turbine blades
- **Feature engineering** → converting raw vibration signals into frequency features
- **Train/test split** → ISO standards require validation on held-out data
- **Overfitting** → a model that only works on your test bench, not in production
- **Cross-validation** → averaging performance over multiple splits gives more reliable estimates

---

## Next Steps → Module 2: Scikit-Learn Deep Dive

After this module, you'll learn:
- The sklearn Estimator API in depth (`fit`, `predict`, `transform`, `score`)
- Building preprocessing Pipelines (avoid data leakage!)
- Systematic hyperparameter search with GridSearchCV
