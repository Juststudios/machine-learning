# Module 3: Regression — Predicting Continuous Values

## Overview

Regression is the most common supervised learning task in engineering and science:
predicting a **continuous number** from input features.

- What will CPU temperature be at 80% load and 3.2 GHz?
- How much power will this wind turbine produce given wind speed and direction?
- How many days until this bearing fails given current vibration signature?

This module builds regression from the ground up: the math, the intuition,
the code, and the engineering interpretation of every parameter.

---

## Prerequisites

- **Modules 1 & 2 completed**: ML workflow, sklearn API, Pipelines
- Linear Algebra: matrix-vector multiplication (`Xw`), vector norms
- Calculus: derivatives, the concept of minimizing a function
- You've seen `LinearRegression` before — now you'll understand *exactly* what it does

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Derive** the intuition behind the least-squares cost function and explain why MSE is minimized
2. **Implement** simple and multiple linear regression, and interpret the coefficients physically
3. **Apply** polynomial regression for nonlinear relationships and select the right degree
4. **Explain** why regularization is needed and what Ridge (L2) vs Lasso (L1) do geometrically
5. **Use** `RidgeCV` and `LassoCV` to automatically select regularization strength
6. **Read** coefficient path plots to understand how regularization performs feature selection
7. **Implement** gradient descent from scratch in numpy (Level 4 challenge)

---

## The Math: What Linear Regression Actually Does

### The Model

For a dataset with `n` samples and `p` features:

```
X  = (n × p) matrix   — your features (e.g., [clock_speed, load])
y  = (n,)    vector   — your targets  (e.g., CPU temperature)
w  = (p,)    vector   — weights (what the model LEARNS)
b  =  scalar          — bias / intercept (also learned)

Prediction: ŷ = Xw + b
```

### The Cost Function (What Gets Minimized)

```
MSE Loss = (1/n) · Σᵢ (yᵢ - ŷᵢ)²
         = (1/n) · ||y - Xw - b||²
```

We want to find `w` and `b` that make this as small as possible.

### Gradient Descent (How It's Minimized)

```
∂MSE/∂w = (-2/n) · Xᵀ(y - Xw)

Update rule:
  w ← w - α · ∂MSE/∂w
  b ← b - α · ∂MSE/∂b
```

where `α` (alpha) is the **learning rate** — a hyperparameter you choose.

sklearn's LinearRegression uses the **closed-form solution** (exact, no iteration):
```
w* = (XᵀX)⁻¹ Xᵀy
```
But gradient descent is important because it's used by every neural network.

---

## Why Regularization?

When you have many features (or polynomial features), the model can overfit:
it fits noise in the training data and fails on new data.

### Ridge (L2 regularization)

```
Ridge Loss = MSE + α · ||w||²₂ = MSE + α · Σ wᵢ²
```

Adds a penalty for large weights → all weights shrink toward zero, but none become exactly zero.
Use when: all features are potentially relevant, you want a stable model.

### Lasso (L1 regularization)

```
Lasso Loss = MSE + α · ||w||₁ = MSE + α · Σ |wᵢ|
```

The L1 penalty creates **sparse solutions** — some weights become exactly 0.
This means Lasso performs **automatic feature selection**.
Use when: you suspect many features are irrelevant.

### ElasticNet (L1 + L2)

```
ElasticNet Loss = MSE + α · (ρ · ||w||₁ + (1-ρ)/2 · ||w||²₂)
```

Best of both: feature selection + stability.

---

## Concept Map

```
Regression
├── Linear Regression
│   ├── Simple (1 feature)     → y = w·x + b
│   ├── Multiple (p features)  → y = Xw + b
│   └── Solved by: OLS (closed form) or Gradient Descent
│
├── Polynomial Regression
│   ├── PolynomialFeatures(degree=d) → creates x², x³, x₁x₂, ...
│   ├── Then fit LinearRegression on transformed features
│   ├── Risk: overfitting grows with degree
│   └── Selection: cross-validation to pick best degree
│
└── Regularized Regression
    ├── Ridge (L2)    → shrinks all weights, none exactly zero
    ├── Lasso (L1)    → drives some weights to exactly zero (feature selection)
    └── ElasticNet    → combines L1 + L2
```

---

## File Guide

| File | What It Teaches | Run Time |
|------|-----------------|----------|
| `01_linear_regression.py` | Math intuition, simple/multiple LR, coefficient interpretation | ~10s |
| `02_polynomial_regression.py` | Nonlinear fitting, PolynomialFeatures, degree selection | ~15s |
| `03_regularization.py` | Ridge, Lasso, ElasticNet; coefficient paths; CV for alpha | ~20s |
| `exercises.py` | 4-tier practice including gradient descent from scratch | varies |
| `solutions/regression_solutions.py` | Complete solutions (study after attempting!) | — |

### Running Order
```bash
cd 03_regression/
python 01_linear_regression.py    # Saves plots to output/
python 02_polynomial_regression.py
python 03_regularization.py
python exercises.py
```

---

## Real-World Engineering Examples

| Application | Features (X) | Target (y) | Algorithm |
|-------------|--------------|------------|-----------|
| CPU thermal model | clock_speed, load, core_count | temperature | Linear/Ridge |
| Wind power prediction | wind_speed, direction, density | power_output | Polynomial |
| Chemical yield | reagent_conc, temperature, pressure | product_yield | ElasticNet |
| Remaining useful life | vibration_rms, cycles, temp | hours_remaining | Regularized |
| Building energy | occupancy, outdoor_temp, hour | kWh_consumed | Polynomial |

---

## Gradient Descent Intuition (Preview)

```
Loss
  │
  │        ●  ← start (random weights)
  │       / \
  │      /   \
  │     /     \
  │    /       ● ← after 1 step
  │   /         \
  │  /           ● ← after 2 steps
  │ /             \
  │/               ★ ← minimum (optimal weights)
  └─────────────────────── weights (w)
```

Each step moves in the direction of steepest descent (negative gradient).
Step size = learning rate α.
Too large: oscillates, diverges. Too small: takes forever.

---

## Next Steps → Module 4: Classification

Regression predicts numbers. Classification predicts categories.
You'll use logistic regression, decision trees, SVMs, and ensemble methods.
