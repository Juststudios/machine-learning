# Module 9: Neural Networks — Deep Dive

## Overview

Module 8 taught you the PyTorch training recipe. This module goes *inside* the recipe: why do specific architectural choices matter? You will understand activation functions, weight initialization, batch normalization, and dropout — the four pillars that make deep networks trainable and generalizable.

By the end you will be able to diagnose and fix common training pathologies, and build a production-ready deep MLP for a real engineering problem.

---

## Prerequisites

- Completed Module 8 (PyTorch Fundamentals) — especially `nn.Module`, training loop
- Calculus: chain rule (for vanishing gradient discussion)
- Linear algebra: matrix multiplication

---

## Learning Objectives

After completing this module you will be able to:

1. Explain why nonlinear activations are necessary and choose the right one
2. Diagnose vanishing/exploding gradient problems
3. Understand and apply Xavier and He weight initialization
4. Use batch normalization correctly in train vs eval mode
5. Use dropout to prevent overfitting
6. Build a complete production deep MLP with all four components

---

## How Neurons Work

A single neuron computes:

```
output = activation( w₁x₁ + w₂x₂ + ... + wₙxₙ + b )
       = activation( wᵀx + b )
```

- **`wᵀx + b`** — weighted sum (linear transformation)
- **`activation(·)`** — nonlinear function deciding whether/how strongly the neuron "fires"

Without the activation, stacking neurons is pointless:
`W₂(W₁x + b₁) + b₂ = Wx + b` — still just linear!

---

## Why Depth Matters

Each layer learns a different *level of abstraction*:

```
Raw Features → Low-level Patterns → Mid-level Combinations → High-level Concepts → Decision
  (pixels)      (edges, corners)     (shapes, textures)       (faces, objects)    (cat/dog)
```

The Universal Approximation Theorem proves a sufficiently wide single hidden layer can approximate *any* continuous function — but **depth** is much more parameter-efficient in practice.

---

## The Four Pillars of Trainable Deep Networks

```
Problem                    Solution
─────────────────────────────────────────────────────────────
Vanishing gradients   →    ReLU / GELU activations
Symmetry breaking     →    Xavier / He initialization
Internal covariate    →    Batch Normalization
  shift
Overfitting           →    Dropout
```

---

## Concept Map

```
Input
  │
  ▼
[Linear Layer] ──► Weight Init (Xavier/He) determines starting point
  │
  ▼
[Batch Norm]  ──► Normalizes activations → stable training
  │
  ▼
[Activation]  ──► Nonlinearity → network can learn complex functions
  │
  ▼
[Dropout]     ──► Randomly zero neurons → forces redundancy → generalization
  │
  ▼
[Next Layer]
  │
  ▼
[Loss] → gradients flow BACKWARD through all these layers
```

---

## The Path to CNNs and Transformers

| Component | MLP | CNN | Transformer |
|---|---|---|---|
| Activation | ReLU/GELU | ReLU/GELU | GELU |
| Init | He | He | Xavier |
| Normalization | BatchNorm1d | BatchNorm2d | LayerNorm |
| Dropout | Dropout | Dropout | Dropout |

The principles don't change — only the layer types do.

---

## File Guide

| File | What You Learn |
|---|---|
| `01_activation_functions.py` | Sigmoid, Tanh, ReLU, GELU, vanishing gradients |
| `02_weight_initialization.py` | Xavier, He init; bad vs good init experiment |
| `03_batch_normalization.py` | Internal covariate shift, train vs eval modes |
| `04_dropout.py` | Overfitting prevention, ensemble interpretation |
| `05_deep_mlp_project.py` | Full engineering fault detection MLP project |
| `exercises.py` | 4-tier exercises |
| `exercises_solutions.py` | Solutions (attempt first!) |

---

## Running the Files

```bash
cd 09_neural_networks/

python 01_activation_functions.py  # saves output/activations.png
python 02_weight_initialization.py # saves output/init_comparison.png
python 03_batch_normalization.py   # saves output/batchnorm_effect.png
python 04_dropout.py               # saves output/dropout_effect.png
python 05_deep_mlp_project.py      # saves output/confusion_matrix.png + training_curves.png

python exercises.py
```

---

## Real-World Connections

- **Batch normalization** was essential for training ResNet-152 (surpassed human ImageNet accuracy)
- **Dropout** was invented at Toronto and remains standard in language models
- **ReLU** replaced sigmoid ~2010 and caused a step-change in what was trainable
- **He initialization** is default in every modern deep learning framework
- The fault prediction project mirrors real predictive maintenance systems in manufacturing

---

## Next Steps

- **Module 10**: Convolutional Neural Networks — same building blocks applied to 2D spatial data
- **Module 11**: Sequences and Transformers — attention, BERT/GPT architecture
