# Module 8: PyTorch Fundamentals

## Overview

This module introduces PyTorch — the dominant framework for research and production deep learning. You will build a solid mental model of how PyTorch works from the ground up: tensors, automatic differentiation, model building blocks, loss functions, optimizers, and the complete training loop.

By the end of this module you will be able to write a complete neural network training pipeline from scratch.

---

## Prerequisites

- Python: functions, classes, loops, NumPy
- Linear algebra: matrix multiplication, dot products
- Calculus: chain rule, partial derivatives (used by autograd)
- Probability: cross-entropy, likelihood
- Completed Modules 1–7 (sklearn fundamentals)

---

## Learning Objectives

After completing this module you will be able to:

1. Explain the PyTorch computational graph and why it enables automatic differentiation
2. Build custom models by subclassing `nn.Module`
3. Choose the correct loss function for regression and classification tasks
4. Understand what each optimizer does and when to use Adam vs SGD
5. Write a complete, professional-grade training loop with validation and early stopping
6. Train an MLP classifier and compare it to sklearn baselines

---

## Concept Map

```
Raw Data (Tensors)
       │
       ▼
   nn.Module         ◄── your model (layers defined in __init__, computation in forward())
       │
       ▼
  Forward Pass       ── model(X) → predictions
       │
       ▼
  Loss Function      ── criterion(predictions, targets) → scalar loss
       │
       ▼
  Backward Pass      ── loss.backward() → computes ∂loss/∂weights for all parameters
       │
       ▼
   Optimizer         ── optimizer.step() → updates weights using gradients
       │
       ▼
  Repeat (epochs)
```

---

## PyTorch vs. Scikit-learn

| Feature | scikit-learn | PyTorch |
|---|---|---|
| Core abstraction | Estimator (fit/predict) | nn.Module (forward) |
| Optimization | Built-in (often analytical) | Explicit gradient descent |
| Computation graph | None | Dynamic (define-by-run) |
| GPU support | No | Yes (trivial `.to(device)`) |
| Custom architectures | Very limited | Unlimited |
| Best for | Classical ML, baselines | Deep learning, research |

**Key difference:** sklearn optimizes internally. In PyTorch, *you* write the optimization loop. This gives full control but requires understanding gradient-based learning.

---

## The PyTorch Training Recipe (Always 5 Steps)

Every PyTorch training loop reduces to these 5 steps per batch:

```python
for batch_X, batch_y in train_loader:
    optimizer.zero_grad()                    # 1. Clear accumulated gradients
    predictions = model(batch_X)             # 2. Forward pass
    loss = criterion(predictions, batch_y)   # 3. Compute loss
    loss.backward()                          # 4. Backward pass (autograd)
    optimizer.step()                         # 5. Update weights
```

Memorize this. Every variation is just this with extra bookkeeping.

---

## PyTorch Architecture

```
┌─────────────────────────────────────────────────┐
│                   PyTorch Core                   │
│                                                  │
│  Tensor  ──►  Autograd  ──►  nn.Module           │
│  (data)      (gradients)    (model layers)       │
│                                                  │
│  optim  ──►  DataLoader  ──►  Training Loop      │
│  (update)   (batching)       (your code)         │
└─────────────────────────────────────────────────┘
```

### Key Components

- **`torch.Tensor`**: N-dimensional array with gradient tracking. Like NumPy but autodiff-capable.
- **`torch.autograd`**: Automatic differentiation via chain rule.
- **`nn.Module`**: Base class for all models. Compose layers, track parameters, switch train/eval modes.
- **`nn.functional`**: Stateless functions (relu, softmax). Used inside `forward()`.
- **`torch.optim`**: Optimizers (SGD, Adam) that update parameters using gradients.
- **`DataLoader`**: Handles batching, shuffling, and parallel data loading.

---

## When to Use PyTorch vs. Sklearn

**Use sklearn when:**
- Tabular data with < 100k rows
- Quick baseline needed
- Interpretability is required
- Standard classical algorithms (tree, SVM, linear models)

**Use PyTorch when:**
- Images, text, audio, time series
- Data > 100k rows (neural nets scale better)
- Custom architecture needed
- Transfer learning / research

---

## File Guide

| File | What You Learn |
|---|---|
| `01_nn_module.py` | Building models: `nn.Module`, layers, parameters |
| `02_loss_functions.py` | Choosing the right loss for regression/classification |
| `03_optimizers.py` | SGD, Adam, learning rate schedulers |
| `04_training_loop.py` | Complete training loop with validation and checkpointing |
| `05_mlp_classification.py` | Full MLP classifier with sklearn comparison |
| `exercises.py` | 4-tier exercises to test your understanding |
| `exercises_solutions.py` | Solutions (attempt first!) |

---

## Running the Files

```bash
cd 08_pytorch_fundamentals/

python 01_nn_module.py
python 02_loss_functions.py
python 03_optimizers.py          # saves output/optimizer_comparison.png
python 04_training_loop.py       # saves output/training_curves.png
python 05_mlp_classification.py  # saves output/decision_boundary.png

python exercises.py
```

---

## Real-World Connections

- **Self-driving cars**: PyTorch models detect objects in camera frames (YOLO, DETR)
- **Drug discovery**: Graph neural networks predict molecule properties
- **Language models**: GPT, BERT — all PyTorch `nn.Module` subclasses
- **Medical imaging**: MRI scan segmentation, cancer detection
- **Finance**: Fraud detection, price forecasting

---

## Next Steps

- **Module 9**: Deep dive into activation functions, batch normalization, dropout
- **Module 10**: Convolutional Neural Networks for image data
- **Module 11**: Recurrent Networks and Transformers for sequences
