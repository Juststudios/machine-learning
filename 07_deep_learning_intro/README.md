# Module 7: Introduction to Deep Learning with PyTorch

## Overview

Classical ML (linear models, trees, SVMs) is powerful but has a fundamental limitation: **it requires humans to engineer features**. For images, text, audio, and sequences, the right features are not obvious — and often not even describable.

Deep learning solves this by **learning the features automatically**, layer by layer, directly from raw data.

This module bridges classical ML to deep learning. You will build neural networks from scratch, understand every mathematical operation, and learn PyTorch — the most research-friendly deep learning framework in the world.

---

## When Classical ML Isn't Enough

| Problem | Classical ML Approach | Why It Struggles |
|---------|----------------------|-----------------|
| Image classification | Manually extract edges, textures | Millions of possible features; doesn't scale |
| Sentiment analysis | Bag-of-words, TF-IDF | Loses word order, context, meaning |
| Speech recognition | Hand-crafted acoustic features | Enormous variation in speakers and environments |
| Protein structure | Expert-designed biochemical features | The search space is astronomical |
| Game playing | Hand-coded heuristics | Cannot enumerate all possible states |

**Deep learning** learns hierarchical representations: raw pixels → edges → shapes → objects.

---

## What Does "Deep" Mean?

"Deep" refers to **multiple layers of learned transformations**, each building on the previous:

```
Input (raw data)
     ↓
Layer 1: Detects simple patterns (edges in images, common word pairs in text)
     ↓
Layer 2: Combines simple patterns into complex ones (shapes, phrases)
     ↓
Layer 3: Combines complex patterns into concepts (faces, opinions)
     ↓
Output (prediction)
```

**Depth = abstraction.** The deeper the network, the more abstract (and powerful) the representations it can learn.

---

## The Progression of Architectures

```
Perceptron (1958)
  └── Single neuron: weighted sum + threshold
      └── MLP — Multi-Layer Perceptron (1986)
            └── Multiple layers, backpropagation enables learning
                └── CNN — Convolutional Neural Network (1989, popularized ~2012)
                      └── Spatially aware: great for images, audio
                          └── RNN — Recurrent Neural Network (1986–2015)
                                └── Memory across time steps: sequences, text
                                    └── Transformer (2017–present)
                                          └── Attention over entire sequence at once
                                              └── The architecture behind GPT, BERT, etc.
```

This module covers **MLP**, the foundation that everything else builds on.

---

## Why PyTorch?

PyTorch was chosen for this curriculum because:

| Feature | PyTorch | Why It Matters |
|---------|---------|---------------|
| **Dynamic graphs** | Yes (define-by-run) | Debug like normal Python — print tensors mid-computation |
| **Pythonic** | Deeply | The code reads like the math |
| **Research adoption** | Dominant | Most ML papers use PyTorch |
| **Production** | TorchScript, ONNX | Deployable when you're ready |
| **Community** | Huge | Stack Overflow, tutorials, pre-trained models everywhere |

---

## What is a Tensor?

A **tensor** is a generalization of arrays to arbitrary dimensions:

```
Scalar (0D tensor):  42
Vector (1D tensor):  [1, 2, 3, 4]
Matrix (2D tensor):  [[1, 2], [3, 4]]
3D tensor:           RGB image (height x width x channels)
4D tensor:           Batch of RGB images (batch x height x width x channels)
```

PyTorch tensors are like NumPy arrays, but with two superpowers:
1. **They can live on a GPU** (for massively parallel computation)
2. **They track operations** for automatic differentiation (autograd)

---

## The PyTorch Ecosystem

```
torch                 <- Core tensor operations
torch.nn              <- Neural network layers (Linear, Conv2d, LSTM, ...)
torch.optim           <- Optimizers (SGD, Adam, RMSprop, ...)
torch.utils.data      <- Dataset and DataLoader for efficient data pipelines
torchvision           <- Pre-built datasets and models for computer vision
torchaudio            <- Audio processing
torchtext             <- NLP utilities
torch.hub             <- Download pre-trained models in one line
```

---

## File Guide

| File | What You Learn | Lines |
|------|---------------|-------|
| `01_tensors.py` | Tensors: creation, ops, numpy bridge, autograd intro | ~180 |
| `02_autograd.py` | Automatic differentiation from scratch | ~170 |
| `03_linear_model_in_pytorch.py` | Full training loop: forward→loss→backward→update | ~200 |
| `04_datasets_and_dataloaders.py` | Efficient data pipelines with Dataset & DataLoader | ~160 |
| `exercises.py` | 4-tier exercises | — |

---

## Prerequisites Check

Before diving in, confirm you understand:

- [ ] NumPy array operations (indexing, broadcasting, shape)
- [ ] What a derivative is (conceptually — slope of a function)
- [ ] What linear regression does (fit a line to data)
- [ ] The concept of a loss function (measure how wrong you are)

---

## Running Instructions

```bash
cd 07_deep_learning_intro/

# Check PyTorch is installed
python -c "import torch; print(torch.__version__)"

# Run lessons in order
python 01_tensors.py
python 02_autograd.py
python 03_linear_model_in_pytorch.py   # saves plots to output/
python 04_datasets_and_dataloaders.py

# Exercises
python exercises.py
```

---

## Real-World Connections

- **01_tensors.py**: Images, audio, video are all tensors. A 4K video = 4D tensor with time dimension
- **02_autograd.py**: Every neural network — GPT, ResNet, AlphaFold — uses exactly this mechanism
- **03_linear_model_in_pytorch.py**: This training loop is identical to training a 100-billion parameter model — just scaled up
- **04_datasets_and_dataloaders.py**: Efficient data loading is often the bottleneck in training

---

## Next Steps

After completing this module:
- **Module 8**: Multi-Layer Perceptrons — add hidden layers, add nonlinearities, get deep
- **Module 9**: CNNs for images — spatial feature learning
- **Module 10**: Training techniques — batch normalization, dropout, learning rate scheduling

---

*"Deep learning is just matrix multiplication, nonlinearities, and gradient descent — repeated many times."*
