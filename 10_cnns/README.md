# Module 10: Convolutional Neural Networks (CNNs)

## Overview

In Module 9 you built neural networks where every neuron connects to every other neuron — **fully-connected (dense) layers**. This works for tabular data, but it catastrophically fails for images. Why? A 256×256 RGB image has 196,608 pixel values. A single hidden layer of 1024 neurons would need **201 million parameters** just for the first layer — and it would have no idea that nearby pixels are related.

CNNs solve this by **sharing weights across space**. One small filter (say, 3×3) slides over the entire image, learning to detect the same feature (like a horizontal edge) everywhere it appears. This is **parameter efficiency** driven by the physics of images: the world looks the same whether an edge is in the top-left or bottom-right of a photo.

---

## Prerequisites

- Module 8: PyTorch fundamentals (tensors, autograd)
- Module 9: Neural networks (layers, training loops, loss functions)
- Linear algebra: matrix multiplication, dot products
- Calculus: chain rule (backprop through conv layers)

---

## Learning Objectives

By the end of this module, you will be able to:

1. Explain **why** CNNs outperform MLPs on image data (spatial locality, weight sharing)
2. Implement **convolution manually** with NumPy and understand what each parameter does
3. Describe **feature maps** and how stacking convolutional layers builds hierarchical representations
4. Use **pooling** for spatial invariance and dimensionality reduction
5. Build and train a **complete CNN** for image classification in PyTorch
6. Explain **transfer learning** and when/how to use pretrained models

---

## Why CNNs? The Problem with MLPs on Images

### The MLP failure mode

Suppose you train an MLP to recognize cats. It learns pixel #1247 (top-left ear area) should be bright. Now the cat moves 5 pixels to the right. The MLP fails — pixel #1247 is now background. It has no concept of **spatial relationships**.

```
MLP on images:
- Flattens 28x28 image → 784 numbers
- Loses ALL spatial structure
- Every pixel treated independently
- Fails if object shifts position
- Huge parameter count = overfitting
```

### The CNN solution: spatial locality + weight sharing

```
CNN approach:
- Keeps 2D structure: image stays a grid
- Small filters slide over the image
- Same filter detects the same feature ANYWHERE
- Parameters shared across all positions
- Enormously fewer parameters
```

**Real-world analogy:** A border inspector doesn't memorize every possible passport layout. They learn a few key patterns ("look for a photo, look for text, look for a stamp") and apply those patterns regardless of exact position. That's what a CNN filter does.

---

## Key Concepts

### 1. Convolution and Filters

A **filter** (also called a **kernel**) is a small matrix of learnable weights — typically 3×3 or 5×5. The convolution operation slides this filter over the input, computing the **dot product** between the filter and each local patch:

```
Input patch:    Filter (Sobel horizontal):    Output:
1  2  3         -1  -2  -1                    (1×-1 + 2×-2 + 3×-1 +
4  5  6    ×     0   0   0      =              4×0  + 5×0  + 6×0  +
7  8  9          1   2   1                    7×1  + 8×2  + 9×1)
                                            = -1-4-3 + 0 + 7+16+9 = 24
```

**What the filter learns:**
- Sobel-like filters → edges
- Blob-like filters → corners, textures
- Later layers → eyes, wheels, faces, disease patterns

### 2. Key Parameters

| Parameter | What it does | Typical values |
|-----------|-------------|----------------|
| `kernel_size` | Size of the filter (k×k) | 3, 5, 7 |
| `stride` | How many pixels to move at each step | 1 (default), 2 |
| `padding` | Zeros added around border to control output size | 0, 1, 'same' |
| `in_channels` | Number of input feature maps | 1 (grayscale), 3 (RGB), 64, 128... |
| `out_channels` | Number of filters = number of output feature maps | 16, 32, 64, 128... |

**Output size formula:**
```
output_size = floor((input_size + 2×padding - kernel_size) / stride) + 1
```

### 3. Feature Maps

Each filter produces one **feature map** — a 2D grid showing where in the image that filter's pattern was detected. With 32 filters, you get 32 feature maps.

```
Input image (1×28×28)
      ↓  32 filters of 3×3
Feature maps (32×26×26)   ← 32 different "views" of the image
      ↓  64 filters of 3×3
Feature maps (64×24×24)   ← 64 richer features
```

### 4. Pooling: Spatial Invariance

**MaxPooling** takes the maximum value in each local region:
```
4  3  1  2          4  3
2  1  0  1   →      2  3      (2×2 max pool, stride=2)
3  2  3  1
1  0  2  1
```

**Why it works:**
- If an edge detector fires at position (5,7) or (6,7), it probably means the same edge — pooling collapses these nearby detections
- Reduces spatial size → fewer parameters in subsequent layers
- Provides **translation invariance** (small shifts don't matter)

### 5. Receptive Field

The **receptive field** is how much of the original image one neuron in a deep layer "sees." Stacking conv+pool layers exponentially grows the receptive field:

```
Layer 1 (3×3 conv):  sees 3×3 = 9 pixels
Layer 2 (3×3 conv):  sees 5×5 = 25 pixels
After pool:          sees 10×10 = 100 pixels
Layer 3 (3×3 conv):  sees 12×12 = 144 pixels
```

Deep CNNs have neurons that integrate information from the entire image.

### 6. The Feature Hierarchy

CNNs learn a hierarchy of features through depth:

```
Raw pixels
    ↓  (Conv Layer 1)
Edges and gradients: /, \, |, —
    ↓  (Conv Layer 2)
Textures and shapes: corners, curves, blobs
    ↓  (Conv Layer 3)
Parts: eyes, ears, wheels, nuclei
    ↓  (Conv Layer 4+)
Objects: cats, cars, tumors, crops
```

This mirrors how the human visual cortex is organized (V1 → V2 → V4 → IT cortex).

---

## Standard CNN Architecture Pattern

```
Input Image
    │
    ▼
┌─────────────────────┐
│  Conv2d (k=3, p=1)  │  ← Learn local features
│  BatchNorm2d        │  ← Stabilize training
│  ReLU               │  ← Non-linearity
│  MaxPool2d (2×2)    │  ← Reduce spatial size
└─────────────────────┘ × N blocks (deeper = richer features)
    │
    ▼
AdaptiveAvgPool2d or Flatten
    │
    ▼
Linear → ReLU → Dropout → Linear → Softmax
```

---

## Real-World Applications

| Domain | Task | CNN Role |
|--------|------|----------|
| **Medical imaging** | Detecting tumors in CT scans | Feature maps learn tissue texture patterns |
| **Satellite imagery** | Land use classification | Spectral + spatial features combined |
| **Manufacturing** | Defect detection on assembly lines | Anomaly detection in texture |
| **Self-driving** | Lane detection, object recognition | Real-time spatial reasoning |
| **Agriculture** | Crop disease identification | Phone photos → instant diagnosis |
| **Astronomy** | Galaxy classification | Morphological feature extraction |

---

## Concept Map

```
Images as tensors (C × H × W)
         │
    Convolution (filter slides over input)
         │
    Feature maps (pattern detectors)
         │
    Pooling (spatial invariance, size reduction)
         │
    Stacked layers → feature hierarchy
         │
    Flatten → Fully Connected → Classification
         │
    Transfer learning: reuse pretrained features
```

---

## File Guide

| File | Concept | Key Takeaway |
|------|---------|-------------|
| `01_convolution.py` | Conv operation, filters, feature maps | Filters are learnable pattern detectors |
| `02_pooling_and_architecture.py` | Pooling, BatchNorm, CNN design | Standard (Conv→BN→ReLU→Pool) pattern |
| `03_cnn_for_images.py` | Full CNN training pipeline | End-to-end image classification |
| `04_transfer_learning.py` | Pretrained models, fine-tuning | Don't train from scratch if you can borrow |
| `exercises.py` | Practice problems | 4-level difficulty progression |

---

## Running Instructions

```bash
cd 10_cnns/

# Run in order:
python 01_convolution.py          # ~10 seconds
python 02_pooling_and_architecture.py  # ~5 seconds
python 03_cnn_for_images.py       # ~60 seconds (training)
python 04_transfer_learning.py    # ~10 seconds

# Each script saves plots to output/
ls output/
```

---

## Real-World Connection

When you take a chest X-ray in a modern hospital, a CNN (often a ResNet or DenseNet variant) may screen it for pneumonia, fractures, or cardiomegaly. The same mathematical operations you'll implement in `01_convolution.py` — sliding a 3×3 filter over pixel values — underpin that system. The only difference is depth (hundreds of layers) and scale (millions of training examples).

In climate science, CNNs process satellite imagery at planetary scale to track deforestation, monitor ice caps, and predict extreme weather events. Understanding convolution is understanding a large part of modern automated perception.

---

## Next Steps

- **Module 11**: Transformers — when sequential attention matters more than spatial locality
- **Projects**: Build a real image classifier on your own dataset using pretrained ResNet
- **Further reading**: "Deep Learning" (Goodfellow et al.), Chapter 9; CS231n lecture notes (Stanford)
