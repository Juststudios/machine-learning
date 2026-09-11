"""
PyTorch Tensors: The Foundation of Deep Learning
==================================================
Everything in deep learning — data, model weights, gradients, activations —
is a tensor.

What is a tensor?
  - Scalar: 0D tensor (a single number)
  - Vector: 1D tensor [1, 2, 3]
  - Matrix: 2D tensor [[1, 2], [3, 4]]
  - 3D+:   Batches of images, sequences, etc.

WHY tensors over numpy arrays?
  - Automatic differentiation (autograd) — compute gradients automatically
  - GPU acceleration — move tensors to GPU for massive parallelism
  - Deep learning ecosystem — all PyTorch layers work with tensors natively

This lesson: tensors on CPU (torch 2.13.0+cpu)
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 60)
print("PyTorch Tensor Fundamentals")
print(f"PyTorch version: {torch.__version__}")
print("=" * 60)

# ===========================================================================
# PART 1: Creating Tensors
# ===========================================================================
print("\n--- PART 1: Creating Tensors ---")

# From Python lists
t1 = torch.tensor([1.0, 2.0, 3.0])
print(f"From list:         {t1}")
print(f"  shape: {t1.shape}, dtype: {t1.dtype}, device: {t1.device}")

# 2D tensor (matrix)
t2 = torch.tensor([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]])
print(f"\nMatrix:\n{t2}")
print(f"  shape: {t2.shape}")  # torch.Size([2, 3])

# Factory functions (like numpy equivalents)
zeros    = torch.zeros(3, 4)       # all zeros
ones     = torch.ones(2, 3)        # all ones
rand     = torch.rand(3, 3)        # uniform [0, 1)
randn    = torch.randn(3, 3)       # standard normal N(0, 1)
eye      = torch.eye(4)            # identity matrix
arange   = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5) # [0.0, 0.25, 0.5, 0.75, 1.0]

print(f"\ntorch.zeros(3,4):\n{zeros}")
print(f"\ntorch.randn(3,3):\n{randn.round(decimals=3)}")
print(f"\ntorch.arange(0,10,2): {arange}")
print(f"\ntorch.linspace(0,1,5): {linspace}")

# Specifying dtype
int_t   = torch.tensor([1, 2, 3], dtype=torch.int64)
float32 = torch.tensor([1.0, 2.0], dtype=torch.float32)  # most common in ML
float64 = torch.tensor([1.0, 2.0], dtype=torch.float64)

print(f"\ndtype examples:")
print(f"  int64:   {int_t.dtype}    → {int_t}")
print(f"  float32: {float32.dtype}  → {float32}  (default for neural nets)")
print(f"  float64: {float64.dtype}  → {float64}")

# ===========================================================================
# PART 2: Tensor Operations
# ===========================================================================
print("\n--- PART 2: Tensor Operations ---")

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

print("Element-wise operations:")
print(f"  a + b:\n{a + b}")
print(f"  a * b:\n{a * b}")      # element-wise (NOT matrix multiply)
print(f"  a / b:\n{(a / b).round(decimals=3)}")
print(f"  a ** 2:\n{a ** 2}")

print("\nMatrix operations:")
print(f"  a @ b (matrix multiply):\n{a @ b}")    # equivalent to torch.matmul(a, b)
print(f"  a.T (transpose):\n{a.T}")

print("\nReduction operations:")
x = torch.randn(4, 3)
print(f"  x:\n{x.round(decimals=3)}")
print(f"  x.sum():            {x.sum():.4f}   (all elements)")
print(f"  x.sum(dim=0):       {x.sum(dim=0).round(decimals=3)}  (sum down rows → 3 values)")
print(f"  x.sum(dim=1):       {x.sum(dim=1).round(decimals=3)}  (sum across cols → 4 values)")
print(f"  x.mean():           {x.mean():.4f}")
print(f"  x.max():            {x.max():.4f}")
print(f"  x.argmax():         {x.argmax()}   (flat index of max)")

print("\nShape manipulation:")
t = torch.arange(12, dtype=torch.float32)
print(f"  Original: {t.shape} → {t}")
print(f"  view(3,4):\n{t.view(3, 4)}")
print(f"  view(2,6):\n{t.view(2, 6)}")
print(f"  view(1,-1): shape {t.view(1, -1).shape}  (-1 infers the dimension)")
print(f"  unsqueeze(0): {t.unsqueeze(0).shape}  (add dimension at position 0)")
print(f"  squeeze():    removes size-1 dimensions")

# ===========================================================================
# PART 3: The Numpy Bridge
# ===========================================================================
print("\n--- PART 3: NumPy ↔ PyTorch Bridge ---")

print("""
WHY the bridge matters:
  - Data loading often uses numpy (pandas, scikit-learn, OpenCV)
  - PyTorch models need tensors
  - You'll constantly convert between the two
""")

# numpy → tensor
arr = np.array([1.0, 2.0, 3.0])
t_from_np = torch.from_numpy(arr)
print(f"numpy → torch:")
print(f"  arr:         {arr}  (numpy, dtype={arr.dtype})")
print(f"  tensor:      {t_from_np}  (torch, dtype={t_from_np.dtype})")

# IMPORTANT: from_numpy shares memory with the original array!
arr[0] = 99.0
print(f"\n  After arr[0]=99: tensor = {t_from_np}  ← shared memory!")
arr[0] = 1.0  # reset

# To avoid shared memory, use .clone()
t_copy = torch.from_numpy(arr).clone()

# tensor → numpy
tensor = torch.tensor([4.0, 5.0, 6.0])
np_from_t = tensor.numpy()
print(f"\ntorch → numpy:")
print(f"  tensor:      {tensor}")
print(f"  numpy array: {np_from_t}  (dtype={np_from_t.dtype})")

# If tensor has gradients, must detach first
t_with_grad = torch.tensor([1.0, 2.0], requires_grad=True)
np_from_grad = t_with_grad.detach().numpy()
print(f"\n  Tensor with gradients → must detach() first: {np_from_grad}")

# torch.tensor() vs torch.from_numpy() — difference
print("""
Key difference:
  torch.from_numpy(arr)  → shares memory, no copy (fast, but dangerous)
  torch.tensor(arr)      → copies data (safe, slightly slower)
""")

# ===========================================================================
# PART 4: Autograd — The Key Difference from NumPy
# ===========================================================================
print("--- PART 4: Autograd — Why Tensors Enable Deep Learning ---")

print("""
The KEY feature tensors have that numpy arrays don't:
  AUTOMATIC DIFFERENTIATION (autograd)

Deep learning trains by computing gradients of the loss with respect to
every weight in the network. For millions of parameters, doing this by hand
is impossible.

PyTorch builds a computation graph as you do operations.
When you call .backward(), it computes ALL gradients automatically.

requires_grad=True: "track all operations on this tensor for differentiation"
""")

# Simple example: y = 3x² + 2x + 1
# True derivative: dy/dx = 6x + 2
# At x = 2: dy/dx = 14

x = torch.tensor(2.0, requires_grad=True)

# Forward pass (computation graph is built here)
y = 3 * x**2 + 2 * x + 1

print(f"x = {x.data}")
print(f"y = 3x² + 2x + 1 = {y.item():.1f}")

# Backward pass: compute dy/dx
y.backward()

print(f"\nComputed gradient dy/dx at x=2:")
print(f"  x.grad = {x.grad}  (expected: 6*2 + 2 = 14) ✓")

# Using with neural network parameters
print("""
In a neural network:
  x → represents the WEIGHTS (w)
  y → represents the LOSS
  y.backward() computes ∂loss/∂w for every weight simultaneously

This is what optimizer.step() uses to update weights!
""")

# Disable gradient tracking for inference
x2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
with torch.no_grad():
    y2 = x2 * 2 + 1
print(f"Inside torch.no_grad(), result has grad_fn: {y2.grad_fn}")  # None

# ===========================================================================
# PART 5: Indexing, Slicing, Boolean Masking
# ===========================================================================
print("\n--- PART 5: Indexing and Slicing ---")

t = torch.arange(16, dtype=torch.float32).view(4, 4)
print(f"Matrix:\n{t}")
print(f"\nIndexing:")
print(f"  t[1, 2]:    {t[1, 2].item()}")  # element at row 1, col 2
print(f"  t[0]:       {t[0]}")             # first row
print(f"  t[:, 2]:    {t[:, 2]}")          # all rows, col 2
print(f"  t[1:3, 0:2]:\n{t[1:3, 0:2]}")   # submatrix

# Boolean masking
print(f"\nBoolean masking:")
mask = t > 8
print(f"  t > 8:\n{mask}")
print(f"  t[t > 8]: {t[mask]}")

# ===========================================================================
# PART 6: Tensor vs NumPy — Quick Summary Plot
# ===========================================================================
x_plot = torch.linspace(-3, 3, 100)
y_relu    = torch.relu(x_plot)
y_sigmoid = torch.sigmoid(x_plot)
y_tanh    = torch.tanh(x_plot)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_plot.numpy(), y_relu.numpy(),    label='ReLU', linewidth=2)
ax.plot(x_plot.numpy(), y_sigmoid.numpy(), label='Sigmoid', linewidth=2)
ax.plot(x_plot.numpy(), y_tanh.numpy(),    label='Tanh', linewidth=2)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output")
ax.set_title("Common Activation Functions (computed with PyTorch tensors)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "activation_functions_tensor.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nActivation function plot saved: {OUTPUT_DIR}/activation_functions_tensor.png")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. Tensors are N-dimensional arrays — same as numpy but with superpowers
2. torch.tensor(), torch.zeros(), torch.randn() — create tensors
3. Operations: +, *, @, .T, .sum(), .mean(), .view() — same as numpy style
4. Bridge: torch.from_numpy() and .numpy() for conversion (CPU tensors only)
5. requires_grad=True → PyTorch tracks operations for automatic differentiation
6. .backward() → computes all gradients simultaneously (the engine of training)
7. torch.no_grad() → disable tracking for inference (faster, less memory)

Next: 02_autograd.py — Deep dive into automatic differentiation
""")
