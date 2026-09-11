"""
Autograd: Automatic Differentiation
=====================================
How does a neural network learn? It computes gradients.
How does it compute gradients for millions of parameters? Autograd.

This lesson explains:
- What a computational graph is
- How backward() traces the graph to compute all gradients
- The chain rule (backpropagation) — implemented automatically
- Gradient descent from scratch using only autograd

WHY this matters:
  Without autograd, implementing backpropagation for even a 3-layer
  network requires dozens of complex matrix derivative formulas.
  Autograd makes it trivial — define your forward pass, call backward().
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

torch.manual_seed(42)

# ===========================================================================
# PART 1: The Computational Graph
# ===========================================================================
print("=" * 60)
print("PART 1: The Computational Graph")
print("=" * 60)

print("""
When you compute with tensors that have requires_grad=True,
PyTorch secretly builds a COMPUTATION GRAPH.

Example:
  x = 3.0
  a = x * 2       (node: multiply by 2)
  b = a + 1       (node: add 1)
  c = b ** 2      (node: square)

  Graph: x → [*2] → a → [+1] → b → [**2] → c

When you call c.backward():
  PyTorch traverses the graph BACKWARDS, applying the chain rule
  at each node to compute dc/dx.
""")

x = torch.tensor(3.0, requires_grad=True)

a = x * 2
b = a + 1
c = b ** 2

print(f"x = {x.item()}")
print(f"a = x * 2 = {a.item()}")
print(f"b = a + 1 = {b.item()}")
print(f"c = b ** 2 = {c.item()}")
print(f"\nc.grad_fn: {c.grad_fn}  ← the graph node for 'square' operation")
print(f"b.grad_fn: {b.grad_fn}  ← the graph node for 'add' operation")
print(f"a.grad_fn: {a.grad_fn}  ← the graph node for 'multiply' operation")
print(f"x.grad_fn: {x.grad_fn}  ← x is a leaf (no upstream operation)")

# Compute gradient
c.backward()

print(f"\nAfter backward():")
print(f"  x.grad = {x.grad}")

# Manual verification using chain rule:
# dc/db = 2b = 2 * 7 = 14
# db/da = 1
# da/dx = 2
# dc/dx = dc/db * db/da * da/dx = 14 * 1 * 2 = 28
print(f"\nManual chain rule: dc/dx = 2b * 1 * 2 = 2*{b.item()} * 2 = {2*b.item()*2}")
print("  ↑ Matches!" if x.grad.item() == 28 else "  ↑ Error!")

# ===========================================================================
# PART 2: Gradient Accumulation — zero_grad() is Critical
# ===========================================================================
print("\n--- PART 2: Gradient Accumulation Warning ---")

print("""
IMPORTANT: PyTorch ACCUMULATES gradients by default.
Every call to .backward() ADDS to existing gradients.

In neural network training loops, you MUST call optimizer.zero_grad()
before each backward pass. Otherwise gradients from previous batches
contaminate the current update.
""")

x = torch.tensor(2.0, requires_grad=True)

# First backward
(x ** 2).backward()
print(f"After 1st backward: x.grad = {x.grad}")  # 2*2 = 4

# Second backward WITHOUT zeroing
(x ** 2).backward()
print(f"After 2nd backward (no zero_grad): x.grad = {x.grad}")  # 4 + 4 = 8!  WRONG!

# Correct: zero before each backward
x.grad.zero_()
(x ** 2).backward()
print(f"After zero_() + backward: x.grad = {x.grad}")  # 4 (correct)

print("\n⚠️  Always call optimizer.zero_grad() before loss.backward() in your training loop!")

# ===========================================================================
# PART 3: Gradient Descent From Scratch
# ===========================================================================
print("\n--- PART 3: Gradient Descent From Scratch ---")

print("""
TASK: Find the minimum of f(x) = x² + 4x + 4 = (x + 2)²
      True minimum: x* = -2

Gradient descent algorithm:
  1. Start at some initial x
  2. Compute gradient df/dx
  3. Step in the NEGATIVE gradient direction (downhill)
  4. Repeat until convergence

  x_new = x_old - lr * (df/dx)
  
  With autograd, step 2 is just: f(x).backward()
""")

# Initialize x at some point
x = torch.tensor(8.0, requires_grad=True)
lr = 0.1

history = {'x': [x.item()], 'f': [(x**2 + 4*x + 4).item()]}

print(f"Starting point: x = {x.item():.4f}")

for step in range(30):
    # Forward: compute loss
    f = x**2 + 4*x + 4
    
    # Backward: compute gradient
    f.backward()
    
    # Update x (manually since we're not using an optimizer)
    with torch.no_grad():  # disable gradient tracking during the update step
        x -= lr * x.grad
    
    # IMPORTANT: zero the gradient after using it
    x.grad.zero_()
    
    history['x'].append(x.item())
    history['f'].append(f.item())
    
    if step % 5 == 0:
        print(f"  Step {step:2d}: x = {x.item():+.4f}, f(x) = {f.item():.4f}")

print(f"\nFinal: x = {x.item():.6f}  (true minimum: x* = -2.0)")

# ===========================================================================
# PART 4: Linear Regression With Autograd (From Scratch)
# ===========================================================================
print("\n--- PART 4: Linear Regression With Autograd ---")

print("""
Now let's use autograd to fit y = wx + b to data.
This IS what sklearn's LinearRegression does internally
(though sklearn uses the analytical solution, not gradient descent).

WHY show this? Because this EXACT pattern scales to:
  - Linear regression (2 parameters)
  - Neural network (millions of parameters)
  - The code looks the same — only the model changes!
""")

# True relationship: y = 2x + 1
torch.manual_seed(42)
n = 50
X_data = torch.linspace(-3, 3, n)
y_data = 2 * X_data + 1 + torch.randn(n) * 0.5

# Parameters to learn (randomly initialized)
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f"Initial w = {w.item():.4f}, b = {b.item():.4f}  (random)")

lr = 0.01
loss_history = []

for epoch in range(200):
    # 1. Forward pass
    y_pred = w * X_data + b
    
    # 2. Loss (MSE)
    loss = ((y_pred - y_data) ** 2).mean()
    
    # 3. Backward pass
    loss.backward()
    
    # 4. Update parameters
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    
    # 5. Zero gradients
    w.grad.zero_()
    b.grad.zero_()
    
    loss_history.append(loss.item())
    
    if epoch % 40 == 0:
        print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print(f"\nFinal: w = {w.item():.4f}, b = {b.item():.4f}")
print(f"True:  w = 2.0000, b = 1.0000")

# Plot training
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Loss curve
axes[0].plot(loss_history, color='steelblue')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Training Loss (Gradient Descent)")
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# Fit
axes[1].scatter(X_data.numpy(), y_data.numpy(), alpha=0.5, label='Data', color='coral')
x_line = torch.linspace(-3, 3, 100)
y_line = (w.detach() * x_line + b.detach()).numpy()
axes[1].plot(x_line.numpy(), y_line, 'b-', linewidth=2,
             label=f'Learned: y = {w.item():.2f}x + {b.item():.2f}')
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("Linear Regression via Gradient Descent")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "autograd_linear_regression.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: {OUTPUT_DIR}/autograd_linear_regression.png")

# ===========================================================================
# PART 5: torch.no_grad() — Inference Mode
# ===========================================================================
print("\n--- PART 5: torch.no_grad() ---")

print("""
During INFERENCE (prediction time), you don't need gradients.
Building the computation graph uses memory and time.

torch.no_grad() context manager disables graph building:
  - Faster predictions
  - Less memory usage
  - REQUIRED when you don't want autograd side effects (e.g., during eval)
""")

x_test = torch.tensor([1.0, 2.0, 3.0])

# With gradient tracking (during training)
y_with_grad = w * x_test + b
print(f"With grad tracking:    y_with_grad.requires_grad = {y_with_grad.requires_grad}")

# Without gradient tracking (during inference)
with torch.no_grad():
    y_no_grad = w * x_test + b
print(f"With torch.no_grad():  y_no_grad.requires_grad   = {y_no_grad.requires_grad}")

# Also: tensor.detach() removes a tensor from the graph
y_detached = y_with_grad.detach()
print(f"After .detach():       y_detached.requires_grad  = {y_detached.requires_grad}")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. PyTorch builds a computation graph as you perform operations
2. .backward() applies chain rule backwards to compute all gradients
3. Gradients accumulate — ALWAYS call zero_grad() before backward()
4. Gradient descent: x = x - lr * x.grad   (the learning step)
5. torch.no_grad(): disable graph for inference (faster, less memory)
6. This SAME pattern (forward → loss → backward → update) scales to
   networks with millions of parameters — the code looks identical!

Next: 03_linear_model_in_pytorch.py — Build regression with nn.Module
""")
