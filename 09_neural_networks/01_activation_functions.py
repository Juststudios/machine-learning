"""
Neural Networks: Activation Functions
=======================================
Why do we need activation functions?

A neural network without activations is just matrix multiplication:
  layer1 = X @ W1 + b1
  layer2 = layer1 @ W2 + b2
  output = layer2 @ W3 + b3

But matrix multiplication of matrices = one matrix multiplication.
No matter how many "layers" you stack, without activations
the whole network collapses to a single linear transformation.

Activation functions introduce NON-LINEARITY, allowing the network
to learn curved decision boundaries and complex patterns.

Universal Approximation Theorem: a network with even ONE hidden layer
and a nonlinear activation can approximate ANY continuous function.
(Given enough neurons.)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

torch.manual_seed(42)

# ===========================================================================
# PART 1: Why Activations? Demonstration
# ===========================================================================
print("=" * 60)
print("PART 1: Why Activation Functions?")
print("=" * 60)

print("""
Proof: Linear network = linear model, no matter how deep.

Linear network (2 layers, no activation):
  h = X @ W1 + b1
  y = h @ W2 + b2
  y = X @ (W1 @ W2) + (b1 @ W2 + b2)  ← still linear!
  
So adding more linear layers gives you NOTHING over a single linear layer.
The activation function is what makes depth meaningful.
""")

# Demonstrate: deep linear network = single linear layer
torch.manual_seed(0)
X = torch.randn(10, 3)

# Deep "linear" network (no activations) — 4 layers
W1, W2, W3, W4 = [torch.randn(3, 3) for _ in range(4)]
b1, b2, b3, b4 = [torch.randn(3) for _ in range(4)]

# 4-layer computation
h1 = X @ W1 + b1
h2 = h1 @ W2 + b2
h3 = h2 @ W3 + b3
out_deep = h3 @ W4 + b4

# Equivalent single-layer computation (collapse all matrix multiplications)
W_combined = W1 @ W2 @ W3 @ W4
b_combined = b1 @ W2 @ W3 @ W4 + b2 @ W3 @ W4 + b3 @ W4 + b4
out_single = X @ W_combined + b_combined

diff = (out_deep - out_single).abs().max().item()
print(f"Difference between 4-layer linear and 1-layer: {diff:.8f}")
print(f"  (near zero = they're mathematically equivalent!)")

# ===========================================================================
# PART 2: Common Activation Functions
# ===========================================================================
print("\n--- PART 2: Common Activation Functions ---")

x = torch.linspace(-5, 5, 200)

activations = {
    'Sigmoid':      (torch.sigmoid(x),      "σ(x) = 1/(1+e^-x) | Output: (0,1)"),
    'Tanh':         (torch.tanh(x),         "tanh(x) | Output: (-1, 1), zero-centered"),
    'ReLU':         (torch.relu(x),         "max(0, x) | Fast, sparse, solves vanishing gradient"),
    'Leaky ReLU':   (torch.where(x>0, x, 0.01*x), "max(0.01x, x) | Fixes 'dying ReLU' problem"),
    'GELU':         (x * torch.sigmoid(1.702 * x), "Smooth ReLU variant, used in Transformers"),
    'Softplus':     (torch.log(1 + torch.exp(x)),  "Smooth approximation of ReLU"),
}

print("Activation functions and their properties:")
for name, (_, desc) in activations.items():
    print(f"  {name:15s}: {desc}")

# PyTorch implementations
print("\nPyTorch activation functions:")
x_demo = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"  Input:    {x_demo.tolist()}")
print(f"  ReLU:     {torch.relu(x_demo).tolist()}")
print(f"  Sigmoid:  {torch.sigmoid(x_demo).round(decimals=4).tolist()}")
print(f"  Tanh:     {torch.tanh(x_demo).round(decimals=4).tolist()}")
print(f"  Softmax:  {torch.softmax(x_demo, dim=0).round(decimals=4).tolist()}")

# ===========================================================================
# PART 3: Vanishing Gradient — Why Sigmoid/Tanh Struggle
# ===========================================================================
print("\n--- PART 3: The Vanishing Gradient Problem ---")

print("""
Problem with Sigmoid/Tanh in deep networks:
  - Sigmoid saturates at 0 and 1 (gradient ≈ 0 there)
  - During backprop, gradient = product of many activation gradients
  - Each layer multiplies by a value between 0 and 1
  - After 10 layers: 0.25^10 = 0.000001 → gradient vanishes!

The earlier layers receive essentially ZERO gradient signal.
They cannot learn. The network fails.

ReLU solves this:
  - Gradient = 1 for positive inputs (doesn't shrink the gradient)
  - Gradient = 0 for negative inputs ("dead" neurons — trade-off)
""")

# Demonstrate vanishing gradients
def sigmoid_grad(x):
    s = torch.sigmoid(x)
    return s * (1 - s)

def relu_grad(x):
    return (x > 0).float()

x_test = torch.tensor(0.5)  # moderate activation
print(f"At x = {x_test.item()}:")
print(f"  Sigmoid gradient: {sigmoid_grad(x_test).item():.6f}")
print(f"  ReLU gradient:    {relu_grad(x_test).item():.6f}")

# After 10 layers
sig_10 = sigmoid_grad(x_test).item() ** 10
relu_10 = relu_grad(x_test).item() ** 10
print(f"\nAfter 10 layers of chain rule multiplication:")
print(f"  Sigmoid: {sig_10:.8f}  ← nearly zero!")
print(f"  ReLU:    {relu_10:.6f}  ← still 1.0 (no vanishing)")

# ===========================================================================
# PART 4: Softmax for Multi-Class Output
# ===========================================================================
print("\n--- PART 4: Softmax — Converting Logits to Probabilities ---")

print("""
Softmax converts raw network outputs (logits) to probabilities.
  softmax(x_i) = exp(x_i) / sum(exp(x_j))

Properties:
  - All outputs are positive (exp is always positive)
  - All outputs sum to 1 (valid probability distribution)
  - Preserves ranking (larger logit → larger probability)
""")

logits = torch.tensor([2.0, 1.0, 0.5, -1.0])
probs = torch.softmax(logits, dim=0)

print(f"Logits: {logits.tolist()}")
print(f"Probabilities: {probs.round(decimals=4).tolist()}")
print(f"Sum: {probs.sum().item():.6f}  (always sums to 1)")
print(f"Argmax: class {probs.argmax().item()} (highest probability)")

# ===========================================================================
# PART 5: Visualization
# ===========================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Activation Functions and Their Gradients", fontsize=14, fontweight='bold')

x_np = x.numpy()

activation_configs = [
    ('Sigmoid', torch.sigmoid(x), torch.sigmoid(x) * (1 - torch.sigmoid(x))),
    ('Tanh',    torch.tanh(x),    1 - torch.tanh(x)**2),
    ('ReLU',    torch.relu(x),    (x > 0).float()),
    ('Leaky ReLU', torch.where(x>0, x, 0.01*x), torch.where(x>0, torch.ones_like(x), 0.01*torch.ones_like(x))),
    ('GELU',    x * torch.sigmoid(1.702 * x), None),  # gradient complex, skip
    ('Softplus',torch.log(1 + torch.exp(x)), torch.sigmoid(x)),
]

for ax, (name, values, grad) in zip(axes.flat, activation_configs):
    ax.plot(x_np, values.numpy(), 'b-', linewidth=2, label='f(x)')
    if grad is not None:
        ax.plot(x_np, grad.numpy(), 'r--', linewidth=1.5, alpha=0.8, label="f'(x)")
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel("x")
    ax.set_ylim(-1.5, 2.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "activation_functions.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nActivation functions plot saved: {OUTPUT_DIR}/activation_functions.png")

print("""
===========================================
WHEN TO USE WHICH ACTIVATION
===========================================
Hidden layers:
  ReLU        → Default choice. Fast, simple, works well.
  Leaky ReLU  → When you see "dying ReLU" (many dead neurons).
  GELU        → Modern transformers, BERT, GPT use this.
  Tanh        → RNNs/LSTMs, when you need (-1, 1) range.

Output layer:
  Sigmoid     → Binary classification (outputs probability 0-1).
  Softmax     → Multi-class classification (outputs class probabilities).
  Linear (none) → Regression (need unbounded output).

NEVER use Sigmoid/Tanh in deep hidden layers → vanishing gradients!

Next: 02_weight_initialization.py — Why initialization matters for training
""")
