"""
Exercises: Neural Networks (Module 9)
========================================
4-tier exercises covering activation functions, backpropagation,
deep MLP architecture, and neural network training from scratch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("NEURAL NETWORKS — EXERCISES")
print("=" * 60)

# ===========================================================================
# LEVEL 1: RECALL
# ===========================================================================
print("\n--- LEVEL 1: RECALL ---")
print("""
Study the demo below and answer:

1a. Which activation function should you use in hidden layers by default?
    Why not Sigmoid or Tanh?

1b. What does a "dying ReLU" problem look like? How do you fix it?

1c. What is the vanishing gradient problem?
    Name two ways PyTorch architectures solve it.

1d. In BatchNorm, what do gamma and beta learn? What is their initialization?

1e. If train loss = 0.02 and val loss = 0.95, what is the problem?
    Name TWO solutions specific to neural networks.
""")

# Demo: activation function gradients
x = torch.linspace(-3, 3, 100)
acts = {
    'Sigmoid': (torch.sigmoid(x), torch.sigmoid(x) * (1 - torch.sigmoid(x))),
    'Tanh':    (torch.tanh(x),    1 - torch.tanh(x)**2),
    'ReLU':    (torch.relu(x),    (x > 0).float()),
}

print("Activation gradients at x=2:")
for name, (val, grad) in acts.items():
    print(f"  {name}: f(2)={val[66].item():.4f}, f'(2)={grad[66].item():.4f}")

print("\nGradient after 10 layers (product of 10 activation gradients at x=2):")
for name, (val, grad) in acts.items():
    prod = grad[66].item() ** 10
    print(f"  {name}: {prod:.8f}  {'← vanishes!' if prod < 0.001 else ''}")

# ===========================================================================
# LEVEL 2: DEBUGGING
# ===========================================================================
print("\n--- LEVEL 2: DEBUGGING ---")
print("""
This neural network refuses to learn. Find and fix 4 bugs.
""")

buggy = '''
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)
X = torch.randn(300, 10)
y = (X[:, 0] + X[:, 2] - X[:, 5] > 0).long()

# BUG 1: Xavier init is intended for tanh, but using with sigmoid layers below
class BuggyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        # Missing initialization — PyTorch default (Kaiming) is actually fine for ReLU
        # but the activations below use sigmoid which has vanishing gradient:
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # BUG 2: sigmoid in hidden layer → vanishing gradient
        x = torch.sigmoid(self.fc2(x))  # BUG 2: same issue
        return self.fc3(x)

model = BuggyMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=100.0)  # BUG 3: LR WAY too large (10 is way too big)

for epoch in range(50):
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    # BUG 4: Missing optimizer.zero_grad() → gradient accumulation

print(f"Final loss: {loss.item():.4f}")
print(f"Accuracy: {(logits.argmax(1) == y).float().mean():.4f}")
'''
print(buggy)
print("""
Bug 1/2: ________________________________________________ (activation issue)
Bug 3:   ________________________________________________ (optimizer issue)
Bug 4:   ________________________________________________ (training loop issue)

TODO: Write the corrected version
""")

# ===========================================================================
# LEVEL 3: APPLICATION — Build a Fault Severity Predictor
# ===========================================================================
print("\n--- LEVEL 3: APPLICATION ---")
print("""
TASK: Design and train a deep MLP for 3-class fault detection.

Dataset (use the sensor data below):
  8 sensors → 3 fault severity classes (OK, WARNING, FAULT)
  1200 training, 300 test samples

Architecture requirements:
  - At least 3 hidden layers
  - BatchNorm1d after each linear layer
  - ReLU activations
  - Dropout(0.3) for regularization
  - Correct output: 3 logits (CrossEntropyLoss, no softmax in forward)

Training requirements:
  - Adam optimizer (lr=0.001, weight_decay=1e-4)
  - 150 epochs
  - ReduceLROnPlateau scheduler (patience=10)
  - Early stopping (patience=20)
  - Save best model state

Evaluation:
  - Final test accuracy
  - Compare with a single hidden layer MLP (same data)
  - Show how depth helps (both should beat a simple baseline)

Save training history plot to output/nn_exercise_training.png
""")

# Data
n_per = 500
X0 = np.random.randn(n_per, 8) + np.array([70, 1.0, 3000, 15, 5, 60, 220, 20])
X1 = np.random.randn(n_per, 8) + np.array([82, 2.5, 2800, 19, 5.8, 62, 218, 35])
X2 = np.random.randn(n_per, 8) + np.array([95, 5.5, 2500, 27, 7, 65, 213, 60])
X_all = np.vstack([X0, X1, X2]).astype(np.float32)
y_all = np.array([0]*n_per + [1]*n_per + [2]*n_per)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
sc = StandardScaler()
X_tr_s = torch.tensor(sc.fit_transform(X_tr))
X_te_s  = torch.tensor(sc.transform(X_te))
y_tr_t  = torch.tensor(y_tr, dtype=torch.long)
y_te_t  = torch.tensor(y_te, dtype=torch.long)

print(f"Data: {X_tr_s.shape} train, {X_te_s.shape} test")
print("TODO: Build and train your deep MLP")

# ===========================================================================
# LEVEL 4: CHALLENGE — Neural Network From Scratch (No PyTorch nn.Module)
# ===========================================================================
print("\n--- LEVEL 4: CHALLENGE ---")
print("""
TASK: Implement a 2-hidden-layer neural network from scratch using ONLY
      PyTorch tensors and autograd (no nn.Module, no nn.Linear).

class TwoLayerNet:
    '''
    Network: x → Linear(D, H1) → ReLU → Linear(H1, H2) → ReLU → Linear(H2, C)
    All weights and biases are torch.tensor with requires_grad=True
    '''
    def __init__(self, input_dim, h1, h2, output_dim):
        # Initialize weights using Xavier uniform initialization
        # W1: (input_dim, h1), b1: (h1,)
        # W2: (h1, h2), b2: (h2,)
        # W3: (h2, output_dim), b3: (output_dim,)
        pass
    
    def forward(self, x):
        # Compute: z1 = relu(x @ W1 + b1)
        #          z2 = relu(z1 @ W2 + b2)
        #          out = z2 @ W3 + b3
        pass
    
    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
    
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

# Training loop (manual optimizer: SGD with momentum)
# loss = CrossEntropyLoss (implement it: -log(softmax(logits)[y]))

Requirements:
  - Implement TwoLayerNet with only tensor operations
  - Implement cross-entropy loss manually: 
    loss = -mean(log(softmax(logits))[range(n), y])
  - Implement SGD update: param -= lr * param.grad
  - Train on the sensor data from Level 3
  - Target: test accuracy ≥ 85%
  - Compare with nn.Module equivalent (should get same accuracy)
""")

class TwoLayerNet:
    """TODO: Implement this from scratch."""
    
    def __init__(self, input_dim, h1, h2, output_dim, lr=0.01):
        self.lr = lr
        # TODO: Initialize weights (Xavier init recommended)
        # self.W1 = ...
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass
    
    def parameters(self):
        pass
    
    def zero_grad(self):
        for p in self.parameters():
            if p is not None and p.grad is not None:
                p.grad.zero_()

def cross_entropy_loss_manual(logits, y):
    """TODO: Implement cross-entropy loss manually."""
    # HINT:
    # softmax = exp(logits) / sum(exp(logits), axis=1, keepdim=True)
    # log_softmax = log(softmax)
    # loss = -mean(log_softmax[range(n), y])
    pass

print("TODO: Implement TwoLayerNet and cross_entropy_loss_manual")
print("  Target test accuracy on sensor data: ≥ 85%")

print("\n" + "=" * 60)
print("EXERCISES COMPLETE")
print("Check solutions/neural_networks_solutions.py for reference answers.")
print("=" * 60)
