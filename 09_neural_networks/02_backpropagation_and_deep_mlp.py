"""
Backpropagation and Deep MLP Architecture
==========================================
Now that we understand activations and weight init,
let's trace exactly HOW a neural network learns.

Backpropagation = "backward propagation of errors"
It efficiently computes the gradient of the loss with respect to
EVERY weight in the network by applying the chain rule layer by layer,
starting from the output and working backwards.

Why understanding backprop matters:
  - Explains WHY certain activations cause vanishing gradients
  - Explains WHY certain initializations are critical
  - Explains WHY skip connections (ResNets) work so well
  - Helps debug training problems (NaN loss, no learning, etc.)
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
import copy

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

torch.manual_seed(42)
np.random.seed(42)

# ===========================================================================
# PART 1: Backpropagation — Tracing Through a Simple Network
# ===========================================================================
print("=" * 60)
print("PART 1: Tracing Backpropagation")
print("=" * 60)

print("""
Consider a 1-layer network: y = ReLU(xW + b)
  Loss = MSE(y, y_true) = (y - y_true)²

Forward pass:
  z = xW + b         (linear combination)
  y = ReLU(z)        (activation)
  L = (y - y_true)²  (loss)

Backward pass (chain rule):
  dL/dy = 2(y - y_true)
  dy/dz = 1 if z > 0 else 0   (ReLU derivative)
  dz/dW = x
  dz/db = 1

  dL/dW = dL/dy · dy/dz · dz/dW = 2(y-y_t) · ReLU'(z) · x
  dL/db = dL/dy · dy/dz · dz/db = 2(y-y_t) · ReLU'(z)

For deep networks: same chain rule, just applied layer by layer
  dL/dW_layer_k = dL/d(output) · chain of all derivatives in between
""")

# Trace gradients manually vs PyTorch
x = torch.tensor([[1.0, 2.0, -1.0]])  # 1 sample, 3 features
y_true = torch.tensor([[1.5]])          # 1 target

# Manual setup
W = torch.tensor([[0.5], [-0.3], [0.8]], requires_grad=True)  # (3, 1)
b = torch.tensor([0.1], requires_grad=True)

# Forward
z = x @ W + b
y_pred = torch.relu(z)
loss = ((y_pred - y_true) ** 2).mean()

print(f"Forward pass:")
print(f"  x = {x.tolist()}")
print(f"  z = xW + b = {z.item():.4f}")
print(f"  y = ReLU(z) = {y_pred.item():.4f}")
print(f"  L = (y - y_true)² = {loss.item():.4f}")

# Backward
loss.backward()
print(f"\nBackward pass (PyTorch):")
print(f"  dL/dW = {W.grad.T.tolist()}")
print(f"  dL/db = {b.grad.tolist()}")

# Manual computation
dL_dy = 2 * (y_pred - y_true)                    # dL/dy
dy_dz = (z > 0).float()                          # ReLU derivative
dL_dW_manual = (dL_dy * dy_dz).T @ x             # chain rule
dL_db_manual = (dL_dy * dy_dz).sum()

print(f"\nManual computation:")
print(f"  dL/dW = {dL_dW_manual.detach().tolist()}")
print(f"  dL/db = {dL_db_manual.item():.6f}")
print(f"  Match: {torch.allclose(W.grad.T, dL_dW_manual)}")

# ===========================================================================
# PART 2: Deep MLP Architecture — Layer-by-Layer
# ===========================================================================
print("\n--- PART 2: Deep MLP Design Patterns ---")

print("""
Key design decisions for a deep MLP:
  
  1. NUMBER OF LAYERS (depth):
     - 1-2 hidden layers: most tabular data problems
     - 3-5 hidden layers: complex tabular, time series
     - Many layers: only if you have lots of data and use skip connections
  
  2. LAYER WIDTH (neurons per layer):
     - Common: 512 → 256 → 128 → 64 (halve each layer)
     - Or: constant width (all layers same size)
     - Bottleneck: narrow middle (128 → 32 → 128) for feature compression
  
  3. ACTIVATION FUNCTION (hidden layers):
     - ReLU: default choice, fast, simple
     - GELU: smoother, used in modern architectures
     - No sigmoid/tanh in hidden layers! (vanishing gradients)
  
  4. NORMALIZATION:
     - BatchNorm1d after Linear: stabilizes training, allows higher LR
     - LayerNorm: when batch size is small or variable-length sequences
  
  5. REGULARIZATION:
     - Dropout: randomly zero neurons (disable during eval!)
     - Weight decay (L2): in optimizer (weight_decay=1e-4)
""")

# Three architectures: shallow, standard, deep+residual
class ShallowMLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x): return self.net(x)

class StandardMLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    def forward(self, x): return self.net(x)

class ResidualBlock(nn.Module):
    """Skip connection: output = f(x) + x (if dimensions match)."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x + self.block(x))  # skip connection!

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, 128)
        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.head = nn.Linear(128, n_classes)
    
    def forward(self, x):
        x = torch.relu(self.input_proj(x))
        x = self.res_blocks(x)
        return self.head(x)

# Count parameters
for name, model in [('ShallowMLP', ShallowMLP(20, 3)),
                     ('StandardMLP', StandardMLP(20, 3)),
                     ('ResidualMLP', ResidualMLP(20, 3))]:
    params = sum(p.numel() for p in model.parameters())
    print(f"  {name:15s}: {params:,} parameters")

# ===========================================================================
# PART 3: Train and Compare Architectures
# ===========================================================================
print("\n--- PART 3: Architecture Comparison ---")

# Generate data
from sklearn.datasets import make_classification
X_np, y_np = make_classification(n_samples=2000, n_features=20, n_informative=15,
                                   n_classes=3, n_clusters_per_class=2, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_tr, X_te, y_tr, y_te = train_test_split(X_np, y_np, test_size=0.2, stratify=y_np, random_state=42)
sc = StandardScaler()
X_tr_s = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32)
X_te_s  = torch.tensor(sc.transform(X_te), dtype=torch.float32)
y_tr_t  = torch.tensor(y_tr, dtype=torch.long)
y_te_t  = torch.tensor(y_te, dtype=torch.long)

loader = DataLoader(TensorDataset(X_tr_s, y_tr_t), batch_size=64, shuffle=True)

def train_and_eval(model, loader, X_te, y_te, epochs=80):
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_state = 0, None
    
    for ep in range(epochs):
        model.train()
        for Xb, yb in loader:
            loss = criterion(model(Xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        
        model.eval()
        with torch.no_grad():
            preds = model(X_te).argmax(dim=1)
            acc = (preds == y_te).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_state)
    return best_acc

architectures = [
    ('ShallowMLP',   ShallowMLP(20, 3)),
    ('StandardMLP',  StandardMLP(20, 3)),
    ('ResidualMLP',  ResidualMLP(20, 3)),
]

print("Training and comparing architectures...")
results = {}
for name, model in architectures:
    acc = train_and_eval(model, loader, X_te_s, y_te_t, epochs=80)
    results[name] = acc
    print(f"  {name:15s}: Test Accuracy = {acc:.4f}")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. Backprop: chain rule applied from loss backward through all layers
2. PyTorch autograd handles backprop automatically — just call .backward()
3. Gradient flows backwards; activation gradients multiply at each layer
   → Why vanishing gradients kill deep networks with sigmoid/tanh
4. Residual connections (x + f(x)) allow gradient to "skip" problematic layers
5. BatchNorm stabilizes gradients and allows larger learning rates
6. Dropout: use for regularization, disable during eval (model.eval())
7. 2-3 hidden layers sufficient for most tabular problems

Next: exercises.py — Build deep networks and explore backprop
""")
