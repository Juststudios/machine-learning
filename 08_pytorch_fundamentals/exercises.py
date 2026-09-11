"""
Exercises: PyTorch Fundamentals (Module 8)
==========================================
4-tier exercises for nn.Module, loss functions, optimizers, and training loops.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("PYTORCH FUNDAMENTALS — EXERCISES")
print("=" * 60)

# ===========================================================================
# LEVEL 1: RECALL
# ===========================================================================
print("\n--- LEVEL 1: RECALL ---")
print("""
Answer by studying the demo code:

1a. What is the difference between nn.ReLU() and torch.relu(x)?
    When would you use each?

1b. In the training loop: model.train() then model.eval().
    Which layers behave differently in each mode?

1c. What is the correct order of these 5 operations?
    A. optimizer.step()
    B. y_pred = model(X)
    C. optimizer.zero_grad()
    D. loss = criterion(y_pred, y)
    E. loss.backward()

1d. BCEWithLogitsLoss vs CrossEntropyLoss: when do you use each?

1e. What is weight_decay in optim.Adam? What problem does it solve?
""")

# Demo
model_demo = nn.Sequential(
    nn.Linear(4, 16),
    nn.BatchNorm1d(16),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(16, 1),
)
print("Model in training mode:")
model_demo.train()
X_demo = torch.randn(8, 4)
out_train = model_demo(X_demo)
print(f"  Output shape: {out_train.shape}")

model_demo.eval()
with torch.no_grad():
    out_eval = model_demo(X_demo)
print(f"  Output shape (eval): {out_eval.shape}")
print(f"  Outputs differ (due to Dropout/BatchNorm): {not torch.allclose(out_train.detach(), out_eval)}")

# ===========================================================================
# LEVEL 2: DEBUGGING
# ===========================================================================
print("\n--- LEVEL 2: DEBUGGING ---")
print("""
The MLP below has 4 bugs. Find and fix them.
""")

buggy = '''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
X = torch.randn(200, 6)
y = (X[:, 0] + X[:, 1] > 0).float()  # binary classification

# BUG 1: Wrong loss function (CrossEntropyLoss needs class indices, not floats)
class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))  # BUG 2: sigmoid before CrossEntropyLoss = wrong!

model = BadModel()
criterion = nn.CrossEntropyLoss()  # BUG 1: should be BCEWithLogitsLoss for binary
optimizer = optim.Adam(model.parameters(), lr=0.001)
loader = DataLoader(TensorDataset(X, y.unsqueeze(1)), batch_size=32)

for epoch in range(5):
    for Xb, yb in loader:
        y_pred = model(Xb)
        loss = criterion(y_pred, yb)     # BUG 1 causes error here
        loss.backward()
        optimizer.step()                  # BUG 3: Missing optimizer.zero_grad()!

# BUG 4: Evaluating without model.eval() and torch.no_grad()
acc = (model(X).round() == y.unsqueeze(1)).float().mean()
print(f"Accuracy: {acc.item():.4f}")
'''
print(buggy)
print("""
Bug 1: ________________________________________________
Bug 2: ________________________________________________
Bug 3: ________________________________________________
Bug 4: ________________________________________________

TODO: Write the corrected version
""")

# ===========================================================================
# LEVEL 3: APPLICATION — Multi-class MLP From Scratch
# ===========================================================================
print("\n--- LEVEL 3: APPLICATION ---")
print("""
TASK: Build and train an MLP for multi-class classification on sensor data.

Dataset (generate it):
  - 4 sensor features (temperature, vibration, rpm, current)
  - 3 classes: OK=0, WARNING=1, FAULT=2
  - 600 training samples, 150 test samples

Requirements:
  1. Build an MLP: Linear(4→64) → ReLU → Dropout(0.3) → Linear(64→32) → ReLU → Linear(32→3)
  2. Use CrossEntropyLoss (raw logits — no softmax in forward!)
  3. Use Adam with lr=0.001
  4. Training loop: 100 epochs, batch_size=32
  5. After each epoch, compute val accuracy
  6. Use early stopping: stop if val accuracy doesn't improve for 10 epochs
  7. Restore best model and evaluate on test set
  8. Plot training history (train loss + val accuracy) — save to output/
  9. Compare with sklearn LogisticRegression on same data
""")

# Starter data
n_per = 200
X_ok    = np.random.randn(n_per, 4) + np.array([70, 1.0, 3000, 15])
X_warn  = np.random.randn(n_per, 4) + np.array([82, 2.5, 2800, 19])
X_fault = np.random.randn(n_per, 4) + np.array([95, 5.5, 2500, 27])
X_app   = np.vstack([X_ok, X_warn, X_fault]).astype(np.float32)
y_app   = np.array([0]*n_per + [1]*n_per + [2]*n_per)

X_atr, X_ate, y_atr, y_ate = train_test_split(X_app, y_app, test_size=0.2, random_state=42)

sc = StandardScaler()
X_atr_s = sc.fit_transform(X_atr)
X_ate_s  = sc.transform(X_ate)

print(f"Data: X_train={X_atr_s.shape}, X_test={X_ate_s.shape}")
print("TODO: Implement the MLP training loop above")

# ===========================================================================
# LEVEL 4: CHALLENGE — Learning Rate Finder
# ===========================================================================
print("\n--- LEVEL 4: CHALLENGE ---")
print("""
TASK: Implement a Learning Rate Finder (LR Range Test).

The LR Range Test (Smith, 2017) helps find the optimal learning rate:
  1. Start with a very small learning rate (lr_min = 1e-7)
  2. Gradually increase lr (exponentially) over many mini-batches
  3. Record loss at each step
  4. Plot lr (log scale) vs loss
  5. The optimal lr is just before the loss starts to explode

Algorithm:
  def lr_finder(model, loader, criterion, lr_min=1e-7, lr_max=10, num_steps=100):
      optimizer = optim.Adam(model.parameters(), lr=lr_min)
      
      # Create a learning rate scheduler that multiplies lr each step
      # lr after step i: lr_min * (lr_max/lr_min)^(i/num_steps)
      
      lrs, losses = [], []
      
      for i, (X_batch, y_batch) in enumerate(cycle(loader)):  # cycle over loader
          if i >= num_steps: break
          
          # Forward
          loss = criterion(model(X_batch), y_batch)
          lrs.append(optimizer.param_groups[0]['lr'])
          losses.append(loss.item())
          
          # Backward
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          # Increase lr (exponentially)
          new_lr = lr_min * (lr_max/lr_min) ** ((i+1)/num_steps)
          optimizer.param_groups[0]['lr'] = new_lr
          
          # Stop if loss explodes
          if loss.item() > 10 * min(losses):
              break
      
      # Plot lr vs loss, find the lr just before loss starts rising
      return lrs, losses

Requirements:
  - Implement lr_finder()
  - Run it on the sensor fault dataset from Level 3
  - Plot LR vs Loss (log scale for LR axis)
  - Identify the "best" LR (lowest point on the curve, or just before explosion)
  - Save plot to output/lr_finder.png
  - Train the final model using that LR and report test accuracy
""")
from itertools import cycle

def lr_finder(model, loader, criterion, lr_min=1e-7, lr_max=10, num_steps=100):
    """TODO: Implement the LR Range Test."""
    lrs, losses = [], []
    # YOUR IMPLEMENTATION HERE
    return lrs, losses

print("TODO: Implement lr_finder() and use it to find the optimal learning rate")

print("\n" + "=" * 60)
print("EXERCISES COMPLETE")
print("Check solutions/pytorch_fundamentals_solutions.py for reference answers.")
print("=" * 60)
