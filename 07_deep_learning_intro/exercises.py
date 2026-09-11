"""
Exercises: Deep Learning Intro (Modules 7-9)
=============================================
4-tier exercises for PyTorch tensors, autograd, training loops,
and neural network design.

IMPORTANT: Attempt each exercise yourself before checking
solutions/pytorch_solutions.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("DEEP LEARNING INTRO — EXERCISES")
print("=" * 60)

# ===========================================================================
# LEVEL 1: RECALL — Tensor and autograd basics
# ===========================================================================
print("\n--- LEVEL 1: RECALL ---")

print("""
Run and study the code below. Then answer the questions in comments.

1a. What is the shape, dtype, and device of a torch.randn(3, 4) tensor?
1b. What is the difference between torch.from_numpy() and torch.tensor()?
1c. What does requires_grad=True do?
1d. What happens if you call .backward() without calling .zero_grad() first?
1e. What is the difference between model.train() and model.eval()?
""")

# Demo to study
x = torch.randn(3, 4)
print(f"x shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")

a = torch.tensor(2.0, requires_grad=True)
b = a ** 3
b.backward()
print(f"a = {a.item()}, b = a³ = {b.item()}, db/da = {a.grad.item()}")
print(f"  (Expected: db/da = 3a² = {3 * a.item()**2})")

model = nn.Linear(4, 2)
model.train()
print(f"\nmodel.training = {model.training}")
model.eval()
print(f"model.training after eval() = {model.training}")

# ===========================================================================
# LEVEL 2: UNDERSTANDING — Debug the broken training loop
# ===========================================================================
print("\n--- LEVEL 2: DEBUGGING ---")
print("""
The following PyTorch training loop has 4 bugs. Find and fix them.
""")

buggy_training_loop = '''
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)
X = torch.randn(100, 4)
y = (X[:, 0] > 0).long()   # binary classification

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    # BUG 1: Missing model.train()
    
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # BUG 2: Missing optimizer.zero_grad() before backward
    loss.backward()
    optimizer.step()
    
    # BUG 3: Evaluating during training without torch.no_grad()
    accuracy = (model(X).argmax(dim=1) == y).float().mean()
    
    if epoch % 5 == 0:
        # BUG 4: Can't call .numpy() on tensor with gradients
        loss_val = loss.numpy()
        print(f"Epoch {epoch}: loss={loss_val:.4f}, acc={accuracy:.4f}")
'''

print("Buggy training loop (printed as string):")
print(buggy_training_loop)

print("""
Find and fix:
  Bug 1: ________________________________________________
  Bug 2: ________________________________________________
  Bug 3: ________________________________________________
  Bug 4: ________________________________________________
""")

# YOUR FIXED CODE HERE:
print("TODO: Write the fixed version of the training loop")

# ===========================================================================
# LEVEL 3: APPLICATION — Implement Logistic Regression in PyTorch
# ===========================================================================
print("\n--- LEVEL 3: APPLICATION ---")
print("""
TASK: Implement binary logistic regression in PyTorch from scratch.

Requirements:
  1. Generate synthetic binary classification data (n=300, 2 features)
  2. Build a model: nn.Linear(2, 1) + Sigmoid (or use BCEWithLogitsLoss)
  3. Train with BCEWithLogitsLoss + Adam (lr=0.05)
  4. Run for 200 epochs
  5. Report final accuracy on a 20% holdout test set
  6. Compare with sklearn LogisticRegression on the same data

Expected accuracy: > 90% for linearly separable data.

Hint for BCEWithLogitsLoss:
  - model output: raw logit (no sigmoid in model!)
  - y_true: float tensor of 0s and 1s
  - loss = BCEWithLogitsLoss()(logit, y_float)
""")

# YOUR CODE HERE:
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_np, y_np = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                   n_informative=2, random_state=42)
X_tr_np, X_te_np, y_tr_np, y_te_np = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

print("TODO: Implement PyTorch logistic regression and compare with sklearn")
print(f"  Data: X.shape={X_np.shape}, classes={np.bincount(y_np)}")

# sklearn reference (for comparison)
sk_model = LogisticRegression()
sk_model.fit(X_tr_np, y_tr_np)
sk_acc = sk_model.score(X_te_np, y_te_np)
print(f"  sklearn accuracy: {sk_acc:.4f}")
print("  YOUR PyTorch accuracy: ???")


# ===========================================================================
# LEVEL 4: CHALLENGE — Build a Custom Dataset From a CSV File
# ===========================================================================
print("\n--- LEVEL 4: CHALLENGE ---")
print("""
TASK: Build a custom PyTorch Dataset class that:
  1. Reads industrial_sensor_train.csv from the datasets/ directory
  2. Applies StandardScaler normalization
  3. Returns (feature_tensor, label_tensor) in __getitem__
  4. Trains a 3-class MLP using the DataLoader
  5. Reports final test accuracy

BONUS: Add data augmentation — randomly add Gaussian noise (σ=0.05)
       to features during training (in __getitem__), but not during evaluation.
       This is a form of regularization.

Dataset location: ../../datasets/industrial_sensor_train.csv
Feature columns: temperature, vibration, pressure, current, rpm, humidity, voltage, acoustic_emission
Target column: fault_severity (0=OK, 1=WARNING, 2=FAULT)
""")

from pathlib import Path
from torch.utils.data import Dataset

class IndustrialSensorDataset(Dataset):
    """TODO: Implement this Dataset class."""
    
    def __init__(self, csv_path, scaler=None, training=True):
        """
        Args:
            csv_path: path to the CSV file
            scaler: fitted StandardScaler (fit on train data, reuse for test)
            training: if True, apply data augmentation
        """
        # TODO: Load CSV with pandas
        # TODO: Extract features and labels
        # TODO: Handle missing values
        # TODO: Scale features
        # TODO: Convert to tensors
        pass
    
    def __len__(self):
        # TODO: return number of samples
        pass
    
    def __getitem__(self, idx):
        # TODO: return (feature_tensor, label_tensor)
        # TODO: apply noise augmentation if self.training is True
        pass

print("TODO: Implement IndustrialSensorDataset and train a 3-class MLP")
print("  Expected test accuracy: > 85%")

print("\n" + "=" * 60)
print("EXERCISES COMPLETE")
print("Check solutions/pytorch_solutions.py for reference answers.")
print("=" * 60)
