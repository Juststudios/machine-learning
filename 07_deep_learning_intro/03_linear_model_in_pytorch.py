"""
Linear Regression in PyTorch (Hello World of PyTorch)
======================================================
Now we use nn.Module — the building block of all PyTorch models.

Before: we manually defined w and b as tensors with requires_grad=True.
Now:    nn.Linear handles this automatically, plus gives us a clean API.

The 5-step training loop (MEMORIZE THIS):
  1. y_pred = model(X)             Forward pass
  2. loss = criterion(y_pred, y)   Compute loss
  3. optimizer.zero_grad()         Zero gradients
  4. loss.backward()               Backward pass
  5. optimizer.step()              Update weights
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

torch.manual_seed(42)
np.random.seed(42)

# ===========================================================================
# PART 1: nn.Module — The Building Block of All PyTorch Models
# ===========================================================================
print("=" * 60)
print("PART 1: nn.Module")
print("=" * 60)

print("""
nn.Module is the base class for ALL neural network models in PyTorch.

To create a model:
  1. Inherit from nn.Module
  2. Define layers in __init__()
  3. Define the forward computation in forward()

nn.Linear(in_features, out_features):
  Represents: y = xW^T + b
  Where W has shape (out_features, in_features) and b has shape (out_features)
  It AUTOMATICALLY creates W and b as learnable parameters.
""")

class LinearModel(nn.Module):
    """Simple linear regression model: y = wx + b."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # WHY super().__init__(): register this module with PyTorch's bookkeeping system
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass (computation)."""
        return self.linear(x)

model = LinearModel(in_features=3, out_features=1)
print("Model architecture:")
print(model)

print(f"\nAll learnable parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={param.shape}, values={param.data.flatten().round(decimals=4)[:4]}")

print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# ===========================================================================
# PART 2: Loss Functions & Optimizers
# ===========================================================================
print("\n--- PART 2: Loss Functions & Optimizers ---")

print("""
Loss function: measures how wrong the model's predictions are.
  For regression: nn.MSELoss() — Mean Squared Error
                  higher sensitivity to outliers

Optimizer: uses gradients to update weights.
  optim.SGD(params, lr):    Stochastic Gradient Descent
  optim.Adam(params, lr):   Adaptive Moment Estimation (usually best default)
""")

# Loss functions demo
criterion_mse = nn.MSELoss()
y_pred_demo = torch.tensor([3.0, 5.0, 7.0])
y_true_demo = torch.tensor([2.5, 5.5, 6.0])

mse_loss = criterion_mse(y_pred_demo, y_true_demo)
print(f"MSE Loss: {mse_loss.item():.4f}")
print(f"Manual:   {((y_pred_demo - y_true_demo)**2).mean().item():.4f}  ← same!")

# Optimizer demo
model_demo = LinearModel(1, 1)
optimizer_sgd  = optim.SGD(model_demo.parameters(),  lr=0.01)
optimizer_adam = optim.Adam(model_demo.parameters(), lr=0.001)

print(f"\nSGD  optimizer: {optimizer_sgd}")
print(f"Adam optimizer: {optimizer_adam}")

# ===========================================================================
# PART 3: The Complete Training Loop
# ===========================================================================
print("\n--- PART 3: The Complete Training Loop ---")

# --- Data preparation ---
n = 100
X_np = np.random.uniform(-5, 5, (n, 3))
# True relationship: y = 2x1 - 1.5x2 + 0.5x3 + 3
y_np = 2*X_np[:,0] - 1.5*X_np[:,1] + 0.5*X_np[:,2] + 3 + np.random.randn(n)*0.8

# Split
split = int(0.8 * n)
X_train_np, X_test_np = X_np[:split], X_np[split:]
y_train_np, y_test_np = y_np[:split], y_np[split:]

# Convert to tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test  = torch.tensor(X_test_np,  dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)  # shape (n, 1)
y_test  = torch.tensor(y_test_np,  dtype=torch.float32).unsqueeze(1)

# --- Model, loss, optimizer ---
model = LinearModel(in_features=3, out_features=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

print("Training linear regression with PyTorch:")
print(f"  Model: LinearModel(3 → 1)")
print(f"  Loss:  MSELoss")
print(f"  Optimizer: Adam (lr=0.05)")
print()

train_losses = []
test_losses  = []

# THE CANONICAL PYTORCH TRAINING LOOP
for epoch in range(300):
    # ---- Training mode ----
    model.train()                          # sets model to training mode
    
    # Step 1: Forward pass
    y_pred = model(X_train)
    
    # Step 2: Compute loss
    loss = criterion(y_pred, y_train)
    
    # Step 3: Zero gradients (MUST come before backward)
    optimizer.zero_grad()
    
    # Step 4: Backward pass
    loss.backward()
    
    # Step 5: Update weights
    optimizer.step()
    
    train_losses.append(loss.item())
    
    # ---- Evaluation (no gradient needed) ----
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = criterion(y_test_pred, y_test)
        test_losses.append(test_loss.item())
    
    if epoch % 50 == 0:
        print(f"  Epoch {epoch:3d}: Train Loss = {loss.item():.4f}, Test Loss = {test_loss.item():.4f}")

print(f"\nFinal weights (should be close to [2, -1.5, 0.5]):")
for name, param in model.named_parameters():
    print(f"  {name}: {param.data.flatten().tolist()}")

# ===========================================================================
# PART 4: Compare With sklearn (Should Get Similar Results)
# ===========================================================================
print("\n--- PART 4: Comparison With sklearn ---")

sklearn_model = LinearRegression()
sklearn_model.fit(X_train_np, y_train_np)
y_sk_pred = sklearn_model.predict(X_test_np)

# PyTorch predictions
model.eval()
with torch.no_grad():
    y_pt_pred = model(X_test).numpy().flatten()

r2_sk = r2_score(y_test_np, y_sk_pred)
r2_pt = r2_score(y_test_np, y_pt_pred)

print(f"sklearn LinearRegression R²: {r2_sk:.4f}")
print(f"PyTorch LinearModel R²:      {r2_pt:.4f}")
print(f"\nsklearn coefficients: {sklearn_model.coef_.round(4)}")
print(f"PyTorch coefficients: {model.linear.weight.data.numpy().flatten().round(4)}")

print("""
Results should be similar! sklearn uses the exact normal equation solution,
while PyTorch uses gradient descent — both converge to the same answer.
""")

# ===========================================================================
# PART 5: Visualize Training
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Loss curves
axes[0].plot(train_losses, label='Train Loss', color='steelblue')
axes[0].plot(test_losses,  label='Test Loss',  color='coral')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Training & Test Loss Over Time")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Prediction vs actual
axes[1].scatter(y_test_np, y_pt_pred, alpha=0.7, color='steelblue', label='PyTorch predictions')
axes[1].scatter(y_test_np, y_sk_pred, alpha=0.7, color='coral', marker='^', label='sklearn predictions')
min_val = min(y_test_np.min(), y_pt_pred.min())
max_val = max(y_test_np.max(), y_pt_pred.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect predictions')
axes[1].set_xlabel("True Values")
axes[1].set_ylabel("Predicted Values")
axes[1].set_title("Predictions vs True Values")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pytorch_linear_training.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nTraining plot saved: {OUTPUT_DIR}/pytorch_linear_training.png")

print("""
===========================================
THE CANONICAL PYTORCH TRAINING LOOP
===========================================
for epoch in range(num_epochs):
    model.train()                   # training mode
    
    y_pred = model(X_train)         # 1. forward pass
    loss = criterion(y_pred, y)     # 2. compute loss
    optimizer.zero_grad()           # 3. zero gradients (NEVER FORGET!)
    loss.backward()                 # 4. backward pass
    optimizer.step()                # 5. update weights
    
    model.eval()                    # evaluation mode
    with torch.no_grad():           # no gradient tracking during eval
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)

Next: 04_datasets_and_dataloaders.py — Efficient data pipeline for large datasets
""")
