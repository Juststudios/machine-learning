"""
The Complete PyTorch Training Loop
====================================
This lesson teaches the professional training loop used in real ML projects.

Beyond the basic 5-step loop, production training includes:
  - Validation monitoring every epoch
  - Early stopping: stop when validation stops improving
  - Learning rate scheduling: reduce LR when stuck
  - Model checkpointing: save the best model seen so far
  - Gradient clipping: prevent exploding gradients

After this lesson, you'll know how to train ANY PyTorch model professionally.
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
# DATA PREPARATION
# ===========================================================================
print("=" * 60)
print("The Professional PyTorch Training Loop")
print("=" * 60)

# Generate regression data
n = 1000
X_np = np.random.randn(n, 8).astype(np.float32)
# True weights
w_true = np.array([3.0, -2.0, 1.5, 0.0, -1.0, 2.5, 0.0, -0.5], dtype=np.float32)
y_np = X_np @ w_true + 0.5 + np.random.randn(n).astype(np.float32) * 0.8

# Split: 70% train, 15% val, 15% test
n_train = int(0.7 * n)
n_val   = int(0.15 * n)

X_train = torch.tensor(X_np[:n_train])
y_train = torch.tensor(y_np[:n_train]).unsqueeze(1)
X_val   = torch.tensor(X_np[n_train:n_train+n_val])
y_val   = torch.tensor(y_np[n_train:n_train+n_val]).unsqueeze(1)
X_test  = torch.tensor(X_np[n_train+n_val:])
y_test  = torch.tensor(y_np[n_train+n_val:]).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=32, shuffle=False)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ===========================================================================
# MODEL DEFINITION
# ===========================================================================
class MLP(nn.Module):
    """Multi-layer perceptron for regression."""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int = 1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

model = MLP(input_dim=8, hidden_dims=[64, 32])
print(f"\nModel: {model}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

# ===========================================================================
# PART 1: Basic Training Loop (Review)
# ===========================================================================
print("\n--- The 5-Step Training Loop (canonical) ---")
print("""
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)                 # 1. Forward
        loss = criterion(y_pred, y_batch)       # 2. Loss
        optimizer.zero_grad()                   # 3. Zero grads
        loss.backward()                         # 4. Backward
        optimizer.step()                        # 5. Update
""")

# ===========================================================================
# PART 2: Professional Training Loop With All Features
# ===========================================================================
print("--- Professional Training Loop ---\n")

def train_model(model, train_loader, val_loader,
                criterion, optimizer, scheduler=None,
                num_epochs=100, patience=10,
                checkpoint_path=None, clip_grad_norm=None):
    """
    Professional training loop with:
    - Train/validation monitoring
    - Early stopping
    - LR scheduling
    - Model checkpointing
    - Gradient clipping
    
    Returns: history dict with train/val losses per epoch
    """
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # ---- TRAINING PHASE ----
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping: prevents gradient explosion in deep networks
            # WHY: if gradients become huge, weight updates blow up
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # ---- VALIDATION PHASE ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']
        
        # ---- LEARNING RATE SCHEDULING ----
        if scheduler is not None:
            # ReduceLROnPlateau: reduce LR when validation loss plateaus
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # ---- CHECKPOINTING ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
        
        # ---- EARLY STOPPING ----
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
        
        # ---- LOGGING ----
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)
        
        if epoch % 10 == 0 or epoch < 5:
            print(f"  Epoch {epoch+1:3d}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}, lr={current_lr:.6f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model (val_loss={best_val_loss:.4f})")
    
    return history


# Setup
model = MLP(input_dim=8, hidden_dims=[64, 32])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Learning rate scheduler: reduce LR by 0.5 when val loss plateaus for 5 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',           # monitor minimum (val loss)
    factor=0.5,           # multiply LR by 0.5
    patience=5,           # wait 5 epochs before reducing
    verbose=False,
)

checkpoint_path = OUTPUT_DIR / "best_model.pt"

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=150,
    patience=20,
    checkpoint_path=checkpoint_path,
    clip_grad_norm=1.0,   # clip gradient norm to 1.0
)

# ===========================================================================
# PART 3: Evaluating the Best Model on Test Set
# ===========================================================================
print("\n--- Final Test Evaluation ---")

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)
    test_loss = criterion(y_test_pred, y_test)

# Compute R²
ss_res = ((y_test - y_test_pred) ** 2).sum()
ss_tot = ((y_test - y_test.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot

print(f"Test MSE:  {test_loss.item():.4f}")
print(f"Test RMSE: {np.sqrt(test_loss.item()):.4f}")
print(f"Test R²:   {r2.item():.4f}")

# ===========================================================================
# PART 4: Loading a Saved Checkpoint
# ===========================================================================
print("\n--- Loading a Saved Checkpoint ---")

# This is how you reload a saved model
new_model = MLP(input_dim=8, hidden_dims=[64, 32])  # must have same architecture
new_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
new_model.eval()

with torch.no_grad():
    y_check = new_model(X_test[:5])
    print(f"Loaded model predictions (first 5):")
    print(f"  Predicted: {y_check.flatten().tolist()}")
    print(f"  True:      {y_test[:5].flatten().tolist()}")

# ===========================================================================
# PART 5: Visualize Training History
# ===========================================================================
epochs = range(1, len(history['train_loss']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Loss curves
axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='steelblue')
axes[0].plot(epochs, history['val_loss'],   label='Val Loss',   color='coral')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Training & Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Learning rate
axes[1].plot(epochs, history['lr'], color='green', linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Learning Rate")
axes[1].set_title("Learning Rate Schedule (ReduceLROnPlateau)")
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_loop_history.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nTraining history saved: {OUTPUT_DIR}/training_loop_history.png")

print("""
===========================================
PROFESSIONAL TRAINING LOOP TEMPLATE
===========================================

for epoch in range(num_epochs):
    # === TRAINING ===
    model.train()
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # optional
        optimizer.step()
    
    # === VALIDATION ===
    model.eval()
    with torch.no_grad():
        val_loss = compute_val_loss(model, val_loader, criterion)
    
    # === SCHEDULING ===
    scheduler.step(val_loss)  # for ReduceLROnPlateau
    
    # === EARLY STOPPING ===
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        break

# Reload best model before final test evaluation
model.load_state_dict(torch.load('best_model.pt'))
""")
