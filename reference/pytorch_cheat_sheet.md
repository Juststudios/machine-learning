# PyTorch Cheat Sheet

> Quick reference for PyTorch tensors, layers, training loops, and debugging.

---

## Tensors

```python
import torch

# Creation
t = torch.tensor([1.0, 2.0, 3.0])                 # from list
t = torch.zeros(3, 4)                              # all zeros
t = torch.ones(2, 3)                               # all ones
t = torch.rand(3, 3)                               # uniform [0, 1)
t = torch.randn(3, 3)                              # normal N(0, 1)
t = torch.eye(4)                                   # identity matrix
t = torch.arange(0, 10, 2)                        # [0, 2, 4, 6, 8]
t = torch.linspace(0, 1, 5)                       # [0, 0.25, 0.5, 0.75, 1]
t = torch.from_numpy(np_array)                    # from numpy (shares memory)
t = torch.tensor(np_array, dtype=torch.float32)  # copy from numpy

# Properties
t.shape        # torch.Size([3, 4])
t.dtype        # torch.float32, torch.int64, etc.
t.device       # cpu or cuda:0
t.requires_grad  # whether autograd tracks this tensor

# Shape operations
t.view(2, 6)            # reshape (must be compatible, shares data)
t.reshape(2, 6)         # reshape (may copy)
t.unsqueeze(0)          # add dimension: [3,4] → [1,3,4]
t.squeeze()             # remove size-1 dims: [1,3,1] → [3]
t.permute(2, 0, 1)      # rearrange dimensions
t.transpose(0, 1)       # swap two dimensions
t.flatten()             # flatten to 1D
t.contiguous()          # make memory contiguous (needed after permute)

# Numpy bridge
arr = t.numpy()            # tensor → numpy (CPU tensors only, shares memory)
arr = t.detach().numpy()   # if t has gradients, must detach first
```

---

## Tensor Operations

```python
# Arithmetic (element-wise)
a + b, a - b, a * b, a / b, a ** 2
torch.sqrt(a), torch.exp(a), torch.log(a), torch.abs(a)

# Matrix operations
a @ b                   # matrix multiply (preferred)
torch.matmul(a, b)      # same as @
a.T                     # transpose (2D)
torch.mm(a, b)          # matrix multiply (2D only)
torch.bmm(a, b)         # batch matrix multiply (3D: batch × m × k @ batch × k × n)

# Reduction
t.sum()                 # sum all elements
t.sum(dim=0)            # sum along rows → one value per column
t.mean(), t.std(), t.max(), t.min()
t.argmax(), t.argmin()  # index of max/min element
t.argmax(dim=1)         # index of max per row

# Comparison
t > 0                   # boolean mask
t.clamp(min=0, max=1)   # clip values
t.masked_fill(mask, value)  # fill masked positions

# Indexing / slicing (same as numpy)
t[0]          # first row
t[:, 2]       # all rows, column 2
t[mask]       # boolean indexing
```

---

## Common Layer Types (nn.Module)

```python
import torch.nn as nn

# Linear / Dense
nn.Linear(in_features=128, out_features=64)       # y = xW^T + b

# Activations
nn.ReLU()
nn.LeakyReLU(negative_slope=0.01)
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=1)    # for multi-class output
nn.GELU()            # used in transformers

# Normalization
nn.BatchNorm1d(num_features=64)    # for 2D input (batch × features)
nn.BatchNorm2d(num_features=32)    # for 4D input (batch × channels × H × W)
nn.LayerNorm(normalized_shape=64)  # normalize each sample independently

# Regularization
nn.Dropout(p=0.5)    # randomly zero p fraction of neurons during training

# Convolutional
nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
nn.MaxPool2d(kernel_size=2, stride=2)
nn.AdaptiveAvgPool2d(output_size=(1, 1))  # global average pool

# Embedding (for NLP)
nn.Embedding(num_embeddings=10000, embedding_dim=256)

# Transformer components
nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1)
nn.TransformerEncoderLayer(d_model=256, nhead=8)
nn.TransformerEncoder(encoder_layer, num_layers=6)
```

---

## Building Models with nn.Module

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define ALL layers here
        self.fc1 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        # Define forward computation here
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.drop(x)
        return self.fc2(x)

model = MyModel()

# Using nn.Sequential (for simple sequential models)
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 10),
)

# Model info
print(model)                                                  # architecture
print(sum(p.numel() for p in model.parameters()))             # param count
print(list(model.named_parameters()))                         # all params + names
model.state_dict()                                            # all weights as dict
```

---

## Loss Functions

```python
# Regression
nn.MSELoss()                    # mean squared error (sensitive to outliers)
nn.L1Loss()                     # mean absolute error (robust to outliers)
nn.HuberLoss(delta=1.0)         # smooth L1 (best of both)

# Binary Classification
nn.BCEWithLogitsLoss()          # preferred: BCE + sigmoid (numerically stable)
nn.BCELoss()                    # use with explicit sigmoid output

# Multi-class Classification
nn.CrossEntropyLoss()           # preferred: softmax + NLL in one (numerically stable)
nn.NLLLoss()                    # use with explicit log_softmax output

# Note: CrossEntropyLoss expects RAW LOGITS (no softmax!)
#       BCEWithLogitsLoss expects RAW LOGITS (no sigmoid!)

# Class weighting (for imbalanced data)
weights = torch.tensor([1.0, 3.0, 5.0])  # weight minority classes higher
criterion = nn.CrossEntropyLoss(weight=weights)
```

---

## Optimizers

```python
import torch.optim as optim

# Common optimizers
optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)  # Adam + decoupled weight decay
optim.RMSprop(model.parameters(), lr=1e-3)

# Learning rate schedulers
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=100)

# Update scheduler (call after each epoch)
scheduler.step()                    # for StepLR, CosineAnnealingLR
scheduler.step(val_loss)            # for ReduceLROnPlateau
```

---

## The Canonical Training Loop

```python
NUM_EPOCHS = 100
PATIENCE   = 10

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    # === TRAIN ===
    model.train()                           # enable dropout, batch norm training mode
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)             # 1. Forward
        loss = criterion(y_pred, y_batch)   # 2. Loss
        optimizer.zero_grad()               # 3. Zero grads  ← NEVER FORGET
        loss.backward()                     # 4. Backward
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 5. Clip (optional)
        optimizer.step()                    # 6. Update
    
    # === VALIDATE ===
    model.eval()                            # disable dropout, batch norm uses running stats
    with torch.no_grad():                   # disable gradient computation (faster)
        val_losses = [criterion(model(X_b), y_b).item() for X_b, y_b in val_loader]
        val_loss = sum(val_losses) / len(val_losses)
    
    # === SCHEDULE ===
    scheduler.step(val_loss)               # ReduceLROnPlateau
    
    # === CHECKPOINT + EARLY STOP ===
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

# Reload best
model.load_state_dict(torch.load('best_model.pt', weights_only=True))
```

---

## Data Loading

```python
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# Quick: TensorDataset from existing tensors
dataset = TensorDataset(X_tensor, y_tensor)

# Full: custom Dataset class
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# Split
train_ds, val_ds, test_ds = random_split(dataset, [700, 150, 150])
```

---

## Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `Expected input batch_size (N) to match target batch_size (M)` | y not squeezed | `y = y.unsqueeze(1)` for regression |
| `RuntimeError: expected scalar type Long but found Float` | Wrong dtype for CrossEntropyLoss | `y = y.long()` |
| `Expected 4D input (NCHW) but got 3D` | Missing batch dim | `x = x.unsqueeze(0)` |
| Loss is NaN | Exploding gradients or bad init | `clip_grad_norm_`, check LR |
| Loss doesn't decrease | Wrong loss, LR too small/large | Verify loss fn, adjust LR |
| GPU tensor / CPU tensor mismatch | Device mismatch | Move both to same device: `.to(device)` |
| `Can't call numpy() on tensor that requires grad` | Gradient attached | `.detach().numpy()` |
| Validation loss >> Train loss | Overfitting | Add Dropout, reduce model size, get more data |
| `optimizer.zero_grad()` not called | Gradient accumulation | Add `zero_grad()` before `backward()` |

---

## Device Management (CPU/GPU)

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

model = model.to(device)

for X_batch, y_batch in loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    # ... rest of training loop

# Save/load (always loads to CPU first)
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt', map_location='cpu', weights_only=True))
```
