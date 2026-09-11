"""
PyTorch Datasets and DataLoaders
==================================
Training on large datasets requires more than a single tensor.

Problems with loading ALL data at once:
  - 1 million images don't fit in RAM
  - Training on all data at once (batch gradient descent) is slow
  - Memory bandwidth becomes the bottleneck

Solution: Mini-batch training with DataLoader
  - Load data in small chunks (batches)
  - Each batch: forward → loss → backward → update
  - Randomize order each epoch (shuffling)

This is the standard data pipeline for ALL serious PyTorch training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

torch.manual_seed(42)
np.random.seed(42)

# ===========================================================================
# PART 1: The Dataset Class
# ===========================================================================
print("=" * 60)
print("PART 1: The Dataset Class")
print("=" * 60)

print("""
torch.utils.data.Dataset is an abstract class.
To create a custom dataset, implement:
  __len__(self):         returns number of samples
  __getitem__(self, idx): returns one sample (features, label)

PyTorch's DataLoader then handles:
  - Batching: grouping samples into mini-batches
  - Shuffling: randomize order each epoch
  - Parallel loading: multiple worker processes
""")

class SensorDataset(Dataset):
    """
    Custom Dataset for industrial sensor readings.
    
    Each sample: (sensor_readings, fault_label)
    sensor_readings: 6 sensor values (temperature, vibration, rpm, current, pressure, voltage)
    fault_label: 0=OK, 1=WARNING, 2=FAULT
    """
    
    def __init__(self, n_samples: int = 1000):
        """Generate synthetic sensor data."""
        np.random.seed(42)
        
        # Generate 3 classes with different sensor profiles
        n_per_class = n_samples // 3
        
        # Class 0: OK — all sensors nominal
        X_ok = np.random.randn(n_per_class, 6) * 0.5 + np.array([70, 1.0, 3000, 15, 5, 220])
        
        # Class 1: WARNING — elevated temperature and vibration
        X_warn = np.random.randn(n_per_class, 6) * 0.8 + np.array([82, 2.5, 2800, 18, 5.5, 218])
        
        # Class 2: FAULT — high temperature, high vibration, abnormal current
        X_fault = np.random.randn(n_per_class, 6) * 1.2 + np.array([95, 5.0, 2500, 25, 6, 215])
        
        X = np.vstack([X_ok, X_warn, X_fault]).astype(np.float32)
        y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class, dtype=np.int64)
        
        # Store as tensors
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        
        # Normalize features
        self.mean = self.X.mean(dim=0)
        self.std  = self.X.std(dim=0)
        self.X = (self.X - self.mean) / self.std
    
    def __len__(self):
        """Return total number of samples."""
        return len(self.y)
    
    def __getitem__(self, idx):
        """Return one sample: (features tensor, label tensor)."""
        return self.X[idx], self.y[idx]

# Create dataset
dataset = SensorDataset(n_samples=900)
print(f"Dataset size: {len(dataset)} samples")
print(f"Feature shape: {dataset.X.shape}")
print(f"Labels shape:  {dataset.y.shape}")
print(f"Label counts:  {[(dataset.y == i).sum().item() for i in range(3)]} (OK, WARNING, FAULT)")

# Access individual samples
sample_X, sample_y = dataset[0]
print(f"\nSample 0: X.shape={sample_X.shape}, y={sample_y.item()} (0=OK)")

# ===========================================================================
# PART 2: Splitting With random_split
# ===========================================================================
print("\n--- PART 2: Splitting the Dataset ---")

# random_split: split dataset into train/val/test
total = len(dataset)
train_size = int(0.7 * total)
val_size   = int(0.15 * total)
test_size  = total - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # reproducible split
)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# ===========================================================================
# PART 3: DataLoader — Batching, Shuffling, Parallel Loading
# ===========================================================================
print("\n--- PART 3: DataLoader ---")

print("""
DataLoader wraps a Dataset and provides:
  batch_size:  how many samples per batch
  shuffle:     randomize order each epoch (True for training, False for val/test)
  num_workers: parallel data loading processes (0 = main process only)
  drop_last:   drop the last incomplete batch (useful for batch norm)
""")

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,      # WHY: randomize to prevent model learning order patterns
    num_workers=0,     # 0 for compatibility; increase for real datasets
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,     # WHY: no need to shuffle validation data
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

print(f"\nTrain loader: {len(train_loader)} batches of ≤32 samples")
print(f"Val loader:   {len(val_loader)} batches")
print(f"Test loader:  {len(test_loader)} batches")

# Inspect one batch
batch_X, batch_y = next(iter(train_loader))
print(f"\nOne batch: X.shape={batch_X.shape}, y.shape={batch_y.shape}")
print(f"  First 5 labels: {batch_y[:5].tolist()}")

# ===========================================================================
# PART 4: Training With DataLoader
# ===========================================================================
print("\n--- PART 4: Mini-Batch Training With DataLoader ---")

class FaultClassifier(nn.Module):
    """Multi-layer perceptron for 3-class fault classification."""
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # 3 output classes: OK, WARNING, FAULT
        )
    
    def forward(self, x):
        return self.network(x)

model = FaultClassifier()
criterion = nn.CrossEntropyLoss()  # for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model: {model}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
print()

train_acc_hist = []
val_acc_hist   = []

def compute_accuracy(loader, model):
    """Compute accuracy over a DataLoader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    return correct / total

num_epochs = 30

for epoch in range(num_epochs):
    # ---- Training phase: iterate over all batches ----
    model.train()
    for X_batch, y_batch in train_loader:
        logits = model(X_batch)             # forward pass
        loss = criterion(logits, y_batch)   # compute loss
        optimizer.zero_grad()               # zero gradients
        loss.backward()                     # backward pass
        optimizer.step()                    # update weights
    
    # ---- Evaluation phase ----
    train_acc = compute_accuracy(train_loader, model)
    val_acc   = compute_accuracy(val_loader, model)
    
    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)
    
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:2d}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

# Final test evaluation
test_acc = compute_accuracy(test_loader, model)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# ===========================================================================
# PART 5: TensorDataset — Quick Shortcut
# ===========================================================================
print("\n--- PART 5: TensorDataset (Quick Shortcut) ---")

print("""
When your data is already in tensors, use TensorDataset.
No need to write a custom Dataset class.
""")

# Quick dataset from existing tensors
X_quick = torch.randn(200, 4)
y_quick = torch.randint(0, 2, (200,))

quick_dataset = TensorDataset(X_quick, y_quick)
quick_loader  = DataLoader(quick_dataset, batch_size=16, shuffle=True)

print(f"TensorDataset size: {len(quick_dataset)}")
batch = next(iter(quick_loader))
print(f"Batch X shape: {batch[0].shape}, y shape: {batch[1].shape}")

# ===========================================================================
# PART 6: Plot Training History
# ===========================================================================
fig, ax = plt.subplots(figsize=(8, 5))
epochs = range(len(train_acc_hist))
ax.plot(epochs, train_acc_hist, 'o-', color='steelblue', label='Training Accuracy')
ax.plot(epochs, val_acc_hist,   'o-', color='coral',     label='Validation Accuracy')
ax.axhline(y=test_acc, color='green', linestyle='--', label=f'Test Accuracy: {test_acc:.3f}')
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Sensor Fault Classification: Training History")
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dataloader_training.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nTraining history plot saved: {OUTPUT_DIR}/dataloader_training.png")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. Dataset: defines __len__ and __getitem__ — the interface PyTorch expects
2. DataLoader: wraps Dataset, handles batching, shuffling, parallel loading
3. Training loop: iterate over loader → batch forward → loss → backward → step
4. Shuffle=True for training (prevents learning order), False for val/test
5. TensorDataset: quick shortcut when data already in tensor form
6. mini-batch size (32-256) balances: memory, gradient quality, training speed

The DataLoader pattern scales to:
  - Datasets too large for RAM (stream from disk)
  - Image datasets (with transforms applied per-batch)
  - Text datasets with variable-length sequences (with padding)

Next: exercises.py — Practice what you've learned about PyTorch basics
""")
