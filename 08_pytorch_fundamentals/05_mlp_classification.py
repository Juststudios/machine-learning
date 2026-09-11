"""
PyTorch MLP for Classification — Full Project
==============================================
Build and train a complete Multi-Layer Perceptron (MLP) for classification.

We'll tackle a realistic 3-class classification problem using:
  - sklearn's make_classification for a well-understood dataset
  - Complete PyTorch MLP with professional training loop
  - Visualization of decision boundaries and training progress
  - Comparison with sklearn baselines

This is the "capstone" of the PyTorch Fundamentals module before
moving to more complex architectures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import copy

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cpu')

# ===========================================================================
# DATA SETUP
# ===========================================================================
print("=" * 60)
print("MLP Classification — Full Project")
print("=" * 60)

X_np, y_np = make_classification(
    n_samples=1000,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=2,
    random_state=42,
)

print(f"Dataset: {X_np.shape}, classes: {np.bincount(y_np)}")

X_tr, X_te, y_tr, y_te = train_test_split(X_np, y_np, test_size=0.2, stratify=y_np, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(X_te)

print(f"Split — Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}")

# Convert to tensors
def to_tensor(X, y):
    return (torch.tensor(X, dtype=torch.float32).to(DEVICE),
            torch.tensor(y, dtype=torch.long).to(DEVICE))

X_tr_t, y_tr_t = to_tensor(X_tr_s, y_tr)
X_va_t, y_va_t = to_tensor(X_va_s, y_va)
X_te_t, y_te_t = to_tensor(X_te_s, y_te)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=32, shuffle=False)

# ===========================================================================
# PART 1: Define the MLP Architecture
# ===========================================================================
print("\n--- PART 1: MLP Architecture ---")

class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron for multi-class classification.
    
    Architecture:
      Input → Linear(128) → BN → ReLU → Dropout
           → Linear(64)  → BN → ReLU → Dropout
           → Linear(32)  → BN → ReLU → Dropout
           → Linear(n_classes)  (logits)
    """
    def __init__(self, input_dim: int, hidden_dims: list, n_classes: int, dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),   # stabilize training
                nn.ReLU(),
                nn.Dropout(dropout),          # regularize
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_classes))  # output layer: no activation (CrossEntropyLoss handles it)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

model = MLPClassifier(
    input_dim=12,
    hidden_dims=[128, 64, 32],
    n_classes=3,
    dropout=0.3,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {model}")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ===========================================================================
# PART 2: Training Loop
# ===========================================================================
print("\n--- PART 2: Training ---")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8, factor=0.5)

best_val_acc  = 0.0
best_state    = None
patience_counter = 0
PATIENCE = 20

history = {'train_loss': [], 'val_acc': [], 'train_acc': [], 'lr': []}

def compute_accuracy(loader, model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for Xb, yb in loader:
            preds = model(Xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += len(yb)
    return correct / total

print(f"Training for up to 200 epochs with early stopping (patience={PATIENCE})...\n")

for epoch in range(200):
    model.train()
    epoch_losses = []
    
    for Xb, yb in train_loader:
        logits = model(Xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_losses.append(loss.item())
    
    train_acc = compute_accuracy(train_loader, model)
    val_acc   = compute_accuracy(val_loader,   model)
    avg_loss  = np.mean(epoch_losses)
    current_lr = optimizer.param_groups[0]['lr']
    
    scheduler.step(val_acc)
    
    history['train_loss'].append(avg_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
    
    if epoch % 20 == 0 or epoch < 5:
        print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    
    if patience_counter >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch+1}")
        break

# Restore best model
model.load_state_dict(best_state)
print(f"\nBest validation accuracy: {best_val_acc:.4f}")

# ===========================================================================
# PART 3: Final Evaluation
# ===========================================================================
print("\n--- PART 3: Final Evaluation ---")

model.eval()
with torch.no_grad():
    test_logits = model(X_te_t)
    test_preds  = test_logits.argmax(dim=1).cpu().numpy()

test_acc = accuracy_score(y_te, test_preds)
test_f1  = f1_score(y_te, test_preds, average='macro')

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Macro-F1: {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_te, test_preds, target_names=['Class 0', 'Class 1', 'Class 2']))

# ===========================================================================
# PART 4: Compare With sklearn Baselines
# ===========================================================================
print("--- PART 4: Comparison With sklearn Baselines ---")

baselines = [
    ('LogisticRegression', LogisticRegression(max_iter=500, C=1.0)),
    ('RandomForest(100)',  RandomForestClassifier(n_estimators=100, random_state=42)),
]

for name, clf in baselines:
    clf.fit(X_tr_s, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te_s))
    f1  = f1_score(y_te, clf.predict(X_te_s), average='macro')
    print(f"  {name:30s}: Accuracy={acc:.4f}, Macro-F1={f1:.4f}")

print(f"  {'PyTorch MLP':30s}: Accuracy={test_acc:.4f}, Macro-F1={test_f1:.4f}")

# ===========================================================================
# PART 5: Training Visualization
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("MLP Classification — Training History", fontsize=12, fontweight='bold')

epochs = range(1, len(history['train_loss']) + 1)

axes[0].plot(epochs, history['train_loss'], color='steelblue')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].set_title("Training Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, history['train_acc'], label='Train', color='steelblue')
axes[1].plot(epochs, history['val_acc'],   label='Val',   color='coral')
axes[1].axhline(test_acc, color='green', linestyle='--', label=f'Test: {test_acc:.3f}')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Train vs Validation Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs, history['lr'], color='purple')
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Learning Rate")
axes[2].set_title("Learning Rate Schedule")
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mlp_classification_training.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nTraining visualization saved: {OUTPUT_DIR}/mlp_classification_training.png")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. MLPClassifier: stack Linear → BatchNorm → ReLU → Dropout layers
2. CrossEntropyLoss: takes raw logits (NO softmax in forward!)
3. BatchNorm: stabilizes training, allows higher learning rates
4. Dropout: strong regularizer, must disable during eval with model.eval()
5. Early stopping + checkpoint: always restore the best model seen
6. ReduceLROnPlateau: reduce LR when validation stops improving
7. Compare with sklearn baselines to know if deep learning is worth it
   (For small tabular data, RandomForest often wins — don't over-engineer!)

Next: 09_neural_networks/ — Deep dive into what makes networks work
""")
