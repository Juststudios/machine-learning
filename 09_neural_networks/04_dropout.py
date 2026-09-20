"""
Neural Networks: Dropout Regularization
========================================
Why do large neural networks overfit?

Deep neural networks possess massive parameter counts. When trained on finite
data, neurons often develop complex CO-ADAPTATIONS — where one neuron learns to
correct the specific errors of another neuron. While this can drive training error
to zero, the resulting representations are brittle and fail to generalize to
unseen data.

Dropout (Srivastava, Hinton, et al., 2014) is an exceptionally effective
regularization technique:
  1. During each training forward pass, randomly drop (zero out) each neuron's
     activation with probability p (retention probability 1 - p).
  2. Forces each neuron to learn self-reliant, robust features that work well in
     conjunction with diverse random subsets of other neurons.
  3. Can be understood as training an exponential ENSEMBLE of 2^N thinned
     subnetworks sharing weights.

Inverted Dropout (PyTorch Implementation):
  - Standard Dropout: Scales outputs during inference by (1 - p).
  - Inverted Dropout: Scales outputs during TRAINING by 1 / (1 - p).
  - Benefit: Inference is completely free of scaling and identity-mapped!
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

# ===========================================================================
# PART 1: The Problem — Co-Adaptation & Overfitting
# ===========================================================================
print("=" * 70)
print("PART 1: Neuron Co-Adaptation & The Need for Dropout")
print("=" * 70)
print("""
In unregularized deep networks:
- Neurons learn to rely on the presence of specific other neurons.
- The network memorizes idiosyncratic training quirks and high-frequency noise.
- Co-adapted features fail completely when tested on slightly perturbed data.

Dropout breaks co-adaptation:
- By randomly zeroing activations with probability p, no neuron can rely on
  any specific neighbor always being active.
- Every neuron must learn features that are individually useful and resilient.
""")

# ===========================================================================
# PART 2: Inverted Dropout Math & Train vs Eval Mode
# ===========================================================================
print("=" * 70)
print("PART 2: Inverted Dropout Mathematical Formulation")
print("=" * 70)
print("""
Mathematical Formulation:

Let h in R^d be the layer activation vector and p in [0, 1) be the drop probability.

1. Binary mask generation:
     m_j ~ Bernoulli(1 - p)   where P(m_j = 1) = 1 - p,  P(m_j = 0) = p

2. INVERTED DROPOUT scaling (PyTorch standard):
     h_dropped = (m * h) / (1 - p)

Expectation Proof:
     E[h_dropped] = E[m * h / (1 - p)]
                  = (h / (1 - p)) * E[m]
                  = (h / (1 - p)) * (1 - p)
                  = h

Why is this ingenious?
Because E[h_train] == h_eval, we do NOT need any scaling operations during
inference! At test time, dropout is simply the identity function:
     h_eval = h
""")

# Demonstrate Inverted Dropout numerically
p = 0.4
scale = 1.0 / (1.0 - p)
x_test = torch.ones(10)

dropout_layer = nn.Dropout(p=p)

dropout_layer.train()
y_train_1 = dropout_layer(x_test)
y_train_2 = dropout_layer(x_test)

dropout_layer.eval()
y_eval = dropout_layer(x_test)

print(f"Dropout probability p = {p}  (scaling factor = 1/(1-p) = {scale:.4f})")
print(f"Input vector:                   {x_test.numpy().tolist()}")
print(f"Train pass 1 (random drops):    {y_train_1.numpy().round(3).tolist()}")
print(f"Train pass 2 (different drops): {y_train_2.numpy().round(3).tolist()}")
print(f"Eval pass (identity mapped):    {y_eval.numpy().round(3).tolist()}")
print(f"Mean of active train units:     {y_train_1[y_train_1 > 0].mean().item():.4f} (scaled to compensate zeros)")
print(f"Mean across large train batch:  {dropout_layer(torch.ones(10000)).mean().item():.4f} (matches input mean 1.0)")
print("=> Inverted dropout perfectly preserves expectation during training!\n")

# ===========================================================================
# PART 3: The Ensemble Interpretation
# ===========================================================================
print("=" * 70)
print("PART 3: Ensemble Interpretation of Dropout")
print("=" * 70)
print("""
A network with N hidden units can be configured in 2^N distinct subnetwork topologies.
During training:
  - Each mini-batch updates a different randomly sampled subnetwork.
  - All 2^N subnetworks share the underlying weight matrix W.
During evaluation:
  - We run the full network with all weights active.
  - Hinton et al. proved that this single forward pass computes an accurate
    geometric mean of the predictions across all 2^N subnetworks!
  - You effectively get the predictive power of a massive ensemble for the
    computational cost of a single model.
""")

# ===========================================================================
# PART 4: Overfitting Regularization Experiment
# ===========================================================================
print("=" * 70)
print("PART 4: Overfitting Experiment — Wide MLP on Small Noisy Dataset")
print("=" * 70)

# Create a challenging overfitting scenario:
# 250 samples, 20 features (10 informative, 10 pure noise), 2 classes
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_raw, y_raw = make_classification(
    n_samples=250,
    n_features=20,
    n_informative=6,
    n_redundant=4,
    n_classes=2,
    flip_y=0.15,  # High label noise
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_raw, y_raw, test_size=0.4, random_state=42, stratify=y_raw
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

X_tr_t = torch.tensor(X_train_s, dtype=torch.float32)
y_tr_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=16, shuffle=True)

print(f"Dataset split: {len(X_train)} train samples, {len(X_val)} validation samples")

# Define Wide MLP Architecture
class WideMLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256, dropout_rate=0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        return self.out(x)

# Instantiate models: Without Dropout vs With Dropout (p=0.5)
torch.manual_seed(42)
model_no_drop = WideMLP(dropout_rate=0.0)
torch.manual_seed(42)
model_with_drop = WideMLP(dropout_rate=0.5)

criterion = nn.CrossEntropyLoss()
epochs = 90
lr = 0.002

opt_no_drop = optim.Adam(model_no_drop.parameters(), lr=lr)
opt_with_drop = optim.Adam(model_with_drop.parameters(), lr=lr)

history = {
    'no_drop':   {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
    'with_drop': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
}

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = criterion(logits, y).item()
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return loss, acc

print(f"Training Wide MLPs (3x256 hidden units) for {epochs} epochs...")

for epoch in range(epochs):
    # Train No-Dropout model
    model_no_drop.train()
    for bx, by in train_loader:
        opt_no_drop.zero_grad()
        loss = criterion(model_no_drop(bx), by)
        loss.backward()
        opt_no_drop.step()
        
    # Train With-Dropout model
    model_with_drop.train()
    for bx, by in train_loader:
        opt_with_drop.zero_grad()
        loss = criterion(model_with_drop(bx), by)
        loss.backward()
        opt_with_drop.step()
        
    # Record metrics in eval mode
    tr_l1, tr_a1 = evaluate(model_no_drop, X_tr_t, y_tr_t)
    va_l1, va_a1 = evaluate(model_no_drop, X_val_t, y_val_t)
    history['no_drop']['train_loss'].append(tr_l1)
    history['no_drop']['val_loss'].append(va_l1)
    history['no_drop']['train_acc'].append(tr_a1)
    history['no_drop']['val_acc'].append(va_a1)
    
    tr_l2, tr_a2 = evaluate(model_with_drop, X_tr_t, y_tr_t)
    va_l2, va_a2 = evaluate(model_with_drop, X_val_t, y_val_t)
    history['with_drop']['train_loss'].append(tr_l2)
    history['with_drop']['val_loss'].append(va_l2)
    history['with_drop']['train_acc'].append(tr_a2)
    history['with_drop']['val_acc'].append(va_a2)
    
    if (epoch + 1) % 15 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"NoDrop ValLoss: {va_l1:.3f} (Acc: {va_a1*100:.1f}%) | "
              f"WithDrop ValLoss: {va_l2:.3f} (Acc: {va_a2*100:.1f}%)")

print("\nFinal Performance Comparison:")
print(f"  WITHOUT Dropout: Train Loss={history['no_drop']['train_loss'][-1]:.4f}, Train Acc={history['no_drop']['train_acc'][-1]*100:.1f}% | "
      f"Val Loss={history['no_drop']['val_loss'][-1]:.4f}, Val Acc={history['no_drop']['val_acc'][-1]*100:.1f}%")
print(f"  WITH Dropout:    Train Loss={history['with_drop']['train_loss'][-1]:.4f}, Train Acc={history['with_drop']['train_acc'][-1]*100:.1f}% | "
      f"Val Loss={history['with_drop']['val_loss'][-1]:.4f}, Val Acc={history['with_drop']['val_acc'][-1]*100:.1f}%")

# ===========================================================================
# PART 5: Visualizing Dropout Effect
# ===========================================================================
print("\n" + "=" * 70)
print("PART 5: Saving Visualization to output/dropout_effect.png")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Loss curves
epochs_range = range(1, epochs + 1)
ax1.plot(epochs_range, history['no_drop']['train_loss'], 'r--', label='Train Loss (No Dropout)', alpha=0.7)
ax1.plot(epochs_range, history['no_drop']['val_loss'], 'r-', label='Val Loss (No Dropout)', linewidth=2)
ax1.plot(epochs_range, history['with_drop']['train_loss'], 'b--', label='Train Loss (Dropout p=0.5)', alpha=0.7)
ax1.plot(epochs_range, history['with_drop']['val_loss'], 'b-', label='Val Loss (Dropout p=0.5)', linewidth=2)
ax1.set_title("Training & Validation Loss", fontsize=13, fontweight='bold')
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("CrossEntropy Loss", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Subplot 2: Accuracy curves
ax2.plot(epochs_range, history['no_drop']['train_acc'], 'r--', label='Train Acc (No Dropout)', alpha=0.7)
ax2.plot(epochs_range, history['no_drop']['val_acc'], 'r-', label='Val Acc (No Dropout)', linewidth=2)
ax2.plot(epochs_range, history['with_drop']['train_acc'], 'b--', label='Train Acc (Dropout p=0.5)', alpha=0.7)
ax2.plot(epochs_range, history['with_drop']['val_acc'], 'b-', label='Val Acc (Dropout p=0.5)', linewidth=2)
ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight='bold')
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Accuracy", fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

plt.tight_layout()
fig_path = OUTPUT_DIR / "dropout_effect.png"
plt.savefig(fig_path, dpi=120)
plt.close()

print(f"✅ Saved plot to: {fig_path}")

# ===========================================================================
# PART 6: Practical Rules of Thumb
# ===========================================================================
print("\n" + "=" * 70)
print("DROPOUT — PRACTICAL RULES OF THUMB")
print("=" * 70)
print("""
1. Standard Drop Rate:
   - For hidden layers in MLPs: p in [0.2, 0.5] (0.5 was recommended in original paper).
   - For input layers: keep p low (p in [0.1, 0.2]) or avoid dropout on inputs.
2. Architecture Compatibility:
   - Placement: Linear -> BatchNorm1d -> ReLU -> Dropout
   - When using BatchNorm, lower dropout rates (e.g., p=0.2 - 0.3) are typically ideal
     because BatchNorm itself provides implicit regularization.
3. Overfitting Diagnostic:
   - If Train Loss << Val Loss: INCREASE dropout probability p.
   - If Train Loss is too high (underfitting): DECREASE or remove dropout.
4. Always check `model.eval()`:
   - Failing to call `model.eval()` before test evaluation leaves dropout active,
     producing noisy, degraded test scores.
""")
