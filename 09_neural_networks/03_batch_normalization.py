"""
Neural Networks: Batch Normalization
=====================================
Why do deep neural networks struggle to train?

As gradients flow backward and weights update during training, the distribution
of each layer's inputs changes continually. This phenomenon is termed
INTERNAL COVARIATE SHIFT. Subsequent layers must continually adapt to these
shifting input distributions, forcing practitioners to use very small learning
rates and hyper-careful weight initialization to avoid exploding or vanishing activations.

Batch Normalization (Ioffe & Szegedy, 2015) solves this by explicitly normalizing
layer activations across each mini-batch:
  1. Computes mini-batch mean and variance for each feature.
  2. Normalizes activations to zero mean and unit variance.
  3. Applies learnable affine parameters (gamma, beta) so the network can
     recover any optimal representation (including the identity mapping).
  4. Maintains running statistics (EMA) for deterministic single-sample inference.

Key Advantages:
  - Enables significantly higher learning rates without divergence.
  - Dampens dependence on hyper-sensitive weight initializations.
  - Acts as a mild regularizer due to batch-sampling noise.
  - Prevents activation saturation across deep layers.
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
# PART 1: Internal Covariate Shift Explanation
# ===========================================================================
print("=" * 70)
print("PART 1: Understanding Internal Covariate Shift")
print("=" * 70)
print("""
In deep architectures, layer k's input is the output of layer k-1:
    h_k = f(W_k * h_{k-1} + b_k)

When W_1, W_2, ..., W_{k-1} update during backpropagation, the distribution
of h_{k-1} shifts. Even if the shift per layer is modest, compounding across
5, 10, or 50 layers causes deep representations to drift wildly.

Consequences without normalization:
  - Activations easily explode to extreme magnitudes or vanish toward zero.
  - Gradients through saturating nonlinearities vanish completely.
  - Learning rates must remain tiny (e.g., 1e-4), making optimization sluggish.
""")

# ===========================================================================
# PART 2: Mathematical Formulation — Step-by-Step Manual vs PyTorch nn.BatchNorm1d
# ===========================================================================
print("=" * 70)
print("PART 2: Mathematical Formulation of Batch Normalization")
print("=" * 70)
print("""
Given a mini-batch B = {x_1, x_2, ..., x_m} of size m for feature channel j:

1. Mini-batch Mean:
     mu_B = (1 / m) * sum_{i=1}^m x_i

2. Mini-batch Variance:
     sigma_B^2 = (1 / m) * sum_{i=1}^m (x_i - mu_B)^2

3. Normalization (zero mean, unit variance):
     x_hat_i = (x_i - mu_B) / sqrt(sigma_B^2 + eps)
     where eps is a small numerical stability constant (e.g. 1e-5)

4. Scale and Shift (Learnable Affine Transformation):
     y_i = gamma * x_hat_i + beta

Why gamma and beta?
If identity mapping is optimal, the network can learn:
     gamma = sqrt(sigma_B^2 + eps),  beta = mu_B
This guarantees that BatchNorm NEVER constrains what the network can express!
""")

# Let's verify manual formulation against PyTorch nn.BatchNorm1d
batch_size, num_features = 5, 3
x_sample = torch.randn(batch_size, num_features)

# Manual calculation
eps = 1e-5
mu_manual = x_sample.mean(dim=0, keepdim=True)
# PyTorch uses biased variance (1/m) during training normalization
var_manual = ((x_sample - mu_manual) ** 2).mean(dim=0, keepdim=True)
x_hat_manual = (x_sample - mu_manual) / torch.sqrt(var_manual + eps)

gamma = torch.ones(1, num_features)
beta = torch.zeros(1, num_features)
y_manual = gamma * x_hat_manual + beta

# PyTorch official module
bn_module = nn.BatchNorm1d(num_features, eps=eps, momentum=0.1)
bn_module.train()  # Training mode
y_pytorch = bn_module(x_sample)

diff = (y_manual - y_pytorch).abs().max().item()
print(f"Sample input shape: {list(x_sample.shape)} (batch_size={batch_size}, features={num_features})")
print(f"Manual normalized mean: {y_manual.mean(dim=0).detach().numpy().round(4)}")
print(f"Manual normalized std:  {y_manual.std(dim=0, unbiased=False).detach().numpy().round(4)}")
print(f"Max absolute difference between Manual and nn.BatchNorm1d: {diff:.8e}")
assert diff < 1e-5, "Manual BatchNorm does not match PyTorch implementation!"
print("=> Mathematical formulation verified successfully!\n")

# ===========================================================================
# PART 3: Train vs Eval Modes & Running Statistics (EMA)
# ===========================================================================
print("=" * 70)
print("PART 3: Train Mode vs Eval Mode & Running Statistics")
print("=" * 70)
print("""
Crucial Operational Difference:
- During TRAINING (model.train()):
    mu_B and sigma_B^2 are computed directly from the current mini-batch.
    Simultaneously, running estimates are updated via Exponential Moving Average (EMA):
        running_mean = (1 - momentum) * running_mean + momentum * mu_B
        running_var  = (1 - momentum) * running_var  + momentum * sigma_B_unbiased^2

- During INFERENCE / EVALUATION (model.eval()):
    We must be able to make deterministic predictions for single samples or arbitrary batches!
    Therefore, BatchNorm FREEZES running statistics and uses:
        x_norm = (x - running_mean) / sqrt(running_var + eps)
        y = gamma * x_norm + beta
""")

# Demonstrate running statistics accumulation
bn_demo = nn.BatchNorm1d(num_features=2, momentum=0.1)
bn_demo.train()
print(f"Initial running mean: {bn_demo.running_mean.numpy()}")
print(f"Initial running var:  {bn_demo.running_var.numpy()}")

for step in range(5):
    # Pass mini-batches with mean [10.0, -5.0] and std [2.0, 3.0]
    batch_data = torch.randn(32, 2) * torch.tensor([2.0, 3.0]) + torch.tensor([10.0, -5.0])
    _ = bn_demo(batch_data)

print(f"After 5 batches: running mean: {bn_demo.running_mean.numpy().round(3)}")
print(f"                 running var:  {bn_demo.running_var.numpy().round(3)}")

# Switch to eval mode
bn_demo.eval()
single_sample = torch.tensor([[10.0, -5.0]])
out_eval = bn_demo(single_sample)
print(f"Inference on single sample {single_sample.numpy().tolist()} under eval mode:")
print(f"  Normalized output: {out_eval.detach().numpy().round(4)} (close to zero as expected!)\n")

# ===========================================================================
# PART 4: Deep MLP Experiment: With vs Without BatchNorm at High LR
# ===========================================================================
print("=" * 70)
print("PART 4: Comparative Experiment: Deep MLP at High Learning Rate")
print("=" * 70)

# Generate synthetic classification dataset: 1000 samples, 10 features, 2 classes
from sklearn.datasets import make_classification
X_np, y_np = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    flip_y=0.05,
    random_state=42
)
X_tensor = torch.tensor(X_np, dtype=torch.float32)
y_tensor = torch.tensor(y_np, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Deep MLP without BatchNorm (6 layers)
class DeepMLPWithoutBN(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, num_classes=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ])
        self.out = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x, return_activations=False):
        activations = []
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
            if return_activations:
                activations.append(x.detach())
        logits = self.out(x)
        if return_activations:
            return logits, activations
        return logits

# Deep MLP with BatchNorm (6 layers)
class DeepMLPWithBN(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, num_classes=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(5)
        ])
        self.out = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x, return_activations=False):
        activations = []
        for layer, bn in zip(self.layers, self.bns):
            x = layer(x)
            x = bn(x)
            x = self.relu(x)
            if return_activations:
                activations.append(x.detach())
        logits = self.out(x)
        if return_activations:
            return logits, activations
        return logits

# Train both models at a relatively aggressive learning rate (lr = 0.08)
torch.manual_seed(42)
model_no_bn = DeepMLPWithoutBN()
model_bn = DeepMLPWithBN()

criterion = nn.CrossEntropyLoss()
lr = 0.08
epochs = 40

opt_no_bn = optim.SGD(model_no_bn.parameters(), lr=lr)
opt_bn = optim.SGD(model_bn.parameters(), lr=lr)

loss_history_no_bn = []
loss_history_bn = []

print(f"Training 6-layer Deep MLPs for {epochs} epochs at learning rate = {lr}...")

for epoch in range(epochs):
    # Model without BN
    model_no_bn.train()
    running_loss_no_bn = 0.0
    for X_batch, y_batch in train_loader:
        opt_no_bn.zero_grad()
        preds = model_no_bn(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        # Gradient clipping to prevent complete NaN divergence in unnormalized net
        nn.utils.clip_grad_norm_(model_no_bn.parameters(), max_norm=5.0)
        opt_no_bn.step()
        running_loss_no_bn += loss.item() * len(y_batch)
    loss_history_no_bn.append(running_loss_no_bn / len(dataset))
    
    # Model with BN
    model_bn.train()
    running_loss_bn = 0.0
    for X_batch, y_batch in train_loader:
        opt_bn.zero_grad()
        preds = model_bn(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        opt_bn.step()
        running_loss_bn += loss.item() * len(y_batch)
    loss_history_bn.append(running_loss_bn / len(dataset))
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/{epochs} | Without BN Loss: {loss_history_no_bn[-1]:.4f} | With BN Loss: {loss_history_bn[-1]:.4f}")

# Extract deep layer activations (Layer 5) for evaluation batch
model_no_bn.eval()
model_bn.eval()
with torch.no_grad():
    _, acts_no_bn = model_no_bn(X_tensor[:200], return_activations=True)
    _, acts_bn = model_bn(X_tensor[:200], return_activations=True)

deep_act_no_bn = acts_no_bn[-1].cpu().numpy().flatten()
deep_act_bn = acts_bn[-1].cpu().numpy().flatten()

print(f"\nLayer 5 Activation Stats:")
print(f"  Without BN: mean = {deep_act_no_bn.mean():.4f}, std = {deep_act_no_bn.std():.4f}, zeros = {(deep_act_no_bn == 0).mean()*100:.1f}%")
print(f"  With BN:    mean = {deep_act_bn.mean():.4f}, std = {deep_act_bn.std():.4f}, zeros = {(deep_act_bn == 0).mean()*100:.1f}%")

# ===========================================================================
# PART 5: Visualizing Loss Curves & Activation Distributions
# ===========================================================================
print("\n" + "=" * 70)
print("PART 5: Saving Visualization to output/batchnorm_effect.png")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Loss curves
ax1.plot(range(1, epochs + 1), loss_history_no_bn, 'r--', label='Without BatchNorm (SGD lr=0.08)', linewidth=2)
ax1.plot(range(1, epochs + 1), loss_history_bn, 'b-', label='With BatchNorm1d (SGD lr=0.08)', linewidth=2)
ax1.set_title("Training Loss: Deep MLP (6 Layers)", fontsize=13, fontweight='bold')
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("CrossEntropy Loss", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Subplot 2: Activation distributions at Layer 5
bins = np.linspace(0, 4, 40)
ax2.hist(deep_act_no_bn[deep_act_no_bn > 0], bins=bins, color='red', alpha=0.5, density=True, label='Without BatchNorm')
ax2.hist(deep_act_bn[deep_act_bn > 0], bins=bins, color='blue', alpha=0.5, density=True, label='With BatchNorm1d')
ax2.set_title("Layer 5 Post-ReLU Non-Zero Activations", fontsize=13, fontweight='bold')
ax2.set_xlabel("Activation Magnitude", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
fig_path = OUTPUT_DIR / "batchnorm_effect.png"
plt.savefig(fig_path, dpi=120)
plt.close()

print(f"✅ Saved plot to: {fig_path}")

# ===========================================================================
# PART 6: Summary & Best Practices
# ===========================================================================
print("\n" + "=" * 70)
print("BATCH NORMALIZATION — SUMMARY & PRACTICAL CHECKLIST")
print("=" * 70)
print("""
Key Practical Guidelines:
1. Placement: Apply BatchNorm AFTER Linear/Conv and BEFORE the nonlinear activation:
     Linear -> BatchNorm1d -> ReLU -> Dropout
2. Bias parameter: When using BatchNorm immediately after Linear/Conv, set `bias=False`:
     nn.Linear(in_dim, out_dim, bias=False)
     BatchNorm's beta parameter already provides the necessary shift!
3. Modes: ALWAYS switch between modes:
     `model.train()` during training (updates running stats).
     `model.eval()` during validation/testing/deployment (uses frozen running stats).
4. Batch Size: Ensure batch size is >= 16 (preferably >= 32). Very small batches
     produce noisy batch statistics and harm training stability.
""")
