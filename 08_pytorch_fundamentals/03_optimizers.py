"""
03_optimizers.py — Optimizers: How to Minimise the Loss

WHAT THIS LESSON TEACHES:
    An optimizer takes the gradient computed by loss.backward() and decides
    HOW MUCH and in WHAT DIRECTION to move the model's weights.

WHY IT MATTERS:
    Different optimizers have very different convergence properties.
    Using SGD where Adam is appropriate can mean the difference between
    your model converging in 100 epochs vs 10,000 epochs — or not at all.

LESSON STRUCTURE:
    1. Gradient descent review: what does the gradient tell us?
    2. SGD: simple but noisy
    3. SGD with Momentum: build velocity in consistent directions
    4. Adam: adaptive learning rates (most commonly used)
    5. Learning rate schedulers
    6. Experiment: compare optimizers on the same problem
    7. Save optimizer comparison plot
"""

import matplotlib
matplotlib.use('Agg')  # WHY: non-interactive backend — saves PNGs without display
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from pathlib import Path

torch.manual_seed(42)

# Output directory
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: What Does the Gradient Tell Us?
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: What Does the Gradient Tell Us?")
print("=" * 60)

# Simple 1D example: minimise loss = (w - 3)²  → optimum at w = 3
w = torch.tensor(0.0, requires_grad=True)
loss = (w - 3.0) ** 2
loss.backward()

print(f"w = {w.item()}, loss = {loss.item():.2f}")
print(f"gradient ∂loss/∂w = {w.grad.item():.2f}")
print(f"Interpretation: gradient is NEGATIVE of the direction to the optimum.")
print(f"  Moving w in direction -gradient (w → w - lr * grad) reduces loss.")
print(f"  Update: w ← {w.item()} - 0.1 × {w.grad.item():.2f} = {w.item() - 0.1 * w.grad.item():.2f}")
print(f"  (Optimal w = 3.0)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: SGD (Stochastic Gradient Descent)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2: SGD — Simple But Noisy")
print("=" * 60)

# Update rule:   w ← w - lr × gradient
# WHY 'Stochastic': in practice, gradient is computed on a RANDOM MINI-BATCH
# of data (not the full dataset). This is much faster but introduces noise.
#
# Problem: if the loss landscape has steep directions and shallow directions,
# SGD oscillates in steep directions and crawls in shallow ones.

w = torch.tensor(0.0, requires_grad=True)
sgd = torch.optim.SGD([w], lr=0.1)

print(f"Initial w = {w.item():.4f} (target = 3.0)")
for step in range(5):
    sgd.zero_grad()           # ALWAYS clear old gradients first
    loss = (w - 3.0) ** 2
    loss.backward()
    sgd.step()
    print(f"  Step {step+1}: w = {w.item():.4f}, loss = {loss.item():.4f}")

print(f"\nKey hyperparameter: learning_rate (lr)")
print(f"  Too large → overshoots, diverges")
print(f"  Too small → converges very slowly")
print(f"  LR is the most important hyperparameter in deep learning")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SGD with Momentum
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3: SGD with Momentum")
print("=" * 60)

# Update rule:  v ← momentum × v - lr × gradient
#               w ← w + v
# WHY: Momentum acts like a ball rolling down a hill — it builds up speed in
# consistent downhill directions and dampens oscillations in inconsistent ones.
# The 'velocity' v accumulates past gradients.
#
# momentum=0.9 means 90% of the old velocity is kept.

w = torch.tensor(0.0, requires_grad=True)
sgd_momentum = torch.optim.SGD([w], lr=0.1, momentum=0.9)

print(f"Initial w = {w.item():.4f} (target = 3.0)")
for step in range(5):
    sgd_momentum.zero_grad()
    loss = (w - 3.0) ** 2
    loss.backward()
    sgd_momentum.step()
    print(f"  Step {step+1}: w = {w.item():.4f}, loss = {loss.item():.4f}")

print(f"\nNotice: with momentum, we overshoot slightly (ball rolling too fast)")
print(f"but converge faster overall on well-conditioned problems.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Adam (Adaptive Moment Estimation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4: Adam — Adaptive Learning Rates")
print("=" * 60)

# Update rule (simplified):
#   m ← β₁m + (1-β₁)g        (first moment: running mean of gradients)
#   v ← β₂v + (1-β₂)g²       (second moment: running mean of squared gradients)
#   w ← w - lr × m / (√v + ε)
#
# WHY: Each parameter gets its OWN learning rate, scaled by how consistent
# its gradient has been. Parameters with small, consistent gradients get
# larger effective LR; parameters with noisy/large gradients get smaller LR.
#
# WHY MOST POPULAR:
#   - Works well out-of-the-box (lr=1e-3 is a good default)
#   - Handles sparse gradients well (great for NLP)
#   - Less sensitive to LR choice than SGD

w = torch.tensor(0.0, requires_grad=True)
adam = torch.optim.Adam([w], lr=0.1)

print(f"Initial w = {w.item():.4f} (target = 3.0)")
for step in range(5):
    adam.zero_grad()
    loss = (w - 3.0) ** 2
    loss.backward()
    adam.step()
    print(f"  Step {step+1}: w = {w.item():.4f}, loss = {loss.item():.4f}")

print(f"\nDefault Adam hyperparameters (rarely need to change):")
print(f"  lr=1e-3, betas=(0.9, 0.999), eps=1e-8")
print(f"  If training is unstable, first try reducing lr to 1e-4")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Learning Rate Schedulers
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5: Learning Rate Schedulers")
print("=" * 60)

# WHY: Starting with a large LR finds a good region quickly.
# Then decaying LR allows precise convergence within that region.
# Fixed LR: often oscillates around the minimum without converging.

model_for_sched = nn.Linear(4, 1)
optimizer_sched = torch.optim.Adam(model_for_sched.parameters(), lr=0.1)

# ── StepLR: multiply LR by gamma every step_size epochs ──────────────────
step_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer_sched, step_size=10, gamma=0.5
)
# Every 10 epochs: lr = lr × 0.5
lrs_step = []
for epoch in range(40):
    lrs_step.append(optimizer_sched.param_groups[0]['lr'])
    step_scheduler.step()

# Reset LR for next demonstration
for g in optimizer_sched.param_groups:
    g['lr'] = 0.1

# ── CosineAnnealingLR: smooth cosine decay to minimum LR ─────────────────
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_sched, T_max=40, eta_min=0.001
)
# WHY cosine: smooth decay avoids abrupt LR drops that can destabilise training
lrs_cosine = []
for epoch in range(40):
    lrs_cosine.append(optimizer_sched.param_groups[0]['lr'])
    cosine_scheduler.step()

print("Learning Rate Schedules (40 epochs):")
print(f"  StepLR:   start={lrs_step[0]:.3f}, mid={lrs_step[20]:.4f}, end={lrs_step[-1]:.4f}")
print(f"  Cosine:   start={lrs_cosine[0]:.3f}, mid={lrs_cosine[20]:.4f}, end={lrs_cosine[-1]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Optimizer Comparison Experiment
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6: Optimizer Comparison Experiment")
print("=" * 60)

from sklearn.datasets import make_regression
import numpy as np

# Generate a regression problem with noisy features
X_np, y_np = make_regression(n_samples=300, n_features=10, noise=20.0, random_state=42)
X = torch.FloatTensor(X_np)
y = torch.FloatTensor(y_np).unsqueeze(1)

# Normalise inputs and targets
X = (X - X.mean(0)) / (X.std(0) + 1e-8)
y = (y - y.mean()) / (y.std() + 1e-8)


def make_model():
    """Build a fresh model for fair comparison."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


def train_and_record(optimizer_name: str, num_epochs: int = 80) -> list:
    """Train model and record per-epoch loss."""
    model = make_model()
    criterion = nn.MSELoss()

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    elif optimizer_name == "SGD+Momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    elif optimizer_name == "Adam+Cosine":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-4
        )

    losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        if optimizer_name == "Adam+Cosine":
            scheduler.step()
        losses.append(loss.item())

    return losses


optimizers_to_compare = ["SGD", "SGD+Momentum", "Adam", "Adam+Cosine"]
all_losses = {}

for opt_name in optimizers_to_compare:
    losses = train_and_record(opt_name)
    all_losses[opt_name] = losses
    print(f"  {opt_name:15s}: final loss = {losses[-1]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Save Comparison Plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Optimizer Comparison on Regression Task", fontsize=14, fontweight='bold')

colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

# Left: full training curves
ax = axes[0]
for (name, losses), color in zip(all_losses.items(), colors):
    ax.plot(losses, label=name, color=color, linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Full Training Curves")
ax.legend()
ax.grid(True, alpha=0.3)

# Right: zoomed last 40 epochs
ax = axes[1]
for (name, losses), color in zip(all_losses.items(), colors):
    ax.plot(losses[40:], label=name, color=color, linewidth=2)
ax.set_xlabel("Epoch (last 40)")
ax.set_ylabel("MSE Loss")
ax.set_title("Convergence Zoom (Last 40 Epochs)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "optimizer_comparison.png"
plt.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to: {out_path}")

print("\n✓ 03_optimizers.py complete")
print("\nOPTIMIZER GUIDE:")
print("  SGD:           good for CV (images) with careful LR tuning")
print("  SGD+Momentum:  better than plain SGD for most tasks")
print("  Adam:          best default for most tasks (lr=1e-3)")
print("  Adam+Schedule: production choice — warm up then decay")
print("\n  Learning rate: MOST IMPORTANT hyperparameter")
print("  Default Adam lr=1e-3 → try 1e-4 if unstable, 1e-2 if too slow")
