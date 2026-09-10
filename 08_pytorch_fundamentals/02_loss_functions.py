"""
02_loss_functions.py — Loss Functions: What to Minimise

WHAT THIS LESSON TEACHES:
    A loss function translates "how wrong is the model?" into a number that
    gradient descent can minimise. Choosing the WRONG loss ruins training
    even if your model architecture is perfect.

WHY IT MATTERS:
    The loss function IS the objective. It defines what "good" means to
    your model. MSELoss trained on classification? It will partially work
    but never give calibrated probabilities. CrossEntropyLoss on regression?
    Disaster. Getting this right is step zero.

LESSON STRUCTURE:
    1. Regression losses: MSE, MAE, Huber
    2. Classification losses: BCE, BCEWithLogits, CrossEntropy
    3. Why the right loss matters (experiment)
    4. Custom loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Regression Losses
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: Regression Losses")
print("=" * 60)

# Synthetic regression data: predicting house prices
y_true = torch.tensor([200.0, 250.0, 180.0, 300.0, 150.0])  # actual prices (£k)
y_pred = torch.tensor([210.0, 240.0, 185.0, 500.0, 155.0])  # model predictions

print(f"True values:      {y_true.tolist()}")
print(f"Predicted values: {y_pred.tolist()}")
print(f"Note: one large outlier at index 3 (500 vs 300)")

# ── 1a. MSE Loss (Mean Squared Error / L2 Loss) ──────────────────────────────
# Formula: loss = mean((y_pred - y_true)²)
# WHY: Squaring errors makes larger errors MUCH more expensive.
# CONSEQUENCE: A single outlier dominates the loss.
mse = nn.MSELoss()
mse_val = mse(y_pred, y_true)
print(f"\nMSE Loss:   {mse_val.item():.2f}")
print(f"  Manual:   {((y_pred - y_true)**2).mean().item():.2f}")
print(f"  WHY: The outlier (error=200) contributes 200²=40000 to the sum!")
print(f"  BEST FOR: When outliers are rare AND you care about all errors equally")

# ── 1b. L1 Loss (Mean Absolute Error / MAE) ──────────────────────────────────
# Formula: loss = mean(|y_pred - y_true|)
# WHY: Absolute value treats all errors proportionally.
# CONSEQUENCE: More robust to outliers.
l1 = nn.L1Loss()
l1_val = l1(y_pred, y_true)
print(f"\nL1 Loss (MAE): {l1_val.item():.2f}")
print(f"  Manual:      {(y_pred - y_true).abs().mean().item():.2f}")
print(f"  WHY: Outlier contributes only 200 (not 40000). Far less dominant.")
print(f"  BEST FOR: Regression with noisy targets / sensor data")

# ── 1c. Huber Loss ───────────────────────────────────────────────────────────
# Formula: L2 for small errors, L1 for large errors (threshold = delta)
# WHY: Best of both worlds — smooth gradients near zero, robust to outliers
huber = nn.HuberLoss(delta=50.0)
huber_val = huber(y_pred, y_true)
print(f"\nHuber Loss (delta=50): {huber_val.item():.2f}")
print(f"  WHY: Quadratic within ±50 (sensitive to small errors)")
print(f"       Linear beyond ±50 (not dominated by the outlier)")
print(f"  BEST FOR: Most regression tasks — robust + differentiable")

print("\n── Comparison Summary ──")
print(f"  MSE:   {mse_val.item():.1f}  (dominated by outlier)")
print(f"  MAE:   {l1_val.item():.1f}  (robust to outlier)")
print(f"  Huber: {huber_val.item():.1f}  (balanced)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Classification Losses
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2: Classification Losses")
print("=" * 60)

# ── 2a. Binary Cross-Entropy: BCELoss ────────────────────────────────────────
# Task: predict whether email is spam (1) or not (0)
# WHY: Measures how well the predicted PROBABILITY matches the true label.
# Formula: -[y*log(p) + (1-y)*log(1-p)]

print("\n── Binary Classification (spam detection) ──")
y_binary = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])  # true labels

# BCELoss requires inputs to be PROBABILITIES (0 to 1)
# So we must apply sigmoid BEFORE computing loss
logits_binary = torch.tensor([2.0, -1.5, 0.8, 3.0, -0.5])  # raw model output
probs_binary = torch.sigmoid(logits_binary)  # convert to probabilities

bce = nn.BCELoss()
bce_val = bce(probs_binary, y_binary)
print(f"Logits:      {logits_binary.tolist()}")
print(f"Probs:       {[f'{p:.3f}' for p in probs_binary.tolist()]}")
print(f"True labels: {y_binary.int().tolist()}")
print(f"BCELoss:     {bce_val.item():.4f}")

# ── 2b. BCEWithLogitsLoss (PREFERRED for binary classification) ───────────────
# WHY PREFERRED: Combines sigmoid + BCE in one step using the log-sum-exp trick.
# This is NUMERICALLY STABLE (avoids log(0) errors with extreme logits).
# In practice: ALWAYS use BCEWithLogitsLoss instead of sigmoid + BCELoss.
bce_logits = nn.BCEWithLogitsLoss()
bce_logits_val = bce_logits(logits_binary, y_binary)
print(f"\nBCEWithLogitsLoss: {bce_logits_val.item():.4f}")
print(f"BCELoss:           {bce_val.item():.4f}")
print(f"Identical? {torch.isclose(bce_logits_val, bce_val, atol=1e-5).item()}")
print(f"RULE: Use BCEWithLogitsLoss — feed RAW logits, not probabilities!")

# ── 2c. CrossEntropyLoss (multi-class classification) ────────────────────────
# Task: classify digits 0-9 (10 classes)
# WHY: Generalisation of BCE to multiple classes.
# What it computes: softmax(logits) → log → negate → average
# Formula: loss = -log(softmax(logits)[true_class])

print("\n── Multi-Class Classification (3 classes) ──")
# Model outputs: raw logits for each class
logits_multi = torch.tensor([
    [2.0, 0.5, 0.1],   # sample 0: confidently class 0 → correct (label=0)
    [0.1, 3.0, 0.2],   # sample 1: confidently class 1 → correct (label=1)
    [0.5, 0.5, 0.5],   # sample 2: uncertain → wrong? (label=2, barely predicts 2)
    [3.0, 2.0, 0.1],   # sample 3: wrong! (label=2, predicts class 0)
])
labels_multi = torch.tensor([0, 1, 2, 2])  # true class indices (NOT one-hot!)

ce_loss = nn.CrossEntropyLoss()
ce_val = ce_loss(logits_multi, labels_multi)
print(f"CrossEntropyLoss: {ce_val.item():.4f}")

# Show what CrossEntropyLoss actually computes:
probs_multi = torch.softmax(logits_multi, dim=1)
nll = -torch.log(probs_multi[torch.arange(4), labels_multi])
print(f"Manual (softmax + NLL): {nll.mean().item():.4f}")
print(f"  Sample 0 loss: {nll[0].item():.4f} (confident + correct → low loss)")
print(f"  Sample 3 loss: {nll[3].item():.4f} (confident + WRONG → high loss)")

print("\nIMPORTANT RULES for CrossEntropyLoss:")
print("  - Feed RAW LOGITS (not softmax output)")
print("  - Labels must be INTEGER class indices (not one-hot vectors)")
print("  - Your final layer should have shape [batch, num_classes]")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Why the Right Loss Matters
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3: Why the Right Loss Matters")
print("=" * 60)

# Experiment: Train a 2-class classifier with MSE loss vs CrossEntropy loss
# Show that MSE "works" but gives worse calibration

from sklearn.datasets import make_classification
from torch.utils.data import TensorDataset, DataLoader

X_np, y_np = make_classification(n_samples=500, n_features=4, n_classes=2,
                                  n_informative=3, random_state=42)
X = torch.FloatTensor(X_np)
y = torch.LongTensor(y_np)

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x):
        return self.net(x)

def train_with_loss(loss_name: str, num_epochs: int = 30):
    """Train a small network and return final accuracy."""
    net = SmallNet()
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    if loss_name == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_name == "MSE":
        # WHY this is wrong: MSE treats class labels as continuous numbers.
        # It penalises predicting "0.6" for class 1 less than predicting "0.0".
        # But class labels (0, 1) are CATEGORIES, not magnitudes.
        criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = net(X)
        
        if loss_name == "MSE":
            # MSE needs float targets in [0,1] range — very hacky for classification
            y_onehot = torch.zeros(len(y), 2)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            loss = criterion(torch.softmax(logits, dim=1), y_onehot)
        else:
            loss = criterion(logits, y)
        
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        preds = net(X).argmax(dim=1)
        accuracy = (preds == y).float().mean().item()
    return accuracy

acc_ce = train_with_loss("CrossEntropy")
acc_mse = train_with_loss("MSE")

print(f"CrossEntropyLoss accuracy: {acc_ce:.3f} (correct choice)")
print(f"MSE Loss accuracy:         {acc_mse:.3f} (wrong choice)")
print(f"\nConclusion: CrossEntropy is designed for classification.")
print(f"It directly minimises the negative log-likelihood of correct classes.")
print(f"MSE treats labels as magnitudes, not categories — conceptually wrong.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Custom Loss Functions
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4: Custom Loss Functions")
print("=" * 60)

# WHY: Sometimes you need domain-specific loss functions.
# Example: Weighted MSE (penalise underprediction more than overprediction)
# Relevant for: safety-critical forecasting (better to overestimate flood height)

class AsymmetricMSELoss(nn.Module):
    """
    Asymmetric MSE: penalises underpredictions more than overpredictions.

    Use case: flood forecasting — underestimating flood height is much more
    dangerous than overestimating it, so we penalise it more.

    Args:
        under_weight: multiplier for underprediction errors (pred < true)
        over_weight:  multiplier for overprediction errors  (pred > true)
    """

    def __init__(self, under_weight: float = 3.0, over_weight: float = 1.0):
        super().__init__()
        self.under_weight = under_weight
        self.over_weight = over_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        errors = y_pred - y_true   # positive = overpredict, negative = underpredict

        # WHY torch.where: apply different weights based on sign of error
        weights = torch.where(
            errors < 0,                        # if underpredicting
            torch.tensor(self.under_weight),   # apply heavy penalty
            torch.tensor(self.over_weight),    # else light penalty
        )
        return (weights * errors ** 2).mean()

# Demonstrate the asymmetric loss
asym_loss = AsymmetricMSELoss(under_weight=3.0, over_weight=1.0)
standard_mse = nn.MSELoss()

y_true_flood = torch.tensor([10.0, 8.0, 12.0])
y_under = torch.tensor([8.0, 6.0, 10.0])   # always under by 2
y_over  = torch.tensor([12.0, 10.0, 14.0]) # always over by 2

print("Flood height prediction (underprediction is dangerous):")
print(f"  Standard MSE — under:  {standard_mse(y_under, y_true_flood).item():.2f}")
print(f"  Standard MSE — over:   {standard_mse(y_over, y_true_flood).item():.2f}")
print(f"  Asym MSE    — under:   {asym_loss(y_under, y_true_flood).item():.2f}  ← higher penalty!")
print(f"  Asym MSE    — over:    {asym_loss(y_over, y_true_flood).item():.2f}")
print(f"\nThe model trained with AsymmetricMSE will learn to 'prefer'")
print(f"overestimating over underestimating — exactly what we want for safety.")

print("\n✓ 02_loss_functions.py complete")
print("\nLOSS FUNCTION CHEAT SHEET:")
print("  Regression:              MSE (default) | MAE (outliers) | Huber (both)")
print("  Binary classification:   BCEWithLogitsLoss (feed logits!)")
print("  Multi-class:             CrossEntropyLoss (feed logits, integer labels!)")
print("  Custom:                  subclass nn.Module, implement forward()")
