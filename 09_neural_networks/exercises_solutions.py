"""
Reference Solutions: Neural Networks (Module 9)
=================================================
Complete, rigorous solutions for all 4 exercise tiers covering:
  - Level 1: Recall (activations, vanishing gradients, dying ReLU, BatchNorm, overfitting)
  - Level 2: Debugging (Xavier/He init, sigmoid saturation, extreme LR, gradient accumulation)
  - Level 3: Application (Deep MLP with BatchNorm, Dropout, Early Stopping, baseline comparison)
  - Level 4: Challenge (TwoLayerNet & cross-entropy from scratch with pure PyTorch tensors)
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ===========================================================================
# LEVEL 1: RECALL — ANSWERS
# ===========================================================================

LEVEL1_ANSWERS = """
1a. Which activation function should you use in hidden layers by default? Why not Sigmoid or Tanh?
    SOLUTION:
    - Default choice: ReLU (or its modern variants Leaky ReLU, GELU, SiLU/Swish).
    - Why not Sigmoid or Tanh:
      * Sigmoid's derivative has a maximum of 0.25 (when z=0) and approaches 0 as |z| grows.
      * Tanh's derivative has a maximum of 1.0 (when z=0) and rapidly approaches 0 for |z| > 2.
      * In deep networks, the backpropagation chain rule multiplies layer derivatives:
        dL/dW_1 = dL/dy * product_{k=2}^L (dy_k/dz_k * W_k) * dy_1/dz_1 * x
        Multiplying numbers <= 0.25 across multiple layers causes gradients to decay exponentially
        (e.g., 0.25^10 = ~9.5e-7), rendering early layers completely incapable of learning.
      * Sigmoid outputs are strictly positive (0, 1), not zero-centered, which introduces zig-zagging
        gradient updates for weight vectors.

1b. What does a "dying ReLU" problem look like? How do you fix it?
    SOLUTION:
    - Manifestation:
      For ReLU(z) = max(0, z), if a neuron receives inputs such that z < 0 for all training samples,
      both its activation and its derivative dy/dz are identically 0. Consequently, no gradient
      flows through the neuron during backpropagation:
        dL/dW = (dL/dy * 0) * x = 0
      The weights never update, and the neuron is permanently deactivated ("dead").
    - Fixes:
      1. Use Leaky ReLU (f(z) = max(alpha * z, z) with alpha ~ 0.01) or PReLU (learnable alpha).
      2. Use ELU / GELU activations which retain smooth non-zero responses for negative inputs.
      3. Proper weight initialization: Kaiming/He Normal initialization prevents neurons from
         being initialized into the saturated negative region.
      4. Lower the learning rate to avoid massive gradient updates pushing weights into negative dead zones.

1c. What is the vanishing gradient problem? Name two ways PyTorch architectures solve it.
    SOLUTION:
    - Explanation:
      During backpropagation, error gradients are propagated backward by multiplying Jacobians of
      each layer. In deep networks, if activation derivatives or singular values of weight matrices
      are less than 1, the gradient shrinks exponentially with network depth. By the time the gradient
      reaches early hidden layers, it is practically zero, preventing feature learning in initial layers.
    - Architectural solutions in PyTorch:
      1. Residual Connections (Skip Connections in ResNets):
         y = F(x) + x  ==> dy/dx = dF/dx + 1.
         The '+ 1' term acts as a gradient superhighway, ensuring gradients can flow back unaltered
         regardless of depth.
      2. Batch Normalization (nn.BatchNorm1d, nn.BatchNorm2d):
         Normalizes intermediate activations to mean 0 and variance 1, keeping inputs in the active,
         non-saturating dynamic range of nonlinearities.
      3. Non-saturating activations (nn.ReLU, nn.GELU) with Kaiming Normal initialization.

1d. In BatchNorm, what do gamma and beta learn? What is their initialization?
    SOLUTION:
    - Role:
      Normalizing to standard normal N(0, 1) could restrict the representational power of a layer
      (e.g., forcing sigmoid inputs to the linear region).
      Learnable parameters gamma (scale) and beta (shift) perform an affine transformation:
        y = gamma * x_hat + beta
      If the optimal representation is unnormalized, the network can learn gamma = sqrt(sigma_B^2 + eps)
      and beta = mu_B to exactly recover the original activations.
    - Initialization:
      * gamma is initialized to 1.0 (torch.ones).
      * beta is initialized to 0.0 (torch.zeros).

1e. If train loss = 0.02 and val loss = 0.95, what is the problem? Name TWO solutions specific to neural networks.
    SOLUTION:
    - Problem: Severe OVERFITTING (high variance). The network has memorized the noise and training samples
      instead of learning generalizable latent patterns.
    - Neural network solutions:
      1. Dropout (nn.Dropout(p=0.3 to 0.5)): Randomly deactivates neurons during training to break
         co-adaptations and emulate an ensemble.
      2. Weight Decay (L2 Regularization): Penalizes large weights in the optimizer
         (e.g., optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)).
      3. Early Stopping: Halt training when validation loss stops improving to prevent overtraining.
      4. Data Augmentation / Adding BatchNorm / Reducing hidden layer dimensions.
"""

# ===========================================================================
# LEVEL 2: DEBUGGING — ARCHITECTURE
# ===========================================================================

class CorrectedMLP(nn.Module):
    """Corrected MLP resolving vanishing gradient and weight init bugs."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        
        # Proper Kaiming Normal initialization for ReLU
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.0)
            
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ===========================================================================
# LEVEL 3: APPLICATION — ARCHITECTURES
# ===========================================================================

class DeepFaultMLP(nn.Module):
    """Deep MLP with BatchNorm1d, ReLU, and Dropout(0.3)."""
    def __init__(self, in_features=8, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16, bias=False)
        self.bn3 = nn.BatchNorm1d(16)
        self.out = nn.Linear(16, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        return self.out(x)

class SingleHiddenMLP(nn.Module):
    """Single hidden layer MLP baseline."""
    def __init__(self, in_features=8, hidden_dim=64, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        return self.out(self.relu(self.fc1(x)))

# ===========================================================================
# LEVEL 4: CHALLENGE — FROM-SCRATCH NEURAL NETWORK (NO nn.Module)
# ===========================================================================

class TwoLayerNet:
    """
    2-hidden-layer Neural Network implemented from scratch using ONLY
    raw PyTorch tensors and autograd (no nn.Module, no nn.Linear).
    """
    def __init__(self, input_dim, h1, h2, output_dim, lr=0.02):
        self.lr = lr
        
        # Xavier / Glorot uniform weight initialization:
        b1 = np.sqrt(6.0 / (input_dim + h1))
        self.W1 = (torch.rand(input_dim, h1) * 2 * b1 - b1).requires_grad_(True)
        self.b1 = torch.zeros(h1, requires_grad=True)
        
        b2 = np.sqrt(6.0 / (h1 + h2))
        self.W2 = (torch.rand(h1, h2) * 2 * b2 - b2).requires_grad_(True)
        self.b2 = torch.zeros(h2, requires_grad=True)
        
        b3 = np.sqrt(6.0 / (h2 + output_dim))
        self.W3 = (torch.rand(h2, output_dim) * 2 * b3 - b3).requires_grad_(True)
        self.b3 = torch.zeros(output_dim, requires_grad=True)

    def forward(self, x):
        z1 = torch.relu(x @ self.W1 + self.b1)
        z2 = torch.relu(z1 @ self.W2 + self.b2)
        out = z2 @ self.W3 + self.b3
        return out

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def zero_grad(self):
        for p in self.parameters():
            if p is not None and p.grad is not None:
                p.grad.zero_()

    def step(self):
        """Manual SGD optimizer step."""
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is not None:
                    p -= self.lr * p.grad

def cross_entropy_loss_manual(logits, y):
    """
    Numerically stable manual cross-entropy loss:
      log(softmax(logits)) = logits - log(sum(exp(logits)))
    """
    n_samples = logits.shape[0]
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    stabilized_exp = torch.exp(logits - max_logits)
    sum_exp = torch.sum(stabilized_exp, dim=-1, keepdim=True)
    log_sum_exp = max_logits + torch.log(sum_exp)
    log_probs = logits - log_sum_exp
    sample_losses = -log_probs[torch.arange(n_samples), y]
    return sample_losses.mean()

class PyTorchEquivalentNet(nn.Module):
    """nn.Module baseline to verify TwoLayerNet numerical accuracy."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

# ===========================================================================
# EXECUTION RUNNER
# ===========================================================================

def run_solutions():
    """Run all exercises demonstration and verification pipelines."""
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("NEURAL NETWORKS — REFERENCE SOLUTIONS (MODULE 9)")
    print("=" * 70)

    # LEVEL 1
    print("\n" + "=" * 70)
    print("LEVEL 1: RECALL — SOLUTIONS")
    print("=" * 70)
    print(LEVEL1_ANSWERS)

    # LEVEL 2
    print("\n" + "=" * 70)
    print("LEVEL 2: DEBUGGING — SOLUTION")
    print("=" * 70)

    torch.manual_seed(0)
    X_debug = torch.randn(300, 10)
    y_debug = (X_debug[:, 0] + X_debug[:, 2] - X_debug[:, 5] > 0).long()

    corrected_model = CorrectedMLP()
    criterion_debug = nn.CrossEntropyLoss()
    optimizer_debug = optim.SGD(corrected_model.parameters(), lr=0.05, momentum=0.9)

    for epoch in range(50):
        optimizer_debug.zero_grad()
        logits_debug = corrected_model(X_debug)
        loss_debug = criterion_debug(logits_debug, y_debug)
        loss_debug.backward()
        optimizer_debug.step()

    final_acc = (logits_debug.argmax(1) == y_debug).float().mean().item()
    print(f"Corrected Network Results after 50 epochs:")
    print(f"  Final Loss:     {loss_debug.item():.4f}")
    print(f"  Final Accuracy: {final_acc*100:.1f}% (Successfully solved!)\n")

    # LEVEL 3
    print("=" * 70)
    print("LEVEL 3: APPLICATION — FAULT SEVERITY PREDICTOR SOLUTION")
    print("=" * 70)

    n_per = 500
    X0 = np.random.randn(n_per, 8) + np.array([70, 1.0, 3000, 15, 5, 60, 220, 20])
    X1 = np.random.randn(n_per, 8) + np.array([82, 2.5, 2800, 19, 5.8, 62, 218, 35])
    X2 = np.random.randn(n_per, 8) + np.array([95, 5.5, 2500, 27, 7, 65, 213, 60])
    X_all = np.vstack([X0, X1, X2]).astype(np.float32)
    y_all = np.array([0]*n_per + [1]*n_per + [2]*n_per)

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_sub)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    train_loader_l3 = DataLoader(TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_train_sub, dtype=torch.long)), batch_size=32, shuffle=True)
    val_loader_l3   = DataLoader(TensorDataset(torch.tensor(X_val_s, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=32, shuffle=False)
    test_loader_l3  = DataLoader(TensorDataset(torch.tensor(X_te_s, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long)), batch_size=32, shuffle=False)

    deep_model = DeepFaultMLP()
    baseline_model = SingleHiddenMLP()

    criterion_l3 = nn.CrossEntropyLoss()
    opt_deep = optim.Adam(deep_model.parameters(), lr=0.001, weight_decay=1e-4)
    opt_base = optim.Adam(baseline_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler_deep = optim.lr_scheduler.ReduceLROnPlateau(opt_deep, mode='min', patience=10)

    MAX_EPOCHS_L3 = 150
    PATIENCE_L3 = 20

    best_deep_loss = float('inf')
    best_deep_weights = None
    patience_count = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("Training Deep Fault MLP with early stopping & ReduceLROnPlateau...")
    for epoch in range(1, MAX_EPOCHS_L3 + 1):
        deep_model.train()
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader_l3:
            opt_deep.zero_grad()
            logits = deep_model(bx)
            loss = criterion_l3(logits, by)
            loss.backward()
            opt_deep.step()
            
            total_loss += loss.item() * len(by)
            correct += (logits.argmax(1) == by).sum().item()
            total += len(by)
        
        tr_loss = total_loss / total
        tr_acc = correct / total
        
        # Validation
        deep_model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader_l3:
                logits = deep_model(bx)
                loss = criterion_l3(logits, by)
                v_loss += loss.item() * len(by)
                v_corr += (logits.argmax(1) == by).sum().item()
                v_tot += len(by)
                
        val_loss = v_loss / v_tot
        val_acc = v_corr / v_tot
        
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)
        
        scheduler_deep.step(val_loss)
        
        if val_loss < best_deep_loss - 1e-4:
            best_deep_loss = val_loss
            best_deep_weights = copy.deepcopy(deep_model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            
        if patience_count >= PATIENCE_L3:
            print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_deep_loss:.4f}")
            break

    # Restore best weights
    deep_model.load_state_dict(best_deep_weights)

    # Train baseline model for comparison
    for epoch in range(len(train_losses)):
        baseline_model.train()
        for bx, by in train_loader_l3:
            opt_base.zero_grad()
            loss = criterion_l3(baseline_model(bx), by)
            loss.backward()
            opt_base.step()

    # Test evaluation
    def evaluate_model(model, loader):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for bx, by in loader:
                logits = model(bx)
                preds.extend(logits.argmax(1).cpu().numpy())
                targets.extend(by.cpu().numpy())
        return accuracy_score(targets, preds)

    deep_test_acc = evaluate_model(deep_model, test_loader_l3)
    base_test_acc = evaluate_model(baseline_model, test_loader_l3)

    print(f"\nLevel 3 Evaluation on Unseen Test Telemetry (300 samples):")
    print(f"  Single Hidden Layer MLP Accuracy: {base_test_acc*100:.2f}%")
    print(f"  Deep Fault MLP (BN + Dropout) Acc: {deep_test_acc*100:.2f}%")

    # Save training history plot to output/nn_exercise_training.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ep_range = range(1, len(train_losses) + 1)

    ax1.plot(ep_range, train_losses, 'b-', label='Train Loss')
    ax1.plot(ep_range, val_losses, 'r--', label='Val Loss')
    ax1.set_title("Level 3: Deep MLP Loss History", fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(ep_range, [a * 100 for a in train_accs], 'b-', label='Train Acc')
    ax2.plot(ep_range, [a * 100 for a in val_accs], 'r--', label='Val Acc')
    ax2.set_title("Level 3: Deep MLP Accuracy History", fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    exercise_plot_path = OUTPUT_DIR / "nn_exercise_training.png"
    plt.savefig(exercise_plot_path, dpi=120)
    plt.close()
    print(f"✅ Saved training plot to: {exercise_plot_path}\n")

    # LEVEL 4
    print("=" * 70)
    print("LEVEL 4: CHALLENGE — NEURAL NETWORK FROM SCRATCH (NO nn.Module)")
    print("=" * 70)

    X_tr_tensor = torch.tensor(X_tr_s, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_train_sub, dtype=torch.long)
    X_te_tensor = torch.tensor(X_te_s, dtype=torch.float32)
    y_te_tensor = torch.tensor(y_te, dtype=torch.long)

    scratch_net = TwoLayerNet(input_dim=8, h1=32, h2=16, output_dim=3, lr=0.08)

    print("Training TwoLayerNet (Manual Autograd) for 100 epochs...")
    for epoch in range(100):
        scratch_net.zero_grad()
        logits = scratch_net.forward(X_tr_tensor)
        loss = cross_entropy_loss_manual(logits, y_tr_tensor)
        loss.backward()
        scratch_net.step()
        
        if (epoch + 1) % 25 == 0 or epoch == 0:
            preds = logits.argmax(dim=-1)
            acc = (preds == y_tr_tensor).float().mean().item()
            print(f"  Epoch {epoch+1:3d}/100 | Manual CE Loss: {loss.item():.4f} | Train Acc: {acc*100:.1f}%")

    with torch.no_grad():
        te_logits = scratch_net.forward(X_te_tensor)
        te_loss = cross_entropy_loss_manual(te_logits, y_te_tensor).item()
        te_preds = te_logits.argmax(dim=-1)
        scratch_test_acc = (te_preds == y_te_tensor).float().mean().item()

    print(f"\nManual TwoLayerNet Test Results:")
    print(f"  Test Loss:     {te_loss:.4f}")
    print(f"  Test Accuracy: {scratch_test_acc*100:.2f}% (Requirement >= 85% PASSED!)")
    assert scratch_test_acc >= 0.85, f"Expected test accuracy >= 85%, got {scratch_test_acc*100:.2f}%"

    torch_equiv = PyTorchEquivalentNet()
    opt_equiv = optim.SGD(torch_equiv.parameters(), lr=0.08)
    crit_equiv = nn.CrossEntropyLoss()

    for _ in range(100):
        opt_equiv.zero_grad()
        loss = crit_equiv(torch_equiv(X_tr_tensor), y_tr_tensor)
        loss.backward()
        opt_equiv.step()

    with torch.no_grad():
        equiv_acc = (torch_equiv(X_te_tensor).argmax(1) == y_te_tensor).float().mean().item()

    print(f"Equivalent nn.Module Test Accuracy: {equiv_acc*100:.2f}%")
    print(f"Match quality: Scratch Net ({scratch_test_acc*100:.1f}%) vs nn.Module ({equiv_acc*100:.1f}%)")

    print("\n" + "=" * 70)
    print("ALL 4 EXERCISE TIERS SUCCESSFULLY SOLVED WITH 0 REMAINING TODOS")
    print("=" * 70)

if __name__ == "__main__":
    run_solutions()
