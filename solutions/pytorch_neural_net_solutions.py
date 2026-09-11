"""
Solutions: PyTorch and Neural Networks (Key Exercises)
=======================================================
Reference solutions for the most important / challenging exercises
across modules 7-11. Study AFTER genuinely attempting each exercise.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import copy

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True, parents=True)
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("SOLUTIONS: PYTORCH & NEURAL NETWORKS")
print("=" * 60)

# ===========================================================================
# SOLUTION A: Fixed Training Loop (Module 7/8 debugging exercise)
# ===========================================================================
print("\n--- SOLUTION A: Corrected Training Loop ---")

X = torch.randn(200, 6)
y = (X[:, 0] + X[:, 1] > 0).float()

model = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1))
criterion = nn.BCEWithLogitsLoss()      # FIX 1: BCEWithLogits for binary, no sigmoid in model
optimizer = optim.Adam(model.parameters(), lr=0.01)
loader = DataLoader(TensorDataset(X, y.unsqueeze(1)), batch_size=32)

for epoch in range(30):
    model.train()                        # FIX 2: set training mode
    for Xb, yb in loader:
        y_pred = model(Xb)
        loss = criterion(y_pred, yb)
        optimizer.zero_grad()            # FIX 3: zero grads before backward!
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():                    # FIX 4: no_grad for evaluation
    preds = (torch.sigmoid(model(X)) > 0.5).float()
    acc = (preds == y.unsqueeze(1)).float().mean()
print(f"Fixed training loop accuracy: {acc.item():.4f}")

# ===========================================================================
# SOLUTION B: Custom Dataset From CSV (Module 7 challenge)
# ===========================================================================
print("\n--- SOLUTION B: Custom Dataset ---")

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class IndustrialSensorDataset(Dataset):
    """Load industrial sensor data from CSV."""
    
    def __init__(self, csv_path, scaler=None, training=True, noise_std=0.05):
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        feature_cols = ['temperature', 'vibration', 'pressure', 'current',
                        'rpm', 'humidity', 'voltage', 'acoustic_emission']
        
        X = df[feature_cols].values.astype(np.float32)
        y = df['fault_severity'].values.astype(np.int64)
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        # Scale (fit on train, apply to both)
        if scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X).astype(np.float32)
        else:
            self.scaler = scaler
            X = self.scaler.transform(X).astype(np.float32)
        
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.training = training
        self.noise_std = noise_std
    
    def __len__(self): return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.training and self.noise_std > 0:
            x += torch.randn_like(x) * self.noise_std  # data augmentation
        return x, self.y[idx]

# Try to load the dataset if it exists
try:
    DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"
    train_ds = IndustrialSensorDataset(DATA_DIR / "industrial_sensor_train.csv", training=True)
    test_ds  = IndustrialSensorDataset(DATA_DIR / "industrial_sensor_test.csv",
                                        scaler=train_ds.scaler, training=False)
    print(f"Dataset loaded: {len(train_ds)} train, {len(test_ds)} test")
    
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    Xb, yb = next(iter(loader))
    print(f"Batch: X={Xb.shape}, y={yb.shape}")
except FileNotFoundError:
    print("Dataset not found. Run 12_capstone/generate_capstone_data.py first.")

# ===========================================================================
# SOLUTION C: TwoLayerNet from scratch (Module 9 challenge)
# ===========================================================================
print("\n--- SOLUTION C: Neural Net from Scratch ---")

class TwoLayerNetSolution:
    """2-hidden-layer net using only tensors and autograd."""
    
    def __init__(self, input_dim, h1, h2, output_dim, lr=0.01):
        self.lr = lr
        
        # Xavier uniform initialization
        def xavier(fan_in, fan_out):
            std = math.sqrt(2.0 / (fan_in + fan_out))
            return torch.randn(fan_in, fan_out) * std
        
        self.W1 = xavier(input_dim, h1).requires_grad_(True)
        self.b1 = torch.zeros(h1, requires_grad=True)
        self.W2 = xavier(h1, h2).requires_grad_(True)
        self.b2 = torch.zeros(h2, requires_grad=True)
        self.W3 = xavier(h2, output_dim).requires_grad_(True)
        self.b3 = torch.zeros(output_dim, requires_grad=True)
    
    def forward(self, x):
        self.z1 = torch.relu(x @ self.W1 + self.b1)
        self.z2 = torch.relu(self.z1 @ self.W2 + self.b2)
        return self.z2 @ self.W3 + self.b3
    
    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
    
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None: p.grad.zero_()

def cross_entropy_manual(logits, y):
    """Manual cross-entropy loss."""
    n = logits.shape[0]
    # Numerically stable softmax
    logits = logits - logits.max(dim=1, keepdim=True).values
    log_probs = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True))
    return -log_probs[range(n), y].mean()

# Quick test
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_np, y_np = make_classification(n_samples=500, n_features=10, n_classes=3,
                                   n_informative=8, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
sc = StandardScaler()
X_tr_s = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32)
X_te_s  = torch.tensor(sc.transform(X_te), dtype=torch.float32)
y_tr_t  = torch.tensor(y_tr, dtype=torch.long)
y_te_t  = torch.tensor(y_te, dtype=torch.long)

net = TwoLayerNetSolution(10, 64, 32, 3, lr=0.01)
loader = DataLoader(TensorDataset(X_tr_s, y_tr_t), batch_size=32, shuffle=True)

for epoch in range(100):
    for Xb, yb in loader:
        logits = net.forward(Xb)
        loss = cross_entropy_manual(logits, yb)
        net.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in net.parameters():
                if p.grad is not None:
                    p -= net.lr * p.grad

net.zero_grad()
with torch.no_grad():
    preds = net.forward(X_te_s).argmax(dim=1)
    acc = (preds == y_te_t).float().mean().item()
print(f"TwoLayerNet scratch accuracy: {acc:.4f}")

# ===========================================================================
# SOLUTION D: MultiHead Attention from Scratch (Module 11 challenge)
# ===========================================================================
print("\n--- SOLUTION D: Scaled Dot-Product Attention ---")

def scaled_dot_product_attention_solution(Q, K, V, mask=None):
    """
    Q, K, V: (batch, n_heads, seq, d_k)
    Returns output (same shape) and weights (batch, n_heads, seq, seq)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    
    weights = torch.softmax(scores, dim=-1)
    output  = weights @ V
    return output, weights

class MultiHeadAttentionSolution(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, Q_in, K_in, V_in, mask=None):
        B, T, d = Q_in.shape
        
        # 1. Project
        Q = self.W_q(Q_in).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        K = self.W_k(K_in).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V_in).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Attention
        output, _ = scaled_dot_product_attention_solution(Q, K, V, mask)
        
        # 3. Concatenate heads
        output = output.transpose(1, 2).contiguous().view(B, T, d)
        
        # 4. Output projection
        return self.W_o(output)

# Test
d, heads = 32, 4
mha_sol = MultiHeadAttentionSolution(d, heads)
X_t = torch.randn(2, 8, d)
with torch.no_grad():
    out_sol = mha_sol(X_t, X_t, X_t)
print(f"MultiHeadAttention scratch output: {out_sol.shape}  (should be (2, 8, {d}))")

# Demonstrate causal mask
T = 8
causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
with torch.no_grad():
    out_causal, weights_causal = scaled_dot_product_attention_solution(
        X_t[:, :T, :d//heads].unsqueeze(1),  # dummy Q
        X_t[:, :T, :d//heads].unsqueeze(1),  # dummy K
        X_t[:, :T, :d//heads].unsqueeze(1),  # dummy V
        mask=causal_mask
    )
print(f"\nCausal attention weights (should be lower-triangular):")
print(weights_causal[0, 0].round(decimals=3))

print("""
===========================================
SOLUTIONS SUMMARY
===========================================
A. Training loop: BCEWithLogitsLoss + model.train() + zero_grad() + no_grad()
B. Custom Dataset: pandas + imputation + StandardScaler + noise augmentation
C. Two-layer net: Xavier init + forward + manual cross-entropy + SGD
D. MHA from scratch: Q/K/V projections + reshape to heads + attention + concat + W_O
""")
