"""
Exercises: Transformers (Module 11)
=====================================
4-tier exercises covering self-attention, transformer architecture,
and applying transformers to sequence data.
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

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("TRANSFORMERS — EXERCISES")
print("=" * 60)

# ===========================================================================
# LEVEL 1: RECALL
# ===========================================================================
print("\n--- LEVEL 1: RECALL ---")
print("""
Answer by studying the concepts from 01_attention_and_transformers.py:

1a. What problem does self-attention solve that RNNs couldn't?
    Give a concrete example.

1b. In Attention(Q, K, V) = softmax(QK^T / √d_k) · V,
    why do we divide by √d_k?

1c. A MultiheadAttention has d_model=128 and num_heads=8.
    What is the dimension of each individual head?
    How many parameters total? (Assume square projection matrices)

1d. Why does the Transformer use LayerNorm instead of BatchNorm?
    What changes when batch size = 1?

1e. What is a CLS (classification) token? How is it used?
    Which token position in the sequence is read for classification?
""")

# Demo: shape of attention
B, T, d = 4, 15, 64
n_heads = 8
mha = nn.MultiheadAttention(d, n_heads, batch_first=True)
X = torch.randn(B, T, d)
with torch.no_grad():
    out, weights = mha(X, X, X)
print(f"MHA input: {X.shape}")
print(f"MHA output: {out.shape}")
print(f"Attention weights: {weights.shape}  (batch, seq, seq)")
print(f"MHA params: {sum(p.numel() for p in mha.parameters()):,}")

# LayerNorm vs BatchNorm
ln = nn.LayerNorm(64)
bn = nn.BatchNorm1d(64)
x_demo = torch.randn(4, 64)
print(f"\nLayerNorm on (4,64): {ln(x_demo).shape}  — normalizes each sample independently")
print(f"BatchNorm1d on (4,64): {bn(x_demo).shape}  — normalizes each feature across the batch")

# ===========================================================================
# LEVEL 2: DEBUGGING
# ===========================================================================
print("\n--- LEVEL 2: DEBUGGING ---")
print("""
Find 3 bugs in this self-attention implementation.
""")

buggy = '''
import torch
import torch.nn as nn
import math

class BuggyAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, d = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k)
        
        # Transpose for attention: (B, n_heads, T, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention scores
        # BUG 1: Missing the √d_k scaling factor
        scores = Q @ K.transpose(-2, -1)
        
        # BUG 2: Missing softmax! Using raw scores as attention weights
        attn_weights = scores
        
        output = attn_weights @ V
        
        # Reassemble heads
        output = output.transpose(1, 2).contiguous().view(B, T, d)
        
        # BUG 3: Missing the final output projection self.W_o
        return output

model = BuggyAttention(d_model=64, n_heads=8)
X = torch.randn(4, 10, 64)
out = model(X)
print(f"Output shape: {out.shape}")
'''
print(buggy)
print("""
Bug 1: ________________________________________________
Bug 2: ________________________________________________
Bug 3: ________________________________________________

TODO: Write the corrected version
""")

# ===========================================================================
# LEVEL 3: APPLICATION — Classify Fault Sequences
# ===========================================================================
print("\n--- LEVEL 3: APPLICATION ---")
print("""
TASK: Use a Transformer to classify machine fault sequences.

Dataset:
  Input: 50 time steps × 6 sensor readings
  Labels: 0=OK (60%), 1=WARNING (25%), 2=FAULT (15%)
  n=1500 samples (imbalanced)

Requirements:
  1. Generate the time-series dataset
     (OK: smooth sinusoids, WARNING: noisy, FAULT: ramp + spikes)
  2. Build a Transformer classifier:
     Linear(6 → 32) → 2× TransformerEncoderBlock(32, 4 heads) → CLS → FC(32→3)
  3. Handle class imbalance: use class_weight in CrossEntropyLoss
  4. Train with Adam (lr=3e-4), CosineAnnealingLR, 60 epochs
  5. Early stopping (patience=15)
  6. Report: test accuracy AND per-class F1 (important with imbalance!)
  7. Visualize attention weights for 3 test samples (one of each class)
     (Save to output/attention_visualization.png)
  8. Compare with an LSTM (nn.LSTM) on the same data:
     Which is more accurate? Which trains faster?
""")

# Data generation
def gen_fault_sequences(n_samples=1500, seq_len=50, n_features=6):
    """Generate fault classification time series."""
    n_ok    = int(0.60 * n_samples)
    n_warn  = int(0.25 * n_samples)
    n_fault = n_samples - n_ok - n_warn
    
    t = np.linspace(0, 1, seq_len)
    X, y = [], []
    
    for _ in range(n_ok):  # OK: smooth
        seq = np.array([np.sin(2*np.pi*t*f + np.random.rand()) + np.random.randn(seq_len)*0.1
                        for f in np.random.uniform(0.5, 2, n_features)]).T
        X.append(seq); y.append(0)
    
    for _ in range(n_warn):  # WARNING: noisy
        seq = np.array([np.sin(2*np.pi*t*f + np.random.rand()) + np.random.randn(seq_len)*0.4
                        for f in np.random.uniform(0.5, 2, n_features)]).T
        X.append(seq); y.append(1)
    
    for _ in range(n_fault):  # FAULT: ramp with spikes
        seq = np.array([t + np.random.randn(seq_len)*0.2
                        + np.where(np.random.random(seq_len) < 0.1, 2.0, 0.0)
                        for _ in range(n_features)]).T
        X.append(seq); y.append(2)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

X_fault, y_fault = gen_fault_sequences()
print(f"Dataset: {X_fault.shape}, class dist: {np.bincount(y_fault)}")
print("TODO: Build the Transformer classifier above")

# ===========================================================================
# LEVEL 4: CHALLENGE — Implement Scaled Dot-Product Attention From Scratch
# ===========================================================================
print("\n--- LEVEL 4: CHALLENGE ---")
print("""
TASK: Implement multi-head self-attention from scratch using only tensors.

def scaled_dot_product_attention(Q, K, V, mask=None):
    '''
    Q, K, V: (batch, n_heads, seq_len, d_k)
    mask:    optional (batch, 1, seq_len, seq_len) bool mask
             True = positions to MASK OUT (set to -inf before softmax)
    
    Returns:
      output:  (batch, n_heads, seq_len, d_k) — attended values
      weights: (batch, n_heads, seq_len, seq_len) — attention weights
    '''
    d_k = Q.shape[-1]
    
    # Step 1: scores = Q @ K^T / sqrt(d_k)
    # Step 2: If mask given, set masked positions to -1e9
    # Step 3: weights = softmax(scores, dim=-1)
    # Step 4: output = weights @ V
    
    return output, weights

class MultiHeadAttentionScratch(nn.Module):
    def __init__(self, d_model, n_heads):
        # W_q, W_k, W_v, W_o: nn.Linear(d_model, d_model)
        # self.d_k = d_model // n_heads
    
    def forward(self, Q, K, V, mask=None):
        B, T, _ = Q.shape
        
        # Step 1: Project Q, K, V
        # Step 2: Reshape to (B, n_heads, T, d_k)
        # Step 3: Apply scaled_dot_product_attention
        # Step 4: Concatenate heads back to (B, T, d_model)
        # Step 5: Apply output projection

Requirements:
  - Your MultiHeadAttentionScratch should produce outputs identical to
    nn.MultiheadAttention (with the same weights)
  - Implement causal masking: the mask argument allows each position
    to attend only to PREVIOUS positions (for autoregressive generation)
  - Test it: show that attention weights for a causal mask are lower-triangular

Bonus: implement rotary positional embeddings (RoPE) instead of
learned positional embeddings — used in LLaMA, Mistral, Gemma.
""")

def scaled_dot_product_attention(Q, K, V, mask=None):
    """TODO: Implement from scratch."""
    d_k = Q.shape[-1]
    # TODO: compute scores, apply mask, softmax, attend
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # starter
    weights = torch.softmax(scores, dim=-1)             # starter (incomplete — no mask)
    output = weights @ V                                # starter
    return output, weights

class MultiHeadAttentionScratch(nn.Module):
    """TODO: Complete this implementation."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, Q_in, K_in, V_in, mask=None):
        B, T, d = Q_in.shape
        # TODO: project, reshape, attend, concatenate, project out
        return self.W_o(Q_in)  # REPLACE with correct implementation

# Test
d, heads = 32, 4
mha_scratch = MultiHeadAttentionScratch(d, heads)
mha_ref     = nn.MultiheadAttention(d, heads, batch_first=True, bias=False)
X_test = torch.randn(2, 8, d)
with torch.no_grad():
    out_ref, _ = mha_ref(X_test, X_test, X_test)
    out_scratch, _ = mha_scratch(X_test, X_test, X_test), None
print(f"Reference MHA output: {out_ref.shape}")
print("TODO: Your scratch implementation should match nn.MultiheadAttention output shape and values")
print("  (After copying weights: mha_scratch.W_q.weight.data = mha_ref.in_proj_weight[:d])")

print("\n" + "=" * 60)
print("EXERCISES COMPLETE")
print("Check solutions/transformer_solutions.py for reference answers.")
print("=" * 60)
