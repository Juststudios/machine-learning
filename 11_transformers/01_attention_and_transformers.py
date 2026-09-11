"""
Transformers: Attention Is All You Need
=========================================
The Transformer (Vaswani et al., 2017) replaced RNNs for sequence tasks
and became the foundation of:
  - GPT (text generation)
  - BERT (text understanding)
  - Vision Transformers / ViT (image classification)
  - Whisper (speech recognition)
  - AlphaFold2 (protein structure prediction)

THE KEY IDEA: Self-Attention
  Instead of processing sequences left-to-right (RNN),
  every element in a sequence can DIRECTLY attend to EVERY other element.
  
  This solves the RNN's problem: distant dependencies.
  "The cat that sat on the mat, which was blue, was happy."
  → "was happy" refers to "cat", 9 words away. RNN forgets. Transformer doesn't.

This lesson:
  1. Intuition for self-attention
  2. The math: Q, K, V matrices
  3. Multi-head attention
  4. Transformer encoder block
  5. Sequence classification with PyTorch nn.Transformer
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
import math
import copy

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

torch.manual_seed(42)
np.random.seed(42)

# ===========================================================================
# PART 1: The Attention Intuition
# ===========================================================================
print("=" * 60)
print("PART 1: What is Self-Attention?")
print("=" * 60)

print("""
Imagine reading: "The engineer fixed the machine because it was broken."

What does "it" refer to? The machine. (Not the engineer!)
Your brain computed ATTENTION — "it" attends to "machine", not "engineer".

Self-attention formalizes this:
  Every word/token QUERIES: "which other tokens are relevant to me?"
  Every word/token has KEYS: "here's what I represent"
  Every word/token has VALUES: "here's my information, if you attend to me"

Attention(Q, K, V) = softmax(QK^T / √d_k) · V

  QK^T:  how much each pair of tokens should attend to each other
  / √d_k: scale to prevent softmax from saturating
  softmax: convert scores to attention weights (sum to 1)
  · V:   weighted sum of value vectors → the attended representation
""")

# ===========================================================================
# PART 2: Self-Attention — Step by Step
# ===========================================================================
print("--- PART 2: Self-Attention Computation ---")

torch.manual_seed(0)

# A small sequence: batch=1, seq_len=4 tokens, d_model=8
batch, seq_len, d_model = 1, 4, 8
d_k = d_model  # key/query dimension

X = torch.randn(batch, seq_len, d_model)  # input embeddings

# Projection matrices (learned)
W_Q = torch.randn(d_model, d_k) * 0.1
W_K = torch.randn(d_model, d_k) * 0.1
W_V = torch.randn(d_model, d_k) * 0.1

# Step 1: Compute Q, K, V
Q = X @ W_Q  # (batch, seq_len, d_k)
K = X @ W_K
V = X @ W_V

print(f"Input X: {X.shape}")
print(f"Q, K, V: {Q.shape} each")

# Step 2: Compute attention scores
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (batch, seq_len, seq_len)
print(f"\nRaw attention scores (seq_len × seq_len):\n{scores[0].round(decimals=3)}")
print(f"  → entry [i,j] = how much token i attends to token j")

# Step 3: Softmax to get attention weights
attn_weights = torch.softmax(scores, dim=-1)
print(f"\nAttention weights (after softmax, rows sum to 1):\n{attn_weights[0].round(decimals=3)}")
print(f"  Row sums: {attn_weights[0].sum(dim=-1).tolist()}")

# Step 4: Weighted sum of values
output = attn_weights @ V  # (batch, seq_len, d_k)
print(f"\nAttended output: {output.shape}")

# ===========================================================================
# PART 3: Multi-Head Attention
# ===========================================================================
print("\n--- PART 3: Multi-Head Attention ---")

print("""
Problem with single-head attention: ONE attention pattern per layer.
  "it" can only attend to ONE other token at a time.

Multi-head attention: H independent attention heads in parallel.
  Each head learns a DIFFERENT type of relationship:
    Head 1: syntactic structure (subject-verb)
    Head 2: coreference ("it" → "machine")
    Head 3: proximity (nearby words)
    Head 4: semantic similarity

  Results from all heads are CONCATENATED then projected:
  MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · W_O
""")

# nn.MultiheadAttention
d_model_mha = 64
n_heads = 8  # n_heads must divide d_model evenly
d_head = d_model_mha // n_heads  # 64/8 = 8 per head

mha = nn.MultiheadAttention(embed_dim=d_model_mha, num_heads=n_heads, batch_first=True)
print(f"MultiheadAttention(d={d_model_mha}, heads={n_heads}): {sum(p.numel() for p in mha.parameters()):,} params")

X_mha = torch.randn(2, 10, d_model_mha)  # batch=2, seq_len=10, d=64
with torch.no_grad():
    out_mha, attn_map = mha(X_mha, X_mha, X_mha)  # Q=K=V=X for self-attention

print(f"Input: {X_mha.shape}")
print(f"Output: {out_mha.shape}")
print(f"Attention map: {attn_map.shape}  (averaged over heads)")

# ===========================================================================
# PART 4: Transformer Encoder Block
# ===========================================================================
print("\n--- PART 4: Transformer Encoder Block ---")

print("""
A Transformer Encoder Block:
  Input X
    ↓
  MultiHeadAttention(X, X, X)   ← self-attention
    ↓
  Add & LayerNorm   → X = LayerNorm(X + attention_out)
    ↓
  FeedForward: Linear(d→4d) → ReLU → Linear(4d→d)
    ↓
  Add & LayerNorm   → X = LayerNorm(X + feedforward_out)
    ↓
  Output X'

The residual connections (Add) + LayerNorm prevent vanishing gradients.
The FeedForward adds non-linear processing power to each position independently.
""")

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block.
    Implements: Self-Attention → Add&Norm → FFN → Add&Norm
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),  # expand
            nn.GELU(),                 # GELU is standard in modern transformers
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),  # project back
        )
        self.norm1 = nn.LayerNorm(d_model)  # LayerNorm (not BatchNorm!)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop(attn_out))
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x

block = TransformerEncoderBlock(d_model=64, n_heads=8, d_ff=256)
params = sum(p.numel() for p in block.parameters())
print(f"TransformerEncoderBlock params: {params:,}")

X_block = torch.randn(4, 20, 64)  # batch=4, seq_len=20, d=64
with torch.no_grad():
    out_block = block(X_block)
print(f"Block input: {X_block.shape} → output: {out_block.shape}")

# ===========================================================================
# PART 5: Sequence Classification With Transformers
# ===========================================================================
print("\n--- PART 5: Sequence Classification ---")

class SequenceClassifier(nn.Module):
    """
    Transformer encoder for sequence classification.
    Uses CLS token: prepend a learnable [CLS] token, read its final representation.
    """
    def __init__(self, input_dim: int, d_model: int, n_heads: int,
                 n_layers: int, n_classes: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Project input features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, n_classes)
        """
        B, T, _ = x.shape
        
        # Project to model dim
        x = self.input_proj(x)  # (B, T, d_model)
        
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)        # (B, T+1, d_model)
        
        # Add positional encoding
        x = x + self.pos_embed[:, :T+1, :]
        
        # Transformer encoder
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        
        # Classification from CLS token (position 0)
        cls_output = x[:, 0, :]  # (B, d_model)
        return self.head(cls_output)

# ===========================================================================
# PART 6: Train on Synthetic Multivariate Time Series
# ===========================================================================
print("--- PART 6: Training on Time-Series Classification ---")
print("""
Task: Classify 3 types of machine operational states from sensor sequences.
Each sample: 30 time steps × 8 sensor readings → class {0, 1, 2}
""")

# Generate synthetic time-series
def gen_ts_data(n_per_class=300, seq_len=30, n_features=8):
    X, y = [], []
    for cls in range(3):
        for _ in range(n_per_class):
            t = np.linspace(0, 1, seq_len)
            if cls == 0:  # Normal: smooth, low amplitude
                seq = np.stack([np.sin(2*np.pi*t + np.random.rand())
                                + np.random.randn(seq_len)*0.1 for _ in range(n_features)], axis=1)
            elif cls == 1:  # High-freq vibration
                seq = np.stack([np.sin(8*np.pi*t + np.random.rand())
                                + np.random.randn(seq_len)*0.15 for _ in range(n_features)], axis=1)
            else:  # Ramp (degrading)
                seq = np.stack([t + np.random.randn(seq_len)*0.1
                                for _ in range(n_features)], axis=1)
            X.append(seq)
            y.append(cls)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

X_ts, y_ts = gen_ts_data(n_per_class=300)
print(f"Time-series dataset: {X_ts.shape}  (N, seq_len, n_features)")

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X_ts, y_ts, test_size=0.2, stratify=y_ts, random_state=42)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
X_te_t  = torch.tensor(X_te, dtype=torch.float32)
y_tr_t  = torch.tensor(y_tr, dtype=torch.long)
y_te_t  = torch.tensor(y_te, dtype=torch.long)

loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)

# Build and train the Transformer
model = SequenceClassifier(
    input_dim=8, d_model=32, n_heads=4,
    n_layers=2, n_classes=3, max_len=30, dropout=0.1
)
total_p = sum(p.numel() for p in model.parameters())
print(f"Transformer model: {total_p:,} parameters")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

val_accs, train_losses = [], []
best_acc, best_state = 0, None

print("\nTraining Transformer for time-series classification...")
for epoch in range(40):
    model.train()
    ep_losses = []
    for Xb, yb in loader:
        logits = model(Xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(); loss.backward(); 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_losses.append(loss.item())
    
    scheduler.step()
    
    model.eval()
    with torch.no_grad():
        acc = (model(X_te_t).argmax(dim=1) == y_te_t).float().mean().item()
    
    val_accs.append(acc)
    train_losses.append(np.mean(ep_losses))
    
    if acc > best_acc:
        best_acc = acc
        best_state = copy.deepcopy(model.state_dict())
    
    if epoch % 10 == 0:
        print(f"  Epoch {epoch+1:2d}: loss={np.mean(ep_losses):.4f}, val_acc={acc:.4f}")

print(f"\nBest accuracy: {best_acc:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Transformer for Time-Series Classification", fontsize=12, fontweight='bold')
axes[0].plot(train_losses, color='steelblue')
axes[0].set_title("Training Loss"), axes[0].set_xlabel("Epoch"), axes[0].grid(True, alpha=0.3)
axes[1].plot(val_accs, color='coral')
axes[1].axhline(best_acc, color='green', linestyle='--', label=f'Best: {best_acc:.3f}')
axes[1].set_title("Validation Accuracy"), axes[1].set_xlabel("Epoch")
axes[1].legend(), axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "transformer_time_series.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"Training plot saved: {OUTPUT_DIR}/transformer_time_series.png")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. Self-Attention: every token attends to every other token directly
   → No vanishing gradient over long distances (unlike RNN)
2. Q, K, V: queries (what am I looking for), keys (what do I have),
   values (what I contribute when attended to)
3. Multi-head: multiple attention heads learn different relationships
4. Encoder block: Self-Attn → Add&Norm → FFN → Add&Norm
5. LayerNorm (not BatchNorm!): normalizes per-sample, not per-batch
6. CLS token: prepend and read for sequence-level classification
7. Positional encoding: add position info (transformers are order-agnostic otherwise)

Where Transformers win:
  ✓ Long sequences with long-range dependencies
  ✓ Text, code, audio, protein sequences
  ✓ Large datasets (transformers scale better than RNNs with data)

Where MLPs/CNNs still win:
  ✓ Small tabular datasets (Transformer overkill)
  ✓ Short sequences with only local patterns (CNN often faster)
  ✓ Limited compute budget

Next: 11_transformers/02_attention_visualization.py — Visualize what attention learns
""")
