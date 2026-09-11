# Module 11: Transformers

## What You'll Learn

The Transformer architecture — the foundation of every major AI breakthrough since 2017.

---

## Why Transformers?

| Problem | RNN Approach | Transformer Approach |
|---|---|---|
| Long-range dependency | Hidden state (forgets!) | Direct attention (no forgetting) |
| Parallelism | Sequential (slow) | Fully parallel (fast) |
| Scaling | Saturates | Scales with data and compute |
| Applications | Text only (mostly) | Text, images, audio, protein, code |

---

## Lesson Files

| File | Topic | Concepts |
|---|---|---|
| [`01_attention_and_transformers.py`](01_attention_and_transformers.py) | Self-attention, multi-head, encoder block | Q, K, V matrices, softmax, residual+norm |
| [`exercises.py`](exercises.py) | All levels | Recall → implement from scratch |

---

## The Core Architecture

```
Input Sequence [x₁, x₂, ..., xₙ]
        ↓
   Input Embeddings + Positional Encoding
        ↓
   ┌──────────────────────────────┐
   │   TransformerEncoderBlock    │  × N layers
   │                              │
   │   MultiHeadAttention(x,x,x)  │
   │   Add & LayerNorm            │
   │   FeedForward(Linear→GELU→Linear) │
   │   Add & LayerNorm            │
   └──────────────────────────────┘
        ↓
   Output Representations
        ↓
   [CLS] token → Linear → n_classes  (for classification)
```

---

## Self-Attention Formula

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

Where:
  Q = X · W_Q    (queries: what am I looking for?)
  K = X · W_K    (keys:    what does each position contain?)
  V = X · W_V    (values:  what information to extract?)
  d_k = dimension of Q and K (for scaling)
```

---

## PyTorch Usage

```python
import torch
import torch.nn as nn

# Multi-head self-attention
mha = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1, batch_first=True)

# Self-attention call (Q=K=V=x)
x = torch.randn(batch, seq_len, 256)
output, attn_weights = mha(x, x, x)

# Transformer encoder layer (complete block)
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256, nhead=8, dim_feedforward=1024,
    dropout=0.1, activation='gelu', batch_first=True
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
output = encoder(x)
```

---

## Real-World Transformer Models

| Model | Task | Framework |
|---|---|---|
| BERT | Text classification, NER, Q&A | HuggingFace |
| GPT-2/3/4 | Text generation | HuggingFace / OpenAI |
| T5 | Text-to-text | HuggingFace |
| ViT | Image classification | HuggingFace / torchvision |
| Whisper | Speech recognition | HuggingFace / OpenAI |
| CodeBERT | Code understanding | HuggingFace |

### Using HuggingFace (after installing `transformers`):
```python
# NOTE: requires: pip install transformers
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
```

---

## Prerequisites
- Module 7: PyTorch tensors and autograd  
- Module 8: Training loops, nn.Module
- Module 9: Activation functions, deep networks

---

## Next
After completing this module, move to **Module 12: Capstone Project** — apply everything you've learned to a real engineering problem.
