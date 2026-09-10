"""
01_nn_module.py — The nn.Module: PyTorch's Building Block for All Models

WHAT THIS LESSON TEACHES:
    nn.Module is the base class for EVERY PyTorch model — from a 2-layer MLP
    to GPT. Understanding it deeply means you can build anything.

WHY IT MATTERS:
    Unlike sklearn's estimators (black boxes), nn.Module gives you full
    transparency: you define exactly what layers exist and exactly how data
    flows through them. This control is what enables custom architectures
    like attention, residual connections, and graph networks.

LESSON STRUCTURE:
    1. The minimal nn.Module
    2. Common layer types
    3. Composing modules (module inside module)
    4. Inspecting parameters
    5. Saving and loading weights
    6. train() vs eval() mode
    7. A complete MLP built step by step
"""

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: The Minimal nn.Module
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: The Minimal nn.Module")
print("=" * 60)

# WHY: Every model must subclass nn.Module. The two methods you MUST implement:
#   - __init__: define your layers (this registers them as "parameters")
#   - forward(): define the computation (how data flows through layers)

class LinearRegression(nn.Module):
    """A single linear layer: y = Wx + b (same as sklearn LinearRegression)."""

    def __init__(self, input_features: int, output_features: int):
        # WHY super().__init__(): registers this class with PyTorch's module system
        # so it can track parameters, handle .to(device), etc.
        super().__init__()

        # WHY nn.Linear: not just a matrix — it initialises weights properly,
        # registers them as parameters (so optimizer can find them),
        # and handles batches automatically.
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # WHY: forward() is called when you do model(x).
        # PyTorch calls this AND sets up the computation graph for autograd.
        return self.linear(x)

# Create the model
model = LinearRegression(input_features=3, output_features=1)
print(f"Model: {model}")

# Test with dummy input
x = torch.randn(5, 3)  # 5 samples, 3 features
y_pred = model(x)       # calls forward() automatically
print(f"Input shape:  {x.shape}")
print(f"Output shape: {y_pred.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Common Layer Types
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2: Common Layer Types")
print("=" * 60)

x_demo = torch.randn(4, 8)  # 4 samples, 8 features

# nn.Linear — the workhorse: y = Wx + b
linear = nn.Linear(8, 4)
print(f"nn.Linear(8→4):    {linear(x_demo).shape}")

# nn.ReLU — Rectified Linear Unit: max(0, x)
# WHY: Most common activation. Simple, fast, avoids vanishing gradients.
relu = nn.ReLU()
print(f"nn.ReLU output range: [{relu(x_demo).min().item():.2f}, {relu(x_demo).max().item():.2f}]")

# nn.Sigmoid — squashes to (0, 1), used for binary classification output
sigmoid = nn.Sigmoid()
print(f"nn.Sigmoid output range: [{sigmoid(x_demo).min().item():.2f}, {sigmoid(x_demo).max().item():.2f}]")

# nn.BatchNorm1d — normalises each feature across the batch
# WHY: Keeps activations in a healthy range, enables higher learning rates
bn = nn.BatchNorm1d(8)
print(f"nn.BatchNorm1d:    {bn(x_demo).shape}")

# nn.Dropout — randomly zeros neurons during training
# WHY: Acts as regularisation, prevents over-reliance on any single neuron
dropout = nn.Dropout(p=0.3)  # 30% of neurons zeroed each forward pass
print(f"nn.Dropout (train mode, p=0.3):   {(dropout(x_demo) == 0).float().mean().item():.2f} fraction zeroed")

# nn.Sequential — the simplest way to chain layers in order
sequential = nn.Sequential(
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
)
print(f"nn.Sequential(8→4→ReLU→2):  {sequential(x_demo).shape}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Composing Modules (Module Inside Module)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3: Composing Modules")
print("=" * 60)

# WHY: Complex models (ResNets, Transformers) are built by composing
# smaller modules. PyTorch automatically finds ALL parameters recursively.

class Encoder(nn.Module):
    """A small encoder block — maps high-dim features to a bottleneck."""

    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, bottleneck_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class Classifier(nn.Module):
    """Full model: Encoder + classification head."""

    def __init__(self, input_dim: int, bottleneck_dim: int, num_classes: int):
        super().__init__()
        # WHY: storing sub-modules as attributes registers their parameters
        # automatically. model.parameters() will find them all.
        self.encoder = Encoder(input_dim, bottleneck_dim)
        self.head = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)   # encode to bottleneck
        logits = self.head(features) # classify from bottleneck
        return logits


composite_model = Classifier(input_dim=16, bottleneck_dim=4, num_classes=3)
x_test = torch.randn(8, 16)
out = composite_model(x_test)
print(f"Composite model output shape: {out.shape}")
print(composite_model)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Inspecting Parameters
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4: Inspecting Parameters")
print("=" * 60)

# model.parameters() — iterator over all learnable tensors
# WHY: The optimizer needs this list to know WHAT to update.
total_params = 0
for name, param in composite_model.named_parameters():
    print(f"  {name:30s} shape={str(param.shape):20s} requires_grad={param.requires_grad}")
    total_params += param.numel()

print(f"\nTotal trainable parameters: {total_params:,}")

# model.state_dict() — ordered dict of {parameter_name: tensor}
# WHY: This is how you SAVE and LOAD model weights.
state = composite_model.state_dict()
print(f"\nstate_dict keys: {list(state.keys())}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Saving and Loading Weights
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5: Saving and Loading Weights")
print("=" * 60)

import tempfile, os

# Save weights to a file
save_path = "/tmp/demo_model.pt"
torch.save(composite_model.state_dict(), save_path)
print(f"Model saved to {save_path}")

# Load weights into a NEW model instance
loaded_model = Classifier(input_dim=16, bottleneck_dim=4, num_classes=3)
loaded_model.load_state_dict(torch.load(save_path, weights_only=True))
print("Model loaded successfully")

# Verify identical outputs
with torch.no_grad():
    out_original = composite_model(x_test)
    out_loaded = loaded_model(x_test)
    print(f"Outputs identical: {torch.allclose(out_original, out_loaded)}")

# WHY map_location: allows loading a GPU-saved model on CPU (important!)
# loaded_model.load_state_dict(torch.load(path, map_location='cpu'))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: train() vs eval() Mode
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6: train() vs eval() Mode")
print("=" * 60)

# WHY: Some layers behave DIFFERENTLY during training vs inference:
#   - nn.Dropout: active (zeroes neurons) in train(), disabled in eval()
#   - nn.BatchNorm: uses batch statistics in train(), running stats in eval()
#
# Forgetting model.eval() during validation is a common bug!

class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.dropout(self.fc(x))

demo_model = ModelWithDropout()
x_small = torch.ones(1, 4)

# Train mode: dropout active — each call gives different result
demo_model.train()
out1 = demo_model(x_small)
out2 = demo_model(x_small)
print(f"Train mode — same input, different outputs: {not torch.allclose(out1, out2)}")

# Eval mode: dropout disabled — deterministic
demo_model.eval()
with torch.no_grad():
    out3 = demo_model(x_small)
    out4 = demo_model(x_small)
print(f"Eval mode — same input, same outputs: {torch.allclose(out3, out4)}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: A Complete MLP — Step by Step
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7: Complete MLP Built Step by Step")
print("=" * 60)

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for classification.

    Architecture:
        Input → Linear → BatchNorm → ReLU → Dropout
              → Linear → BatchNorm → ReLU → Dropout
              → Linear → output logits

    WHY this architecture:
        - BatchNorm before activation: stabilises training, enables higher LR
        - ReLU: avoids vanishing gradients
        - Dropout: regularisation — prevents overfitting
        - No activation on final layer: CrossEntropyLoss expects raw logits
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim

        # Final output layer — no activation (loss function handles it)
        layers.append(nn.Linear(prev_dim, output_dim))

        # WHY nn.Sequential here: we built the list dynamically, now wrap it
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# Build an MLP: 20 features → 64 → 32 → 5 classes
mlp = MLP(
    input_dim=20,
    hidden_dims=[64, 32],
    output_dim=5,
    dropout_rate=0.2,
)

print(mlp)

# Count parameters
n_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {n_params:,}")

# Forward pass
x_sample = torch.randn(16, 20)  # batch of 16, 20 features each
mlp.train()
logits = mlp(x_sample)
print(f"\nInput:  {x_sample.shape}")
print(f"Output: {logits.shape}  (raw logits, not probabilities)")

# Convert to probabilities
probs = torch.softmax(logits, dim=1)
predicted_classes = probs.argmax(dim=1)
print(f"Predicted classes: {predicted_classes.tolist()}")

print("\n✓ 01_nn_module.py complete")
print("\nKEY TAKEAWAYS:")
print("  1. All models subclass nn.Module and implement __init__ + forward()")
print("  2. Layers defined in __init__ are automatically registered as parameters")
print("  3. model.parameters() gives the optimizer everything it needs to update")
print("  4. model.train() / model.eval() controls dropout and batchnorm behaviour")
print("  5. model.state_dict() / torch.save() for saving weights")
