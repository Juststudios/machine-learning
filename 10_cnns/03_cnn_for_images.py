"""
CNN for Image Classification
==============================
CNNs (Convolutional Neural Networks) are the reason deep learning
dominated computer vision. They understand SPATIAL structure.

For an 32x32 RGB image (32×32×3 = 3072 pixels):
  - A fully connected layer connecting all pixels to 512 neurons
    needs 3072 × 512 = 1,572,864 parameters for ONE layer.
  - A convolutional layer with 32 filters of size 3×3×3 needs
    32 × (3×3×3 + 1) = 896 parameters — 1756x fewer!

CNN Advantages:
  1. Parameter sharing: same filter applied everywhere → translation invariance
  2. Local connectivity: each neuron sees a small neighborhood, not all pixels
  3. Hierarchical features: edge → shape → object

This lesson: MNIST-like classification on synthetic digit patterns.
(We use synthetic data to avoid downloading dependencies.)
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

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

torch.manual_seed(42)
np.random.seed(42)

# ===========================================================================
# PART 1: CNN Architecture Overview
# ===========================================================================
print("=" * 60)
print("PART 1: CNN Architecture")
print("=" * 60)

print("""
A typical CNN for image classification:

  Input (N, C, H, W)
    ↓
  Conv2d(C, F, kernel_size=3, padding=1)   → same spatial size
  BatchNorm2d(F)
  ReLU
    ↓
  MaxPool2d(2, 2)   → halve spatial dimensions
    ↓
  Conv2d(F, 2F, kernel_size=3, padding=1)
  BatchNorm2d(2F)
  ReLU
    ↓
  MaxPool2d(2, 2)
    ↓
  Flatten()
    ↓
  Linear(2F * H/4 * W/4, 256)
  ReLU
  Dropout(0.5)
    ↓
  Linear(256, n_classes)

N = batch size, C = channels (1 for grayscale, 3 for RGB)
F = number of filters (32, 64, 128, ...)
""")

# ===========================================================================
# PART 2: Build the CNN
# ===========================================================================
print("--- PART 2: Building the CNN ---")

class SimpleCNN(nn.Module):
    """
    Simple CNN for grayscale image classification.
    Input: (batch, 1, 28, 28) — grayscale 28x28 images
    Output: (batch, n_classes) — class logits
    """
    def __init__(self, n_classes: int = 10, dropout: float = 0.4):
        super().__init__()
        
        # --- Feature Extractor (Conv layers) ---
        self.features = nn.Sequential(
            # Block 1: 1 → 32 channels, 28×28 → 14×14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (N, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # (N, 32, 14, 14)
            
            # Block 2: 32 → 64 channels, 14×14 → 7×7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (N, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # (N, 64, 7, 7)
        )
        
        # --- Classifier (FC layers) ---
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # (N, 64*7*7) = (N, 3136)
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

model = SimpleCNN(n_classes=10)
print(f"CNN Architecture:\n{model}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
conv_params  = sum(p.numel() for p in model.features.parameters())
fc_params    = sum(p.numel() for p in model.classifier.parameters())
print(f"\nTotal parameters:    {total_params:,}")
print(f"  Conv layers:       {conv_params:,}  ({100*conv_params/total_params:.1f}%)")
print(f"  FC layers:         {fc_params:,}    ({100*fc_params/total_params:.1f}%)")

# Verify shapes with a dummy input
dummy_input = torch.randn(4, 1, 28, 28)  # batch of 4 grayscale 28×28 images
with torch.no_grad():
    dummy_out = model(dummy_input)
print(f"\nShape check: input {list(dummy_input.shape)} → output {list(dummy_out.shape)}")

# ===========================================================================
# PART 3: Feature Map Visualization (what CNNs "see")
# ===========================================================================
print("\n--- PART 3: What CNNs Learn ---")

print("""
Understanding CNN feature maps:
  Early layers: detect edges, corners, simple textures
  Middle layers: detect shapes, patterns, parts of objects
  Late layers: detect class-specific features (faces, wheels, etc.)

This hierarchy is WHY CNNs are so powerful for images.
Each layer builds upon the features discovered by previous layers.
""")

# ===========================================================================
# PART 4: Generate Synthetic Training Data (grayscale 28x28)
# ===========================================================================
print("--- PART 4: Generate Synthetic Image Data ---")

def generate_synthetic_images(n_per_class=200, img_size=28, n_classes=4):
    """
    Generate simple synthetic images:
    Class 0: bright circle in center
    Class 1: horizontal bar
    Class 2: vertical bar
    Class 3: bright corners
    """
    images = []
    labels = []
    
    for cls in range(n_classes):
        for _ in range(n_per_class):
            img = np.zeros((img_size, img_size), dtype=np.float32)
            noise = np.random.randn(img_size, img_size) * 0.1
            
            if cls == 0:  # Circle
                cx, cy = img_size//2 + np.random.randint(-3, 3), img_size//2 + np.random.randint(-3, 3)
                r = np.random.randint(4, 8)
                for i in range(img_size):
                    for j in range(img_size):
                        if (i - cx)**2 + (j - cy)**2 < r**2:
                            img[i, j] = 1.0
            elif cls == 1:  # Horizontal bar
                row = np.random.randint(8, img_size-8)
                img[row-2:row+2, 2:-2] = 1.0
            elif cls == 2:  # Vertical bar
                col = np.random.randint(8, img_size-8)
                img[2:-2, col-2:col+2] = 1.0
            elif cls == 3:  # Bright corners
                sz = np.random.randint(4, 8)
                img[:sz, :sz] = 1.0
                img[:sz, -sz:] = 1.0
                img[-sz:, :sz] = 1.0
                img[-sz:, -sz:] = 1.0
            
            img = np.clip(img + noise, 0, 1)
            images.append(img)
            labels.append(cls)
    
    X = np.array(images)[:, np.newaxis, :, :]  # add channel dim → (N, 1, H, W)
    y = np.array(labels)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

X_img, y_img = generate_synthetic_images(n_per_class=300, img_size=28, n_classes=4)
print(f"Synthetic dataset: {X_img.shape} (N, C, H, W)")
print(f"Classes: {np.bincount(y_img)} (circle, h-bar, v-bar, corners)")

# Split
from sklearn.model_selection import train_test_split

X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(X_img, y_img, test_size=0.2, stratify=y_img, random_state=42)

X_tr_t = torch.tensor(X_tr_i, dtype=torch.float32)
X_te_t  = torch.tensor(X_te_i, dtype=torch.float32)
y_tr_t  = torch.tensor(y_tr_i, dtype=torch.long)
y_te_t  = torch.tensor(y_te_i, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)

# ===========================================================================
# PART 5: Train the CNN
# ===========================================================================
print("\n--- PART 5: Training the CNN ---")

model = SimpleCNN(n_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_acc, best_state = 0.0, None
history = {'loss': [], 'val_acc': []}

for epoch in range(30):
    model.train()
    epoch_losses = []
    for Xb, yb in train_loader:
        out = model(Xb)
        loss = criterion(out, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        val_preds = model(X_te_t).argmax(dim=1)
        val_acc = (val_preds == y_te_t).float().mean().item()
    
    history['loss'].append(np.mean(epoch_losses))
    history['val_acc'].append(val_acc)
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if epoch % 5 == 0:
        print(f"  Epoch {epoch+1:2d}: loss={np.mean(epoch_losses):.4f}, val_acc={val_acc:.4f}")

model.load_state_dict(best_state)
print(f"\nBest val accuracy: {best_acc:.4f}")

# ===========================================================================
# PART 6: Visualize Results
# ===========================================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("CNN for Synthetic Image Classification", fontsize=13, fontweight='bold')

# Sample images
class_names = ['Circle', 'H-Bar', 'V-Bar', 'Corners']
for i, cls in enumerate([0, 1, 2, 3]):
    mask = y_img == cls
    ax = axes[0, i] if i < 3 else axes[1, 0]
    ax.imshow(X_img[mask][0, 0], cmap='gray')
    ax.set_title(f"Class {cls}: {class_names[cls]}")
    ax.axis('off')

# Training curves
axes[1, 1].plot(history['loss'], color='steelblue', linewidth=2)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Cross-Entropy Loss")
axes[1, 1].set_title("Training Loss")
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].plot(history['val_acc'], color='coral', linewidth=2)
axes[1, 2].axhline(best_acc, color='green', linestyle='--', label=f'Best: {best_acc:.3f}')
axes[1, 2].set_xlabel("Epoch")
axes[1, 2].set_ylabel("Validation Accuracy")
axes[1, 2].set_title("Validation Accuracy")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cnn_image_classification.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nCNN training plot saved: {OUTPUT_DIR}/cnn_image_classification.png")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. Conv2d(in_ch, out_ch, kernel_size, padding): applies a learned filter bank
2. MaxPool2d(2, 2): halves spatial dimensions, keeps strongest activations
3. BatchNorm2d: normalize feature maps → stabilizes training
4. Feature hierarchy: edge → shape → object (automatic feature engineering!)
5. Flatten() before FC layers: convert (N, C, H, W) → (N, C*H*W)
6. CNNs are translation equivariant: same filter applied everywhere

When to use CNN vs MLP:
  CNN → data with SPATIAL structure (images, 2D signals, spectrograms)
  MLP → tabular data, 1D features with no spatial meaning

Next: 04_transfer_learning.py — Use pretrained CNNs for your own problems
""")
