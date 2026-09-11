"""
Transfer Learning: Standing on the Shoulders of Giants
========================================================
Training a deep CNN from scratch requires:
  - Millions of labeled images
  - Days/weeks on powerful GPUs
  - Deep ML expertise to get right

Transfer learning lets you avoid all of this:
  1. Take a model pretrained on millions of images (e.g., ResNet on ImageNet)
  2. Replace just the final classification layer for your task
  3. Fine-tune on YOUR small dataset (100-10,000 images is enough!)

Why it works:
  Pretrained models have already learned universal visual features:
  - Layer 1: edges, colors, simple textures
  - Layer 2: corners, curves, patterns
  - Layer 3-5: eyes, wheels, windows, fur, etc.
  
  Your custom task (e.g., detecting PCB defects) uses THESE same low-level
  features! You only need to teach the final layer the new class concepts.

Two strategies:
  A. Feature Extraction: freeze ALL conv layers, train only the new FC head
     → Fast, good when your data is similar to ImageNet
  B. Fine-tuning: unfreeze some/all layers and retrain with a small LR
     → More powerful, needs more data and time
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
# PART 1: Transfer Learning Concept
# ===========================================================================
print("=" * 60)
print("PART 1: Transfer Learning Concept")
print("=" * 60)

print("""
WHY transfer learning works:

  ImageNet pretrained model learned on 1.2 million images, 1000 classes.
  The features it learned are GENERAL visual features.

  For a new task (e.g., 5 types of leaf diseases):
    - You may have only 500 images
    - Training from scratch → massive overfitting
    - Transfer learning: start from excellent initialization → much better!

  Two workflows:

  A. Feature Extraction (Frozen backbone):
     model.features: FROZEN (no gradient, no update)
     model.head: TRAINABLE (only these weights update)
     
     Advantages: fast, doesn't need GPU, few training epochs needed
     When to use: small dataset, task similar to ImageNet

  B. Fine-tuning (Partial/Full Unfreeze):
     model.features[-few layers]: TRAINABLE (with very small LR)
     model.head: TRAINABLE (with normal LR)
     
     Advantages: better accuracy on new domain
     When to use: medium dataset, or target domain differs significantly from ImageNet
""")

# ===========================================================================
# PART 2: Simulate a Pretrained CNN
# ===========================================================================
print("--- PART 2: Simulating a Pretrained CNN ---")
print("""
NOTE: In practice you would load a real pretrained model:
  import torchvision.models as models
  backbone = models.resnet18(pretrained=True)  # or weights='DEFAULT'
  
Here we simulate a pretrained model since torchvision is not installed.
The principle and code structure are identical.
""")

class PretrainedCNNSimulated(nn.Module):
    """
    Simulates a pretrained CNN backbone (e.g., ResNet-like).
    In real code: use torchvision.models.resnet18(weights='DEFAULT')
    """
    def __init__(self, feature_dim=512):
        super().__init__()
        # Simulate the pretrained backbone (already trained on ImageNet)
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling → (N, 128, 1, 1)
            nn.Flatten(),
        )
        # "Pretrained" classification head (ImageNet: 1000 classes)
        self.head = nn.Linear(128, 10)  # 10-class original head
        
        # Simulate "pretrained" weights by initializing meaningfully
        self._simulate_pretrained_init()
    
    def _simulate_pretrained_init(self):
        """Initialize as if pretrained (better than default random init)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
    
    def get_feature_dim(self):
        """Return dimension of backbone output features."""
        return 128

pretrained_model = PretrainedCNNSimulated()
backbone_params = sum(p.numel() for p in pretrained_model.backbone.parameters())
head_params     = sum(p.numel() for p in pretrained_model.head.parameters())
print(f"Pretrained model: {backbone_params:,} backbone params + {head_params:,} head params")

# ===========================================================================
# PART 3: Strategy A — Feature Extraction (Freeze Backbone)
# ===========================================================================
print("\n--- PART 3: Feature Extraction (Frozen Backbone) ---")

def create_feature_extractor(pretrained_model, n_new_classes: int):
    """
    Adapt a pretrained model for a new classification task.
    FREEZES the backbone, REPLACES and trains only the head.
    """
    # Step 1: Freeze ALL backbone layers
    for param in pretrained_model.backbone.parameters():
        param.requires_grad = False
    
    # Step 2: Replace the head with new one for our task
    feature_dim = pretrained_model.get_feature_dim()
    pretrained_model.head = nn.Sequential(
        nn.Linear(feature_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, n_new_classes),
    )
    
    return pretrained_model

extractor = create_feature_extractor(pretrained_model, n_new_classes=4)

# Count trainable params
trainable = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
total     = sum(p.numel() for p in extractor.parameters())
print(f"Feature Extraction:")
print(f"  Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")
print(f"  Backbone frozen: YES")

# ===========================================================================
# PART 4: Strategy B — Fine-tuning (Partial Unfreeze)
# ===========================================================================
print("\n--- PART 4: Fine-tuning (Partial Unfreeze) ---")

def create_finetuner(feature_dim: int = 128, n_new_classes: int = 4):
    """
    Create a fresh model (simulating fine-tuning from pretrained).
    In practice: load pretrained, then unfreeze last few layers.
    """
    model = PretrainedCNNSimulated()
    
    # Unfreeze only the LAST conv block (fine-tune it at small LR)
    for param in model.backbone[:6].parameters():  # freeze first 2 blocks
        param.requires_grad = False
    for param in model.backbone[6:].parameters():  # unfreeze last block
        param.requires_grad = True
    
    # Replace head
    model.head = nn.Sequential(
        nn.Linear(feature_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, n_new_classes),
    )
    
    return model

finetuner = create_finetuner(n_new_classes=4)
trainable_ft = sum(p.numel() for p in finetuner.parameters() if p.requires_grad)
total_ft     = sum(p.numel() for p in finetuner.parameters())
print(f"Fine-tuning:")
print(f"  Trainable: {trainable_ft:,} / {total_ft:,} params ({100*trainable_ft/total_ft:.1f}%)")
print(f"  Backbone: last block unfrozen")

# ===========================================================================
# PART 5: Training With Different Learning Rates Per Layer Group
# ===========================================================================
print("\n--- PART 5: Layer-wise Learning Rates ---")
print("""
When fine-tuning, use SMALLER learning rate for pretrained layers.
WHY: Pretrained weights are already good. Large LR would destroy them.

Use param_groups to specify different LR per module:
  optimizer = optim.Adam([
      {'params': model.backbone.parameters(), 'lr': 1e-5},  # very small
      {'params': model.head.parameters(),     'lr': 1e-3},  # normal
  ])
""")

# Example of layer-wise learning rates
finetuner2 = create_finetuner(n_new_classes=4)
optimizer_layerwise = optim.Adam([
    {'params': [p for p in finetuner2.backbone.parameters() if p.requires_grad],
     'lr': 1e-5, 'weight_decay': 1e-4},  # small LR for pretrained layers
    {'params': finetuner2.head.parameters(),
     'lr': 1e-3, 'weight_decay': 1e-4},  # normal LR for new head
])
print(f"Layer-wise optimizer created with {len(optimizer_layerwise.param_groups)} param groups")
for i, g in enumerate(optimizer_layerwise.param_groups):
    n_params = sum(p.numel() for p in g['params'])
    print(f"  Group {i}: lr={g['lr']}, n_params={n_params:,}")

# ===========================================================================
# PART 6: Quick Training Comparison
# ===========================================================================
print("\n--- PART 6: Comparing Strategies ---")

# Use our synthetic image dataset
def generate_synthetic_images(n_per_class=100, img_size=28, n_classes=4):
    images, labels = [], []
    for cls in range(n_classes):
        for _ in range(n_per_class):
            img = np.zeros((img_size, img_size), dtype=np.float32)
            noise = np.random.randn(img_size, img_size) * 0.1
            if cls == 0:
                cx, cy = img_size//2, img_size//2
                for i in range(img_size):
                    for j in range(img_size):
                        if (i-cx)**2 + (j-cy)**2 < 6**2:
                            img[i, j] = 1.0
            elif cls == 1:
                img[12:16, 2:-2] = 1.0
            elif cls == 2:
                img[2:-2, 12:16] = 1.0
            else:
                img[:6, :6] = img[:6, -6:] = img[-6:, :6] = img[-6:, -6:] = 1.0
            images.append(np.clip(img + noise, 0, 1))
            labels.append(cls)
    X = np.array(images)[:, np.newaxis, :, :]
    y = np.array(labels)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

from sklearn.model_selection import train_test_split

X_img, y_img = generate_synthetic_images(n_per_class=150)
X_tr, X_te, y_tr, y_te = train_test_split(X_img, y_img, test_size=0.2, stratify=y_img, random_state=42)
X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
X_te_t  = torch.tensor(X_te, dtype=torch.float32)
y_tr_t  = torch.tensor(y_tr, dtype=torch.long)
y_te_t  = torch.tensor(y_te, dtype=torch.long)
loader  = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)

def train_quick(model, loader, X_te, y_te, epochs=20, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    accs = []
    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            loss = criterion(model(Xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X_te).argmax(dim=1) == y_te).float().mean().item()
        accs.append(acc)
    return accs

print("Training from scratch (20 epochs)...")
scratch_model = PretrainedCNNSimulated()
scratch_model.head = nn.Linear(128, 4)
scratch_accs = train_quick(scratch_model, loader, X_te_t, y_te_t, epochs=20)

print("Training with feature extraction (20 epochs)...")
extractor2 = create_feature_extractor(PretrainedCNNSimulated(), n_new_classes=4)
extract_accs = train_quick(extractor2, loader, X_te_t, y_te_t, epochs=20)

print(f"\nFinal accuracy after 20 epochs:")
print(f"  Training from scratch:  {scratch_accs[-1]:.4f}")
print(f"  Feature extraction:     {extract_accs[-1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(scratch_accs,  'o-', color='steelblue', label='From Scratch')
ax.plot(extract_accs, 's-', color='coral',     label='Feature Extraction')
ax.set_xlabel("Epoch")
ax.set_ylabel("Test Accuracy")
ax.set_title("Transfer Learning vs Training from Scratch")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "transfer_learning_comparison.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nComparison plot saved: {OUTPUT_DIR}/transfer_learning_comparison.png")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. Transfer learning: reuse pretrained weights, train only new head
2. Feature Extraction: freeze backbone, fast, works with tiny datasets
3. Fine-tuning: unfreeze some layers, needs more data, better accuracy
4. Layer-wise LR: use very small LR (1e-5) for pretrained layers
5. In practice: use torchvision.models (ResNet, EfficientNet, ViT) pretrained on ImageNet
6. Rule of thumb:
     < 100 images per class  → Feature extraction only
     100-1000 images/class   → Fine-tune last few layers
     > 1000 images/class     → Full fine-tuning
7. Transfer learning = biggest practical win in computer vision today

Real code template:
  import torchvision.models as models
  backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
  for p in backbone.parameters(): p.requires_grad = False   # freeze
  backbone.fc = nn.Linear(512, N_CLASSES)                   # replace head
""")
