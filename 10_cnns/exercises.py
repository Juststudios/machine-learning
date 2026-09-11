"""
Exercises: CNNs (Module 10)
=============================
4-tier exercises for convolution, pooling, CNN architecture, and transfer learning.
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

print("=" * 60)
print("CNNs — EXERCISES")
print("=" * 60)

# ===========================================================================
# LEVEL 1: RECALL
# ===========================================================================
print("\n--- LEVEL 1: RECALL ---")
print("""
Answer by working through the shape calculations:

1a. An input is (batch=8, channels=3, height=64, width=64).
    After Conv2d(3, 32, kernel_size=3, padding=1), what is the output shape?

1b. After MaxPool2d(2, 2), what is the output shape from (8, 32, 64, 64)?

1c. How many learnable parameters does Conv2d(3, 64, kernel_size=3) have?
    (Don't forget the bias terms)

1d. What is the purpose of padding='same' in convolution?

1e. What does AdaptiveAvgPool2d((1,1)) produce from (N, 128, 7, 7)?
    Why is it useful?
""")

# Shape demonstration
X_demo = torch.randn(8, 3, 64, 64)
conv    = nn.Conv2d(3, 32, kernel_size=3, padding=1)
pool    = nn.MaxPool2d(2, 2)
gap     = nn.AdaptiveAvgPool2d((1, 1))
flatten = nn.Flatten()

with torch.no_grad():
    z1 = conv(X_demo)
    z2 = pool(z1)
    z3 = gap(z2)
    z4 = flatten(z3)

print(f"Input:               {list(X_demo.shape)}")
print(f"After Conv2d(3→32):  {list(z1.shape)}")
print(f"After MaxPool2d(2):  {list(z2.shape)}")
print(f"After AdaptiveAvgPool2d((1,1)): {list(z3.shape)}")
print(f"After Flatten():     {list(z4.shape)}")

conv_params = sum(p.numel() for p in conv.parameters())
print(f"\nConv2d(3, 32, 3x3) parameters: {conv_params}")
print(f"  Weights: 32 filters × (3×3×3) = {32*3*3*3}")
print(f"  Biases:  32")

# ===========================================================================
# LEVEL 2: DEBUGGING
# ===========================================================================
print("\n--- LEVEL 2: DEBUGGING ---")
print("""
Find the 4 bugs in this CNN.
""")

buggy = '''
import torch
import torch.nn as nn

# BUG 1: Wrong input channels to second Conv (should match output of first)
class BuggyCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        # BUG 1: Second conv uses in_channels=1, should be 32
        self.conv2 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # BUG 2: Hardcoded flatten dimension (will break for different input sizes)
        self.fc1   = nn.Linear(64 * 28 * 28, 128)   # wrong! after 2 pooling: 28→7
        self.fc2   = nn.Linear(128, n_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        # BUG 3: Missing Flatten (need to reshape before fc1)
        x = self.fc1(x)
        # BUG 4: Applying softmax before CrossEntropyLoss (double-softmax!)
        return torch.softmax(self.fc2(torch.relu(x)), dim=1)

model = BuggyCNN()
X = torch.randn(4, 1, 28, 28)
out = model(X)   # will error
'''
print(buggy)
print("""
Bug 1: ________________________________________________
Bug 2: ________________________________________________
Bug 3: ________________________________________________
Bug 4: ________________________________________________

TODO: Write the corrected CNN
""")

# ===========================================================================
# LEVEL 3: APPLICATION — Build a CNN for PCB Defect Detection
# ===========================================================================
print("\n--- LEVEL 3: APPLICATION ---")
print("""
TASK: Build a CNN to detect defects in synthetic PCB (circuit board) images.

PCB defect categories:
  Class 0: OK (no defect)
  Class 1: Missing component (dark spot)
  Class 2: Solder bridge (bright line between pads)
  Class 3: Component misalignment (off-center component)

Image size: 32x32, grayscale (1 channel)
n_samples: 200 per class

Requirements:
  1. Generate the 4-class synthetic PCB image dataset (use generate_pcb below)
  2. Build a CNN with:
     - 2 convolutional blocks (conv → bn → relu → pool)
     - Global average pooling before the FC head
     - Dropout(0.4) in the FC head
     - Output: 4 class logits
  3. Train with Adam(lr=0.001), CrossEntropyLoss, 40 epochs
  4. Plot training loss + validation accuracy curves
     (save to output/pcb_cnn_training.png)
  5. Report final test accuracy and classification report
  6. Compare with a flattened MLP on the same data
     (flatten 32x32 = 1024 features, 2 hidden layers)
     Which approach gets higher accuracy? Why?
""")

def generate_pcb_images(n_per_class=200, img_size=32):
    """Generate simple synthetic PCB defect images."""
    images, labels = [], []
    
    for cls in range(4):
        for _ in range(n_per_class):
            img = np.full((img_size, img_size), 0.2, dtype=np.float32)  # gray background
            
            # Add PCB traces (horizontal and vertical lines)
            for row in [8, 16, 24]:
                img[row-1:row+1, :] = 0.5
            for col in [8, 16, 24]:
                img[:, col-1:col+1] = 0.5
            
            if cls == 0:  # OK: add component in center
                img[13:19, 13:19] = 0.8
            elif cls == 1:  # Missing component: dark spot
                img[13:19, 13:19] = 0.0
            elif cls == 2:  # Solder bridge: bright line
                r = np.random.randint(5, img_size-5)
                img[r, 5:-5] = 1.0
            elif cls == 3:  # Misalignment: off-center component
                cx = 13 + np.random.randint(-4, 4)
                cy = 13 + np.random.randint(-4, 4)
                img[cx:cx+6, cy:cy+6] = 0.8
            
            noise = np.random.randn(img_size, img_size) * 0.05
            images.append(np.clip(img + noise, 0, 1))
            labels.append(cls)
    
    X = np.array(images)[:, np.newaxis, :, :]
    y = np.array(labels)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

X_pcb, y_pcb = generate_pcb_images(n_per_class=200)
print(f"PCB dataset: {X_pcb.shape}, classes: {np.bincount(y_pcb)}")
print("TODO: Build the CNN architecture above and train it")

# ===========================================================================
# LEVEL 4: CHALLENGE — Implement a Residual Block From Scratch
# ===========================================================================
print("\n--- LEVEL 4: CHALLENGE ---")
print("""
TASK: Implement a CNN with residual connections (ResNet-style).

A Residual Block:
    y = f(x) + x    (if input and output dimensions match)
    y = f(x) + g(x) (if dimensions differ → g is a 1×1 conv to match)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        # Skip connection: only needed if dimensions change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)      # ← SKIP CONNECTION
        return torch.relu(out)

Requirements:
  1. Implement ResidualBlock (template above)
  2. Build a small ResNet-like model using it:
     Input(1, 32, 32)
     → Conv(1→32) → BN → ReLU
     → ResidualBlock(32, 32)        (same dimensions)
     → ResidualBlock(32, 64, stride=2) (downsamples: 32→16)
     → ResidualBlock(64, 64)
     → AdaptiveAvgPool2d((1,1))
     → Flatten → Linear(64, 4)
  3. Train on PCB dataset from Level 3
  4. Compare accuracy with plain CNN (Level 3)
  5. Do residual connections help on this small dataset? Discuss why/why not.
""")

class ResidualBlock(nn.Module):
    """TODO: Implement this."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # TODO: implement (template provided above)
        pass
    
    def forward(self, x):
        # TODO: return f(x) + shortcut(x), then relu
        return x  # REPLACE

print("TODO: Implement ResidualBlock and test on PCB dataset")

# Quick test that block compiles
rb = ResidualBlock(32, 32)
x_rb = torch.randn(4, 32, 16, 16)
# out = rb(x_rb)  # Uncomment after implementing

print("\n" + "=" * 60)
print("EXERCISES COMPLETE")
print("Check solutions/cnn_solutions.py for reference answers.")
print("=" * 60)
