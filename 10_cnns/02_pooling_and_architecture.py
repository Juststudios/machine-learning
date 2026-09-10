"""
Module 10, Lesson 2: Pooling and CNN Architecture
==================================================

WHAT THIS LESSON TEACHES:
    After convolution extracts local features, we need to:
    1. Make the network robust to small position shifts (pooling)
    2. Reduce spatial dimensions to keep computation manageable
    3. Organize these operations into a standard architectural pattern

WHY THIS MATTERS:
    The (Conv -> BatchNorm -> ReLU -> Pool) block is the workhorse of modern
    computer vision. ResNet, VGG, EfficientNet all use variants of this pattern.
    Understanding it lets you adapt any CNN architecture to your problem.

LEARNING PATH:
    1. MaxPooling2d and AveragePooling2d
    2. AdaptiveAvgPool2d: handle any input size
    3. Batch Normalization: why CNNs need it
    4. Standard CNN block: Conv -> BN -> ReLU -> Pool
    5. Building a complete small CNN
    6. Tracing shapes through the network
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# SECTION 1: MAX POOLING
# =============================================================================

print("=" * 65)
print("SECTION 1: MaxPooling2d - Keeping the Strongest Signal")
print("=" * 65)

# WHY max pooling?
# After a filter fires (detects its pattern), we don't care EXACTLY where.
# If an eye detector fires at pixel (12,15) or (13,15), it's the same eye.
# MaxPool collapses a 2x2 region into 1 number: the MAXIMUM activation.
# This provides TRANSLATION INVARIANCE: small shifts don't change the output.

# Example: 4x4 feature map
feature_map = torch.tensor([
    [1.0, 3.0, 2.0, 4.0],
    [5.0, 2.0, 0.0, 1.0],
    [3.0, 1.0, 7.0, 2.0],
    [0.0, 4.0, 3.0, 6.0]
]).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 4, 4)

max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
result_max = max_pool(feature_map)

print("Input feature map (4x4):")
print(feature_map.squeeze().numpy())
print(f"\nAfter MaxPool2d(kernel=2, stride=2) -> shape {result_max.shape}:")
print(result_max.squeeze().numpy())
print("\nInterpretation:")
print("  Top-left 2x2: [1,3,5,2] -> max = 5  (the 5 in row1,col0)")
print("  Top-right 2x2: [2,4,0,1] -> max = 4  (the 4 in row0,col3)")
print("  Bottom-left 2x2: [3,1,0,4] -> max = 4  (the 4 in row3,col1)")
print("  Bottom-right 2x2: [7,2,3,6] -> max = 7  (the 7 in row2,col2)")

# =============================================================================
# SECTION 2: AVERAGE POOLING
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 2: AveragePooling2d - Smoothed Summary")
print("=" * 65)

avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
result_avg = avg_pool(feature_map)

print("After AvgPool2d(kernel=2, stride=2):")
print(result_avg.squeeze().numpy())
print("\nMax pool vs Avg pool:")
print("  MaxPool: keep most PROMINENT feature (good for classification - 'was it there?')")
print("  AvgPool: smooth average (good for texture, density estimation)")
print("  Modern CNNs mostly use MaxPool or strided convolutions")

# =============================================================================
# SECTION 3: ADAPTIVE AVERAGE POOLING - The Modern Solution
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 3: AdaptiveAvgPool2d - Handle Any Input Size")
print("=" * 65)

# WHY adaptive pooling?
# Traditional pooling requires knowing the exact input size to set kernel/stride.
# AdaptiveAvgPool2d(output_size=(1,1)) collapses ANY spatial size to 1x1.
# This is how ResNet, VGG, etc. handle variable input sizes!
# It's essentially "global average pooling" (GAP).

adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

print("Testing AdaptiveAvgPool2d with different input sizes:")
for h, w in [(4, 4), (7, 7), (14, 14), (28, 28)]:
    x = torch.randn(1, 32, h, w)
    out = adaptive_pool(x)
    print(f"  Input: (1, 32, {h}, {w}) -> Output: {tuple(out.shape)}")

print("\nThis is how 'Global Average Pooling' works:")
print("  Conv layers extract features at any spatial resolution")
print("  GAP at the end collapses to (batch, channels, 1, 1)")
print("  Then flatten -> linear classifier")
print("  No need to know H,W in advance!")

# =============================================================================
# SECTION 4: BATCH NORMALIZATION
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 4: Batch Normalization - Stable Training")
print("=" * 65)

# WHY BatchNorm in CNNs?
# Without BN: activations can explode/vanish in deep networks.
# With BN: normalize feature maps to zero mean, unit variance per batch.
# Then LEARN to scale/shift: gamma * x_normalized + beta
# Benefits:
#   1. Higher learning rates (training is more stable)
#   2. Less sensitive to weight initialization
#   3. Slight regularization effect
#   4. Allows MUCH deeper networks to train

# For CNNs: BatchNorm2d normalizes across (batch, H, W) dimensions PER channel
batch_norm = nn.BatchNorm2d(num_features=16)  # one set of (gamma, beta) per channel

x = torch.randn(8, 16, 8, 8)  # batch=8, 16 channels, 8x8 spatial
print(f"Input shape: {x.shape}")

# Check mean/std before
print(f"Before BN: mean={x.mean():.3f}, std={x.std():.3f}")

bn_out = batch_norm(x)
# After BN: each channel is normalized across (batch, H, W)
print(f"After BN:  mean={bn_out.mean():.4f} (near 0), std={bn_out.std():.4f} (near 1)")

print("\nBN parameter count:", sum(p.numel() for p in batch_norm.parameters()))
print("  = 2 x num_features (gamma + beta, one per channel)")
print("  Very cheap! 32 parameters for 16-channel feature map.")

# =============================================================================
# SECTION 5: STANDARD CNN BLOCK
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 5: Standard CNN Block - Conv -> BN -> ReLU -> Pool")
print("=" * 65)

class ConvBNReLUPool(nn.Module):
    """
    Standard convolutional block used in most modern CNNs.

    WHY this order?
    - Conv: extract features (linear operation)
    - BN:   normalize activations for stable training
    - ReLU: introduce non-linearity (without this, all layers = one linear layer)
    - Pool: spatial invariance + downsampling

    Note: Some modern networks use pre-activation (BN -> ReLU -> Conv).
    The exact order is an active research area, but Conv->BN->ReLU->Pool
    is the most common and a safe default.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, bias=False),  # WHY bias=False? BN absorbs the bias
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # WHY inplace=True? Saves memory (modifies in place)
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# Test the block
block = ConvBNReLUPool(in_channels=1, out_channels=32)
x = torch.randn(4, 1, 16, 16)  # batch=4, grayscale, 16x16
out = block(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}  <- channels: 1->32, spatial: 16->8 (MaxPool halved it)")

# =============================================================================
# SECTION 6: BUILDING A COMPLETE CNN
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 6: Building a Complete CNN Architecture")
print("=" * 65)

class SimpleCNN(nn.Module):
    """
    A complete CNN for classifying 28x28 grayscale images into 2 classes.

    Architecture:
        Block 1: 1->16 channels, 28x28 -> 14x14
        Block 2: 16->32 channels, 14x14 -> 7x7
        Global Average Pooling: 7x7 -> 1x1
        Classifier: 32 -> 64 -> 2

    WHY this design?
    - Double channels as we halve spatial dims: keeps total computation roughly constant
    - GAP instead of flatten+large-FC: fewer parameters, less overfitting
    - Two FC layers: capacity to learn non-linear class boundaries
    """
    def __init__(self, num_classes=2):
        super().__init__()

        # Feature extractor: learns image representations
        self.features = nn.Sequential(
            ConvBNReLUPool(in_channels=1,  out_channels=16, pool=True),   # 28->14
            ConvBNReLUPool(in_channels=16, out_channels=32, pool=True),   # 14->7
            ConvBNReLUPool(in_channels=32, out_channels=64, pool=False),  # 7->7 (no pool)
        )

        # Global average pooling: (batch, 64, 7, 7) -> (batch, 64, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier: takes the 64-dim vector and predicts class
        self.classifier = nn.Sequential(
            nn.Flatten(),                # (batch, 64, 1, 1) -> (batch, 64)
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),             # WHY dropout? Regularization - prevent overfitting
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


model = SimpleCNN(num_classes=2)
print("SimpleCNN Architecture:")
print(model)

# =============================================================================
# SECTION 7: TRACING SHAPES THROUGH THE NETWORK
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 7: Shape Tracing - Understanding Data Flow")
print("=" * 65)

# This is one of the most important debugging skills for CNNs!
# Always trace shapes manually when you get dimension errors.

print("Tracing a 28x28 grayscale image through SimpleCNN:")
print()

x_trace = torch.randn(1, 1, 28, 28)  # single image
print(f"Input:                  {tuple(x_trace.shape)}")

with torch.no_grad():
    # Feature blocks
    x_trace = model.features[0](x_trace)
    print(f"After Block 1 (1->16):  {tuple(x_trace.shape)}  [MaxPool: 28->14]")

    x_trace = model.features[1](x_trace)
    print(f"After Block 2 (16->32): {tuple(x_trace.shape)}  [MaxPool: 14->7]")

    x_trace = model.features[2](x_trace)
    print(f"After Block 3 (32->64): {tuple(x_trace.shape)}  [no pool]")

    x_trace = model.gap(x_trace)
    print(f"After GAP:              {tuple(x_trace.shape)}  [7x7 -> 1x1]")

    x_trace = model.classifier[0](x_trace)  # Flatten
    print(f"After Flatten:          {tuple(x_trace.shape)}  [ready for Linear]")

    x_trace_out = model(torch.randn(1, 1, 28, 28))
    print(f"Final output:           {tuple(x_trace_out.shape)}  [2 class logits]")

# Parameter count analysis
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters:      {total_params:,}")
print(f"Trainable parameters:  {trainable:,}")
print(f"\nFor comparison: MLP with same depth on 28x28 images:")
mlp = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2))
mlp_params = sum(p.numel() for p in mlp.parameters())
print(f"  MLP parameters: {mlp_params:,}  ({mlp_params/total_params:.1f}x more than CNN)")
print("  AND the MLP loses all spatial structure!")

# =============================================================================
# SECTION 8: EFFECT OF POOLING ON SPATIAL INVARIANCE
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 8: Demonstrating Translation Invariance from Pooling")
print("=" * 65)

# WHY: show that pooling makes CNN robust to small translations
# Create two versions of an image: original and slightly shifted

def make_spot_image(size=16, spot_row=6, spot_col=6, spot_size=3):
    """Create image with a bright spot at given position."""
    img = np.zeros((size, size), dtype=np.float32)
    img[spot_row:spot_row+spot_size, spot_col:spot_col+spot_size] = 1.0
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)

img_original = make_spot_image(spot_row=6, spot_col=6)
img_shifted  = make_spot_image(spot_row=7, spot_col=7)  # shifted by 1 pixel

# Compare max-pooled versions
pool2x2 = nn.MaxPool2d(2, 2)
pool4x4 = nn.MaxPool2d(4, 4)

out_orig_2x = pool2x2(img_original).squeeze().numpy()
out_shft_2x = pool2x2(img_shifted).squeeze().numpy()

diff_no_pool = np.abs(img_original.numpy() - img_shifted.numpy()).max()
diff_pool2x  = np.abs(out_orig_2x - out_shft_2x).max()

out_orig_4x = pool4x4(img_original).squeeze().numpy()
out_shft_4x = pool4x4(img_shifted).squeeze().numpy()
diff_pool4x = np.abs(out_orig_4x - out_shft_4x).max()

print(f"Max difference (original vs 1-pixel shift):")
print(f"  No pooling:    {diff_no_pool:.3f}  <- images look very different pixel-by-pixel")
print(f"  MaxPool 2x2:   {diff_pool2x:.3f}  <- pooling absorbs the shift!")
print(f"  MaxPool 4x4:   {diff_pool4x:.3f}  <- larger pool = more invariance")
print("\nThis is WHY pooling is essential: small object movements don't fool the network.")

# =============================================================================
# SECTION 9: VISUALIZATION
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 9: Visualizations")
print("=" * 65)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Pooling Operations and CNN Architecture\n"
             "Pooling reduces spatial size while preserving the strongest features",
             fontsize=12, fontweight='bold')

# Test feature map
fm = feature_map.squeeze().numpy()

axes[0, 0].imshow(fm, cmap='hot', vmin=0, vmax=7)
axes[0, 0].set_title("Input Feature Map\n(4x4)", fontsize=10, fontweight='bold')
for i in range(4):
    for j in range(4):
        axes[0, 0].text(j, i, f'{fm[i,j]:.0f}', ha='center', va='center', fontsize=12, color='white')
axes[0, 0].set_xticks([])
axes[0, 0].set_yticks([])

# MaxPool result
mp = result_max.squeeze().numpy()
axes[0, 1].imshow(mp, cmap='hot', vmin=0, vmax=7)
axes[0, 1].set_title("After MaxPool2d(2,2)\n(2x2)", fontsize=10, fontweight='bold')
for i in range(2):
    for j in range(2):
        axes[0, 1].text(j, i, f'{mp[i,j]:.0f}', ha='center', va='center', fontsize=14, color='white')
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])

# AvgPool result
ap = result_avg.squeeze().numpy()
axes[0, 2].imshow(ap, cmap='hot', vmin=0, vmax=7)
axes[0, 2].set_title("After AvgPool2d(2,2)\n(2x2)", fontsize=10, fontweight='bold')
for i in range(2):
    for j in range(2):
        axes[0, 2].text(j, i, f'{ap[i,j]:.1f}', ha='center', va='center', fontsize=14, color='white')
axes[0, 2].set_xticks([])
axes[0, 2].set_yticks([])

# Translation invariance demo
axes[0, 3].bar(['No Pool', 'MaxPool 2x2', 'MaxPool 4x4'],
               [diff_no_pool, diff_pool2x, diff_pool4x],
               color=['#e74c3c', '#f39c12', '#27ae60'])
axes[0, 3].set_title("Translation Invariance\n(diff for 1-pixel shift)", fontsize=9, fontweight='bold')
axes[0, 3].set_ylabel("Max absolute difference")
axes[0, 3].set_ylim(0, diff_no_pool * 1.3)

# Bottom row: architecture diagram as text
axes[1, 0].axis('off')
arch_text = ("Conv Block Pattern:\n\n"
             "Conv2d(in, out, k=3, p=1)\n"
             "         |\n"
             "BatchNorm2d(out)\n"
             "         |\n"
             "    ReLU (non-linear)\n"
             "         |\n"
             "MaxPool2d(2, 2)\n"
             "         |\n"
             "  H,W halved, channels grow")
axes[1, 0].text(0.05, 0.95, arch_text, transform=axes[1, 0].transAxes,
                fontsize=9, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 0].set_title("Standard CNN Block", fontsize=10, fontweight='bold')

# Shape trace visualization
axes[1, 1].axis('off')
shapes = [
    ("Input", "(1, 1, 28, 28)"),
    ("Block 1", "(1, 16, 14, 14)"),
    ("Block 2", "(1, 32, 7, 7)"),
    ("Block 3", "(1, 64, 7, 7)"),
    ("GAP", "(1, 64, 1, 1)"),
    ("Flatten", "(1, 64)"),
    ("Output", "(1, 2)"),
]
for idx, (stage, shape) in enumerate(shapes):
    y = 0.9 - idx * 0.13
    axes[1, 1].text(0.1, y, f"{stage}:", fontsize=8, fontweight='bold',
                    transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.45, y, shape, fontsize=8, fontfamily='monospace',
                    transform=axes[1, 1].transAxes)
axes[1, 1].set_title("Shape Trace Through CNN", fontsize=10, fontweight='bold')

# Parameter comparison
axes[1, 2].bar(['CNN\n(SimpleCNN)', 'MLP\n(same depth)'],
               [total_params, mlp_params],
               color=['#3498db', '#e74c3c'])
axes[1, 2].set_title(f"Parameter Count\nCNN: {total_params:,} vs MLP: {mlp_params:,}", fontsize=9)
axes[1, 2].set_ylabel("# Parameters")

# BN effect
axes[1, 3].axis('off')
bn_text = ("Batch Normalization:\n\n"
           "For each channel c:\n"
           "  mu = mean(x_c over batch,H,W)\n"
           "  sigma = std(x_c)\n"
           "  x_norm = (x - mu) / sigma\n"
           "  output = gamma*x_norm + beta\n\n"
           "gamma, beta are LEARNED\n"
           "One pair per channel\n\n"
           "Benefits:\n"
           "  Higher learning rates OK\n"
           "  Less init sensitivity\n"
           "  Slight regularization")
axes[1, 3].text(0.05, 0.95, bn_text, transform=axes[1, 3].transAxes,
                fontsize=8, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
axes[1, 3].set_title("Batch Normalization", fontsize=10, fontweight='bold')

plt.tight_layout()
out_path = OUTPUT_DIR / "02_pooling_and_architecture.png"
plt.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_path}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("SUMMARY: CNN Design Principles")
print("=" * 65)
print("""
1. MAX POOLING: keep strongest activation per region
   - kernel=2, stride=2: halves spatial dimensions
   - Provides translation invariance (small shifts OK)

2. ADAPTIVE AVG POOLING: handle any spatial size
   - AdaptiveAvgPool2d((1,1)) = Global Average Pooling
   - Collapses spatial dims; output is (batch, channels, 1, 1)

3. BATCH NORMALIZATION:
   - Normalize feature maps per channel per batch
   - Enables deeper networks, higher learning rates
   - bias=False in Conv when followed by BN (BN has its own bias)

4. STANDARD CNN BLOCK: Conv -> BN -> ReLU -> Pool
   - Spatial dims: halved at each pool (28->14->7->...)
   - Channels: usually double (16->32->64->128...)
   - This keeps computation roughly constant per block

5. GLOBAL AVG POOLING at the end:
   - Replaces Flatten + huge Linear layer
   - Works with any input size
   - Fewer parameters, less overfitting

Next: Full CNN training on synthetic image data ->
""")
