"""
Module 10, Lesson 1: The Convolution Operation
===============================================

WHAT THIS LESSON TEACHES:
    The convolution operation is the core building block of all CNNs.
    Before using nn.Conv2d as a black box, we need to understand WHAT it
    computes: a sliding dot product between a small filter and local image patches.

WHY THIS MATTERS:
    Every time a CNN processes a medical scan, satellite photo, or selfie,
    it performs millions of these dot products. Understanding the operation
    lets you debug shape errors, design better architectures, and interpret
    what your model is actually learning.

LEARNING PATH:
    1. Manual convolution with NumPy (edge detection)
    2. Key parameters: stride, padding, multiple filters
    3. nn.Conv2d in PyTorch
    4. Visualizing feature maps
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # WHY: non-interactive backend for saving PNGs without display
import matplotlib.pyplot as plt
from pathlib import Path

# WHY: resolve() gives absolute path; .parent is the directory this script lives in
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# SECTION 1: WHAT IS A FILTER?
# =============================================================================

print("=" * 65)
print("SECTION 1: Filters as Pattern Detectors")
print("=" * 65)

# A filter is a small matrix of weights.
# Before training, weights are random. After training, they detect specific patterns.
# Classic example: the Sobel filter detects EDGES (gradient direction).

# Sobel filter for HORIZONTAL edges (bright top -> dark bottom = edge)
sobel_horizontal = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# Sobel filter for VERTICAL edges
sobel_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

print("Sobel horizontal filter (detects horizontal edges):")
print(sobel_horizontal)
print("\nIntuition: top row negative -> bright pixels above edge get subtracted")
print("           bottom row positive -> bright pixels below edge get added")
print("           Strong response ONLY where bright transitions to dark vertically")

# =============================================================================
# SECTION 2: MANUAL CONVOLUTION WITH NUMPY
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 2: Manual Convolution Step-by-Step")
print("=" * 65)

def convolve2d_manual(image, kernel, stride=1, padding=0):
    """
    Compute 2D convolution manually using nested loops.

    WHY implement manually?
        nn.Conv2d is fast but opaque. Seeing the loops makes it clear
        exactly what 'sliding a filter over an image' means mathematically.

    Args:
        image:   2D numpy array, shape (H, W)
        kernel:  2D numpy array, shape (kH, kW)
        stride:  step size between filter positions (default 1)
        padding: zero-padding around image borders (default 0)

    Returns:
        Feature map: 2D numpy array
    """
    H, W = image.shape
    kH, kW = kernel.shape

    # WHY padding? Without it, the output shrinks each layer.
    # padding=1 with a 3x3 kernel keeps H,W the same (same-padding).
    if padding > 0:
        image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
        H += 2 * padding
        W += 2 * padding

    # Output size formula: floor((in_size + 2*padding - kernel_size) / stride) + 1
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    output = np.zeros((out_H, out_W), dtype=np.float32)

    # The core operation: slide filter over every valid position
    for i in range(out_H):
        for j in range(out_W):
            # Extract the local patch (same size as kernel)
            row_start = i * stride
            col_start = j * stride
            patch = image[row_start:row_start + kH, col_start:col_start + kW]

            # Element-wise multiply + sum = dot product = one output value
            output[i, j] = np.sum(patch * kernel)

    return output


# Create a synthetic test image: bright square on dark background
image = np.zeros((16, 16), dtype=np.float32)
image[4:12, 4:12] = 1.0  # bright square in center

print("Input image (16x16, bright square in center):")
print(f"  Shape: {image.shape}")
print(f"  Min: {image.min():.1f}, Max: {image.max():.1f}")

# Apply Sobel filters
edge_h = convolve2d_manual(image, sobel_horizontal, stride=1, padding=0)
edge_v = convolve2d_manual(image, sobel_vertical, stride=1, padding=0)

print(f"\nHorizontal edge map shape: {edge_h.shape}")
print(f"  (shrinks by 2 in each dimension: 16-2=14, no padding)")
print(f"  Strong responses at top/bottom edges: {edge_h.max():.1f}")

edge_magnitude = np.sqrt(edge_h**2 + edge_v**2)
print(f"\nEdge magnitude (combined H+V):")
print(f"  Max response: {edge_magnitude.max():.1f}  <- at the square's borders")
print(f"  Interior response: {edge_magnitude[7,7]:.1f}  <- flat region = no edge")

# =============================================================================
# SECTION 3: KEY PARAMETER - STRIDE
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 3: Understanding Stride")
print("=" * 65)

# WHY stride > 1?
# - Stride 1: output nearly same size as input (fine-grained)
# - Stride 2: output half the spatial size (coarser, fewer computations)
# - Stride 2 can REPLACE MaxPooling (more modern approach, e.g. ResNet)

for stride in [1, 2]:
    result = convolve2d_manual(image, sobel_horizontal, stride=stride, padding=0)
    print(f"  Stride={stride}: output shape = {result.shape}")

print("\nRule: Larger stride -> smaller output -> fewer parameters downstream")
print("      Stride 2 approx= Conv + MaxPool in one step")

# =============================================================================
# SECTION 4: KEY PARAMETER - PADDING
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 4: Understanding Padding")
print("=" * 65)

for pad in [0, 1]:
    result = convolve2d_manual(image, sobel_horizontal, stride=1, padding=pad)
    print(f"  Padding={pad}: output shape = {result.shape}")

print("\nWith padding=1 and kernel_size=3:")
print("  Output same spatial size as input ('same' padding)")
print("  Border pixels participate in convolution")
print("  Used when you want to preserve spatial dimensions")

# =============================================================================
# SECTION 5: MULTIPLE FILTERS = MULTIPLE FEATURE MAPS
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 5: Multiple Filters -> Multiple Feature Maps")
print("=" * 65)

# WHY multiple filters?
# One filter detects ONE type of pattern.
# 32 filters detect 32 DIFFERENT patterns simultaneously.

filter_bank = {
    "Horizontal edges": sobel_horizontal,
    "Vertical edges":   sobel_vertical,
    "Top-left blob":    np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], dtype=np.float32),
    "Blur":             np.ones((3, 3), dtype=np.float32) / 9.0,
}

feature_maps = {}
for name, filt in filter_bank.items():
    fmap = convolve2d_manual(image, filt, stride=1, padding=1)
    feature_maps[name] = fmap
    print(f"  Filter '{name}': output shape = {fmap.shape}, max = {fmap.max():.2f}")

print(f"\nWith {len(filter_bank)} filters: output is shape")
print(f"  (n_filters=4, H=16, W=16) -- this is a 'feature map volume'")

# =============================================================================
# SECTION 6: nn.Conv2d IN PYTORCH
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 6: nn.Conv2d in PyTorch")
print("=" * 65)

# PyTorch uses 4D tensors for images: (batch, channels, height, width)
# WHY batch dimension? We process multiple images at once for efficiency.

image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # (1, 1, 16, 16)
print(f"Input tensor shape: {image_tensor.shape}  [batch, channels, H, W]")

conv_layer = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False)

with torch.no_grad():
    # WHY no_grad? We're manually setting weights, not doing a forward pass
    conv_layer.weight[0, 0] = torch.tensor(sobel_horizontal)
    conv_layer.weight[1, 0] = torch.tensor(sobel_vertical)

output_tensor = conv_layer(image_tensor)
print(f"Output tensor shape: {output_tensor.shape}  [batch, n_filters, H, W]")

# Verify manual matches PyTorch
manual_result = convolve2d_manual(image, sobel_horizontal, stride=1, padding=1)
pytorch_result = output_tensor[0, 0].detach().numpy()
max_diff = np.abs(manual_result - pytorch_result).max()
print(f"\nManual vs PyTorch Sobel max difference: {max_diff:.6f}")
print("  (Should be near zero -- same computation, different implementation)")

# =============================================================================
# SECTION 7: MULTI-CHANNEL CONVOLUTION (RGB images)
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 7: Multi-Channel Input (RGB images)")
print("=" * 65)

rgb_image = torch.randn(1, 3, 16, 16)  # batch=1, RGB, 16x16
print(f"RGB input shape: {rgb_image.shape}")

conv_rgb = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
output_rgb = conv_rgb(rgb_image)
print(f"RGB output shape: {output_rgb.shape}")

params = sum(p.numel() for p in conv_rgb.parameters())
print(f"Parameter count: {params}")
print(f"  = out_channels x in_channels x kH x kW + bias")
print(f"  = 16 x 3 x 3 x 3 + 16 = {16*3*3*3 + 16}")
print("\nCompare to dense layer: 16x16x3 inputs -> 16 outputs = 12,304 params")
print("WHY conv wins: shared weights across all 16x16 spatial locations")

# =============================================================================
# SECTION 8: VISUALIZATION
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 8: Visualizing Filters and Feature Maps")
print("=" * 65)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Convolution: Filters and Their Feature Maps\n"
             "Each filter detects a different pattern; output shows where that pattern appears",
             fontsize=12, fontweight='bold')

filter_list = list(filter_bank.items())
for col, (name, filt) in enumerate(filter_list):
    im0 = axes[0, col].imshow(filt, cmap='RdBu_r', vmin=-2, vmax=2)
    axes[0, col].set_title(f"Filter: {name}", fontsize=9, fontweight='bold')
    axes[0, col].set_xticks([])
    axes[0, col].set_yticks([])
    plt.colorbar(im0, ax=axes[0, col], fraction=0.046)

    fmap = feature_maps[name]
    im1 = axes[1, col].imshow(fmap, cmap='viridis')
    axes[1, col].set_title(f"Feature Map\n(filter response)", fontsize=9)
    axes[1, col].set_xticks([])
    axes[1, col].set_yticks([])
    plt.colorbar(im1, ax=axes[1, col], fraction=0.046)

axes[0, 0].set_ylabel("Filters\n(pattern detectors)", fontsize=9, fontweight='bold')
axes[1, 0].set_ylabel("Feature Maps\n(responses)", fontsize=9, fontweight='bold')

plt.tight_layout()
out_path = OUTPUT_DIR / "01_convolution_filters_featuremaps.png"
plt.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_path}")

# Edge detection on richer synthetic image
fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
fig2.suptitle("Edge Detection on Synthetic Image\n"
              "Sobel filters find gradient direction; magnitude = edge strength",
              fontsize=12, fontweight='bold')

x = np.linspace(0, 4 * np.pi, 64)
rich_image = np.outer(np.sin(x), np.cos(x))
rich_image = (rich_image - rich_image.min()) / (rich_image.max() - rich_image.min())

edge_h2 = convolve2d_manual(rich_image, sobel_horizontal, padding=1)
edge_v2 = convolve2d_manual(rich_image, sobel_vertical, padding=1)
edge_mag2 = np.sqrt(edge_h2**2 + edge_v2**2)

axes2[0].imshow(rich_image, cmap='gray')
axes2[0].set_title("Original Image\n(sin x cos pattern)", fontsize=9)
axes2[1].imshow(edge_h2, cmap='RdBu_r')
axes2[1].set_title("Horizontal Edges\n(Sobel-H response)", fontsize=9)
axes2[2].imshow(edge_v2, cmap='RdBu_r')
axes2[2].set_title("Vertical Edges\n(Sobel-V response)", fontsize=9)
axes2[3].imshow(edge_mag2, cmap='hot')
axes2[3].set_title("Edge Magnitude\nsqrt(H^2 + V^2)", fontsize=9)

for ax in axes2:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
out_path2 = OUTPUT_DIR / "01_edge_detection.png"
plt.savefig(out_path2, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {out_path2}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("SUMMARY: What You Learned")
print("=" * 65)
print("""
1. CONVOLUTION = sliding filter -> element-wise multiply -> sum
   Output shape: floor((in + 2p - k) / s) + 1

2. FILTERS are learnable pattern detectors
   - Before training: random noise
   - After training: edges, textures, shapes, objects

3. MULTIPLE FILTERS = multiple feature maps
   - Each filter looks for ONE pattern type
   - Stack of feature maps encodes MANY patterns simultaneously

4. KEY PARAMETERS:
   - kernel_size: larger -> bigger patterns, more parameters
   - stride: larger -> smaller output, faster computation
   - padding=1 (for k=3): preserves spatial dimensions

5. MULTI-CHANNEL: filter weight shape = (out_ch, in_ch, kH, kW)
   Dot product computed across ALL input channels at once

Next: pooling (spatial invariance) and full CNN architecture ->
""")
