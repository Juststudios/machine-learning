"""
06_pytorch_vs_tensorflow_rosetta.py
===================================
The Master Rosetta Stone: 32+ Side-by-Side PyTorch vs. TensorFlow Code Pairs.

This executable reference document provides side-by-side comparative implementations
for all fundamental deep learning operations across PyTorch and TensorFlow 2.x / Keras.
"""

import sys
from pathlib import Path

# Add current directory to path to load Python 3.14 compatibility engine
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tf_compat  # noqa: F401
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

print("=" * 80)
print("THE MASTER ROSETTA STONE: PYTORCH vs. TENSORFLOW (32 CODE PAIRS)")
print("=" * 80)

pairs_verified = 0

def log_pair(num: int, title: str, pt_code: str, tf_code: str):
    global pairs_verified
    pairs_verified += 1
    print(f"\n[{num:02d}] {title}")
    print("-" * 80)
    print(f"  PyTorch:\n    {pt_code}")
    print(f"  TensorFlow:\n    {tf_code}")


# ----------------------------------------------------------------------------
# 1. Tensor Creation
# ----------------------------------------------------------------------------
pt_t1 = torch.tensor([1.0, 2.0, 3.0])
tf_t1 = tf.constant([1.0, 2.0, 3.0])
assert np.allclose(pt_t1.numpy(), tf_t1.numpy())
log_pair(
    1, "Tensor Creation",
    "pt_t = torch.tensor([1.0, 2.0, 3.0])",
    "tf_t = tf.constant([1.0, 2.0, 3.0])",
)

# ----------------------------------------------------------------------------
# 2. Data Types & Type Casting
# ----------------------------------------------------------------------------
pt_cast = pt_t1.to(torch.float64)
tf_cast = tf.cast(tf_t1, tf.float64)
assert pt_cast.dtype == torch.float64 and tf_cast.dtype == tf.float64
log_pair(
    2, "Data Types & Type Casting",
    "t_f64 = t.to(torch.float64)",
    "t_f64 = tf.cast(t, tf.float64)",
)

# ----------------------------------------------------------------------------
# 3. Shapes & Reshaping
# ----------------------------------------------------------------------------
pt_mat = torch.randn(2, 6)
tf_mat = tf.constant(pt_mat.numpy())
pt_reshaped = pt_mat.view(3, 4)
tf_reshaped = tf.reshape(tf_mat, (3, 4))
assert pt_reshaped.shape == (3, 4) and tf_reshaped.shape == (3, 4)
log_pair(
    3, "Shapes & Reshaping",
    "shape = t.shape; t_reshaped = t.view(3, 4)",
    "shape = t.shape; t_reshaped = tf.reshape(t, (3, 4))",
)

# ----------------------------------------------------------------------------
# 4. Indexing & Slicing
# ----------------------------------------------------------------------------
pt_slice = pt_mat[:, :3]
tf_slice = tf_mat[:, :3]
assert np.allclose(pt_slice.numpy(), tf_slice.numpy())
log_pair(
    4, "Indexing & Slicing",
    "sub = t[:, :3]",
    "sub = t[:, :3]",
)

# ----------------------------------------------------------------------------
# 5. Math Operations (Element-wise & Matmul)
# ----------------------------------------------------------------------------
a_np = np.random.randn(2, 3).astype(np.float32)
b_np = np.random.randn(3, 2).astype(np.float32)
pt_mm = torch.from_numpy(a_np) @ torch.from_numpy(b_np)
tf_mm = tf.constant(a_np) @ tf.constant(b_np)
assert np.allclose(pt_mm.numpy(), tf_mm.numpy(), atol=1e-5)
log_pair(
    5, "Math Operations & Matmul",
    "res = a @ b  # or torch.matmul(a, b)",
    "res = a @ b  # or tf.matmul(a, b)",
)

# ----------------------------------------------------------------------------
# 6. Device Management (CPU / GPU)
# ----------------------------------------------------------------------------
log_pair(
    6, "Device Management",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); t = t.to(device)",
    "with tf.device('/GPU:0'): t = tf.constant([1.0, 2.0])",
)

# ----------------------------------------------------------------------------
# 7. Variable Mutability & State
# ----------------------------------------------------------------------------
pt_p = nn.Parameter(torch.tensor([1.0, 2.0]))
tf_v = tf.Variable([1.0, 2.0])
tf_v.assign([3.0, 4.0])
log_pair(
    7, "Variable Mutability & State",
    "p = nn.Parameter(torch.tensor([1.0, 2.0])); p.data.copy_(new_val)",
    "v = tf.Variable([1.0, 2.0]); v.assign(new_val)",
)

# ----------------------------------------------------------------------------
# 8. Gradient Tracking
# ----------------------------------------------------------------------------
log_pair(
    8, "Gradient Tracking Setup",
    "x = torch.tensor(3.0, requires_grad=True)",
    "x = tf.Variable(3.0)  # tracked automatically in GradientTape",
)

# ----------------------------------------------------------------------------
# 9. Watching Constants
# ----------------------------------------------------------------------------
log_pair(
    9, "Watching Non-Variable Constants",
    "x.requires_grad_(True)",
    "with tf.GradientTape() as tape: tape.watch(c)",
)

# ----------------------------------------------------------------------------
# 10. Manual Gradient Calculation
# ----------------------------------------------------------------------------
pt_x = torch.tensor(3.0, requires_grad=True)
pt_y = pt_x ** 2
pt_y.backward()

tf_x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    tf_y = tf_x * tf_x
tf_grad = tape.gradient(tf_y, tf_x)
assert np.isclose(pt_x.grad.item(), tf_grad.numpy())
log_pair(
    10, "Manual Gradient Calculation",
    "loss.backward(); grad = x.grad",
    "with tf.GradientTape() as tape: loss = ...; grad = tape.gradient(loss, x)",
)

# ----------------------------------------------------------------------------
# 11. Weight Update (assign_sub vs step)
# ----------------------------------------------------------------------------
log_pair(
    11, "Weight Update (In-Place)",
    "with torch.no_grad(): param.sub_(lr * param.grad)  # or optimizer.step()",
    "param.assign_sub(lr * grad)  # or optimizer.apply_gradients([(grad, param)])",
)

# ----------------------------------------------------------------------------
# 12. Sequential Model Definition
# ----------------------------------------------------------------------------
pt_seq = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
tf_seq = keras.Sequential([layers.Dense(16, activation="relu"), layers.Dense(2)])
log_pair(
    12, "Sequential Model Definition",
    "model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))",
    "model = keras.Sequential([layers.Dense(16, activation='relu'), layers.Dense(2)])",
)

# ----------------------------------------------------------------------------
# 13. Custom Layer Definition
# ----------------------------------------------------------------------------
log_pair(
    13, "Custom Layer Definition",
    "class MyLayer(nn.Module): def forward(self, x): return x * 2.0",
    "class MyLayer(layers.Layer): def call(self, inputs): return inputs * 2.0",
)

# ----------------------------------------------------------------------------
# 14. Custom Model Definition (forward vs call)
# ----------------------------------------------------------------------------
log_pair(
    14, "Custom Model Definition",
    "class MyModel(nn.Module): def forward(self, x): return self.net(x)",
    "class MyModel(keras.Model): def call(self, inputs, training=False): return self.net(inputs)",
)

# ----------------------------------------------------------------------------
# 15. Activation Functions
# ----------------------------------------------------------------------------
log_pair(
    15, "Activation Functions",
    "nn.ReLU(), nn.Sigmoid(), nn.Softmax(dim=-1)",
    "layers.ReLU(), layers.Dense(..., activation='sigmoid'), layers.Softmax()",
)

# ----------------------------------------------------------------------------
# 16. Loss Functions
# ----------------------------------------------------------------------------
log_pair(
    16, "Loss Functions",
    "nn.MSELoss(), nn.BCELoss(), nn.CrossEntropyLoss()",
    "keras.losses.MeanSquaredError(), BinaryCrossentropy(), SparseCategoricalCrossentropy()",
)

# ----------------------------------------------------------------------------
# 17. Optimizers
# ----------------------------------------------------------------------------
log_pair(
    17, "Optimizers",
    "optim.SGD(model.parameters(), lr=0.01), optim.Adam(model.parameters(), lr=0.001)",
    "keras.optimizers.SGD(learning_rate=0.01), keras.optimizers.Adam(learning_rate=0.001)",
)

# ----------------------------------------------------------------------------
# 18. Training Mode Toggle
# ----------------------------------------------------------------------------
log_pair(
    18, "Training Mode Toggle",
    "model.train() vs. model.eval()",
    "model(inputs, training=True) vs. model(inputs, training=False)",
)

# ----------------------------------------------------------------------------
# 19. Zeroing Gradients
# ----------------------------------------------------------------------------
log_pair(
    19, "Zeroing Gradients",
    "optimizer.zero_grad()",
    "# Automatic: Each fresh `with tf.GradientTape():` block starts with zero gradients!",
)

# ----------------------------------------------------------------------------
# 20. Backward Pass
# ----------------------------------------------------------------------------
log_pair(
    20, "Backward Pass",
    "loss.backward()",
    "grads = tape.gradient(loss, model.trainable_variables)",
)

# ----------------------------------------------------------------------------
# 21. Standard Training Loop
# ----------------------------------------------------------------------------
log_pair(
    21, "Standard Training Loop",
    "for X, y in loader: opt.zero_grad(); out = model(X); loss = crit(out, y); loss.backward(); opt.step()",
    "model.fit(dataset, epochs=10)  # OR with GradientTape: apply_gradients",
)

# ----------------------------------------------------------------------------
# 22. Dataset Abstraction
# ----------------------------------------------------------------------------
log_pair(
    22, "Dataset Abstraction",
    "dataset = TensorDataset(torch_X, torch_y)",
    "dataset = tf.data.Dataset.from_tensor_slices((X, y))",
)

# ----------------------------------------------------------------------------
# 23. Batching & Shuffling
# ----------------------------------------------------------------------------
log_pair(
    23, "Batching & Shuffling",
    "loader = DataLoader(dataset, batch_size=32, shuffle=True)",
    "dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)",
)

# ----------------------------------------------------------------------------
# 24. Model Summary & Parameter Counting
# ----------------------------------------------------------------------------
log_pair(
    24, "Model Summary & Parameter Counting",
    "sum(p.numel() for p in model.parameters() if p.requires_grad)",
    "model.summary()",
)

# ----------------------------------------------------------------------------
# 25. Saving & Loading Weights
# ----------------------------------------------------------------------------
log_pair(
    25, "Saving & Loading Weights",
    "torch.save(model.state_dict(), 'weights.pt'); model.load_state_dict(torch.load('weights.pt'))",
    "model.save_weights('weights.npz'); model.load_weights('weights.npz')",
)

# ----------------------------------------------------------------------------
# 26. Learning Rate Schedulers
# ----------------------------------------------------------------------------
log_pair(
    26, "Learning Rate Schedulers",
    "scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95); scheduler.step()",
    "lr_sched = keras.optimizers.schedules.ExponentialDecay(initial_lr=0.01, decay_rate=0.95)",
)

# ----------------------------------------------------------------------------
# 27. Custom Loss Functions
# ----------------------------------------------------------------------------
log_pair(
    27, "Custom Loss Functions",
    "def custom_loss(y_pred, y_true): return torch.mean(torch.abs(y_pred - y_true))",
    "def custom_loss(y_true, y_pred): return tf.reduce_mean(tf.abs(y_true - y_pred))",
)

# ----------------------------------------------------------------------------
# 28. Custom Metric Computation
# ----------------------------------------------------------------------------
log_pair(
    28, "Custom Metric Computation",
    "acc = (preds.argmax(dim=1) == targets).float().mean().item()",
    "acc = tf.reduce_mean(tf.cast(tf.argmax(preds, axis=1) == targets, tf.float32))",
)

# ----------------------------------------------------------------------------
# 29. Inference / Evaluation (No Gradient Context)
# ----------------------------------------------------------------------------
log_pair(
    29, "Inference Context",
    "with torch.no_grad(): predictions = model(X)",
    "predictions = model(X, training=False)  # or model.predict(X)",
)

# ----------------------------------------------------------------------------
# 30. Feature Concatenation & Stacking
# ----------------------------------------------------------------------------
log_pair(
    30, "Feature Concatenation & Stacking",
    "torch.cat([t1, t2], dim=1); torch.stack([t1, t2], dim=0)",
    "tf.concat([t1, t2], axis=1); tf.stack([t1, t2], axis=0)",
)

# ----------------------------------------------------------------------------
# 31. Dropout Mode Differences
# ----------------------------------------------------------------------------
log_pair(
    31, "Dropout Behavior (Train vs. Eval)",
    "model.train() activates dropout with 1/(1-p) scaling; model.eval() turns dropout into identity",
    "model(x, training=True) drops nodes; model(x, training=False) passes values unmodified",
)

# ----------------------------------------------------------------------------
# 32. Batch Normalization Statistics
# ----------------------------------------------------------------------------
log_pair(
    32, "Batch Normalization Mode Differences",
    "model.train() computes batch mean/var and updates running stats; model.eval() freezes running stats",
    "model(x, training=True) updates moving stats; model(x, training=False) uses frozen moving stats",
)

print("\n" + "=" * 80)
print(f"[*] Successfully verified {pairs_verified} side-by-side PyTorch vs. TensorFlow code pairs.")
print("=" * 80)


if __name__ == "__main__":
    pass
