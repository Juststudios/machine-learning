"""
04_training_workflows.py
========================
Lesson 4: Deep Learning Training Workflows.

Demonstrates:
- Workflow 1: High-level declarative training via model.compile() and model.fit()
- Workflow 2: Custom imperative training with tf.GradientTape and optimizer.apply_gradients
- Workflow 3: PyTorch side-by-side comparison with its explicit 5-step training loop
- Callbacks: EarlyStopping with validation loss monitoring
- Convergence comparison across all three training paradigms
"""

import sys
from pathlib import Path

# Add current directory to path to load Python 3.14 compatibility engine
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tf_compat  # noqa: F401
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 70)
print("TENSORFLOW FUNDAMENTALS — LESSON 4: TRAINING WORKFLOWS")
print("=" * 70)

# ============================================================================
# Synthetic Dataset Generation (Binary Classification)
# ============================================================================
np.random.seed(42)
torch.manual_seed(42)

N = 600
X_np = np.random.randn(N, 6).astype(np.float32)
# True decision boundary: x0 + 1.5*x1 - 0.8*x2 > 0
logits_true = X_np[:, 0] + 1.5 * X_np[:, 1] - 0.8 * X_np[:, 2]
y_np = (logits_true > 0).astype(np.float32).reshape(-1, 1)

# Split into 80% train, 20% validation
split = int(0.8 * N)
X_train_np, X_val_np = X_np[:split], X_np[split:]
y_train_np, y_val_np = y_np[:split], y_np[split:]

print(f"Dataset: {N} samples total ({len(X_train_np)} train, {len(X_val_np)} val), 6 features.")


# ============================================================================
# WORKFLOW 1: HIGH-LEVEL DECLARATIVE (model.compile + model.fit)
# ============================================================================
print("\n--- Workflow 1: Declarative Training (model.compile & model.fit) ---")

model_wf1 = keras.Sequential([
    layers.Dense(16, activation="relu", name="wf1_dense1"),
    layers.Dropout(0.1, name="wf1_dropout"),
    layers.Dense(1, activation="sigmoid", name="wf1_output"),
], name="declarative_model")

model_wf1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

history = model_wf1.fit(
    X_train_np,
    y_train_np,
    epochs=25,
    batch_size=32,
    verbose=1,
    validation_data=(X_val_np, y_val_np),
    callbacks=[early_stopping],
)

eval_wf1 = model_wf1.evaluate(X_val_np, y_val_np, verbose=0)
print(f"Workflow 1 (model.fit) Final Val Loss: {eval_wf1[0]:.4f} | Val Accuracy: {eval_wf1[1]:.4f}")


# ============================================================================
# WORKFLOW 2: CUSTOM IMPERATIVE LOOP (tf.GradientTape)
# ============================================================================
print("\n--- Workflow 2: Custom Imperative Loop (tf.GradientTape) ---")

model_wf2 = keras.Sequential([
    layers.Dense(16, activation="relu", name="wf2_dense1"),
    layers.Dropout(0.1, name="wf2_dropout"),
    layers.Dense(1, activation="sigmoid", name="wf2_output"),
], name="custom_tape_model")

optimizer_wf2 = keras.optimizers.Adam(learning_rate=0.01)
loss_fn_wf2 = keras.losses.BinaryCrossentropy()

# Create tf.data.Dataset
batch_size = 32
train_ds = tf.data.Dataset.from_tensor_slices((X_train_np, y_train_np)).shuffle(100).batch(batch_size)

epochs = 25
print("Running custom tf.GradientTape training loop...")
for epoch in range(epochs):
    epoch_loss = []
    for step, (x_batch, y_batch) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            preds = model_wf2(x_batch, training=True)
            loss = loss_fn_wf2(y_batch, preds)

        # 1. Compute gradients explicitly
        grads = tape.gradient(loss, model_wf2.trainable_variables)
        # 2. Apply gradients to optimizer
        optimizer_wf2.apply_gradients(zip(grads, model_wf2.trainable_variables))
        epoch_loss.append(float(loss.numpy()))

    if (epoch + 1) % 5 == 0 or epoch == 0:
        val_preds = model_wf2(tf.constant(X_val_np), training=False).numpy()
        val_acc = np.mean((val_preds >= 0.5) == y_val_np)
        print(f"  Epoch {epoch + 1:02d}/{epochs:02d} | Train Loss: {np.mean(epoch_loss):.4f} | Val Acc: {val_acc:.4f}")

val_preds_wf2 = model_wf2(tf.constant(X_val_np), training=False).numpy()
val_acc_wf2 = float(np.mean((val_preds_wf2 >= 0.5) == y_val_np))
print(f"Workflow 2 (GradientTape) Final Val Accuracy: {val_acc_wf2:.4f}")


# ============================================================================
# WORKFLOW 3: PYTORCH SIDE-BY-SIDE COMPARISON
# ============================================================================
print("\n--- Workflow 3: PyTorch Side-by-Side Comparison ---")

class PyTorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

torch_model = PyTorchMLP()
torch_opt = optim.Adam(torch_model.parameters(), lr=0.01)
torch_crit = nn.BCELoss()

torch_X_train = torch.from_numpy(X_train_np)
torch_y_train = torch.from_numpy(y_train_np)
torch_X_val = torch.from_numpy(X_val_np)
torch_y_val = torch.from_numpy(y_val_np)

print("Running PyTorch 5-step loop:")
print("  1. optimizer.zero_grad()")
print("  2. predictions = model(batch)")
print("  3. loss = criterion(predictions, targets)")
print("  4. loss.backward()")
print("  5. optimizer.step()")

for epoch in range(epochs):
    torch_model.train()
    torch_opt.zero_grad()                        # 1. Zero grads
    preds = torch_model(torch_X_train)          # 2. Forward pass
    loss = torch_crit(preds, torch_y_train)     # 3. Loss computation
    loss.backward()                              # 4. Backward autograd
    torch_opt.step()                             # 5. Optimizer step

torch_model.eval()
with torch.no_grad():
    pt_val_preds = (torch_model(torch_X_val).numpy() >= 0.5).astype(np.float32)
    pt_val_acc = float(np.mean(pt_val_preds == y_val_np))

print(f"Workflow 3 (PyTorch) Final Val Accuracy: {pt_val_acc:.4f}")


# ============================================================================
# PARADIGM COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("WORKFLOW COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Workflow':<35} {'Lines of Code':<15} {'Control Level':<15} {'Val Accuracy':<15}")
print("-" * 70)
print(f"{'1. Keras model.fit()':<35} {'Minimal (1 line)':<15} {'Declarative':<15} {eval_wf1[1]:<15.4f}")
print(f"{'2. TF Custom GradientTape':<35} {'Moderate (~15)':<15} {'Complete':<15} {val_acc_wf2:<15.4f}")
print(f"{'3. PyTorch 5-step Loop':<35} {'Moderate (~15)':<15} {'Complete':<15} {pt_val_acc:<15.4f}")
print("-" * 70)
print("[*] Lesson 4 completed successfully.")


if __name__ == "__main__":
    pass
