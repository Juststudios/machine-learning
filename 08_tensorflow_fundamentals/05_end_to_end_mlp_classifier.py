"""
05_end_to_end_mlp_classifier.py
===============================
Lesson 5: Complete End-to-End Multi-Class MLP Classification Pipeline.

Demonstrates:
- Multi-class industrial equipment failure diagnosis dataset
- Data preprocessing & Z-score feature normalization
- Pipeline ETL with tf.data.Dataset (shuffling, batching)
- Deep MLP architecture combining Dense, BatchNormalization, Dropout, and Softmax
- Model training with validation monitoring and EarlyStopping
- Comprehensive evaluation: test accuracy, confusion matrix, precision, recall, F1
- Saving and reloading weights with inference verification
- Plotting loss and accuracy curves to output/tf_mlp_training_curves.png
"""

import sys
import os
from pathlib import Path

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add current directory to path to load Python 3.14 compatibility engine
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tf_compat  # noqa: F401
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 70)
print("TENSORFLOW FUNDAMENTALS — LESSON 5: END-TO-END MLP PIPELINE")
print("=" * 70)

# ============================================================================
# 1. DATASET GENERATION: 3-CLASS INDUSTRIAL DIAGNOSTICS
# ============================================================================
print("\n--- 1. Generating Industrial Diagnostic Dataset ---")
np.random.seed(42)
tf.random.set_seed(42)

# 3 Classes: 0 = Normal, 1 = Bearing Wear, 2 = Rotor Imbalance
N_per_class = 350
total_N = N_per_class * 3

# Class 0: Normal
X_c0 = np.random.multivariate_normal(
    mean=[45.0, 1.8, 28.0, 1800.0],
    cov=np.diag([4.0, 0.2, 5.0, 400.0]),
    size=N_per_class,
)
# Class 1: Bearing Wear (High vibration, high acoustic)
X_c1 = np.random.multivariate_normal(
    mean=[58.0, 5.2, 55.0, 1780.0],
    cov=np.diag([9.0, 0.6, 12.0, 500.0]),
    size=N_per_class,
)
# Class 2: Rotor Imbalance (High temperature, moderate vibration)
X_c2 = np.random.multivariate_normal(
    mean=[75.0, 3.8, 38.0, 1720.0],
    cov=np.diag([12.0, 0.4, 8.0, 800.0]),
    size=N_per_class,
)

X_all = np.vstack([X_c0, X_c1, X_c2]).astype(np.float32)
y_all = np.array([0] * N_per_class + [1] * N_per_class + [2] * N_per_class, dtype=np.int64)

# Shuffle
perm = np.random.permutation(total_N)
X_all, y_all = X_all[perm], y_all[perm]

# Train (70%) / Val (15%) / Test (15%) splits
n_train = int(0.70 * total_N)
n_val = int(0.15 * total_N)

X_train_raw, y_train = X_all[:n_train], y_all[:n_train]
X_val_raw, y_val = X_all[n_train : n_train + n_val], y_all[n_train : n_train + n_val]
X_test_raw, y_test = X_all[n_train + n_val :], y_all[n_train + n_val :]

# Standardization (Z-score)
mean = np.mean(X_train_raw, axis=0)
std = np.std(X_train_raw, axis=0) + 1e-7

X_train = (X_train_raw - mean) / std
X_val = (X_val_raw - mean) / std
X_test = (X_test_raw - mean) / std

print(f"Dataset partitioned: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
print(f"Features: [Temperature, Vibration, Acoustic Emission, Speed]")


# ============================================================================
# 2. ETL PIPELINE WITH tf.data.Dataset
# ============================================================================
print("\n--- 2. Building tf.data.Dataset Pipelines ---")
BATCH_SIZE = 32
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=len(X_train), seed=42)
    .batch(BATCH_SIZE)
)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
print(f"Constructed input pipeline with batch size {BATCH_SIZE}.")


# ============================================================================
# 3. MODEL ARCHITECTURE (DEEP MLP CLASSIFIER)
# ============================================================================
print("\n--- 3. Defining Keras MLP Classifier Architecture ---")
model = keras.Sequential([
    layers.Dense(32, activation="relu", name="fc1"),
    layers.BatchNormalization(name="bn1"),
    layers.Dropout(0.2, name="dropout1"),
    layers.Dense(16, activation="relu", name="fc2"),
    layers.BatchNormalization(name="bn2"),
    layers.Dense(3, activation="softmax", name="output_logits"),
], name="industrial_fault_mlp")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.summary()


# ============================================================================
# 4. TRAINING WITH EARLY STOPPING
# ============================================================================
print("\n--- 4. Training Model with Validation Early Stopping ---")
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

history = model.fit(
    X_train,
    y_train,
    epochs=35,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
)


# ============================================================================
# 5. EVALUATION: TEST ACCURACY & CONFUSION MATRIX
# ============================================================================
print("\n--- 5. Evaluating Model on Held-Out Test Set ---")
test_res = model.evaluate(X_test, y_test, verbose=0)
test_loss = test_res[0]
test_acc = test_res[1] if len(test_res) > 1 else 0.0

test_preds = model.predict(X_test)
y_pred_classes = np.argmax(test_preds, axis=1)

print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# Compute Confusion Matrix
n_classes = 3
conf_mat = np.zeros((n_classes, n_classes), dtype=np.int32)
for true_lbl, pred_lbl in zip(y_test, y_pred_classes):
    conf_mat[true_lbl, pred_lbl] += 1

print("\nConfusion Matrix (Rows = True Class, Cols = Predicted Class):")
print(f"{'':<12} {'Pred Class 0':<15} {'Pred Class 1':<15} {'Pred Class 2':<15}")
for r in range(n_classes):
    class_names = ["0 (Normal)", "1 (Bearing)", "2 (Rotor)"]
    print(f"{class_names[r]:<12} {conf_mat[r, 0]:<15} {conf_mat[r, 1]:<15} {conf_mat[r, 2]:<15}")

# Per-class metrics
for c in range(n_classes):
    tp = conf_mat[c, c]
    fp = np.sum(conf_mat[:, c]) - tp
    fn = np.sum(conf_mat[c, :]) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"  Class {c}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")


# ============================================================================
# 6. WEIGHT PERSISTENCE & INFERENCE VERIFICATION
# ============================================================================
print("\n--- 6. Model Weight Persistence (Save & Reload) ---")
weights_path = str(OUT_DIR / "mlp_weights.npz")
model.save_weights(weights_path)
print(f"Saved model weights to {weights_path}")

# Construct fresh model and load weights
new_model = keras.Sequential([
    layers.Dense(32, activation="relu", name="fc1"),
    layers.BatchNormalization(name="bn1"),
    layers.Dropout(0.2, name="dropout1"),
    layers.Dense(16, activation="relu", name="fc2"),
    layers.BatchNormalization(name="bn2"),
    layers.Dense(3, activation="softmax", name="output_logits"),
], name="reloaded_mlp")
# Initialize shapes
_ = new_model(tf.constant(X_test[:2]))
new_model.load_weights(weights_path)

reloaded_preds = new_model.predict(X_test)
assert np.allclose(test_preds, reloaded_preds, atol=1e-5), "Reloaded model outputs differ!"
print("[+] Weight reload verified: 100% numerical parity confirmed.")


# ============================================================================
# 7. DIAGNOSTIC TRAINING PLOT EXPORT
# ============================================================================
print("\n--- 7. Exporting Training Curves ---")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

epochs_range = range(1, len(history.history["loss"]) + 1)

# Loss plot
axes[0].plot(epochs_range, history.history["loss"], "b-o", label="Training Loss")
if "val_loss" in history.history:
    axes[0].plot(epochs_range, history.history["val_loss"], "r--s", label="Validation Loss")
axes[0].set_title("Cross-Entropy Loss vs Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, linestyle="--", alpha=0.6)
axes[0].legend()

# Accuracy plot
if "accuracy" in history.history:
    axes[1].plot(epochs_range, history.history["accuracy"], "b-o", label="Training Accuracy")
if "val_accuracy" in history.history:
    axes[1].plot(epochs_range, history.history["val_accuracy"], "r--s", label="Validation Accuracy")
axes[1].set_title("Classification Accuracy vs Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].grid(True, linestyle="--", alpha=0.6)
axes[1].legend()

plot_path = OUT_DIR / "tf_mlp_training_curves.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Exported diagnostic plot to: {plot_path}")
print("\n[*] Lesson 5 completed successfully.")


if __name__ == "__main__":
    pass
