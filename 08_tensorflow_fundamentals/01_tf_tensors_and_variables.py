"""
01_tf_tensors_and_variables.py
==============================
Lesson 1: Fundamental Tensors and Variables in TensorFlow.

Demonstrates:
- Creating immutable tensors with tf.constant
- Creating mutable state with tf.Variable
- Tensor shapes, ranks, dimensions, and data types (tf.float32, tf.int32)
- Type casting with tf.cast
- In-place variable updates: assign, assign_add, assign_sub
- Tensor arithmetic, matrix multiplication (tf.matmul, @), and reductions
- Indexing, slicing, and reshaping (tf.reshape)
- The bidirectional NumPy bridge: tensor.numpy() and tf.convert_to_tensor
"""

import sys
from pathlib import Path

# Add current directory to path to load Python 3.14 compatibility engine
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tf_compat  # noqa: F401
import tensorflow as tf
import numpy as np

print("=" * 70)
print(f"TENSORFLOW FUNDAMENTALS — LESSON 1: TENSORS & VARIABLES (TF {tf.__version__})")
print("=" * 70)

# ============================================================================
# 1. IMMUTABLE TENSORS: tf.constant
# ============================================================================
print("\n--- 1. tf.constant (Immutable Tensors) ---")

# Scalar (rank 0)
scalar = tf.constant(3.14159, dtype=tf.float32)
print(f"Scalar: {scalar} | Shape: {scalar.shape} | Rank: {scalar.ndim}")

# Vector (rank 1)
vector = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
print(f"Vector: {vector} | Shape: {vector.shape} | Rank: {vector.ndim}")

# Matrix (rank 2)
matrix = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
print(f"Matrix:\n{matrix}")
print(f"Matrix shape: {matrix.shape} (rows={matrix.shape[0]}, cols={matrix.shape[1]})")

# Type Casting with tf.cast
int_tensor = tf.constant([1, 2, 3, 4], dtype=tf.int32)
float_converted = tf.cast(int_tensor, dtype=tf.float32)
print(f"Original int tensor: {int_tensor.dtype} -> Cast to: {float_converted.dtype}")

# Key Mental Model: tf.constant is IMMUTABLE
print("Immutability note: Once created, elements of a tf.constant cannot be reassigned.")


# ============================================================================
# 2. MUTABLE STATE: tf.Variable
# ============================================================================
print("\n--- 2. tf.Variable (Mutable Parameters) ---")
# Variables represent shared, persistent state manipulated by a program (e.g. neural network weights)
weights = tf.Variable([[0.5, -0.2], [0.1, 0.8]], trainable=True, name="layer_weights")
print(f"Initial Variable:\n{weights}")
print(f"Trainable flag: {weights.trainable}")

# In-place reassignment: assign()
print("\nMutating state:")
weights.assign([[1.0, 1.0], [1.0, 1.0]])
print(f"After .assign():\n{weights}")

# In-place addition: assign_add()
weights.assign_add([[0.5, 0.5], [0.5, 0.5]])
print(f"After .assign_add():\n{weights}")

# In-place subtraction: assign_sub() (Standard for Gradient Descent weight update!)
lr = 0.1
grad_dummy = tf.constant([[0.2, 0.1], [0.4, 0.3]])
weights.assign_sub(lr * grad_dummy)
print(f"After .assign_sub(lr * grad):\n{weights}")


# ============================================================================
# 3. TENSOR MATHEMATICAL OPERATIONS
# ============================================================================
print("\n--- 3. Mathematical Operations & Linear Algebra ---")
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# Element-wise operations
print(f"Element-wise addition (a + b):\n{a + b}")
print(f"Element-wise multiplication (a * b):\n{a * b}")

# Matrix multiplication (@ or tf.matmul)
matmul_res = a @ b
print(f"Matrix multiplication (a @ b):\n{matmul_res}")
assert np.allclose(matmul_res.numpy(), np.array([[19.0, 22.0], [43.0, 50.0]])), "Matmul error"

# Reductions
data = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(f"\nTensor:\n{data}")
print(f"Overall Mean: {tf.reduce_mean(data).numpy()}")
print(f"Column-wise Mean (axis=0): {tf.reduce_mean(data, axis=0).numpy()}")
print(f"Row-wise Sum (axis=1): {tf.reduce_sum(data, axis=1).numpy()}")


# ============================================================================
# 4. INDEXING, SLICING & RESHAPING
# ============================================================================
print("\n--- 4. Indexing, Slicing & Reshaping ---")
t = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=tf.int32)
print(f"3x3 Grid:\n{t}")
print(f"Element at row 1, col 2: {t[1, 2].numpy()}")
print(f"First two rows:\n{t[:2, :]}")
print(f"Last column:\n{t[:, -1]}")

# Reshaping
flat = tf.reshape(t, (9,))
print(f"Reshaped to 1D vector (9,): {flat}")
reconstructed = tf.reshape(flat, (3, 3))
print(f"Reconstructed to (3, 3):\n{reconstructed}")


# ============================================================================
# 5. THE NUMPY BRIDGE
# ============================================================================
print("\n--- 5. The NumPy Interoperability Bridge ---")
# NumPy to TensorFlow
np_array = np.array([1.5, 3.0, 4.5], dtype=np.float32)
tf_tensor = tf.convert_to_tensor(np_array)
print(f"NumPy array -> tf.Tensor: {tf_tensor}")

# TensorFlow to NumPy
back_to_np = tf_tensor.numpy()
print(f"tf.Tensor.numpy() -> NumPy array: {back_to_np} (Type: {type(back_to_np).__name__})")
assert isinstance(back_to_np, np.ndarray), "Failed to convert back to NumPy"

print("\n[*] Lesson 1 completed successfully.")


if __name__ == "__main__":
    pass
