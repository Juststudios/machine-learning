"""
exercises.py
============
Module 8: TensorFlow & Keras Fundamentals — 4-Tier Progressive Exercises.

Tiers:
- Tier 1: Recall & Conceptual Foundations
- Tier 2: Understanding & Debugging (4 Subtle TensorFlow/Keras Bugs)
- Tier 3: Application (Custom Regression Loop with Huber Loss & LR Decay)
- Tier 4: Challenge (Residual MLP Block across Functional & Subclassing Paradigms)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tf_compat  # noqa: F401
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("=" * 70)
print("MODULE 8: TENSORFLOW & KERAS FUNDAMENTALS — EXERCISES")
print("=" * 70)

# ============================================================================
# TIER 1: RECALL & CONCEPTUAL FOUNDATIONS
# ============================================================================
"""
Questions:
1. What is the difference between tf.constant and tf.Variable? Why does GradientTape
   track tf.Variable automatically, but requires tape.watch() for tf.constant?
2. In PyTorch, gradients are accumulated onto tensor.grad until optimizer.zero_grad().
   How does TensorFlow handle gradient clearing between iterations?
3. Contrast the 3 Keras paradigms: Sequential, Functional, and Subclassing.
   When is the Functional API strictly preferred over Sequential?
4. How does the 'training' boolean parameter in Keras call(inputs, training=...)
   affect Dropout and BatchNormalization layers?
"""

def demo_tier1():
    print("\n--- TIER 1 DEMO: Tape Tracking & Gradient Inspection ---")
    w = tf.Variable(2.5, name="weight")
    b = tf.Variable(1.0, name="bias")
    x = tf.constant(4.0)

    with tf.GradientTape() as tape:
        y = w * x + b  # y = 2.5 * 4 + 1 = 11.0

    dw, db = tape.gradient(y, [w, b])
    print(f"y = w * x + b -> y = {y.numpy()}")
    print(f"dy/dw = {dw.numpy()} (Expected: x = 4.0)")
    print(f"dy/db = {db.numpy()} (Expected: 1.0)")

demo_tier1()


# ============================================================================
# TIER 2: UNDERSTANDING & DEBUGGING
# ============================================================================
"""
The training script below contains 4 critical bugs:
- Bug 1: Trying to use .assign() on a tf.constant (only tf.Variable is mutable).
- Bug 2: GradientTape exits BEFORE computing the loss, causing tape.gradient() to return None!
- Bug 3: CategoricalCrossentropy used on integer class labels (should be SparseCategoricalCrossentropy).
- Bug 4: Testing/evaluating model without setting training=False, causing Dropout to remain active.
"""

buggy_tf_script = """
# BUG 1: Constant is immutable
w = tf.constant([1.0, 2.0])
w.assign([2.0, 3.0])

# BUG 2: Computing loss outside tape scope
with tf.GradientTape() as tape:
    y_pred = model(x)
loss = loss_fn(y_true, y_pred)  # Outside tape! tape cannot trace graph!
grads = tape.gradient(loss, model.trainable_variables)

# BUG 3: Loss mismatch
loss_fn = keras.losses.CategoricalCrossentropy()
loss = loss_fn(y_integer_labels, y_pred_probs)  # Needs SparseCategoricalCrossentropy!

# BUG 4: Evaluating with active dropout
eval_preds = model(x_test, training=True)  # Drops units during evaluation!
"""

# Exercise 2 Starter:
def fix_buggy_training_step(model, optimizer, x_batch, y_batch):
    """
    TODO for Student:
    Implement a corrected training step using:
    1. with tf.GradientTape() as tape:
    2. preds = model(x_batch, training=True)
    3. loss = keras.losses.SparseCategoricalCrossentropy()(y_batch, preds)
    4. grads = tape.gradient(loss, model.trainable_variables)
    5. optimizer.apply_gradients(zip(grads, model.trainable_variables))
    """
    pass


# ============================================================================
# TIER 3: APPLICATION — CUSTOM REGRESSION WITH HUBER LOSS & LR DECAY
# ============================================================================
class CustomRegressionTrainer:
    """
    TODO for Student:
    Implement a custom regression training loop from scratch using tf.GradientTape.
    Requirements:
    1. Huber Loss function with configurable delta (default 1.0):
       if |error| <= delta: 0.5 * error^2
       else: delta * (|error| - 0.5 * delta)
    2. Learning rate decay per epoch: lr = initial_lr * (decay_rate ** epoch)
    3. Tracks training loss per epoch and returns history list.
    """

    def __init__(self, initial_lr: float = 0.05, decay_rate: float = 0.95, delta: float = 1.0):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.delta = delta
        self.w = tf.Variable(0.0)
        self.b = tf.Variable(0.0)

    def huber_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """TODO: Implement Huber loss formula."""
        pass

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 30) -> list:
        """TODO: Execute manual gradient descent loop with learning rate decay."""
        pass


# ============================================================================
# TIER 4: CHALLENGE — RESIDUAL BLOCK (FUNCTIONAL & SUBCLASSING PARITY)
# ============================================================================
"""
TODO for Student:
Implement a Residual MLP Block:
  F(x) = Dense(units, activation='relu')(Dense(units, activation='relu')(x))
  Output = F(x) + x  (Skip connection)

Implement this architecture using:
1. build_functional_residual(input_dim, units) -> keras.Model
2. SubclassedResidualBlock(keras.Model)

Verify both models produce identical output shapes and have identical parameter counts.
"""

def build_functional_residual(input_dim: int, units: int) -> keras.Model:
    pass


class SubclassedResidualBlock(keras.Model):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        pass

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        pass


if __name__ == "__main__":
    print("\n[!] Module 8 exercise templates loaded.")
    print("Refer to machine-learning/solutions/tensorflow_fundamentals_solutions.py for reference implementations.")
