"""
tensorflow_fundamentals_solutions.py
====================================
Complete reference solutions for Module 8: TensorFlow & Keras Fundamentals.
Contains 0 TODOs. Fully implemented and verified.
"""

import sys
from pathlib import Path

# Add TensorFlow module directory to path for tf_compat engine
TF_MODULE_DIR = Path(__file__).resolve().parent.parent / "08_tensorflow_fundamentals"
sys.path.insert(0, str(TF_MODULE_DIR))
import tf_compat  # noqa: F401
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("=" * 70)
print("SOLUTIONS: TENSORFLOW & KERAS FUNDAMENTALS (MODULE 8)")
print("=" * 70)

# ============================================================================
# SOLUTION TIER 1: RECALL & CONCEPTUAL FOUNDATIONS
# ============================================================================
RECALL_ANSWERS = {
    "1_constant_vs_variable": (
        "tf.constant represents immutable data arrays (like fixed inputs or constants). "
        "tf.Variable represents mutable state that can be modified in-place using .assign_sub(), "
        "making it ideal for tracking trainable model weights. By default, tf.GradientTape automatically "
        "watches all tf.Variable instances in its context, whereas a tf.constant must be explicitly "
        "registered via tape.watch(c) to record its operations."
    ),
    "2_gradient_clearing": (
        "In PyTorch, gradients accumulate in tensor.grad until explicitly cleared with optimizer.zero_grad(). "
        "In TensorFlow, each fresh `with tf.GradientTape() as tape:` context block starts with a clean slate, "
        "dynamically tracking operations executed inside the context. tape.gradient() returns a fresh list of "
        "gradient tensors without accumulating historical values."
    ),
    "3_keras_paradigms": (
        "Sequential API: A simple linear stack of layers with 1 input and 1 output. "
        "Functional API: Defines a directed acyclic graph (DAG) of layers, enabling multi-input, multi-output, "
        "and residual skip connections (x + residual). "
        "Model Subclassing: Imperative, object-oriented modeling overriding call(self, inputs, training=False), "
        "offering maximum flexibility for dynamic control flow and custom loops."
    ),
    "4_training_mode_boolean": (
        "The 'training' boolean flag controls stochastic and state-updating layers: "
        "During training=True, Dropout zeroes random activations with 1/(1-p) scaling, and BatchNormalization "
        "computes mini-batch mean/variance while updating exponential moving statistics. "
        "During training=False (eval/inference), Dropout acts as the identity pass, and BatchNormalization uses "
        "frozen moving statistics without updating."
    ),
}

# ============================================================================
# SOLUTION TIER 2: CORRECTED TRAINING STEP
# ============================================================================
def fix_buggy_training_step(model, optimizer, x_batch, y_batch):
    """
    Executes a single corrected forward pass, loss computation, gradient
    calculation, and optimizer parameter update within the GradientTape scope.
    """
    with tf.GradientTape() as tape:
        # 1. Forward pass with training mode enabled
        preds = model(x_batch, training=True)
        # 2. Loss computation INSIDE the tape context
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        loss = loss_fn(y_batch, preds)

    # 3. Extract gradients
    grads = tape.gradient(loss, model.trainable_variables)
    # 4. Apply gradients
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss.numpy())


# ============================================================================
# SOLUTION TIER 3: CUSTOM REGRESSION WITH HUBER LOSS & LR DECAY
# ============================================================================
class CustomRegressionTrainer:
    """
    Custom regression optimization loop with analytical Huber Loss and
    exponential learning rate decay.
    """

    def __init__(self, initial_lr: float = 0.05, decay_rate: float = 0.95, delta: float = 1.0):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.delta = delta
        self.w = tf.Variable(0.0, name="w")
        self.b = tf.Variable(0.0, name="b")

    def huber_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Huber Loss formulation:
        For |error| <= delta: 0.5 * error^2
        For |error| > delta:  delta * (|error| - 0.5 * delta)
        """
        error = y_true - y_pred
        abs_err = tf.abs(error)
        linear_mask = tf.cast(abs_err <= self.delta, tf.float32)

        quadratic = 0.5 * (error * error)
        linear = self.delta * (abs_err - 0.5 * self.delta)

        # Smooth piecewise blend
        return tf.reduce_mean(linear_mask * quadratic + (1.0 - linear_mask) * linear)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 30) -> list:
        """
        Executes manual gradient descent with learning rate decay:
        lr = initial_lr * (decay_rate ** epoch)
        """
        X_t = tf.constant(X.astype(np.float32))
        y_t = tf.constant(y.astype(np.float32))
        history = []

        for epoch in range(epochs):
            lr = self.initial_lr * (self.decay_rate ** epoch)

            with tf.GradientTape() as tape:
                y_pred = X_t * self.w + self.b
                loss = self.huber_loss(y_t, y_pred)

            dw, db = tape.gradient(loss, [self.w, self.b])
            self.w.assign_sub(lr * dw)
            self.b.assign_sub(lr * db)

            history.append(float(loss.numpy()))

        return history


# ============================================================================
# SOLUTION TIER 4: RESIDUAL BLOCK (FUNCTIONAL & SUBCLASSING)
# ============================================================================
def build_functional_residual(input_dim: int, units: int) -> keras.Model:
    """
    Builds a Residual MLP Block using Keras Functional API.
    F(x) = Dense(units)(Dense(units)(x)) + x
    """
    inputs = keras.Input(shape=(input_dim,), name="res_in")
    x = layers.Dense(units, activation="relu", name="res_dense1")(inputs)
    x = layers.Dense(units, activation="relu", name="res_dense2")(x)
    # Skip connection addition
    outputs = x + inputs
    return keras.Model(inputs=inputs, outputs=outputs, name="functional_res_block")


class SubclassedResidualBlock(keras.Model):
    """
    Builds a Residual MLP Block using Keras Model Subclassing.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(name="subclassed_res_block", **kwargs)
        self.dense1 = layers.Dense(units, activation="relu", name="sub_res_dense1")
        self.dense2 = layers.Dense(units, activation="relu", name="sub_res_dense2")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        fx = self.dense1(inputs)
        fx = self.dense2(fx)
        return fx + inputs


# ============================================================================
# VERIFICATION HARNESS
# ============================================================================
def verify_solutions():
    print("[*] Verifying Module 8 Reference Solutions...")

    # 1. Verify fix_buggy_training_step
    test_model = keras.Sequential([layers.Dense(4, activation="softmax")])
    test_opt = keras.optimizers.Adam(learning_rate=0.01)
    xb = tf.constant(np.random.randn(8, 6).astype(np.float32))
    yb = tf.constant(np.random.randint(0, 4, size=(8,)).astype(np.int64))
    step_loss = fix_buggy_training_step(test_model, test_opt, xb, yb)
    assert step_loss > 0, "Training step failed"
    print("  [+] Buggy Training Step Fix: PASS")

    # 2. Verify CustomRegressionTrainer
    trainer = CustomRegressionTrainer(initial_lr=0.5, delta=1.0, decay_rate=0.99)
    X_reg = np.linspace(-1, 1, 50).reshape(-1, 1)
    y_reg = 2.0 * X_reg + 0.5
    loss_hist = trainer.train(X_reg, y_reg, epochs=60)
    assert loss_hist[-1] < loss_hist[0], "Trainer failed to minimize loss"
    assert np.isclose(trainer.w.numpy(), 2.0, atol=0.2), "Slope failed convergence"
    print("  [+] Custom Regression with Huber Loss & LR Decay: PASS")

    # 3. Verify Residual Blocks
    func_res = build_functional_residual(input_dim=8, units=8)
    sub_res = SubclassedResidualBlock(units=8)

    sample_x = tf.constant(np.ones((4, 8), dtype=np.float32))
    out_f = func_res(sample_x)
    out_s = sub_res(sample_x)

    assert out_f.shape == (4, 8), f"Expected shape (4, 8), got {out_f.shape}"
    assert out_s.shape == (4, 8), f"Expected shape (4, 8), got {out_s.shape}"
    print("  [+] Residual Block Functional & Subclassing Parity: PASS")
    print("[*] All Module 8 Solutions verified successfully.")


if __name__ == "__main__":
    verify_solutions()
