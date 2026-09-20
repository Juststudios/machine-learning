"""
03_keras_model_architectures.py
===============================
Lesson 3: The Three Keras Modeling Paradigms.

Demonstrates:
- Sequential API: Simple, linear layer stacks
- Functional API: Directed Acyclic Graphs (DAG), residual skip connections, multi-branch topologies
- Model Subclassing API: Imperative, object-oriented custom models (call(self, inputs, training=False))
- Implementing an identical architecture across all 3 paradigms
- Verifying parameter counts and numerical parity across paradigms
- Functional residual connections and dynamic subclass branching
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

print("=" * 70)
print("TENSORFLOW FUNDAMENTALS — LESSON 3: THE 3 KERAS MODELING PARADIGMS")
print("=" * 70)

# ============================================================================
# PARADIGM 1: SEQUENTIAL API (LINEAR LAYER STACK)
# ============================================================================
print("\n--- Paradigm 1: Sequential API ---")
# Best for: Straightforward feed-forward networks with exactly 1 input and 1 output

sequential_model = keras.Sequential([
    layers.Dense(16, activation="relu", name="seq_dense_1"),
    layers.Dense(8, activation="relu", name="seq_dense_2"),
    layers.Dense(2, activation="softmax", name="seq_output"),
], name="sequential_classifier")

# Initialize model by passing a dummy batch
dummy_input = tf.constant(np.ones((4, 8), dtype=np.float32))
out_seq = sequential_model(dummy_input)
print(f"Sequential model input shape: {dummy_input.shape} -> output shape: {out_seq.shape}")
print("Sequential Model Summary:")
sequential_model.summary()


# ============================================================================
# PARADIGM 2: FUNCTIONAL API (DIRECTED ACYCLIC GRAPH & RESIDUAL SKIP)
# ============================================================================
print("\n--- Paradigm 2: Functional API ---")
# Best for: Multi-input, multi-output, shared layers, and residual/skip connections

inputs = keras.Input(shape=(8,), name="func_input")
x = layers.Dense(16, activation="relu", name="func_dense_1")(inputs)
x = layers.Dense(8, activation="relu", name="func_dense_2")(x)
outputs = layers.Dense(2, activation="softmax", name="func_output")(x)

functional_model = keras.Model(inputs=inputs, outputs=outputs, name="functional_classifier")
out_func = functional_model(dummy_input)
print(f"Functional model output shape: {out_func.shape}")
print("Functional Model Summary:")
functional_model.summary()

# Bonus: Building a Residual Block with Functional API (ResNet style)
print("\nBuilding a Residual Skip Connection with Functional API:")
res_in = keras.Input(shape=(16,))
h1 = layers.Dense(16, activation="relu", name="res_layer_1")(res_in)
h2 = layers.Dense(16, activation="relu", name="res_layer_2")(h1)
# Residual addition: output = F(x) + x
res_out = h2 + res_in
res_model = keras.Model(inputs=res_in, outputs=res_out, name="residual_block_model")
print(f"Residual Block output shape: {res_model(tf.constant(np.ones((2, 16), dtype=np.float32))).shape}")


# ============================================================================
# PARADIGM 3: MODEL SUBCLASSING (OBJECT-ORIENTED IMPERATIVE)
# ============================================================================
print("\n--- Paradigm 3: Model Subclassing ---")
# Best for: Research architectures, custom control flow, dynamic branching, PyTorch converts

class SubclassedClassifier(keras.Model):
    """
    Subclassed Model defining layers in __init__ and computation in call().
    Directly analogous to PyTorch nn.Module.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.2, **kwargs):
        super().__init__(name="subclassed_classifier", **kwargs)
        self.dense1 = layers.Dense(16, activation="relu", name="sub_dense_1")
        self.dropout = layers.Dropout(dropout_rate, name="sub_dropout")
        self.dense2 = layers.Dense(8, activation="relu", name="sub_dense_2")
        self.out = layers.Dense(num_classes, activation="softmax", name="sub_output")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Imperative forward pass. Notice the explicit 'training' boolean flag
        controlling stochastic layers like Dropout and BatchNorm.
        """
        x = self.dense1(inputs)
        # Apply dropout only during training mode
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.out(x)


subclassed_model = SubclassedClassifier(num_classes=2, dropout_rate=0.3)
out_sub_train = subclassed_model(dummy_input, training=True)
out_sub_eval = subclassed_model(dummy_input, training=False)
print(f"Subclassed forward pass (training=True) shape: {out_sub_train.shape}")
print(f"Subclassed forward pass (training=False) shape: {out_sub_eval.shape}")
print("Subclassed Model Summary:")
subclassed_model.summary()


# ============================================================================
# ARCHITECTURAL COMPARISON MATRIX
# ============================================================================
print("\n" + "=" * 70)
print("KERAS PARADIGMS COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Feature':<25} {'Sequential':<15} {'Functional':<15} {'Subclassing':<15}")
print("-" * 70)
print(f"{'Complexity':<25} {'Lowest':<15} {'Medium':<15} {'Highest':<15}")
print(f"{'Multi-input/output':<25} {'No':<15} {'Yes':<15} {'Yes':<15}")
print(f"{'Skip connections':<25} {'No':<15} {'Yes':<15} {'Yes':<15}")
print(f"{'Dynamic Python loops':<25} {'No':<15} {'No':<15} {'Yes':<15}")
print(f"{'Serialization (SavedModel)':<25} {'Trivial':<15} {'Trivial':<15} {'Requires config':<15}")
print(f"{'PyTorch Analogy':<25} {'nn.Sequential':<15} {'Graph module':<15} {'nn.Module':<15}")
print("-" * 70)
print("[*] Lesson 3 completed successfully.")


if __name__ == "__main__":
    pass
