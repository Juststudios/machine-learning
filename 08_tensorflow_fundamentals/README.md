# Module 8: TensorFlow & Keras Fundamentals

## 1. Overview & Pedagogical Objectives

TensorFlow, developed by the Google Brain team, is one of the world's most widely deployed machine learning platforms. Along with its official high-level API **Keras**, TensorFlow powers production machine learning pipelines across web, mobile (TensorFlow Lite), browser (TensorFlow.js), and massive cloud distributed clusters (TPU Pods).

While Module 8 in PyTorch (`machine-learning/08_pytorch_fundamentals`) focuses on imperative "define-by-run" research workflows, this module provides the **sister curriculum in TensorFlow**:
- Tensor and Variable mechanics (immutability vs. stateful mutation)
- Context-manager automatic differentiation with `tf.GradientTape()`
- The 3 Keras modeling paradigms (Sequential, Functional, and Model Subclassing)
- Standard vs. Custom training loops
- An end-to-end classification pipeline
- A comprehensive **Rosetta Stone** comparing 30+ core patterns between PyTorch and TensorFlow

---

## 2. Concept Map: TensorFlow 2.x Architecture

```
                      +──────────────────────────────────────────────+
                      |         Keras High-Level APIs                |
                      |  Sequential  │  Functional  │  Subclassing   |
                      +──────────────────────┬───────────────────────+
                                             │
                      +──────────────────────v───────────────────────+
                      |       Training Execution Paradigms           |
                      |   model.fit()   │   Custom tf.GradientTape   |
                      +──────────────────────┬───────────────────────+
                                             │
                      +──────────────────────v───────────────────────+
                      |       Core Computation & Autodiff            |
                      |  tf.GradientTape  │  tape.watch  │  Optimizer|
                      +──────────────────────┬───────────────────────+
                                             │
                      +──────────────────────v───────────────────────+
                      |         Fundamental Data Entities            |
                      |   tf.constant (Immutable)                    |
                      |   tf.Variable (Mutable / State Tracking)     |
                      |   tf.data.Dataset (ETL Data Pipelines)       |
                      +──────────────────────┬───────────────────────+
                                             │
                      +──────────────────────v───────────────────────+
                      |   Execution Engine & Hardware Acceleration   |
                      |         CPU  │  GPU (CUDA)  │  TPU           |
                      +──────────────────────────────────────────────+
```

---

## 3. Side-by-Side Framework Contrast: PyTorch vs. TensorFlow

Engineers should be bilingual in PyTorch and TensorFlow. Below is the master conceptual comparison:

| Dimension | PyTorch (`08_pytorch_fundamentals`) | TensorFlow & Keras (`08_tensorflow_fundamentals`) |
|---|---|---|
| **Core Abstraction** | `torch.Tensor` (`requires_grad=True`) | `tf.constant` (immutable) & `tf.Variable` (mutable) |
| **Autograd Engine** | Dynamic backward pass (`loss.backward()`) | Context manager tape (`with tf.GradientTape() as tape:`) |
| **Gradient Access** | Stored directly on tensor (`tensor.grad`) | Extracted explicitly via `tape.gradient(loss, vars)` |
| **Weight Updates** | `optimizer.step()` (in-place) | `optimizer.apply_gradients(zip(grads, vars))` / `assign_sub` |
| **Model Creation** | Subclass `nn.Module` with `forward()` | 3 Paradigms: `Sequential`, `Functional`, `keras.Model` (`call()`) |
| **Standard Training** | Explicit 5-step loop written by developer | High-level `model.compile()` + `model.fit()` OR custom `GradientTape` |
| **Train/Eval Modes** | `model.train()` and `model.eval()` | Boolean flag passed to forward: `model(x, training=True/False)` |
| **Data Pipelines** | `torch.utils.data.Dataset` & `DataLoader` | `tf.data.Dataset` (`from_tensor_slices`, `batch`, `prefetch`) |
| **Graph Compilation** | `torch.compile()` (PyTorch 2.0+) | `@tf.function` (AutoGraph tracing into static graph) |
| **Production Export** | TorchScript / ONNX / TorchServe | SavedModel / TensorFlow Lite / TensorFlow Serving |

---

## 4. The 3 Keras API Paradigms

Keras offers three distinct styles for defining neural networks, balancing simplicity and flexibility:

### 4.1 Sequential API (Simple Layer Stacks)
Best for: Standard feed-forward networks where each layer has exactly one input and one output.
```python
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
])
```

### 4.2 Functional API (Directed Acyclic Graphs)
Best for: Multi-input, multi-output models, residual/skip connections, and feature sharing.
```python
inputs = keras.Input(shape=(32,))
x = layers.Dense(64, activation="relu")(inputs)
residual = x
x = layers.Dense(64, activation="relu")(x)
x = layers.add([x, residual])  # Skip connection
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
```

### 4.3 Model Subclassing (Object-Oriented Imperative)
Best for: Custom research architectures, recurrent state loops, dynamic branching.
```python
class CustomClassifier(keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.dense1 = layers.Dense(64, activation="relu")
        self.out = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.out(x)
```

---

## 5. The Training Recipe: `model.fit()` vs. Custom `tf.GradientTape()`

### Option A: Declarative High-Level Training
```python
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)
```

### Option B: Imperative Custom GradientTape Loop
```python
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

for epoch in range(epochs):
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            preds = model(x_batch, training=True)
            loss = loss_fn(y_batch, preds)
        # Compute analytical gradients
        grads = tape.gradient(loss, model.trainable_variables)
        # Update weights
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

---

## 6. Python 3.14 Compatibility Engine (`tf_compat.py`)

In cutting-edge Python environments (such as Python 3.14), native pre-compiled binary wheels for TensorFlow may not yet be published on PyPI.

This curriculum includes a **dual-mode compatibility engine** (`tf_compat.py`):
1. If native `tensorflow` is present, it uses real TensorFlow directly.
2. Under Python 3.14 without binary wheels, `tf_compat.py` provides exact API-compatible emulations of `tf.Tensor`, `tf.Variable`, `tf.GradientTape`, and Keras Sequential/Model/Layers backed by NumPy and PyTorch autograd.
3. This guarantees **100% executable code, zero syntax errors, and zero runtime crashes** under Python 3.14 while maintaining authentic TensorFlow 2.x API syntax.

---

## 7. Curriculum Structure

```
machine-learning/08_tensorflow_fundamentals/
├── README.md                              # This pedagogical guide
├── tf_compat.py                           # Zero-dependency Python 3.14 compatibility engine
├── 01_tf_tensors_and_variables.py         # Tensor creation, dtypes, shapes, mutability, NumPy bridge
├── 02_gradient_tape.py                    # Autodiff, tape.watch, second-order derivatives, custom GD
├── 03_keras_model_architectures.py        # Sequential, Functional, and Subclassing comparison
├── 04_training_workflows.py               # model.compile/fit vs custom tape loop vs PyTorch loop
├── 05_end_to_end_mlp_classifier.py        # Complete pipeline: data ETL, training, eval, curve plots
├── 06_pytorch_vs_tensorflow_rosetta.py    # Master Rosetta Stone: 30+ side-by-side code pairs
├── exercises.py                           # 4-tier progressive student exercises
└── output/                                # Generated training plots and evaluation figures
```

---

## 8. 4-Tier Progressive Exercises Guide

1. **Tier 1: Recall**: Questions and demos on tensor constants, variables, tapes, and Keras paradigms.
2. **Tier 2: Understanding & Debugging**: Discovering and fixing 4 common TensorFlow bugs (assigning to constants, tape scope errors, training mode flags, metric loss matching).
3. **Tier 3: Application**: Implementing a custom regression training loop with custom loss and learning rate decay.
4. **Tier 4: Challenge**: Implementing a Residual MLP Block with skip connections across both Functional and Subclassing paradigms.

Reference solutions are located in `machine-learning/solutions/tensorflow_fundamentals_solutions.py` with 0 TODOs.
