"""
02_gradient_tape.py
===================
Lesson 2: Automatic Differentiation with tf.GradientTape.

Demonstrates:
- Context-manager automatic differentiation with tf.GradientTape
- Evaluating scalar and multi-variable gradients
- Watching non-Variable constants using tape.watch()
- Persistent tapes (tf.GradientTape(persistent=True))
- Higher-order derivatives via nested gradient tapes
- Implementing a complete Gradient Descent optimization loop from scratch
- Fitting a Linear Regression model using manual tape gradients and assign_sub
"""

import sys
from pathlib import Path

# Add current directory to path to load Python 3.14 compatibility engine
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tf_compat  # noqa: F401
import tensorflow as tf
import numpy as np

print("=" * 70)
print("TENSORFLOW FUNDAMENTALS — LESSON 2: tf.GradientTape & AUTODIFF")
print("=" * 70)

# ============================================================================
# 1. BASIC GRADIENT COMPUTATION (SCALAR FUNCTION)
# ============================================================================
print("\n--- 1. Basic Gradient Computation ---")
# Compute dy/dx for y = x^2 at x = 3.0. Analytical answer: dy/dx = 2x = 6.0
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x * x

grad_y = tape.gradient(y, x)
print(f"Function: y = x^2 at x = {x.numpy()}")
print(f"Computed dy/dx: {grad_y.numpy()} (Analytical expected: 6.0)")
assert np.isclose(grad_y.numpy(), 6.0), "Gradient mismatch!"


# ============================================================================
# 2. MULTI-VARIABLE GRADIENTS
# ============================================================================
print("\n--- 2. Multi-Variable Gradients ---")
# Compute gradients for z = 2*u^3 + v^2 with respect to [u, v]
# dz/du = 6*u^2, dz/dv = 2*v
u = tf.Variable(2.0, name="u")
v = tf.Variable(5.0, name="v")

with tf.GradientTape() as tape:
    z = 2.0 * (u * u * u) + (v * v)

dz_du, dz_dv = tape.gradient(z, [u, v])
print(f"Function: z = 2*u^3 + v^2 at u={u.numpy()}, v={v.numpy()}")
print(f"Computed dz/du: {dz_du.numpy()} (Analytical: 6*u^2 = 24.0)")
print(f"Computed dz/dv: {dz_dv.numpy()} (Analytical: 2*v = 10.0)")
assert np.isclose(dz_du.numpy(), 24.0) and np.isclose(dz_dv.numpy(), 10.0)


# ============================================================================
# 3. WATCHING CONSTANTS WITH tape.watch()
# ============================================================================
print("\n--- 3. Watching Constants with tape.watch() ---")
# By default, tf.GradientTape only records operations on tf.Variable instances.
# To compute gradients with respect to a tf.constant, call tape.watch(c).
c = tf.constant(4.0)

with tf.GradientTape() as tape:
    tape.watch(c)  # Explicitly tell tape to track this constant
    res = c * c * c  # res = c^3, d(res)/dc = 3*c^2 = 3 * 16 = 48.0

d_res = tape.gradient(res, c)
print(f"Constant c = {c.numpy()} watched explicitly")
print(f"Computed d(c^3)/dc: {d_res.numpy()} (Analytical: 48.0)")
assert np.isclose(d_res.numpy(), 48.0)


# ============================================================================
# 4. PERSISTENT TAPES
# ============================================================================
print("\n--- 4. Persistent Gradient Tapes ---")
# By default, tape.gradient() releases tracked resources.
# Setting persistent=True allows multiple gradient calls from one tape.
w = tf.Variable(3.0)

with tf.GradientTape(persistent=True) as tape:
    y1 = w * w
    y2 = w * w * w

dy1_dw = tape.gradient(y1, w)  # 2*w = 6.0
dy2_dw = tape.gradient(y2, w)  # 3*w^2 = 27.0
del tape  # Explicitly release resources of persistent tape

print(f"dy1/dw (2*w): {dy1_dw.numpy()} | dy2/dw (3*w^2): {dy2_dw.numpy()}")
assert np.isclose(dy1_dw.numpy(), 6.0) and np.isclose(dy2_dw.numpy(), 27.0)


# ============================================================================
# 5. HIGHER-ORDER DERIVATIVES (NESTED TAPES)
# ============================================================================
print("\n--- 5. Second-Order Derivatives (Nested Tapes) ---")
# Compute d^2y/dx^2 for y = x^3 at x = 2.0
# dy/dx = 3*x^2 (12.0)
# d^2y/dx^2 = 6*x (12.0)
val = tf.Variable(2.0)

with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        output = val * val * val
    first_derivative = inner_tape.gradient(output, val)
second_derivative = outer_tape.gradient(first_derivative, val)

print(f"At x = {val.numpy()}:")
print(f"  First derivative dy/dx: {first_derivative.numpy()}")
print(f"  Second derivative d^2y/dx^2: {second_derivative.numpy()} (Analytical: 12.0)")
assert np.isclose(second_derivative.numpy(), 12.0)


# ============================================================================
# 6. FROM-SCRATCH GRADIENT DESCENT: LINEAR REGRESSION
# ============================================================================
print("\n--- 6. Custom Gradient Descent Loop: Linear Regression ---")
# Synthetic Ground Truth: y = 3.5 * X + 1.2 + noise
np.random.seed(42)
X_np = np.linspace(-2.0, 2.0, 100, dtype=np.float32).reshape(-1, 1)
noise = np.random.normal(0.0, 0.2, size=X_np.shape).astype(np.float32)
y_np = 3.5 * X_np + 1.2 + noise

X_train = tf.constant(X_np)
y_train = tf.constant(y_np)

# Trainable parameters initialized randomly
w_learned = tf.Variable(0.0, name="slope")
b_learned = tf.Variable(0.0, name="intercept")

learning_rate = 0.1
epochs = 50

print(f"Initial parameters: w = {w_learned.numpy():.2f}, b = {b_learned.numpy():.2f}")
print("Training Linear Regression using manual GradientTape updates...")

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # Forward pass: y_hat = X * w + b
        y_pred = X_train * w_learned + b_learned
        # Loss: Mean Squared Error
        error = y_pred - y_train
        loss = tf.reduce_mean(error * error)

    # Compute gradients of loss with respect to parameters
    grad_w, grad_b = tape.gradient(loss, [w_learned, b_learned])

    # Gradient descent update: w = w - lr * grad_w
    w_learned.assign_sub(learning_rate * grad_w)
    b_learned.assign_sub(learning_rate * grad_b)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch + 1:02d}/{epochs:02d} | Loss: {loss.numpy():.4f} | w: {w_learned.numpy():.3f} | b: {b_learned.numpy():.3f}")

print(f"\nFinal learned parameters: w = {w_learned.numpy():.3f} (True: 3.500), b = {b_learned.numpy():.3f} (True: 1.200)")
assert np.isclose(w_learned.numpy(), 3.5, atol=0.15), "Slope failed to converge"
assert np.isclose(b_learned.numpy(), 1.2, atol=0.15), "Intercept failed to converge"
print("[*] Custom Gradient Descent verified successfully.")


if __name__ == "__main__":
    pass
