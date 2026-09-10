"""
Lesson: Logistic Regression for Classification
===============================================
WHY THIS LESSON EXISTS:
    Linear regression predicts a number. But sometimes you need to predict
    a *category*: OK or FAULT, spam or not spam, disease or healthy.
    Logistic Regression is the simplest and most interpretable classifier —
    it's the foundation of neural networks and the go-to for binary decisions.

WHAT YOU WILL LEARN:
    1. Why we can't use linear regression for classification
    2. The sigmoid function: squashing any number into [0, 1]
    3. Interpreting coefficients as log-odds
    4. Binary classification: machine fault detection
    5. Multi-class classification using one-vs-rest
    6. Reading probability outputs with predict_proba()

REAL ENGINEERING APPLICATION:
    Predictive maintenance: a vibration sensor + temperature sensor on a
    motor. Predict whether the motor is OK or about to FAULT so maintenance
    can be scheduled *before* the machine breaks.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # WHY: non-interactive backend — saves PNG without display
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# WHY: Use __file__ so paths work from any working directory
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# PART 1: Why Not Linear Regression for Classification?
# =============================================================================
print("=" * 65)
print("PART 1: The Problem With Linear Regression for Classification")
print("=" * 65)

# WHY: If we use linear regression to predict 0 (OK) or 1 (FAULT),
# the model can output values like -0.3 or 1.8, which have no meaning
# as probabilities. We need outputs in [0, 1].
print("""
Problem: You have sensor data. You want to predict:
    0 = machine is OK
    1 = machine will FAULT in next hour

If you use LinearRegression:
    - It can predict 1.8 or -0.3  ← what does that mean?
    - It assumes errors are Gaussian — wrong for binary outcomes
    - Decision boundary is unbounded

Solution: Logistic Regression uses the SIGMOID function to squash
    any real number into a valid probability between 0 and 1.
""")

# =============================================================================
# PART 2: The Sigmoid Function
# =============================================================================
print("=" * 65)
print("PART 2: The Sigmoid Function  σ(z) = 1 / (1 + e^(-z))")
print("=" * 65)

def sigmoid(z):
    """The sigmoid (logistic) function — squashes any number to [0, 1].
    
    WHY: When z is very negative → output ≈ 0 (confidently class 0)
         When z = 0             → output = 0.5 (uncertain)
         When z is very positive → output ≈ 1 (confidently class 1)
    """
    return 1.0 / (1.0 + np.exp(-z))

z_values = np.linspace(-8, 8, 300)
sigmoid_values = sigmoid(z_values)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: sigmoid curve
ax = axes[0]
ax.plot(z_values, sigmoid_values, 'b-', linewidth=2.5, label='σ(z)')
ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='threshold = 0.5')
ax.axvline(0, color='gray', linestyle=':', alpha=0.6)
ax.fill_between(z_values, 0.5, sigmoid_values,
                where=(sigmoid_values > 0.5), alpha=0.15, color='green',
                label='Predict class 1')
ax.fill_between(z_values, sigmoid_values, 0.5,
                where=(sigmoid_values < 0.5), alpha=0.15, color='red',
                label='Predict class 0')
ax.set_xlabel('z  (linear combination of features)')
ax.set_ylabel('σ(z)  =  P(class = 1)')
ax.set_title('The Sigmoid Function')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# Right: what logistic regression computes
ax2 = axes[1]
ax2.text(0.05, 0.85, "Logistic Regression Model:", fontsize=11,
         fontweight='bold', transform=ax2.transAxes)
ax2.text(0.05, 0.70, "z  =  w₀ + w₁·x₁ + w₂·x₂ + ...", fontsize=10,
         transform=ax2.transAxes, family='monospace')
ax2.text(0.05, 0.55, "P(y=1|x) = σ(z) = 1/(1 + e^{-z})", fontsize=10,
         transform=ax2.transAxes, family='monospace')
ax2.text(0.05, 0.40, "Predict class 1 if P(y=1|x) > 0.5", fontsize=10,
         transform=ax2.transAxes)
ax2.text(0.05, 0.25, "The model LEARNS w₀, w₁, w₂, ...", fontsize=10,
         transform=ax2.transAxes)
ax2.text(0.05, 0.12, "by maximizing log-likelihood", fontsize=10,
         transform=ax2.transAxes, color='blue')
ax2.axis('off')
ax2.set_title('How Logistic Regression Works')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_sigmoid_function.png", dpi=120, bbox_inches='tight')
plt.close()
print("✓ Saved: output/01_sigmoid_function.png")

print("""
Key sigmoid properties:
  σ(0)   = 0.500  (uncertain — on the boundary)
  σ(2)   = {:.3f}  (lean toward class 1)
  σ(4)   = {:.3f}  (strongly class 1)
  σ(-4)  = {:.3f}  (strongly class 0)
""".format(sigmoid(2), sigmoid(4), sigmoid(-4)))

# =============================================================================
# PART 3: Binary Classification — Machine Fault Detection
# =============================================================================
print("=" * 65)
print("PART 3: Binary Classification — Machine Fault Detection")
print("=" * 65)
print("""
Scenario: Monitors measure temperature (°C) and vibration (mm/s)
on industrial motors every minute.
Label:  0 = OK (running normally)
        1 = FAULT (will fail within the hour)
Goal: Predict FAULT before it happens → schedule maintenance!
""")

# WHY: Synthetic data with known properties lets us verify the model
# understands the right relationships
np.random.seed(42)
n_samples = 600

# OK class: normal temperature (60-75°C), low vibration (1-3 mm/s)
n_ok = 400
temp_ok = np.random.normal(67, 4, n_ok)
vib_ok = np.random.normal(2.0, 0.5, n_ok)

# FAULT class: higher temperature (78-95°C), higher vibration (4-8 mm/s)
n_fault = 200
temp_fault = np.random.normal(86, 5, n_fault)
vib_fault = np.random.normal(6.0, 1.0, n_fault)

X = np.column_stack([
    np.concatenate([temp_ok, temp_fault]),
    np.concatenate([vib_ok, vib_fault])
])
y = np.array([0] * n_ok + [1] * n_fault)

# WHY: Shuffle so train/test split doesn't have all one class at the end
shuffle_idx = np.random.permutation(len(y))
X, y = X[shuffle_idx], y[shuffle_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# WHY: StandardScaler is important for logistic regression. The gradient
# descent optimizer converges much faster when features have similar scale.
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)  # fit on train ONLY
X_test_sc = scaler.transform(X_test)         # apply same transform to test

# Train logistic regression
# WHY max_iter=1000: gradient descent needs enough steps to converge
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_sc, y_train)

y_pred = clf.predict(X_test_sc)
y_prob = clf.predict_proba(X_test_sc)  # probability for each class

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['OK', 'FAULT']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"              Predicted OK   Predicted FAULT")
print(f"Actual OK:       {cm[0,0]:4d}           {cm[0,1]:4d}")
print(f"Actual FAULT:    {cm[1,0]:4d}           {cm[1,1]:4d}")

# WHY: Coefficients tell us which features matter and in which direction
print("\nModel Coefficients (on scaled features):")
feature_names = ['Temperature', 'Vibration']
for fname, coef in zip(feature_names, clf.coef_[0]):
    direction = "↑ increases FAULT probability" if coef > 0 else "↓ decreases FAULT probability"
    print(f"  {fname}: {coef:.3f}  {direction}")
print(f"  Intercept: {clf.intercept_[0]:.3f}")

# Probability outputs — engineering value
print("\nPrediction samples (first 5 test points):")
print(f"{'Temp':>8} {'Vibr':>8} {'True':>8} {'P(OK)':>8} {'P(FAULT)':>10} {'Pred':>8}")
for i in range(5):
    t, v = X_test[i]
    print(f"{t:8.1f} {v:8.2f} {['OK','FAULT'][y_test[i]]:>8} "
          f"{y_prob[i,0]:8.3f} {y_prob[i,1]:10.3f} "
          f"{['OK','FAULT'][y_pred[i]]:>8}")

# Decision boundary plot
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#2196F3', '#F44336']
labels = ['OK', 'FAULT']
markers = ['o', 's']

for cls in [0, 1]:
    mask = y_train == cls
    ax.scatter(X_train[mask, 0], X_train[mask, 1],
               c=colors[cls], label=f'{labels[cls]} (train)',
               marker=markers[cls], alpha=0.6, s=30)

# WHY: We compute the decision boundary in original (unscaled) space
# by creating a meshgrid and inverse-transforming before plotting
t_range = np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 200)
v_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
T, V = np.meshgrid(t_range, v_range)
grid = np.column_stack([T.ravel(), V.ravel()])
grid_sc = scaler.transform(grid)
Z = clf.predict_proba(grid_sc)[:, 1].reshape(T.shape)

contour = ax.contourf(T, V, Z, levels=20, cmap='RdBu_r', alpha=0.3)
ax.contour(T, V, Z, levels=[0.5], colors='black', linewidths=2,
           linestyles='--')
plt.colorbar(contour, ax=ax, label='P(FAULT)')

ax.set_xlabel('Temperature (°C)', fontsize=12)
ax.set_ylabel('Vibration (mm/s)', fontsize=12)
ax.set_title('Logistic Regression Decision Boundary\nMachine Fault Detection', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_logistic_decision_boundary.png", dpi=120, bbox_inches='tight')
plt.close()
print("\n✓ Saved: output/01_logistic_decision_boundary.png")

# =============================================================================
# PART 4: Multi-class Logistic Regression (One-vs-Rest)
# =============================================================================
print("\n" + "=" * 65)
print("PART 4: Multi-class Classification — 3 Machine States")
print("=" * 65)
print("""
Same sensor data, but now THREE classes:
  0 = OK       (normal operation)
  1 = WARNING  (performance degrading, schedule maintenance soon)
  2 = FAULT    (imminent failure — stop immediately!)

Logistic Regression handles this with One-vs-Rest (OVR):
  Train 3 binary classifiers:
    Classifier A: Is it OK? (vs all others)
    Classifier B: Is it WARNING? (vs all others)
    Classifier C: Is it FAULT? (vs all others)
  Predict the class with the highest probability.
""")

np.random.seed(7)
n_ok2, n_warn, n_fault2 = 300, 200, 150

X_mc = np.row_stack([
    np.column_stack([np.random.normal(66, 3, n_ok2),
                     np.random.normal(1.8, 0.4, n_ok2)]),
    np.column_stack([np.random.normal(78, 3, n_warn),
                     np.random.normal(4.0, 0.6, n_warn)]),
    np.column_stack([np.random.normal(90, 4, n_fault2),
                     np.random.normal(7.0, 0.8, n_fault2)]),
])
y_mc = np.array([0]*n_ok2 + [1]*n_warn + [2]*n_fault2)

idx = np.random.permutation(len(y_mc))
X_mc, y_mc = X_mc[idx], y_mc[idx]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_mc, y_mc, test_size=0.25, random_state=42, stratify=y_mc
)
sc2 = StandardScaler()
X_tr_sc = sc2.fit_transform(X_tr)
X_te_sc = sc2.transform(X_te)

# WHY multi_class='ovr': explicitly one-vs-rest for interpretability
clf_mc = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42)
clf_mc.fit(X_tr_sc, y_tr)

y_te_pred = clf_mc.predict(X_te_sc)
print(f"Multi-class accuracy: {accuracy_score(y_te, y_te_pred):.3f}")
print()
print("Classification Report (3-class):")
print(classification_report(y_te, y_te_pred, target_names=['OK', 'WARNING', 'FAULT']))

print("""
KEY TAKEAWAYS — Logistic Regression:
  ✓ Outputs probabilities, not just labels → calibrated uncertainty
  ✓ Coefficients are interpretable (larger coef = stronger effect)
  ✓ Fast to train, works well on linearly separable data
  ✓ Scales to large datasets with stochastic gradient descent
  ✗ Assumes linear decision boundary — may underfit complex patterns
  ✗ Needs feature scaling for gradient descent to converge well
""")
