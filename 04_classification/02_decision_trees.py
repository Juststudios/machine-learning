"""
Lesson: Decision Trees for Classification
==========================================
WHY THIS LESSON EXISTS:
    Decision trees are one of the most interpretable machine learning models.
    Unlike logistic regression (which gives you a formula), a decision tree
    gives you a set of IF-THEN rules that a human engineer can read, verify,
    and explain to a non-technical manager.

    "IF temperature > 82°C AND vibration > 4.5 mm/s THEN → FAULT"

    This interpretability is critical in regulated industries: aerospace,
    medical devices, financial services, safety-critical systems.

WHAT YOU WILL LEARN:
    1. How a tree splits data using Gini impurity
    2. Training a DecisionTreeClassifier in sklearn
    3. Visualizing the tree structure (save to PNG)
    4. max_depth: the key hyperparameter controlling overfitting
    5. Feature importance: which sensors matter most?
    6. Real example: 3-class sensor fault classification

REAL ENGINEERING APPLICATION:
    A maintenance engineer asks: "Can you explain WHY the model flagged
    this machine for maintenance?" With a decision tree, you can point to
    the exact rule. With a neural network, you cannot.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# PART 1: Intuition — How Does a Tree Split Data?
# =============================================================================
print("=" * 65)
print("PART 1: How Decision Trees Split Data")
print("=" * 65)
print("""
A Decision Tree answers a series of YES/NO questions:

    Is temperature > 82°C?
    ├── NO  → Is vibration > 3.5 mm/s?
    │         ├── NO  → Predict: OK
    │         └── YES → Predict: WARNING
    └── YES → Is vibration > 5.0 mm/s?
              ├── NO  → Predict: WARNING
              └── YES → Predict: FAULT

How does it choose WHERE to split?
    It measures IMPURITY — how mixed are the classes in each region?
    
GINI IMPURITY for a node with classes proportions p₁, p₂, ...:
    Gini = 1 - Σ(pᵢ²)
    
    Pure node (all same class):  Gini = 1 - 1² = 0   ← best
    50/50 split (2 classes):     Gini = 1 - (0.5² + 0.5²) = 0.5  ← worst

The tree picks the split (feature + threshold) that MOST REDUCES
average Gini impurity across the two child nodes.
""")

# Show Gini impurity visually
p = np.linspace(0.001, 0.999, 200)
gini = 1 - (p**2 + (1-p)**2)
entropy = -(p * np.log2(p + 1e-10) + (1-p) * np.log2(1-p + 1e-10))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(p, gini, 'b-', linewidth=2, label='Gini impurity')
ax.plot(p, entropy / 2, 'r--', linewidth=2, label='Entropy / 2 (scaled)')
ax.set_xlabel('Fraction of class 1 in node (p)', fontsize=11)
ax.set_ylabel('Impurity measure', fontsize=11)
ax.set_title('Gini Impurity — Measures How Mixed a Node Is', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axvline(0.5, color='gray', linestyle=':', alpha=0.7, label='Most impure (50/50)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_gini_impurity.png", dpi=120, bbox_inches='tight')
plt.close()
print("✓ Saved: output/02_gini_impurity.png")

# =============================================================================
# PART 2: Generate Sensor Data — 3 Classes
# =============================================================================
print("\n" + "=" * 65)
print("PART 2: 3-Class Sensor Classification — OK / WARNING / FAULT")
print("=" * 65)

np.random.seed(2024)

def make_sensor_data(n_ok=350, n_warn=250, n_fault=150):
    """Generate synthetic sensor data with 3 health states.
    
    Features:
        temperature: motor temperature in °C
        vibration:   vibration amplitude in mm/s
        pressure:    hydraulic pressure in bar
    """
    # OK: cool, quiet, normal pressure
    ok = np.column_stack([
        np.random.normal(65, 3, n_ok),   # temperature
        np.random.normal(1.8, 0.4, n_ok), # vibration
        np.random.normal(50, 3, n_ok),    # pressure
    ])
    # WARNING: moderately elevated readings
    warn = np.column_stack([
        np.random.normal(78, 4, n_warn),
        np.random.normal(4.0, 0.7, n_warn),
        np.random.normal(44, 4, n_warn),
    ])
    # FAULT: high temperature, strong vibration, low pressure
    fault = np.column_stack([
        np.random.normal(91, 5, n_fault),
        np.random.normal(7.5, 1.0, n_fault),
        np.random.normal(38, 4, n_fault),
    ])
    X = np.vstack([ok, warn, fault])
    y = np.array([0]*n_ok + [1]*n_warn + [2]*n_fault)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

X, y = make_sensor_data()
feature_names = ['Temperature (°C)', 'Vibration (mm/s)', 'Pressure (bar)']
class_names = ['OK', 'WARNING', 'FAULT']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Dataset: {len(X)} samples, {X.shape[1]} features, 3 classes")
print(f"Class distribution — OK: {(y==0).sum()}, WARNING: {(y==1).sum()}, FAULT: {(y==2).sum()}")

# =============================================================================
# PART 3: Train Decision Tree & Visualize
# =============================================================================
print("\n" + "=" * 65)
print("PART 3: Training and Visualizing the Decision Tree")
print("=" * 65)

# WHY max_depth=4: deeper trees memorize training data (overfit).
# A shallow tree (depth 3-5) is more interpretable and generalizes better.
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print(f"Decision Tree (max_depth=4) accuracy: {accuracy_score(y_test, y_pred):.3f}")
print()
print(classification_report(y_test, y_pred, target_names=class_names))

# Visualize the tree
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(
    tree,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,          # WHY filled=True: color nodes by majority class
    rounded=True,
    fontsize=8,
    ax=ax
)
ax.set_title('Decision Tree (max_depth=4) — Machine Health Classification\n'
             'Color = majority class | Darker = purer node', fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_decision_tree_visualization.png", dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved: output/02_decision_tree_visualization.png")

# =============================================================================
# PART 4: Overfitting — Effect of max_depth
# =============================================================================
print("\n" + "=" * 65)
print("PART 4: Overfitting — How max_depth Controls Model Complexity")
print("=" * 65)
print("""
max_depth = 1  → underfits (only 1 split, too simple)
max_depth = 3  → good balance
max_depth = 10 → overfits (memorizes training data)
max_depth = None → fully grown tree, guaranteed to overfit
""")

depths = [1, 2, 3, 4, 5, 6, 8, 10, 15, None]
train_accs = []
test_accs = []

for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=42)
    t.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, t.predict(X_train)))
    test_accs.append(accuracy_score(y_test, t.predict(X_test)))

depth_labels = [str(d) if d is not None else 'None\n(full)' for d in depths]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(len(depths)), train_accs, 'bo-', linewidth=2, label='Training accuracy')
ax.plot(range(len(depths)), test_accs, 'rs-', linewidth=2, label='Test accuracy')
ax.set_xticks(range(len(depths)))
ax.set_xticklabels(depth_labels)
ax.set_xlabel('max_depth', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Decision Tree: Overfitting with Increasing Depth', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 1.02)

# Mark sweet spot
best_test_idx = int(np.argmax(test_accs))
ax.axvline(best_test_idx, color='green', linestyle='--', alpha=0.7)
ax.text(best_test_idx + 0.1, 0.55, f'Best test\ndepth={depths[best_test_idx]}',
        color='green', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_depth_vs_accuracy.png", dpi=120, bbox_inches='tight')
plt.close()
print("✓ Saved: output/02_depth_vs_accuracy.png")

# Print table
print(f"\n{'max_depth':>12} {'Train Acc':>10} {'Test Acc':>10}")
print("-" * 35)
for d, tr, te in zip(depths, train_accs, test_accs):
    overfit_flag = " ← overfitting" if tr - te > 0.05 else ""
    dname = str(d) if d is not None else "None"
    print(f"{dname:>12} {tr:>10.3f} {te:>10.3f}{overfit_flag}")

# =============================================================================
# PART 5: Feature Importance
# =============================================================================
print("\n" + "=" * 65)
print("PART 5: Feature Importance — Which Sensors Matter Most?")
print("=" * 65)
print("""
Feature importance = how much each feature reduces impurity across all splits.
Normalized to sum to 1.0.
""")

# Use the best-depth tree
tree_best = DecisionTreeClassifier(max_depth=depths[best_test_idx], random_state=42)
tree_best.fit(X_train, y_train)
importances = tree_best.feature_importances_

sorted_idx = np.argsort(importances)[::-1]
for rank, i in enumerate(sorted_idx, 1):
    bar = "█" * int(importances[i] * 50)
    print(f"  {rank}. {feature_names[i]:22s}: {importances[i]:.3f}  {bar}")

fig, ax = plt.subplots(figsize=(7, 4))
colors = ['#1976D2', '#388E3C', '#F57C00']
bars = ax.bar(feature_names, importances[sorted_idx[::-1]],  # sorted low to high for horizontal
              color=['#1976D2', '#388E3C', '#F57C00'])

# Replot sorted descending
ax.cla()
y_pos = np.arange(len(feature_names))
sorted_names = [feature_names[i] for i in sorted_idx]
sorted_imps = importances[sorted_idx]
bars = ax.barh(y_pos, sorted_imps, color=['#F44336', '#FF9800', '#2196F3'])
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names, fontsize=11)
ax.set_xlabel('Feature Importance (Gini reduction)', fontsize=11)
ax.set_title('Which Sensor Matters Most for Fault Detection?', fontsize=12)
for bar, val in zip(bars, sorted_imps):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_feature_importance.png", dpi=120, bbox_inches='tight')
plt.close()
print("\n✓ Saved: output/02_feature_importance.png")

print("""
KEY TAKEAWAYS — Decision Trees:
  ✓ Highly interpretable: produces human-readable rules
  ✓ No feature scaling needed (splits are based on thresholds)
  ✓ Handles non-linear boundaries naturally
  ✓ Built-in feature importance scores
  ✗ Prone to overfitting (needs max_depth or min_samples tuning)
  ✗ High variance: small data changes → very different trees
  ✗ Axis-aligned boundaries only (less flexible than SVM or NN)
  
  → Solution to high variance: use a RANDOM FOREST (next lesson!)
""")
