"""
Lesson 3: Data and Features
============================
What makes good machine learning data?

In this lesson you will learn:
- What are features and targets (the vocabulary of ML)
- Why feature quality matters MORE than algorithm choice
- Feature types: numerical, categorical, ordinal
- Feature engineering: creating better inputs from existing data
- One-hot encoding: converting categories into numbers
- Feature scaling: why distances and gradients care about scale

The 80% rule: 80% of real ML work is data preparation.
The best algorithm on bad features loses to a simple model on good features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
)
from sklearn.compose import ColumnTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

np.random.seed(42)

# ===========================================================================
# PART 1: Features vs. Targets
# ===========================================================================
print("=" * 60)
print("PART 1: Features vs. Targets")
print("=" * 60)

# WHY: Every ML problem has two sides:
#   X = features (inputs, predictors, independent variables)
#   y = target   (output, label, dependent variable)

# Think of it like a function: y = f(X)
# ML's job: figure out what f() is from examples of (X, y) pairs

print("""
Machine Learning vocabulary:
  X = features (what we know about each sample)
  y = target   (what we want to predict)

Example - Predicting house prices:
  Features (X): square footage, bedrooms, neighborhood, year built
  Target   (y): price ($)

Example - Detecting machine faults:
  Features (X): temperature, vibration, RPM, current draw
  Target   (y): OK / WARNING / FAULT
""")

# Create a sample dataset
data = {
    'study_hours':    [2, 5, 1, 8, 3, 6, 4, 7, 1, 9],
    'attendance_pct': [70, 90, 60, 95, 75, 85, 80, 92, 55, 98],
    'prev_grade':     [65, 80, 55, 90, 70, 82, 75, 88, 50, 95],
    'major':          ['CS', 'EE', 'ME', 'CS', 'EE', 'ME', 'CS', 'EE', 'ME', 'CS'],
    'final_grade':    [68, 82, 58, 91, 72, 84, 77, 89, 53, 96],
}
df = pd.DataFrame(data)

print("Sample student dataset:")
print(df.to_string(index=False))

# Features vs target
X = df.drop(columns=['final_grade'])   # Everything except what we predict
y = df['final_grade']                  # What we want to predict

print(f"\nFeature matrix X shape: {X.shape} ({X.shape[0]} samples, {X.shape[1]} features)")
print(f"Target vector y shape:  {y.shape}")

# ===========================================================================
# PART 2: Feature Types
# ===========================================================================
print("\n" + "=" * 60)
print("PART 2: Feature Types")
print("=" * 60)

print("""
Feature types:
1. NUMERICAL (continuous): study_hours=5.5, temperature=72.3
   - Can be directly used by most algorithms
   - May need scaling (see Part 4)

2. CATEGORICAL (nominal): major='CS', color='red'
   - No inherent order: CS is not "greater than" EE
   - Must be encoded as numbers before feeding to most models
   - Use: One-Hot Encoding

3. ORDINAL (ordered categories): grade='A' > 'B' > 'C'
   - Has meaningful order, but gaps between levels may be unequal
   - Can use Label Encoding OR custom integer mapping

4. BINARY: pass=True/False, fault=0/1
   - Special case of categorical with only 2 values
""")

print("Feature types in our dataset:")
for col in df.columns:
    dtype = df[col].dtype
    unique_count = df[col].nunique()
    print(f"  {col:20s}: dtype={dtype}, unique values={unique_count}")

# ===========================================================================
# PART 3: Feature Engineering
# ===========================================================================
print("\n" + "=" * 60)
print("PART 3: Feature Engineering — Creating Better Features")
print("=" * 60)

# WHY: Raw features often don't capture the true signal.
# A feature engineer creates new features that are more informative.

print("""
WHY feature engineering:
  Raw data: study_hours=2, attendance=90
  
  But what matters might be the COMBINATION:
  - A student who studies 2h but attends 90% classes
    might outperform someone who studies 10h but only attends 50%
  - Create: study_efficiency = study_hours * attendance_pct / 100
""")

df_engineered = df.copy()

# Feature 1: Study efficiency (interaction feature)
df_engineered['study_efficiency'] = (
    df['study_hours'] * df['attendance_pct'] / 100
)

# Feature 2: Academic momentum (how much they improved from previous grade)
# WHY: improving students perform better than stagnating ones
df_engineered['grade_gap'] = df['final_grade'] - df['prev_grade']

# Feature 3: High performer flag
# WHY: Binary features can help tree-based models make sharp splits
df_engineered['high_attender'] = (df['attendance_pct'] >= 85).astype(int)

print("Original features + engineered features:")
print(df_engineered[['study_hours', 'attendance_pct', 
                       'study_efficiency', 'grade_gap', 'high_attender']].to_string(index=False))

# ===========================================================================
# PART 4: Encoding Categorical Features
# ===========================================================================
print("\n" + "=" * 60)
print("PART 4: Encoding Categorical Features")
print("=" * 60)

print("""
Problem: ML models need numbers. 'CS', 'EE', 'ME' are strings.

BAD approach (Label Encoding for nominal data):
  CS → 0, EE → 1, ME → 2
  Problem: model thinks ME (2) is "twice as big" as CS (0)!
  This introduces false ordering into the data.

GOOD approach (One-Hot Encoding):
  CS → [1, 0, 0]
  EE → [0, 1, 0]  
  ME → [0, 0, 1]
  Each category gets its own binary column. No false ordering.
""")

# Label Encoding — ONLY appropriate for ordinal data
le = LabelEncoder()
df['major_label'] = le.fit_transform(df['major'])
print("Label Encoded majors (BAD for nominal data):")
print(df[['major', 'major_label']].drop_duplicates().sort_values('major').to_string(index=False))

# One-Hot Encoding — correct for nominal categories
ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids dummy variable trap
major_encoded = ohe.fit_transform(df[['major']])
major_df = pd.DataFrame(major_encoded, columns=ohe.get_feature_names_out(['major']))

print("\nOne-Hot Encoded majors (GOOD for nominal data):")
result = pd.concat([df['major'], major_df], axis=1).drop_duplicates().sort_values('major')
print(result.to_string(index=False))

# Pandas get_dummies: convenient one-liner
print("\nUsing pd.get_dummies (convenient shortcut):")
df_dummies = pd.get_dummies(df[['major']], drop_first=True)
print(df_dummies.head(5).to_string())

# ===========================================================================
# PART 5: Feature Scaling
# ===========================================================================
print("\n" + "=" * 60)
print("PART 5: Feature Scaling — Why It Matters")
print("=" * 60)

print("""
WHY scaling matters:

Example without scaling:
  feature_1: study_hours    → range [1, 9]
  feature_2: attendance_pct → range [55, 98]

For algorithms that use DISTANCES (KNN, SVM) or GRADIENTS (Linear Regression, 
Neural Networks), a feature with larger numbers dominates.

Gradient descent will spend most of its time navigating steep hills 
in the attendance dimension and tiny slopes in study_hours.

SOLUTION: Scale all features to the same range.
""")

# Generate data with very different scales
np.random.seed(42)
n = 100
sensor_data = pd.DataFrame({
    'temperature_C': np.random.normal(75, 10, n),     # range ~45-105
    'rpm':           np.random.normal(3000, 500, n),  # range ~1500-4500
    'current_A':     np.random.normal(15, 3, n),      # range ~6-24
    'voltage_V':     np.random.normal(220, 5, n),     # range ~200-240
})

print("Raw sensor data statistics:")
print(sensor_data.describe().round(1).to_string())

# StandardScaler: (x - mean) / std → mean=0, std=1
# WHY: Works well for normally distributed data, preserves outlier info
std_scaler = StandardScaler()
data_standard = std_scaler.fit_transform(sensor_data)
df_standard = pd.DataFrame(data_standard, columns=sensor_data.columns)

print("\nAfter StandardScaler (mean=0, std=1):")
print(df_standard.describe().round(3).to_string())

# MinMaxScaler: (x - min) / (max - min) → range [0, 1]
# WHY: Good when algorithm expects values in [0,1] (e.g., neural networks)
#      BUT: very sensitive to outliers (one outlier squishes everything)
mm_scaler = MinMaxScaler()
data_minmax = mm_scaler.fit_transform(sensor_data)
df_minmax = pd.DataFrame(data_minmax, columns=sensor_data.columns)

print("\nAfter MinMaxScaler (range [0, 1]):")
print(df_minmax.describe().round(3).to_string())

print("""
When to use which scaler:
  StandardScaler:  Default choice. Good for most algorithms.
                   Keeps outliers (they become large values but stay).
  MinMaxScaler:    When you need values in [0, 1].
                   Use for neural network inputs, image pixels.
  RobustScaler:    When data has many outliers.
                   Uses median and IQR instead of mean and std.
""")

# ===========================================================================
# PART 6: Putting It All Together — Preprocessing Pipeline Concept
# ===========================================================================
print("\n" + "=" * 60)
print("PART 6: Mixed Data — Numerical + Categorical Together")
print("=" * 60)

# Realistic pipeline: scale numbers, encode categories
numerical_cols = ['study_hours', 'attendance_pct', 'prev_grade']
categorical_cols = ['major']

# ColumnTransformer applies different transforms to different columns
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
])

X_raw = df[numerical_cols + categorical_cols]
X_processed = preprocessor.fit_transform(X_raw)

print(f"Raw X shape: {X_raw.shape}")
print(f"Processed X shape: {X_processed.shape}  (3 numerical + 2 one-hot = 5 columns)")
print("\nFirst 5 rows of processed features:")
print(X_processed[:5].round(3))

print("""
Note: The preprocessor is fitted ONLY on training data.
      Then applied to test data using the same parameters.
      (This prevents data leakage — covered in Module 6)
""")

# ===========================================================================
# VISUALIZATION: Feature distributions before and after scaling
# ===========================================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Feature Distributions: Raw vs. Scaled", fontsize=14, fontweight='bold')

for i, col in enumerate(sensor_data.columns):
    # Raw
    axes[0, i].hist(sensor_data[col], bins=20, color='steelblue', alpha=0.7)
    axes[0, i].set_title(f"Raw: {col}")
    axes[0, i].set_ylabel("Count" if i == 0 else "")
    
    # Scaled
    axes[1, i].hist(df_standard[col], bins=20, color='coral', alpha=0.7)
    axes[1, i].set_title(f"Scaled: {col}")
    axes[1, i].set_ylabel("Count" if i == 0 else "")

plt.tight_layout()
plot_path = OUTPUT_DIR / "feature_scaling_comparison.png"
plt.savefig(plot_path, dpi=100, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: {plot_path}")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. Features (X) are inputs; target (y) is what you predict
2. Feature quality > algorithm choice (garbage in = garbage out)
3. Numerical features: scale them (StandardScaler is the default choice)
4. Categorical features: one-hot encode them (not label encode)
5. Feature engineering (creating new features) often gives the biggest gains
6. Always fit scalers/encoders on TRAINING data only, then transform test data

Next: 04_train_test_split.py — How to honestly evaluate your model
""")
