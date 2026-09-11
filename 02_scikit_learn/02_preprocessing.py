"""
Preprocessing: Preparing Data for Machine Learning
====================================================
"Garbage in, garbage out."

Preprocessing is one of the most important — and most underestimated —
steps in any ML pipeline. No algorithm can compensate for bad data.

This lesson covers the complete sklearn preprocessing toolkit:
  - Scalers: StandardScaler, MinMaxScaler, RobustScaler
  - Encoders: LabelEncoder, OneHotEncoder, OrdinalEncoder
  - Imputers: SimpleImputer for missing values
  - ColumnTransformer: different preprocessing for different columns
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PowerTransformer, Binarizer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

np.random.seed(42)

# ===========================================================================
# PART 1: Scalers — When Numbers Have Different Scales
# ===========================================================================
print("=" * 60)
print("PART 1: Feature Scalers")
print("=" * 60)

print("""
WHY scaling matters:
  Building sensor data: 
    - outdoor_temp:  -10 to 40 °C
    - floor_area:    50 to 5000 m²
    - num_occupants: 1 to 500
    
  KNN, SVM, Neural Networks use DISTANCES or GRADIENTS.
  A 1-unit change in floor_area (50→51 m²) has 100x more
  'distance impact' than a 1°C change in temperature.
  The model incorrectly treats floor_area as 100x more important!
""")

# Create a realistic mixed-scale dataset
n = 200
data = pd.DataFrame({
    'outdoor_temp_C':  np.random.uniform(-10, 40, n),   # small range
    'floor_area_m2':   np.random.uniform(50, 5000, n),  # large range
    'num_occupants':   np.random.randint(1, 500, n).astype(float),
    'energy_kWh':      np.random.uniform(100, 10000, n),
})
X = data.values

print("Raw data statistics:")
print(data.describe().round(1).to_string())

# --- StandardScaler ---
print("\n--- StandardScaler: (x - mean) / std ---")
print("""
Result: mean=0, std=1 for each feature
Use when: data is approximately normal, algorithm assumes zero-mean
Best for: linear models, logistic regression, neural networks, PCA
""")
std_scaler = StandardScaler()
X_standard = std_scaler.fit_transform(X)
print(f"After StandardScaler — mean: {X_standard.mean(axis=0).round(4)}")
print(f"After StandardScaler — std:  {X_standard.std(axis=0).round(4)}")

# --- MinMaxScaler ---
print("\n--- MinMaxScaler: (x - min) / (max - min) ---")
print("""
Result: all values in [0, 1]
Use when: algorithm expects bounded inputs (image pixels, embeddings)
Weakness: ONE outlier can compress all other values to a tiny range!
""")
mm_scaler = MinMaxScaler()
X_minmax = mm_scaler.fit_transform(X)
print(f"After MinMaxScaler — min: {X_minmax.min(axis=0).round(4)}")
print(f"After MinMaxScaler — max: {X_minmax.max(axis=0).round(4)}")

# --- RobustScaler ---
print("\n--- RobustScaler: (x - median) / IQR ---")
print("""
Result: uses median and interquartile range instead of mean/std
Use when: data has OUTLIERS (sensor malfunctions, data entry errors)
WHY robust: median and IQR are not affected by extreme values
""")
# Add some outliers to demonstrate
X_with_outliers = X.copy()
X_with_outliers[0, 0] = 1000  # extreme temperature outlier
X_with_outliers[1, 1] = 50000 # extreme area outlier

from sklearn.preprocessing import RobustScaler
rob_scaler = RobustScaler()
X_robust = rob_scaler.fit_transform(X_with_outliers)

std_scaler2 = StandardScaler()
X_standard2 = std_scaler2.fit_transform(X_with_outliers)

print(f"With outliers — StandardScaler max: {X_standard2[:, 0].max():.2f}  (outlier dominates!)")
print(f"With outliers — RobustScaler   max: {X_robust[:, 0].max():.2f}  (outlier controlled)")

# ===========================================================================
# PART 2: Handling Missing Values
# ===========================================================================
print("\n" + "=" * 60)
print("PART 2: Missing Values")
print("=" * 60)

print("""
Real datasets ALWAYS have missing values. Common causes:
  - Sensor malfunction (reading = NaN)
  - Survey non-response  
  - Data entry errors
  - Different data sources with mismatched fields

NEVER just drop all rows with NaN — you might lose 30% of your data!
NEVER leave NaN in your data — most algorithms will crash.
""")

# Create a dataset with missing values
df_missing = data.copy()
# Randomly introduce NaN values (simulate sensor failures)
for col in df_missing.columns:
    mask = np.random.random(n) < 0.1  # 10% missing per column
    df_missing.loc[mask, col] = np.nan

print(f"Missing values per column:")
print(df_missing.isnull().sum().to_string())
print(f"Total missing: {df_missing.isnull().sum().sum()} / {df_missing.size} values")

# --- SimpleImputer ---
print("\n--- SimpleImputer ---")

# Mean imputation (for normally distributed numerical data)
mean_imputer = SimpleImputer(strategy='mean')
X_mean_imp = mean_imputer.fit_transform(df_missing.values)
print(f"After mean imputation — NaN count: {np.isnan(X_mean_imp).sum()}")

# Median imputation (better for skewed data or data with outliers)
median_imputer = SimpleImputer(strategy='median')
X_median_imp = median_imputer.fit_transform(df_missing.values)
print(f"After median imputation — NaN count: {np.isnan(X_median_imp).sum()}")

# Most-frequent (for categorical data)
freq_imputer = SimpleImputer(strategy='most_frequent')

# Constant value
const_imputer = SimpleImputer(strategy='constant', fill_value=0)

print("""
Strategy selection guide:
  'mean'         → numerical, roughly normal distribution
  'median'       → numerical, skewed or has outliers  
  'most_frequent'→ categorical, or heavily skewed numerical
  'constant'     → when 0 or a specific value makes domain sense
""")

# --- KNNImputer (more sophisticated) ---
print("\n--- KNNImputer (uses neighboring samples) ---")
print("""
KNNImputer: fill missing value with weighted mean of K nearest neighbors.
WHY better: uses relationships between features, not just global statistics.
WHY slower: must compute distances to all training samples.
Use when: you have time and the imputation quality matters a lot.
""")
knn_imputer = KNNImputer(n_neighbors=5)
X_knn_imp = knn_imputer.fit_transform(df_missing.values)
print(f"After KNN imputation — NaN count: {np.isnan(X_knn_imp).sum()}")

# ===========================================================================
# PART 3: Encoding Categorical Variables
# ===========================================================================
print("\n" + "=" * 60)
print("PART 3: Categorical Encoders")
print("=" * 60)

# Sample data with categoricals
df_cat = pd.DataFrame({
    'city':      ['London', 'Paris', 'Berlin', 'London', 'Tokyo', 'Paris'],
    'quality':   ['Low', 'Medium', 'High', 'High', 'Low', 'Medium'],  # ordinal
    'has_solar': [1, 0, 1, 0, 1, 0],  # binary, already numeric
    'energy':    [450, 320, 580, 410, 290, 350],
})

print("Original categorical data:")
print(df_cat.to_string())

# --- LabelEncoder: only for ordinal data! ---
print("\n--- LabelEncoder (ONLY for ordinal/binary features) ---")
le = LabelEncoder()
df_cat['quality_encoded'] = le.fit_transform(df_cat['quality'])
print(f"Quality encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")
print(f"  ← This is WRONG for 'quality': High=0, Low=1, Medium=2 (no correct order!)")

# OrdinalEncoder: specify the correct order
print("\n--- OrdinalEncoder (for truly ordinal features with known order) ---")
ord_enc = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df_cat['quality_ordinal'] = ord_enc.fit_transform(df_cat[['quality']])
print(f"After OrdinalEncoder: Low→0, Medium→1, High→2")
print(df_cat[['quality', 'quality_ordinal']].to_string())

# OneHotEncoder: for nominal categories (no order)
print("\n--- OneHotEncoder (for nominal features like city) ---")
ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids collinearity
city_encoded = ohe.fit_transform(df_cat[['city']])
city_df = pd.DataFrame(city_encoded, columns=ohe.get_feature_names_out(['city']))
print(city_df.to_string())
print(f"\nDrop 'first' (London): {len(ohe.categories_[0])-1} columns instead of {len(ohe.categories_[0])}")

# ===========================================================================
# PART 4: ColumnTransformer — Different Treatment for Different Columns
# ===========================================================================
print("\n" + "=" * 60)
print("PART 4: ColumnTransformer — Mixed Data Pipeline")
print("=" * 60)

print("""
Real datasets have mixed types: some columns numerical, some categorical.
ColumnTransformer applies different transformers to different columns simultaneously.
""")

# Realistic building energy dataset
n_buildings = 300
df_buildings = pd.DataFrame({
    'floor_area':   np.random.uniform(50, 500, n_buildings),
    'num_floors':   np.random.randint(1, 20, n_buildings).astype(float),
    'year_built':   np.random.randint(1950, 2024, n_buildings).astype(float),
    'building_type': np.random.choice(['Residential', 'Commercial', 'Industrial'], n_buildings),
    'heating_type':  np.random.choice(['Gas', 'Electric', 'Heat Pump'], n_buildings),
    'energy_kwh':   np.random.uniform(5000, 50000, n_buildings),
})

# Add some missing values
df_buildings.loc[np.random.choice(n_buildings, 20, replace=False), 'floor_area'] = np.nan
df_buildings.loc[np.random.choice(n_buildings, 15, replace=False), 'num_floors'] = np.nan

print(f"Building dataset: {df_buildings.shape}")
print(df_buildings.dtypes.to_string())
print(f"Missing values: {df_buildings.isnull().sum().sum()}")

# Define feature types
numerical_cols   = ['floor_area', 'num_floors', 'year_built']
categorical_cols = ['building_type', 'heating_type']

X_raw = df_buildings[numerical_cols + categorical_cols]
y_raw = df_buildings['energy_kwh'].values

# ColumnTransformer: chain imputation + scaling for numerical,
#                    imputation + encoding for categorical
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline,   numerical_cols),
    ('cat', categorical_pipeline, categorical_cols),
])

# Full pipeline with model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model',        LogisticRegression(max_iter=500)),
])

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# Just show the preprocessing result
X_processed = preprocessor.fit_transform(X_train_b)
print(f"\nRaw X shape:       {X_raw.shape}")
print(f"Processed X shape: {X_processed.shape}  (3 scaled + imputed + 4 one-hot = {X_processed.shape[1]} cols)")
print(f"No NaN remaining:  {~np.isnan(X_processed).any()}")

# ===========================================================================
# PART 5: Visualization — Effect of Scaling
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

colors = ['steelblue', 'coral', 'green', 'purple']
labels = ['outdoor_temp', 'floor_area', 'occupants', 'energy']

for i, ax, title, X_scaled in [
    (0, axes[0], "Raw Data", X),
    (1, axes[1], "StandardScaler", X_standard),
    (2, axes[2], "RobustScaler", X_robust),
]:
    bp = ax.boxplot(X_scaled, patch_artist=True, notch=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title)
    ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

plt.suptitle("Effect of Scaling on Feature Distributions", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "preprocessing_scaling.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\nScaling comparison plot saved: {OUTPUT_DIR}/preprocessing_scaling.png")

print("""
===========================================
KEY TAKEAWAYS
===========================================
1. StandardScaler → default choice for most algorithms
2. MinMaxScaler   → when you need [0,1] range (neural nets)
3. RobustScaler   → when outliers are present
4. SimpleImputer  → fill NaN before any algorithm
5. OneHotEncoder  → for nominal categories (city, color, type)
6. OrdinalEncoder → for ordinal categories (Low < Medium < High)
7. ColumnTransformer → apply different preprocessing to different columns
8. Always fit preprocessors on TRAINING data, apply to both train + test

Next: 03_pipelines.py — Combine preprocessing + model cleanly
""")
