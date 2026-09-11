"""
Exercises: Scikit-learn API, Preprocessing, Pipelines
=======================================================
4-tier exercises testing your ability to build sklearn pipelines,
handle real-world preprocessing challenges, and prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42)

print("=" * 60)
print("SCIKIT-LEARN API & PREPROCESSING — EXERCISES")
print("=" * 60)

# ===========================================================================
# LEVEL 1: RECALL
# ===========================================================================
print("\n--- LEVEL 1: RECALL ---")
print("""
Answer these questions by running the demo code and reading the output:

1a. What is the difference between fit(), transform(), and fit_transform()?
1b. Why must we fit the scaler ONLY on training data?
1c. In a Pipeline, which step can be the last step? Which can be middle steps?
1d. What does SimpleImputer(strategy='median') do with NaN values?
1e. If a Pipeline has steps [('scaler', StandardScaler()), ('ridge', Ridge())],
    how do you set ridge's alpha to 10 using set_params()?
""")

# Demo
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=4, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)  # learn stats from train only
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"scaler.mean_: {scaler.mean_.round(3)}")
print(f"After scaling, X_train_s.mean(axis=0): {X_train_s.mean(axis=0).round(4)}")
print(f"After scaling, X_test_s.mean(axis=0):  {X_test_s.mean(axis=0).round(4)}  (not exactly 0 — correct!)")

pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])
print(f"\nDefault pipeline params: {pipe.get_params()}")

# ===========================================================================
# LEVEL 2: DEBUGGING
# ===========================================================================
print("\n--- LEVEL 2: DEBUGGING ---")
print("""
The following code has data leakage and an incorrect evaluation.
Find and fix all 3 bugs.
""")

buggy = '''
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score

np.random.seed(42)
X = np.random.randn(200, 8)
y = X @ np.array([2, -1, 0.5, 0, 1.5, 0, -2, 0.3]) + np.random.randn(200)

# BUG 1: Fit scaler before split (data leakage!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# BUG 2: Cross-validate the scaled X (CV doesn't re-fit the scaler per fold → leakage!)
ridge = Ridge()
cv_scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring="r2")
print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# BUG 3: Evaluate on training set and call it "test performance"
ridge.fit(X_train, y_train)
test_r2 = ridge.score(X_train, y_train)  # using X_train instead of X_test!
print(f"Test R²: {test_r2:.4f}")
'''
print(buggy)
print("""
Bugs found:
  Bug 1: ________________________________________________
  Bug 2: ________________________________________________
  Bug 3: ________________________________________________
""")
print("TODO: Write the correct version below")

# ===========================================================================
# LEVEL 3: APPLICATION — Real-World Housing Pipeline
# ===========================================================================
print("\n--- LEVEL 3: APPLICATION ---")
print("""
TASK: Build a complete ML pipeline for predicting house energy consumption.

Dataset (generate it):
  Numerical features (may have NaN):
    - floor_area_m2:  50-500 (10% missing)
    - year_built:     1960-2024 (5% missing)
    - num_rooms:      2-12 (5% missing)
    - avg_temp_C:     -5 to 25 (no missing)
    
  Categorical features:
    - building_type:  ['Apartment', 'House', 'Office', 'Warehouse']
    - heating_system: ['Gas', 'Electric', 'Heat Pump', 'None']
  
  Target:
    - energy_kwh_per_year (continuous, regression target)

Requirements:
  1. Generate 400 samples (use a realistic formula for energy)
  2. Split 80/20 (train/test)
  3. Build a ColumnTransformer:
     - Numerical: SimpleImputer(median) → StandardScaler
     - Categorical: SimpleImputer(most_frequent) → OneHotEncoder(drop='first')
  4. Build a full Pipeline with the preprocessor + RandomForestRegressor
  5. 5-fold cross-validate on training set (report mean ± std R²)
  6. Evaluate on test set (report R² and RMSE)
  7. Run GridSearchCV over at least 2 RandomForest hyperparameters
  8. Report the best parameters and final test R²
""")
print("TODO: Implement the housing energy pipeline above")

# Starter: data generation
n_samples = 400
building_types  = ['Apartment', 'House', 'Office', 'Warehouse']
heating_systems = ['Gas', 'Electric', 'Heat Pump', 'None']

floor_area   = np.random.uniform(50, 500, n_samples)
year_built   = np.random.randint(1960, 2024, n_samples).astype(float)
num_rooms    = np.random.randint(2, 12, n_samples).astype(float)
avg_temp     = np.random.uniform(-5, 25, n_samples)
btype        = np.random.choice(building_types, n_samples)
heating      = np.random.choice(heating_systems, n_samples)

# Simple energy formula (your pipeline should recover this)
energy = (
    50 * floor_area
    + 200 * (2024 - year_built)
    + 1000 * num_rooms
    - 500 * avg_temp
    + np.where(heating == 'None', 5000, 0)
    + np.random.normal(0, 3000, n_samples)
)
energy = np.clip(energy, 1000, None)

# Add missing values
floor_area[np.random.choice(n_samples, int(0.1*n_samples))] = np.nan
year_built[np.random.choice(n_samples, int(0.05*n_samples))] = np.nan
num_rooms[np.random.choice(n_samples, int(0.05*n_samples))]  = np.nan

df = pd.DataFrame({
    'floor_area': floor_area, 'year_built': year_built,
    'num_rooms': num_rooms, 'avg_temp': avg_temp,
    'building_type': btype, 'heating_system': heating,
})

print(f"\nDataset created: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print("TODO: Build the ColumnTransformer pipeline and GridSearchCV above")


# ===========================================================================
# LEVEL 4: CHALLENGE — Custom Transformer
# ===========================================================================
print("\n--- LEVEL 4: CHALLENGE ---")
print("""
TASK: Implement a custom sklearn-compatible transformer using BaseEstimator
and TransformerMixin. Your transformer should:

  class OutlierClipper(BaseEstimator, TransformerMixin):
      '''Clip values beyond n_std standard deviations to the boundary.'''
      
      def __init__(self, n_std=3.0):
          self.n_std = n_std
      
      def fit(self, X, y=None):
          # Compute mean and std from TRAINING data
          # Store as self.lower_ and self.upper_
          return self
      
      def transform(self, X, y=None):
          # Clip values to [lower_, upper_] bounds
          return X_clipped

Requirements:
  - Works inside a Pipeline (implements fit + transform)
  - fit() stores the clip boundaries from training data
  - transform() applies those boundaries to any data
  - Test it: add 5% extreme outliers to data, show it clips them
  - Use it in a Pipeline: OutlierClipper → StandardScaler → Ridge
  - Cross-validate the pipeline and report R²

Starter:
""")
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierClipper(BaseEstimator, TransformerMixin):
    """TODO: Implement this transformer."""
    
    def __init__(self, n_std=3.0):
        self.n_std = n_std
    
    def fit(self, X, y=None):
        """Compute clip boundaries from training data."""
        # TODO: compute self.lower_ and self.upper_
        return self
    
    def transform(self, X, y=None):
        """Clip values to the learned boundaries."""
        # TODO: apply clipping
        return X  # REPLACE with clipped version

print("TODO: Implement OutlierClipper and integrate it into a Pipeline")
print("  Test: verify it reduces the max value after adding extreme outliers")

print("\n" + "=" * 60)
print("EXERCISES COMPLETE")
print("Check solutions/sklearn_solutions.py for reference answers.")
print("=" * 60)
