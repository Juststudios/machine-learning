"""
LESSON: What Is Machine Learning?
==================================
This lesson contrasts traditional programming with machine learning using a
concrete house-price prediction example. By the end, you will understand:

  - WHY ML exists: some rules are too complex to hand-code
  - WHAT a model "learns": numerical coefficients from data
  - HOW to execute the basic sklearn workflow: fit → predict → evaluate
  - WHAT the model's output (coefficients) physically means

Real engineering context: The same workflow you see here — fitting a model
to historical sensor data and using it to predict future behaviour — appears
in predictive maintenance, process control, and structural health monitoring.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Reproducibility: fix the random seed so every run produces the same data
np.random.seed(42)

# ============================================================
# PART 1: TRADITIONAL PROGRAMMING
# ============================================================
print("=" * 60)
print("PART 1: Traditional Programming (Rule-Based)")
print("=" * 60)

# WHY: The programmer manually writes every rule. This works when the rules
# are simple and known. But what if the rules are complex, or there are
# thousands of interacting factors?

def traditional_house_price(sqft, bedrooms, distance_to_city_km=None, **kwargs):
    """
    A hand-coded pricing rule written by a domain expert.
    Problem: these coefficients are guesses. Are they right?
    Real estate markets are far too complex for hand-coded rules.
    """
    dist = kwargs.get("distance_km", distance_to_city_km if distance_to_city_km is not None else 0)
    price = 0
    price += sqft * 150          # $150 per square foot (guessed)
    price += bedrooms * 8000     # $8k per bedroom (guessed)
    price -= dist * 2000         # Penalize distance (guessed)
    price += 50000               # Base price (guessed)
    return price

# Test the hand-coded rule
example_house = {"sqft": 1800, "bedrooms": 3, "distance_km": 5}
traditional_pred = traditional_house_price(**example_house)
print(f"\nHouse: {example_house}")
print(f"Traditional rule prediction: ${traditional_pred:,.0f}")
print("\nProblem: The coefficients ($150/sqft, $8k/bedroom, etc.) were GUESSED.")
print("A machine learning model will LEARN better coefficients from data.")

# ============================================================
# PART 2: GENERATE SYNTHETIC TRAINING DATA
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Creating Training Data")
print("=" * 60)

# WHY synthetic data: we need a dataset where we know the true answer
# so we can verify the model learned correctly. In practice, you would
# use real market data (e.g., from a property database).

n_houses = 500  # We have 500 historical house sales to learn from

# Generate realistic feature values
sqft          = np.random.uniform(800, 3500, n_houses)      # 800–3500 sq ft
bedrooms      = np.random.randint(1, 6, n_houses)           # 1–5 bedrooms
distance_km   = np.random.uniform(1, 30, n_houses)          # 1–30 km from city

# TRUE pricing rule (hidden from the model — it must discover this from data):
#   price = 120·sqft + 10000·bedrooms - 3000·distance + 45000 + noise
true_w_sqft     = 120    # $/sqft
true_w_bedroom  = 10000  # $ per bedroom
true_w_distance = -3000  # $ per km from city
true_intercept  = 45000  # base price
noise           = np.random.normal(0, 15000, n_houses)  # market noise

prices = (true_w_sqft * sqft +
          true_w_bedroom * bedrooms +
          true_w_distance * distance_km +
          true_intercept + noise)

print(f"\nDataset created: {n_houses} house sales")
print(f"Price range: ${prices.min():,.0f} — ${prices.max():,.0f}")
print(f"Mean price: ${prices.mean():,.0f}")
print(f"\nTrue rule (hidden from model):")
print(f"  price = {true_w_sqft}·sqft + {true_w_bedroom}·bedrooms")
print(f"          + ({true_w_distance})·distance_km + {true_intercept}")

# ============================================================
# PART 3: PREPARE DATA FOR SKLEARN
# ============================================================
print("\n" + "=" * 60)
print("PART 3: Preparing Data — Features Matrix X and Target Vector y")
print("=" * 60)

# WHY: sklearn expects X as a 2D matrix (n_samples, n_features)
# and y as a 1D vector (n_samples,)
# Each ROW of X is one house. Each COLUMN is one feature.

# Stack the individual feature arrays into a matrix: shape (500, 3)
X = np.column_stack([sqft, bedrooms, distance_km])
y = prices

print(f"\nFeature matrix X shape: {X.shape}")
print(f"  → {X.shape[0]} houses × {X.shape[1]} features")
print(f"Target vector y shape:  {y.shape}")
print(f"\nFirst 3 rows of X (sqft, bedrooms, distance_km):")
print(X[:3].round(1))
print(f"\nFirst 3 prices (y):")
print(y[:3].round(0))

# ============================================================
# PART 4: TRAIN/TEST SPLIT — WHY THIS IS NOT OPTIONAL
# ============================================================
print("\n" + "=" * 60)
print("PART 4: Train/Test Split")
print("=" * 60)

# WHY: If we evaluate performance on the same data we trained on, we're
# cheating — like a student grading their own practice problems.
# The TEST SET simulates data the model has never seen.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% held out for testing (100 houses)
    random_state=42   # Fixed seed → reproducible split
)

print(f"\nTraining set: {X_train.shape[0]} houses  (model sees this)")
print(f"Test set:     {X_test.shape[0]}  houses  (model never sees this during training)")
print("\nThe model will learn coefficients from training data.")
print("We will evaluate it on test data to measure real-world performance.")

# ============================================================
# PART 5: MACHINE LEARNING — LET THE MODEL LEARN THE RULES
# ============================================================
print("\n" + "=" * 60)
print("PART 5: Machine Learning — model.fit()")
print("=" * 60)

# WHY LinearRegression: Our data was generated by a linear relationship
# (price = w1·sqft + w2·bedrooms + w3·distance + b).
# Linear regression finds the w's and b that minimize prediction error.

model = LinearRegression()

print("\nBefore training, the model knows nothing.")
print("Calling model.fit(X_train, y_train) ...")
print("  → Internally, this solves: w* = (XᵀX)⁻¹ Xᵀy")
print("  → This finds the OPTIMAL coefficients that minimize mean squared error")

model.fit(X_train, y_train)

print("\nTraining complete!")
print("\nWhat the model LEARNED (its internal parameters):")
feature_names = ["sqft", "bedrooms", "distance_km"]
for name, coeff in zip(feature_names, model.coef_):
    print(f"  w_{name:12s} = {coeff:8.1f}  (True value: {[true_w_sqft, true_w_bedroom, true_w_distance][feature_names.index(name)]})")
print(f"  intercept (b)  = {model.intercept_:8.1f}  (True value: {true_intercept})")
print("\nThe model recovered the true coefficients from data alone — no hand-coding!")

# ============================================================
# PART 6: PREDICTIONS ON NEW DATA
# ============================================================
print("\n" + "=" * 60)
print("PART 6: Making Predictions — model.predict()")
print("=" * 60)

# WHY: After training, we can use the model to price any house,
# even ones not in the dataset. This is the whole point of ML.

y_pred = model.predict(X_test)

print(f"\nPredictions on {len(y_test)} unseen houses:")
print(f"\n{'Actual Price':>15} {'Predicted Price':>16} {'Error':>12}")
print("-" * 45)
for actual, predicted in zip(y_test[:8], y_pred[:8]):
    error = predicted - actual
    print(f"  ${actual:>12,.0f}   ${predicted:>12,.0f}   ${error:>10,.0f}")

# ============================================================
# PART 7: EVALUATION — HOW GOOD IS THE MODEL?
# ============================================================
print("\n" + "=" * 60)
print("PART 7: Evaluation — model.score() and metrics")
print("=" * 60)

# WHY multiple metrics: Each metric tells a different story about performance.
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)
rmse   = np.sqrt(np.mean((y_test - y_pred) ** 2))

print(f"\nModel Performance on Test Set ({len(y_test)} houses):")
print(f"  MAE  (Mean Absolute Error)  = ${mae:>10,.0f}")
print(f"       → On average, predictions are off by ${mae:,.0f}")
print(f"  RMSE (Root Mean Sq. Error)  = ${rmse:>10,.0f}")
print(f"       → Like MAE but penalizes large errors more")
print(f"  R²   (Coefficient of Det.)  = {r2:.4f}")
print(f"       → {r2*100:.1f}% of price variance explained by the model")
print(f"       → 1.0 = perfect, 0.0 = no better than predicting the mean")

# ============================================================
# PART 8: THE KEY INSIGHT
# ============================================================
print("\n" + "=" * 60)
print("PART 8: The Key Insight — Rules vs. Learned Patterns")
print("=" * 60)

print("""
TRADITIONAL PROGRAMMING:
  You write rules → program applies them
  Problem: rules are hand-coded, may be wrong, can't handle complexity

MACHINE LEARNING:
  You provide data (examples + answers) → algorithm discovers rules
  
  The "rules" are the MODEL PARAMETERS (coefficients, weights):
    - LinearRegression: w1, w2, ..., b
    - Neural network:   billions of weights
    - Decision tree:    thresholds and branches

  After training, prediction is just arithmetic:
    price = w_sqft · sqft + w_bedrooms · bedrooms + w_distance · distance + b

  The magic: these numbers were LEARNED, not written by a human.
""")

print("Compare our ML model vs. the hand-coded rule:")
# Predict the same example house with both approaches
example_X = np.array([[1800, 3, 5]])  # 1800 sqft, 3 bed, 5 km
ml_pred = model.predict(example_X)[0]
rule_pred = traditional_house_price(1800, 3, 5)

# True price (no noise for comparison)
true_price = (true_w_sqft * 1800 + true_w_bedroom * 3 +
              true_w_distance * 5 + true_intercept)

print(f"  House: 1800 sqft, 3 bedrooms, 5 km from city")
print(f"  True price (no noise):      ${true_price:>10,.0f}")
print(f"  ML prediction:              ${ml_pred:>10,.0f}   (error: ${abs(ml_pred - true_price):,.0f})")
print(f"  Hand-coded rule prediction: ${rule_pred:>10,.0f}   (error: ${abs(rule_pred - true_price):,.0f})")
print(f"\n  → ML learned better coefficients from data than the hand-coded guess!")

print("\n" + "=" * 60)
print("SUMMARY: What You Just Did")
print("=" * 60)
print("""
1. Created data: X (features) and y (target)
2. Split: 80% train / 20% test (held out!)
3. model.fit(X_train, y_train)  → model learned w1, w2, w3, b
4. model.predict(X_test)        → applied learned rules to new data
5. Evaluated: MAE, RMSE, R²    → measured real-world performance

Next: 02_ml_workflow.py — walk through the FULL ML workflow step by step
""")
