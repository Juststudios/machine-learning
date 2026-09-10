# Module 12: Capstone Project
## Predictive Failure Detection System

> *"Every day, industrial machines fail unexpectedly — costing manufacturers billions in downtime. Your job: use sensor data to predict failure before it happens."*

---

## The Scenario

You are a data scientist hired by **Meridian Industrial**, a manufacturer that operates a fleet of critical pumping machines. Each machine is instrumented with **8 sensors** that record operational data every hour.

The engineering team has been logging this data for months. They know from historical records when machines have failed and what the fault severity was at failure time. Your task is to:

1. **Explore** the sensor data and understand what normal vs. faulty operation looks like
2. **Build predictive models** that can:
   - Predict *how many hours* until the next failure (regression)
   - Predict *current fault severity* (classification: OK / WARNING / FAULT)
3. **Compare** classical ML and deep learning approaches
4. **Deliver** a recommendation to the engineering team

This is a real engineering workflow. No hand-holding. You have a starter template, a dataset, and the rubric below.

---

## The Dataset

Run `python generate_capstone_data.py` to create:
- `datasets/industrial_sensor_train.csv` — 1000 training samples
- `datasets/industrial_sensor_test.csv` — 200 test samples

### Features (8 sensor channels)

| Column | Units | Description |
|--------|-------|-------------|
| `temperature` | °C | Bearing temperature |
| `vibration` | mm/s | Vibration amplitude (RMS) |
| `pressure` | bar | Internal pressure |
| `current` | A | Motor current draw |
| `rpm` | rev/min | Shaft rotation speed |
| `humidity` | % | Ambient humidity |
| `voltage` | V | Supply voltage |
| `acoustic_emission` | dB | High-frequency acoustic signal |

### Targets (two prediction tasks)

| Column | Type | Description |
|--------|------|-------------|
| `remaining_useful_life` | float (0–500 hrs) | Hours until predicted failure |
| `fault_severity` | int (0/1/2) | 0=OK, 1=WARNING, 2=FAULT |

---

## Three Tracks

Choose **one track** to submit. Each track is worth full marks on the rubric.

| Track | Models Required | Best For |
|-------|----------------|----------|
| **Classical ML** | Random Forest + SVM (both tasks) | Stronger classical ML foundation |
| **Deep Learning** | PyTorch MLP (both tasks) | Stronger DL background |
| **Combined** | Classical + DL, compared side-by-side | Full marks on comparison section |

The **Combined** track is recommended if you have completed Modules 1–11.

---

## Deliverables

1. **`capstone_submission.py`** — Your complete Python solution script
   - Must run from top to bottom without errors
   - Must save all plots to `output/` directory
   - Must print a final summary of all model metrics

2. **`capstone_report.md`** — Written report (500–1000 words)
   - Describe your data exploration findings
   - Justify your model choices
   - Interpret your evaluation metrics
   - Write 3 specific recommendations for the engineering team

---

## Grading Rubric (200 Points)

### Section 1: Data Exploration and Preprocessing (30 pts)

| Criterion | Points |
|-----------|--------|
| Load and inspect the dataset correctly (shapes, dtypes, head) | 5 |
| Visualize distributions of all 8 features | 5 |
| Identify and handle missing values (don't just drop rows) | 10 |
| Visualize correlation structure (correlation heatmap or pairplot) | 5 |
| Split features from targets correctly, no data leakage | 5 |

### Section 2: Feature Engineering (20 pts)

| Criterion | Points |
|-----------|--------|
| Apply appropriate scaling (StandardScaler or MinMaxScaler) | 5 |
| Justify scaling choice in comments | 5 |
| Create at least 1 engineered feature (e.g., temperature × vibration interaction) | 5 |
| Handle the two targets correctly (regression vs. classification) | 5 |

### Section 3: Classical ML Models (40 pts)

| Criterion | Points |
|-----------|--------|
| Train at least one regression model for RUL prediction | 10 |
| Train at least one classification model for fault severity | 10 |
| Use cross-validation (not just a single train/test split) | 10 |
| Perform hyperparameter tuning (GridSearchCV or RandomizedSearchCV) | 10 |

### Section 4: Deep Learning Model (50 pts)

| Criterion | Points |
|-----------|--------|
| Define a PyTorch MLP with at least 2 hidden layers | 10 |
| Implement a correct training loop (forward, loss, backward, step) | 15 |
| Track and plot training loss over epochs | 10 |
| Train on regression task OR classification task (your choice) | 10 |
| Use appropriate loss function (MSE for regression, CrossEntropy for classification) | 5 |

### Section 5: Evaluation and Comparison (30 pts)

| Criterion | Points |
|-----------|--------|
| Report RMSE and R² for all regression models | 10 |
| Report accuracy, precision, recall, F1 for all classification models | 10 |
| Present a confusion matrix for classification | 5 |
| Compare classical vs. DL performance and explain differences | 5 |

### Section 6: Interpretability and Recommendations (30 pts)

| Criterion | Points |
|-----------|--------|
| Plot feature importances (from Random Forest or similar) | 10 |
| Write 3 specific, actionable recommendations for Meridian Industrial | 10 |
| Identify at least 1 limitation of your approach | 5 |
| Discuss what additional data would improve predictions | 5 |

---

## Timeline

| Week | Tasks |
|------|-------|
| **Week 1** | Data exploration, preprocessing, classical ML models |
| **Week 2** | Deep learning model, evaluation, report writing |

---

## Getting Started

```bash
# Step 1: Generate the dataset
python generate_capstone_data.py

# Step 2: Copy the starter template
cp starter_template.py capstone_submission.py

# Step 3: Work through each TODO section
# (Do NOT look at solutions/capstone_solution.py until you submit)

# Step 4: Run your solution
python capstone_submission.py
```

---

## Tips from Engineers Who Have Done This

- **Start with EDA.** You cannot build a good model without understanding the data first.
- **Check class balance** before choosing metrics. Accuracy is misleading on imbalanced data.
- **A simple model that you understand is better than a complex model you don't.**
- **Your written recommendations matter as much as your code.** Engineers need to act on your findings.
- **Plot everything.** A plot reveals what 100 print statements cannot.

Good luck. Build something real.
