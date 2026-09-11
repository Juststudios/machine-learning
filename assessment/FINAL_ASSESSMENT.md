# FINAL ASSESSMENT: Machine Learning & Deep Learning

## Overview

This assessment covers the complete Level 3+4 curriculum:
- **Modules 1–6:** Classical ML (sklearn)
- **Modules 7–10:** PyTorch, Neural Networks, CNNs
- **Module 11:** Transformers

**Total time:** Approximately 4–6 hours  
**Format:** Written answers + Python code  
**Grading:** Self-assessed against the rubric at the bottom

---

## SECTION 1: Conceptual Questions (30 marks)

Answer each question clearly and concisely.

### 1.1 Core ML Concepts (10 marks)

**Q1.** Explain the bias-variance tradeoff. A student says "my model has Train R²=0.99 but Test R²=0.52." Diagnose the problem and name TWO model-specific solutions.

**Q2.** You are building a model to detect manufacturing defects. Only 2% of products are defective. A colleague trains a model and reports "98% accuracy!" Why is this misleading? Which 3 metrics would you use instead, and why?

**Q3.** What is data leakage? Give a concrete example of how it occurs with a StandardScaler. How does using a Pipeline prevent it?

**Q4.** Compare Ridge (L2) and Lasso (L1) regularization:
- What does each penalize?
- Which one performs automatic feature selection? Why?
- When would you prefer Ridge over Lasso?

**Q5.** A neural network trained for 200 epochs shows: train loss steadily decreasing, val loss decreasing until epoch 50 then increasing. Describe what is happening and the THREE standard techniques to fix it.

---

### 1.2 Deep Learning Concepts (10 marks)

**Q6.** Why do we use ReLU instead of Sigmoid as the activation function in hidden layers of deep networks? Explain in terms of gradients.

**Q7.** What does BatchNorm1d do? Explain what happens during `model.train()` vs `model.eval()` with BatchNorm and Dropout.

**Q8.** You have 200 images per class (4 classes). You want to use a CNN. Should you train from scratch or use transfer learning? Justify your answer. What are the two main strategies for transfer learning?

**Q9.** Explain the self-attention mechanism in transformers:
- What are Q, K, and V?
- Write the attention formula.
- Why does multi-head attention use multiple heads instead of one?

**Q10.** Compare these sequence models for a 500-timestep sensor series classification:
- Simple MLP (flatten → linear layers)
- CNN (1D convolutions)
- Transformer (self-attention)

Name one advantage and one disadvantage of each.

---

### 1.3 Practical Judgment (10 marks)

**Q11.** You have a dataset with 50 samples and 200 features. Which of these models would you choose and why?
- LogisticRegression with no regularization
- Ridge (L2)
- Lasso (L1)
- Random Forest (100 trees)
- Deep MLP (5 layers)

**Q12.** DBSCAN vs K-Means: for each scenario, which would you use and why?
- Scenario A: 1 million customer records, expect 10 market segments of similar size
- Scenario B: GPS location data for a fleet of vehicles, some vehicles have erratic paths (anomalies)

**Q13.** A colleague asks: "I have image data. Should I use a CNN or flatten the images into a vector and use a plain MLP?" Give a complete answer including when each approach is appropriate.

**Q14.** You are evaluating a fraud detection model. The PR-AUC is 0.42, and ROC-AUC is 0.91. Which metric do you trust more? Why is there such a large difference between the two scores?

**Q15.** A student's Transformer model for time-series classification achieves 65% accuracy while an LSTM achieves 78% on the same data with 300 training samples. What might explain this? How would you fix it?

---

## SECTION 2: Code Implementation (50 marks)

Write clean, runnable Python code for each task.

### 2.1 Classical ML Pipeline (15 marks)

Build a complete ML pipeline for the sensor fault dataset at `datasets/industrial_sensor_train.csv`.

```python
# Starter:
import pandas as pd
train = pd.read_csv("datasets/industrial_sensor_train.csv")
test  = pd.read_csv("datasets/industrial_sensor_test.csv")
```

**Requirements:**
1. Load both datasets and print class distribution of `fault_severity`
2. Build a ColumnTransformer that:
   - Imputes missing values in `pressure` and `current` (check which strategy is appropriate)
   - Scales all 8 numerical features
3. Build a full Pipeline: preprocessor → `RandomForestClassifier(class_weight='balanced')`
4. 5-fold stratified cross-validation on the training set (report macro-F1 mean ± std)
5. Use GridSearchCV to tune at least 2 RandomForest hyperparameters
6. Evaluate the best model on the TEST set:
   - Confusion matrix (as a heatmap)
   - Classification report (per-class precision, recall, F1)
   - Overall macro-F1

---

### 2.2 Neural Network From Scratch (15 marks)

Implement a PyTorch MLP for `fault_severity` classification.

**Requirements:**
1. Load and preprocess `industrial_sensor_train.csv` (impute + scale)
2. Build a PyTorch MLP: `Linear(8→64) → BN → ReLU → Dropout(0.3) → Linear(64→32) → BN → ReLU → Linear(32→3)`
3. Use `CrossEntropyLoss` with class weights computed from training data:
   ```python
   weights = torch.tensor([n_total / (n_classes * class_count[c]) for c in range(n_classes)])
   criterion = nn.CrossEntropyLoss(weight=weights)
   ```
4. Adam optimizer (lr=0.001), `ReduceLROnPlateau(patience=10)`
5. Training loop: 200 epochs, early stopping (patience=20), save best model state
6. Report: test accuracy and macro-F1
7. Plot training loss + validation accuracy curves
8. Compare with your sklearn Random Forest from Task 2.1 — discuss results

---

### 2.3 CNN Implementation (10 marks)

Build and train a CNN for your own synthetic image dataset.

**Requirements:**
1. Generate 300 samples per class (4 classes, 32×32 grayscale) — use any pattern distinctions
2. CNN architecture: 2 conv blocks (Conv → BN → ReLU → Pool) → FC head → 4 logits
3. Train with Adam (lr=0.001), CrossEntropyLoss, 50 epochs
4. Plot sample images from each class + training curves
5. Compare CNN accuracy vs a flattened-image MLP on the same data

---

### 2.4 Transformer for Time-Series (10 marks)

Build a Transformer-based classifier for multivariate time-series.

**Requirements:**
1. Generate time-series data: 3 classes, 40 timesteps, 6 features
2. Build a Transformer encoder:
   - Input projection: `Linear(6 → 32)`
   - 2× TransformerEncoderBlock (d=32, heads=4)
   - CLS token → FC → 3 logits
3. Train 80 epochs (Adam lr=3e-4, CosineAnnealingLR)
4. Evaluate: accuracy and per-class F1
5. Visualize one attention heatmap (30×30 attention map for a single sample)

---

## SECTION 3: Capstone Integration (20 marks)

Use `12_capstone/starter_template.py` as your starting point.

**Task:** Build a complete ML pipeline for the industrial sensor capstone dataset.

**Requirements:**

1. **Exploratory Data Analysis (4 marks)**
   - Distribution of each feature (histograms)
   - Correlation matrix heatmap
   - Class balance of `fault_severity`
   - Distribution of `remaining_useful_life`
   - Missing value summary

2. **Preprocessing Pipeline (4 marks)**
   - Handle all missing values appropriately
   - Scale all features
   - Use a ColumnTransformer pipeline (no separate steps)

3. **Model Development (8 marks)**
   - Train THREE models for `fault_severity` classification:
     a. RandomForestClassifier (sklearn)
     b. PyTorch MLP (at least 2 hidden layers)
     c. Your choice of any other model
   - Cross-validate all three (5-fold, stratified, macro-F1)
   - Select the best model for the test set

4. **Final Evaluation (4 marks)**
   - Best model evaluated on the holdout test set
   - Confusion matrix + classification report for `fault_severity`
   - Final macro-F1 ≥ 0.70 (sensor data is noisy — this is a realistic target)
   - 1-paragraph reflection: what was hardest? what would you try next?

---

## Rubric

| Section | Points | Criteria |
|---|---|---|
| 1.1 Core ML Concepts | 10 | Correct, clear explanations |
| 1.2 Deep Learning | 10 | Technical accuracy + intuition |
| 1.3 Practical Judgment | 10 | Reasoning quality + real-world awareness |
| 2.1 Classical Pipeline | 15 | Code runs, correct preprocessing, CV, GridSearchCV |
| 2.2 PyTorch MLP | 15 | Correct architecture, training loop, early stopping |
| 2.3 CNN | 10 | Correct Conv architecture, comparison |
| 2.4 Transformer | 10 | Working transformer, attention visualization |
| 3. Capstone | 20 | EDA depth, pipeline quality, model comparison, reflection |
| **Total** | **100** | |

### Score Interpretation
- **90–100:** Ready for applied ML engineering roles
- **75–89:** Strong understanding; ready for ML projects
- **60–74:** Good foundation; review weak areas and rebuild
- **< 60:** Review fundamentals and revisit modules

---

> [!TIP]
> **Exam strategy:** Start with Section 3 (Capstone) as it integrates everything and you can build on that code for other sections.

> [!NOTE]
> **Allowed resources:** All lesson files in this package, sklearn/PyTorch documentation, your own notes. This is an open-book learning assessment.
