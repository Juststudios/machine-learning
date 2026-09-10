# Machine Learning & Deep Learning Curriculum
## Level 3 & 4 of the Engineering + AI Learning Path

> *"The goal is not to learn algorithms. The goal is to think like an engineer who can reach for the right tool, understand why it works, and know when it fails."*

---

## The Full Learning Path

This package is part of a structured, four-level curriculum designed to take you from Python basics to production-ready deep learning systems.

| Level | Package | Topics |
|-------|---------|--------|
| **Level 1** | Python Data Tools | NumPy, Pandas, Matplotlib — the foundation of all data work |
| **Level 2** | Engineering Mathematics | Linear Algebra, Calculus, Probability, Statistics with MATLAB |
| **Level 3** | **Machine Learning** ← *You are here* | Supervised Learning, Unsupervised Learning, Model Evaluation |
| **Level 4** | **Deep Learning** ← *Also here* | PyTorch, Neural Networks, CNNs, Transformers |

Each level builds directly on the previous. You will use NumPy and Pandas *constantly* in this package. The mathematics from Level 2 will appear in every lesson — not as theory, but as the reason algorithms work.

---

## What You Will Build

By completing this curriculum, you will have built **five real engineering systems**:

1. **Predictive Maintenance Classifier** — Detect machine faults before they happen using sensor data (Modules 3–4)
2. **Customer Segmentation Engine** — Group users by behavior for a SaaS product using clustering (Module 5)
3. **Medical Diagnosis Pipeline** — Classify chest X-ray findings with evaluation-first thinking (Module 6)
4. **MNIST Digit Recognizer** — Build a neural network from scratch in PyTorch, then with CNNs (Modules 8–10)
5. **Industrial Failure Detection System** — End-to-end ML + Deep Learning capstone project (Module 12)

---

## Package Overview

This package contains **11 teaching modules** plus a capstone project, organized from classical ML fundamentals through modern deep learning.

### Module Map

```
Modules 1–2: Foundation            → What is ML? How does sklearn work?
Modules 3–4: Supervised Learning   → Regression and Classification
Module  5:   Unsupervised Learning → Clustering (K-Means, DBSCAN, hierarchical)
Module  6:   Model Evaluation      → Cross-validation, metrics, bias-variance
Modules 7–9: Deep Learning Intro   → PyTorch basics, training loops, MLPs
Module 10:   Convolutional Nets    → CNNs for image and time-series data
Module 11:   Transformers          → Attention, BERT fine-tuning, intro to LLMs
Module 12:   Capstone              → Full industrial ML/DL system
```

### Teaching Philosophy

Every lesson follows the same structure:
1. **Real Problem First** — Why would an engineer need this?
2. **Intuition** — What does this algorithm actually *do*?
3. **Mathematics** — The equations, explained in English
4. **Code** — Implementation with `# WHY:` comments throughout
5. **Results** — Visualize and interpret the output
6. **Exercises** — Four-tier practice system (see below)

---

## Technology Stack

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.14+ | Core language |
| **scikit-learn** | 1.9+ | Classical ML algorithms |
| **PyTorch** | 2.13+ (CPU) | Deep learning framework |
| **NumPy** | 2.0+ | Array operations |
| **Pandas** | 2.0+ | Data manipulation |
| **Matplotlib** | 3.7+ | Visualization |

> **Note on TensorFlow:** TensorFlow is referenced in optional, historical sections for context. The primary deep learning framework in this curriculum is **PyTorch**. All runnable code uses PyTorch only.

---

## Prerequisites

Before starting this package, you should be comfortable with:

**From Level 1 (Python Data Tools):**
- [ ] NumPy: array creation, indexing, broadcasting, `np.dot`, `np.linalg`
- [ ] Pandas: DataFrame creation, filtering, groupby, CSV I/O
- [ ] Matplotlib: line plots, scatter plots, subplots, histograms

**From Level 2 (Engineering Mathematics):**
- [ ] Linear Algebra: dot product, matrix multiplication, eigenvalues, SVD
- [ ] Calculus: derivatives, partial derivatives, the chain rule
- [ ] Probability: distributions, conditional probability, Bayes' theorem
- [ ] Statistics: mean, variance, standard deviation, correlation

If you are missing any of these, complete those levels first. The difficulty cliff is real.

---

## Directory Structure

```
machine-learning/
│
├── README.md                          ← This file
├── requirements.txt                   ← All dependencies
│
├── 01_ml_fundamentals/                ← What is ML? Supervised vs unsupervised
├── 02_scikit_learn/                   ← The sklearn API: fit, predict, transform
├── 03_regression/                     ← Linear, Ridge, Lasso, SVR, Decision Trees
├── 04_classification/                 ← Logistic Regression, SVM, Random Forest
├── 05_clustering/                     ← K-Means, DBSCAN, Hierarchical
├── 06_model_evaluation/               ← Cross-val, metrics, imbalanced data
├── 07_deep_learning_intro/            ← Why deep learning? Perceptrons, activations
├── 08_pytorch_fundamentals/           ← Tensors, autograd, computational graphs
├── 09_neural_networks/                ← MLP, training loops, overfitting
├── 10_cnns/                           ← Convolutions, pooling, architectures
├── 11_transformers/                   ← Attention, BERT, LLM foundations
│
├── 12_capstone/                       ← Full industrial ML/DL project
│   ├── README.md
│   ├── generate_capstone_data.py
│   └── starter_template.py
│
├── solutions/                         ← Complete worked solutions
│   ├── ml_fundamentals_solutions.py
│   ├── sklearn_solutions.py
│   ├── regression_solutions.py
│   ├── classification_solutions.py
│   ├── clustering_solutions.py
│   ├── model_evaluation_solutions.py
│   ├── pytorch_solutions.py
│   ├── advanced_dl_solutions.py
│   └── capstone_solution.py
│
├── reference/                         ← Quick-reference cheat sheets
│   ├── ml_cheat_sheet.md
│   ├── pytorch_cheat_sheet.md
│   ├── ml_mathematics.md
│   └── algorithm_guide.md
│
├── assessment/                        ← Final assessment
│   ├── FINAL_ASSESSMENT.md
│   └── practical_test.py
│
└── datasets/                          ← Shared datasets
    ├── industrial_sensor_train.csv
    └── industrial_sensor_test.csv
```

---

## How to Use This Package

### Running Order

Work through the modules in order. Each module builds on the previous.

```bash
# Start here — no exceptions
cd 01_ml_fundamentals
python 01_what_is_ml.py

# Progress through modules in sequence
cd ../02_scikit_learn && python 01_sklearn_api.py
# ... and so on through module 12
```

### For Each Module

1. **Read the `README.md`** first — it tells you the real-world motivation
2. **Run the lesson files** — watch the output and read the comments
3. **Do the exercises** — attempt all four tiers before checking solutions
4. **Check solutions** — only in the `solutions/` folder, never alongside exercises

### Running Lesson Files

All lesson files:
- Save plots to `output/` subdirectories (no interactive windows)
- Print explanations to the terminal
- Are self-contained (no hidden dependencies)

```bash
python 01_linear_regression.py
# → Prints explanations to terminal
# → Saves plots to output/
```

### Generating Datasets

The capstone dataset must be generated before running the capstone:

```bash
cd 12_capstone
python generate_capstone_data.py
# → Creates datasets/industrial_sensor_train.csv
# → Creates datasets/industrial_sensor_test.csv
```

---

## The 4-Tier Exercise System

Every module has exercises at four difficulty levels. This is not optional — the tiers are designed to build different skills.

| Tier | Name | What It Tests | Example |
|------|------|--------------|---------|
| **Level 1** | Recall | Can you remember the concept? | "What does the `C` parameter in SVM control?" |
| **Level 2** | Understanding | Can you debug and explain? | "This code has a data leakage bug — find it" |
| **Level 3** | Application | Can you solve a new problem? | "Apply Ridge regression to this housing dataset" |
| **Level 4** | Challenge | Can you design a system? | "Build a full pipeline with cross-validation and explain every choice" |

**Always attempt all four tiers before looking at solutions.** Getting stuck on Level 4 is expected and valuable. The struggle is the learning.

---

## Computational Resources

This entire curriculum runs on a **CPU**. No GPU required.

| Task | Runtime (CPU) |
|------|-------------|
| sklearn models | < 10 seconds |
| PyTorch MLP training | 30–120 seconds |
| CNN training (small dataset) | 2–5 minutes |
| Transformer fine-tuning | 10–30 minutes |

```python
# Standard device setup used throughout this package
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## Academic Integrity Policy

The `solutions/` directory contains complete, commented solutions to all exercises.

**The rule:** Attempt the exercise fully before opening any solution file. The goal is not to pass exercises — it is to build a mental model that will serve you in real engineering work. Solutions are provided because learning from worked examples *after genuine struggle* is one of the most effective study methods in existence. Use them as intended.

---

## Real-World Connections

Every concept connects to active engineering problems:

- **Regression** → Predicting remaining useful life of industrial equipment
- **Classification** → Medical image diagnosis, fault detection, spam filtering
- **Clustering** → Customer segmentation, anomaly detection, gene expression analysis
- **CNNs** → Autonomous vehicle vision, satellite imagery analysis, quality inspection
- **Transformers** → Code completion, scientific paper summarization, drug discovery

---

## Next Steps After This Package

1. **Kaggle Competitions** — Apply your skills on real competition datasets
2. **Research Papers** — Read originals: "Attention Is All You Need", "Deep Residual Learning"
3. **Deployment** — Serve models as REST APIs with FastAPI
4. **MLOps** — MLflow for experiment tracking, Docker for reproducibility
5. **Specialization** — Computer Vision (torchvision), NLP (HuggingFace), Reinforcement Learning

---

*Built for engineers who want to understand, not just use.*
