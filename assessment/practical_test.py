#!/usr/bin/env python3
"""
Practical Assessment: Machine Learning & Deep Learning
=======================================================
This script validates whether the ML package is complete and runnable.
Run this to check all modules before the final assessment.

Usage: python assessment/practical_test.py
"""

import subprocess
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
PYTHON  = sys.executable

TESTS = [
    # (description, file_path)
    ("Module 1: What is ML?",       "01_ml_fundamentals/01_what_is_ml.py"),
    ("Module 1: ML Workflow",       "01_ml_fundamentals/02_ml_workflow.py"),
    ("Module 2: Sklearn API",       "02_scikit_learn/01_sklearn_api.py"),
    ("Module 2: Preprocessing",     "02_scikit_learn/02_preprocessing.py"),
    ("Module 2: Pipelines",         "02_scikit_learn/03_pipelines.py"),
    ("Module 3: Linear Regression", "03_regression/01_linear_regression.py"),
    ("Module 3: Polynomial Reg",    "03_regression/02_polynomial_regression.py"),
    ("Module 3: Regularization",    "03_regression/03_regularization.py"),
    ("Module 5: K-Means",           "05_clustering/01_kmeans.py"),
    ("Module 5: Hierarchical",      "05_clustering/02_hierarchical_clustering.py"),
    ("Module 5: DBSCAN",            "05_clustering/03_dbscan.py"),
    ("Module 5: PCA",               "05_clustering/04_dimensionality_reduction.py"),
    ("Module 6: Metrics",           "06_model_evaluation/01_metrics.py"),
    ("Module 6: Cross-Validation",  "06_model_evaluation/02_cross_validation.py"),
    ("Module 6: Bias-Variance",     "06_model_evaluation/04_bias_variance_tradeoff.py"),
    ("Module 7: Tensors",           "07_deep_learning_intro/01_tensors.py"),
    ("Module 7: Autograd",          "07_deep_learning_intro/02_autograd.py"),
    ("Module 8: MLP Classification","08_pytorch_fundamentals/05_mlp_classification.py"),
    ("Module 9: Activations",       "09_neural_networks/01_activation_functions.py"),
    ("Module 9: Backprop & MLP",    "09_neural_networks/02_backpropagation_and_deep_mlp.py"),
    ("Module 10: Convolution",      "10_cnns/01_convolution.py"),
    ("Module 10: CNN for Images",   "10_cnns/03_cnn_for_images.py"),
    ("Module 11: Transformers",     "11_transformers/01_attention_and_transformers.py"),
]

def run_test(desc, rel_path, timeout=120):
    """Run a single lesson file and return (passed, message)."""
    file_path = ML_DIR / rel_path
    if not file_path.exists():
        return False, "FILE NOT FOUND"
    
    try:
        result = subprocess.run(
            [PYTHON, str(file_path)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(file_path.parent)
        )
        if result.returncode == 0:
            return True, "PASS"
        else:
            err = result.stderr[-200:] if result.stderr else "no output"
            return False, f"FAIL: {err.strip()}"
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (> 120s)"
    except Exception as e:
        return False, f"ERROR: {e}"

def main():
    print("=" * 65)
    print("MACHINE LEARNING PACKAGE — PRACTICAL VALIDATION")
    print("=" * 65)
    print(f"Testing {len(TESTS)} lesson files...")
    print()
    
    passed = 0
    failed = []
    
    for desc, rel_path in TESTS:
        status, msg = run_test(desc, rel_path)
        symbol = "✅" if status else "❌"
        print(f"  {symbol}  {desc:40s}  {msg}")
        if status:
            passed += 1
        else:
            failed.append((desc, rel_path, msg))
    
    print()
    print("=" * 65)
    print(f"Results: {passed}/{len(TESTS)} tests passed")
    
    if failed:
        print(f"\nFailed tests:")
        for desc, path, msg in failed:
            print(f"  ❌ {desc}")
            print(f"     Path: {path}")
            print(f"     Reason: {msg}")
    else:
        print("\n🎉 All tests passed! Ready for the final assessment.")
    
    print()
    
    # Check file completeness
    print("File completeness check:")
    modules = {
        "01_ml_fundamentals": ["01_what_is_ml.py", "02_ml_workflow.py", "03_data_and_features.py",
                                "04_train_test_split.py", "exercises.py", "README.md"],
        "02_scikit_learn":    ["01_sklearn_api.py", "02_preprocessing.py", "03_pipelines.py",
                                "exercises.py", "README.md"],
        "03_regression":      ["01_linear_regression.py", "02_polynomial_regression.py",
                                "03_regularization.py", "exercises.py", "README.md"],
        "04_classification":  ["01_logistic_regression.py", "02_decision_trees.py",
                                "03_random_forests.py", "04_svm.py", "05_knn.py",
                                "exercises.py", "README.md"],
        "05_clustering":      ["01_kmeans.py", "02_hierarchical_clustering.py", "03_dbscan.py",
                                "04_dimensionality_reduction.py", "exercises.py", "README.md"],
        "06_model_evaluation":["01_metrics.py", "02_cross_validation.py", "03_hyperparameter_tuning.py",
                                "04_bias_variance_tradeoff.py", "05_imbalanced_data.py",
                                "exercises.py", "README.md"],
        "07_deep_learning_intro": ["01_tensors.py", "02_autograd.py", "03_linear_model_in_pytorch.py",
                                    "04_datasets_and_dataloaders.py", "exercises.py", "README.md"],
        "08_pytorch_fundamentals": ["01_nn_module.py", "02_loss_functions.py", "03_optimizers.py",
                                     "04_training_loop.py", "05_mlp_classification.py",
                                     "exercises.py", "README.md"],
        "09_neural_networks": ["01_activation_functions.py", "02_backpropagation_and_deep_mlp.py",
                                "exercises.py", "README.md"],
        "10_cnns":            ["01_convolution.py", "02_pooling_and_architecture.py",
                                "03_cnn_for_images.py", "04_transfer_learning.py",
                                "exercises.py", "README.md"],
        "11_transformers":    ["01_attention_and_transformers.py", "exercises.py", "README.md"],
        "12_capstone":        ["generate_capstone_data.py", "starter_template.py", "README.md"],
    }
    
    total_expected = sum(len(v) for v in modules.values())
    total_present  = 0
    missing = []
    
    for module, files in modules.items():
        for f in files:
            if (ML_DIR / module / f).exists():
                total_present += 1
            else:
                missing.append(f"{module}/{f}")
    
    print(f"  Files: {total_present}/{total_expected} present")
    if missing:
        print("  Missing files:")
        for m in missing:
            print(f"    - {m}")
    else:
        print("  ✅ All expected files present!")

if __name__ == "__main__":
    main()
