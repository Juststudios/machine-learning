"""
Logistic Regression Gradient Descent From Scratch
===================================================
WHY THIS SCRIPT EXISTS:
    While scikit-learn's `LogisticRegression` solves classification problems
    using C-level optimizers (L-BFGS, Liblinear), this script implements
    Logistic Regression and One-vs-Rest multi-class classification from
    mathematical first principles using NumPy.

MATHEMATICAL FOUNDATION:
    1. Linear Combination (Logits):
       z = X @ w + b

    2. Numerically Stable Sigmoid Activation:
       sigma(z) = 1 / (1 + exp(-z))
       For numerical stability across extreme values:
       sigma(z) = { 1 / (1 + exp(-z))   if z >= 0
                  { exp(z) / (1 + exp(z)) if z < 0

    3. Binary Cross-Entropy (BCE) Loss with L2 Regularization:
       J(w, b) = - (1/m) * sum_{i=1}^m [ y_i * ln(y_hat_i + eps) +
                                        (1 - y_i) * ln(1 - y_hat_i + eps) ]
                 + (lambda / (2 * m)) * ||w||^2

    4. Analytical Gradients:
       grad_w = (1 / m) * X.T @ (y_hat - y) + (lambda / m) * w
       grad_b = (1 / m) * sum_{i=1}^m (y_hat_i - y_i)

    5. Gradient Descent Updates:
       w <- w - alpha * grad_w
       b <- b - alpha * grad_b

    6. One-vs-Rest (OvR) Multiclass:
       For K classes, train K independent binary classifiers:
       Classifier k predicts P(y = k | X).
       Predicted class: y* = argmax_k P(y = k | X).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def stable_sigmoid(z):
    """
    Numerically stable sigmoid function avoiding overflow/underflow.
    
    Parameters
    ----------
    z : array-like
        Input values or logits.
        
    Returns
    -------
    ndarray
        Sigmoid probabilities in [0.0, 1.0].
    """
    z = np.asarray(z, dtype=float)
    # Clip z to prevent exp overflow in extreme regimes
    z_clipped = np.clip(z, -500.0, 500.0)
    # Piecewise branch: for z >= 0 use 1 / (1 + exp(-z)), for z < 0 use exp(z) / (1 + exp(z))
    positive_mask = (z_clipped >= 0)
    result = np.empty_like(z_clipped)
    result[positive_mask] = 1.0 / (1.0 + np.exp(-z_clipped[positive_mask]))
    exp_neg = np.exp(z_clipped[~positive_mask])
    result[~positive_mask] = exp_neg / (1.0 + exp_neg)
    return result


class LogisticRegressionGD:
    """
    Binary Logistic Regression classifier using Gradient Descent from scratch.
    
    Parameters
    ----------
    learning_rate : float, default=0.05
        Gradient descent step size (alpha).
    max_iter : int, default=2000
        Maximum number of gradient descent iterations.
    l2_reg : float, default=0.0
        L2 regularization strength (lambda).
    tol : float, default=1e-7
        Convergence tolerance on loss change.
    random_state : int or None, default=42
        Seed for weight initialization reproducibility.
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Learned feature weights (w).
    intercept_ : float
        Learned bias term (b).
    losses_ : list of float
        Binary cross-entropy loss recorded per iteration.
    n_iter_ : int
        Number of iterations executed until convergence.
    """

    def __init__(self, learning_rate=0.05, max_iter=2000, l2_reg=0.0, tol=1e-7, random_state=42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2_reg = l2_reg
        self.tol = tol
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = 0.0
        self.losses_ = []
        self.n_iter_ = 0

    def fit(self, X, y):
        """
        Fit the binary logistic regression model via Gradient Descent.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Binary target labels {0, 1}.
            
        Returns
        -------
        self : LogisticRegressionGD
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X.shape}")
        if len(y) != len(X):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

        m, d = X.shape
        rng = np.random.RandomState(self.random_state)
        # Small random initialization (Xavier/He style for single unit)
        self.coef_ = rng.normal(0.0, 0.01, size=d)
        self.intercept_ = 0.0
        self.losses_ = []
        eps = 1e-15

        for i in range(self.max_iter):
            # 1. Forward pass: logits & predictions
            logits = X @ self.coef_ + self.intercept_
            y_hat = stable_sigmoid(logits)

            # 2. Binary Cross-Entropy Loss computation
            # Clamping predictions to (eps, 1 - eps) prevents log(0)
            y_hat_safe = np.clip(y_hat, eps, 1.0 - eps)
            bce_loss = - (1.0 / m) * np.sum(
                y * np.log(y_hat_safe) + (1.0 - y) * np.log(1.0 - y_hat_safe)
            )
            reg_penalty = (self.l2_reg / (2.0 * m)) * np.sum(self.coef_ ** 2)
            total_loss = bce_loss + reg_penalty
            self.losses_.append(total_loss)

            # Check early stopping
            if i > 0 and abs(self.losses_[-2] - self.losses_[-1]) < self.tol:
                self.n_iter_ = i + 1
                break

            # 3. Analytical gradients
            error = y_hat - y  # shape (m,)
            grad_w = (1.0 / m) * (X.T @ error) + (self.l2_reg / m) * self.coef_
            grad_b = (1.0 / m) * np.sum(error)

            # 4. Gradient descent updates
            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b
        else:
            self.n_iter_ = self.max_iter

        return self

    def predict_proba(self, X):
        """
        Compute class probabilities for test samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        
        Returns
        -------
        ndarray of shape (n_samples, 2)
            Probabilities [P(y=0), P(y=1)].
        """
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        X = np.asarray(X, dtype=float)
        logits = X @ self.coef_ + self.intercept_
        p1 = stable_sigmoid(logits)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X, threshold=0.5):
        """
        Predict binary classes for test samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        threshold : float, default=0.5
            Decision threshold.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Binary predictions in {0, 1}.
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


class LogisticRegressionOVR:
    """
    One-vs-Rest (OvR) Multi-class Logistic Regression using LogisticRegressionGD.
    
    Parameters
    ----------
    learning_rate : float, default=0.05
    max_iter : int, default=2000
    l2_reg : float, default=0.0
    tol : float, default=1e-7
    random_state : int, default=42
    """

    def __init__(self, learning_rate=0.05, max_iter=2000, l2_reg=0.0, tol=1e-7, random_state=42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2_reg = l2_reg
        self.tol = tol
        self.random_state = random_state
        self.classes_ = None
        self.classifiers_ = {}
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Train K binary classifiers, one for each unique class in y.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.classifiers_ = {}
        coefs = []
        intercepts = []

        for idx, cls in enumerate(self.classes_):
            # Target is 1 if sample == cls, else 0
            binary_y = (y == cls).astype(int)
            clf = LogisticRegressionGD(
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                l2_reg=self.l2_reg,
                tol=self.tol,
                random_state=self.random_state + idx
            )
            clf.fit(X, binary_y)
            self.classifiers_[cls] = clf
            coefs.append(clf.coef_)
            intercepts.append(clf.intercept_)

        self.coef_ = np.array(coefs)
        self.intercept_ = np.array(intercepts)
        return self

    def predict_proba(self, X):
        """
        Predict normalized multi-class probabilities.
        """
        X = np.asarray(X, dtype=float)
        # Collect raw probabilities for class == 1 from each binary model
        raw_probs = np.column_stack([
            self.classifiers_[cls].predict_proba(X)[:, 1]
            for cls in self.classes_
        ])
        # Normalize probabilities across classes so rows sum to 1.0
        row_sums = raw_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return raw_probs / row_sums

    def predict(self, X):
        """
        Predict class with the highest probability.
        """
        proba = self.predict_proba(X)
        best_indices = np.argmax(proba, axis=1)
        return self.classes_[best_indices]


def generate_binary_fault_data(n_samples=500, random_state=42):
    """
    Generate synthetic sensor data for industrial machine fault detection:
    Features: Temperature (°C), Vibration (mm/s)
    Classes: 0 = OK, 1 = FAULT
    """
    rng = np.random.RandomState(random_state)
    n_ok = int(n_samples * 0.6)
    n_fault = n_samples - n_ok

    # Normal machine operation
    temp_ok = rng.normal(68.0, 3.5, n_ok)
    vib_ok = rng.normal(1.8, 0.4, n_ok)

    # Machine fault (bearing degradation + thermal overheating)
    temp_fault = rng.normal(86.0, 4.5, n_fault)
    vib_fault = rng.normal(4.8, 0.8, n_fault)

    X = np.vstack([
        np.column_stack([temp_ok, vib_ok]),
        np.column_stack([temp_fault, vib_fault])
    ])
    y = np.array([0] * n_ok + [1] * n_fault)
    
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def generate_multiclass_fault_data(n_samples=600, random_state=7):
    """Generate 3-class machine state data: OK (0), WARNING (1), FAULT (2)."""
    rng = np.random.RandomState(random_state)
    n_ok, n_warn, n_fault = 300, 180, 120

    X_ok = np.column_stack([rng.normal(66.0, 3.0, n_ok), rng.normal(1.8, 0.4, n_ok)])
    X_warn = np.column_stack([rng.normal(78.0, 3.5, n_warn), rng.normal(3.8, 0.6, n_warn)])
    X_fault = np.column_stack([rng.normal(91.0, 4.0, n_fault), rng.normal(6.5, 0.8, n_fault)])

    X = np.vstack([X_ok, X_warn, X_fault])
    y = np.array([0] * n_ok + [1] * n_warn + [2] * n_fault)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


if __name__ == "__main__":
    print("=" * 65)
    print("LOGISTIC REGRESSION GD FROM SCRATCH — VERIFICATION")
    print("=" * 65)

    # 1. Binary Classification Experiment
    X, y = generate_binary_fault_data(n_samples=500, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Train manual LogisticRegressionGD
    clf_scratch = LogisticRegressionGD(learning_rate=0.1, max_iter=3000, l2_reg=0.01)
    clf_scratch.fit(X_train_sc, y_train)

    # Train Scikit-Learn LogisticRegression (with matching L2 penalty)
    # In sklearn, C = 1 / lambda, so lambda = 0.01 means C = 100
    clf_sklearn = LogisticRegression(C=100.0, max_iter=3000, random_state=42)
    clf_sklearn.fit(X_train_sc, y_train)

    y_pred_scratch = clf_scratch.predict(X_test_sc)
    y_pred_sklearn = clf_sklearn.predict(X_test_sc)
    acc_scratch = accuracy_score(y_test, y_pred_scratch)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

    print("\n--- Binary Classification Benchmark ---")
    print(f"Scratch GD Accuracy:  {acc_scratch:.4f}")
    print(f"Sklearn Accuracy:     {acc_sklearn:.4f}")
    print(f"Scratch Iterations:   {clf_scratch.n_iter_}")
    print(f"Scratch Final Loss:   {clf_scratch.losses_[-1]:.6f}")
    print(f"Scratch Weights:      {clf_scratch.coef_}, Intercept: {clf_scratch.intercept_:.4f}")
    print(f"Sklearn Weights:      {clf_sklearn.coef_[0]}, Intercept: {clf_sklearn.intercept_[0]:.4f}")

    # Verify accuracy is high (>= 95%) and closely matches sklearn
    assert acc_scratch >= 0.95, f"Scratch accuracy too low: {acc_scratch}"
    assert abs(acc_scratch - acc_sklearn) <= 0.02, "Scratch accuracy diverges from sklearn"

    # Verify probability calibration
    p_scratch = clf_scratch.predict_proba(X_test_sc)
    p_sklearn = clf_sklearn.predict_proba(X_test_sc)
    prob_diff = np.mean(np.abs(p_scratch - p_sklearn))
    print(f"Mean absolute probability difference vs sklearn: {prob_diff:.4f}")
    assert prob_diff < 0.05, f"Probability calibration mismatch: {prob_diff}"

    # 2. Multi-Class One-vs-Rest Experiment
    print("\n--- Multi-Class (3 States: OK, WARNING, FAULT) ---")
    X_mc, y_mc = generate_multiclass_fault_data(n_samples=600, random_state=7)
    X_mc_tr, X_mc_te, y_mc_tr, y_mc_te = train_test_split(
        X_mc, y_mc, test_size=0.25, random_state=42, stratify=y_mc
    )
    sc_mc = StandardScaler()
    X_mc_tr_sc = sc_mc.fit_transform(X_mc_tr)
    X_mc_te_sc = sc_mc.transform(X_mc_te)

    clf_ovr = LogisticRegressionOVR(learning_rate=0.1, max_iter=3000, l2_reg=0.01)
    clf_ovr.fit(X_mc_tr_sc, y_mc_tr)
    y_mc_pred = clf_ovr.predict(X_mc_te_sc)
    acc_ovr = accuracy_score(y_mc_te, y_mc_pred)
    print(f"OvR Multi-class Accuracy: {acc_ovr:.4f}")
    assert acc_ovr >= 0.95, f"OvR accuracy too low: {acc_ovr}"

    print("\nOvR Classification Report:")
    print(classification_report(y_mc_te, y_mc_pred, target_names=['OK', 'WARNING', 'FAULT']))

    # 3. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Logistic Regression GD From Scratch", fontsize=13, fontweight='bold')

    # Subplot 1: Loss History
    axes[0].plot(clf_scratch.losses_, color='crimson', linewidth=2)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].set_title(f"Gradient Descent Loss Convergence (Iter: {clf_scratch.n_iter_})")
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Binary Decision Boundary Comparison
    # Create meshgrid
    x_min, x_max = X_test_sc[:, 0].min() - 0.5, X_test_sc[:, 0].max() + 0.5
    y_min, y_max = X_test_sc[:, 1].min() - 0.5, X_test_sc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Scratch boundary
    Z_scratch_prob = clf_scratch.predict_proba(grid)[:, 1].reshape(xx.shape)
    axes[1].contourf(xx, yy, Z_scratch_prob, levels=20, cmap='RdBu_r', alpha=0.4)
    axes[1].contour(xx, yy, Z_scratch_prob, levels=[0.5], colors='black', linewidths=2)
    
    # Scatter test points
    scatter0 = axes[1].scatter(
        X_test_sc[y_test == 0, 0], X_test_sc[y_test == 0, 1],
        c='blue', edgecolors='k', label='OK (0)', s=30
    )
    scatter1 = axes[1].scatter(
        X_test_sc[y_test == 1, 0], X_test_sc[y_test == 1, 1],
        c='red', edgecolors='k', label='FAULT (1)', s=30
    )
    axes[1].set_xlabel("Temperature (standardized)")
    axes[1].set_ylabel("Vibration (standardized)")
    axes[1].set_title("Decision Boundary (Scratch GD)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: Multi-class Decision Regions
    x_min_mc, x_max_mc = X_mc_te_sc[:, 0].min() - 0.5, X_mc_te_sc[:, 0].max() + 0.5
    y_min_mc, y_max_mc = X_mc_te_sc[:, 1].min() - 0.5, X_mc_te_sc[:, 1].max() + 0.5
    xx_mc, yy_mc = np.meshgrid(np.linspace(x_min_mc, x_max_mc, 200), np.linspace(y_min_mc, y_max_mc, 200))
    grid_mc = np.c_[xx_mc.ravel(), yy_mc.ravel()]
    Z_mc = clf_ovr.predict(grid_mc).reshape(xx_mc.shape)

    axes[2].contourf(xx_mc, yy_mc, Z_mc, levels=[-0.5, 0.5, 1.5, 2.5], cmap='viridis', alpha=0.3)
    colors_mc = ['navy', 'darkorange', 'darkred']
    names_mc = ['OK', 'WARNING', 'FAULT']
    for c_idx in [0, 1, 2]:
        mask = (y_mc_te == c_idx)
        axes[2].scatter(
            X_mc_te_sc[mask, 0], X_mc_te_sc[mask, 1],
            c=colors_mc[c_idx], label=names_mc[c_idx], edgecolors='k', s=30
        )
    axes[2].set_xlabel("Temperature (standardized)")
    axes[2].set_ylabel("Vibration (standardized)")
    axes[2].set_title("OvR Multi-class Regions")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "logistic_regression_gd.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"\nVisualization saved to: {plot_path}")
    print("=" * 65)
    print("LOGISTIC REGRESSION GD TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)
