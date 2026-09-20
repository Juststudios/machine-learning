"""
PCA From Scratch: NumPy Mathematical Implementation
=====================================================
WHY THIS SCRIPT EXISTS:
    While scikit-learn provides `sklearn.decomposition.PCA`, real engineering
    requires understanding the mathematical engine beneath the hood:
    mean-centering, sample covariance matrix computation, eigendecomposition,
    and singular value decomposition (SVD) equivalence.

MATHEMATICAL FOUNDATION:
    1. Mean Centering:
       mu = (1 / N) * sum_{i=1}^N x_i
       X_tilde = X - mu

    2. Sample Covariance Matrix (Sigma):
       Sigma = (1 / (N - 1)) * (X_tilde.T @ X_tilde)
       Shape: (D, D), symmetric positive semi-definite

    3. Eigendecomposition:
       Sigma @ v_i = lambda_i * v_i
       Eigenvalues lambda_i measure variance along principal direction v_i.
       Sorted in descending order: lambda_1 >= lambda_2 >= ... >= lambda_D

    4. Explained Variance Ratio:
       EVR_i = lambda_i / sum_{j=1}^D lambda_j

    5. Projection (Dimensionality Reduction):
       W = [v_1, v_2, ..., v_k]  (D x k matrix of top k eigenvectors)
       Z = X_tilde @ W           (N x k lower-dimensional projection)

    6. Reconstruction (Inverse Transform):
       X_hat = Z @ W.T + mu      (N x D reconstruction in original feature space)

    7. Economy SVD Equivalence:
       X_tilde = U @ S @ V_h
       Sigma = (1 / (N - 1)) * (V_h.T @ S^2 @ V_h)
       ==> lambda_i = S_i^2 / (N - 1)
       ==> Principal axes v_i match rows of V_h (up to sign flip)
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class PCAScratch:
    """
    Principal Component Analysis (PCA) implemented from scratch using NumPy.
    
    Parameters
    ----------
    n_components : int, float, or None
        Number of components to keep.
        - If int >= 1: keep exactly n_components.
        - If float in (0, 1): keep enough components to explain >= n_components of variance.
        - If None: keep all min(N, D) components.
    
    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data.
    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
    singular_values_ : ndarray of shape (n_components,)
        The singular values corresponding to each of the selected components.
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
    covariance_ : ndarray of shape (n_features, n_features)
        Empirical covariance matrix estimated from training set.
    n_components_ : int
        The resolved number of components.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.covariance_ = None
        self.n_components_ = None

    def fit(self, X):
        """
        Fit the PCA model with X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        self : PCAScratch
            The fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("PCA requires at least 2 samples to compute sample covariance.")

        # Step 1: Compute empirical mean and center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: Compute sample covariance matrix Sigma = 1 / (N - 1) * (X_tilde.T @ X_tilde)
        self.covariance_ = (X_centered.T @ X_centered) / (n_samples - 1)

        # Step 3: Eigendecomposition via np.linalg.eigh (for real symmetric matrices)
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_)

        # Step 4: Sort eigenvalues and eigenvectors in descending order
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]

        # Numerical cleanup: clamp tiny negative eigenvalues to 0.0
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Step 5: Explained variance and ratios
        total_variance = np.sum(eigenvalues)
        if total_variance > 0:
            explained_variance_ratio = eigenvalues / total_variance
        else:
            explained_variance_ratio = np.zeros_like(eigenvalues)

        # Step 6: Determine number of components k
        max_possible = min(n_samples - 1, n_features)
        if self.n_components is None:
            k = max_possible
        elif isinstance(self.n_components, (int, np.integer)):
            if self.n_components <= 0 or self.n_components > n_features:
                raise ValueError(f"n_components={self.n_components} must be between 1 and {n_features}")
            k = min(int(self.n_components), max_possible)
        elif isinstance(self.n_components, (float, np.floating)):
            if not (0.0 < self.n_components < 1.0):
                raise ValueError("n_components as float must be in range (0.0, 1.0)")
            cum_var = np.cumsum(explained_variance_ratio)
            k = int(np.searchsorted(cum_var, self.n_components) + 1)
            k = min(k, max_possible)
        else:
            raise TypeError("n_components must be int, float, or None")

        self.n_components_ = k
        self.explained_variance_ = eigenvalues[:k]
        self.explained_variance_ratio_ = explained_variance_ratio[:k]
        self.singular_values_ = np.sqrt(self.explained_variance_ * (n_samples - 1))

        # Eigenvectors shape: (n_features, k). Sklearn convention components_ shape: (k, n_features)
        components = eigenvectors[:, :k].T

        # Scikit-learn deterministic sign convention:
        # Force the component vector with largest absolute value to be positive
        max_abs_cols = np.argmax(np.abs(components), axis=1)
        signs = np.sign(components[np.arange(k), max_abs_cols])
        # Replace zero signs with 1
        signs[signs == 0] = 1.0
        self.components_ = components * signs[:, np.newaxis]

        return self

    def transform(self, X):
        """
        Apply dimensionality reduction on X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        
        Returns
        -------
        Z : ndarray of shape (n_samples, n_components_)
            Projected lower-dimensional coordinates.
        """
        if self.components_ is None:
            raise RuntimeError("PCAScratch must be fitted before calling transform.")
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """Fit the model with X and apply the dimensionality reduction on X."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        """
        Transform data back to its original feature space.
        
        Parameters
        ----------
        Z : array-like of shape (n_samples, n_components_)
            Coordinates in latent principal component space.
            
        Returns
        -------
        X_hat : ndarray of shape (n_samples, n_features)
            Reconstructed data in original space.
        """
        if self.components_ is None:
            raise RuntimeError("PCAScratch must be fitted before calling inverse_transform.")
        Z = np.asarray(Z, dtype=float)
        return Z @ self.components_ + self.mean_


def verify_svd_equivalence(X):
    """
    Demonstrate and verify mathematical equivalence between:
    1. Sample Covariance Eigendecomposition: Sigma = 1/(N-1) * X_tilde.T @ X_tilde
    2. Economy SVD: X_tilde = U @ S @ V_h
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    # Method 1: Covariance + Eigendecomposition
    cov = (X_centered.T @ X_centered) / (N - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx], 0.0)
    eigvecs = eigvecs[:, idx]

    # Method 2: Economy SVD on X_centered
    U, S, V_h = np.linalg.svd(X_centered, full_matrices=False)
    svd_eigenvalues = (S ** 2) / (N - 1)

    print("=" * 65)
    print("ECONOMY SVD EQUIVALENCE VERIFICATION")
    print("=" * 65)
    print(f"Top 5 Eigenvalues from Covariance Matrix:\n  {eigvals[:5]}")
    print(f"Top 5 Eigenvalues from SVD (S^2 / (N-1)):\n  {svd_eigenvalues[:5]}")

    eig_close = np.allclose(eigvals[:len(S)], svd_eigenvalues, atol=1e-8)
    print(f"Eigenvalue equivalence matches within 1e-8: {eig_close}")

    # Check that principal direction vectors span the same subspace (dot product magnitude = 1)
    k_check = min(5, len(S))
    cosines = np.abs(np.sum(eigvecs[:, :k_check] * V_h[:k_check, :].T, axis=0))
    print(f"Direction alignment (|cos(theta)| for top {k_check} components):\n  {cosines}")
    direction_close = np.allclose(cosines, 1.0, atol=1e-7)
    print(f"Principal directions match up to sign flip: {direction_close}")
    return eig_close and direction_close


def generate_sensor_dataset(n_samples=600, random_state=42):
    """Generate realistic 8-channel industrial machine telemetry data."""
    rng = np.random.RandomState(random_state)
    n_per_state = n_samples // 3

    # Normal machine state
    state_0 = rng.normal(
        loc=[70.0, 1.0, 3000.0, 15.0, 5.0, 60.0, 220.0, 20.0],
        scale=[2.0, 0.2, 50.0, 1.0, 0.3, 3.0, 5.0, 2.0],
        size=(n_per_state, 8)
    )

    # Warning state (thermal friction + imbalance)
    state_1 = rng.normal(
        loc=[84.0, 2.8, 2850.0, 19.5, 5.8, 62.0, 218.0, 36.0],
        scale=[3.0, 0.4, 70.0, 1.5, 0.4, 3.5, 6.0, 3.0],
        size=(n_per_state, 8)
    )

    # Fault state (severe bearing damage + overheating)
    state_2 = rng.normal(
        loc=[98.0, 6.2, 2450.0, 28.0, 7.2, 65.0, 212.0, 62.0],
        scale=[4.0, 0.8, 100.0, 2.2, 0.6, 4.0, 7.0, 5.0],
        size=(n_per_state, 8)
    )

    X = np.vstack([state_0, state_1, state_2])
    y = np.array([0] * n_per_state + [1] * n_per_state + [2] * n_per_state)
    feature_names = [
        "Temperature (°C)", "Vibration (mm/s)", "RPM", "Current (A)",
        "Pressure (bar)", "Humidity (%)", "Voltage (V)", "Acoustic (dB)"
    ]
    return X, y, feature_names


if __name__ == "__main__":
    print("=" * 65)
    print("NUMPY PCA FROM SCRATCH — RIGOROUS VERIFICATION")
    print("=" * 65)

    # 1. Generate multi-sensor dataset
    X_raw, y_states, feature_names = generate_sensor_dataset(n_samples=600)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(f"Dataset shape: {X_scaled.shape} (600 samples, 8 features)")

    # 2. Economy SVD equivalence check
    svd_ok = verify_svd_equivalence(X_scaled)
    assert svd_ok, "SVD equivalence check failed!"

    # 3. Fit PCAScratch
    pca_scratch = PCAScratch(n_components=2)
    Z_scratch = pca_scratch.fit_transform(X_scaled)

    # 4. Fit Scikit-Learn PCA for direct comparison
    pca_sklearn = PCA(n_components=2)
    Z_sklearn = pca_sklearn.fit_transform(X_scaled)

    print("\n" + "=" * 65)
    print("COMPARISON: PCAScratch vs sklearn.decomposition.PCA")
    print("=" * 65)
    print(f"Scratch Explained Variance Ratio:\n  {pca_scratch.explained_variance_ratio_}")
    print(f"Sklearn Explained Variance Ratio:\n  {pca_sklearn.explained_variance_ratio_}")
    
    evr_diff = np.max(np.abs(pca_scratch.explained_variance_ratio_ - pca_sklearn.explained_variance_ratio_))
    print(f"Max EVR absolute difference: {evr_diff:.2e}")
    assert evr_diff < 1e-7, "Explained variance ratios do not match!"

    # Components comparison (accounting for deterministic sign convention)
    comp_diff = np.max(np.abs(pca_scratch.components_ - pca_sklearn.components_))
    print(f"Max Components absolute difference: {comp_diff:.2e}")
    assert comp_diff < 1e-7, "Principal component vectors do not match!"

    # Projection coordinates comparison
    proj_diff = np.max(np.abs(Z_scratch - Z_sklearn))
    print(f"Max Projection coordinate difference: {proj_diff:.2e}")
    assert proj_diff < 1e-7, "Projections do not match!"

    # Inverse transform (reconstruction) comparison
    X_hat_scratch = pca_scratch.inverse_transform(Z_scratch)
    X_hat_sklearn = pca_sklearn.inverse_transform(Z_sklearn)
    recon_diff = np.max(np.abs(X_hat_scratch - X_hat_sklearn))
    print(f"Max Reconstruction absolute difference: {recon_diff:.2e}")
    assert recon_diff < 1e-7, "Reconstructions do not match!"

    # 5. Full spectrum PCA (8 components) for scree analysis
    pca_full = PCAScratch(n_components=8)
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    print("\nExplained Variance by Component (1 to 8):")
    for i, (evr, cv) in enumerate(zip(pca_full.explained_variance_ratio_, cum_var), 1):
        print(f"  PC{i}: {evr * 100:6.2f}% | Cumulative: {cv * 100:6.2f}%")

    # 6. Visualization: Scree plot, 2D projection, and reconstruction error
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("PCA From Scratch (NumPy) vs Scikit-Learn Verification", fontsize=13, fontweight='bold')

    # Subplot 1: Scree Plot
    axes[0].bar(range(1, 9), pca_full.explained_variance_ratio_, color='steelblue', alpha=0.8, label='Individual EVR')
    axes[0].plot(range(1, 9), cum_var, 'ro-', linewidth=2, label='Cumulative EVR')
    axes[0].axhline(0.95, color='green', linestyle='--', label='95% Threshold')
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("Scree Plot & Cumulative Variance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: 2D Projection using PCAScratch
    colors = ['#2b5c8f', '#d95f02', '#7570b3']
    labels = ['Normal (OK)', 'Warning', 'Fault']
    for state_idx in [0, 1, 2]:
        mask = (y_states == state_idx)
        axes[1].scatter(
            Z_scratch[mask, 0], Z_scratch[mask, 1],
            c=colors[state_idx], label=labels[state_idx],
            alpha=0.7, s=25
        )
    axes[1].set_xlabel(f"PC1 ({pca_scratch.explained_variance_ratio_[0] * 100:.1f}% var)")
    axes[1].set_ylabel(f"PC2 ({pca_scratch.explained_variance_ratio_[1] * 100:.1f}% var)")
    axes[1].set_title("2D Projection via PCAScratch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: Reconstruction Error vs Number of Components
    recon_errors = []
    for k in range(1, 9):
        p_k = PCAScratch(n_components=k).fit(X_scaled)
        rec = p_k.inverse_transform(p_k.transform(X_scaled))
        mse = np.mean((X_scaled - rec) ** 2)
        recon_errors.append(mse)
    
    axes[2].plot(range(1, 9), recon_errors, 's-', color='crimson', linewidth=2)
    axes[2].set_xlabel("Number of Components (k)")
    axes[2].set_ylabel("Reconstruction Mean Squared Error")
    axes[2].set_title("Reconstruction MSE vs Latent Dimensions")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "pca_from_scratch.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"\nVerification plot saved to: {plot_path}")
    print("=" * 65)
    print("ALL PCA FROM SCRATCH TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)
