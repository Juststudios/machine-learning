"""
Deep MLP Project: Industrial Machine Fault Classification
==========================================================
End-to-End Deep Learning Engineering Project:
Multi-class Fault Severity Predictor on 8-Channel Telemetry.

Modern manufacturing and aerospace operations deploy extensive sensor networks
to continuously monitor high-value turbomachinery, pumps, and gearboxes.
Predicting early degradation before catastrophic failure saves millions in downtime.

This project synthesizes all core deep learning techniques:
  1. Synthetic 8-Channel Industrial Telemetry Generator:
     - Vibration RMS (mm/s), Vibration Peak (g), Bearing Temp (°C), Motor RPM,
       Phase Current (A), Acoustic Emission (dB), Hydraulic Pressure (bar),
       Exhaust Temp (°C).
     - 3 Fault Severity Classes: Normal (OK), Warning (Degraded), Critical Fault.
  2. Complete Deep MLP Architecture (`DeepFaultClassifier`):
     - Kaiming Normal (He) weight initialization for ReLU networks.
     - Deep hidden representations: 8 -> 256 -> 128 -> 64 -> 3.
     - Batch Normalization (`nn.BatchNorm1d`) for training stability and accelerated convergence.
     - Dropout (`p=0.25 - 0.30`) to prevent co-adaptation and overfitting.
  3. Production Training Loop:
     - Adam optimizer with weight decay (L2 regularization).
     - Learning rate scheduler (`ReduceLROnPlateau`).
     - Validation-based Early Stopping with deep-copy model checkpointing.
  4. Rigorous Evaluation & Visualizations:
     - Test classification report (Precision, Recall, F1-Score).
     - Normalized confusion matrix heatmap (`output/confusion_matrix.png`).
     - Epoch-wise training & validation curves (`output/training_curves.png`).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SEED = 42

FEATURE_NAMES = [
    "Vibration_RMS",      # mm/s
    "Vibration_Peak",     # g
    "Bearing_Temp",       # °C
    "Motor_RPM",          # RPM
    "Phase_Current",      # A
    "Acoustic_Emission",  # dB
    "Hydraulic_Pressure", # bar
    "Exhaust_Temp"        # °C
]
CLASS_NAMES = ["Normal (OK)", "Warning (Degraded)", "Critical Fault"]

def generate_telemetry(n_per_class=500, seed=42):
    """Generate synthetic 8-channel telemetry across 3 health classes."""
    rng = np.random.RandomState(seed)
    
    # Class 0: Normal / Healthy operation
    c0 = np.column_stack([
        rng.normal(1.2, 0.25, n_per_class),     # Vibration_RMS
        rng.normal(2.4, 0.40, n_per_class),     # Vibration_Peak
        rng.normal(64.0, 3.50, n_per_class),    # Bearing_Temp
        rng.normal(3000.0, 35.0, n_per_class),  # Motor_RPM
        rng.normal(14.8, 0.80, n_per_class),    # Phase_Current
        rng.normal(52.0, 3.00, n_per_class),    # Acoustic_Emission
        rng.normal(212.0, 4.50, n_per_class),   # Hydraulic_Pressure
        rng.normal(44.0, 2.50, n_per_class)     # Exhaust_Temp
    ])
    
    # Class 1: Warning / Mild degradation (elevated vibration & temperature)
    c1 = np.column_stack([
        rng.normal(2.9, 0.45, n_per_class),     # Vibration_RMS elevated
        rng.normal(5.2, 0.80, n_per_class),     # Vibration_Peak elevated
        rng.normal(79.0, 5.00, n_per_class),    # Bearing_Temp higher
        rng.normal(2940.0, 55.0, n_per_class),  # Motor_RPM slightly dragged
        rng.normal(18.2, 1.30, n_per_class),    # Phase_Current increased
        rng.normal(69.0, 4.00, n_per_class),    # Acoustic_Emission higher
        rng.normal(194.0, 7.50, n_per_class),   # Hydraulic_Pressure drop
        rng.normal(59.0, 4.00, n_per_class)     # Exhaust_Temp higher
    ])
    
    # Class 2: Critical Fault (severe anomalies, cavitation, thermal runaway)
    c2 = np.column_stack([
        rng.normal(6.5, 1.10, n_per_class),     # Vibration_RMS severe
        rng.normal(12.5, 2.20, n_per_class),    # Vibration_Peak critical
        rng.normal(99.0, 7.00, n_per_class),    # Bearing_Temp overheating
        rng.normal(2740.0, 110.0, n_per_class), # Motor_RPM heavy load
        rng.normal(26.5, 2.80, n_per_class),    # Phase_Current spike
        rng.normal(88.0, 5.50, n_per_class),    # Acoustic_Emission alarming
        rng.normal(158.0, 12.0, n_per_class),   # Hydraulic_Pressure major loss
        rng.normal(83.0, 6.00, n_per_class)     # Exhaust_Temp critical
    ])
    
    X = np.vstack([c0, c1, c2]).astype(np.float32)
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class, dtype=np.int64)
    return X, y

def init_weights_kaiming(m):
    """
    Kaiming (He) Normal Initialization:
    Std = sqrt(2 / fan_in) tailored for ReLU activation.
    BatchNorm weights initialized to gamma=1, beta=0.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

class DeepFaultClassifier(nn.Module):
    """
    Deep Industrial Telemetry Classifier
    Architecture:
      Input (8) -> Linear(256) -> BatchNorm1d -> ReLU -> Dropout(0.25)
                -> Linear(128) -> BatchNorm1d -> ReLU -> Dropout(0.25)
                -> Linear(64)  -> BatchNorm1d -> ReLU -> Dropout(0.20)
                -> Linear(3) Logits
    """
    def __init__(self, in_features=8, num_classes=3, dropout_rate=0.25):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(in_features, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 3
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            
            # Output Layer
            nn.Linear(64, num_classes)
        )
        self.apply(init_weights_kaiming)

    def forward(self, x):
        return self.net(x)

def run_project():
    """Run full industrial telemetry classification pipeline."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    print("=" * 75)
    print("INDUSTRIAL MACHINE FAULT DETECTION — DEEP MLP CLASSIFIER")
    print("=" * 75)

    # STEP 1: Telemetry Generation & Preprocessing
    print("\n--- STEP 1: Generating 8-Channel Industrial Telemetry ---")
    X_raw, y_raw = generate_telemetry(n_per_class=500, seed=SEED)
    print(f"Total Telemetry Samples: {len(X_raw)} (Features: {len(FEATURE_NAMES)}, Classes: {len(CLASS_NAMES)})")

    # Stratified 70% Train, 15% Validation, 15% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y_raw, test_size=0.15, random_state=SEED, stratify=y_raw
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(0.15 / 0.85), random_state=SEED, stratify=y_train_val
    )
    print(f"Dataset Partitions: Train = {len(X_train)} (70%), Val = {len(X_val)} (15%), Test = {len(X_test)} (15%)")

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # PyTorch DataLoaders
    BATCH_SIZE = 32
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset   = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset  = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # STEP 2: Architecture & Weight Init
    print("\n--- STEP 2: Architecture & Weight Initialization ---")
    model = DeepFaultClassifier(in_features=8, num_classes=3, dropout_rate=0.25)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DeepFaultClassifier instantiated with {total_params:,} trainable parameters.")
    print(model)

    # STEP 3: Training Loop with Early Stopping & LR Scheduling
    print("\n--- STEP 3: Training Loop with Validation & Early Stopping ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    MAX_EPOCHS = 100
    PATIENCE = 15

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    best_model_weights = None
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss':   [],
        'train_acc':  [],
        'val_acc':    [],
        'lr':         []
    }

    for epoch in range(1, MAX_EPOCHS + 1):
        # Training pass
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += len(y_batch)
            
        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation pass
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                running_val_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                correct_val += (preds == y_batch).sum().item()
                total_val += len(y_batch)
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)
        
        # Scheduler step
        scheduler.step(epoch_val_loss)
        
        # Check early stopping
        if epoch_val_loss < best_val_loss - 1e-4:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | Train Loss: {epoch_train_loss:.4f} (Acc: {epoch_train_acc*100:.1f}%) | "
                  f"Val Loss: {epoch_val_loss:.4f} (Acc: {epoch_val_acc*100:.1f}%) | lr: {current_lr:.6f}")
            
        if patience_counter >= PATIENCE:
            print(f"\n>> Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}.")
            break

    # Restore best checkpoint
    model.load_state_dict(best_model_weights)
    print(f"Restored best model state from epoch {best_epoch} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc*100:.2f}%)")

    # STEP 4: Test Evaluation
    print("\n--- STEP 4: Test Evaluation on Unseen Telemetry ---")
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            test_loss += loss.item() * len(y_batch)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    test_loss /= len(test_dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_acc = (all_preds == all_targets).mean()

    print(f"Final Test Loss:     {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc*100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix (Counts):")
    print(cm)

    # STEP 5: Visualizations
    print("\n--- STEP 5: Generating Engineering Visualizations ---")
    fig1, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_trained = len(history['train_loss'])
    epochs_x = range(1, epochs_trained + 1)

    # Loss curves
    ax_l.plot(epochs_x, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax_l.plot(epochs_x, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    ax_l.axvline(best_epoch, color='green', linestyle=':', label=f'Best Checkpoint (Epoch {best_epoch})', linewidth=2)
    ax_l.set_title("Cross-Entropy Loss vs Epoch", fontsize=13, fontweight='bold')
    ax_l.set_xlabel("Epoch", fontsize=11)
    ax_l.set_ylabel("Loss", fontsize=11)
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(fontsize=10)

    # Accuracy curves
    ax_a.plot(epochs_x, [a * 100 for a in history['train_acc']], 'b-', label='Train Accuracy', linewidth=2)
    ax_a.plot(epochs_x, [a * 100 for a in history['val_acc']], 'r--', label='Validation Accuracy', linewidth=2)
    ax_a.axvline(best_epoch, color='green', linestyle=':', label=f'Best Checkpoint (Epoch {best_epoch})', linewidth=2)
    ax_a.set_title("Classification Accuracy vs Epoch", fontsize=13, fontweight='bold')
    ax_a.set_xlabel("Epoch", fontsize=11)
    ax_a.set_ylabel("Accuracy (%)", fontsize=11)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=10)

    plt.tight_layout()
    curves_path = OUTPUT_DIR / "training_curves.png"
    plt.savefig(curves_path, dpi=130)
    plt.close(fig1)
    print(f"✅ Saved training curves to: {curves_path}")

    # Confusion Matrix
    fig2, ax_cm = plt.subplots(figsize=(7, 6))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax_cm.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    cbar = fig2.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Proportion", fontsize=10)

    ax_cm.set_xticks(range(len(CLASS_NAMES)))
    ax_cm.set_yticks(range(len(CLASS_NAMES)))
    ax_cm.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=10)
    ax_cm.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax_cm.set_xlabel("Predicted Condition", fontsize=11, fontweight='bold')
    ax_cm.set_ylabel("True Condition", fontsize=11, fontweight='bold')
    ax_cm.set_title("Confusion Matrix — Industrial Fault Classifier", fontsize=12, fontweight='bold', pad=12)

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_color = "white" if cm_norm[i, j] > thresh else "black"
            ax_cm.text(j, i, f"{count}\n({pct:.1f}%)",
                       ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')

    plt.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=130)
    plt.close(fig2)
    print(f"✅ Saved confusion matrix to: {cm_path}")

    # STEP 6: Single Sample Diagnostic
    print("\n--- STEP 6: Single-Sample Inference Telemetry Diagnostic ---")
    sample_idx = 0
    raw_telemetry_sample = X_test[sample_idx]
    true_label = CLASS_NAMES[y_test[sample_idx]]

    scaled_sample = torch.tensor(scaler.transform(raw_telemetry_sample.reshape(1, -1)), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        sample_logits = model(scaled_sample)
        sample_probs = torch.softmax(sample_logits, dim=1).numpy()[0]
        pred_idx = np.argmax(sample_probs)
        pred_label = CLASS_NAMES[pred_idx]

    print(f"Diagnostic Test Telemetry:")
    for fname, val in zip(FEATURE_NAMES, raw_telemetry_sample):
        print(f"  {fname:20s}: {val:.2f}")
    print(f"\nModel Assessment:")
    for cname, prob in zip(CLASS_NAMES, sample_probs):
        bar = "█" * int(prob * 30)
        print(f"  {cname:20s}: {prob*100:5.1f}% | {bar}")
    print(f"\nVerdict: Predicted [{pred_label}] | Ground Truth: [{true_label}]")
    assert pred_idx == y_test[sample_idx], "Verification prediction failed!"
    print("=> Diagnostic pipeline verified successfully!\n")

if __name__ == "__main__":
    run_project()
