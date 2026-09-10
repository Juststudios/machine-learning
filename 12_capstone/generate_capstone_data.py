"""
generate_capstone_data.py
=========================
Generates the synthetic industrial sensor dataset for the capstone project.

WHY synthetic data?
    - We control the ground truth, so solutions can be verified
    - We can inject realistic patterns (high temp correlates with fault)
    - We can add controlled noise and missing values for preprocessing practice

Dataset design rationale:
    - 8 sensor features mimic a real industrial pump monitoring setup
    - Two targets: regression (remaining useful life) and classification (fault severity)
    - Fault state is realistically correlated with multiple sensors
    - ~5% missing values in two sensor columns for preprocessing practice

Run this script BEFORE running any capstone solution or template.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─── Reproducibility ─────────────────────────────────────────────────────────
# WHY: Fixed seed ensures every student gets the same dataset
np.random.seed(42)

# ─── Output directory ─────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
DATASETS_DIR = HERE.parent / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)


def generate_sensor_data(n_samples: int, split: str = "train") -> pd.DataFrame:
    """
    Generate one split of the industrial sensor dataset.

    Design decisions:
    - fault_severity is the primary label (0=OK, 1=WARNING, 2=FAULT)
    - Sensor values are shifted depending on fault_severity to create
      learnable signal
    - remaining_useful_life is derived from fault_severity + noise
    - Missing values are injected post-hoc in 'pressure' and 'current' columns
    """

    # ─── Step 1: Sample fault states ──────────────────────────────────────────
    # WHY: Realistic class imbalance — most readings are normal operation
    # Class distribution: 60% OK, 25% WARNING, 15% FAULT
    fault_probs = [0.60, 0.25, 0.15]
    fault_severity = np.random.choice([0, 1, 2], size=n_samples, p=fault_probs)

    # ─── Step 2: Generate sensor features conditioned on fault state ───────────
    # Each sensor has a baseline range and a fault-induced shift.
    # WHY: This creates the learnable signal — a model should discover that
    #      high temperature + high vibration → FAULT

    # Temperature: normal ~65°C, rises with fault severity
    temp_base = {0: 65.0, 1: 78.0, 2: 95.0}
    temperature = np.array([
        np.random.normal(temp_base[f], 5.0) for f in fault_severity
    ])

    # Vibration: normal ~2.5 mm/s, increases with fault
    vib_base = {0: 2.5, 1: 5.5, 2: 9.2}
    vibration = np.array([
        np.random.normal(vib_base[f], 0.8) for f in fault_severity
    ])

    # Pressure: normal ~4.5 bar, drops at FAULT (seal failure)
    pres_base = {0: 4.5, 1: 4.1, 2: 3.2}
    pressure = np.array([
        np.random.normal(pres_base[f], 0.3) for f in fault_severity
    ])

    # Motor current: rises with fault (pump working harder)
    curr_base = {0: 12.0, 1: 15.5, 2: 21.0}
    current = np.array([
        np.random.normal(curr_base[f], 1.5) for f in fault_severity
    ])

    # RPM: normal ~1480 rpm, drops slightly at FAULT
    rpm_base = {0: 1480.0, 1: 1450.0, 2: 1390.0}
    rpm = np.array([
        np.random.normal(rpm_base[f], 20.0) for f in fault_severity
    ])

    # Humidity: environmental, mostly independent of fault (slight correlation)
    # WHY: Including a near-irrelevant feature teaches feature importance analysis
    humidity = np.random.normal(55.0, 10.0, n_samples)
    # Small fault-correlated shift (condensation near hot components)
    humidity += fault_severity * 3.0

    # Voltage: supply voltage, slightly unstable during fault
    volt_base = {0: 400.0, 1: 398.0, 2: 392.0}
    voltage = np.array([
        np.random.normal(volt_base[f], 2.0) for f in fault_severity
    ])

    # Acoustic emission: high-frequency sound, very diagnostic of fault
    acoustic_base = {0: 48.0, 1: 61.0, 2: 78.0}
    acoustic_emission = np.array([
        np.random.normal(acoustic_base[f], 4.0) for f in fault_severity
    ])

    # ─── Step 3: Generate regression target ───────────────────────────────────
    # Remaining Useful Life (RUL) in hours
    # WHY: RUL is inversely related to fault severity with added noise
    rul_base = {0: 400.0, 1: 150.0, 2: 30.0}
    remaining_useful_life = np.array([
        max(0.0, np.random.normal(rul_base[f], 40.0)) for f in fault_severity
    ])
    # Clip to realistic range [0, 500]
    remaining_useful_life = np.clip(remaining_useful_life, 0.0, 500.0)

    # ─── Step 4: Clip sensor values to physical limits ────────────────────────
    temperature = np.clip(temperature, 30.0, 150.0)
    vibration = np.clip(vibration, 0.1, 20.0)
    pressure = np.clip(pressure, 0.5, 8.0)
    current = np.clip(current, 5.0, 40.0)
    rpm = np.clip(rpm, 800.0, 1600.0)
    humidity = np.clip(humidity, 10.0, 95.0)
    voltage = np.clip(voltage, 370.0, 420.0)
    acoustic_emission = np.clip(acoustic_emission, 30.0, 100.0)

    # ─── Step 5: Assemble DataFrame ───────────────────────────────────────────
    df = pd.DataFrame({
        'temperature':        np.round(temperature, 2),
        'vibration':          np.round(vibration, 3),
        'pressure':           np.round(pressure, 3),
        'current':            np.round(current, 2),
        'rpm':                np.round(rpm, 1),
        'humidity':           np.round(humidity, 1),
        'voltage':            np.round(voltage, 1),
        'acoustic_emission':  np.round(acoustic_emission, 1),
        'remaining_useful_life': np.round(remaining_useful_life, 1),
        'fault_severity':     fault_severity.astype(int),
    })

    # ─── Step 6: Inject missing values ────────────────────────────────────────
    # WHY: Real sensor data has dropouts. Students must handle NaN values.
    # ~5% missing in 'pressure' (sensor calibration drift)
    # ~3% missing in 'current' (intermittent connection)
    n_missing_pressure = int(n_samples * 0.05)
    n_missing_current = int(n_samples * 0.03)

    missing_pressure_idx = np.random.choice(df.index, n_missing_pressure, replace=False)
    missing_current_idx = np.random.choice(df.index, n_missing_current, replace=False)

    df.loc[missing_pressure_idx, 'pressure'] = np.nan
    df.loc[missing_current_idx, 'current'] = np.nan

    # ─── Step 7: Shuffle rows ─────────────────────────────────────────────────
    # WHY: Real data isn't sorted by fault state. Prevents trivial patterns.
    df = df.sample(frac=1, random_state=42 if split == "train" else 99).reset_index(drop=True)

    return df


def main():
    print("=" * 60)
    print("  Meridian Industrial — Sensor Dataset Generator")
    print("=" * 60)

    # Generate training set
    print("\n[1/2] Generating training set (1000 samples)...")
    train_df = generate_sensor_data(n_samples=1000, split="train")

    train_path = DATASETS_DIR / "industrial_sensor_train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"      Saved → {train_path}")

    # Generate test set
    print("[2/2] Generating test set (200 samples)...")

    # WHY separate seed: ensures test set has no overlap with train generation
    np.random.seed(99)
    test_df = generate_sensor_data(n_samples=200, split="test")

    test_path = DATASETS_DIR / "industrial_sensor_test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"      Saved → {test_path}")

    # ─── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Dataset Summary")
    print("=" * 60)

    print(f"\nTraining set shape: {train_df.shape}")
    print(f"\nFault severity distribution (train):")
    counts = train_df['fault_severity'].value_counts().sort_index()
    labels = {0: "OK", 1: "WARNING", 2: "FAULT"}
    for cls, count in counts.items():
        pct = count / len(train_df) * 100
        print(f"  Class {cls} ({labels[cls]:8s}): {count:4d} samples ({pct:.1f}%)")

    print(f"\nRemaining Useful Life stats (train):")
    print(f"  Min:  {train_df['remaining_useful_life'].min():.1f} hrs")
    print(f"  Max:  {train_df['remaining_useful_life'].max():.1f} hrs")
    print(f"  Mean: {train_df['remaining_useful_life'].mean():.1f} hrs")
    print(f"  Std:  {train_df['remaining_useful_life'].std():.1f} hrs")

    print(f"\nMissing values (train):")
    missing = train_df.isnull().sum()
    for col, n in missing[missing > 0].items():
        print(f"  {col}: {n} missing ({n/len(train_df)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("  Ready! You can now run starter_template.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
