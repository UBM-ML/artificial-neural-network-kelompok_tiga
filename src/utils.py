"""
Utilitas bersama: pengaturan seed, plotting kurva training,
dan penyimpanan history training agar dapat digabungkan di
notebook perbandingan.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import RANDOM_SEED, RESULTS_DIR


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Mengunci seed untuk numpy, random, dan tensorflow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_history(history, variant_name: str) -> str:
    """
    Menyimpan history training ke file CSV di folder results/.
    File ini akan dibaca oleh 05_comparison.ipynb.

    Args:
        history: objek History yang dikembalikan oleh model.fit()
        variant_name: identifier varian, mis. "01_single_layer"

    Returns:
        Path ke file CSV yang disimpan.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(history.history)
    df["epoch"] = df.index + 1
    df["variant"] = variant_name
    path = os.path.join(RESULTS_DIR, f"{variant_name}.csv")
    df.to_csv(path, index=False)
    print(f"✅ History tersimpan di: {path}")
    return path


def plot_training_curves(history, variant_name: str) -> None:
    """Plot loss dan accuracy (training + validation) dalam satu figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title(f"{variant_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history["accuracy"], label="train")
    if "val_accuracy" in history.history:
        axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title(f"{variant_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def evaluate_and_report(model, X_test, y_test, variant_name: str) -> dict:
    """Cetak metrik test dan kembalikan ringkasan dict untuk digabung nanti."""
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    n_params = model.count_params()
    print(f"\n📊 {variant_name}")
    print(f"   Test loss:     {loss:.4f}")
    print(f"   Test accuracy: {acc:.4f}")
    print(f"   Jumlah parameter: {n_params:,}")
    return {
        "variant": variant_name,
        "test_loss": loss,
        "test_accuracy": acc,
        "n_params": n_params,
    }
