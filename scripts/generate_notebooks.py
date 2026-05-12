"""
Script untuk menghasilkan 4 notebook varian + 1 notebook perbandingan.
Dijalankan sekali oleh pembuat template, tidak dijalankan oleh mahasiswa.
"""

import json
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
os.makedirs(OUT_DIR, exist_ok=True)


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src if isinstance(src, list) else [src]}

def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src if isinstance(src, list) else [src]}


COLAB_BADGE = (
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/REPLACE-WITH-YOUR-ORG/REPLACE-WITH-YOUR-REPO/blob/main/notebooks/{filename})\n"
)

SETUP_CLONE = [
    "# Jalankan cell ini HANYA jika kamu berada di Google Colab.\n",
    "# Kalau kamu menjalankan di lokal/Jupyter, cukup pastikan kamu berada di root repo.\n",
    "\n",
    "import os\n",
    "if not os.path.exists('src'):\n",
    "    # Ganti URL di bawah dengan URL repo kelompok kamu\n",
    "    REPO_URL = 'https://github.com/REPLACE-WITH-YOUR-ORG/REPLACE-WITH-YOUR-REPO.git'\n",
    "    !git clone $REPO_URL repo\n",
    "    %cd repo\n",
    "print('Working dir:', os.getcwd())\n",
    "print('Contents:', os.listdir('.'))\n",
]

COMMON_IMPORTS = [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Input\n",
    "\n",
    "from src.data_loader import load_iris_data\n",
    "from src.utils import set_global_seed, plot_training_curves, save_history, evaluate_and_report\n",
    "from src.config import EPOCHS, BATCH_SIZE, OPTIMIZER, LOSS, METRICS, VALIDATION_SPLIT, RANDOM_SEED\n",
    "\n",
    "set_global_seed(RANDOM_SEED)\n",
    "print('TensorFlow version:', tf.__version__)\n",
]

LOAD_DATA = [
    "X_train, X_test, y_train, y_test, n_features, n_classes = load_iris_data()\n",
    "print(f'Jumlah fitur: {n_features}, jumlah kelas: {n_classes}')\n",
    "print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')\n",
]


def build_variant_notebook(filename, title, anggota_slot, arsitektur_desc,
                           aktivasi_desc, model_code_lines, reflection_questions):
    cells = []
    cells.append(md([
        f"# {title}\n",
        "\n",
        COLAB_BADGE.format(filename=filename),
        "\n",
        f"**Anggota yang mengerjakan:** _{anggota_slot}_\n",
        "\n",
        "---\n",
        "\n",
        f"## 🏗️ Arsitektur\n{arsitektur_desc}\n",
        "\n",
        f"## ⚡ Fungsi Aktivasi\n{aktivasi_desc}\n",
        "\n",
        "## 🎯 Goal\n",
        "Menjalankan eksperimen ini, menyimpan history training, lalu commit notebook ini "
        "(dengan output yang sudah ter-render) ke repo GitHub kelompok.\n",
    ]))

    cells.append(md(["## 1. Setup environment"]))
    cells.append(code(SETUP_CLONE))

    cells.append(md(["## 2. Import library"]))
    cells.append(code(COMMON_IMPORTS))

    cells.append(md([
        "## 3. Load data\n",
        "Catatan: data sudah otomatis di-split, di-shuffle, dan dinormalisasi sesuai "
        "konfigurasi bersama di `src/config.py`. **Jangan diubah** supaya perbandingan adil."
    ]))
    cells.append(code(LOAD_DATA))

    cells.append(md(["## 4. Bangun model"]))
    cells.append(code(model_code_lines))

    cells.append(md([
        "## 5. Latih model\n",
        "Hyperparameter (epochs, batch_size, optimizer) diambil dari `src/config.py` "
        "supaya identik dengan varian lain."
    ]))
    cells.append(code([
        "history = model.fit(\n",
        "    X_train, y_train,\n",
        "    epochs=EPOCHS,\n",
        "    batch_size=BATCH_SIZE,\n",
        "    validation_split=VALIDATION_SPLIT,\n",
        "    verbose=2,\n",
        ")\n",
    ]))

    cells.append(md(["## 6. Visualisasi kurva training"]))
    cells.append(code([f"plot_training_curves(history, variant_name='{filename.replace('.ipynb','')}')\n"]))

    cells.append(md(["## 7. Evaluasi di test set"]))
    cells.append(code([
        f"summary = evaluate_and_report(model, X_test, y_test, variant_name='{filename.replace('.ipynb','')}')\n",
        f"save_history(history, variant_name='{filename.replace('.ipynb','')}')\n",
        "summary\n",
    ]))

    cells.append(md([
        "## 8. Refleksi singkat\n",
        "_Diisi oleh anggota yang mengerjakan notebook ini._ Tuliskan jawaban dalam cell markdown di bawah:\n",
        "\n",
        *[f"{i+1}. {q}\n" for i, q in enumerate(reflection_questions)],
    ]))
    cells.append(md(["_Tulis jawabanmu di sini..._"]))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    path = os.path.join(OUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"✅ {path}")


# ============================================================
# VARIAN 1: Single layer (no hidden) — baseline linier
# ============================================================
build_variant_notebook(
    filename="01_single_layer.ipynb",
    title="Varian 01 — Single Layer (Baseline Linier)",
    anggota_slot="Anggota 1",
    arsitektur_desc=(
        "Single-layer network: **input langsung ke output**, tanpa hidden layer. "
        "Ini setara dengan logistic regression multi-kelas (sesuai sub-bab 2.2 di slide)."
    ),
    aktivasi_desc=(
        "Tidak ada fungsi aktivasi di hidden layer (karena tidak ada hidden layer). "
        "Output layer menggunakan **softmax** untuk klasifikasi 3 kelas."
    ),
    model_code_lines=[
        "model = Sequential([\n",
        "    Input(shape=(n_features,)),\n",
        "    Dense(n_classes, activation='softmax'),  # langsung ke output\n",
        "])\n",
        "model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)\n",
        "model.summary()\n",
    ],
    reflection_questions=[
        "Berapa total jumlah parameter model ini? Bandingkan dengan varian multi-layer.",
        "Menurutmu, apakah model ini cukup ekspresif untuk Iris? Mengapa?",
        "Jika dataset diganti dengan yang non-linear separable (mis. XOR), apa prediksimu?",
    ],
)


# ============================================================
# VARIAN 2: MLP — 1 hidden layer dengan Sigmoid
# ============================================================
build_variant_notebook(
    filename="02_mlp_sigmoid.ipynb",
    title="Varian 02 — MLP 1 Hidden Layer (Sigmoid)",
    anggota_slot="Anggota 2",
    arsitektur_desc="1 hidden layer berisi **16 neuron**, lalu output 3 neuron.",
    aktivasi_desc=(
        "Hidden layer menggunakan **Sigmoid** (rumus: f(x) = 1 / (1 + e^-x)). "
        "Range output 0..1. Output layer tetap softmax."
    ),
    model_code_lines=[
        "model = Sequential([\n",
        "    Input(shape=(n_features,)),\n",
        "    Dense(16, activation='sigmoid'),\n",
        "    Dense(n_classes, activation='softmax'),\n",
        "])\n",
        "model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)\n",
        "model.summary()\n",
    ],
    reflection_questions=[
        "Lihat kurva training-mu. Pada epoch berapa loss mulai stabil?",
        "Apakah Sigmoid cocok untuk hidden layer? Coba kaitkan dengan fenomena 'vanishing gradient'.",
        "Jika kamu menambah hidden layer kedua (lagi Sigmoid), apakah hasilnya membaik atau memburuk?",
    ],
)


# ============================================================
# VARIAN 3: MLP — 1 hidden layer dengan Tanh
# ============================================================
build_variant_notebook(
    filename="03_mlp_tanh.ipynb",
    title="Varian 03 — MLP 1 Hidden Layer (Tanh)",
    anggota_slot="Anggota 3",
    arsitektur_desc=(
        "1 hidden layer berisi **16 neuron**, lalu output 3 neuron. "
        "Arsitektur identik dengan Varian 02 — yang berbeda hanya fungsi aktivasinya."
    ),
    aktivasi_desc=(
        "Hidden layer menggunakan **Tanh** (rumus: f(x) = (e^2x - 1) / (e^2x + 1)). "
        "Range output -1..1, zero-centered."
    ),
    model_code_lines=[
        "model = Sequential([\n",
        "    Input(shape=(n_features,)),\n",
        "    Dense(16, activation='tanh'),\n",
        "    Dense(n_classes, activation='softmax'),\n",
        "])\n",
        "model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)\n",
        "model.summary()\n",
    ],
    reflection_questions=[
        "Slide 2.7 mengklaim Tanh 'mempercepat pembelajaran karena zero-centered'. Apakah klaim ini terbukti pada hasilmu (bandingkan dengan Varian 02 — Sigmoid)?",
        "Coba bandingkan epoch ke berapa Varian 02 dan Varian 03 mencapai val_accuracy ≥ 0.90.",
        "Menurutmu, kenapa zero-centered penting untuk gradient descent?",
    ],
)


# ============================================================
# VARIAN 4: MLP — 2 hidden layer dengan ReLU
# ============================================================
build_variant_notebook(
    filename="04_mlp_relu.ipynb",
    title="Varian 04 — MLP 2 Hidden Layer (ReLU)",
    anggota_slot="Anggota 4",
    arsitektur_desc=(
        "2 hidden layer: **32 neuron → 16 neuron**, lalu output 3 neuron. "
        "Arsitektur paling dalam dan paling banyak parameter di kelompok ini."
    ),
    aktivasi_desc=(
        "Kedua hidden layer menggunakan **ReLU** (rumus: f(x) = max(0, x)). "
        "Cocok untuk model yang lebih dalam karena tidak terkena vanishing gradient."
    ),
    model_code_lines=[
        "model = Sequential([\n",
        "    Input(shape=(n_features,)),\n",
        "    Dense(32, activation='relu'),\n",
        "    Dense(16, activation='relu'),\n",
        "    Dense(n_classes, activation='softmax'),\n",
        "])\n",
        "model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)\n",
        "model.summary()\n",
    ],
    reflection_questions=[
        "Slide 2.8 mengklaim ReLU 'konvergen lebih cepat'. Apakah grafik loss-mu mendukung klaim ini?",
        "Model ini punya parameter paling banyak — apakah accuracy-nya selalu paling tinggi? Kalau tidak, mengapa?",
        "Apakah ada tanda-tanda overfitting (val_loss naik sementara train_loss turun)? Pada epoch berapa?",
    ],
)


# ============================================================
# NOTEBOOK 5: Comparison
# ============================================================
comparison_cells = []
comparison_cells.append(md([
    "# Notebook 05 — Perbandingan Antar Varian\n",
    "\n",
    COLAB_BADGE.format(filename="05_comparison.ipynb"),
    "\n",
    "**Anggota yang menggabungkan:** _isi nama di sini_\n",
    "\n",
    "---\n",
    "\n",
    "Notebook ini **membaca file CSV history** yang sudah dihasilkan oleh keempat varian "
    "(file ada di folder `results/`), lalu menggabungkannya menjadi tabel ringkasan dan grafik komparatif.\n",
    "\n",
    "**Prasyarat:** keempat anggota sudah commit notebook varian masing-masing dan file `results/*.csv` sudah ada di repo."
]))

comparison_cells.append(md(["## 1. Setup"]))
comparison_cells.append(code(SETUP_CLONE))

comparison_cells.append(code([
    "import os\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "RESULTS_DIR = 'results'\n",
    "files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')])\n",
    "print('Ditemukan file:', files)\n",
]))

comparison_cells.append(md(["## 2. Gabungkan semua history"]))
comparison_cells.append(code([
    "dfs = []\n",
    "for f in files:\n",
    "    df = pd.read_csv(os.path.join(RESULTS_DIR, f))\n",
    "    dfs.append(df)\n",
    "all_history = pd.concat(dfs, ignore_index=True)\n",
    "all_history.head()\n",
]))

comparison_cells.append(md(["## 3. Tabel akurasi akhir per varian"]))
comparison_cells.append(code([
    "final = all_history.sort_values('epoch').groupby('variant').tail(1)\n",
    "summary = final[['variant', 'loss', 'val_loss', 'accuracy', 'val_accuracy']]\n",
    "summary = summary.sort_values('val_accuracy', ascending=False).reset_index(drop=True)\n",
    "summary\n",
]))

comparison_cells.append(md(["## 4. Plot validation accuracy semua varian"]))
comparison_cells.append(code([
    "plt.figure(figsize=(10, 5))\n",
    "for variant, df in all_history.groupby('variant'):\n",
    "    plt.plot(df['epoch'], df['val_accuracy'], label=variant)\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Validation Accuracy')\n",
    "plt.title('Perbandingan Validation Accuracy Antar Varian')\n",
    "plt.legend()\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()\n",
]))

comparison_cells.append(md(["## 5. Plot validation loss semua varian"]))
comparison_cells.append(code([
    "plt.figure(figsize=(10, 5))\n",
    "for variant, df in all_history.groupby('variant'):\n",
    "    plt.plot(df['epoch'], df['val_loss'], label=variant)\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Validation Loss')\n",
    "plt.title('Perbandingan Validation Loss Antar Varian')\n",
    "plt.legend()\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()\n",
]))

comparison_cells.append(md([
    "## 6. Diskusi kelompok\n",
    "Berdasarkan tabel dan grafik di atas, **isi `docs/REPORT.md`** dengan analisis kelompok. "
    "Jangan hanya mendeskripsikan grafik — kaitkan dengan teori di slide kuliah."
]))

comparison_nb = {
    "cells": comparison_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = os.path.join(OUT_DIR, "05_comparison.ipynb")
with open(path, "w") as f:
    json.dump(comparison_nb, f, indent=1)
print(f"✅ {path}")
