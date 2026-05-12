"""
Konfigurasi bersama untuk semua varian eksperimen.

PENTING: JANGAN MENGUBAH NILAI DI FILE INI tanpa kesepakatan kelompok.
Semua varian (single-layer, MLP-sigmoid, MLP-tanh, MLP-relu) harus
menggunakan konfigurasi yang sama agar perbandingan adil (controlled experiment).
"""

# Reproducibility — semua anggota wajib memakai seed yang sama
RANDOM_SEED = 42

# Pembagian data
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2  # diambil dari training set saat model.fit()

# Hyperparameter training
EPOCHS = 100
BATCH_SIZE = 8
OPTIMIZER = "adam"
LOSS = "categorical_crossentropy"
METRICS = ["accuracy"]

# Output direktori untuk menyimpan history (.csv) tiap varian
RESULTS_DIR = "results"
