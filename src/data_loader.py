"""
Data loader bersama untuk dataset Iris.

JANGAN MENGUBAH FUNGSI INI. Semua varian eksperimen harus memanggil
load_iris_data() supaya menerima train/test split yang identik.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

from .config import RANDOM_SEED, TEST_SIZE


def load_iris_data():
    """
    Memuat dataset Iris, melakukan one-hot encoding pada label,
    melakukan train/test split, dan menormalisasi fitur.

    Returns:
        X_train, X_test, y_train, y_test (numpy arrays)
        n_features (int): jumlah fitur input
        n_classes (int):  jumlah kelas output
    """
    # 1. Memuat dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    # 2. One-hot encoding label (3 kelas: setosa, versicolor, virginica)
    y = to_categorical(y, n_classes)

    # 3. Split train/test — seed tetap supaya reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # 4. Normalisasi (fit hanya di training, transform di keduanya)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, n_features, n_classes


if __name__ == "__main__":
    # Sanity check — jalankan `python -m src.data_loader` untuk verifikasi
    X_train, X_test, y_train, y_test, n_feat, n_cls = load_iris_data()
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test  shape: {y_test.shape}")
    print(f"Jumlah fitur: {n_feat}, Jumlah kelas: {n_cls}")
