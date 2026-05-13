# Laporan Kelompok — ANN Bake-Off

## Identitas Kelompok

- **Nama Kelompok:** kelompok_tiga
- **Anggota:**
  1. Dionisius Rafael — 32230013 — 05 Comparison
  2. Nathasia Oktarina Riyani — 32230025 — 04 ReLU
  3. Caryn Caroline — 32230042 — 02 Sigmoid
  4. Nadia Aurelia Clarissa — 32230052 — Laporan
  5. Edward Soeyanto — 32230060 — 05 Comparison
  6. Aliya Cahyanti Wijaya — 32230063 — 01 Single Layer
  7. Regina Hillary — 32230070 — 03 Tanh

---

## 1. Ringkasan Hasil Eksperimen

Tabel berikut merangkum hasil epoch terakhir (epoch 100) dari keempat varian model. Nilai yang ditampilkan adalah **validation accuracy** dan **validation loss** (digunakan sebagai ukuran performa generalisasi), serta **train accuracy** dan **train loss** sebagai pembanding.

| Varian | Arsitektur | Aktivasi | Val Accuracy | Val Loss | Train Accuracy | Train Loss |
|--------|------------|----------|:------------:|:--------:|:--------------:|:----------:|
| 01 | Single layer (input → output) | — (Softmax) | 0.7500 | 0.5437 | 0.8646 | 0.4167 |
| 02 | 1 hidden layer (16 unit) | Sigmoid | 0.9583 | 0.3086 | 0.9167 | 0.2511 |
| 03 | 1 hidden layer (16 unit) | Tanh | 0.9583 | 0.1265 | 0.9688 | 0.1138 |
| 04 | 2 hidden layers (32 → 16 unit) | ReLU | 0.9583 | 0.0807 | 0.9896 | 0.0346 |

> **Catatan jumlah parameter:** Varian 01 hanya memiliki bobot dari input langsung ke output (tanpa hidden layer), sehingga jumlah parameternya paling sedikit. Varian 02 dan 03 menambahkan 1 hidden layer (16 unit), sedangkan Varian 04 memiliki 2 hidden layer (32→16 unit) sehingga parameter paling banyak.

---

## 2. Analisis & Diskusi

### 2.1 Varian mana yang terbaik dan mengapa?

Varian 02 (Sigmoid), 03 (Tanh), dan 04 (ReLU) mencapai **validation accuracy yang sama di epoch 100 yaitu 0.9583**. Namun jika dilihat dari **validation loss**, Varian 04 (ReLU) menghasilkan val loss terendah (0.0807), diikuti Varian 03 (0.1265), lalu Varian 02 (0.3086). Ini menunjukkan bahwa model dengan arsitektur lebih dalam (2 hidden layers) dan fungsi aktivasi ReLU menghasilkan **distribusi probabilitas prediksi yang lebih tajam dan percaya diri**, meskipun akurasi akhirnya setara.

Varian 01 (Single Layer) adalah yang terlemah dengan val accuracy hanya 0.7500. Hal ini sesuai dengan teori: tanpa hidden layer, model hanya mampu mempelajari batas keputusan yang **linear**, sehingga tidak cukup ekspresif untuk data yang membutuhkan representasi non-linear.

### 2.2 Apakah ada tanda overfitting?

Overfitting dapat dideteksi dari gap antara train accuracy dan val accuracy. Hasil analisis per varian:

- **Varian 01 (Single Layer):** Gap = 0.1146 → **OVERFIT**. Meskipun model sederhana, ia mulai "menghafal" pola training yang tidak tergeneralisasi dengan baik. Ini mungkin disebabkan oleh tidak adanya kapasitas representasi yang memadai sehingga model gagal menangkap pola umum.
- **Varian 02 (Sigmoid):** Gap = −0.042 → **OK** (val accuracy bahkan lebih tinggi dari train accuracy, yang bisa menandakan regularisasi implisit atau data validasi yang relatif mudah).
- **Varian 03 (Tanh):** Gap = 0.010 → **OK**, generalisasi sangat baik.
- **Varian 04 (ReLU):** Gap = 0.031 → **OK**, sedikit gap tapi masih dalam batas wajar.

Secara keseluruhan, penambahan hidden layer justru **mengurangi overfitting** dibanding model tanpa hidden layer, karena hidden layer membantu model belajar fitur yang lebih representatif dan tidak sekadar "menghafal".

### 2.3 Pengaruh fungsi aktivasi terhadap kecepatan konvergensi

Kecepatan konvergensi diukur dari epoch pertama saat val accuracy mencapai ≥ 0.90:

- **Varian 01:** Tidak pernah mencapai 0.90 dalam 100 epoch.
- **Varian 02 (Sigmoid):** Epoch 34.
- **Varian 03 (Tanh):** Epoch 24.
- **Varian 04 (ReLU):** Epoch 14 tercepat.

Hasil ini konsisten dengan teori. ReLU menghindari *vanishing gradient* karena derivatifnya bernilai konstan (1 untuk input positif), sehingga sinyal gradien tidak mengecil saat backpropagation. Sebaliknya, Sigmoid rentan terhadap *vanishing gradient* karena gradiennya mendekati nol di ujung kurva saturasi, membuat proses belajar lebih lambat. Tanh lebih baik dari Sigmoid karena zero-centered (output rata-rata mendekati nol) sehingga update bobot lebih stabil, namun tetap mengalami saturasi di nilai ekstrem.

---

## 3. Refleksi Proses Kerja Kelompok

Dalam pengerjaan proyek ini, kelompok membagi tugas secara sederhana karena tugas utama hanya menjalankan beberapa file Google Colab yang telah disediakan. Total terdapat lima file Colab yang berkaitan dengan implementasi neural network, sehingga setiap anggota bertanggung jawab pada bagian masing-masing. Sebagian anggota menjalankan dan memahami isi file Colab, sementara satu anggota bertugas menyusun laporan hasil pengerjaan agar seluruh hasil dapat terdokumentasi dengan baik. Pembagian tugas dilakukan agar proses pengerjaan lebih cepat dan setiap anggota tetap memiliki kontribusi dalam proyek.

Selama pengerjaan, terdapat beberapa kendala baik teknis maupun non-teknis. Dari sisi teknis, beberapa anggota mengalami error saat menjalankan file Colab, seperti masalah library yang belum terpasang, runtime yang disconnect, atau hasil output yang berbeda. Selain itu, ada juga anggota yang belum terlalu memahami konsep neural network sehingga memerlukan waktu tambahan untuk memahami alur program yang dijalankan. Dari sisi non-teknis, kendala yang muncul lebih kepada koordinasi antaranggota dan pembagian waktu, karena setiap anggota memiliki jadwal dan kesibukan masing-masing.

Untuk mengatasi kendala tersebut, kelompok melakukan komunikasi melalui diskusi singkat dan saling membantu ketika ada error atau bagian yang belum dipahami. Anggota yang lebih memahami kode membantu menjelaskan proses yang ada di dalam Colab kepada anggota lainnya. Selain itu, kelompok juga mencari referensi tambahan dari internet maupun dokumentasi library yang digunakan agar masalah teknis dapat diselesaikan lebih cepat. Pembagian tugas yang jelas juga membantu pengerjaan menjadi lebih terarah dan mengurangi kemungkinan pekerjaan menumpuk pada satu orang saja.

Melalui proyek ini, kelompok memperoleh beberapa pelajaran penting yang dapat diterapkan pada proyek machine learning berikutnya. Salah satunya adalah pentingnya kerja sama dan komunikasi dalam tim, meskipun tugas yang dikerjakan relatif sederhana. Selain itu, kelompok juga belajar bahwa pemahaman dasar mengenai alur machine learning dan neural network tetap diperlukan agar tidak hanya menjalankan program tanpa memahami prosesnya. Pengalaman ini juga memberikan pemahaman bahwa dokumentasi dan pembagian tugas yang jelas dapat membuat pengerjaan proyek menjadi lebih efektif dan efisien.

---

## 4. Kontribusi Tiap Anggota

| Anggota | Kontribusi konkret | % Effort |
|---|---|---|
| Dionisius Rafael | Menjalankan & menganalisis notebook 05 Comparison, menyusun grafik komparatif | 14.3% |
| Nathasia Oktarina Riyani | Menjalankan notebook 04 ReLU, menghasilkan file `04_mlp_relu.csv` | 14.3% |
| Caryn Caroline | Menjalankan notebook 02 Sigmoid, menghasilkan file `02_mlp_sigmoid.csv` | 14.3% |
| Nadia Aurelia Clarissa | Menyusun laporan kelompok (REPORT.md), merangkum hasil dan analisis | 14.3% |
| Edward Soeyanto | Menjalankan & menganalisis notebook 05 Comparison, membuat ringkasan hasil | 14.3% |
| Aliya Cahyanti Wijaya | Menjalankan notebook 01 Single Layer, menghasilkan file `01_single_layer.csv` | 14.3% |
| Regina Hillary | Menjalankan notebook 03 Tanh, menghasilkan file `03_mlp_tanh.csv` | 14.3% |
| | **Total** | **100%** |

---

## 5. Referensi

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org
2. Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. *ICML 2010*.
3. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444.
4. Dokumentasi Keras — Activation Functions. https://keras.io/api/layers/activations/
