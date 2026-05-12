# 🧠 ANN Bake-Off — Group Assignment

> Mata kuliah: **DSB07 — Machine Learning**
> Topik: **Artificial Neural Network**
> Format: Kelompok 4 mahasiswa, dikerjakan di **Google Colab**, hasil di-commit ke **GitHub**.

---

## 🎯 Tujuan

Setelah menyelesaikan tugas ini, kelompok akan mampu:

1. Mengimplementasikan model ANN menggunakan Keras/TensorFlow.
2. Membandingkan dampak **arsitektur** (single-layer vs multi-layer) terhadap performa.
3. Membandingkan dampak **fungsi aktivasi** (Sigmoid, Tanh, ReLU) terhadap kecepatan konvergensi dan akurasi.
4. Berkolaborasi melalui Git/GitHub untuk eksperimen ilmiah yang reproducible.

---

## 📋 Pembagian Peran Kelompok

Setiap anggota kelompok **wajib** mengerjakan satu varian. Notebook starter sudah disediakan di folder `notebooks/`.

| Anggota | File Notebook | Arsitektur | Fungsi Aktivasi Hidden |
|---|---|---|---|
| 1 | `notebooks/01_single_layer.ipynb` | Tanpa hidden layer | — (output saja) |
| 2 | `notebooks/02_mlp_sigmoid.ipynb` | 1 hidden layer (16 neuron) | **Sigmoid** |
| 3 | `notebooks/03_mlp_tanh.ipynb` | 1 hidden layer (16 neuron) | **Tanh** |
| 4 | `notebooks/04_mlp_relu.ipynb` | 2 hidden layer (32 → 16 neuron) | **ReLU** |

Setelah keempat notebook selesai, **satu anggota** menggabungkan hasilnya di `notebooks/05_comparison.ipynb`.

> ⚠️ **Aturan main:** Semua varian **harus** menggunakan dataset, random seed, split, optimizer (Adam), batch size, dan jumlah epoch yang sama. Yang berbeda **hanya arsitektur dan fungsi aktivasi**. Ini supaya perbandingannya adil (controlled experiment).

---

## 🚀 Cara Memulai (Quickstart)

### 1. Accept assignment di GitHub Classroom
Klik link invitation dari dosen → buat tim → repo otomatis dibuat untuk kelompok Anda.

### 2. Buka di Google Colab
Setiap notebook punya tombol **"Open in Colab"** di bagian atas. Klik tombol tersebut, lalu di Colab:

```
File → Save a copy in GitHub → pilih repo kelompok Anda → commit
```

Atau, clone repo dan buka file `.ipynb` langsung dari Colab:

```python
!git clone https://github.com/UBM-ML/<nama-repo-kelompok>.git
%cd <nama-repo-kelompok>
```

### 3. Jalankan notebook varian Anda
- Lengkapi bagian yang ditandai `# TODO`.
- Jalankan semua cell. Pastikan tidak ada error.
- Output (akurasi, plot) **harus tersimpan** di notebook saat commit.

### 4. Commit hasil ke GitHub
Dari Colab: `File → Save a copy in GitHub`.
Dari terminal: `git add notebooks/0X_*.ipynb && git commit -m "..." && git push`.

### 5. Gabungkan di `05_comparison.ipynb`
Anggota yang ditugaskan menjalankan notebook gabungan untuk menghasilkan tabel + grafik komparatif.

### 6. Tulis `REPORT.md`
Refleksi singkat (300–500 kata) di file `docs/REPORT.md`. Template sudah disediakan.

---

## 📦 Struktur Repo

```
.
├── README.md                    # File ini
├── notebooks/
│   ├── 01_single_layer.ipynb    # Anggota 1
│   ├── 02_mlp_sigmoid.ipynb     # Anggota 2
│   ├── 03_mlp_tanh.ipynb        # Anggota 3
│   ├── 04_mlp_relu.ipynb        # Anggota 4
│   └── 05_comparison.ipynb      # Notebook gabungan
├── src/
│   ├── data_loader.py           # Fungsi load_dataset() — JANGAN DIUBAH
│   ├── utils.py                 # Plotting, evaluasi — boleh dikembangkan
│   └── config.py                # Hyperparameter bersama (seed, epochs, dll)
├── docs/
│   ├── REPORT.md                # Template laporan kelompok
│   └── RUBRIC.md                # Rubrik penilaian
└── requirements.txt             # Untuk environment lokal (opsional)
```

---

## 📊 Dataset

Kami menggunakan dataset **Iris** (`sklearn.datasets.load_iris`) — sama seperti contoh di slide kuliah. Dataset ini dipilih karena:
- Cukup kecil untuk latihan cepat di Colab CPU.
- 4 fitur, 3 kelas → cocok untuk multi-class classification.
- Hasilnya cukup bervariasi antar varian untuk dibandingkan secara menarik.

> Bonus (opsional): Anda boleh menambahkan eksperimen dengan dataset **Breast Cancer Wisconsin** (`load_breast_cancer`) di notebook terpisah untuk nilai tambahan.

---

## 📝 Penilaian

Lihat `docs/RUBRIC.md` untuk rincian poin. Garis besar:

- **40%** — Keempat notebook varian berjalan tanpa error & output lengkap.
- **25%** — Notebook `05_comparison.ipynb` benar dan informatif.
- **20%** — `REPORT.md` berkualitas (analisis, bukan sekadar deskriptif).
- **15%** — Praktik Git/kolaborasi (commit dari setiap anggota terlihat, pesan commit jelas).

---

## ❓ Pertanyaan Diskusi untuk Laporan

Saat menulis `REPORT.md`, kelompok wajib membahas minimal 3 dari pertanyaan berikut:

1. Apakah single-layer mampu mencapai akurasi yang sebanding dengan multi-layer? Mengapa?
2. Pada dataset ini, apakah ReLU benar-benar konvergen lebih cepat dibanding Sigmoid? Buktikan dengan grafik loss.
3. Apa yang terjadi pada validation accuracy ketika model terlalu dalam (overfitting)? Cocokkan dengan klaim di slide 1.9.
4. Jika kelompok menambah hidden layer ke 3 atau 4, apakah akurasi terus naik? Lakukan eksperimen tambahan.
5. Bandingkan klaim slide 2.7 ("Tanh mempercepat pembelajaran karena zero-centered") dengan hasil empiris Anda.

---

## 🆘 Bantuan

- Slack channel: `#dsb07-ml`
- Office hour dosen: lihat LMS.
- Untuk error TensorFlow di Colab, biasanya cukup `Runtime → Restart`.

Selamat ber-eksperimen! 🚀
