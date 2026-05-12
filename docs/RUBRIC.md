# Rubrik Penilaian — ANN Bake-Off

Total: **100 poin**.

## A. Keempat Notebook Varian (40 poin)

| Kriteria | Excellent (10) | Good (7) | Cukup (4) | Kurang (0) |
|---|---|---|---|---|
| Notebook 01 | Berjalan tanpa error, output lengkap, refleksi diisi mendalam | Berjalan, refleksi ada tapi singkat | Berjalan tapi refleksi kosong | Tidak ada / error |
| Notebook 02 | _idem_ | _idem_ | _idem_ | _idem_ |
| Notebook 03 | _idem_ | _idem_ | _idem_ | _idem_ |
| Notebook 04 | _idem_ | _idem_ | _idem_ | _idem_ |

## B. Notebook Comparison `05_comparison.ipynb` (25 poin)

- Berhasil membaca keempat CSV history dan menampilkan tabel ringkasan: **10 poin**
- Grafik perbandingan val_accuracy dan val_loss dirender dengan benar: **10 poin**
- Tabel diurutkan dan informatif (mudah dibaca): **5 poin**

## C. Laporan `docs/REPORT.md` (20 poin)

- Ringkasan hasil eksperimen lengkap: **3 poin**
- Minimal 3 pertanyaan diskusi dijawab: **9 poin** (3 poin/pertanyaan)
- Refleksi proses kerja kelompok: **5 poin**
- Kontribusi anggota jelas dan jujur: **3 poin**

**Catatan:** Jawaban yang sekadar mendeskripsikan grafik tanpa mengaitkan dengan teori akan mendapat poin minimum. Yang dinilai adalah **kualitas analisis**, bukan panjang tulisan.

## D. Praktik Git/Kolaborasi (15 poin)

- Setiap anggota punya minimal 1 commit dari akunnya sendiri: **8 poin**
- Pesan commit deskriptif (bukan "update" atau "asdf"): **4 poin**
- Tidak ada konflik merge yang belum diresolusi di branch main: **3 poin**

## Pengurangan Poin (Penalty)

- Notebook tidak menyimpan output (kosong saat dibuka): **-5 per notebook**
- Mengubah `src/config.py` atau `src/data_loader.py` tanpa alasan jelas (merusak fair comparison): **-10**
- Plagiarisme antar kelompok (notebook persis sama termasuk komentar): **0 untuk semua kelompok yang terlibat**

## Bonus (+10 poin)

Diberikan untuk salah satu (atau kombinasi) hal berikut:
- Menambahkan varian ke-5 (misal: ReLU dengan dropout, atau Leaky ReLU) dengan analisis sendiri.
- Melakukan eksperimen pada dataset kedua (mis. Breast Cancer Wisconsin) dan membandingkan hasilnya.
- Memvisualisasikan decision boundary atau feature embedding hidden layer.
