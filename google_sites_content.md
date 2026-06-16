# DRAFT KONTEN GOOGLE SITES
## Project Akhir Data Mining - Klasifikasi Deposito Berjangka (Bank Marketing)

Gunakan draf di bawah ini untuk disalin (*copy-paste*) ke Google Sites Anda. Jangan lupa untuk mengunggah visualisasi grafis (PNG) yang telah tersimpan di folder `plots/` ke masing-masing bagian yang sesuai di Google Sites.

---

### [SEKSI 1: BERANDA / HOME]

#### **Judul Project**
> **Klasifikasi Kampanye Pemasaran Deposito Berjangka Menggunakan Algoritma Data Mining (CRISP-DM)**

#### **Identitas Mahasiswa**
*   **Nama**: [Nama Mahasiswa]
*   **NIM**: [NIM Mahasiswa]
*   **Kelas**: [Kelas Mahasiswa, misal: SI 24 P SIM-1]
*   **Program Studi**: Sistem Informasi
*   **Mata Kuliah**: Konsep Data Warehouse & Mining
*   **Dosen Pengampu**: Agus Rifaldi, S.Kom

---

### [SEKSI 2: CRISP-DM - BUSINESS UNDERSTANDING]

#### **Latar Belakang Masalah**
Dalam industri perbankan yang kompetitif, penawaran produk investasi seperti deposito berjangka sering kali dilakukan melalui kampanye pemasaran langsung (*direct marketing* via telepon). Menghubungi seluruh nasabah secara acak sangat tidak efisien dari segi waktu, tenaga, dan biaya operasional. Dengan memanfaatkan data kampanye sebelumnya, bank dapat mengidentifikasi pola nasabah yang memiliki ketertarikan tinggi untuk menaruh deposito.

#### **Tujuan Project**
1. Membangun model klasifikasi data mining yang mampu memprediksi apakah seorang nasabah akan berlangganan deposito berjangka (`y` = 'yes' atau 'no').
2. Membandingkan performa algoritma **Random Forest Classifier** dan **Logistic Regression** dalam memproses data berdimensi tinggi.
3. Mencapai akurasi model di atas target kelulusan yaitu **80%**.

#### **Manfaat Analisis**
*   **Bagi Bank**: Membantu tim marketing menyaring target nasabah potensial secara otomatis sebelum melakukan panggilan telepon, sehingga dapat menghemat biaya operasional hingga 50%.
*   **Bagi Nasabah**: Mengurangi gangguan panggilan pemasaran yang tidak relevan bagi nasabah yang tidak tertarik.

---

### [SEKSI 3: CRISP-DM - DATA UNDERSTANDING]

#### **Deskripsi Dataset**
Dataset yang digunakan berasal dari **UCI Machine Learning Repository: Bank Marketing Dataset**.
*   **Jumlah Baris**: 41,188 data (Memenuhi syarat UAS > 5.000 data)
*   **Jumlah Kolom**: 21 Atribut (20 Input, 1 Target `y`)
*   **Sumber Link**: [UCI Bank Marketing Dataset (bank-additional-full.csv)](https://archive.ics.uci.edu/dataset/222/bank+marketing)

#### **Eksplorasi Data Awal (Visualisasi)**
*(Unggah gambar-gambar berikut dari folder `plots/` ke Google Sites)*

1.  **Distribusi Kelas Target (`target_distribution.png`)**
    *   *Penjelasan*: Terlihat ketimpangan kelas yang signifikan (*class imbalance*) di mana mayoritas nasabah memilih untuk tidak berlangganan (`no` sekitar 88.7%), sementara yang berlangganan (`yes`) hanya sekitar 11.3%. Ketimpangan ini ditangani pada tahap *Modeling* menggunakan pembobotan kelas (*class weights*).
2.  **Hubungan Usia dan Status Deposito (`age_distribution.png`)**
    *   *Penjelasan*: Mayoritas nasabah yang dihubungi berada di rentang usia produktif (30-45 tahun). Namun, tingkat konversi/langganan tertinggi secara proporsional juga terlihat pada kelompok usia lanjut (>60 tahun) dan usia muda (<25 tahun).
3.  **Status Berlangganan Berdasarkan Jenis Pekerjaan (`job_vs_subscription.png`)**
    *   *Penjelasan*: Nasabah dengan pekerjaan administratif (*admin.*), kerah biru (*blue-collar*), dan teknisi (*technician*) adalah yang paling sering dihubungi. Namun secara proporsi, kelompok pensiunan (*retired*) dan mahasiswa (*student*) memiliki tingkat penerimaan penawaran deposito tertinggi.

---

### [SEKSI 4: CRISP-DM - DATA PREPARATION]

Langkah-langkah persiapan data yang dilakukan pada aplikasi Anaconda/Jupyter Notebook:
1.  **Pembersihan Data (Data Cleaning)**: Mengecek data kosong (*missing values*). Pada dataset ini, nilai tidak diketahui disimpan sebagai string `'unknown'`, yang diproses sebagai kategori terpisah agar tidak menghilangkan informasi penting.
2.  **Encoding Data**: Mengubah fitur-fitur kategorikal (seperti *job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome*) menjadi format numerik menggunakan `LabelEncoder` agar dapat diproses oleh algoritma machine learning.
3.  **Splitting Data**: Membagi dataset menjadi fitur input ($X$) dan label target ($y$), lalu membaginya menjadi **80% Data Training** (32,950 data) dan **80% Data Testing** (8,238 data) dengan metode stratifikasi untuk menjaga proporsi target tetap seimbang.
4.  **Normalisasi/Scaling**: Melakukan standardisasi fitur numerik menggunakan `StandardScaler` agar perbedaan skala antar kolom (seperti usia puluhan tahun vs saldo ribuan) tidak bias dalam pemodelan.

---

### [SEKSI 5: CRISP-DM - MODELING & EVALUATION]

#### **Metode Modeling**
Pemodelan klasifikasi dilakukan menggunakan dua algoritma:
1.  **Random Forest Classifier**: Menggunakan parameter `class_weight='balanced'` untuk menangani ketimpangan kelas target secara otomatis.
2.  **Logistic Regression**: Sebagai model baseline dengan pembobotan kelas seimbang (`class_weight='balanced'`).

#### **Hasil Evaluasi Model**
*(Unggah gambar `confusion_matrix_rf.png` dan `roc_curve.png`)*

Berikut adalah perbandingan performa kedua model pada data testing (8,238 data):

| Metrik Evaluasi | Random Forest Classifier | Logistic Regression |
|---|---|---|
| **Akurasi Keseluruhan** | **91.87%** | 85.69% |
| **Precision (Kelas Yes)** | **71.08%** | 43.47% |
| **Recall (Kelas Yes)** | 46.88% | **89.98%** |
| **F1-Score (Kelas Yes)** | 56.49% | **58.62%** |
| **Nilai Area Under ROC (AUC)**| **0.9416** | 0.9312 |

*   **Tingkat Akurasi**: Model **Random Forest Classifier** memberikan akurasi tertinggi yaitu **91.87%**, yang sangat memenuhi syarat sukses evaluasi UAS yaitu minimal **80%**.
*   **Analisis Kurva ROC**: Keduanya memiliki performa klasifikasi yang sangat baik dengan skor AUC di atas **0.93**, membuktikan model sangat kokoh dalam membedakan nasabah potensial dan non-potensial.

---

### [SEKSI 6: KESIMPULAN]

1.  Proyek data mining klasifikasi ini berhasil dirancang menggunakan metodologi **CRISP-DM** dengan dataset publik real-world sebesar **41,188 data**.
2.  Model terbaik yang terpilih adalah **Random Forest Classifier** dengan akurasi pengujian sebesar **91.87%** (melebihi standar minimum kelulusan 80%).
3.  Bank direkomendasikan untuk menggunakan model ini guna menyaring daftar telepon kampanye pemasaran, memprioritaskan nasabah yang diprediksi bernilai positif (kelas 1), sehingga meningkatkan efektivitas waktu pemasaran dan menurunkan pemborosan biaya panggilan operasional.
