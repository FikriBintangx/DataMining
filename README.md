# Project Akhir Data Mining: Klasifikasi Deposito Berjangka (Bank Marketing)

Repository ini dibuat untuk memenuhi syarat Ujian Akhir Semester (UAS) Genap TA 2025/2026 pada program studi Sistem Informasi, Institut Teknologi & Bisnis Bina Sarana Global.

---

## 🔗 Tautan Penting
*   **Link Google Sites Portofolio**: [https://sites.google.com/view/isi-link-google-sites-anda](https://sites.google.com/view/isi-link-google-sites-anda) *(Silakan ganti dengan link Google Sites Anda setelah dipublikasikan)*
*   **Link Video Presentasi YouTube**: *(Opsional / Tidak Wajib Sesuai Request Klien)*

---

## 📝 Deskripsi Project
Project ini bertujuan untuk membangun model klasifikasi data mining dengan menggunakan framework **CRISP-DM** untuk memprediksi apakah nasabah bank akan berlangganan deposito berjangka (`y` = 'yes' atau 'no'). 

Pihak bank dapat menggunakan model ini untuk menyaring calon nasabah potensial guna meningkatkan efisiensi kampanye pemasaran langsung (*direct marketing campaign*).

---

## 📂 Struktur Folder & File
*   `bank-additional/` : Folder berisi dataset Bank Marketing (UCI Repository)
    *   `bank-additional-full.csv` : Dataset utama dengan 41.188 baris data
*   `plots/` : Folder berisi grafik visualisasi EDA & Evaluasi Model
    *   `target_distribution.png` : Visualisasi sebaran kelas target
    *   `age_distribution.png` : Distribusi usia nasabah
    *   `job_vs_subscription.png` : Analisis pekerjaan vs langganan deposito
    *   `confusion_matrix_rf.png` : Matriks kebingungan model Random Forest
    *   `roc_curve.png` : Perbandingan kurva ROC-AUC model Random Forest & Logistic Regression
*   `uas_classification.ipynb` : File Jupyter Notebook utama (Anaconda) berisi kode end-to-end CRISP-DM
*   `google_sites_content.md` : Draf teks panduan salin-tempel untuk Google Sites

---

## ⚙️ Cara Menjalankan Kode (Anaconda / Jupyter Notebook)
1. Buka aplikasi **Anaconda Navigator** lalu jalankan **Jupyter Notebook**.
2. Arahkan direktori Jupyter ke folder project ini.
3. Jalankan file `uas_classification.ipynb`.
4. Jalankan setiap sel (*cell*) secara berurutan (`Shift + Enter`).

---

## 📊 Ringkasan Hasil Evaluasi
*   **Model Terbaik**: Random Forest Classifier
*   **Akurasi Model**: **91.87%** (Melebihi standar minimum kelulusan UAS yaitu 80%)
*   **AUC Score**: **0.9416** (Performa klasifikasi sangat baik)
