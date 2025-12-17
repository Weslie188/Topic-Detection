# 📝 Blog Topic Classification with NLP

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange.svg)

Aplikasi berbasis web ini dikembangkan untuk mengklasifikasikan topik dari postingan blog secara otomatis menggunakan teknik **Natural Language Processing (NLP)**. [cite_start]Proyek ini dibangun sebagai bagian dari **Laporan Ujian Akhir Semester (UAS)** mata kuliah NLP di **Universitas Bunda Mulia**.

[cite_start]Aplikasi ini menggunakan model *Machine Learning* yang dilatih pada dataset **Blog Authorship Corpus**[cite: 2, 45].

## 👥 Tim Penyusun
[cite_start]Proyek ini disusun oleh mahasiswa Program Studi Data Sains[cite: 8]:

| NIM | Nama |
| :--- | :--- |
| **36220017** | [cite_start]Denis Winata [cite: 4] |
| **36230003** | [cite_start]Hans Christian Wijaya [cite: 5] |
| **36230009** | [cite_start]Ferdiantono [cite: 6] |
| **36230013** | [cite_start]Weslie Austin [cite: 7] |

## 🔍 Gambaran Proyek

### Latar Belakang
Banyaknya data teks yang tersedia di internet, khususnya blog, menjadikan proses analisis manual tidak efisien. [cite_start]Penelitian ini bertujuan membangun model otomatis untuk mengelompokkan dokumen ke dalam kategori topik tertentu[cite: 18, 20].

### Dataset
Dataset yang digunakan adalah **Blog Authorship Corpus** yang terdiri dari postingan blog dari ribuan penulis. [cite_start]Untuk optimalisasi model, 40 kategori topik asli digeneralisasi menjadi **7 Topik Utama**[cite: 76, 77]:
1. Education & Academia
2. Technology & Internet
3. Media, Arts & Culture
4. Business & Finance
5. Government & Law
6. Environment & Social
7. Industry & Infrastructure

### Metodologi
[cite_start]Proyek ini membandingkan dua pendekatan ekstraksi fitur[cite: 147, 154]:
1.  **LSA (Latent Semantic Analysis):** Menggunakan TF-IDF dan TruncatedSVD.
2.  **LDA (Latent Dirichlet Allocation):** Menggunakan CountVectorizer dan model probabilistik LDA.

Kedua metode tersebut digunakan sebagai fitur untuk klasifikasi menggunakan algoritma **Logistic Regression**.

---

## 🚀 Cara Menjalankan Aplikasi

Aplikasi ini dibangun menggunakan **Streamlit**. Karena adanya ketergantungan pada versi NumPy lawas (untuk kompatibilitas model `.joblib`), disarankan menggunakan **Python 3.9**.

### 1. Instalasi Lokal

Pastikan Anda memiliki Python (disarankan v3.9) dan jalankan perintah berikut:

```bash
# Clone repository ini
git clone [https://github.com/username-anda/nama-repo.git](https://github.com/username-anda/nama-repo.git)

# Masuk ke direktori
cd nama-repo

# Install dependencies (PENTING: Gunakan requirements.txt yang disediakan)
pip install -r requirements.txt