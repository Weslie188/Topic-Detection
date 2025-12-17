import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Topik Blog (LDA)", layout="centered")

# --- 2. DOWNLOAD RESOURCE NLTK ---
# Kita perlu mendownload resource ini agar preprocessing berjalan
# (Sesuai tahap Tokenization & Stopwords removal di laporan)
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

download_nltk_resources()

# --- 3. FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    # Memuat file model yang Anda unggah
    # Pastikan nama file sesuai dengan yang ada di folder
    model = joblib.load('lda_bow_logreg_best.joblib')
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("File 'lda_bow_logreg_best (1).joblib' tidak ditemukan. Pastikan file ada di folder yang sama.")
    st.stop()

# --- 4. FUNGSI PREPROCESSING (Sesuai BAB III Laporan) ---
def preprocess_text(text):
    # a. Lowercase 
    text = text.lower()
    
    # b. Remove Special Characters 
    # Menghapus angka, simbol, tanda baca
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # c. Tokenization [cite: 130]
    tokens = text.split()
    
    # d. Remove Stopwords [cite: 134]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # e. Lemmatization 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Gabungkan kembali menjadi string karena Pipeline Sklearn menerima input string
    return " ".join(tokens)

# --- 5. TAMPILAN ANTARMUKA (UI) ---
st.title("🔍 Klasifikasi Topik Blog")
st.markdown("""
Aplikasi ini menggunakan model **Latent Dirichlet Allocation (LDA)** yang telah di-tuning untuk memprediksi kategori topik dari teks blog.
""")

st.write("---")

# Area Input Teks
input_text = st.text_area("Masukkan Teks Blog (Bahasa Inggris):", height=200, placeholder="Write your blog content here...")

# Tombol Prediksi
if st.button("Prediksi Topik"):
    if input_text:
        with st.spinner('Sedang memproses teks...'):
            # 1. Tampilkan teks asli
            st.subheader("Teks Asli")
            st.info(input_text)
            
            # 2. Lakukan Preprocessing
            clean_text = preprocess_text(input_text)
            
            # (Opsional) Tampilkan hasil cleaning untuk debug/demonstrasi
            with st.expander("Lihat Hasil Preprocessing (Clean Text)"):
                st.text(clean_text)
            
            # 3. Prediksi menggunakan Model
            # Model pipeline sudah berisi Vectorizer -> LDA -> Classifier
            try:
                prediction = model.predict([clean_text])[0]
                
                # Menampilkan Hasil
                st.success(f"Topik Diprediksi: **{prediction}**")
                
                # Penjelasan Tambahan (Opsional)
                st.markdown(f"""
                *Model mengklasifikasikan teks ini ke dalam kategori **{prediction}** berdasarkan pola kata yang ditemukan.*
                """)
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
                st.warning("Kemungkinan versi scikit-learn Anda berbeda dengan versi saat training.")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")

# Footer
st.markdown("---")
st.caption("Dibuat berdasarkan Laporan UAS NLP - Universitas Bunda Mulia")