# 🧠 RuangRasa: Mental Well-being Analytics & AI Platform ✨

Selamat datang di repositori **RuangRasa**! Proyek ini merupakan hasil akhir (*Capstone Project*) untuk kelulusan program studi independen **Dicoding: Data Science Path** (Team Capstone CC26-PSU309).

Dashboard interaktif ini dibangun menggunakan **Streamlit** untuk memvisualisasikan data eksplorasi terkait kesehatan mental, yang bersumber dari **9.700 data jurnal emosi tekstual** dan **10.000 data rekam evaluasi screening**. Selain analitik, platform ini juga mendemonstrasikan implementasi **Artificial Intelligence (NLP)** untuk analisis sentimen dan klasifikasi emosi secara *real-time*.

## 🎯 Business Questions Answered
Proyek ini dirancang untuk menjawab serangkaian pertanyaan strategis yang dibagi ke dalam dua domain utama:

**Dari Analisis Jurnal Teks Emosi (`capstone_jurnal.ipynb`):**
1. Bagaimana distribusi proporsi emosi pengguna dalam data journaling, dan emosi apa yang paling dominan terekam?
2. Apakah terdapat perbedaan panjang teks (jumlah kata) antar kategori emosi, dan emosi mana yang cenderung menghasilkan curhatan lebih panjang?
3. Kata-kata (leksikon) apa yang paling sering muncul pada masing-masing emosi, dan tema apa yang paling dominan dalam curhatan pengguna?

**Dari Analisis Evaluasi Screening (`capstone_screening.ipynb`):**
1. **Keandalan Sistem:** Mampukah model menangani ketidakseimbangan data awal secara adil dan menghasilkan keandalan deteksi (*Recall*) di atas 90%?
2. **Faktor Risiko Utama:** Faktor perilaku sehari-hari dan gejala spesifik apa saja yang paling dominan menjadi indikator utama masalah kesehatan mental?
3. **Efisiensi Pengguna:** Mungkinkah mengoptimalkan *user experience* dengan memangkas jumlah pertanyaan kuesioner awal (dari 50 pertanyaan) menjadi jauh lebih ringkas tanpa mengorbankan tingkat akurasi sistem?

## 📂 Struktur Proyek
- `dashboard/`: Direktori utama untuk aplikasi Streamlit.
  - `views/`: Halaman-halaman modular dari dashboard (*Home*, *Analytics*, dan *AI Lab*).
  - `utils/`: Skrip pendukung (*helper*) untuk pemrosesan data, konfigurasi grafik Plotly, dan AI *inference logic*.
  - `models/`: Tempat menyimpan file *pre-trained* model Machine Learning (`.h5` / arsitektur jaringan saraf tiruan).
  - `data/`: Dataset *clean* yang telah melalui proses wrangling/cleaning dan siap disajikan ke dalam visualisasi.
- `capstone_jurnal.ipynb` & `capstone_screening.ipynb`: Berkas Jupyter Notebook yang mencakup keseluruhan alur kerja *Data Science* (mulai dari *Gathering*, *Wrangling*, EDA, hingga mendefinisikan pertanyaan bisnis). *(Catatan: Proses pengembangan model AI dilakukan oleh tim AI Engineer`)*.

## 🛠️ Setup Environment

Gunakan panduan di bawah ini untuk menyiapkan *environment* pengembangan di komputer lokal Anda.

### Menggunakan Anaconda (Rekomendasi)
```bash
conda create --name .env python=3.12
conda activate .env
pip install -r requirements.txt
```

### Menggunakan Python venv
```bash
python -m venv .env

# Mengaktifkan environment (Windows PowerShell):
.env\Scripts\Activate.ps1

# Mengaktifkan environment (Mac/Linux):
source .env/bin/activate

pip install -r requirements.txt
```

## 🚀 Run Streamlit App
Setelah environment terpasang dan semua *dependencies* (seperti `streamlit`, `pandas`, `plotly`, `tensorflow-cpu`) terunduh, Anda dapat menjalankan dashboard secara lokal dengan perintah berikut:

```bash
# Pastikan Anda berada di root folder proyek
streamlit run dashboard/dashboard.py
```
*Aplikasi akan otomatis terbuka di browser melalui `http://localhost:8501`*

## 🌐 Live Dashboard
Aplikasi analitik ini telah di-deploy secara publik melalui **Streamlit Community Cloud**. Anda dapat mengakses seluruh fungsionalitasnya (termasuk fitur interaktif AI Lab) melalui tautan berikut:

👉 **[Akses Dashboard RuangRasa di Sini](https://capstone-dicoding-data-science-path.streamlit.app/)**

---
_Created with ❤️ by **Dary Ihsan Amanullah** - Dicoding Capstone CC26-PSU309 (2026)_
