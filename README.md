# 🎮 Dashboard Analisis Data X (Twitter) - GTA VI

Aplikasi web interaktif berbasis **Streamlit** untuk menganalisis percakapan publik mengenai **Grand Theft Auto VI (GTA VI)** di platform X (Twitter). Proyek ini dibuat untuk memenuhi Tugas Teori dan Praktik Aplikasi Web.

---

## 📌 Fitur Utama Aplikasi

Aplikasi ini dibagi menjadi 4 Tab utama dan 1 bagian ringkasan analisis:

1. **📊 Tab 1: Data**
   * Pratinjau dataset `gta6_clean.csv`.
   * Validasi keberadaan kolom wajib (`text`, `created_at`, `like_count`).
   * Statistik ringkas data (total postingan, data likes valid, dan jumlah kolom).
2. **💬 Tab 2: Sentimen**
   * Distribusi frekuensi dan persentase kategori sentimen (Positif, Netral, Negatif).
   * Visualisasi grafik *pie chart* interaktif.
   * Tabel detail hasil klasifikasi teks.
   * Modul evaluasi manual 20 sampel postingan (`evaluasi_sentimen.csv`) beserta kalkulasi persentase kesesuaian/akurasi.
3. **🧩 Tab 3: Clustering Teks**
   * Pengelompokan teks menjadi 3 *cluster* menggunakan kombinasi **TF-IDF Vectorizer** dan **K-Means Clustering**.
   * Identifikasi 5 kata kunci dominan per *cluster*.
   * Penamaan tema otomatis berdasarkan isi pembahasan tiap *cluster*.
   * Penjelasan interpretasi perbedaan dan tumpang tindih tema.
4. **📈 Tab 4: Regresi Linear**
   * Pemodelan Regresi Linear Sederhana untuk memperkirakan jumlah *likes* berdasarkan **Jumlah Kata** dan **Jumlah Hashtag**.
   * Pembagian data 80% Data Latih (*Train*) dan 20% Data Uji (*Test*) dengan `random_state=42`.
   * Evaluasi performa model menggunakan metrik **MAE (Mean Absolute Error)** dan **$R^2$ Score**.
   * Visualisasi grafik *scatter plot* perbandingan Likes Aktual vs Prediksi.
   * Form interaktif simulasi estimasi jumlah *likes*.
5. **📌 Bagian Akhir: Temuan & Keterbatasan**
   * Ringkasan 3 temuan utama dari hasil eksplorasi data.
   * Ringkasan 2 keterbatasan data atau metode analisis yang digunakan.

---

## 📁 Struktur Folder Proyek

```text
Kelompok_GTAVI/
├── app.py                      # Berkas utama aplikasi Streamlit
├── requirements.txt            # Daftar dependensi library Python
├── README.md                   # Petunjuk instalasi dan penggunaan
├── laporan.pdf                 # Laporan ringkas tugas
├── data/                       # Folder penyimpan dataset
│   ├── gta6_clean.csv          # Dataset utama percakapan GTA VI
│   └── evaluasi_sentimen.csv   # Dataset 20 sampel evaluasi manual
└── tabs/                       # Modul halaman tab Streamlit
    ├── __init__.py             # Inisialisasi paket Python
    ├── tab_data.py             # Tampilan & logika Tab Data
    ├── tab_sentimen.py         # Tampilan & logika Tab Sentimen
    ├── tab_clustering.py       # Tampilan & logika Tab Clustering
    └── tab_regresi.py          # Tampilan & logika Tab Regresi Linear
```

---

## 🛠️ Prasyarat (Requirements)

* **Python** versi 3.8 atau yang lebih baru.
* Library Python utama:
  * `streamlit`
  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `plotly`

---

## 🚀 Petunjuk Instalasi & Menjalankan Aplikasi

Lakukan langkah-langkah berikut untuk menjalankan aplikasi secara lokal:

### 1. Ekstrak Berkas Proyek
Ekstrak berkas ZIP proyek ke direktori pilihan kamu dan buka terminal/Command Prompt di direktori tersebut.

### 2. (Opsional) Buat Virtual Environment
Disarankan menggunakan *virtual environment* agar dependensi terisolasi:

* **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Instal Dependensi
Jalankan perintah berikut untuk menginstal seluruh pustaka yang dibutuhkan:

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi Streamlit
Jalankan perintah Streamlit berikut:

```bash
streamlit run app.py
```

Aplikasi secara otomatis akan terbuka di peramban (browser) web kamu pada alamat default: `http://localhost:8501`.

---

## 📄 Penjelasan Dataset (`data/`)

* **`data/gta6_clean.csv`**: Berisi data postingan unik dari X terkait GTA VI dengan kolom minimal `text`, `clean_text`, `created_at`, dan `like_count`.
* **`data/evaluasi_sentimen.csv`**: Berisi 20 sampel postingan acak yang dilabeli manual (`label_manual`) dan dibandingkan dengan label sistem (`label_sistem`).
* **Fitur Unggah Baru (Opsional):** Kamu juga dapat mengunggah dataset CSV baru melalui bilah samping (*sidebar*) jika ingin menguji dataset lain secara dinamis.

---

## 💡 Catatan Evaluasi & Temuan Utama

1. **Distribusi Engagement (Likes) Timpang:** Sebagian besar postingan memiliki jumlah *likes* kecil, sementara hanya sedikit postingan viral yang memiliki *likes* sangat tinggi (*right-skewed distribution*).
2. **Keterbatasan Regresi Linear:** Fitur jumlah kata dan hashtag belum cukup kuat untuk memprediksi jumlah *likes* secara akurat ($R^2$ mendekati 0 atau negatif), karena popularitas postingan lebih dipengaruhi oleh faktor eksternal seperti jangkauan akun (*followers*) dan usia postingan.