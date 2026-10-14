## 📌 Fitur Utama Aplikasi

Aplikasi terdiri dari 4 tab utama dan 1 bagian ringkasan di bawahnya.

1. 📊 **Tab Data**
   * Unggah dataset CSV (opsional); jika kosong, aplikasi membaca `data/gta6_clean.csv`.
   * Pemeriksaan kolom wajib (`text`, `created_at`, `like_count`) beserta pesan kesalahan bila kolom tidak lengkap atau file tidak dapat dibaca.
   * Ringkasan data: jumlah postingan, jumlah likes valid, dan rentang tanggal postingan.
   * Pratinjau 50 baris pertama (teks asli dan teks bersih).
2. 💬 **Tab Sentimen**
   * Klasifikasi sentimen dengan VADER (Positif, Netral, Negatif).
   * Jumlah dan persentase tiap kategori.
   * Grafik batang distribusi sentimen.
   * Tabel hasil klasifikasi beserta skor *compound*, dapat difilter per kategori.
   * Ringkasan pemeriksaan manual 20 postingan acak (`evaluasi_sentimen.csv`): jumlah dan persentase yang sesuai, serta tabel postingan yang tidak sesuai.
3. 🧩 **Tab Clustering**
   * Pengelompokan teks menjadi 3 cluster dengan TF-IDF dan K-Means (`random_state=42`).
   * Grafik ukuran tiap cluster.
   * Lima kata dominan dan word cloud untuk tiap cluster.
   * Tiga contoh postingan yang paling mewakili tiap cluster.
   * Nama tema tiap cluster yang ditentukan kelompok setelah membaca kata dominan dan contoh postingan.
   * Interpretasi perbedaan dan tumpang tindih antar-cluster.
4. 📈 **Tab Regresi**
   * Regresi linear untuk memperkirakan jumlah likes dari jumlah kata dan jumlah hashtag.
   * Pembagian 80% data latih dan 20% data uji dengan `random_state=42`.
   * Evaluasi dengan MAE (*Mean Absolute Error*) dan R² pada data uji, beserta persamaan regresinya.
   * Grafik scatter likes aktual vs prediksi, dengan pilihan skala logaritmik.
   * Heatmap korelasi jumlah kata, jumlah hashtag, dan jumlah likes (Pearson atau Spearman).
   * Form estimasi jumlah likes, dengan peringatan bila hasil estimasi negatif.
5. 📌 **Bagian Akhir: Temuan & Keterbatasan**
   * Tiga temuan utama dari hasil analisis.
   * Dua keterbatasan data atau metode.
