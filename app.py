import streamlit as st
import pandas as pd
import re

# Import modul dari folder tabs
from tabs.tab_data import render_tab_data
from tabs.tab_sentimen import render_tab_sentimen
from tabs.tab_clustering import render_tab_clustering
from tabs.tab_regresi import render_tab_regresi

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Dashboard Analisis X - GTA VI", layout="wide")

st.title("🎮 Dashboard Analisis Percakapan GTA VI di X (Twitter)")
st.caption("Aplikasi Web Analisis Sentimen, Clustering Teks, dan Regresi Linear Likes")

REQUIRED_COLUMNS = ['text', 'created_at', 'like_count']

def count_words(text):
    return len(str(text).split())

def count_hashtags(text):
    return len(re.findall(r'#\w+', str(text)))

def validate_dataframe(df):
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]

# -----------------------------------------------------------------------------
# SIDEBAR - UNGGAH DATASET
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Unggah Dataset")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV utama", type=["csv"])
uploaded_eval_file = st.sidebar.file_uploader("Pilih file evaluasi_sentimen.csv (opsional)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        missing_cols = validate_dataframe(df)
        
        if missing_cols:
            st.error(f"❌ Kolom wajib berikut tidak ditemukan dalam dataset: {', '.join(missing_cols)}")
            st.stop()
        
        # Preprocessing Sederhana
        if 'cleaned_text' not in df.columns:
            df['cleaned_text'] = df['text'].fillna('')
        else:
            df['cleaned_text'] = df['cleaned_text'].fillna('')
            
        df['word_count'] = df['text'].apply(count_words)
        df['hashtag_count'] = df['text'].apply(count_hashtags)
        df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce')
        
        # Jalankan KMeans clustering di awal untuk mencegah modifikasi/side-effect di dalam tab
        from tabs.tab_clustering import run_kmeans_clustering
        df['cluster'], st.session_state.cluster_terms, st.session_state.cluster_centroids = run_kmeans_clustering(df['cleaned_text'])

        df_valid_likes = df.dropna(subset=['like_count']).copy()
        
        eval_df = None
        if uploaded_eval_file is not None:
            eval_df = pd.read_csv(uploaded_eval_file)

    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")
        st.stop()
else:
    st.info("👋 Silakan unggah file CSV dataset GTA VI pada menu sidebar untuk memulai analisis.")
    st.stop()

# -----------------------------------------------------------------------------
# INISIALISASI TAB
# -----------------------------------------------------------------------------
tab_data, tab_sentiment, tab_clustering, tab_regression = st.tabs([
    "📊 Data", 
    "💬 Sentimen", 
    "🧩 Clustering", 
    "📈 Regresi Linear"
])

with tab_data:
    render_tab_data(df, df_valid_likes, REQUIRED_COLUMNS)

with tab_sentiment:
    render_tab_sentimen(df, eval_df)

with tab_clustering:
    render_tab_clustering(df)

with tab_regression:
    render_tab_regresi(df_valid_likes)

# -----------------------------------------------------------------------------
# BAGIAN AKHIR: TEMUAN & KETERBATASAN
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("📌 Temuan Utama & Keterbatasan")

col_f1, col_f2 = st.columns(2)

with col_f1:
    st.subheader("3 Temuan Utama")
    st.markdown("""
    1. **Dominasi Sentimen Antusias:** Mayoritas postingan tentang GTA VI menunjukkan sentimen positif yang tinggi dipicu oleh antusiasme terhadap *gameplay* dan tanggal rilis.
    2. **Fokus Pembahasan Tersegmentasi:** Hasil clustering menunjukkan percakapan netizen terbagi jelas antara spekulasi jadwal rilis, analisa spesifikasi grafik, dan konten humor/memes.
    3. **Lemahnya Hubungan Kata & Likes:** Jumlah kata maupun jumlah hashtag tidak memiliki korelasi linear yang kuat terhadap jumlah *likes* yang diterima postingan.
    """)

with col_f2:
    st.subheader("2 Keterbatasan Data / Metode")
    st.markdown("""
    1. **Keterbatasan Fitur Regresi:** Model regresi hanya memakai 2 fitur teks tanpa memperhitungkan variabel kunci lain seperti jangkauan akun (*followers*), ketersediaan media (gambar/video), dan *engagement rate*.
    2. **Bias Bahasa dan Sarcasm:** Metode analisis sentimen dan TF-IDF masih kesulitan mendeteksi sarkasme serta istilah/slang khas komunitas *gaming*.
    """)