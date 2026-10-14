import os
import re
import pandas as pd
import streamlit as st

# Import fungsi render dari folder tabs
from tabs.tab_data import render_tab_data
from tabs.tab_sentimen import render_tab_sentimen
from tabs.tab_clustering import render_tab_clustering
from tabs.tab_regresi import render_tab_regresi

st.set_page_config(page_title="Dashboard Analisis X - GTA VI", layout="wide")

st.title("🎮 Dashboard Analisis Percakapan GTA VI di X (Twitter)")
st.caption("Aplikasi Web Analisis Sentimen, Clustering Teks, dan Regresi Linear Likes")

REQUIRED_COLUMNS = ['text', 'created_at', 'like_count']

def count_words(text):
    return len(str(text).split())

def count_hashtags(text):
    return len(re.findall(r'#\w+', str(text)))

# -----------------------------------------------------------------------------
# MEMUAT DATASET (data/gta6_clean.csv & data/evaluasi_sentimen.csv)
# -----------------------------------------------------------------------------
MAIN_CSV_PATH = os.path.join("data", "gta6_clean.csv")
EVAL_CSV_PATH = os.path.join("data", "evaluasi_sentimen.csv")

@st.cache_data
def load_main_data():
    if not os.path.exists(MAIN_CSV_PATH):
        st.error(f"❌ File `{MAIN_CSV_PATH}` tidak ditemukan! Pastikan file disimpan di dalam folder `data/`.")
        st.stop()
    return pd.read_csv(MAIN_CSV_PATH)

@st.cache_data
def load_eval_data():
    if os.path.exists(EVAL_CSV_PATH):
        return pd.read_csv(EVAL_CSV_PATH)
    return None

# Load Datasets
df_raw = load_main_data()
eval_df = load_eval_data()

# Validasi Kolom Wajib
missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_raw.columns]
if missing_cols:
    st.error(f"❌ File `{MAIN_CSV_PATH}` tidak memiliki kolom wajib: {', '.join(missing_cols)}")
    st.stop()

# Preprocessing Dataset Utama
df = df_raw.copy()
if 'cleaned_text' not in df.columns:
    df['cleaned_text'] = df['text']

df['word_count'] = df['text'].apply(count_words)
df['hashtag_count'] = df['text'].apply(count_hashtags)
df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce')
df_valid_likes = df.dropna(subset=['like_count']).copy()

# Sidebar Info
st.sidebar.header("ℹ️ Status Dataset")
st.sidebar.success(f"📁 Dataset Utama:\n`{MAIN_CSV_PATH}`")
st.sidebar.metric("Total Postingan", len(df))

if eval_df is not None:
    st.sidebar.success(f"✅ `{EVAL_CSV_PATH}` terdeteksi.")
else:
    st.sidebar.warning(f"⚠️ `{EVAL_CSV_PATH}` belum ditemukan.")

# -----------------------------------------------------------------------------
# TAB NAVIGATION
# -----------------------------------------------------------------------------
tab_data, tab_sentiment, tab_clustering, tab_regression = st.tabs([
    "📊 Data", 
    "💬 Sentimen", 
    "🧩 Clustering", 
    "📈 Regresi Linear"
])

filename = os.path.basename(MAIN_CSV_PATH)

with tab_data:
    render_tab_data(df, df_valid_likes, REQUIRED_COLUMNS, filename)

with tab_sentiment:
    render_tab_sentimen(df, eval_df)

with tab_clustering:
    render_tab_clustering(df)

with tab_regression:
    render_tab_regresi(df_valid_likes)

# -----------------------------------------------------------------------------
# TEMUAN UTAMA & KETERBATASAN
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