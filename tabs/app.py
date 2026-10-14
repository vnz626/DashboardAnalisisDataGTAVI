import os
import re
import pandas as pd
import streamlit as st

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
# SIDEBAR - PENGATURAN DATASET
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Pengaturan Dataset")

uploaded_main_file = st.sidebar.file_uploader("Unggah Dataset Baru (Opsional)", type=["csv"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_CSV_PATH = os.path.join(BASE_DIR, "data", "gta6_clean.csv")
EVAL_CSV_PATH = os.path.join(BASE_DIR, "data", "evaluasi_sentimen.csv")

# Memuat Dataset Utama
df_raw = None
filename = ""

if uploaded_main_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_main_file)
        filename = f"{uploaded_main_file.name} (Unggahan)"
        st.sidebar.success(f"📌 Menggunakan dataset terunggah: `{uploaded_main_file.name}`")
    except Exception as e:
        st.error(f"Gagal membaca file CSV unggahan: {e}")
        st.stop()
else:
    if os.path.exists(MAIN_CSV_PATH):
        df_raw = pd.read_csv(MAIN_CSV_PATH)
        filename = "gta6_clean.csv (Default)"
        st.sidebar.info("📁 Menggunakan dataset default:\n`data/gta6_clean.csv`")
    else:
        st.error("❌ File default `data/gta6_clean.csv` tidak ditemukan dan belum ada file terunggah!")
        st.stop()

# Memuat Dataset Evaluasi Sentimen
eval_df = None
if os.path.exists(EVAL_CSV_PATH):
    try:
        eval_df = pd.read_csv(EVAL_CSV_PATH)
        st.sidebar.success("✅ Terdeteksi `data/evaluasi_sentimen.csv`")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca `data/evaluasi_sentimen.csv`: {e}")
else:
    st.sidebar.warning("⚠️ `data/evaluasi_sentimen.csv` belum ditemukan.")

# Validasi Kolom Wajib
missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_raw.columns]
if missing_cols:
    st.error(f"❌ Dataset `{filename}` tidak memiliki kolom wajib: {', '.join(missing_cols)}")
    st.stop()

# Preprocessing Dataset
df = df_raw.copy()
if 'cleaned_text' not in df.columns:
    df['cleaned_text'] = df['clean_text'] if 'clean_text' in df.columns else df['text']

df['word_count'] = df['text'].apply(count_words)
df['hashtag_count'] = df['text'].apply(count_hashtags)
df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce')
df_valid_likes = df.dropna(subset=['like_count']).copy()

st.sidebar.metric("Total Postingan", len(df))

# -----------------------------------------------------------------------------
# NAVIGASI TAB
# -----------------------------------------------------------------------------
tab_data, tab_sentiment, tab_clustering, tab_regression = st.tabs([
    "📊 Data", 
    "💬 Sentimen", 
    "🧩 Clustering", 
    "📈 Regresi Linear"
])

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
    1. **Ketimpangan Ekstrem Engagement (*Outliers*):** Distribusi *like_count* sangat timpang, di mana 75% postingan hanya memperoleh ≤ 4 likes, sementara terdapat postingan viral yang mencapai 31.827 likes.
    2. **Topik Didominasi Obrolan Umum:** Sebanyak 315 dari 383 postingan (82%) berupa obrolan umum. Dua tema khusus yang menonjol adalah pengumuman lagu soundtrack Cardi B "Don't Chart" (47 postingan) dan meme "We got … before GTA 6" (21 postingan).
    3. **Tidak Ada Korelasi Linear Teks Terhadap Popularitas:** Jumlah kata dan hashtag tidak dapat memperkirakan jumlah likes ($R^2 = -0,0090$ dan MAE = 516,60), menunjukkan bahwa popularitas postingan dipengaruhi faktor lain di luar teks, seperti akun pengunggah.
    """)

with col_f2:
    st.subheader("2 Keterbatasan Data / Metode")
    st.markdown("""
    1. **Keterbatasan Model Regresi Linear Biasa:** Regresi linear tanpa transformasi data (seperti *log transform*) gagal menangani pola data likes yang berdistribusi *power-law* dan sarat akan pencilan ekstrem. Data juga hanya mencakup sekitar 3,5 jam pada 9 Oktober 2026.
    2. **Kelemahan Deteksi Konteks pada Sentimen:** Pengklasifikasi berbasis kamus (VADER) kesulitan membedakan teks netral berisi berita dengan ekspresi opini, serta tidak memahami judul lagu, kritik halus, dan slang, sehingga kecocokan dengan label manual hanya 11 dari 20 sampel (55%).
    """)