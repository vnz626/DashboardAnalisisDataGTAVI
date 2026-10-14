"""
Dashboard Analisis Postingan X tentang GTA 6 Menjelang Perilisan
Tugas Teori dan Praktik Aplikasi Web

Jalankan:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as an

try:
    from wordcloud import WordCloud
except ImportError:  # aplikasi tetap jalan walau pustaka belum terpasang
    WordCloud = None

BASE_DIR = Path(__file__).parent
DEFAULT_DATA = BASE_DIR / "data" / "gta6_clean.csv"
DEFAULT_EVAL = BASE_DIR / "data" / "evaluasi_sentimen.csv"

# ===========================================================================
# ISI BAGIAN INI SETELAH ANALISIS SELESAI
# Semua teks di bawah harus ditulis kelompok berdasarkan hasil data kalian.
# Jangan mengarang: baca dulu hasil di tiap tab, baru isi.
# ===========================================================================

# Nama tema tiap cluster, ditentukan SETELAH melihat kata dominan & contoh postingan.
CLUSTER_NAMES = {
    0: "TODO: nama tema cluster 0",
    1: "TODO: nama tema cluster 1",
    2: "TODO: nama tema cluster 2",
}

# Penjelasan singkat perbedaan antar-cluster dan tumpang tindihnya.
CLUSTER_INTERPRETATION = "TODO: jelaskan apa yang membedakan ketiga cluster dan apakah masih ada pembahasan yang tumpang tindih."

# Interpretasi hasil regresi.
REGRESSION_INTERPRETATION = "TODO: jelaskan makna MAE, makna R² (termasuk jika negatif), dan keterbatasan dua fitur ini."

# 3 temuan utama dan 2 keterbatasan.
FINDINGS = [
    "TODO: temuan 1",
    "TODO: temuan 2",
    "TODO: temuan 3",
]
LIMITATIONS = [
    "TODO: keterbatasan 1",
    "TODO: keterbatasan 2",
]

# ===========================================================================

SENTIMENT_COLORS = {"Positif": "#2e9d5b", "Netral": "#8a8f98", "Negatif": "#d1453b"}
SENTIMENT_ORDER = ["Positif", "Netral", "Negatif"]


def show_text_or_todo(text: str):
    if text.strip().startswith("TODO"):
        st.warning(f"Belum diisi kelompok: {text}")
    else:
        st.write(text)


# ---------------------------------------------------------------------------
st.set_page_config(page_title="Analisis X: GTA 6", layout="wide")
st.title("Dashboard Analisis Postingan X tentang GTA 6")
st.caption("Analisis sentimen, clustering teks, dan regresi linear pada postingan X menjelang perilisan GTA 6.")


@st.cache_data
def load_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


@st.cache_data
def compute_all(df: pd.DataFrame):
    df = an.prepare_dataframe(df)
    df = an.add_sentiment(df)
    return df


# ---------------------------------------------------------------------------
# Muat data (dipakai oleh semua tab)
# ---------------------------------------------------------------------------
tab_data, tab_sent, tab_clust, tab_reg = st.tabs(["Data", "Sentimen", "Clustering", "Regresi"])

with tab_data:
    st.subheader("Dataset")
    uploaded = st.file_uploader("Unggah dataset CSV (kosongkan untuk memakai data/gta6_clean.csv)", type="csv")

raw_df = None
try:
    if uploaded is not None:
        raw_df = load_csv(uploaded)
    elif DEFAULT_DATA.exists():
        raw_df = load_csv(DEFAULT_DATA)
except Exception as exc:  # file rusak, bukan CSV, encoding salah, dll.
    with tab_data:
        st.error(f"File tidak dapat dibaca sebagai CSV: {exc}")
    st.stop()

if raw_df is None:
    with tab_data:
        st.info("Belum ada data. Unggah file CSV atau letakkan dataset di data/gta6_clean.csv.")
    st.stop()

missing = an.check_required_columns(raw_df)
with tab_data:
    st.write("**Pemeriksaan kolom wajib**")
    st.dataframe(
        pd.DataFrame({
            "Kolom": an.REQUIRED_COLUMNS,
            "Status": ["Ada" if c in raw_df.columns else "Tidak ada" for c in an.REQUIRED_COLUMNS],
        }),
        hide_index=True,
    )
if missing:
    with tab_data:
        st.error(f"Kolom wajib tidak ditemukan: {', '.join(missing)}. Analisis tidak dapat dilanjutkan.")
    st.stop()

if raw_df["text"].isna().all() or len(raw_df) == 0:
    with tab_data:
        st.error("Dataset kosong atau kolom text tidak berisi data.")
    st.stop()

try:
    df = compute_all(raw_df)
except Exception as exc:
    with tab_data:
        st.error(f"Data tidak dapat diproses: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Tab Data
# ---------------------------------------------------------------------------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah postingan", len(df))
    c2.metric("Likes valid", int(df["like_count"].notna().sum()))
    created = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if created.notna().any():
        c3.metric("Rentang tanggal", f"{created.min():%d %b %Y} – {created.max():%d %b %Y}")
    st.write("**Pratinjau dataset**")
    st.dataframe(df[["text", "clean_text", "created_at", "like_count"]].head(50), width="stretch")


# ---------------------------------------------------------------------------
# Tab Sentimen
# ---------------------------------------------------------------------------
with tab_sent:
    st.subheader("Analisis Sentimen (VADER)")
    counts = df["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    pct = counts / counts.sum() * 100

    cols = st.columns(3)
    for col, label in zip(cols, SENTIMENT_ORDER):
        col.metric(label, f"{counts[label]} postingan", f"{pct[label]:.1f}%", delta_color="off")

    fig = px.bar(
        x=counts.index, y=counts.values, color=counts.index,
        color_discrete_map=SENTIMENT_COLORS,
        labels={"x": "Sentimen", "y": "Jumlah postingan"},
        text=[f"{v} ({p:.1f}%)" for v, p in zip(counts.values, pct.values)],
    )
    fig.update_layout(showlegend=False, title="Distribusi sentimen")
    st.plotly_chart(fig, width="stretch")

    st.write("**Tabel hasil klasifikasi**")
    pilihan = st.multiselect("Filter sentimen", SENTIMENT_ORDER, default=SENTIMENT_ORDER)
    st.dataframe(
        df[df["sentiment"].isin(pilihan)][["text", "sentiment", "sentiment_score"]],
        width="stretch",
    )

    st.write("**Ringkasan pemeriksaan manual (20 postingan acak)**")
    if DEFAULT_EVAL.exists():
        try:
            ev = an.evaluate_manual_labels(load_csv(DEFAULT_EVAL))
            st.write(f"Sesuai: **{ev['n_match']} dari {ev['n']} postingan ({ev['pct_match']:.1f}%)**")
            st.write("Postingan yang tidak sesuai:")
            st.dataframe(ev["mismatches"][["text", "label_manual", "prediksi_sistem"]], width="stretch")
            st.caption("Evaluasi terbatas pada 20 sampel dan belum menggambarkan akurasi pada seluruh percakapan di X.")
        except Exception as exc:
            st.error(f"evaluasi_sentimen.csv tidak dapat diproses: {exc}")
    else:
        st.info("File data/evaluasi_sentimen.csv belum ada. Buat lewat notebook setelah pelabelan manual.")

# ---------------------------------------------------------------------------
# Tab Clustering
# ---------------------------------------------------------------------------
with tab_clust:
    st.subheader("Clustering Teks (TF-IDF + K-Means, 3 cluster)")
    try:
        clustered, summary, excl = an.run_clustering(df)
    except Exception as exc:
        st.error(f"Clustering gagal dijalankan: {exc}")
        summary = None
    if summary:
        if excl:
            st.caption(f"{excl} postingan dikeluarkan karena teks bersihnya kosong.")
        sizes = pd.DataFrame({
            "Cluster": [f"{s['cluster']}: {CLUSTER_NAMES.get(s['cluster'], '')}" for s in summary],
            "Jumlah": [s["jumlah"] for s in summary],
        })
        st.plotly_chart(
            px.bar(sizes, x="Cluster", y="Jumlah", text="Jumlah", title="Ukuran cluster"),
            width="stretch",
        )
        for s in summary:
            with st.container(border=True):
                st.markdown(f"#### Cluster {s['cluster']}: {CLUSTER_NAMES.get(s['cluster'], '')}")
                st.write(f"Jumlah postingan: **{s['jumlah']}**")
                st.write("Lima kata dominan: " + ", ".join(f"`{w}`" for w in s["kata_dominan"]))
                freq = an.cluster_word_frequencies(clustered, s["cluster"])
                if WordCloud is not None and freq:
                    wc = WordCloud(width=900, height=320, background_color="white",
                                   colormap="viridis", random_state=an.RANDOM_STATE)
                    st.image(wc.generate_from_frequencies(freq).to_image(), width="stretch",
                             caption="Word cloud: makin besar kata, makin sering muncul di cluster ini")
                elif WordCloud is None:
                    st.caption("Pasang pustaka `wordcloud` untuk menampilkan word cloud.")
                st.write("Contoh postingan:")
                for ex in s["contoh"]:
                    st.markdown(f"> {ex}")
        st.write("**Interpretasi**")
        show_text_or_todo(CLUSTER_INTERPRETATION)

# ---------------------------------------------------------------------------
# Tab Regresi
# ---------------------------------------------------------------------------
with tab_reg:
    st.subheader("Regresi Linear: Jumlah Kata & Hashtag → Likes")
    try:
        reg = an.run_regression(df)
    except Exception as exc:
        st.error(f"Regresi gagal dijalankan: {exc}")
        reg = None
    if reg:
        if reg["excluded"]:
            st.caption(f"{reg['excluded']} postingan dikeluarkan karena jumlah likes kosong atau tidak valid (tidak diganti nol).")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (data uji)", f"{reg['mae']:.2f} likes")
        c2.metric("R² (data uji)", f"{reg['r2']:.3f}")
        c3.metric("Data latih / uji", f"{reg['n_train']} / {reg['n_test']}")

        m = reg["model"]
        st.latex(
            rf"\hat{{likes}} = {m.intercept_:.2f} + ({m.coef_[0]:.2f})\times kata + ({m.coef_[1]:.2f})\times hashtag"
        )

        res = reg["test_result"].copy()
        log_scale = st.toggle(
            "Skala logaritmik", value=True,
            help="Jumlah likes sangat timpang (sebagian kecil postingan viral). "
                 "Skala log membuat postingan dengan likes kecil tetap terlihat.",
        )
        if log_scale:
            # Sumbu log tidak bisa menampilkan 0, jadi semua nilai digeser +1.
            res["x"] = res["likes_aktual"] + 1
            res["y"] = res["likes_prediksi"].clip(lower=0) + 1
            x_label, y_label = "Likes aktual + 1 (log)", "Likes prediksi + 1 (log)"
            lo = 1
        else:
            res["x"], res["y"] = res["likes_aktual"], res["likes_prediksi"]
            x_label, y_label = "Likes aktual", "Likes prediksi"
            lo = 0
        hi = max(res["x"].max(), res["y"].max())
        fig = px.scatter(res, x="x", y="y", log_x=log_scale, log_y=log_scale,
                         labels={"x": x_label, "y": y_label},
                         hover_data={"likes_aktual": True, "likes_prediksi": ":.1f",
                                     "word_count": True, "hashtag_count": True, "x": False, "y": False},
                         title="Likes aktual vs prediksi (data uji)")
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Prediksi sempurna",
                                 line={"dash": "dash", "color": "gray"}))
        st.plotly_chart(fig, width="stretch")
        st.caption("Titik di garis putus-putus = prediksi tepat. Titik di bawah garis = model memprediksi "
                   "likes lebih kecil dari aslinya; di atas garis = lebih besar.")

        st.write("**Korelasi jumlah kata, hashtag, dan likes**")
        metode = st.radio(
            "Metode korelasi", ["Pearson", "Spearman"], horizontal=True,
            help="Pearson mengukur hubungan garis lurus dan sangat dipengaruhi postingan viral. "
                 "Spearman memakai peringkat, jadi lebih tahan terhadap nilai likes yang ekstrem.",
        )
        corr = an.correlation_matrix(df, method=metode.lower())
        fig_corr = px.imshow(
            corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
            aspect="auto", title=f"Heatmap korelasi ({metode})",
        )
        fig_corr.update_layout(coloraxis_colorbar={"title": "r"})
        st.plotly_chart(fig_corr, width="stretch")
        st.caption("Nilai mendekati +1 = hubungan searah kuat, −1 = berlawanan arah kuat, 0 = hampir tidak "
                   "berhubungan. Baris terakhir menunjukkan seberapa kuat tiap fitur berhubungan dengan likes.")

        st.write("**Estimasi likes**")
        with st.form("form_estimasi"):
            n_kata = st.number_input("Jumlah kata", min_value=0, value=20, step=1)
            n_tag = st.number_input("Jumlah hashtag", min_value=0, value=1, step=1)
            submit = st.form_submit_button("Hitung estimasi")
        if submit:
            est = m.predict(pd.DataFrame({"word_count": [n_kata], "hashtag_count": [n_tag]}))[0]
            st.metric("Estimasi likes", f"{est:.1f}")
            if est < 0:
                st.warning("Estimasi negatif tidak bermakna sebagai jumlah likes. "
                           "Ini menunjukkan keterbatasan model linear dengan dua fitur.")

        st.write("**Interpretasi**")
        show_text_or_todo(REGRESSION_INTERPRETATION)

# ---------------------------------------------------------------------------
# Temuan & keterbatasan (di bawah semua tab)
# ---------------------------------------------------------------------------
st.divider()
left, right = st.columns(2)
with left:
    st.subheader("3 Temuan Utama")
    for i, t in enumerate(FINDINGS, 1):
        show_text_or_todo(f"{i}. {t}" if not t.startswith("TODO") else t)
with right:
    st.subheader("2 Keterbatasan")
    for i, t in enumerate(LIMITATIONS, 1):
        show_text_or_todo(f"{i}. {t}" if not t.startswith("TODO") else t)
