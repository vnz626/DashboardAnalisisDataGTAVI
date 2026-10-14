import pandas as pd
import plotly.express as px
import streamlit as st

import analysis as an

try:
    from wordcloud import WordCloud
except ImportError:  # aplikasi tetap jalan walau pustaka belum terpasang
    WordCloud = None

# ---------------------------------------------------------------------------
# DIISI YOHANES setelah membaca kata dominan & contoh postingan tiap cluster.
# Nomor cluster (0, 1, 2) harus sesuai dengan yang tampil di aplikasi.
# ---------------------------------------------------------------------------
THEME_NAMES = {
    0: "Antusiasme dan Dukungan",
    1: "Musik",
    2: "Meme dan Satir",
}

INTERPRETATION = (
    "Cluster 0 (Antusiasme) berisi obrolan umum: harapan, countdown, dan komentar santai.\n\n "
    "Cluster 1 (Musik) sangat spesifik: promosi lagu Cardi B dari 'album GTA 6'. \n\n"
    "Cluster 2 (Meme dan Satire) berpola 'We got X before GTA 6'. \n\n"
    "Cluster 0 menampung 315 dari 383 postingan dengan kata dominan generik, "
    "jadi berfungsi sebagai cluster sisa. \n\n"
    "Tumpang tindih masih ada. Meme bernada antusias bisa masuk Cluster 0. \n\n"
    "Cluster 1 dan 2 batasnya lebih jelas karena memakai kata atau pola khas."
)


@st.cache_data
def _cluster(df: pd.DataFrame):
    # Memakai fungsi yang sama dengan notebook Colab, jadi hasilnya sama dengan laporan.
    return an.run_clustering(df)


def _show_text(text: str):
    if text.strip().startswith("TODO"):
        st.warning(f"Belum diisi kelompok: {text}")
    else:
        st.markdown(text)


def render_tab_clustering(df):
    st.header("Clustering Data Teks (TF-IDF + K-Means)")

    data = df.copy()
    if "clean_text" not in data.columns:
        data["clean_text"] = data["text"].apply(an.clean_text)
    data["clean_text"] = data["clean_text"].fillna("").astype(str)

    try:
        clustered, summary, excluded = _cluster(data)
    except Exception as exc:
        st.error(f"Clustering gagal dijalankan: {exc}")
        return

    if excluded:
        st.caption(f"{excluded} postingan dikeluarkan karena teks bersihnya kosong.")

    sizes = pd.DataFrame({
        "Cluster": [f"{s['cluster']}: {THEME_NAMES.get(s['cluster'], '')}" for s in summary],
        "Jumlah": [s["jumlah"] for s in summary],
    })
    st.plotly_chart(
        px.bar(sizes, x="Cluster", y="Jumlah", text="Jumlah", title="Ukuran cluster"),
        width="stretch",
    )

    for s in summary:
        c = s["cluster"]
        with st.container(border=True):
            st.subheader(f"Cluster {c}: {THEME_NAMES.get(c, '')}")
            st.metric("Jumlah Postingan", s["jumlah"])
            st.markdown("**5 Kata Dominan:** " + ", ".join(f"`{w}`" for w in s["kata_dominan"]))

            freq = an.cluster_word_frequencies(clustered, c)
            if WordCloud is not None and freq:
                wc = WordCloud(width=900, height=320, background_color="white",
                               colormap="viridis", random_state=an.RANDOM_STATE)
                st.image(wc.generate_from_frequencies(freq).to_image(), width="stretch",
                         caption="Word cloud: makin besar kata, makin sering muncul di cluster ini")
            elif WordCloud is None:
                st.caption("Pasang pustaka `wordcloud` untuk menampilkan word cloud.")

            st.markdown("**3 Contoh Postingan (paling mewakili cluster):**")
            for i, sample in enumerate(s["contoh"], 1):
                st.caption(f'{i}. "{sample}"')

    st.markdown("---")
    st.subheader("Interpretasi & Analisis Clustering")
    _show_text(INTERPRETATION)
