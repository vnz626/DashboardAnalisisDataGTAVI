import pandas as pd
import plotly.express as px
import streamlit as st

import analysis as an

SENTIMENT_ORDER = ["Positif", "Netral", "Negatif"]
SENTIMENT_COLORS = {"Positif": "#2ecc71", "Netral": "#95a5a6", "Negatif": "#e74c3c"}


@st.cache_data
def _add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    # VADER dari analysis.py: sama dengan yang dipakai di notebook Colab dan evaluasi manual.
    return an.add_sentiment(df)


def render_tab_sentimen(df, eval_df):
    st.header("Analisis Sentimen Postingan GTA VI (VADER)")
    st.caption("Skor compound ≥ 0,05 = Positif, ≤ −0,05 = Negatif, selain itu Netral.")

    data = _add_sentiment(df)
    sent_counts = data["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    sent_percents = (sent_counts / sent_counts.sum() * 100).round(1)

    summary_df = pd.DataFrame({
        "Jumlah Postingan": sent_counts,
        "Persentase (%)": sent_percents,
    })

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.subheader("Ringkasan Sentimen")
        st.dataframe(summary_df)
    with col_s2:
        st.subheader("Grafik Distribusi Sentimen")
        fig_sent = px.pie(
            names=sent_counts.index,
            values=sent_counts.values,
            color=sent_counts.index,
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.4,
        )
        st.plotly_chart(fig_sent, width="stretch")

    st.subheader("Tabel Detail Klasifikasi Sentimen")
    pilihan = st.multiselect("Filter sentimen", SENTIMENT_ORDER, default=SENTIMENT_ORDER)
    st.dataframe(
        data[data["sentiment"].isin(pilihan)][["text", "sentiment", "sentiment_score"]],
        width="stretch",
    )

    st.markdown("---")
    st.subheader("Pemeriksaan Manual (20 Sampel Evaluasi)")

    if eval_df is None:
        st.warning("📌 Berkas `data/evaluasi_sentimen.csv` tidak ditemukan. "
                   "Pastikan file evaluasi 20 sampel berada di dalam folder `data/`.")
        return

    missing = [c for c in ["text", "label_manual", "prediksi_sistem"] if c not in eval_df.columns]
    if missing:
        st.error(f"evaluasi_sentimen.csv tidak memiliki kolom: {', '.join(missing)}")
        return

    try:
        ev = an.evaluate_manual_labels(eval_df)
    except Exception as exc:
        st.error(f"evaluasi_sentimen.csv tidak dapat diproses: {exc}")
        return

    st.info(f"**Kecocokan label manual dengan VADER:** {ev['n_match']} dari {ev['n']} "
            f"postingan sesuai ({ev['pct_match']:.1f}%)")
    st.dataframe(ev["table"][["text", "label_manual", "prediksi_sistem", "sesuai"]], width="stretch")

    if len(ev["mismatches"]) > 0:
        st.subheader("Postingan yang Tidak Sesuai")
        st.dataframe(ev["mismatches"][["text", "label_manual", "prediksi_sistem"]], width="stretch")

    st.caption("Evaluasi terbatas pada 20 sampel dan belum menggambarkan akurasi pada seluruh percakapan di X.")
