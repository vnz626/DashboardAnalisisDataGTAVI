import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as an


def render_tab_regresi(df_valid_likes):
    st.header("Prediksi Jumlah Likes (Regresi Linear)")

    if len(df_valid_likes) < 10:
        st.error("Jumlah data valid tidak cukup untuk melakukan pembagian data latih dan uji regresi linear.")
        return

    # Memakai fungsi yang sama dengan notebook Colab, jadi hasilnya sama dengan laporan.
    try:
        reg = an.run_regression(df_valid_likes)
    except Exception as exc:
        st.error(f"Regresi gagal dijalankan: {exc}")
        return

    model, mae, r2 = reg["model"], reg["mae"], reg["r2"]

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("MAE (Mean Absolute Error)", f"{mae:.2f} likes")
    col_m2.metric("R² Score", f"{r2:.4f}")
    col_m3.metric("Data latih / uji", f"{reg['n_train']} / {reg['n_test']}")

    st.latex(
        rf"\hat{{likes}} = {model.intercept_:.2f} + ({model.coef_[0]:.2f})\times kata"
        rf" + ({model.coef_[1]:.2f})\times hashtag"
    )

    # --- Grafik aktual vs prediksi --------------------------------------------
    st.subheader("Grafik Likes Aktual vs Hasil Prediksi")
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
        x_label, y_label, lo = "Likes aktual + 1 (log)", "Likes prediksi + 1 (log)", 1
    else:
        res["x"], res["y"] = res["likes_aktual"], res["likes_prediksi"]
        x_label, y_label, lo = "Likes aktual", "Likes prediksi", 0
    hi = max(res["x"].max(), res["y"].max())

    fig_reg = px.scatter(
        res, x="x", y="y", log_x=log_scale, log_y=log_scale,
        labels={"x": x_label, "y": y_label},
        hover_data={"likes_aktual": True, "likes_prediksi": ":.1f",
                    "word_count": True, "hashtag_count": True, "x": False, "y": False},
        title="Perbandingan Likes Aktual vs Prediksi (data uji)",
    )
    fig_reg.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Prediksi sempurna",
                                 line={"dash": "dash", "color": "red"}))
    st.plotly_chart(fig_reg, width="stretch")
    st.caption("Titik di garis putus-putus = prediksi tepat. Di bawah garis = model menebak terlalu kecil; "
               "di atas garis = terlalu besar.")

    # --- Heatmap korelasi -----------------------------------------------------
    st.subheader("Korelasi Jumlah Kata, Hashtag, dan Likes")
    metode = st.radio(
        "Metode korelasi", ["Pearson", "Spearman"], horizontal=True,
        help="Pearson mengukur hubungan garis lurus dan sangat dipengaruhi postingan viral. "
             "Spearman memakai peringkat, jadi lebih tahan terhadap nilai likes yang ekstrem.",
    )
    corr = an.correlation_matrix(df_valid_likes, method=metode.lower())
    fig_corr = px.imshow(
        corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
        aspect="auto", title=f"Heatmap korelasi ({metode})",
    )
    fig_corr.update_layout(coloraxis_colorbar={"title": "r"})
    st.plotly_chart(fig_corr, width="stretch")
    st.caption("Nilai mendekati +1 = hubungan searah kuat, −1 = berlawanan arah kuat, 0 = hampir tidak "
               "berhubungan. Baris terakhir menunjukkan seberapa kuat tiap fitur berhubungan dengan likes.")

    # --- Form estimasi ----------------------------------------------------------
    st.subheader("🔮 Form Estimasi Likes")
    with st.form("prediction_form"):
        input_words = st.number_input("Masukkan Jumlah Kata dalam Postingan:", min_value=1, value=15)
        input_hashtags = st.number_input("Masukkan Jumlah Hashtag:", min_value=0, value=2)
        submit_btn = st.form_submit_button("Hitung Estimasi Likes")

    if submit_btn:
        pred_val = model.predict(
            pd.DataFrame({"word_count": [input_words], "hashtag_count": [input_hashtags]})
        )[0]
        st.success(f"Estimasi Jumlah Likes: **{pred_val:.1f}**")
        if pred_val < 0:
            st.warning("⚠️ Hasil estimasi bernilai negatif. Hal ini menunjukkan keterbatasan model "
                       "regresi linear dengan dua fitur pada data ini.")

    # --- Interpretasi -----------------------------------------------------------
    st.markdown("---")
    st.subheader("Interpretasi Hasil Regresi")
    st.markdown(f"""
    * **Makna MAE ({mae:.2f}):** Rata-rata prediksi model menyimpang sekitar {mae:.2f} likes dari nilai aktualnya.
    * **Makna $R^2$ ({r2:.4f}):** Variabel jumlah kata dan hashtag hanya mampu menjelaskan {max(0, r2 * 100):.2f}% variasi jumlah likes (jika $R^2$ negatif, model linear memberikan prediksi yang lebih buruk daripada sekadar menggunakan nilai rata-rata).
    * **Keterbatasan Model:** Jumlah likes di X sangat dipengaruhi faktor lain di luar teks, seperti jangkauan akun (*followers*), usia postingan, penggunaan media (gambar/video), dan *timing* unggah.
    """)
