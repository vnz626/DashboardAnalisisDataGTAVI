import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def render_tab_regresi(df_valid_likes):
    st.header("Prediksi Jumlah Likes (Regresi Linear)")
    
    X = df_valid_likes[['word_count', 'hashtag_count']]
    y = df_valid_likes['like_count']
    
    if len(df_valid_likes) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
        col_m2.metric("R² Score", f"{r2:.4f}")
        
        st.subheader("Grafik Likes Aktual vs Hasil Prediksi")
        chart_data = pd.DataFrame({
            'Aktual': y_test.values,
            'Prediksi': y_pred
        }).reset_index()
        
        fig_reg = px.scatter(
            chart_data, x='Aktual', y='Prediksi', 
            labels={'Aktual': 'Likes Aktual', 'Prediksi': 'Likes Hasil Prediksi'},
            title="Perbandingan Likes Aktual vs Prediksi"
        )
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig_reg.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                          line=dict(color="Red", dash="dash"))
        
        st.plotly_chart(fig_reg, use_container_width=True)
        
        st.subheader("🔮 Form Estimasi Likes")
        with st.form("prediction_form"):
            input_words = st.number_input("Masukkan Jumlah Kata dalam Postingan:", min_value=1, value=15)
            input_hashtags = st.number_input("Masukkan Jumlah Hashtag:", min_value=0, value=2)
            submit_btn = st.form_submit_button("Hitung Estimasi Likes")
            
            if submit_btn:
                pred_val = model.predict([[input_words, input_hashtags]])[0]
                st.success(f"Estimasi Jumlah Likes: **{pred_val:.1f}**")
                if pred_val < 0:
                    st.warning("⚠️ Hasil estimasi bernilai negatif. Hal ini menunjukkan keterbatasan model regresi linear sederhana pada data ini.")
                    
        st.markdown("---")
        st.subheader("Interpretasi Hasil Regresi")
        st.markdown(f"""
        * **Makna MAE ({mae:.2f}):** Rata-rata kesalahan mutlak prediksi model menyimpang sekitar {mae:.2f} likes dari nilai aktualnya.
        * **Makna $R^2$ ({r2:.4f}):** Variabel jumlah kata dan hashtag hanya mampu menjelaskan {max(0, r2*100):.2f}% variasi jumlah likes (jika $R^2$ negatif, model linear memberikan prediksi yang lebih buruk daripada sekadar menggunakan nilai rata-rata).
        * **Keterbatasan Model:** Jumlah likes di X sangat dipengaruhi faktor eksternal seperti jangkauan akun (*followers*), usia postingan, penggunaan media (gambar/video), dan *timing* unggah.
        """)
    else:
        st.error("Jumlah data valid tidak cukup untuk melakukan pembagian data latih dan uji regresi linear.")