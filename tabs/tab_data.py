import streamlit as st

def render_tab_data(df, df_valid_likes, required_columns):
    st.header("Pratinjau & Pemeriksaan Dataset")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Postingan", len(df))
    col2.metric("Data Likes Valid", len(df_valid_likes))
    col3.metric("Jumlah Kolom", len(df.columns))
    
    st.subheader("Pemeriksaan Kolom Wajib")
    col_status = {col: ("✅ Ada" if col in df.columns else "❌ Tidak Ada") for col in required_columns}
    st.json(col_status)
    
    st.subheader("10 Baris Pertama Dataset")
    st.dataframe(df.head(10), use_container_width=True)