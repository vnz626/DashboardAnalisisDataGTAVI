import streamlit as st
import pandas as pd
import plotly.express as px

def render_tab_sentimen(df, eval_df):
    st.header("Analisis Sentimen Postingan GTA VI")
    
    # Generate label otomatis sederhana jika belum ada kolom 'sentimen'
    if 'sentimen' not in df.columns:
        st.warning("⚠️ Kolom 'sentimen' tidak ditemukan dalam dataset. Menggunakan klasifikasi rule-based sederhana.")
        def simple_sentiment(t):
            t_lower = str(t).lower()
            if any(w in t_lower for w in ['keren', 'hype', 'seru', 'bagus', 'wait', 'cant wait', 'hyped', 'amazing', 'love']):
                return 'Positif'
            elif any(w in t_lower for w in ['kecewa', 'jelek', 'delay', 'lama', 'mahal', 'bad', 'worst', 'hate', 'bug']):
                return 'Negatif'
            return 'Netral'
        df['sentimen'] = df['cleaned_text'].apply(simple_sentiment)

    sent_counts = df['sentimen'].value_counts()
    sent_percents = (df['sentimen'].value_counts(normalize=True) * 100).round(2)
    
    summary_df = pd.DataFrame({
        'Jumlah Postingan': sent_counts,
        'Persentase (%)': sent_percents
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
            color_discrete_map={'Positif': '#2ecc71', 'Netral': '#95a5a6', 'Negatif': '#e74c3c'},
            hole=0.4
        )
        st.plotly_chart(fig_sent, use_container_width=True)
        
    st.subheader("Tabel Detail Klasifikasi Sentimen")
    st.dataframe(df[['text', 'cleaned_text', 'sentimen']], use_container_width=True)
    
    st.markdown("---")
    st.subheader("Pemeriksaan Manual (20 Sampel Evaluasi)")
    
    if eval_df is not None:
        if 'label_manual' not in eval_df.columns or 'label_sistem' not in eval_df.columns:
            st.warning("⚠️ File `evaluasi_sentimen.csv` tidak memiliki kolom 'label_manual' atau 'label_sistem'. Menampilkan pratinjau file saja.")
            st.dataframe(eval_df.head(20), use_container_width=True)
        else:
            st.dataframe(eval_df.head(20), use_container_width=True)
            
            matches = (eval_df['label_manual'] == eval_df['label_sistem']).sum()
            total_eval = len(eval_df)
            acc = (matches / total_eval) * 100
            
            st.info(f"**Akurasi Evaluasi Manual:** {matches} dari {total_eval} postingan sesuai ({acc:.2f}%)")
            
            mismatches = eval_df[eval_df['label_manual'] != eval_df['label_sistem']]
            if len(mismatches) > 0:
                st.subheader("Diskusi Ketidaksesuaian Evaluasi")
                st.dataframe(mismatches.head(3)[['text', 'label_manual', 'label_sistem']])
    else:
        st.write("📌 *Unggah file `evaluasi_sentimen.csv` di sidebar untuk menampilkan ringkasan evaluasi manual 20 postingan.*")