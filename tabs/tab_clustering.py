import streamlit as st

@st.cache_data
def run_kmeans_clustering(text_series):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_series.astype(str))
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    
    return cluster_labels, terms, order_centroids

def render_tab_clustering(df):
    st.header("Clustering Data Teks (TF-IDF + K-Means)")
    
    # Ambil hasil terms dan centroids yang sudah dihitung di app.py
    terms = getattr(st.session_state, 'cluster_terms', None)
    order_centroids = getattr(st.session_state, 'cluster_centroids', None)
    
    if terms is None or order_centroids is None or 'cluster' not in df.columns:
        df['cluster'], terms, order_centroids = run_kmeans_clustering(df['cleaned_text'])
    
    theme_names = {
        0: "Ekspektasi Trailer & Tanggal Rilis",
        1: "Diskusi Spesifikasi & Grafik Game",
        2: "Reaksi Komunitas & Memes"
    }
    
    cols = st.columns(3)
    for cluster_id in range(3):
        with cols[cluster_id]:
            cluster_data = df[df['cluster'] == cluster_id]
            top_words = [terms[ind] for ind in order_centroids[cluster_id, :5]]
            
            st.subheader(f"Cluster {cluster_id}")
            st.markdown(f"**Nama Tema:** {theme_names.get(cluster_id, 'Tema ' + str(cluster_id))}")
            st.metric("Jumlah Postingan", len(cluster_data))
            
            st.markdown("**5 Kata Dominan:**")
            st.write(", ".join(top_words))
            
            st.markdown("**3 Contoh Postingan:**")
            samples = cluster_data['text'].head(3).tolist()
            for i, sample in enumerate(samples, 1):
                st.caption(f"{i}. \"{sample}\"")
                
    st.markdown("---")
    st.subheader("Interpretasi & Analisis Clustering")
    st.markdown("""
    * **Pembeda Isi Cluster:** Cluster 0 berfokus pada spekulasi rilis dan trailer, Cluster 1 lebih didominasi percakapan teknis perangkat dan visual, sedangkan Cluster 2 berisi humor dan reaksi umum netizen.
    * **Tumpang Tindih Pembahasan:** Masih ditemukan sedikit tumpang tindih kata kunci umum seperti *'GTA'*, *'Rockstar'*, dan *'Game'* di seluruh cluster karena pembahasan berasal dari satu topik yang sama.
    """)