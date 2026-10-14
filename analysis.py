"""
Fungsi analisis bersama untuk proyek Dashboard Analisis Postingan X tentang GTA 6.

Dipakai oleh notebook (proses data) dan app.py (aplikasi Streamlit),
supaya hasil di notebook dan di aplikasi selalu sama.
"""

import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["text", "created_at", "like_count"]
RANDOM_STATE = 42
N_CLUSTERS = 3

# Kata negasi TIDAK dibuang (penting untuk makna kalimat).
NEGATION_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "nobody", "nowhere",
    "cannot", "cant", "dont", "doesnt", "didnt", "isnt", "wasnt", "arent",
    "werent", "wont", "wouldnt", "shouldnt", "couldnt", "aint",
}

# Kata topik muncul di hampir semua postingan, jadi dibuang khusus untuk
# clustering agar cluster terbentuk dari isi pembahasan, bukan nama gim.
TOPIC_WORDS = {"gta", "gta6", "gtavi", "vi", "grand", "theft", "auto", "rockstar", "rockstargames"}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
RT_RE = re.compile(r"^RT\s+", flags=re.IGNORECASE)
HTML_ENTITY_RE = re.compile(r"&amp;|&lt;|&gt;|&quot;|&#39;")


# ---------------------------------------------------------------------------
# Validasi & persiapan data
# ---------------------------------------------------------------------------
def check_required_columns(df: pd.DataFrame) -> list:
    """Kembalikan daftar kolom wajib yang tidak ada (kosong = lengkap)."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def text_for_sentiment(text: str) -> str:
    """
    Pembersihan ringan untuk VADER: hapus URL, mention, RT, entitas HTML.
    Huruf kapital, tanda baca, dan emoji DIPERTAHANKAN karena VADER
    memakainya untuk mengukur intensitas sentimen.
    """
    text = str(text)
    text = RT_RE.sub("", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HTML_ENTITY_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Pembersihan untuk clustering: huruf kecil, hapus URL, mention, simbol #,
    angka, dan tanda baca. Kata negasi tidak dihapus di tahap ini.
    """
    text = text_for_sentiment(text).lower()
    text = text.replace("#", " ")
    text = text.replace("'", "").replace("’", "")  # don't -> dont
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Lengkapi kolom turunan yang dibutuhkan analisis (tanpa membuang baris)."""
    df = df.copy()
    df["text"] = df["text"].astype(str)
    if "clean_text" not in df.columns:
        df["clean_text"] = df["text"].apply(clean_text)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce")
    df["word_count"] = df["text"].apply(count_words)
    df["hashtag_count"] = df["text"].apply(count_hashtags)
    return df


# ---------------------------------------------------------------------------
# Analisis sentimen (VADER)
# ---------------------------------------------------------------------------
_analyzer = SentimentIntensityAnalyzer()


def vader_label(text: str) -> tuple:
    """
    Kembalikan (skor compound, label).
    Ambang standar VADER: >= 0.05 positif, <= -0.05 negatif, selain itu netral.
    """
    score = _analyzer.polarity_scores(text_for_sentiment(text))["compound"]
    if score >= 0.05:
        label = "Positif"
    elif score <= -0.05:
        label = "Negatif"
    else:
        label = "Netral"
    return score, label


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    result = df["text"].apply(vader_label)
    df["sentiment_score"] = result.apply(lambda r: r[0])
    df["sentiment"] = result.apply(lambda r: r[1])
    return df


def evaluate_manual_labels(eval_df: pd.DataFrame) -> dict:
    """Bandingkan label manual dengan prediksi sistem."""
    e = eval_df.dropna(subset=["label_manual", "prediksi_sistem"]).copy()
    e["label_manual"] = e["label_manual"].astype(str).str.strip().str.capitalize()
    e["prediksi_sistem"] = e["prediksi_sistem"].astype(str).str.strip().str.capitalize()
    e["sesuai"] = e["label_manual"] == e["prediksi_sistem"]
    n = len(e)
    n_match = int(e["sesuai"].sum())
    return {
        "n": n,
        "n_match": n_match,
        "pct_match": (n_match / n * 100) if n else 0.0,
        "mismatches": e[~e["sesuai"]],
        "table": e,
    }


# ---------------------------------------------------------------------------
# Clustering (TF-IDF + K-Means)
# ---------------------------------------------------------------------------
def clustering_stop_words() -> list:
    return sorted((set(ENGLISH_STOP_WORDS) - NEGATION_WORDS) | TOPIC_WORDS)


def _non_empty(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["clean_text"].str.strip() != ""].copy()


def fit_clustering(df: pd.DataFrame, n_clusters: int = N_CLUSTERS):
    """MELATIH TF-IDF + K-Means. Kembalikan (vectorizer, kmeans)."""
    data = _non_empty(df)
    vectorizer = TfidfVectorizer(
        stop_words=clustering_stop_words(),
        min_df=2,
        max_df=0.9,
        token_pattern=r"(?u)\b[a-z]{3,}\b",
    )
    X = vectorizer.fit_transform(data["clean_text"])
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10).fit(X)
    return vectorizer, km


def summarize_clustering(df: pd.DataFrame, vectorizer, km, top_n: int = 5, n_examples: int = 3):
    """
    MEMAKAI model yang sudah dilatih untuk memberi cluster tiap postingan.
    Baris dengan clean_text kosong dikeluarkan.
    Kembalikan (df_dengan_kolom_cluster, ringkasan_per_cluster, jumlah_baris_dikeluarkan).
    """
    data = _non_empty(df)
    excluded = len(df) - len(data)
    X = vectorizer.transform(data["clean_text"])
    data["cluster"] = km.predict(X)

    terms = np.array(vectorizer.get_feature_names_out())
    summary = []
    for c in range(km.n_clusters):
        idx = np.where(data["cluster"].values == c)[0]
        top_words = terms[km.cluster_centers_[c].argsort()[::-1][:top_n]].tolist()
        # Contoh = postingan paling dekat dengan pusat cluster (paling mewakili).
        dist = km.transform(X[idx])[:, c]
        examples = data.iloc[idx[dist.argsort()[:n_examples]]]["text"].tolist()
        summary.append({
            "cluster": c,
            "jumlah": len(idx),
            "kata_dominan": top_words,
            "contoh": examples,
        })
    return data, summary, excluded


def run_clustering(df: pd.DataFrame, n_clusters: int = N_CLUSTERS, top_n: int = 5, n_examples: int = 3):
    """Latih lalu langsung pakai (dipakai kalau file model belum ada)."""
    vectorizer, km = fit_clustering(df, n_clusters)
    return summarize_clustering(df, vectorizer, km, top_n, n_examples)


def cluster_word_frequencies(clustered: pd.DataFrame, cluster: int, top_n: int = 100) -> dict:
    """
    Frekuensi kata pada satu cluster, untuk word cloud.
    Memakai stopword dan aturan kata (minimal 3 huruf) yang sama dengan clustering.
    """
    stop = set(clustering_stop_words())
    words = (
        clustered.loc[clustered["cluster"] == cluster, "clean_text"]
        .str.findall(r"\b[a-z]{3,}\b")
        .explode()
        .dropna()
    )
    words = words[~words.isin(stop)]
    return words.value_counts().head(top_n).to_dict()


# ---------------------------------------------------------------------------
# Regresi linear
# ---------------------------------------------------------------------------
def count_words(text: str) -> int:
    return len(str(text).split())


def count_hashtags(text: str) -> int:
    return len(HASHTAG_RE.findall(str(text)))


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Korelasi antara jumlah kata, jumlah hashtag, dan jumlah likes.
    method: "pearson" (hubungan linear) atau "spearman" (berbasis peringkat,
    lebih tahan terhadap postingan viral dengan likes ekstrem).
    Baris dengan likes tidak valid dikeluarkan, sama seperti regresi.
    """
    valid = df[df["like_count"].notna() & (df["like_count"] >= 0)]
    corr = valid[["word_count", "hashtag_count", "like_count"]].corr(method=method)
    names = {"word_count": "Jumlah kata", "hashtag_count": "Jumlah hashtag", "like_count": "Jumlah likes"}
    return corr.rename(index=names, columns=names)


def run_regression(df: pd.DataFrame):
    """
    Fitur: word_count, hashtag_count (dari teks asli). Target: like_count.
    Baris dengan like_count kosong/tidak valid/negatif DIKELUARKAN, tidak diganti nol.
    """
    valid = df[df["like_count"].notna() & (df["like_count"] >= 0)].copy()
    excluded = len(df) - len(valid)

    X = valid[["word_count", "hashtag_count"]]
    y = valid["like_count"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = LinearRegression().fit(X_train, y_train)  # hanya data latih
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "excluded": excluded,
        "test_result": pd.DataFrame({
            "word_count": X_test["word_count"].values,
            "hashtag_count": X_test["hashtag_count"].values,
            "likes_aktual": y_test.values,
            "likes_prediksi": y_pred,
        }),
    }
