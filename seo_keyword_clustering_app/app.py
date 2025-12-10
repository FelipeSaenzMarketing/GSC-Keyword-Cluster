import io
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


# =========================
# Data loading and utilities
# =========================

def detect_column_names(columns):
    """
    Detect the relevant column names for keywords and metrics in either English or German.

    Expected English columns:
        - "Top queries"
        - "Clicks"
        - "Impressions"
        - "CTR"

    Expected German columns:
        - "Suchanfrage"
        - "Klicks"
        - "Impressionen"
        - "CTR"

    Returns a dictionary with standard keys:
        {
            "keyword": <column_name_for_keyword>,
            "clicks": <column_name_for_clicks or None>,
            "impressions": <column_name_for_impressions or None>,
            "ctr": <column_name_for_ctr or None>,
        }
    """
    cols = list(columns)

    # Keyword column
    if "Top queries" in cols:
        keyword_col = "Top queries"
    elif "Suchanfrage" in cols:
        keyword_col = "Suchanfrage"
    else:
        # Fallback: use the first column as keyword
        keyword_col = cols[0]

    # Clicks
    if "Clicks" in cols:
        clicks_col = "Clicks"
    elif "Klicks" in cols:
        clicks_col = "Klicks"
    else:
        clicks_col = None

    # Impressions
    if "Impressions" in cols:
        impressions_col = "Impressions"
    elif "Impressionen" in cols:
        impressions_col = "Impressionen"
    else:
        impressions_col = None

    # CTR (same name in both languages)
    ctr_col = "CTR" if "CTR" in cols else None

    return {
        "keyword": keyword_col,
        "clicks": clicks_col,
        "impressions": impressions_col,
        "ctr": ctr_col,
    }


def load_dataframe_from_upload(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """
    Load a DataFrame from a Streamlit UploadedFile object.
    Supports .xls, .xlsx and .csv files.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    elif filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload a .xls, .xlsx, or .csv file.")

    if df.empty:
        raise ValueError("The uploaded file is empty.")

    return df


# =========================
# Text processing and clustering
# =========================

def vectorize_keywords_tfidf(keywords):
    """
    Convert a list/Series of keywords into a TF-IDF matrix.
    Returns the matrix and the fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),  # unigrams and bigrams
    )
    X = vectorizer.fit_transform(keywords)
    return X, vectorizer


def cluster_keywords(X, n_clusters: int = 10, random_state: int = 42):
    """
    Apply KMeans to group keywords into n_clusters.
    Returns the cluster labels for each row and the fitted model.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,  # avoid warnings in new sklearn versions
    )
    labels = kmeans.fit_predict(X)
    return labels, kmeans


def name_clusters_with_top_ngrams(X, labels, vectorizer, top_n: int = 3):
    """
    Create human-readable cluster names based on dominant n-grams for each cluster.

    Strategy:
    - For each cluster, compute the mean TF-IDF vector across all keywords in that cluster.
    - Sort terms by their average TF-IDF weight.
    - Take the top_n terms and build a label like:
        "Cluster 0: term1, term2, term3"

    Returns:
        dict: {cluster_id (int): cluster_name (str)}
    """
    terms = vectorizer.get_feature_names_out()
    labels = np.array(labels)
    unique_clusters = np.unique(labels)

    cluster_name_map = {}

    for cluster_id in unique_clusters:
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            cluster_name_map[cluster_id] = f"Cluster {cluster_id}"
            continue

        # Mean TF-IDF vector for this cluster
        cluster_tfidf = X[idx].mean(axis=0)
        cluster_tfidf = np.asarray(cluster_tfidf).ravel()

        if cluster_tfidf.sum() == 0:
            cluster_name_map[cluster_id] = f"Cluster {cluster_id}"
            continue

        # Get indices of top terms
        top_indices = cluster_tfidf.argsort()[::-1][:top_n]
        top_terms = [terms[i] for i in top_indices if cluster_tfidf[i] > 0]

        if not top_terms:
            cluster_name_map[cluster_id] = f"Cluster {cluster_id}"
        else:
            label_core = ", ".join(top_terms)
            cluster_name_map[cluster_id] = f"Cluster {cluster_id}: {label_core}"

    return cluster_name_map


def create_tsne_2d_projection(X, random_state: int = 42, perplexity: int = 30):
    """
    Reduce the dimensionality of matrix X to 2D using t-SNE for visualization.
    Returns an array of shape (n_samples, 2).

    Note: perplexity must be smaller than the number of samples.
    """
    n_samples = X.shape[0]
    if n_samples < 3:
        raise ValueError("Not enough samples to compute t-SNE projection (need at least 3 keywords).")

    adjusted_perplexity = min(perplexity, max(2, n_samples - 1))

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=adjusted_perplexity,
        init="random",
        learning_rate="auto",
    )
    X_2d = tsne.fit_transform(X)
    return X_2d


def plot_semantic_map(df: pd.DataFrame):
    """
    Create a 2D scatter plot with columns 'x', 'y' and color by 'cluster'.
    Returns a Matplotlib figure (for Streamlit usage).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(df["x"], df["y"], c=df["cluster"], alpha=0.8)

    ax.set_title("Semantic map of keywords (t-SNE + TF-IDF)")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # Optional: colorbar as simple cluster legend (numeric)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster (numeric ID)")

    # Show a small sample of labels to avoid clutter
    sample = df.sample(min(50, len(df)), random_state=42)
    for _, row in sample.iterrows():
        text_label = str(row["keyword"])[:15]
        ax.text(row["x"], row["y"], text_label, fontsize=6)

    fig.tight_layout()
    return fig


# =========================
# Streamlit app
# =========================

def main():
    st.set_page_config(
        page_title="SEO Keyword Clustering",
        layout="wide",
    )

    st.title("SEO Keyword Clustering App")
    st.write(
        """
        Upload a **.xls / .xlsx / .csv** file with your Search Console data and get:
        - A new file with additional **cluster_id** and **cluster_name** columns.
        - A **semantic similarity map** of your keywords based on TF-IDF + t-SNE.

        Supported column languages:
        - English: `Top queries`, `Clicks`, `Impressions`, `CTR`
        - German: `Suchanfrage`, `Klicks`, `Impressionen`, `CTR`
        """
    )

    # Sidebar controls
    st.sidebar.header("Configuration")
    n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=30, value=10, step=1)
    top_n_terms = st.sidebar.slider("Top n-grams per cluster (for naming)", min_value=1, max_value=5, value=3, step=1)
    output_format = st.sidebar.selectbox("Output file format", options=["Excel (.xlsx)", "CSV (.csv)"])

    uploaded_file = st.file_uploader(
        "Upload your .xls / .xlsx / .csv file:",
        type=["xls", "xlsx", "csv"],
    )

    if uploaded_file is None:
        st.info("Please upload a file to start the clustering process.")
        return

    try:
        df_original = load_dataframe_from_upload(uploaded_file)
    except Exception as e:
        st.error(f"Error while reading the file: {e}")
        return

    st.subheader("Preview of uploaded data")
    st.dataframe(df_original.head())

    # Detect relevant columns (keyword + metrics)
    mapping = detect_column_names(df_original.columns)
    keyword_col = mapping["keyword"]

    st.write(f"**Detected keyword column:** `{keyword_col}`")

    if keyword_col not in df_original.columns:
        st.error("Could not detect a valid keyword column.")
        return

    # Prepare keywords for clustering
    df_keywords = df_original.copy()
    df_keywords["keyword"] = (
        df_keywords[keyword_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df_keywords = df_keywords[df_keywords["keyword"] != ""].copy()

    if df_keywords.empty:
        st.error("No valid keywords found after cleaning.")
        return

    # TF-IDF vectorization
    st.subheader("Step 1: TF-IDF vectorization")
    st.write("Transforming keywords into numerical vectors...")
    X_tfidf, vectorizer = vectorize_keywords_tfidf(df_keywords["keyword"].tolist())
    st.write(f"TF-IDF matrix shape: `{X_tfidf.shape}`")

    # Clustering
    st.subheader("Step 2: KMeans clustering")
    st.write(f"Clustering keywords into **{n_clusters}** clusters...")
    labels, kmeans_model = cluster_keywords(X_tfidf, n_clusters=n_clusters)
    df_keywords["cluster_id"] = labels

    # Automatic cluster naming with top n-grams
    st.subheader("Step 3: Naming clusters with dominant n-grams")
    cluster_name_map = name_clusters_with_top_ngrams(X_tfidf, labels, vectorizer, top_n=top_n_terms)
    df_keywords["cluster_name"] = df_keywords["cluster_id"].map(cluster_name_map)

    # Show mapping between cluster_id and cluster_name
    mapping_df = (
        pd.DataFrame(
            {
                "cluster_id": list(cluster_name_map.keys()),
                "cluster_name": list(cluster_name_map.values()),
            }
        )
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )

    st.write("Cluster ID to name mapping:")
    st.dataframe(mapping_df)

    # 2D projection for visualization
    st.subheader("Step 4: t-SNE semantic map")
    st.write("Computing 2D projection for visualization (this may take some seconds for large datasets)...")

    try:
        X_2d = create_tsne_2d_projection(X_tfidf)
        df_keywords["x"] = X_2d[:, 0]
        df_keywords["y"] = X_2d[:, 1]

        # For visualization, we keep numeric cluster IDs
        df_plot = df_keywords[["keyword", "cluster_id", "x", "y"]].rename(columns={"cluster_id": "cluster"})
        fig = plot_semantic_map(df_plot)
        st.pyplot(fig)

        st.caption("Note: Colorbar shows numeric cluster IDs. See the mapping table above for human-readable names.")
    except Exception as e:
        st.warning(f"Could not compute t-SNE projection: {e}")

    # Merge cluster labels back into the original dataframe
    st.subheader("Step 5: Download clustered data")

    # We merge by index to keep the original structure (including metrics)
    df_with_clusters = df_original.copy()
    df_with_clusters.loc[df_keywords.index, "cluster_id"] = df_keywords["cluster_id"]
    df_with_clusters.loc[df_keywords.index, "cluster_name"] = df_keywords["cluster_name"]

    st.write("Sample of the final clustered dataset:")
    st.dataframe(df_with_clusters.head())

    # Prepare file for download
    if output_format.startswith("Excel"):
        output = io.BytesIO()
        df_with_clusters.to_excel(output, index=False)
        output.seek(0)
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        file_label = "Download clustered keywords (Excel)"
        file_name = "clustered_keywords.xlsx"
    else:
        csv_bytes = df_with_clusters.to_csv(index=False).encode("utf-8")
        output = csv_bytes
        mime = "text/csv"
        file_label = "Download clustered keywords (CSV)"
        file_name = "clustered_keywords.csv"

    st.download_button(
        label=file_label,
        data=output,
        file_name=file_name,
        mime=mime,
    )


if __name__ == "__main__":
    main()
