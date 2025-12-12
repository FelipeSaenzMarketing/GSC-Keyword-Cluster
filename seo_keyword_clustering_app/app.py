import io
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


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


def load_dataframe_from_upload(uploaded_file) -> pd.DataFrame:
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
    st.subheader("Step 4: Interactive semantic map (Plotly)")
    st.write("Computing 2D projection for visualization (this may take some seconds for large datasets)...")

    try:
        X_2d = create_tsne_2d_projection(X_tfidf)
        df_keywords["x"] = X_2d[:, 0]
        df_keywords["y"] = X_2d[:, 1]

        # ---- Metrics (English + German support) ----
        def resolve_col(mapping: dict, df_cols, candidates: list):
            """
            Tries to resolve a column name from:
            1) mapping keys (exact),
            2) direct df columns match (case-insensitive).
            Returns the real column name or None.
            """
            # 1) Try mapping first
            for key in candidates:
                col = mapping.get(key)
                if col and col in df_cols:
                    return col

            # 2) Fallback: match by df column names (case-insensitive)
            lower_to_real = {c.lower().strip(): c for c in df_cols}
            for key in candidates:
                if key.lower().strip() in lower_to_real:
                    return lower_to_real[key.lower().strip()]

            return None

        clicks_col = resolve_col(mapping, df_original.columns, ["clicks", "klicks"])
        impr_col   = resolve_col(mapping, df_original.columns, ["impressions", "impressionen"])
        ctr_col    = resolve_col(mapping, df_original.columns, ["ctr"])

        # attach metrics to df_keywords (aligned by index)
        if clicks_col:
            df_keywords["clicks"] = pd.to_numeric(
                df_original.loc[df_keywords.index, clicks_col],
                errors="coerce"
            ).fillna(0)
        else:
            df_keywords["clicks"] = 0

        if impr_col:
            df_keywords["impressions"] = pd.to_numeric(
                df_original.loc[df_keywords.index, impr_col],
                errors="coerce"
            ).fillna(0)
        else:
            df_keywords["impressions"] = 0

        if ctr_col:
            # CTR can come as "0.12" or "12%" depending on export
            raw_ctr = (
                df_original.loc[df_keywords.index, ctr_col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)  # helps if decimal comma appears
            )
            df_keywords["ctr"] = pd.to_numeric(raw_ctr, errors="coerce")

            # If values look like 0-1, convert to %
            if df_keywords["ctr"].dropna().between(0, 1).mean() > 0.8:
                df_keywords["ctr"] = df_keywords["ctr"] * 100

            df_keywords["ctr"] = df_keywords["ctr"].fillna(0)
        else:
            df_keywords["ctr"] = 0

        # ---- Sidebar filters for the chart ----
        st.sidebar.subheader("Semantic map filters")

        cluster_options = (
            df_keywords[["cluster_id", "cluster_name"]]
            .drop_duplicates()
            .sort_values(["cluster_id"])
        )

        default_clusters = cluster_options["cluster_name"].tolist()

        selected_cluster_names = st.sidebar.multiselect(
            "Show clusters",
            options=cluster_options["cluster_name"].tolist(),
            default=default_clusters
        )

        # Range filters
        cmin, cmax = float(df_keywords["clicks"].min()), float(df_keywords["clicks"].max())
        imin, imax = float(df_keywords["impressions"].min()), float(df_keywords["impressions"].max())

        clicks_range = st.sidebar.slider(
            "Clicks range",
            min_value=float(cmin),
            max_value=float(cmax),
            value=(float(cmin), float(cmax)),
            step=max(1.0, float((cmax - cmin) / 100) if cmax > cmin else 1.0),
        )

        impressions_range = st.sidebar.slider(
            "Impressions range",
            min_value=float(imin),
            max_value=float(imax),
            value=(float(imin), float(imax)),
            step=max(1.0, float((imax - imin) / 100) if imax > imin else 1.0),
        )

        keyword_search = st.sidebar.text_input("Search keyword (contains)", value="").strip()

        # Apply filters
        df_plot = df_keywords.copy()
        df_plot = df_plot[df_plot["cluster_name"].isin(selected_cluster_names)]
        df_plot = df_plot[(df_plot["clicks"] >= clicks_range[0]) & (df_plot["clicks"] <= clicks_range[1])]
        df_plot = df_plot[(df_plot["impressions"] >= impressions_range[0]) & (df_plot["impressions"] <= impressions_range[1])]

        if keyword_search:
            df_plot = df_plot[df_plot["keyword"].str.contains(keyword_search, case=False, na=False)]

        # Avoid size=0 for plotly bubble sizing
        df_plot["size_clicks"] = df_plot["clicks"].clip(lower=0)
        if df_plot["size_clicks"].max() == 0:
            df_plot["size_clicks"] = 1

        # ---- Plotly interactive scatter ----
        fig = px.scatter(
            df_plot,
            x="x",
            y="y",
            color="cluster_name",
            size="size_clicks",
            size_max=18,
            hover_data={
                "keyword": True,
                "cluster_id": True,
                "cluster_name": True,
                "clicks": ":.0f",
                "impressions": ":.0f",
                "ctr": ":.2f",
                "x": False,
                "y": False,
                "size_clicks": False,
            },
            title="Semantic map of keywords (t-SNE + TF-IDF)",
        )

        fig.update_layout(
            height=720,
            legend_title_text="Cluster",
            margin=dict(l=20, r=20, t=60, b=20),
        )

        fig.update_traces(marker=dict(opacity=0.85))

        st.plotly_chart(fig, use_container_width=True)

        # ---- Cluster summary table ----
        st.subheader("Cluster summary (filtered view)")
        summary = (
            df_plot.groupby(["cluster_id", "cluster_name"], as_index=False)
            .agg(
                keywords=("keyword", "count"),
                total_clicks=("clicks", "sum"),
                total_impressions=("impressions", "sum"),
                avg_ctr=("ctr", "mean"),
            )
            .sort_values(["total_clicks", "total_impressions"], ascending=False)
        )

        st.dataframe(summary, use_container_width=True)

        st.caption(
            "Tip: Use the sidebar filters + zoom/box-select to explore clusters. "
            "Bubble size = clicks, color = cluster name."
        )

    except Exception as e:
        st.warning(f"Could not compute t-SNE projection: {e}")

    # ---- Download section ----
    st.subheader("Step 5: Download clustered data")
    
    # Prepare output dataframe with original columns + cluster info
    df_output = df_original.copy()
    df_output["cluster_id"] = df_keywords["cluster_id"]
    df_output["cluster_name"] = df_keywords["cluster_name"]
    
    # Generate download file
    if output_format == "Excel (.xlsx)":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_output.to_excel(writer, index=False, sheet_name='Clustered Keywords')
        buffer.seek(0)
        
        st.download_button(
            label="📥 Download Excel file",
            data=buffer,
            file_name="keyword_clusters.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:  # CSV
        csv_buffer = io.StringIO()
        df_output.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download CSV file",
            data=csv_data,
            file_name="keyword_clusters.csv",
            mime="text/csv"
        )
    
    st.success("✅ Clustering complete! Download your results above.")


if __name__ == "__main__":
    main()
