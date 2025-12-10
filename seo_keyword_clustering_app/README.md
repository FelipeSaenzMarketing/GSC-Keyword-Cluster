# SEO Keyword Clustering App

This repository contains a Streamlit application for clustering SEO keywords based on their semantic similarity.

## Features

- Upload **.xls / .xlsx / .csv** files with Search Console data.
- Automatic detection of keyword column in **English** or **German**:
  - English: `Top queries`, `Clicks`, `Impressions`, `CTR`
  - German: `Suchanfrage`, `Klicks`, `Impressionen`, `CTR`
- TF-IDF + KMeans clustering of keywords.
- Automatic **cluster naming** based on dominant n-grams (top TF-IDF terms per cluster).
- t-SNE **semantic map** of keyword clusters.
- Downloadable output with additional `cluster_id` and `cluster_name` columns.

## Installation

```bash
pip install -r requirements.txt
```

## Running the app locally

```bash
streamlit run app.py
```

## Deploying on Streamlit Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Cloud and create a new app.
3. Select your GitHub repo and set:
   - **Main file**: `app.py`
4. Click **Deploy**.

Enjoy exploring and clustering your SEO keywords!
