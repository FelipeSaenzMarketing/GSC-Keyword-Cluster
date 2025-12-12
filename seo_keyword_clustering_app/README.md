# SEO Keyword Clustering App

This repository contains a Streamlit application for clustering SEO keywords based on their semantic similarity.  
The tool is designed for SEO analysis and works with Google Search Console–like datasets in English and German.

---

## Features

- Upload **.xls / .xlsx / .csv** files with Search Console data.
- Automatic detection of the keyword column in **English** or **German**:
  - **English**: `Top queries`, `Clicks`, `Impressions`, `CTR`
  - **German**: `Suchanfrage`, `Klicks`, `Impressionen`, `CTR`
- Keyword vectorization using **TF-IDF**.
- Keyword clustering using **KMeans**.
- Automatic **cluster naming** based on dominant n-grams (top TF-IDF terms per cluster).
- **Interactive t-SNE semantic map** of keyword clusters (Plotly).
- Downloadable output file including additional:
  - `cluster_id`
  - `cluster_name`

---

## Dependencies

This app requires the following Python libraries:

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- plotly (for interactive visualizations)
- openpyxl (for Excel file support)

All dependencies are listed in the `requirements.txt` file.

---

## Installation

Create and activate a virtual environment (recommended), then install the dependencies:

```bash
pip install -r requirements.txt
