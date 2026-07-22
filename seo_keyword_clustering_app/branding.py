"""
Brand theme for Felipe Saenz Streamlit tools.

Applies a consistent visual identity across all apps:
- Montserrat typography
- Felipe Saenz color palette
- Branded header, project description panels, metric guides and signature.

Usage inside an app:

    from branding import apply_branding, brand_header, project_panel, metric_guide, signature

    apply_branding()
    brand_header("Tool Name", "One-line subtitle describing what it does.")
    ...
    signature()
"""

import streamlit as st

# --- Immutable brand palette ------------------------------------------------
BG = "#FFFFFF"
BG_SOFT = "#EEF0FF"
TEXT = "#1A1A2E"
TEXT_MUTED = "#6B7280"
ACCENT = "#3B82F6"
ACCENT_DARK = "#6366F1"
DIVIDER = "#E0E0E0"
HANDLE = "@felipesaenzmkt"


def apply_branding() -> None:
    """Inject the Montserrat font and brand styling into the Streamlit app."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,400;0,500;0,600;0,700;0,800;0,900;1,700&display=swap');

        :root {{
            --brand-bg: {BG};
            --brand-bg-soft: {BG_SOFT};
            --brand-text: {TEXT};
            --brand-muted: {TEXT_MUTED};
            --brand-accent: {ACCENT};
            --brand-accent-dark: {ACCENT_DARK};
            --brand-divider: {DIVIDER};
        }}

        html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {{
            font-family: 'Montserrat', sans-serif !important;
            color: var(--brand-text);
        }}

        .stApp {{ background-color: var(--brand-bg); }}

        h1, h2, h3, h4 {{
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 800 !important;
            color: var(--brand-text) !important;
            letter-spacing: -0.01em;
        }}

        p, li, span, label, .stMarkdown {{
            font-family: 'Montserrat', sans-serif !important;
        }}

        /* Primary buttons and download buttons */
        .stButton > button, .stDownloadButton > button {{
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            border: 1px solid var(--brand-accent) !important;
            background: var(--brand-accent) !important;
            color: #FFFFFF !important;
            transition: filter .15s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            filter: brightness(1.08);
            border-color: var(--brand-accent-dark) !important;
            background: var(--brand-accent-dark) !important;
        }}

        /* Metric cards */
        [data-testid="stMetric"] {{
            background: var(--brand-bg-soft);
            border: 1px solid var(--brand-divider);
            border-radius: 12px;
            padding: 14px 16px;
        }}
        [data-testid="stMetricValue"] {{
            font-weight: 800 !important;
            color: var(--brand-accent-dark) !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--brand-muted) !important;
            font-weight: 500 !important;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: var(--brand-bg-soft);
            border-right: 1px solid var(--brand-divider);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def brand_header(title: str, subtitle: str = "", kicker: str = "SEO & Marketing Tools") -> None:
    """Render a branded header block with kicker, title and subtitle."""
    accent_word = ""
    words = title.split()
    if len(words) > 1:
        # Emphasise the last word in accent italic, on-brand.
        head = " ".join(words[:-1])
        accent_word = words[-1]
        title_html = (
            f"{head} <span style='font-style:italic;color:{ACCENT};'>{accent_word}</span>"
        )
    else:
        title_html = f"<span style='color:{ACCENT};'>{title}</span>"

    subtitle_html = (
        f"<p style='margin:6px 0 0;font-size:1.02rem;color:{TEXT_MUTED};font-weight:500;'>{subtitle}</p>"
        if subtitle
        else ""
    )

    st.markdown(
        f"""
        <div style="padding:18px 0 10px;border-bottom:2px solid {BG_SOFT};margin-bottom:18px;">
            <div style="font-size:.72rem;font-weight:700;letter-spacing:.14em;
                        text-transform:uppercase;color:{ACCENT_DARK};">{kicker}</div>
            <h1 style="margin:4px 0 0;font-size:2.15rem;font-weight:900;line-height:1.1;
                       color:{TEXT};">{title_html}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def project_panel(description: str, points=None) -> None:
    """Render a branded 'About this tool' panel visible to end users."""
    bullets = ""
    if points:
        items = "".join(
            f"<li style='margin:4px 0;color:{TEXT};'>{p}</li>" for p in points
        )
        bullets = f"<ul style='margin:10px 0 0;padding-left:18px;'>{items}</ul>"

    st.markdown(
        f"""
        <div style="background:{BG_SOFT};border:1px solid {DIVIDER};border-left:4px solid {ACCENT};
                    border-radius:12px;padding:16px 18px;margin:0 0 18px;">
            <div style="font-weight:700;color:{ACCENT_DARK};font-size:.8rem;
                        letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">
                About this tool</div>
            <div style="color:{TEXT};font-size:.97rem;line-height:1.5;">{description}</div>
            {bullets}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_guide(title: str, items: dict, expanded: bool = False) -> None:
    """Render an expander explaining what each metric means and how to read it.

    ``items`` maps a metric name -> plain-language explanation.
    """
    with st.expander(title, expanded=expanded):
        rows = ""
        for name, meaning in items.items():
            rows += (
                f"<div style='display:flex;gap:12px;padding:8px 0;"
                f"border-bottom:1px solid {DIVIDER};'>"
                f"<div style='min-width:190px;font-weight:700;color:{ACCENT_DARK};'>{name}</div>"
                f"<div style='color:{TEXT};font-size:.94rem;line-height:1.45;'>{meaning}</div>"
                f"</div>"
            )
        st.markdown(
            f"<div style='margin-top:2px;'>{rows}</div>",
            unsafe_allow_html=True,
        )


def signature() -> None:
    """Render the mandatory brand signature at the bottom of the app."""
    st.markdown(
        f"""
        <div style="margin-top:36px;padding-top:14px;border-top:1px solid {DIVIDER};
                    display:flex;justify-content:space-between;align-items:center;
                    font-family:'Montserrat',sans-serif;font-size:13px;color:{TEXT_MUTED};">
            <span>Built by Felipe Saenz</span>
            <span style="font-weight:600;">{HANDLE}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
