"""Streamlit dashboard for CTGAN synthetic credit-risk research."""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    BEST_MODEL_PATH,
    CLEAN_TRAIN_PATH,
    METRICS_PATH,
    RAW_DATA_PATH,
    SYNTHETIC_DATA_PATH,
)
from src.preprocessing import clean_dataframe
from streamlit_pages import data_analysis, explainability_page, overview, prediction, privacy_fairness_page, synthetic_comparison
from streamlit_pages.style import apply_dark_theme


st.set_page_config(
    page_title="CTGAN Credit Risk AI",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_dark_theme()


@st.cache_data(show_spinner=False)
def load_real_data() -> pd.DataFrame:
    """Load processed training data, or clean the raw Kaggle dataset for preview."""
    if CLEAN_TRAIN_PATH.exists():
        return pd.read_csv(CLEAN_TRAIN_PATH)
    if RAW_DATA_PATH.exists():
        return clean_dataframe(pd.read_csv(RAW_DATA_PATH).drop(columns=["Unnamed: 0"], errors="ignore"))
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_optional_csv(path: str) -> pd.DataFrame | None:
    """Load an optional CSV artifact."""
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_model() -> object | None:
    """Load the best trained model if present."""
    if BEST_MODEL_PATH.exists():
        return joblib.load(BEST_MODEL_PATH)
    return None


real_df = load_real_data()
if real_df.empty:
    st.error("No dataset found. Place `cs-training.csv` in `data/raw` or run preprocessing.")
    st.stop()

synthetic_df = load_optional_csv(str(SYNTHETIC_DATA_PATH)) if SYNTHETIC_DATA_PATH.exists() else None
metrics_df = load_optional_csv(str(METRICS_PATH)) if METRICS_PATH.exists() else None
quality_df = load_optional_csv(str(METRICS_PATH.parent / "synthetic_quality_metrics.csv"))
model = load_model()

with st.sidebar:
    st.title("Credit Risk Lab")
    page = st.radio(
        "Navigation",
        [
            "Project Overview",
            "Data Analysis",
            "Synthetic Data Comparison",
            "Credit Risk Prediction",
            "Explainability",
            "Privacy & Fairness Metrics",
        ],
    )
    st.divider()
    st.caption("Pipeline")
    st.code("python -m src.pipeline --ctgan-epochs 300", language="bash")

if page == "Project Overview":
    overview.render(real_df, synthetic_df, metrics_df)
elif page == "Data Analysis":
    data_analysis.render(real_df)
elif page == "Synthetic Data Comparison":
    synthetic_comparison.render(real_df, synthetic_df, quality_df)
elif page == "Credit Risk Prediction":
    prediction.render(model, real_df)
elif page == "Explainability":
    explainability_page.render()
else:
    privacy_fairness_page.render()
