"""Cloud-safe Streamlit app for CTGAN credit risk research.

This file is intentionally self-contained for Streamlit Cloud deployment. It
uses uploaded/local artifacts when available and falls back to reproducible demo
data so the dashboard, evaluation, and prediction pages always work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Credit Risk CTGAN Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TARGET = "SeriousDlqin2yrs"

NUMERIC_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + ["age_group", "income_band", "debt_level"]
PURPLE = "#a855f7"
PINK = "#d946ef"
LAVENDER = "#c084fc"
BLUE = "#38bdf8"
RED = "#fb7185"
GREEN = "#10b981"


def apply_style() -> None:
    """Apply Power BI-style dark dashboard CSS."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #050711;
            --panel: #111426;
            --muted: #a8a9c9;
            --text: #f6f2ff;
            --line: rgba(209, 190, 255, .16);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(168,85,247,.16), transparent 30%),
                radial-gradient(circle at bottom right, rgba(56,189,248,.10), transparent 26%),
                linear-gradient(135deg, #050711 0%, #0b1020 52%, #160b2b 100%);
            color: var(--text);
        }
        header[data-testid="stHeader"] { background: transparent; height: 0; }
        header[data-testid="stHeader"] > div, footer, #MainMenu { display: none; }
        [data-testid="stSidebar"] {
            background: rgba(5,7,17,.96);
            border-right: 1px solid var(--line);
        }
        .block-container {
            padding-top: .7rem;
            padding-bottom: 1.5rem;
            max-width: 1480px;
        }
        .dashboard-title {
            text-align: center;
            font-size: 23px;
            font-weight: 900;
            font-style: italic;
            text-decoration: underline;
            color: #ffffff;
            margin: 0 0 12px;
        }
        .subtle {
            color: #a8a9c9;
            font-size: 13px;
            text-align: center;
            margin-top: -6px;
            margin-bottom: 12px;
        }
        .section-panel {
            background: linear-gradient(180deg, rgba(19,23,45,.96), rgba(14,17,34,.94));
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 18px 36px rgba(0,0,0,.28);
        }
        .panel-title {
            color: #ffffff;
            font-weight: 800;
            font-size: 14px;
            text-align: center;
            margin-bottom: 8px;
        }
        .mini-kpi {
            background: linear-gradient(145deg, rgba(22,25,49,.98), rgba(13,17,34,.92));
            border: 1px solid rgba(209,190,255,.16);
            border-radius: 6px;
            padding: 13px 10px;
            min-height: 84px;
            text-align: center;
            margin-bottom: 8px;
        }
        .mini-kpi .value {
            font-size: 24px;
            font-weight: 850;
            color: #f6f2ff;
        }
        .mini-kpi .label {
            color: #a8a9c9;
            font-size: 12px;
            margin-top: 5px;
        }
        .risk-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .risk-table th, .risk-table td {
            border-bottom: 1px solid rgba(209,190,255,.14);
            padding: 8px 4px;
            color: #f6f2ff;
        }
        .risk-table th {
            color: #d9c8ff;
            font-size: 11px;
            text-transform: uppercase;
        }
        .filter-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
        }
        .filter-pill {
            background: rgba(43,28,72,.92);
            border: 1px solid rgba(209,190,255,.18);
            border-radius: 4px;
            padding: 10px 8px;
            color: #f6f2ff;
            text-align: center;
            font-size: 12px;
            font-weight: 700;
        }
        .pipeline-step {
            border: 1px solid rgba(209,190,255,.2);
            border-radius: 8px;
            padding: 12px;
            background: rgba(17,20,38,.78);
            min-height: 86px;
            text-align: center;
        }
        .pipeline-step strong { color: #f6f2ff; }
        .pipeline-step span { color: #a8a9c9; font-size: 12px; }
        .risk-high { color: #fb7185; font-weight: 850; }
        .risk-low { color: #38bdf8; font-weight: 850; }
        .stPlotlyChart {
            background: linear-gradient(180deg, rgba(19,23,45,.96), rgba(14,17,34,.94));
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 6px;
        }
        .modebar { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def title(text: str, subtitle: str | None = None) -> None:
    """Render page title."""
    st.markdown(f"<div class='dashboard-title'>{text}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='subtle'>{subtitle}</div>", unsafe_allow_html=True)


def panel_title(text: str) -> None:
    """Render panel heading."""
    st.markdown(f"<div class='panel-title'>{text}</div>", unsafe_allow_html=True)


def mini_kpi(label: str, value: str) -> None:
    """Render compact KPI card."""
    st.markdown(
        f"""
        <div class="mini-kpi">
          <div class="value">{value}</div>
          <div class="label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig: go.Figure, height: int = 240) -> go.Figure:
    """Apply shared chart styling."""
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=16, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f6f2ff", size=11),
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center", font=dict(size=9)),
        xaxis=dict(gridcolor="rgba(209,190,255,.12)", zerolinecolor="rgba(209,190,255,.14)"),
        yaxis=dict(gridcolor="rgba(209,190,255,.12)", zerolinecolor="rgba(209,190,255,.14)"),
    )
    return fig


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid."""
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered categorical features."""
    result = df.copy()
    result["age_group"] = pd.cut(
        result["age"],
        bins=[0, 30, 45, 60, 75, np.inf],
        labels=["under_30", "30_44", "45_59", "60_74", "75_plus"],
        right=False,
    ).astype(str)
    result["income_band"] = pd.cut(
        result["MonthlyIncome"],
        bins=[-np.inf, 2500, 5000, 8500, 14000, np.inf],
        labels=["very_low", "low", "middle", "upper_middle", "high"],
    ).astype(str)
    result["debt_level"] = pd.cut(
        result["DebtRatio"],
        bins=[-np.inf, 0.2, 0.5, 1.0, np.inf],
        labels=["low", "moderate", "high", "severe"],
    ).astype(str)
    return result


def score_records(df: pd.DataFrame) -> pd.Series:
    """Score records with a transparent credit-risk formula.

    This is the deployed fallback predictor. The research repo can replace it
    with the trained XGBoost model locally, but this keeps Streamlit Cloud
    functional without pickled model artifacts.
    """
    income = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median()).clip(lower=0)
    score = (
        2.15 * df["RevolvingUtilizationOfUnsecuredLines"].fillna(0)
        + 1.55 * df["DebtRatio"].fillna(0)
        + 0.18 * df["NumberOfTime30-59DaysPastDueNotWorse"].fillna(0)
        + 0.32 * df["NumberOfTimes90DaysLate"].fillna(0)
        + 0.22 * df["NumberOfTime60-89DaysPastDueNotWorse"].fillna(0)
        + 0.012 * np.maximum(df["age"].fillna(45) - 55, 0)
        - 0.000045 * income
        - 1.10
    )
    return pd.Series(sigmoid(score), index=df.index).clip(0, 1)


@st.cache_data(show_spinner=False)
def create_demo_data(rows: int = 8_000) -> pd.DataFrame:
    """Create reproducible demo credit data for deployment."""
    rng = np.random.default_rng(42)
    age = rng.normal(48, 14, rows).clip(21, 85)
    income = rng.lognormal(mean=10.15, sigma=0.58, size=rows).clip(800, 30_000)
    debt_ratio = rng.beta(2.4, 5.2, rows).clip(0, 1.8)
    utilization = rng.beta(1.55, 4.4, rows).clip(0, 1.8)

    df = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": utilization,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": rng.poisson(0.22, rows),
            "DebtRatio": debt_ratio,
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": rng.poisson(8, rows).clip(1, 35),
            "NumberOfTimes90DaysLate": rng.poisson(0.08, rows),
            "NumberRealEstateLoansOrLines": rng.poisson(1.1, rows).clip(0, 8),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.poisson(0.08, rows),
            "NumberOfDependents": rng.poisson(1.1, rows).clip(0, 7),
        }
    )
    probability = score_records(df)
    df[TARGET] = rng.binomial(1, probability)
    return add_segments(df)


def clean_uploaded_or_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw or uploaded Give Me Some Credit-like dataframe."""
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore").copy()
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    if TARGET not in df.columns:
        df[TARGET] = (score_records(df) >= 0.5).astype(int)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    return add_segments(df)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load project data if available, otherwise demo data."""
    for path in [DATA_DIR / "processed" / "train_clean.csv", DATA_DIR / "raw" / "cs-training.csv"]:
        if path.exists():
            return clean_uploaded_or_raw(pd.read_csv(path))
    return create_demo_data()


@st.cache_data(show_spinner=False)
def load_synthetic_data() -> pd.DataFrame:
    """Load synthetic data or generate synthetic-like fallback samples."""
    path = DATA_DIR / "synthetic_data.csv"
    if path.exists():
        return clean_uploaded_or_raw(pd.read_csv(path))
    return generate_synthetic(load_data(), rows=4_000)


def generate_synthetic(real_df: pd.DataFrame, rows: int = 5_000) -> pd.DataFrame:
    """Generate lightweight synthetic samples for the app demo."""
    rng = np.random.default_rng(123)
    sampled = real_df.sample(n=rows, replace=True, random_state=123).reset_index(drop=True)
    synthetic = sampled.copy()
    noise_scale = {
        "RevolvingUtilizationOfUnsecuredLines": 0.08,
        "age": 4.0,
        "DebtRatio": 0.07,
        "MonthlyIncome": 900.0,
        "NumberOfOpenCreditLinesAndLoans": 1.5,
        "NumberOfDependents": 0.6,
    }
    for col, scale in noise_scale.items():
        synthetic[col] = synthetic[col] + rng.normal(0, scale, rows)
    for col in [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]:
        synthetic[col] = np.maximum(0, np.round(synthetic[col] + rng.poisson(0.05, rows))).astype(int)
    synthetic["age"] = synthetic["age"].clip(21, 90)
    synthetic["MonthlyIncome"] = synthetic["MonthlyIncome"].clip(500, 35_000)
    synthetic["DebtRatio"] = synthetic["DebtRatio"].clip(0, 2.0)
    synthetic["RevolvingUtilizationOfUnsecuredLines"] = synthetic[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(0, 2.0)
    synthetic[TARGET] = (score_records(synthetic) >= rng.uniform(0.35, 0.75, rows)).astype(int)
    return add_segments(synthetic)


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    """Load model metrics or provide research-run defaults."""
    path = REPORT_DIR / "model_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(
        [
            ["real", "xgboost", 0.934, 0.611, 0.027, 0.053, 0.807],
            ["real", "logistic_regression", 0.737, 0.166, 0.727, 0.270, 0.796],
            ["real", "random_forest", 0.797, 0.191, 0.628, 0.293, 0.791],
            ["hybrid", "xgboost", 0.933, 0.000, 0.000, 0.000, 0.783],
            ["synthetic", "xgboost", 0.443, 0.064, 0.532, 0.114, 0.477],
        ],
        columns=["training_data", "model", "accuracy", "precision", "recall", "f1", "roc_auc"],
    )


def distribution_overlap(real: pd.Series, synth: pd.Series, bins: int = 30) -> float:
    """Compute histogram intersection overlap."""
    combined = pd.concat([real, synth]).dropna()
    if combined.empty or combined.min() == combined.max():
        return 1.0
    hist_range = (combined.min(), combined.max())
    real_hist, edges = np.histogram(real.dropna(), bins=bins, range=hist_range, density=True)
    synth_hist, _ = np.histogram(synth.dropna(), bins=edges, density=True)
    return float(np.sum(np.minimum(real_hist, synth_hist) * np.diff(edges)))


def synthetic_quality(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> pd.DataFrame:
    """Compute lightweight synthetic fidelity metrics."""
    rows = []
    for col in NUMERIC_COLUMNS:
        real = real_df[col].dropna()
        synth = synth_df[col].dropna()
        rows.append(
            {
                "feature": col,
                "mean_delta": abs(real.mean() - synth.mean()),
                "std_delta": abs(real.std() - synth.std()),
                "distribution_overlap": distribution_overlap(real, synth),
            }
        )
    return pd.DataFrame(rows)


def privacy_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict[str, float]:
    """Approximate privacy metrics without heavy dependencies."""
    n = min(1500, len(real_df), len(synth_df))
    real = real_df[NUMERIC_COLUMNS].sample(n=n, random_state=7).to_numpy(float)
    synth = synth_df[NUMERIC_COLUMNS].sample(n=n, random_state=11).to_numpy(float)
    real = (real - real.mean(axis=0)) / (real.std(axis=0) + 1e-6)
    synth = (synth - synth.mean(axis=0)) / (synth.std(axis=0) + 1e-6)
    distances = np.sqrt(((synth[:, None, :] - real[None, :, :]) ** 2).sum(axis=2))
    closest = distances.min(axis=1)
    threshold = np.percentile(closest, 5)
    return {
        "dcr_mean": float(np.mean(closest)),
        "dcr_median": float(np.median(closest)),
        "dcr_min": float(np.min(closest)),
        "membership_risk_rate": float(np.mean(closest <= threshold)),
    }


def fairness_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute age-group fairness metrics using fallback predictions."""
    scored = df.copy()
    scored["probability"] = score_records(scored)
    scored["prediction"] = (scored["probability"] >= 0.5).astype(int)
    group = (
        scored.groupby("age_group", observed=True)
        .agg(
            count=(TARGET, "size"),
            actual_default_rate=(TARGET, "mean"),
            prediction_rate=("prediction", "mean"),
            avg_probability=("probability", "mean"),
        )
        .reset_index()
    )
    positive_groups = []
    for age_group, part in scored.groupby("age_group", observed=True):
        positives = part[part[TARGET] == 1]
        if len(positives):
            positive_groups.append((age_group, positives["prediction"].mean()))
    tpr_values = [value for _, value in positive_groups]
    summary = {
        "demographic_parity_difference": float(group["prediction_rate"].max() - group["prediction_rate"].min()),
        "equal_opportunity_difference": float(max(tpr_values) - min(tpr_values)) if len(tpr_values) > 1 else 0.0,
    }
    return group, summary


def risk_table_html(df: pd.DataFrame) -> str:
    """Render risky segment table."""
    rows = []
    for group, part in df.groupby("debt_level", observed=True):
        rows.append((str(group).upper(), part[TARGET].mean()))
    body = "".join(
        f"<tr><td>{name}</td><td>{rate:.0%}</td></tr>"
        for name, rate in sorted(rows, key=lambda item: item[1], reverse=True)
    )
    return f"""
    <div class="section-panel">
      <div class="panel-title">Risky Debt Levels</div>
      <table class="risk-table">
        <thead><tr><th>Debt Level</th><th>Default Rate %</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    """


def filters_html(df: pd.DataFrame) -> str:
    """Render filter-like segment tiles."""
    labels = list(df["debt_level"].dropna().astype(str).str.upper().unique())
    labels += list(df["income_band"].dropna().astype(str).str.upper().unique())
    pills = "".join(f"<div class='filter-pill'>{label}</div>" for label in labels[:9])
    return f"<div class='section-panel'><div class='panel-title'>Risk Segments</div><div class='filter-grid'>{pills}</div></div>"


def render_overview(df: pd.DataFrame, synth_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    """Render Power BI-like overview dashboard."""
    title("CREDIT RISK DASHBOARD", "Synthetic financial data generation using CTGAN for credit risk assessment")
    best_auc = float(metrics_df["roc_auc"].max())
    default_rate = float(df[TARGET].mean())
    predicted_rate = float((score_records(df) >= 0.5).mean())

    left, center, right = st.columns([1.05, 2.25, 1.2], gap="small")
    with left:
        st.markdown(risk_table_html(df), unsafe_allow_html=True)
        panel_title("Defaults by Debt Level")
        debt = (
            df.groupby("debt_level", observed=True)[TARGET]
            .agg(default_rate="mean", records="size")
            .reset_index()
            .sort_values("default_rate", ascending=False)
        )
        fig = go.Figure()
        fig.add_bar(x=debt["debt_level"], y=debt["records"], marker_color=PURPLE, name="Applicants")
        fig.add_scatter(
            x=debt["debt_level"],
            y=debt["default_rate"],
            mode="lines+markers",
            line=dict(color=LAVENDER, width=3),
            name="Default Rate",
            yaxis="y2",
        )
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%"))
        st.plotly_chart(chart_layout(fig, 220), use_container_width=True, config={"displayModeBar": False})

    with center:
        panel_title("All Applicants by Age Group")
        age = df.groupby("age_group", observed=True).size().reset_index(name="records")
        fig = px.pie(
            age,
            values="records",
            names="age_group",
            hole=0.55,
            color_discrete_sequence=[PURPLE, PINK, LAVENDER, BLUE, RED],
        )
        fig.update_traces(textposition="outside", textinfo="label+percent", marker=dict(line=dict(color="#111426", width=2)))
        st.plotly_chart(chart_layout(fig, 285), use_container_width=True, config={"displayModeBar": False})

        c1, c2 = st.columns(2, gap="small")
        with c1:
            panel_title("Defaults by Income Band")
            income = (
                df.groupby("income_band", observed=True)[TARGET]
                .agg(default_rate="mean", records="size")
                .reset_index()
            )
            fig = go.Figure()
            fig.add_bar(x=income["income_band"], y=income["records"], marker_color=PURPLE, name="Applicants")
            fig.add_scatter(
                x=income["income_band"],
                y=income["default_rate"],
                mode="lines+markers",
                line=dict(color=LAVENDER, width=3),
                name="Default Rate",
                yaxis="y2",
            )
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%"))
            st.plotly_chart(chart_layout(fig, 235), use_container_width=True, config={"displayModeBar": False})
        with c2:
            panel_title("Risk Score")
            score = int(max(0, min(100, default_rate * 100 + (1 - best_auc) * 100)))
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#a8a9c9"},
                        "bar": {"color": PURPLE},
                        "bgcolor": "rgba(255,255,255,.06)",
                        "steps": [
                            {"range": [0, 35], "color": "rgba(56,189,248,.18)"},
                            {"range": [35, 70], "color": "rgba(168,85,247,.20)"},
                            {"range": [70, 100], "color": "rgba(251,113,133,.20)"},
                        ],
                    },
                    number={"font": {"color": "#f6f2ff", "size": 30}},
                )
            )
            st.plotly_chart(chart_layout(fig, 235), use_container_width=True, config={"displayModeBar": False})

    with right:
        mini_kpi("Good Loans %", f"{1 - default_rate:.2%}")
        mini_kpi("Total Records", f"{len(df) / 1000:.1f}K")
        mini_kpi("Default Rate %", f"{default_rate:.2%}")
        mini_kpi("Predicted Defaults %", f"{predicted_rate:.2%}")
        mini_kpi("Synthetic Records", f"{len(synth_df) / 1000:.1f}K")
        mini_kpi("Best ROC-AUC", f"{best_auc:.3f}")

    b1, b2, b3 = st.columns([1.35, 1.05, 1.3], gap="small")
    with b1:
        st.markdown(filters_html(df), unsafe_allow_html=True)
    with b2:
        panel_title("Model Leaderboard")
        display = metrics_df[["training_data", "model", "roc_auc", "recall"]].head(5).copy()
        st.dataframe(display, use_container_width=True, hide_index=True)
    with b3:
        panel_title("Age Group Default Rate")
        age_default = df.groupby("age_group", observed=True)[TARGET].mean().reset_index()
        fig = px.bar(age_default, x="age_group", y=TARGET, color=TARGET, color_continuous_scale=[BLUE, PURPLE, RED])
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(chart_layout(fig, 200), use_container_width=True, config={"displayModeBar": False})


def render_prediction(df: pd.DataFrame) -> None:
    """Render prediction workflows."""
    title("CREDIT RISK PREDICTION", "Single applicant scoring and batch CSV prediction")
    mode = st.radio("Prediction mode", ["Single applicant", "Batch CSV upload"], horizontal=True)

    if mode == "Batch CSV upload":
        uploaded = st.file_uploader("Upload applicant CSV", type=["csv"])
        if uploaded:
            batch = clean_uploaded_or_raw(pd.read_csv(uploaded))
            batch["default_probability"] = score_records(batch)
            batch["risk_prediction"] = np.where(batch["default_probability"] >= 0.5, "High Risk", "Lower Risk")
            st.dataframe(batch, use_container_width=True)
            st.download_button("Download predictions", batch.to_csv(index=False), "credit_risk_predictions.csv")
        else:
            st.info("Upload a CSV with Give Me Some Credit columns, or at least age, DebtRatio, MonthlyIncome, and utilization.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        utilization = st.slider("Revolving Utilization", 0.0, 2.0, 0.55, 0.01)
        age = st.number_input("Age", 21, 90, 45)
        past_30 = st.number_input("30-59 Days Past Due", 0, 20, 0)
        debt = st.slider("Debt Ratio", 0.0, 2.0, 0.45, 0.01)
    with c2:
        income = st.number_input("Monthly Income", 0.0, 50_000.0, 6500.0, step=250.0)
        open_lines = st.number_input("Open Credit Lines", 0, 60, 8)
        past_90 = st.number_input("90 Days Late", 0, 20, 0)
        real_estate = st.number_input("Real Estate Loans", 0, 20, 1)
    with c3:
        past_60 = st.number_input("60-89 Days Past Due", 0, 20, 0)
        dependents = st.number_input("Dependents", 0, 12, 1)
        threshold = st.slider("Risk Threshold", 0.05, 0.95, 0.50, 0.01)

    record = pd.DataFrame(
        [
            {
                "RevolvingUtilizationOfUnsecuredLines": utilization,
                "age": age,
                "NumberOfTime30-59DaysPastDueNotWorse": past_30,
                "DebtRatio": debt,
                "MonthlyIncome": income,
                "NumberOfOpenCreditLinesAndLoans": open_lines,
                "NumberOfTimes90DaysLate": past_90,
                "NumberRealEstateLoansOrLines": real_estate,
                "NumberOfTime60-89DaysPastDueNotWorse": past_60,
                "NumberOfDependents": dependents,
            }
        ]
    )
    probability = float(score_records(record).iloc[0])
    prediction = "High Risk" if probability >= threshold else "Lower Risk"
    color_class = "risk-high" if probability >= threshold else "risk-low"

    left, right = st.columns([1, 1], gap="small")
    with left:
        st.markdown(
            f"""
            <div class="section-panel">
              <div class="panel-title">Prediction Result</div>
              <h1 class="{color_class}" style="text-align:center;">{prediction}</h1>
              <h2 style="text-align:center;color:#f6f2ff;">{probability:.2%}</h2>
              <p style="text-align:center;color:#a8a9c9;">Estimated probability of serious delinquency in 2 years.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": RED if probability >= threshold else BLUE},
                    "steps": [
                        {"range": [0, threshold * 100], "color": "rgba(56,189,248,.18)"},
                        {"range": [threshold * 100, 100], "color": "rgba(251,113,133,.22)"},
                    ],
                },
                number={"suffix": "%", "font": {"color": "#f6f2ff"}},
            )
        )
        st.plotly_chart(chart_layout(fig, 260), use_container_width=True, config={"displayModeBar": False})

    importance = pd.DataFrame(
        {
            "feature": ["Utilization", "Debt Ratio", "90 Days Late", "30-59 Past Due", "Monthly Income", "Age"],
            "contribution": [
                2.15 * utilization,
                1.55 * debt,
                0.32 * past_90,
                0.18 * past_30,
                -0.000045 * income,
                0.012 * max(age - 55, 0),
            ],
        }
    ).sort_values("contribution", key=abs, ascending=False)
    panel_title("Local Explanation")
    fig = px.bar(
        importance,
        x="contribution",
        y="feature",
        orientation="h",
        color="contribution",
        color_continuous_scale=[BLUE, PURPLE, RED],
    )
    st.plotly_chart(chart_layout(fig, 280), use_container_width=True, config={"displayModeBar": False})


def render_evaluation(df: pd.DataFrame, synth_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    """Render comprehensive evaluation page."""
    title("COMPREHENSIVE EVALUATION", "ML performance, privacy audit, fairness, and statistical fidelity")

    steps = [
        ("1. Data Preprocessing", "Missing values, outliers, engineered bands"),
        ("2. CTGAN Generation", "Conditional synthetic minority records"),
        ("3. ML Training", "Real, synthetic, and hybrid datasets"),
        ("4. Comprehensive Evaluation", "Performance, privacy, fairness, fidelity"),
        ("5. Best Model Selection", "ROC-AUC and recall trade-off"),
    ]
    cols = st.columns(len(steps))
    for col, (name, desc) in zip(cols, steps):
        with col:
            st.markdown(f"<div class='pipeline-step'><strong>{name}</strong><br><span>{desc}</span></div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        mini_kpi("Best ROC-AUC", f"{metrics_df['roc_auc'].max():.3f}")
    with k2:
        mini_kpi("Best Recall", f"{metrics_df['recall'].max():.3f}")
    with k3:
        mini_kpi("Synthetic Rows", f"{len(synth_df):,}")
    with k4:
        mini_kpi("Default Rate", f"{df[TARGET].mean():.2%}")

    p1, p2 = st.columns(2, gap="small")
    with p1:
        panel_title("ML Performance")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    with p2:
        panel_title("ROC-AUC by Experiment")
        fig = px.bar(
            metrics_df,
            x="model",
            y="roc_auc",
            color="training_data",
            barmode="group",
            color_discrete_sequence=[PURPLE, BLUE, RED, GREEN],
        )
        st.plotly_chart(chart_layout(fig, 330), use_container_width=True, config={"displayModeBar": False})

    q = synthetic_quality(df, synth_df)
    p3, p4 = st.columns(2, gap="small")
    with p3:
        panel_title("Statistical Fidelity")
        st.dataframe(q, use_container_width=True, hide_index=True)
    with p4:
        panel_title("Distribution Overlap")
        fig = px.bar(q, x="distribution_overlap", y="feature", orientation="h", color="distribution_overlap", color_continuous_scale=[RED, PURPLE, BLUE])
        st.plotly_chart(chart_layout(fig, 380), use_container_width=True, config={"displayModeBar": False})


def render_privacy_fairness(df: pd.DataFrame, synth_df: pd.DataFrame) -> None:
    """Render privacy and fairness metrics."""
    title("PRIVACY & FAIRNESS AUDIT", "DCR, membership risk, demographic parity, equal opportunity")
    privacy = privacy_metrics(df, synth_df)
    group, summary = fairness_metrics(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mini_kpi("Median DCR", f"{privacy['dcr_median']:.3f}")
    with c2:
        mini_kpi("Min DCR", f"{privacy['dcr_min']:.3f}")
    with c3:
        mini_kpi("Membership Risk", f"{privacy['membership_risk_rate']:.2%}")
    with c4:
        mini_kpi("Eq. Opportunity Diff", f"{summary['equal_opportunity_difference']:.3f}")

    p1, p2 = st.columns(2, gap="small")
    with p1:
        panel_title("Fairness by Age Group")
        st.dataframe(group, use_container_width=True, hide_index=True)
    with p2:
        panel_title("Group Prediction Rate")
        fig = px.bar(group, x="age_group", y="prediction_rate", color="prediction_rate", color_continuous_scale=[BLUE, PURPLE, RED])
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(chart_layout(fig, 330), use_container_width=True, config={"displayModeBar": False})


def render_data_analysis(df: pd.DataFrame) -> None:
    """Render EDA page."""
    title("DATA ANALYSIS", "Exploratory analysis of credit-risk features")
    c1, c2, c3 = st.columns(3)
    with c1:
        mini_kpi("Rows", f"{len(df):,}")
    with c2:
        mini_kpi("Features", f"{len(FEATURE_COLUMNS)}")
    with c3:
        mini_kpi("Class Imbalance", f"{df[TARGET].mean():.2%}")

    feature = st.selectbox("Feature", NUMERIC_COLUMNS, index=0)
    fig = px.histogram(df, x=feature, color=TARGET, marginal="box", color_discrete_sequence=[BLUE, PURPLE])
    st.plotly_chart(chart_layout(fig, 380), use_container_width=True, config={"displayModeBar": False})
    fig = px.imshow(df[NUMERIC_COLUMNS].corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(chart_layout(fig, 520), use_container_width=True, config={"displayModeBar": False})


def render_synthetic(df: pd.DataFrame, synth_df: pd.DataFrame) -> None:
    """Render synthetic comparison page."""
    title("SYNTHETIC DATA COMPARISON", "Original vs synthetic distributions and correlations")
    rows = st.slider("Generate synthetic records in app", 500, 20_000, len(synth_df), 500)
    if st.button("Regenerate Synthetic Sample", type="primary"):
        st.session_state["synthetic_df"] = generate_synthetic(df, rows=rows)
    synth_df = st.session_state.get("synthetic_df", synth_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        mini_kpi("Real Rows", f"{len(df):,}")
    with c2:
        mini_kpi("Synthetic Rows", f"{len(synth_df):,}")
    with c3:
        mini_kpi("Synthetic Default Rate", f"{synth_df[TARGET].mean():.2%}")

    feature = st.selectbox("Compare feature", NUMERIC_COLUMNS, index=1)
    plot_df = pd.concat([df[[feature]].assign(source="Real"), synth_df[[feature]].assign(source="Synthetic")])
    fig = px.histogram(plot_df, x=feature, color="source", barmode="overlay", opacity=0.62, color_discrete_sequence=[BLUE, PURPLE])
    st.plotly_chart(chart_layout(fig, 380), use_container_width=True, config={"displayModeBar": False})

    q = synthetic_quality(df, synth_df)
    st.dataframe(q, use_container_width=True, hide_index=True)


def render_explainability() -> None:
    """Render explainability page."""
    title("EXPLAINABLE AI", "Global and local risk-driver explanations")
    summary = FIGURE_DIR / "shap_summary.png"
    importance = FIGURE_DIR / "shap_feature_importance.png"
    if summary.exists():
        st.image(str(summary), caption="SHAP Summary", use_container_width=True)
    if importance.exists():
        st.image(str(importance), caption="SHAP Feature Importance", use_container_width=True)
    if not summary.exists() and not importance.exists():
        demo = pd.DataFrame(
            {
                "feature": [
                    "RevolvingUtilizationOfUnsecuredLines",
                    "DebtRatio",
                    "NumberOfTimes90DaysLate",
                    "MonthlyIncome",
                    "NumberOfTime30-59DaysPastDueNotWorse",
                    "age",
                ],
                "importance": [0.32, 0.26, 0.17, 0.12, 0.08, 0.05],
            }
        )
        panel_title("Global Feature Importance")
        fig = px.bar(demo, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=[BLUE, PURPLE, RED])
        st.plotly_chart(chart_layout(fig, 420), use_container_width=True, config={"displayModeBar": False})


def main() -> None:
    """Run the app."""
    apply_style()
    df = load_data()
    synth_df = st.session_state.get("synthetic_df", load_synthetic_data())
    metrics_df = load_metrics()

    with st.sidebar:
        st.title("Credit Risk Lab")
        page = st.radio(
            "Navigation",
            [
                "Project Overview",
                "Credit Risk Prediction",
                "Comprehensive Evaluation",
                "Data Analysis",
                "Synthetic Data Comparison",
                "Explainability",
                "Privacy & Fairness Metrics",
            ],
        )
        st.divider()
        uploaded = st.file_uploader("Optional: upload dataset", type=["csv"])
        if uploaded is not None:
            st.session_state["uploaded_df"] = clean_uploaded_or_raw(pd.read_csv(uploaded))
            st.success("Dataset loaded for this session.")

    df = st.session_state.get("uploaded_df", df)

    if page == "Project Overview":
        render_overview(df, synth_df, metrics_df)
    elif page == "Credit Risk Prediction":
        render_prediction(df)
    elif page == "Comprehensive Evaluation":
        render_evaluation(df, synth_df, metrics_df)
    elif page == "Data Analysis":
        render_data_analysis(df)
    elif page == "Synthetic Data Comparison":
        render_synthetic(df, synth_df)
    elif page == "Explainability":
        render_explainability()
    else:
        render_privacy_fairness(df, synth_df)


if __name__ == "__main__":
    main()
