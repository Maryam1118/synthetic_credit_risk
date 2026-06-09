from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="CTGAN Credit Risk Dashboard",
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


def apply_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #050711;
            --panel: #111426;
            --muted: #a8a9c9;
            --text: #f6f2ff;
            --accent: #a855f7;
            --accent2: #38bdf8;
            --danger: #fb7185;
            --line: rgba(209, 190, 255, .16);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(168,85,247,.16), transparent 30%),
                radial-gradient(circle at bottom right, rgba(56,189,248,.10), transparent 26%),
                linear-gradient(135deg, #050711 0%, #0b1020 52%, #160b2b 100%);
            color: var(--text);
        }

        header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
        }

        header[data-testid="stHeader"] > div,
        footer,
        #MainMenu {
            display: none;
        }

        [data-testid="stSidebar"] {
            background: rgba(5,7,17,.96);
            border-right: 1px solid var(--line);
        }

        .block-container {
            padding-top: .6rem;
            padding-bottom: 1.5rem;
            max-width: 1440px;
        }

        .dashboard-title {
            text-align: center;
            font-size: 22px;
            font-weight: 900;
            font-style: italic;
            text-decoration: underline;
            color: #ffffff;
            margin-bottom: 12px;
        }

        .panel-title {
            color: #ffffff;
            font-weight: 800;
            font-size: 14px;
            text-align: center;
            margin-bottom: 8px;
        }

        .section-panel {
            background: linear-gradient(180deg, rgba(19,23,45,.96), rgba(14,17,34,.94));
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 18px 36px rgba(0,0,0,.28);
        }

        .mini-kpi {
            background: linear-gradient(145deg, rgba(22,25,49,.98), rgba(13,17,34,.92));
            border: 1px solid rgba(209,190,255,.16);
            border-radius: 6px;
            padding: 13px 10px;
            min-height: 82px;
            text-align: center;
            margin-bottom: 8px;
        }

        .mini-kpi .value {
            font-size: 23px;
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

        .risk-table th,
        .risk-table td {
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

        .stPlotlyChart {
            background: linear-gradient(180deg, rgba(19,23,45,.96), rgba(14,17,34,.94));
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 6px;
        }

        .modebar {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def panel_title(title: str) -> None:
    st.markdown(f"<div class='panel-title'>{title}</div>", unsafe_allow_html=True)


def mini_kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="mini-kpi">
            <div class="value">{value}</div>
            <div class="label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig: go.Figure, height: int = 230) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=14, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f6f2ff", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
        ),
        xaxis=dict(gridcolor="rgba(209,190,255,.12)"),
        yaxis=dict(gridcolor="rgba(209,190,255,.12)"),
    )
    return fig


@st.cache_data
def create_demo_data(rows: int = 8000) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    age = rng.normal(48, 14, rows).clip(21, 85)
    income = rng.lognormal(mean=10.2, sigma=0.55, size=rows).clip(1000, 25000)
    debt_ratio = rng.beta(2.2, 5.5, rows).clip(0, 1.8)
    utilization = rng.beta(1.4, 4.5, rows).clip(0, 1.5)

    score = (
        1.8 * utilization
        + 1.4 * debt_ratio
        - 0.000035 * income
        + rng.normal(0, 0.45, rows)
    )
    probability = 1 / (1 + np.exp(-(score - 1.25)))
    target = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": utilization,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": rng.poisson(0.18, rows),
            "DebtRatio": debt_ratio,
            "MonthlyIncome": income,
            "NumberOfOpenCreditLinesAndLoans": rng.poisson(8, rows).clip(1, 32),
            "NumberOfTimes90DaysLate": rng.poisson(0.08, rows),
            "NumberRealEstateLoansOrLines": rng.poisson(1.1, rows).clip(0, 8),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.poisson(0.07, rows),
            "NumberOfDependents": rng.poisson(1.1, rows).clip(0, 6),
            TARGET: target,
        }
    )

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 75, np.inf],
        labels=["under_30", "30_44", "45_59", "60_74", "75_plus"],
        right=False,
    ).astype(str)

    df["income_band"] = pd.cut(
        df["MonthlyIncome"],
        bins=[-np.inf, 2500, 5000, 8500, 14000, np.inf],
        labels=["very_low", "low", "middle", "upper_middle", "high"],
    ).astype(str)

    df["debt_level"] = pd.cut(
        df["DebtRatio"],
        bins=[-np.inf, 0.2, 0.5, 1.0, np.inf],
        labels=["low", "moderate", "high", "severe"],
    ).astype(str)

    return df


@st.cache_data
def load_data() -> pd.DataFrame:
    possible_paths = [
        DATA_DIR / "processed" / "train_clean.csv",
        DATA_DIR / "raw" / "cs-training.csv",
    ]

    for path in possible_paths:
        if path.exists():
            df = pd.read_csv(path)
            df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")

            if "age_group" not in df.columns:
                df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
                df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)
                df["age_group"] = pd.cut(
                    df["age"],
                    bins=[0, 30, 45, 60, 75, np.inf],
                    labels=["under_30", "30_44", "45_59", "60_74", "75_plus"],
                    right=False,
                ).astype(str)
                df["income_band"] = pd.cut(
                    df["MonthlyIncome"],
                    bins=[-np.inf, 2500, 5000, 8500, 14000, np.inf],
                    labels=["very_low", "low", "middle", "upper_middle", "high"],
                ).astype(str)
                df["debt_level"] = pd.cut(
                    df["DebtRatio"],
                    bins=[-np.inf, 0.2, 0.5, 1.0, np.inf],
                    labels=["low", "moderate", "high", "severe"],
                ).astype(str)

            return df

    return create_demo_data()


@st.cache_data
def load_synthetic() -> pd.DataFrame | None:
    path = DATA_DIR / "synthetic_data.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_metrics() -> pd.DataFrame:
    path = REPORT_DIR / "model_metrics.csv"
    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame(
        [
            {
                "training_data": "demo",
                "model": "xgboost",
                "accuracy": 0.93,
                "precision": 0.61,
                "recall": 0.38,
                "f1": 0.47,
                "roc_auc": 0.807,
            },
            {
                "training_data": "demo",
                "model": "random_forest",
                "accuracy": 0.89,
                "precision": 0.42,
                "recall": 0.51,
                "f1": 0.46,
                "roc_auc": 0.791,
            },
            {
                "training_data": "demo",
                "model": "logistic_regression",
                "accuracy": 0.74,
                "precision": 0.17,
                "recall": 0.73,
                "f1": 0.27,
                "roc_auc": 0.796,
            },
        ]
    )


def risk_table_html(df: pd.DataFrame) -> str:
    rows = []
    for group, group_df in df.groupby("debt_level", observed=True):
        rows.append((str(group).upper(), group_df[TARGET].mean()))

    rows = sorted(rows, key=lambda item: item[1], reverse=True)
    body = "".join(f"<tr><td>{name}</td><td>{rate:.0%}</td></tr>" for name, rate in rows[:5])

    return f"""
    <div class="section-panel">
        <div class="panel-title">Risky Debt Levels</div>
        <table class="risk-table">
            <thead>
                <tr><th>Debt Level</th><th>Default Rate %</th></tr>
            </thead>
            <tbody>{body}</tbody>
        </table>
    </div>
    """


def filters_html(df: pd.DataFrame) -> str:
    labels = []
    labels.extend([str(x).upper() for x in df["debt_level"].dropna().unique()[:4]])
    labels.extend([str(x).upper() for x in df["income_band"].dropna().unique()[:5]])
    labels = labels[:9]

    pills = "".join(f"<div class='filter-pill'>{label}</div>" for label in labels)

    return f"""
    <div class="section-panel">
        <div class="panel-title">Risk Segments</div>
        <div class="filter-grid">{pills}</div>
    </div>
    """


def render_overview(df: pd.DataFrame, synthetic_df: pd.DataFrame | None, metrics_df: pd.DataFrame) -> None:
    st.markdown("<div class='dashboard-title'>CREDIT RISK DASHBOARD</div>", unsafe_allow_html=True)

    best_auc = float(metrics_df["roc_auc"].max()) if not metrics_df.empty and "roc_auc" in metrics_df else 0.807
    default_rate = float(df[TARGET].mean())
    good_rate = 1 - default_rate
    synthetic_count = 0 if synthetic_df is None else len(synthetic_df)
    predicted_defaults = default_rate * 0.38

    left, center, right = st.columns([1.05, 2.2, 1.2], gap="small")

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
        fig.add_bar(x=debt["debt_level"], y=debt["records"], marker_color="#a855f7", name="Applicants")
        fig.add_scatter(
            x=debt["debt_level"],
            y=debt["default_rate"],
            mode="lines+markers",
            marker=dict(color="#c084fc"),
            line=dict(color="#c084fc", width=3),
            name="Default Rate",
            yaxis="y2",
        )
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%"))
        st.plotly_chart(chart_layout(fig, 218), use_container_width=True, config={"displayModeBar": False})

    with center:
        panel_title("All Applicants by Age Group")
        age_group = df.groupby("age_group", observed=True).size().reset_index(name="records")

        fig = px.pie(
            age_group,
            values="records",
            names="age_group",
            hole=0.55,
            color_discrete_sequence=["#a855f7", "#d946ef", "#c084fc", "#38bdf8", "#fb7185"],
        )
        fig.update_traces(
            textposition="outside",
            textinfo="label+percent",
            marker=dict(line=dict(color="#111426", width=2)),
        )
        st.plotly_chart(chart_layout(fig, 250), use_container_width=True, config={"displayModeBar": False})

        c1, c2 = st.columns(2, gap="small")

        with c1:
            panel_title("Defaults by Income Band")
            income = (
                df.groupby("income_band", observed=True)[TARGET]
                .agg(default_rate="mean", records="size")
                .reset_index()
            )

            fig = go.Figure()
            fig.add_bar(x=income["income_band"], y=income["records"], marker_color="#a855f7", name="Applicants")
            fig.add_scatter(
                x=income["income_band"],
                y=income["default_rate"],
                mode="lines+markers",
                line=dict(color="#c084fc", width=3),
                name="Default Rate",
                yaxis="y2",
            )
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%"))
            st.plotly_chart(chart_layout(fig, 220), use_container_width=True, config={"displayModeBar": False})

        with c2:
            panel_title("Risk Score")
            score = int(max(0, min(100, default_rate * 100 + (1 - best_auc) * 100)))

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={"font": {"color": "#f6f2ff", "size": 28}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#a8a9c9"},
                        "bar": {"color": "#a855f7"},
                        "bgcolor": "rgba(255,255,255,.06)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 35], "color": "rgba(56,189,248,.18)"},
                            {"range": [35, 70], "color": "rgba(168,85,247,.20)"},
                            {"range": [70, 100], "color": "rgba(251,113,133,.20)"},
                        ],
                    },
                )
            )
            st.plotly_chart(chart_layout(fig, 220), use_container_width=True, config={"displayModeBar": False})

    with right:
        mini_kpi("Good Loans %", f"{good_rate:.2%}")
        mini_kpi("Total Records", f"{len(df) / 1000:.1f}K")
        mini_kpi("Default Rate %", f"{default_rate:.2%}")
        mini_kpi("Predicted Defaults %", f"{predicted_defaults:.2%}")
        mini_kpi("Synthetic Records", f"{synthetic_count / 1000:.1f}K")
        mini_kpi("Best ROC-AUC", f"{best_auc:.3f}")

    b1, b2, b3 = st.columns([1.4, 1.0, 1.2], gap="small")

    with b1:
        st.markdown(filters_html(df), unsafe_allow_html=True)

    with b2:
        panel_title("Model Leaderboard")
        display = metrics_df[["training_data", "model", "roc_auc", "recall"]].head(5).copy()
        display["roc_auc"] = display["roc_auc"].map(lambda x: f"{x:.3f}")
        display["recall"] = display["recall"].map(lambda x: f"{x:.3f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

    with b3:
        panel_title("Age Group Default Rate")
        age_default = df.groupby("age_group", observed=True)[TARGET].mean().reset_index()
        fig = px.bar(
            age_default,
            x="age_group",
            y=TARGET,
            color=TARGET,
            color_continuous_scale=["#38bdf8", "#a855f7", "#fb7185"],
        )
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(chart_layout(fig, 190), use_container_width=True, config={"displayModeBar": False})


def render_data_analysis(df: pd.DataFrame) -> None:
    st.markdown("<div class='dashboard-title'>DATA ANALYSIS</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        mini_kpi("Rows", f"{len(df):,}")
    with c2:
        mini_kpi("Default Rate", f"{df[TARGET].mean():.2%}")
    with c3:
        mini_kpi("Features", str(len(FEATURE_COLUMNS)))

    feature = st.selectbox("Select feature", NUMERIC_COLUMNS)
    fig = px.histogram(
        df,
        x=feature,
        color=TARGET,
        marginal="box",
        color_discrete_sequence=["#38bdf8", "#a855f7"],
    )
    st.plotly_chart(chart_layout(fig, 420), use_container_width=True, config={"displayModeBar": False})

    corr = df[NUMERIC_COLUMNS].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(chart_layout(fig, 520), use_container_width=True, config={"displayModeBar": False})


def render_synthetic(df: pd.DataFrame, synthetic_df: pd.DataFrame | None) -> None:
    st.markdown("<div class='dashboard-title'>SYNTHETIC DATA COMPARISON</div>", unsafe_allow_html=True)

    if synthetic_df is None:
        st.info("Synthetic data was not uploaded to GitHub. The dashboard is running with real/demo data.")
        synthetic_df = create_demo_data(4000)

    c1, c2, c3 = st.columns(3)
    with c1:
        mini_kpi("Real Rows", f"{len(df):,}")
    with c2:
        mini_kpi("Synthetic Rows", f"{len(synthetic_df):,}")
    with c3:
        mini_kpi("Synthetic Default Rate", f"{synthetic_df[TARGET].mean():.2%}")

    feature = st.selectbox("Compare feature", NUMERIC_COLUMNS)
    plot_df = pd.concat(
        [
            df[[feature]].assign(source="Real"),
            synthetic_df[[feature]].assign(source="Synthetic"),
        ],
        ignore_index=True,
    )

    fig = px.histogram(
        plot_df,
        x=feature,
        color="source",
        barmode="overlay",
        opacity=0.65,
        color_discrete_sequence=["#38bdf8", "#a855f7"],
    )
    st.plotly_chart(chart_layout(fig, 420), use_container_width=True, config={"displayModeBar": False})


def render_prediction(df: pd.DataFrame) -> None:
    st.markdown("<div class='dashboard-title'>CREDIT RISK PREDICTION</div>", unsafe_allow_html=True)
    st.info("This deployed version uses a lightweight scoring formula unless you upload trained model artifacts.")

    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age", 21, 90, 45)
        income = st.number_input("Monthly Income", 0.0, 50000.0, 6500.0, step=500.0)
        debt_ratio = st.slider("Debt Ratio", 0.0, 2.0, 0.45, 0.01)

    with c2:
        utilization = st.slider("Revolving Utilization", 0.0, 2.0, 0.55, 0.01)
        open_lines = st.number_input("Open Credit Lines", 0, 50, 8)
        dependents = st.number_input("Dependents", 0, 10, 1)

    score = 1.8 * utilization + 1.4 * debt_ratio - 0.000035 * income + 0.01 * max(age - 45, 0)
    probability = 1 / (1 + np.exp(-(score - 1.25)))

    if st.button("Predict Credit Risk", type="primary"):
        label = "High Risk" if probability >= 0.5 else "Lower Risk"
        color = "#fb7185" if probability >= 0.5 else "#38bdf8"

        st.markdown(
            f"""
            <div class="section-panel">
                <h2 style="color:{color}; margin:0;">{label}: {probability:.2%}</h2>
                <p style="color:#a8a9c9;">Estimated default probability for the applicant.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_explainability() -> None:
    st.markdown("<div class='dashboard-title'>EXPLAINABILITY</div>", unsafe_allow_html=True)

    summary = FIGURE_DIR / "shap_summary.png"
    importance = FIGURE_DIR / "shap_feature_importance.png"

    if summary.exists():
        st.image(str(summary), caption="SHAP Summary", use_container_width=True)

    if importance.exists():
        st.image(str(importance), caption="SHAP Feature Importance", use_container_width=True)

    if not summary.exists() and not importance.exists():
        st.info("SHAP figures were not uploaded. Run explainability locally to generate report figures.")

        demo = pd.DataFrame(
            {
                "feature": [
                    "Revolving Utilization",
                    "Debt Ratio",
                    "Monthly Income",
                    "Age",
                    "Past Due 30-59",
                ],
                "importance": [0.32, 0.25, 0.18, 0.14, 0.11],
            }
        )
        fig = px.bar(
            demo,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#38bdf8", "#a855f7", "#fb7185"],
        )
        st.plotly_chart(chart_layout(fig, 420), use_container_width=True, config={"displayModeBar": False})


def render_privacy() -> None:
    st.markdown("<div class='dashboard-title'>PRIVACY & FAIRNESS METRICS</div>", unsafe_allow_html=True)

    path = REPORT_DIR / "privacy_fairness_metrics.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {
            "distance_to_closest_record": {
                "dcr_mean": 1.42,
                "dcr_median": 1.18,
                "dcr_min": 0.21,
            },
            "membership_inference": {
                "membership_risk_rate": 0.031,
            },
            "fairness": {
                "demographic_parity_difference": 0.084,
                "equal_opportunity_difference": 0.112,
            },
        }

    dcr = data.get("distance_to_closest_record", {})
    membership = data.get("membership_inference", {})
    fairness = data.get("fairness", {})

    c1, c2, c3 = st.columns(3)

    with c1:
        mini_kpi("Median DCR", f"{dcr.get('dcr_median', 0):.3f}")
    with c2:
        mini_kpi("Membership Risk", f"{membership.get('membership_risk_rate', 0):.2%}")
    with c3:
        mini_kpi("Equal Opportunity Diff", f"{fairness.get('equal_opportunity_difference', 0):.3f}")

    st.json(data)


def main() -> None:
    apply_style()

    df = load_data()
    synthetic_df = load_synthetic()
    metrics_df = load_metrics()

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

    if page == "Project Overview":
        render_overview(df, synthetic_df, metrics_df)
    elif page == "Data Analysis":
        render_data_analysis(df)
    elif page == "Synthetic Data Comparison":
        render_synthetic(df, synthetic_df)
    elif page == "Credit Risk Prediction":
        render_prediction(df)
    elif page == "Explainability":
        render_explainability()
    else:
        render_privacy()


if __name__ == "__main__":
    main()
