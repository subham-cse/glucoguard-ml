"""
GlucoGuard: Multi-Factor Diabetes Risk & Early Prevention Intelligence Engine
Production Streamlit Application (app.py)
IBM SkillsBuild / Bharat Cares AICTE Machine Learning Internship Final Submission
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Resolve module paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import (
    RAW_FEATURE_COLUMNS,
    prepare_single_patient_vector,
    prepare_batch_dataframe
)
from src.model_trainer import (
    load_trained_engine,
    GlucoGuardInferenceEngine,
    METADATA_PATH,
    BEST_MODEL_PATH
)
from src.report_generator import generate_patient_pdf, CleanClinicalPDF, FPDF_AVAILABLE

# ---------------------------------------------------------
# Streamlit Application Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="GlucoGuard | Clinical Diabetes Intelligence Engine",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Clinical Theme CSS
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* 4. Remove Clashing Top White Header Bar & Streamlit Decoration Line */
    header[data-testid="stHeader"], .stAppHeader {
        background-color: transparent !important;
        background: transparent !important;
        border-bottom: none !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }

    /* Background and Global Color Defaults */
    .stApp {
        background-color: #0b1120 !important;
        color: #f1f5f9 !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1350px !important;
    }

    /* 5. Standardize Container, Card & Markdown Typography */
    p, .stMarkdown p, .stMarkdown li, div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li {
        color: #e2e8f0 !important;
        line-height: 1.6 !important;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(13, 148, 136, 0.28) 50%, rgba(30, 41, 59, 0.98) 100%) !important;
        border: 1px solid rgba(20, 184, 166, 0.4) !important;
        border-radius: 16px !important;
        padding: 24px 30px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 10px 30px -10px rgba(13, 148, 136, 0.25) !important;
    }
    .hero-title {
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #38bdf8, #2dd4bf, #a7f3d0) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 6px !important;
    }
    .hero-subtitle {
        font-size: 1.02rem !important;
        color: #cbd5e1 !important;
        margin-bottom: 12px !important;
        line-height: 1.5 !important;
    }
    .badge-container {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
    }
    .clinical-badge {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        padding: 4px 10px !important;
        border-radius: 20px !important;
        letter-spacing: 0.3px !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 5px !important;
    }
    .badge-ibm {
        background: rgba(14, 165, 233, 0.2) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(14, 165, 233, 0.45) !important;
    }
    .badge-sdg {
        background: rgba(16, 185, 129, 0.2) !important;
        color: #34d399 !important;
        border: 1px solid rgba(16, 185, 129, 0.45) !important;
    }
    .badge-model {
        background: rgba(245, 158, 11, 0.2) !important;
        color: #fbbf24 !important;
        border: 1px solid rgba(245, 158, 11, 0.45) !important;
    }

    /* Metric Card */
    .metric-card {
        background: #1e293b !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }
    .metric-label {
        font-size: 0.82rem !important;
        text-transform: uppercase !important;
        color: #94a3b8 !important;
        letter-spacing: 0.5px !important;
        font-weight: 600 !important;
    }
    .metric-value {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        margin-top: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Status Pills */
    .risk-pill-low {
        background: rgba(16, 185, 129, 0.25) !important;
        color: #34d399 !important;
        border: 1px solid #10b981 !important;
        padding: 6px 14px !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        display: inline-block !important;
    }
    .risk-pill-moderate {
        background: rgba(245, 158, 11, 0.25) !important;
        color: #fbbf24 !important;
        border: 1px solid #f59e0b !important;
        padding: 6px 14px !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        display: inline-block !important;
    }
    .risk-pill-high {
        background: rgba(239, 68, 68, 0.25) !important;
        color: #f87171 !important;
        border: 1px solid #ef4444 !important;
        padding: 6px 14px !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        display: inline-block !important;
    }

    /* 2. Form Input Reference Range Helper Boxes */
    .ref-range-box {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-left: 3.5px solid #0ea5e9 !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        font-size: 0.78rem !important;
        color: #94a3b8 !important;
        margin-top: -6px !important;
        margin-bottom: 12px !important;
        line-height: 1.4 !important;
    }
    .ref-range-box strong, .ref-range-box b {
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }

    /* Intervention Card */
    .intervention-card {
        background: #1e293b !important;
        border-left: 4px solid #14b8a6 !important;
        border-radius: 0 10px 10px 0 !important;
        padding: 14px 18px !important;
        margin-bottom: 10px !important;
        border-top: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1) !important;
    }
    .intervention-title {
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        color: #2dd4bf !important;
        margin-bottom: 4px !important;
    }
    .intervention-body {
        font-size: 0.85rem !important;
        color: #cbd5e1 !important;
        line-height: 1.5 !important;
    }

    /* 1. Tab Bar Typography & Visibility Overrides */
    div[data-testid="stSidebarNav"] { display: none !important; }
    .stTabs [data-baseweb="tab-list"], div[data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #1e293b !important;
        padding: 8px 12px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
    }
    /* Unselected Tab States */
    .stTabs [data-baseweb="tab"], div[data-baseweb="tab"], button[data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        background-color: transparent !important;
        border: none !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stTabs [data-baseweb="tab"] p, div[data-baseweb="tab"] p, button[data-baseweb="tab"] p {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }
    .stTabs [data-baseweb="tab"]:hover, div[data-baseweb="tab"]:hover, button[data-baseweb="tab"]:hover {
        color: #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.08) !important;
    }
    .stTabs [data-baseweb="tab"]:hover p, div[data-baseweb="tab"]:hover p, button[data-baseweb="tab"]:hover p {
        color: #38bdf8 !important;
    }
    /* Selected Active Tab States */
    .stTabs [aria-selected="true"], div[data-baseweb="tab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(13, 148, 136, 0.2) !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #0d9488 !important;
        border-radius: 8px !important;
    }
    .stTabs [aria-selected="true"] p, div[data-baseweb="tab"][aria-selected="true"] p, button[data-baseweb="tab"][aria-selected="true"] p {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    /* 2. Form Input Labels, Help Text & Slider Controls */
    div[data-testid="stWidgetLabel"] label, 
    div[data-testid="stWidgetLabel"] p, 
    .stNumberInput label, 
    .stSelectbox label, 
    .stSlider label,
    label[data-testid="stWidgetLabel"] {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.01em !important;
    }
    /* Input Box Styles */
    input, select, textarea, div[data-baseweb="input"] input, div[data-baseweb="select"] div {
        color: #f8fafc !important;
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 1px #0d9488 !important;
    }
    /* Number input step controls (+ and -) */
    div[data-testid="stNumberInput"] button {
        color: #f8fafc !important;
        background-color: #1e293b !important;
        border-color: #334155 !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        background-color: #334155 !important;
        color: #38bdf8 !important;
    }
    /* Selectbox dropdowns */
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f8fafc !important;
    }
    li[role="option"] {
        color: #e2e8f0 !important;
    }
    li[role="option"][aria-selected="true"], li[role="option"]:hover {
        background-color: rgba(13, 148, 136, 0.3) !important;
        color: #38bdf8 !important;
    }
    /* Slider value display & labels */
    div[data-testid="stSlider"] div[data-testid="stTickBarMin"], 
    div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
        color: #94a3b8 !important;
    }

    /* 3. Button Styling & Complete Text Invisibility Fix (White-on-White Bug) */
    div.stButton > button, 
    div[data-testid="stDownloadButton"] > button, 
    button[kind="primary"], 
    button[kind="secondary"] {
        background: linear-gradient(135deg, #0d9488 0%, #0284c7 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.4) !important;
        padding: 10px 24px !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover, 
    div[data-testid="stDownloadButton"] > button:hover, 
    button[kind="primary"]:hover, 
    button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #0f766e 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 6px 20px rgba(13, 148, 136, 0.6) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:active, 
    div[data-testid="stDownloadButton"] > button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 8px rgba(13, 148, 136, 0.4) !important;
    }
    /* Deep cascade to all inner elements inside buttons to guarantee text contrast */
    div.stButton > button *, 
    div[data-testid="stDownloadButton"] > button *, 
    div.stButton > button p, 
    div[data-testid="stDownloadButton"] > button p, 
    div.stButton > button span, 
    div[data-testid="stDownloadButton"] > button span, 
    div.stButton > button svg, 
    div[data-testid="stDownloadButton"] > button svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        font-weight: 600 !important;
    }

    /* File uploader container styling */
    div[data-testid="stFileUploader"] section {
        background-color: #1e293b !important;
        border: 2px dashed #0d9488 !important;
        border-radius: 12px !important;
        padding: 24px !important;
    }
    div[data-testid="stFileUploader"] section * {
        color: #cbd5e1 !important;
    }

    /* Dataframe background & table text */
    div[data-testid="stDataFrame"] {
        background-color: #1e293b !important;
        border-radius: 8px !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
    }

    /* 6. Tooltip / Help Popover ("Info Tab") Invisibility & Contrast Fix */
    div[data-baseweb="tooltip"], 
    div[data-baseweb="tooltip"] > div, 
    div[data-testid="stTooltipContent"], 
    div[role="tooltip"] {
        background-color: #0f172a !important;
        background: #0f172a !important;
        color: #f8fafc !important;
        border: 1.5px solid #0d9488 !important;
        border-radius: 8px !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.75) !important;
        padding: 8px 12px !important;
    }

    /* Force all text inside tooltips to bright, high-contrast white */
    div[data-baseweb="tooltip"] *, 
    div[data-testid="stTooltipContent"] *, 
    div[role="tooltip"] *,
    div[data-baseweb="tooltip"] p, 
    div[data-testid="stTooltipContent"] p, 
    div[role="tooltip"] p,
    div[data-baseweb="tooltip"] span, 
    div[data-testid="stTooltipContent"] span, 
    div[role="tooltip"] span {
        color: #f8fafc !important;
        font-size: 0.84rem !important;
        line-height: 1.45 !important;
        font-weight: 500 !important;
    }

    /* Tooltip trigger icon (?) high contrast visibility */
    div[data-testid="stWidgetLabel"] [data-testid="stTooltipHoverTarget"],
    div[data-testid="stWidgetLabel"] button,
    button[data-testid="stTooltipHoverTarget"],
    div[data-testid="stTooltipHoverTarget"],
    div[data-testid="stTooltipHoverTarget"] button,
    [data-testid="stTooltipHoverTarget"] {
        color: #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.18) !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        border-radius: 6px !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
    }

    /* Force all SVGs, paths, circles, and child elements to bright crisp cyan */
    div[data-testid="stWidgetLabel"] button *,
    div[data-testid="stWidgetLabel"] svg,
    div[data-testid="stWidgetLabel"] svg *,
    [data-testid="stTooltipHoverTarget"] *,
    [data-testid="stTooltipHoverTarget"] svg,
    [data-testid="stTooltipHoverTarget"] svg path,
    [data-testid="stTooltipHoverTarget"] svg circle,
    [data-testid="stTooltipHoverTarget"] svg rect {
        color: #38bdf8 !important;
        fill: #38bdf8 !important;
        stroke: #38bdf8 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* Hover State for (?) button */
    div[data-testid="stWidgetLabel"] button:hover,
    button[data-testid="stTooltipHoverTarget"]:hover,
    [data-testid="stTooltipHoverTarget"]:hover {
        background-color: rgba(13, 148, 136, 0.35) !important;
        border-color: #2dd4bf !important;
    }
    div[data-testid="stWidgetLabel"] button:hover *,
    [data-testid="stTooltipHoverTarget"]:hover * {
        color: #2dd4bf !important;
        fill: #2dd4bf !important;
        stroke: #2dd4bf !important;
    }

    /* Alert Notification Boxes (st.info, st.warning, st.error, st.success) */
    div[data-testid="stAlert"] {
        background-color: #1e293b !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
    }
    div[data-testid="stAlert"] * {
        color: #f8fafc !important;
    }
    div[data-testid="stAlert"] svg {
        fill: #38bdf8 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Model & Metadata Loader Cache
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_inference_engine():
    """Caches the trained model pipeline and metadata in memory."""
    engine, metadata = load_trained_engine()
    return engine, metadata


# ---------------------------------------------------------
# Clinical Reference Ranges & Profiles
# ---------------------------------------------------------
CLINICAL_PROFILES = {
    "Custom / Manual Input": None,
    "Healthy Adult (Low Risk)": {
        "Pregnancies": 1,
        "Glucose": 88.0,
        "BloodPressure": 68.0,
        "SkinThickness": 22.0,
        "Insulin": 65.0,
        "BMI": 22.4,
        "DiabetesPedigreeFunction": 0.22,
        "Age": 28
    },
    "Borderline / Pre-Diabetic (Moderate Risk)": {
        "Pregnancies": 3,
        "Glucose": 118.0,
        "BloodPressure": 82.0,
        "SkinThickness": 31.0,
        "Insulin": 135.0,
        "BMI": 28.7,
        "DiabetesPedigreeFunction": 0.48,
        "Age": 42
    },
    "High-Risk Metabolic Patient (High Risk)": {
        "Pregnancies": 5,
        "Glucose": 172.0,
        "BloodPressure": 88.0,
        "SkinThickness": 38.0,
        "Insulin": 220.0,
        "BMI": 36.4,
        "DiabetesPedigreeFunction": 0.88,
        "Age": 54
    }
}


def render_hero_header(meta: Dict[str, Any]):
    """Renders top header banner with IBM SkillsBuild / Bharat Cares / AICTE branding."""
    best_arch = meta.get("best_model_architecture", "Ensemble Classifier")
    best_roc = meta.get("test_benchmarks", {}).get(best_arch, {}).get("roc_auc", 0.84)

    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-title">🩺 GlucoGuard: Clinical Diabetes Intelligence Engine</div>
        <div class="hero-subtitle">
            Multi-Factor Diabetes Risk Stratification, Early Prevention & Explainable Clinical Decision Support System.
        </div>
        <div class="badge-container">
            <span class="clinical-badge badge-ibm">🏢 IBM SkillsBuild | Bharat Cares | AICTE Final Submission</span>
            <span class="clinical-badge badge-sdg">🌍 UN SDG 3: Good Health & Well-Being</span>
            <span class="clinical-badge badge-model">⚡ Active Pipeline: {best_arch} (Test ROC-AUC: {best_roc:.3f})</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_risk_gauge_chart(risk_score_pct: float) -> go.Figure:
    """Generates an interactive medical speedometer gauge for patient risk score."""
    if risk_score_pct < 30.0:
        bar_color = "#10b981"
        tier_text = "LOW RISK"
    elif risk_score_pct < 65.0:
        bar_color = "#f59e0b"
        tier_text = "BORDERLINE / MODERATE"
    else:
        bar_color = "#ef4444"
        tier_text = "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score_pct,
        number={'suffix': "%", 'font': {'size': 44, 'color': '#ffffff', 'family': 'JetBrains Mono'}},
        delta={'reference': 35.0, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}, 'suffix': "% vs cohort avg"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "rgba(30, 41, 59, 0.7)",
            'borderwidth': 1,
            'bordercolor': "rgba(148, 163, 184, 0.2)",
            'steps': [
                {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.15)"},
                {'range': [30, 65], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.15)"}
            ],
            'threshold': {
                'line': {'color': "#f43f5e", 'width': 3},
                'thickness': 0.85,
                'value': 65
            }
        }
    ))

    fig.update_layout(
        height=270,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc", family="Plus Jakarta Sans")
    )
    return fig


def create_explainability_bar_chart(contributions: List[Dict[str, Any]]) -> go.Figure:
    """Renders local feature contribution plot showing risk escalators and mitigators."""
    df_contrib = pd.DataFrame(contributions)
    # Filter out near-zero neutral factors
    df_contrib = df_contrib[df_contrib["risk_contribution_pct"].abs() >= 0.05].copy()
    df_contrib = df_contrib.sort_values(by="risk_contribution_pct", ascending=True)

    colors = ["#ef4444" if val > 0 else "#10b981" for val in df_contrib["risk_contribution_pct"]]

    fig = go.Figure(go.Bar(
        x=df_contrib["risk_contribution_pct"],
        y=df_contrib["feature"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
        text=[f"{val:+.1f}%" for val in df_contrib["risk_contribution_pct"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Patient Value: %{customdata[0]}<br>Cohort Median: %{customdata[1]}<br>Risk Delta: %{x:+.2f}%<extra></extra>",
        customdata=df_contrib[["patient_value", "population_median"]].values
    ))

    fig.update_layout(
        title=dict(text="<b>SHAP-Style Biomarker Risk Attribution (% Impact)</b>", font=dict(size=15, color="#f1f5f9")),
        xaxis=dict(
            title="Contribution to Patient Diabetes Risk (% Delta)",
            zeroline=True,
            zerolinecolor="#94a3b8",
            gridcolor="rgba(148, 163, 184, 0.15)"
        ),
        yaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)"),
        height=320,
        margin=dict(l=20, r=30, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Plus Jakarta Sans")
    )
    return fig


def generate_clinical_interventions(patient_data: Dict[str, Any], risk_score_pct: float) -> List[Dict[str, str]]:
    """Synthesizes clinical lifestyle, nutrition, and surveillance recommendations."""
    interventions = []

    # Fasting Glucose recommendations
    glucose = patient_data.get("Glucose", 100)
    if glucose >= 126:
        interventions.append({
            "title": "🔴 Urgent Glycemic Diagnostic Evaluation",
            "body": f"Current Fasting Glucose ({glucose} mg/dL) exceeds diabetes diagnostic threshold (≥126 mg/dL). Immediate confirmatory venous HbA1c test and endocrinologist consultation recommended."
        })
    elif glucose >= 100:
        interventions.append({
            "title": "🟡 Impaired Fasting Glucose (Pre-Diabetes)",
            "body": f"Glucose ({glucose} mg/dL) indicates pre-diabetes. Adopt low glycemic index (GI) Mediterranean dietary pattern and schedule repeat fasting panel every 6 months."
        })

    # Adiposity / BMI recommendations
    bmi = patient_data.get("BMI", 25.0)
    if bmi >= 30.0:
        target_wt_loss = round((bmi - 25.0) * 2.2, 1)
        interventions.append({
            "title": "⚖️ Metabolic Weight Management Strategy",
            "body": f"Patient BMI ({bmi} kg/m²) indicates clinical obesity. Aim for 7–10% gradual body weight reduction (~{target_wt_loss} lbs) to dramatically enhance peripheral insulin sensitivity."
        })
    elif bmi >= 25.0:
        interventions.append({
            "title": "⚖️ Weight Maintenance & Physical Conditioning",
            "body": f"BMI ({bmi} kg/m²) is in overweight threshold. 150 minutes/week of moderate aerobic exercise plus 2 resistance training sessions recommended to prevent visceral fat deposition."
        })

    # Blood Pressure recommendations
    bp = patient_data.get("BloodPressure", 80)
    if bp >= 130:
        interventions.append({
            "title": "🫀 Cardiovascular Pressure Control (Stage 1/2 HTN)",
            "body": f"Diastolic BP ({bp} mmHg) indicates hypertension risk. Implement DASH diet protocols (sodium < 2,300 mg/day, increased dietary potassium) and daily home BP monitoring."
        })

    # Genetic / Pedigree recommendations
    dpf = patient_data.get("DiabetesPedigreeFunction", 0.3)
    if dpf >= 0.5:
        interventions.append({
            "title": "🧬 Elevated Hereditary Pedigree Surveillance",
            "body": f"Diabetes Pedigree Score ({dpf:.2f}) indicates elevated familial genetic predisposition. Annual lipid panels and microalbuminuria screening are advised."
        })

    # General wellness if low risk
    if risk_score_pct < 30.0 and len(interventions) == 0:
        interventions.append({
            "title": "🟢 Optimal Metabolic Health Maintenance",
            "body": "All primary biomarkers fall within safe physiological limits. Continue balanced whole-food nutrition, adequate sleep (7–9 hrs), and routine annual preventive wellness checkups."
        })

    return interventions


# ---------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------
def main():
    # Load model and metadata
    try:
        with st.spinner("Initializing Clinical AI Models & Calibrating Baselines..."):
            engine, metadata = get_cached_inference_engine()
    except Exception as exc:
        st.error(f"Critical error initializing ML inference engine: {exc}")
        st.stop()

    render_hero_header(metadata)

    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🩺 Patient Diagnostic Assessment",
        "📁 Batch Patient Screening",
        "📊 Model Validation & Performance",
        "🔮 'What-If' Intervention Simulator"
    ])

    # =========================================================
    # TAB 1: Clinical Risk Assessment (Single Patient)
    # =========================================================
    with tab1:
        st.markdown("### 📋 Individual Patient Intake & Diagnostic Assessment")
        st.markdown(
            "Input clinical lab telemetry, patient demographics, and vitals below. Normal clinical reference ranges are shown for validation."
        )

        # -----------------------------------------------------
        # Patient Administrative & Demographic Intake
        # -----------------------------------------------------
        st.markdown("#### 👤 Patient Administrative & Demographic Profile")
        dem_col1, dem_col2, dem_col3 = st.columns(3)
        with dem_col1:
            patient_name = st.text_input(
                "Patient Full Name",
                value="Jane Doe",
                help="Full legal name of the patient for clinical evaluation documentation."
            )
            patient_mrn = st.text_input(
                "Patient ID / Medical Record Number (MRN)",
                value="GG-70492",
                help="Unique patient medical record identifier code."
            )
        with dem_col2:
            patient_gender = st.selectbox(
                "Biological Sex / Gender",
                options=["Female", "Male", "Other"],
                index=0,
                help="Biological sex context for clinical evaluation and documentation."
            )
            patient_blood_group = st.selectbox(
                "Blood Group (Optional)",
                options=["Not Specified", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
                index=0,
                help="ABO / Rh factor blood typing (compiled into formal clinical diagnostic report)."
            )
        with dem_col3:
            attending_physician = st.text_input(
                "Attending Clinician / Physician",
                value="Dr. Sarah Lin, MD",
                help="Evaluating healthcare professional or diagnostic center."
            )
            st.markdown('<div class="ref-range-box" style="margin-top: 14px;"><strong>Administrative Record:</strong> Demographics and Blood Group are compiled into the official clinical PDF diagnostic report.</div>', unsafe_allow_html=True)

        col_prof, _ = st.columns([2, 2])
        with col_prof:
            selected_preset = st.selectbox(
                "Quick-Load Preset Clinical Cohort Profile:",
                options=list(CLINICAL_PROFILES.keys()),
                index=0
            )

        preset_data = CLINICAL_PROFILES[selected_preset] or {}

        # Clinical Input Grid
        col1, col2, col3 = st.columns(3)

        with col1:
            val_glucose = st.number_input(
                "Fasting Plasma Glucose (mg/dL)",
                min_value=40.0, max_value=300.0,
                value=float(preset_data.get("Glucose", 110.0)),
                step=1.0,
                help="Plasma glucose concentration 2 hours after oral glucose tolerance test."
            )
            st.markdown('<div class="ref-range-box">Normal: 70–99 mg/dL | Pre-Diabetes: 100–125 | Diabetes: ≥126</div>', unsafe_allow_html=True)

            val_bp = st.number_input(
                "Diastolic Blood Pressure (mmHg)",
                min_value=30.0, max_value=180.0,
                value=float(preset_data.get("BloodPressure", 72.0)),
                step=1.0,
                help="Diastolic blood pressure resting measurement."
            )
            st.markdown('<div class="ref-range-box">Normal: <80 mmHg | Elevated: 80–89 | Stage 2 HTN: ≥90</div>', unsafe_allow_html=True)

            val_bmi = st.number_input(
                "Body Mass Index - BMI (kg/m²)",
                min_value=12.0, max_value=70.0,
                value=float(preset_data.get("BMI", 27.5)),
                step=0.1,
                help="Weight in kilograms divided by the square of height in meters."
            )
            st.markdown('<div class="ref-range-box">Normal: 18.5–24.9 | Overweight: 25–29.9 | Obese: ≥30</div>', unsafe_allow_html=True)

        with col2:
            val_insulin = st.number_input(
                "2-Hour Serum Insulin (μIU/mL)",
                min_value=0.0, max_value=850.0,
                value=float(preset_data.get("Insulin", 85.0)),
                step=1.0,
                help="2-Hour serum insulin level (0 is treated as missing lab test)."
            )
            st.markdown('<div class="ref-range-box">Fasting Normal: 16–166 μIU/mL (Biological zero handled via KNN/Median)</div>', unsafe_allow_html=True)

            val_skin = st.number_input(
                "Triceps Skinfold Thickness (mm)",
                min_value=0.0, max_value=100.0,
                value=float(preset_data.get("SkinThickness", 25.0)),
                step=1.0,
                help="Subcutaneous fat caliper measurement (mm)."
            )
            st.markdown('<div class="ref-range-box">Typical Median: 20–35 mm (Biological zero auto-imputed)</div>', unsafe_allow_html=True)

            val_age = st.number_input(
                "Patient Age (Years)",
                min_value=18, max_value=100,
                value=int(preset_data.get("Age", 38)),
                step=1,
                help="Patient biological age."
            )
            st.markdown('<div class="ref-range-box">Higher risk window: Age ≥ 45 years</div>', unsafe_allow_html=True)

        with col3:
            val_preg = st.number_input(
                "Number of Pregnancies",
                min_value=0, max_value=20,
                value=int(preset_data.get("Pregnancies", 2)),
                step=1,
                help="Total past pregnancies (Gestational diabetes history factor)."
            )
            st.markdown('<div class="ref-range-box">Gestational stress metric (0 for non-applicable)</div>', unsafe_allow_html=True)

            val_dpf = st.number_input(
                "Diabetes Pedigree Function (DPF)",
                min_value=0.05, max_value=3.0,
                value=float(preset_data.get("DiabetesPedigreeFunction", 0.38)),
                step=0.01,
                help="Genetic history synthesis score based on family pedigree."
            )
            st.markdown('<div class="ref-range-box">Cohort Median: 0.37 | High Genetic Burden: >0.60</div>', unsafe_allow_html=True)

        btn_analyze = st.button("🚀 Analyze Patient Risk & Generate Diagnostics", type="primary", use_container_width=True)

        if btn_analyze or selected_preset != "Custom / Manual Input":
            patient_dict = {
                "Pregnancies": val_preg,
                "Glucose": val_glucose,
                "BloodPressure": val_bp,
                "SkinThickness": val_skin,
                "Insulin": val_insulin,
                "BMI": val_bmi,
                "DiabetesPedigreeFunction": val_dpf,
                "Age": val_age
            }

            # Inference Vector
            patient_vec = prepare_single_patient_vector(patient_dict)
            risk_prob = float(engine.predict_risk_score(patient_vec)[0])
            risk_score_pct = round(risk_prob * 100, 1)

            st.markdown("---")
            st.markdown("### 🔍 Diagnostic Risk Report")

            res_col1, res_col2 = st.columns([1, 1.4])

            with res_col1:
                st.plotly_chart(create_risk_gauge_chart(risk_score_pct), use_container_width=True)

                if risk_score_pct < 30.0:
                    badge_html = '<span class="risk-pill-low">🟢 Low Clinical Risk</span>'
                    desc_text = "Patient demonstrates stable glycemic homeostasis. Low probability of diabetes manifestation."
                elif risk_score_pct < 65.0:
                    badge_html = '<span class="risk-pill-moderate">🟡 Moderate / Borderline Risk</span>'
                    desc_text = "Early metabolic dysfunction detected. Immediate lifestyle and dietary intervention recommended to arrest progression."
                else:
                    badge_html = '<span class="risk-pill-high">🔴 High Clinical Alert</span>'
                    desc_text = "Significant multi-factor diabetic indicators detected. Comprehensive clinical lab workup and physician review required."

                st.markdown(f"""
                <div class="metric-card">
                    <div style="margin-bottom: 8px;">{badge_html}</div>
                    <div class="metric-label">Estimated 5-Year Diabetes Risk Index</div>
                    <div class="metric-value">{risk_score_pct}%</div>
                    <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 8px;">{desc_text}</p>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                # Local Explainability
                contributions = engine.explain_patient_prediction(patient_vec)
                st.plotly_chart(create_explainability_bar_chart(contributions), use_container_width=True)

            # Tailored Clinical Interventions
            st.markdown("### 💊 Tailored Clinical Interventions & Preventative Action Plan")
            interventions = generate_clinical_interventions(patient_dict, risk_score_pct)
            for item in interventions:
                st.markdown(f"""
                <div class="intervention-card">
                    <div class="intervention-title">{item['title']}</div>
                    <div class="intervention-body">{item['body']}</div>
                </div>
                """, unsafe_allow_html=True)

            # Persist latest diagnosis & patient metadata in session state
            risk_tier_label = (
                "Low Risk" if risk_score_pct < 30.0 else ("Borderline Pre-Diabetic" if risk_score_pct < 65.0 else "High Clinical Risk")
            )
            patient_meta_dict = {
                "name": patient_name.strip() if patient_name else "Jane Doe",
                "mrn": patient_mrn.strip() if patient_mrn else "GG-70492",
                "gender": patient_gender,
                "blood_group": patient_blood_group,
                "physician": attending_physician.strip() if attending_physician else "Clinical Decision Support",
                "age": val_age
            }
            st.session_state["latest_patient_data"] = patient_dict
            st.session_state["latest_patient_meta"] = patient_meta_dict
            st.session_state["latest_risk_prob"] = float(risk_prob)
            st.session_state["latest_risk_tier"] = str(risk_tier_label)
            st.session_state["analyzed_patient_data"] = patient_dict
            st.session_state["analyzed_patient_meta"] = patient_meta_dict
            st.session_state["analyzed_risk_score_pct"] = risk_score_pct
            st.session_state["analyzed_risk_tier"] = str(risk_tier_label)
            st.session_state["analyzed_contributions"] = contributions
            st.session_state["analyzed_interventions"] = interventions

        # ---------------------------------------------------------
        # Downloadable Clinical PDF Diagnostic Report Section
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### Official Clinical Diagnostic Documentation")
        st.caption("Export an executive single-page medical summary report for this patient.")

        # Pre-generate the PDF into session state whenever risk is calculated
        if 'latest_patient_data' in st.session_state and 'latest_risk_prob' in st.session_state:
            if not FPDF_AVAILABLE:
                st.warning("⚠️ 'fpdf2' is not installed in the environment. Run `pip install fpdf2` to enable PDF compilation.")
            else:
                try:
                    active_meta = st.session_state.get('latest_patient_meta', {
                        "name": patient_name,
                        "mrn": patient_mrn,
                        "gender": patient_gender,
                        "blood_group": patient_blood_group,
                        "physician": attending_physician,
                        "age": val_age
                    })
                    pdf_bytes = generate_patient_pdf(
                        patient_data=st.session_state['latest_patient_data'],
                        risk_prob=st.session_state['latest_risk_prob'],
                        risk_tier=st.session_state['latest_risk_tier'],
                        key_factors=[],
                        patient_meta=active_meta
                    )
                    
                    p_label_name = active_meta.get("name", "Patient")
                    safe_filename = f"GlucoGuard_Report_{p_label_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    
                    st.download_button(
                        label=f"⬇️ Download Official Clinical Diagnostic Report (PDF) - {p_label_name}",
                        data=pdf_bytes,
                        file_name=safe_filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error compiling diagnostic PDF: {str(e)}")
        else:
            st.info("💡 Run patient diagnostic assessment above to generate and unlock the PDF report.")

    # =========================================================
    # TAB 2: Batch Patient Screening (CSV/XLSX Upload)
    # =========================================================
    with tab2:
        st.markdown("### 📁 Institutional Batch Screening Workflow")
        st.markdown(
            "Upload structured patient cohorts (CSV or Excel) to execute automated high-throughput risk triage."
        )

        sample_test_path = PROJECT_ROOT / "data" / "sample_patients_test.csv"
        if sample_test_path.exists():
            with open(sample_test_path, "rb") as f:
                st.download_button(
                    label="📥 Download Standard Clinical CSV Template (20 Patient Records)",
                    data=f.read(),
                    file_name="glucoguard_sample_batch_template.csv",
                    mime="text/csv"
                )

        uploaded_file = st.file_uploader(
            "Upload Batch Clinical Records (.csv, .xlsx):",
            type=["csv", "xlsx"]
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".xlsx"):
                    df_raw_batch = pd.read_excel(uploaded_file)
                else:
                    df_raw_batch = pd.read_csv(uploaded_file)

                st.success(f"Successfully loaded file '{uploaded_file.name}' with {len(df_raw_batch)} patient records.")

                # Prepare feature vectors
                with st.spinner("Executing automated imputation, feature engineering & batch inference..."):
                    df_features = prepare_batch_dataframe(df_raw_batch)
                    batch_probs = engine.predict_risk_score(df_features)
                    batch_pcts = np.round(batch_probs * 100, 1)

                    # Enrich dataframe
                    df_results = df_raw_batch.copy()
                    df_results["Predicted_Risk_Score_%"] = batch_pcts
                    df_results["Risk_Tier"] = pd.cut(
                        batch_pcts,
                        bins=[-np.inf, 30.0, 65.0, np.inf],
                        labels=["Low Risk", "Borderline Risk", "High Risk"]
                    )
                    df_results["Clinical_Action"] = df_results["Risk_Tier"].map({
                        "Low Risk": "Routine Annual Checkup",
                        "Borderline Risk": "Lifestyle Counseling + 6M FPG",
                        "High Risk": "Immediate Physician Referral + HbA1c Panel"
                    })

                # Batch KPI Cards
                total_screened = len(df_results)
                high_risk_count = int((df_results["Risk_Tier"] == "High Risk").sum())
                high_risk_pct = round((high_risk_count / total_screened) * 100, 1)
                avg_risk = round(float(batch_pcts.mean()), 1)

                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                with b_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Screened</div>
                        <div class="metric-value">{total_screened}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with b_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">High Risk Flagged</div>
                        <div class="metric-value" style="color: #f87171;">{high_risk_count}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with b_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">High Risk Ratio</div>
                        <div class="metric-value" style="color: #fbbf24;">{high_risk_pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with b_col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Cohort Mean Risk</div>
                        <div class="metric-value">{avg_risk}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Batch Visualizations
                plot_c1, plot_c2 = st.columns([1.5, 1])
                with plot_c1:
                    fig_dist = px.histogram(
                        df_results,
                        x="Predicted_Risk_Score_%",
                        nbins=20,
                        color="Risk_Tier",
                        color_discrete_map={"Low Risk": "#10b981", "Borderline Risk": "#f59e0b", "High Risk": "#ef4444"},
                        title="Cohort Risk Score Distribution",
                        labels={"Predicted_Risk_Score_%": "Risk Score (%)"}
                    )
                    fig_dist.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                        height=280,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                with plot_c2:
                    tier_counts = df_results["Risk_Tier"].value_counts().reset_index()
                    tier_counts.columns = ["Risk_Tier", "Count"]
                    fig_pie = px.pie(
                        tier_counts,
                        values="Count",
                        names="Risk_Tier",
                        hole=0.55,
                        color="Risk_Tier",
                        color_discrete_map={"Low Risk": "#10b981", "Borderline Risk": "#f59e0b", "High Risk": "#ef4444"},
                        title="Cohort Stratification Breakdown"
                    )
                    fig_pie.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                        height=280,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Interactive Results Table
                st.markdown("#### 📑 Screened Patient Cohort Roster")
                st.dataframe(df_results, use_container_width=True)

                # Export Enriched Results
                csv_export = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Triage Report as CSV",
                    data=csv_export,
                    file_name="glucoguard_screened_cohort_results.csv",
                    mime="text/csv",
                    type="primary"
                )

            except Exception as e:
                st.error(f"Error parsing uploaded batch file: {e}")
        else:
            st.info("💡 Upload a CSV file or download the pre-loaded template above to test batch clinical screening.")

    # =========================================================
    # TAB 3: Model Performance & Validation Suite
    # =========================================================
    with tab3:
        st.markdown("### 📊 Model Benchmarks & Validation Telemetry")
        st.markdown(
            "Rigorous multi-model evaluation across 5 clinical performance metrics utilizing Stratified 5-Fold Cross-Validation."
        )

        test_benchmarks = metadata.get("test_benchmarks", {})
        cv_benchmarks = metadata.get("cv_benchmarks", {})
        curves_data = metadata.get("curves_data", {})

        # Side-by-side benchmark table
        st.markdown("#### 🏆 Test Set Model Leaderboard")
        bench_rows = []
        for name, metrics in test_benchmarks.items():
            bench_rows.append({
                "Model Architecture": name,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall (Sensitivity)": f"{metrics['recall']:.4f}",
                "F1-Score": f"{metrics['f1_score']:.4f}",
                "ROC-AUC": f"{metrics['roc_auc']:.4f}",
                "Status": "⭐ Selected Production" if name == metadata.get("best_model_architecture") else "Evaluated"
            })
        df_bench = pd.DataFrame(bench_rows)
        st.dataframe(df_bench, use_container_width=True)

        # ROC & PR Curves
        c_curve1, c_curve2 = st.columns(2)

        with c_curve1:
            fig_roc = go.Figure()
            color_map = {
                "Voting Ensemble (Champion)": "#f43f5e",
                "XGBoost": "#fbbf24",
                "Random Forest": "#2dd4bf",
                "Logistic Regression": "#38bdf8"
            }
            for model_name, curve_info in curves_data.items():
                roc_info = curve_info.get("roc", {})
                fig_roc.add_trace(go.Scatter(
                    x=roc_info.get("fpr", []),
                    y=roc_info.get("tpr", []),
                    mode="lines",
                    name=f"{model_name} (AUC={roc_info.get('auc', 0):.3f})",
                    line=dict(width=2.5, color=color_map.get(model_name, "#a78bfa"))
                ))
            # Diagonal random guess
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Random Chance",
                line=dict(dash="dash", color="#64748b")
            ))
            fig_roc.update_layout(
                title="<b>Receiver Operating Characteristic (ROC) Curves</b>",
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Sensitivity / Recall)",
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98, bgcolor="rgba(15,23,42,0.8)")
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with c_curve2:
            fig_pr = go.Figure()
            for model_name, curve_info in curves_data.items():
                pr_info = curve_info.get("pr", {})
                fig_pr.add_trace(go.Scatter(
                    x=pr_info.get("recall", []),
                    y=pr_info.get("precision", []),
                    mode="lines",
                    name=model_name,
                    line=dict(width=2.5, color=color_map.get(model_name, "#a78bfa"))
                ))
            fig_pr.update_layout(
                title="<b>Precision-Recall (PR) Curves</b>",
                xaxis_title="Recall (Sensitivity)",
                yaxis_title="Precision (Positive Predictive Value)",
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor="rgba(15,23,42,0.8)")
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        # Confusion Matrix & Global Feature Importance
        c_diag1, c_diag2 = st.columns(2)

        with c_diag1:
            best_model_name = metadata.get("best_model_architecture", "Random Forest")
            cm_data = test_benchmarks.get(best_model_name, {}).get("confusion_matrix", [[0, 0], [0, 0]])

            fig_cm = px.imshow(
                cm_data,
                text_auto=True,
                color_continuous_scale="Teal",
                labels=dict(x="Predicted Diagnosis", y="Actual Clinical Outcome", color="Patients"),
                x=["Non-Diabetic (0)", "Diabetic (1)"],
                y=["Non-Diabetic (0)", "Diabetic (1)"],
                title=f"<b>Test Set Confusion Matrix ({best_model_name})</b>"
            )
            fig_cm.update_layout(
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="Plus Jakarta Sans")
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with c_diag2:
            feat_imp_dict = test_benchmarks.get(best_model_name, {}).get("feature_importance", {})
            if feat_imp_dict:
                df_imp = pd.DataFrame(list(feat_imp_dict.items()), columns=["Biomarker", "Importance"]).sort_values(
                    by="Importance", ascending=True
                )
                fig_imp = px.bar(
                    df_imp,
                    x="Importance",
                    y="Biomarker",
                    orientation="h",
                    title=f"<b>Global Feature Importance ({best_model_name})</b>",
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                fig_imp.update_layout(
                    height=320,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_imp, use_container_width=True)

    # =========================================================
    # TAB 4: Interactive Risk Sensitivity Simulator
    # =========================================================
    with tab4:
        st.markdown("### 🔮 Interactive 'What-If' Intervention Sensitivity Simulator")
        st.markdown(
            "Explore simulated clinical interventions: quantify the exact risk reduction achieved when a patient "
            "lowers fasting glucose, reduces adiposity (BMI), or optimizes blood pressure."
        )

        sim_col_l, sim_col_r = st.columns([1, 1.2])

        with sim_col_l:
            st.markdown("#### 1. Baseline Patient Parameters")
            sim_glucose = st.slider("Baseline Glucose (mg/dL):", 70.0, 240.0, 155.0, 1.0)
            sim_bmi = st.slider("Baseline BMI (kg/m²):", 18.0, 55.0, 34.0, 0.5)
            sim_bp = st.slider("Baseline Blood Pressure (mmHg):", 50.0, 130.0, 84.0, 1.0)
            sim_age = st.slider("Patient Age:", 20, 80, 48, 1)

            st.markdown("#### 2. Clinical Lifestyle / Therapeutic Interventions")
            delta_glucose = st.slider("🎯 Fasting Glucose Reduction (mg/dL):", 0.0, 60.0, 25.0, 1.0)
            delta_bmi = st.slider("🎯 Body Mass Index (BMI) Reduction (kg/m²):", 0.0, 8.0, 4.0, 0.5)
            delta_bp = st.slider("🎯 Diastolic BP Optimization (mmHg):", 0.0, 25.0, 8.0, 1.0)

        with sim_col_r:
            st.markdown("#### 3. Simulated Clinical Impact & Risk Trajectory")

            # Base Patient
            base_patient = {
                "Pregnancies": 2,
                "Glucose": sim_glucose,
                "BloodPressure": sim_bp,
                "SkinThickness": 28.0,
                "Insulin": 140.0,
                "BMI": sim_bmi,
                "DiabetesPedigreeFunction": 0.42,
                "Age": sim_age
            }
            vec_base = prepare_single_patient_vector(base_patient)
            base_prob = float(engine.predict_risk_score(vec_base)[0])
            base_pct = round(base_prob * 100, 1)

            # Post-Intervention Patient
            sim_patient = base_patient.copy()
            sim_patient["Glucose"] = max(70.0, sim_glucose - delta_glucose)
            sim_patient["BMI"] = max(18.5, sim_bmi - delta_bmi)
            sim_patient["BloodPressure"] = max(60.0, sim_bp - delta_bp)

            vec_sim = prepare_single_patient_vector(sim_patient)
            sim_prob = float(engine.predict_risk_score(vec_sim)[0])
            sim_pct = round(sim_prob * 100, 1)

            abs_reduction = round(base_pct - sim_pct, 1)
            rel_reduction = round(((base_pct - sim_pct) / max(base_pct, 0.1)) * 100, 1)

            # Visual Comparison
            comp_fig = go.Figure(data=[
                go.Bar(
                    name="Baseline Pre-Intervention",
                    x=["Diabetic Risk Index"],
                    y=[base_pct],
                    marker_color="#ef4444" if base_pct >= 65 else ("#f59e0b" if base_pct >= 30 else "#10b981"),
                    text=[f"{base_pct}%"],
                    textposition="auto"
                ),
                go.Bar(
                    name="Simulated Post-Intervention",
                    x=["Diabetic Risk Index"],
                    y=[sim_pct],
                    marker_color="#10b981" if sim_pct < 30 else ("#f59e0b" if sim_pct < 65 else "#ef4444"),
                    text=[f"{sim_pct}%"],
                    textposition="auto"
                )
            ])
            comp_fig.update_layout(
                barmode="group",
                title="<b>Pre vs Post Intervention Risk Comparison</b>",
                yaxis=dict(range=[0, 100], title="Risk Probability (%)"),
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(comp_fig, use_container_width=True)

            # Impact summary cards
            ic1, ic2 = st.columns(2)
            with ic1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Absolute Risk Reduction</div>
                    <div class="metric-value" style="color: #34d399;">-{abs_reduction}%</div>
                </div>
                """, unsafe_allow_html=True)
            with ic2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Relative Risk Reduction</div>
                    <div class="metric-value" style="color: #38bdf8;">-{rel_reduction}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="intervention-card" style="border-left-color: #38bdf8; margin-top: 14px;">
                <div class="intervention-title">🩺 Clinical Physician Interpretation</div>
                <div class="intervention-body">
                    By lowering fasting glucose by <b>{delta_glucose:.0f} mg/dL</b> and reducing BMI by <b>{delta_bmi:.1f} kg/m²</b>,
                    the patient's risk profile drops from <b>{base_pct}%</b> to <b>{sim_pct}%</b>.
                    This yields an outstanding <b>{rel_reduction}% relative reduction</b> in 5-year diabetes incidence risk.
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
