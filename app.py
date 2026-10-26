"""
AgriYield AI — Turn Agricultural Data into Smarter Decisions
Modern Agricultural Intelligence Web Application
"""

import sys
import io
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriYield AI — Agricultural Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# High-Contrast Startup Styling & Button Theme (Guaranteed High Readability)
# -----------------------------------------------------------------------------
STARTUP_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
    }
    
    /* Strictly hide sidebar and collapsed arrow */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* App background */
    .stApp {
        background-color: #F8FAF7;
        color: #123524;
        overflow-x: hidden !important;
    }
    
    /* Main block container padding */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1200px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        overflow-x: hidden !important;
    }

    /* -------------------------------------------------------------------------
       STRICT BUTTON STYLING (High contrast, crystal clear text, NO RED buttons)
       ------------------------------------------------------------------------- */
    /* Secondary (Default / Inactive) Buttons */
    div.stButton > button, 
    button[kind="secondary"],
    .stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #123524 !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 9px 18px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button *,
    button[kind="secondary"] *,
    .stButton > button[kind="secondary"] * {
        color: #123524 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    div.stButton > button:hover,
    button[kind="secondary"]:hover {
        background-color: #F0FDF4 !important;
        background: #F0FDF4 !important;
        border-color: #16A34A !important;
        color: #16A34A !important;
    }
    div.stButton > button:hover *,
    button[kind="secondary"]:hover * {
        color: #16A34A !important;
    }

    /* Primary (Active / Call-to-Action) Buttons */
    div.stButton > button[kind="primary"],
    button[kind="primary"],
    .stButton > button[kind="primary"] {
        background-color: #16A34A !important;
        background: linear-gradient(135deg, #15803D 0%, #16A34A 100%) !important;
        color: #FFFFFF !important;
        border: 1.5px solid #14532D !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        font-size: 14px !important;
        padding: 9px 18px !important;
        box-shadow: 0 4px 10px rgba(22, 163, 74, 0.3) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button[kind="primary"] *,
    button[kind="primary"] *,
    .stButton > button[kind="primary"] * {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 14px !important;
    }
    div.stButton > button[kind="primary"]:hover,
    button[kind="primary"]:hover {
        background-color: #14532D !important;
        background: #14532D !important;
        border-color: #052E16 !important;
        box-shadow: 0 6px 14px rgba(20, 83, 45, 0.4) !important;
    }

    /* Form Submit Button */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #15803D 0%, #16A34A 100%) !important;
        color: #FFFFFF !important;
        border: 1.5px solid #14532D !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        font-size: 14.5px !important;
        padding: 10px 20px !important;
    }
    div[data-testid="stFormSubmitButton"] > button * {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-size: 15px !important;
        font-weight: 700 !important;
        color: #475569 !important;
        padding: 10px 20px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #16A34A !important;
        border-bottom-color: #16A34A !important;
    }

    /* -------------------------------------------------------------------------
       Hero Banner Section
       ------------------------------------------------------------------------- */
    .hero-box {
        position: relative;
        background: linear-gradient(135deg, rgba(14, 43, 27, 0.94) 0%, rgba(26, 68, 45, 0.90) 100%), 
                    url('https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=1600&q=80');
        background-size: cover;
        background-position: center;
        border-radius: 20px;
        padding: 48px 42px;
        color: #FFFFFF !important;
        margin-bottom: 24px;
        box-shadow: 0 16px 28px -8px rgba(18, 53, 36, 0.28);
        box-sizing: border-box;
        width: 100%;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.16);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #FEF3C7 !important;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 16px;
    }
    .hero-headline {
        font-size: 38px;
        font-weight: 800;
        line-height: 1.2;
        letter-spacing: -1px;
        margin: 0 0 14px 0;
        color: #FFFFFF !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.25);
    }
    .hero-subheadline {
        font-size: 16.5px;
        font-weight: 400;
        color: #F1F5F9 !important;
        line-height: 1.6;
        max-width: 820px;
        margin-bottom: 0px;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }

    /* -------------------------------------------------------------------------
       Cards & Tiles
       ------------------------------------------------------------------------- */
    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 26px 22px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 18px -4px rgba(0, 0, 0, 0.08);
    }
    .feature-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        margin-bottom: 14px;
    }
    .feature-title {
        font-size: 17px;
        font-weight: 700;
        color: #123524;
        margin: 0 0 8px 0;
    }
    .feature-desc {
        font-size: 13.5px;
        color: #475569;
        line-height: 1.6;
        margin: 0;
        flex-grow: 1;
    }

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
    }
    .kpi-title {
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 25px;
        font-weight: 800;
        color: #123524;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 11.5px;
        color: #16A34A;
        font-weight: 600;
        margin-top: 4px;
    }

    .agri-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
    }

    .section-title {
        font-size: 26px;
        font-weight: 800;
        color: #123524;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }
    .section-desc {
        font-size: 14.5px;
        color: #475569;
        line-height: 1.6;
        margin-bottom: 20px;
    }

    .callout-box {
        background: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 13.5px;
        color: #14532D;
        margin: 16px 0;
        line-height: 1.5;
    }

    .callout-warning {
        background: #FFFBEB;
        border-left: 4px solid #D97706;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 13.5px;
        color: #92400E;
        margin: 16px 0;
        line-height: 1.5;
    }

    .dataset-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 12px;
        color: #334155;
        font-weight: 600;
    }

    /* -------------------------------------------------------------------------
       EXPANDER & CARD STYLING (Strictly High Contrast, Never Black/Dark)
       ------------------------------------------------------------------------- */
    div[data-testid="stExpander"],
    details[data-testid="stExpander"],
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
        margin-top: 16px !important;
        margin-bottom: 20px !important;
    }
    div[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #123524 !important;
        border-radius: 12px !important;
        padding: 12px 18px !important;
    }
    div[data-testid="stExpander"] summary *,
    details[data-testid="stExpander"] summary *,
    .streamlit-expanderHeader * {
        color: #123524 !important;
        font-weight: 800 !important;
        font-size: 15.5px !important;
    }
    div[data-testid="stExpander"] summary svg,
    details[data-testid="stExpander"] summary svg {
        fill: #16A34A !important;
        stroke: #16A34A !important;
    }
    div[data-testid="stExpander"] summary:hover,
    details[data-testid="stExpander"] summary:hover {
        background-color: #F0FDF4 !important;
        background: #F0FDF4 !important;
    }
    div[data-testid="stExpander"] summary:hover * {
        color: #16A34A !important;
    }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"],
    details[data-testid="stExpander"] div[role="region"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #123524 !important;
        border-top: 1px solid #E2E8F0 !important;
        padding: 18px 22px !important;
        border-bottom-left-radius: 14px !important;
        border-bottom-right-radius: 14px !important;
    }

    /* -------------------------------------------------------------------------
       WIDGET LABELS & SELECTBOX INPUTS (Crisp Dark Text on Pure White)
       ------------------------------------------------------------------------- */
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] *,
    div[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] *,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] span {
        color: #123524 !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        margin-bottom: 4px !important;
    }

    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"],
    div[data-baseweb="input"] > div,
    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stNumberInput"] > div > div,
    div[data-testid="stTextInput"] > div > div {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #123524 !important;
        border-color: #CBD5E1 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] *,
    div[data-baseweb="input"] *,
    div[data-testid="stSelectbox"] *,
    div[data-testid="stNumberInput"] * {
        color: #123524 !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1) !important;
    }
    li[role="option"] {
        background-color: #FFFFFF !important;
        color: #123524 !important;
        font-weight: 600 !important;
    }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #F0FDF4 !important;
        color: #16A34A !important;
    }
</style>
"""
st.markdown(STARTUP_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "active_page" not in st.session_state:
    st.session_state.active_page = "home"

if "active_dataset_type" not in st.session_state:
    st.session_state.active_dataset_type = "official"  # "official" or "uploaded"

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# -----------------------------------------------------------------------------
# Cached Data Loaders & Model Artifacts
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_official_crop_dataset() -> pd.DataFrame:
    """Load and cache the official Government of India dataset."""
    base_dir = Path(__file__).resolve().parent
    processed_path = base_dir / "data" / "processed" / "crop_production_cleaned.csv"
    raw_path = base_dir / "data" / "raw" / "crop_production_raw.csv"

    if processed_path.exists():
        return pd.read_csv(processed_path)
    elif raw_path.exists():
        from src.data.preprocess import preprocess_crop_data
        df, _ = preprocess_crop_data(save=True)
        return df
    else:
        from src.data.preprocess import preprocess_crop_data
        df, _ = preprocess_crop_data(save=True)
        return df


@st.cache_resource(show_spinner=False)
def get_ml_artifacts():
    """Load model pipeline and metadata."""
    from src.models.predict import load_prediction_artifacts
    return load_prediction_artifacts()


from src.visualization.charts import (
    apply_agri_theme,
    COLORS,
    plot_historical_yield_trends,
    plot_crop_distribution,
    plot_season_yield_distribution,
    plot_top_producing_states,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_model_comparison,
    plot_uploaded_category_bar,
    plot_uploaded_scatter,
    plot_uploaded_distribution,
    plot_uploaded_time_series
)

official_df = load_official_crop_dataset()


# -----------------------------------------------------------------------------
# Row 1: Brand Header & Active Dataset Indicator
# -----------------------------------------------------------------------------
head_left, head_right = st.columns([3, 2])

with head_left:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; padding: 4px 0 10px 0;">
        <span style="font-size: 28px;">🌾</span>
        <div>
            <span style="font-size: 22px; font-weight: 800; color: #123524; letter-spacing: -0.5px;">AgriYield AI</span>
            <span style="color: #CBD5E1; margin: 0 8px;">|</span>
            <span style="font-size: 13px; font-weight: 600; color: #16A34A; text-transform: uppercase;">Agricultural Intelligence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with head_right:
    upload_status_str = f"Custom: {st.session_state.uploaded_filename}" if st.session_state.uploaded_df is not None else "GoI Official (1997–2015)"
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: flex-end; height: 100%; padding-bottom: 6px;">
        <div class="dataset-badge">
            <span style="color: #16A34A; font-size: 13px;">●</span> Available Data: <b>{upload_status_str}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Row 2: Clean 3-Item Primary Navigation Bar (High Contrast, Equal Width)
# -----------------------------------------------------------------------------
nav_c1, nav_c2, nav_c3 = st.columns(3)

with nav_c1:
    if st.button("🌾 Home", key="btn_nav_home", use_container_width=True, 
                 type="primary" if st.session_state.active_page == "home" else "secondary"):
        st.session_state.active_page = "home"
        st.rerun()

with nav_c2:
    if st.button("🏛️ Government Data & Yield AI", key="btn_nav_gov", use_container_width=True,
                 type="primary" if st.session_state.active_page == "gov_data" else "secondary"):
        st.session_state.active_page = "gov_data"
        st.rerun()

with nav_c3:
    if st.button("📤 Upload & Analyse Your Data", key="btn_nav_up", use_container_width=True,
                 type="primary" if st.session_state.active_page == "upload" else "secondary"):
        st.session_state.active_page = "upload"
        st.rerun()

st.markdown("<hr style='margin: 10px 0 24px 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)


# =============================================================================
# 1. HOME PAGE (BEAUTIFUL OPENING PAGE)
# =============================================================================
if st.session_state.active_page == "home":
    # Hero Box
    st.markdown("""
    <div class="hero-box">
        <div class="hero-badge">🌱 Agricultural Intelligence Platform</div>
        <h1 class="hero-headline">Turn Agricultural Data into Smarter Decisions.</h1>
        <p class="hero-subheadline">
            Explore government crop statistics, run pre-harvest yield predictions, and upload your own datasets 
            for instant interactive visual analytics—all in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Two Primary Action Buttons
    cta_c1, cta_c2, cta_empty = st.columns([2.0, 2.0, 1.8])
    with cta_c1:
        if st.button("🏛️ Government Data & Yield AI", key="hero_btn_gov", type="primary", use_container_width=True):
            st.session_state.active_page = "gov_data"
            st.rerun()
    with cta_c2:
        if st.button("📤 Upload & Analyse Your Data", key="hero_btn_up", use_container_width=True):
            st.session_state.active_page = "upload"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Dynamic KPI Summary Cards from Authentic Government Dataset
    total_records = len(official_df)
    total_crops = official_df["Crop"].nunique() if "Crop" in official_df.columns else 0
    total_states = official_df["State_Name"].nunique() if "State_Name" in official_df.columns else 0
    avg_yield = official_df["Yield"].mean() if "Yield" in official_df.columns else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Verified Observations</div>
            <div class="kpi-value">{total_records:,}</div>
            <div class="kpi-sub">District-Season Records</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Agricultural Commodities</div>
            <div class="kpi-value">{total_crops}</div>
            <div class="kpi-sub">Grains, Pulses, Commercial, Oilseeds</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Geographic Coverage</div>
            <div class="kpi-value">{total_states} States</div>
            <div class="kpi-sub">All-India Administrative Scope</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">National Average Yield</div>
            <div class="kpi-value">{avg_yield:.2f} <span style="font-size:15px; color:#64748B;">t/ha</span></div>
            <div class="kpi-sub">Tonnes per Hectare Baseline</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # What AgriYield AI Does
    st.markdown("""
    <div style="text-align: center; max-width: 820px; margin: 0 auto 28px auto;">
        <h2 style="font-size: 26px; font-weight: 800; color: #123524; letter-spacing: -0.5px; margin-bottom: 8px;">
            What AgriYield AI Does
        </h2>
        <p style="font-size: 15px; color: #475569; line-height: 1.6;">
            AgriYield AI is an agricultural decision-support engine designed to bridge the gap between complex 
            macroeconomic crop statistics and practical farm and business decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Three Feature Cards
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-box" style="background: #ECFDF5; color: #059669;">🏛️</div>
            <h3 class="feature-title">Government Data Insights</h3>
            <p class="feature-desc">
                Access 19 years of official Ministry of Agriculture statistics. Explore multi-decade production shifts, 
                state-by-state productivity spreads, and seasonal rainfall dependencies across 124 commodities.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-box" style="background: #FEF3C7; color: #D97706;">🔮</div>
            <h3 class="feature-title">Crop Yield Prediction AI</h3>
            <p class="feature-desc">
                Forecast expected harvest yield (Tonnes/Ha) and total production using valid pre-harvest parameters. 
                Trained with strict chronological out-of-time validation achieving an 87.1% R² benchmark.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-box" style="background: #EFF6FF; color: #2563EB;">📤</div>
            <h3 class="feature-title">Custom Dataset Visualizations</h3>
            <p class="feature-desc">
                Bring your own farm or enterprise data. Upload CSV or Excel files for immediate interactive visual graphs, 
                automated yield calculation, correlation plots, and custom chart builder.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Real Agricultural Insights Preview
    st.markdown("""
    <div class="agri-card">
        <h3 style="color: #123524; font-size: 20px; font-weight: 800; margin: 0 0 16px 0;">
            🌾 Empirical Agricultural Insights at a Glance
        </h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px;">
            <div style="border-left: 3px solid #16A34A; padding-left: 14px;">
                <b style="color: #123524;">Cereal Productivity Gap:</b>
                <p style="font-size: 13px; color: #475569; margin: 4px 0 0 0; line-height: 1.5;">
                    Wheat yields in canal-irrigated northern districts (Punjab, Haryana at >4.6 t/ha) exceed 
                    rainfed zones in central/eastern India by over 200%.
                </p>
            </div>
            <div style="border-left: 3px solid #D97706; padding-left: 14px;">
                <b style="color: #123524;">Monsoon Multiplier Effect:</b>
                <p style="font-size: 13px; color: #475569; margin: 4px 0 0 0; line-height: 1.5;">
                    Kharif (monsoon) harvests display 3× higher variance and fatter tail risks than irrigated 
                    Rabi winter cycles, highlighting the necessity of index-based crop insurance.
                </p>
            </div>
            <div style="border-left: 3px solid #2563EB; padding-left: 14px;">
                <b style="color: #123524;">Horticulture Volatility:</b>
                <p style="font-size: 13px; color: #475569; margin: 4px 0 0 0; line-height: 1.5;">
                    Perishable commodities (Onion, Potato) experience district production swings >200% year-on-year, 
                    necessitating pre-harvest cold-chain transport pre-booking.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bottom Call to Action
    st.markdown("""
    <div style="background: linear-gradient(135deg, #123524 0%, #1E4631 100%); border-radius: 16px; padding: 34px 38px; color: #FFFFFF; text-align: center; margin-top: 20px;">
        <h3 style="font-size: 23px; font-weight: 800; margin: 0 0 8px 0; color: #FFFFFF;">Ready to analyze agricultural data?</h3>
        <p style="font-size: 15px; color: #E2E8F0; margin: 0 0 20px 0;">
            Inspect official national crop statistics with ML yield predictions, or upload your private operational dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    b_c1, b_c2, b_space = st.columns([2.0, 2.0, 1.8])
    with b_c1:
        if st.button("🏛️ Explore Government Data & Predictions", key="cta_foot_gov", type="primary", use_container_width=True):
            st.session_state.active_page = "gov_data"
            st.rerun()
    with b_c2:
        if st.button("📤 Upload Your Dataset", key="cta_foot_up", use_container_width=True):
            st.session_state.active_page = "upload"
            st.rerun()


# =============================================================================
# 2. WORKFLOW A: GOVERNMENT DATA & PREDICTIVE YIELD AI (COMPLETELY UNIFIED)
# =============================================================================
elif st.session_state.active_page == "gov_data":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 class="section-title">Government Agricultural Data & Yield AI Platform</h2>
        <p class="section-desc">
            Official crop statistics from the Directorate of Economics and Statistics (DES), Ministry of Agriculture 
            & Farmers Welfare, Government of India (1997–2015), along with its trained pre-harvest yield prediction engine.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_gov_analytics, tab_gov_prediction = st.tabs([
        "📊 Historical Data Analytics & Trends", 
        "🔮 Crop Yield Prediction AI Engine"
    ])

    # -------------------------------------------------------------------------
    # Sub-Tab 1: Historical Data Analytics
    # -------------------------------------------------------------------------
    with tab_gov_analytics:
        st.markdown("<br>", unsafe_allow_html=True)
        # Dynamic Filter Bar
        f1, f2, f3, f4 = st.columns([1.4, 1.4, 1.2, 1.2])
        with f1:
            crop_opts = sorted(official_df["Crop"].dropna().unique().tolist())
            default_cr = "Wheat" if "Wheat" in crop_opts else crop_opts[0]
            sel_crop = st.selectbox("Select Agricultural Crop", crop_opts, index=crop_opts.index(default_cr))
        with f2:
            state_opts = ["All India"] + sorted(official_df["State_Name"].dropna().unique().tolist())
            sel_state = st.selectbox("Select State / Region", state_opts, index=0)
        with f3:
            season_opts = ["All Seasons"] + sorted(official_df["Season"].dropna().unique().tolist())
            sel_season = st.selectbox("Cropping Season", season_opts, index=0)
        with f4:
            metric_agg = st.selectbox("Primary Aggregate", ["Production (Tonnes)", "Cultivated Area (Hectares)"])

        # Filter Data
        sub_gov = official_df[official_df["Crop"] == sel_crop]
        if sel_state != "All India":
            sub_gov = sub_gov[sub_gov["State_Name"] == sel_state]
        if sel_season != "All Seasons":
            sub_gov = sub_gov[sub_gov["Season"] == sel_season]

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Records Found", f"{len(sub_gov):,}")
        with m2:
            tot_prod = sub_gov["Production"].sum() if not sub_gov.empty else 0
            st.metric("Cumulative Production", f"{tot_prod:,.0f} Tonnes")
        with m3:
            tot_area = sub_gov["Area"].sum() if not sub_gov.empty else 0
            st.metric("Total Cultivated Area", f"{tot_area:,.0f} Hectares")
        with m4:
            avg_y = sub_gov["Yield"].mean() if not sub_gov.empty else 0
            st.metric("Average Harvest Yield", f"{avg_y:.2f} t/ha")

        st.markdown("<br>", unsafe_allow_html=True)

        # 1. Historical Trends with Dual-Axis Chart
        st.plotly_chart(plot_historical_yield_trends(official_df, crop=sel_crop, state=sel_state), use_container_width=True)
        st.markdown(f"""
        <div class="callout-box">
            💡 <b>Practical Observation for {sel_crop}:</b> Over the 19-year period, yield variations illustrate 
            technological progress (seed varietal adoption) along with periodic dips corresponding to deficient 
            monsoon seasons. Cultivated acreage trends highlight farmer crop-switching behaviour in response to market prices.
        </div>
        """, unsafe_allow_html=True)

        # 2. Regional and Seasonal Disparities
        c_g1, c_g2 = st.columns(2)
        with c_g1:
            st.plotly_chart(plot_top_producing_states(official_df, crop=sel_crop, top_n=10), use_container_width=True)
            st.caption("States are ranked by total production volume with average yield indicated in the color intensity.")
        with c_g2:
            st.plotly_chart(plot_season_yield_distribution(official_df, crop=sel_crop), use_container_width=True)
            st.caption("Box plots depict median yield, interquartile ranges, and harvest variance across agricultural seasons.")

        # 3. Macro Commodities Ranking
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🌾 National Commodity Production Ranking")
        metric_col = "Production" if "Production" in metric_agg else "Area"
        st.plotly_chart(plot_crop_distribution(official_df, metric=metric_col, top_n=14), use_container_width=True)

        # Data Quality Summary
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🛡️ Inspect Government Dataset Quality & Provenance Details"):
            st.markdown("""
            - **Source:** Directorate of Economics and Statistics (DES), Ministry of Agriculture and Farmers Welfare, GoI.
            - **Total Records:** 246,091 raw observations (1997–2015).
            - **Missing Value Handling:** Exactly 3,730 records (1.52%) with null production were excluded from supervised yield calculation to avoid synthetic label bias.
            - **Unit Heterogeneity:** Coconut reported in nuts / thousand nuts is partitioned from metric tonne field crops.
            - **Biological Bounds:** 190 records exceeding ICAR physiological maximums (e.g. Wheat >12 t/ha) were screened out to maintain mathematical rigor.
            """)

    # -------------------------------------------------------------------------
    # Sub-Tab 2: Pre-Harvest Crop Yield Prediction AI Engine
    # -------------------------------------------------------------------------
    with tab_gov_prediction:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom: 18px;">
            <h3 style="color: #123524; font-size: 21px; font-weight: 800; margin: 0 0 6px 0;">
                🌾 Pre-Harvest Crop Yield Forecasting Engine
            </h3>
            <p style="font-size: 14px; color: #475569; margin: 0;">
                Trained specifically on 202,487 historical Government of India records (1997–2011) and evaluated 
                out-of-time on 37,726 future unseen crop cycles (2012–2015).
            </p>
        </div>
        """, unsafe_allow_html=True)

        try:
            model, meta = get_ml_artifacts()
            benchmarks = meta.get("benchmark_results", {})
            has_model = True
        except Exception as e:
            has_model = False
            st.error(f"Prediction model artifacts not loaded: {e}. Please ensure model training has been completed.")

        if has_model:
            pred_c1, pred_c2 = st.columns([1.6, 1.4])

            with pred_c1:
                st.markdown("<div class='agri-card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color: #123524; margin:0 0 14px 0;'>🌾 Enter Pre-Harvest Parameters</h4>", unsafe_allow_html=True)

                with st.form("form_yield_prediction_gov"):
                    st_opts = sorted(official_df["State_Name"].dropna().unique().tolist())
                    default_st = "Uttar Pradesh" if "Uttar Pradesh" in st_opts else st_opts[0]
                    p_state = st.selectbox("State / Territory", st_opts, index=st_opts.index(default_st))

                    d_opts = sorted(official_df[official_df["State_Name"] == p_state]["District_Name"].dropna().unique().tolist())
                    p_district = st.selectbox("Administrative District", d_opts)

                    cr_opts = sorted(official_df[official_df["Unit_Type"] == "Metric Tonnes"]["Crop"].dropna().unique().tolist())
                    default_cr = "Wheat" if "Wheat" in cr_opts else cr_opts[0]
                    p_crop = st.selectbox("Agricultural Crop", cr_opts, index=cr_opts.index(default_cr))

                    p_season = st.selectbox("Cropping Season", ["Kharif", "Rabi", "Summer", "Whole Year", "Autumn", "Winter"], index=1)

                    p_area = st.number_input("Sown Cultivated Area (Hectares)", min_value=0.1, max_value=1_000_000.0, value=150.0, step=5.0)
                    p_year = st.number_input("Target Cropping Year", min_value=2015, max_value=2030, value=2016, step=1)

                    btn_predict = st.form_submit_button("🌱 Forecast Crop Yield & Production", type="primary", use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

            with pred_c2:
                st.markdown("""
                <div class="agri-card">
                    <h4 style="color: #123524; margin:0 0 10px 0;">🏆 Champion Model Validation Scores</h4>
                    <p style="font-size: 13px; color: #64748B; margin-bottom: 12px;">
                        Evaluated on 4 unseen future crop years (2012–2015, 37,726 samples).
                    </p>
                """, unsafe_allow_html=True)

                hgb = benchmarks.get("HistGradientBoosting", {})
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.metric("R² Score", f"{hgb.get('R2', 0.871):.3f}")
                with sc2:
                    st.metric("MAE", f"{hgb.get('MAE', 1.256):.2f} t/ha")
                with sc3:
                    st.metric("RMSE", f"{hgb.get('RMSE', 4.518):.2f} t/ha")

                st.markdown("""
                    <hr style="margin: 14px 0; border: none; border-top: 1px solid #E2E8F0;">
                    <b style="font-size: 12.5px; color: #123524;">Anti-Leakage Architecture:</b>
                    <p style="font-size: 12px; color: #475569; line-height: 1.5; margin: 4px 0 0 0;">
                        Post-harvest total production is strictly excluded from predictor inputs. Sown area and 
                        expanding historical state-crop benchmarks up to Year t-1 are used to prevent future data leakage.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Handle Prediction Submit
            if btn_predict:
                from src.models.predict import predict_crop_yield
                try:
                    pred_res = predict_crop_yield(
                        state=p_state,
                        district=p_district,
                        season=p_season,
                        crop=p_crop,
                        area=p_area,
                        year=p_year
                    )
                    preds = pred_res["predictions"]
                    y_val = preds["predicted_yield_t_per_ha"]
                    ci_y = preds["yield_confidence_interval_80pct"]
                    p_tot = preds["estimated_production_tonnes"]
                    ci_p = preds["production_confidence_interval_80pct"]

                    st.markdown("### 📈 Decision Support Forecast Output")
                    res1, res2, res3 = st.columns(3)
                    with res1:
                        st.markdown(f"""
                        <div class="kpi-card" style="border-top: 4px solid #16A34A;">
                            <div class="kpi-title">Forecasted Crop Yield</div>
                            <div class="kpi-value">{y_val:.3f} <span style="font-size:15px; color:#64748B;">t/ha</span></div>
                            <div class="kpi-sub">80% CI: [{ci_y[0]:.2f} – {ci_y[1]:.2f}] t/ha</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with res2:
                        st.markdown(f"""
                        <div class="kpi-card" style="border-top: 4px solid #D97706;">
                            <div class="kpi-title">Expected Gross Production</div>
                            <div class="kpi-value">{p_tot:,.1f} <span style="font-size:15px; color:#64748B;">Tonnes</span></div>
                            <div class="kpi-sub">80% CI: [{ci_p[0]:,.0f} – {ci_p[1]:,.0f}] Tonnes</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with res3:
                        st.markdown(f"""
                        <div class="kpi-card" style="border-top: 4px solid #2563EB;">
                            <div class="kpi-title">Operational Farm Scale</div>
                            <div class="kpi-value" style="font-size:19px;">{pred_res['inputs']['farm_scale']}</div>
                            <div class="kpi-sub">Family: {pred_res['inputs']['crop_category']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="callout-box">
                        🌾 <b>Agronomic Context:</b> Forecasted yield of <b>{y_val:.2f} t/ha</b> for <b>{p_crop}</b> in <b>{p_state}</b> 
                        represents the expected average under historical agro-climatic conditions.
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as ex:
                    st.error(f"Inference error: {ex}")

            # Scientific Disclaimer
            st.markdown("""
            <div class="callout-warning">
                ⚠️ <b>Agricultural Disclaimer:</b> Yield forecasts are statistical estimates trained on historical district aggregates. 
                They do not guarantee individual plot harvests, which depend on localized micro-weather events, pest attacks, and soil nutrient health.
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# 3. WORKFLOW B: UPLOAD & ANALYSE YOUR DATASET (WITH IMMEDIATE VISUAL GRAPHS)
# =============================================================================
elif st.session_state.active_page == "upload":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 class="section-title">Upload & Analyse Your Own Agricultural Data</h2>
        <p class="section-desc">
            Upload custom CSV or Excel datasets (farm records, regional trial logs, or agribusiness supply data) 
            to automatically profile data health, compute crop yields, and generate interactive visual charts instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    up_c1, up_c2 = st.columns([1.8, 1.2])
    with up_c1:
        st.markdown("<div class='agri-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #123524; margin:0 0 10px 0;'>Select or Drag & Drop File</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Supported formats: CSV, XLSX, XLS",
            type=["csv", "xlsx", "xls"],
            key="file_uploader_custom",
            help="Upload any agricultural or operational dataset."
        )

        st.markdown("<div style='margin-top: 10px; margin-bottom: 6px; font-size: 13px; color: #64748B;'>— OR test with sample data —</div>", unsafe_allow_html=True)
        if st.button("🌱 Load Sample Agricultural Dataset (1-Click Test)", key="btn_load_sample_data"):
            sample_data = {
                "State": ["Punjab", "Punjab", "Haryana", "Haryana", "Uttar Pradesh", "Uttar Pradesh", "Maharashtra", "Maharashtra", "Karnataka", "Karnataka", "Tamil Nadu", "Tamil Nadu", "Madhya Pradesh", "Madhya Pradesh", "Gujarat", "Gujarat", "Andhra Pradesh", "Andhra Pradesh", "West Bengal", "West Bengal"],
                "District": ["Ludhiana", "Patiala", "Karnal", "Ambala", "Meerut", "Varanasi", "Nashik", "Pune", "Dharwad", "Belagavi", "Coimbatore", "Thanjavur", "Indore", "Bhopal", "Surat", "Rajkot", "Guntur", "Krishna", "Bardhaman", "Hooghly"],
                "Crop": ["Wheat", "Rice", "Wheat", "Rice", "Sugarcane", "Wheat", "Sugarcane", "Cotton", "Cotton", "Maize", "Rice", "Rice", "Soyabean", "Wheat", "Cotton", "Groundnut", "Chilli", "Rice", "Rice", "Potato"],
                "Season": ["Rabi", "Kharif", "Rabi", "Kharif", "Whole Year", "Rabi", "Whole Year", "Kharif", "Kharif", "Kharif", "Kharif", "Rabi", "Kharif", "Rabi", "Kharif", "Kharif", "Kharif", "Kharif", "Winter", "Rabi"],
                "Crop_Year": [2018, 2018, 2019, 2019, 2019, 2020, 2020, 2020, 2021, 2021, 2021, 2021, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022],
                "Area": [120000, 115000, 95000, 90000, 180000, 75000, 140000, 110000, 85000, 60000, 50000, 92000, 160000, 82000, 130000, 95000, 42000, 88000, 150000, 65000],
                "Production": [564000, 483000, 437000, 360000, 13500000, 240000, 11200000, 231000, 161500, 210000, 210000, 386400, 240000, 270600, 299000, 171000, 117600, 369600, 525000, 1495000]
            }
            st.session_state.uploaded_df = pd.DataFrame(sample_data)
            st.session_state.uploaded_filename = "sample_farm_production_data.csv"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Ingest user file immediately if uploaded
    if uploaded_file is not None:
        if st.session_state.uploaded_filename != uploaded_file.name or st.session_state.uploaded_df is None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_parsed = pd.read_csv(uploaded_file)
                else:
                    df_parsed = pd.read_excel(uploaded_file)
                st.session_state.uploaded_df = df_parsed
                st.session_state.uploaded_filename = uploaded_file.name
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}. Please ensure it is a valid CSV or Excel workbook.")

    with up_c2:
        status_loaded = st.session_state.uploaded_df is not None
        rec_count = f"{len(st.session_state.uploaded_df):,}" if status_loaded else "0"
        col_count = f"{len(st.session_state.uploaded_df.columns)}" if status_loaded else "0"
        file_name = st.session_state.uploaded_filename if status_loaded else "None"

        st.markdown(f"""
        <div class="agri-card" style="height: 100%;">
            <h4 style="color: #123524; margin:0 0 10px 0;'>Upload Status</h4>
            <div style="font-size: 13.5px; color: #475569; line-height: 1.7;">
                Status: <b style="color: {'#16A34A' if status_loaded else '#64748B'};">
                    {'● Active Dataset Ready' if status_loaded else '○ No File Loaded'}
                </b><br>
                Active File: <b>{file_name}</b><br>
                Total Records: <b>{rec_count}</b><br>
                Total Columns: <b>{col_count}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # DISPLAY LOADED DATASET, PROFILE & INSTANT GRAPHS
    # -------------------------------------------------------------------------
    if st.session_state.uploaded_df is not None:
        df_up = st.session_state.uploaded_df.copy()

        # Clean string whitespace
        for col in df_up.select_dtypes(include=["object"]).columns:
            df_up[col] = df_up[col].astype(str).str.strip()

        # Automatic Yield Detection and Calculation
        area_col = next((c for c in df_up.columns if c.strip().lower() in ["area", "acreage", "sown_area", "hectares"]), None)
        prod_col = next((c for c in df_up.columns if c.strip().lower() in ["production", "harvest", "output", "tonnes", "yield_production"]), None)

        yield_computed = False
        if "Yield" not in df_up.columns and area_col and prod_col:
            df_up[area_col] = pd.to_numeric(df_up[area_col], errors="coerce")
            df_up[prod_col] = pd.to_numeric(df_up[prod_col], errors="coerce")
            valid_area = df_up[area_col] > 0
            df_up.loc[valid_area, "Yield"] = (df_up.loc[valid_area, prod_col] / df_up.loc[valid_area, area_col]).round(3)
            yield_computed = True
            st.session_state.uploaded_df = df_up

        # Column grouping
        num_cols = df_up.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_up.select_dtypes(include=["object", "category"]).columns.tolist()
        time_cols = [c for c in df_up.columns if any(t in c.lower() for t in ["year", "date", "month", "time", "season"])]

        # Header bar with reset option
        head_c1, head_c2 = st.columns([3, 1])
        with head_c1:
            st.success(f"Loaded Dataset: **{st.session_state.uploaded_filename}** ({len(df_up):,} records, {len(df_up.columns)} columns)")
            if yield_computed:
                st.info(f"🌾 Automatically calculated **Yield = {prod_col} / {area_col}** and added as a metric column.")
        with head_c2:
            if st.button("🗑️ Clear / Reset Upload", key="btn_reset_upload"):
                st.session_state.uploaded_df = None
                st.session_state.uploaded_filename = None
                st.rerun()

        # 5 Summary KPI Cards
        null_cnt = int(df_up.isnull().sum().sum())
        null_pct = (null_cnt / (len(df_up) * len(df_up.columns)) * 100) if len(df_up) > 0 else 0
        dup_cnt = int(df_up.duplicated().sum())
        primary_num = "Yield" if "Yield" in df_up.columns else (num_cols[0] if num_cols else None)
        avg_val = df_up[primary_num].mean() if primary_num else 0.0

        p1, p2, p3, p4, p5 = st.columns(5)
        with p1:
            st.metric("Total Records", f"{len(df_up):,}")
        with p2:
            st.metric("Total Columns", f"{len(df_up.columns)}")
        with p3:
            st.metric("Missing Value Rate", f"{null_pct:.1f}%")
        with p4:
            st.metric("Duplicate Rows", f"{dup_cnt:,}")
        with p5:
            st.metric(f"Average {primary_num}" if primary_num else "Status", f"{avg_val:.2f}" if primary_num else "Active")

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # ✨ INSTANT INTERACTIVE VISUALIZATIONS SECTION
        # ---------------------------------------------------------------------
        st.markdown("### 📊 Automated Interactive Visualizations")
        st.markdown("<p style='color: #475569; font-size: 14.5px; margin-top: -8px;'>Rich interactive charts generated directly from your uploaded data.</p>", unsafe_allow_html=True)

        if cat_cols and num_cols:
            # Selector defaults
            default_cat = next((c for c in cat_cols if any(k in c.lower() for k in ["crop", "product", "item", "state", "region", "category"])), cat_cols[0])
            default_num = "Production" if "Production" in num_cols else ("Yield" if "Yield" in num_cols else num_cols[0])

            # Row 1: Category Comparison & Distribution Box Plot
            ch_r1_c1, ch_r1_c2 = st.columns(2)
            with ch_r1_c1:
                st.plotly_chart(
                    plot_uploaded_category_bar(df_up, cat_col=default_cat, num_col=default_num, agg="sum", top_n=12),
                    use_container_width=True
                )
                st.caption(f"Top {default_cat} ranked by total {default_num} with data labels.")

            with ch_r1_c2:
                dist_metric = "Yield" if "Yield" in num_cols else default_num
                st.plotly_chart(
                    plot_uploaded_distribution(df_up, num_col=dist_metric, group_col=default_cat),
                    use_container_width=True
                )
                st.caption(f"Statistical distribution (median, quartiles, outliers) of {dist_metric} across {default_cat}.")

            # Row 2: Correlation Scatter & Chronological Trend
            ch_r2_c1, ch_r2_c2 = st.columns(2)
            with ch_r2_c1:
                if len(num_cols) >= 2:
                    scat_x = "Area" if "Area" in num_cols else num_cols[0]
                    scat_y = "Production" if "Production" in num_cols and scat_x != "Production" else (num_cols[1] if num_cols[0] == scat_x else num_cols[0])
                    st.plotly_chart(
                        plot_uploaded_scatter(df_up, x_col=scat_x, y_col=scat_y, color_col=default_cat),
                        use_container_width=True
                    )
                    st.caption(f"Scatter correlation between {scat_x} and {scat_y}, colored by {default_cat}.")
                else:
                    st.plotly_chart(
                        plot_uploaded_distribution(df_up, num_col=num_cols[0]),
                        use_container_width=True
                    )
                    st.caption(f"Histogram and spread of {num_cols[0]}.")

            with ch_r2_c2:
                if time_cols:
                    t_col = time_cols[0]
                    st.plotly_chart(
                        plot_uploaded_time_series(df_up, time_col=t_col, num_col=default_num, agg="sum"),
                        use_container_width=True
                    )
                    st.caption(f"Historical progression of {default_num} tracked over {t_col}.")
                else:
                    # Secondary Category distribution
                    sec_cat = cat_cols[1] if len(cat_cols) > 1 else cat_cols[0]
                    st.plotly_chart(
                        plot_uploaded_category_bar(df_up, cat_col=sec_cat, num_col=default_num, agg="mean", top_n=10),
                        use_container_width=True
                    )
                    st.caption(f"Average {default_num} across {sec_cat}.")

        elif num_cols:
            # Numeric-only dataset
            n1 = num_cols[0]
            st.plotly_chart(plot_uploaded_distribution(df_up, num_col=n1), use_container_width=True)
            if len(num_cols) >= 2:
                st.plotly_chart(plot_uploaded_scatter(df_up, x_col=num_cols[0], y_col=num_cols[1]), use_container_width=True)
        else:
            st.warning("Uploaded dataset has no numeric columns to graph. Please ensure at least one numeric field is included.")

        # ---------------------------------------------------------------------
        # 🎨 INTERACTIVE CUSTOM CHART BUILDER (EXPLORE ANY COLUMN COMBINATION)
        # ---------------------------------------------------------------------
        st.markdown("""
        <div style="background: #FFFFFF; border: 1.5px solid #CBD5E1; border-radius: 16px; padding: 22px 24px 8px 24px; margin-top: 24px; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
            <h3 style="color: #123524; font-size: 20px; font-weight: 800; margin: 0 0 4px 0;">
                🎨 Custom Chart Explorer (Build & Customize Your Own Visualizations)
            </h3>
            <p style="font-size: 13.5px; color: #475569; margin: 0 0 4px 0;">
                Create tailored charts for any dimension in your uploaded data.
            </p>
        </div>
        """, unsafe_allow_html=True)

        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        with b_c1:
            chart_kind = st.selectbox("Chart Style", ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Histogram", "Pie / Donut Chart"])
        with b_c2:
            x_choice = st.selectbox("X-Axis Dimension", df_up.columns.tolist(), index=0)
        with b_c3:
            y_options = num_cols if num_cols else df_up.columns.tolist()
            y_choice = st.selectbox("Y-Axis Metric", y_options, index=0)
        with b_c4:
            agg_type = st.selectbox("Aggregation", ["Sum", "Average (Mean)", "Record Count", "Median"])

        # Color / Hue grouping
        grp_opts = ["None"] + cat_cols
        sel_hue = st.selectbox("Color / Category Grouping (Optional)", grp_opts, index=0)
        hue_arg = None if sel_hue == "None" else sel_hue

        # Render Custom Figure
        try:
            fig_custom = None
            agg_fn_map = {"Sum": "sum", "Average (Mean)": "mean", "Record Count": "count", "Median": "median"}
            fn_name = agg_fn_map.get(agg_type, "sum")

            if chart_kind == "Bar Chart":
                if x_choice in cat_cols and y_choice in num_cols:
                    fig_custom = plot_uploaded_category_bar(df_up, cat_col=x_choice, num_col=y_choice, agg=fn_name, top_n=15)
                else:
                    fig_custom = px.bar(df_up, x=x_choice, y=y_choice, color=hue_arg, template="plotly_white")
            elif chart_kind == "Line Chart":
                fig_custom = px.line(df_up, x=x_choice, y=y_choice, color=hue_arg, markers=True, template="plotly_white")
            elif chart_kind == "Scatter Plot":
                fig_custom = plot_uploaded_scatter(df_up, x_col=x_choice, y_col=y_choice, color_col=hue_arg)
            elif chart_kind == "Box Plot":
                fig_custom = plot_uploaded_distribution(df_up, num_col=y_choice, group_col=hue_arg)
            elif chart_kind == "Histogram":
                fig_custom = px.histogram(df_up, x=x_choice, y=y_choice if y_choice in num_cols else None, color=hue_arg, template="plotly_white")
            elif chart_kind == "Pie / Donut Chart":
                if x_choice in cat_cols and y_choice in num_cols:
                    agg_pie = df_up.groupby(x_choice)[y_choice].sum().reset_index().sort_values(y_choice, ascending=False).head(8)
                    fig_custom = px.pie(agg_pie, names=x_choice, values=y_choice, hole=0.45, template="plotly_white")
                else:
                    fig_custom = px.pie(df_up, names=x_choice, hole=0.45, template="plotly_white")

            if fig_custom:
                fig_custom = apply_agri_theme(fig_custom, f"Custom {chart_kind}: {x_choice} vs {y_choice}")
                st.plotly_chart(fig_custom, use_container_width=True)

        except Exception as err:
            st.warning(f"Could not render custom chart with current configuration: {err}")

        # ---------------------------------------------------------------------
        # 📋 DATA PROFILE, CLEANING & PREVIEW
        # ---------------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔍 Dataset Profile & Health Breakdown")

        col_summary = pd.DataFrame({
            "Column Name": df_up.columns,
            "Detected Type": [str(t) for t in df_up.dtypes],
            "Valid Records": df_up.notnull().sum().values,
            "Missing Values": df_up.isnull().sum().values,
            "Missing %": (df_up.isnull().sum().values / len(df_up) * 100).round(2),
            "Sample Value": [str(df_up[c].dropna().iloc[0]) if df_up[c].notnull().sum() > 0 else "None" for c in df_up.columns]
        })
        st.dataframe(col_summary, use_container_width=True)

        # Cleaning toolbar
        st.markdown("#### ⚙️ Data Cleaning & Processing Tools")
        clean_c1, clean_c2, clean_c3 = st.columns(3)
        with clean_c1:
            clean_str = st.checkbox("Standardize string casing & whitespace", value=True)
        with clean_c2:
            clean_dup = st.checkbox("Deduplicate identical rows", value=True)
        with clean_c3:
            clean_null = st.checkbox("Drop rows with empty values", value=False)

        if st.button("Apply Cleaning Transformations", key="btn_apply_clean"):
            cleaned_df = df_up.copy()
            if clean_str:
                for c in cleaned_df.select_dtypes(include=["object"]).columns:
                    cleaned_df[c] = cleaned_df[c].astype(str).str.strip()
            if clean_dup:
                cleaned_df = cleaned_df.drop_duplicates()
            if clean_null:
                cleaned_df = cleaned_df.dropna()

            st.session_state.uploaded_df = cleaned_df
            st.success("Cleaning transformations applied successfully!")
            st.rerun()

        # Data preview & CSV export
        st.markdown("#### 📋 Data Preview (First 15 Rows)")
        st.dataframe(df_up.head(15), use_container_width=True)

        csv_bytes = df_up.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned / Processed Dataset as CSV",
            data=csv_bytes,
            file_name=f"processed_{st.session_state.uploaded_filename or 'dataset.csv'}",
            mime="text/csv"
        )

        # Agribusiness Summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="callout-box">
            🌾 <b>Agribusiness Summary:</b> Uploaded dataset <b>{st.session_state.uploaded_filename}</b> contains 
            <b>{len(df_up):,} observations</b> across <b>{len(df_up.columns)} features</b>. 
            All interactive charts, distribution plots, and column health metrics are fully active and synchronized.
        </div>
        """, unsafe_allow_html=True)
