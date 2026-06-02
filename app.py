# =========================================
# RAPIQ — Rapid IQ Classification System
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="RapIQ — Intelligence Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# CUSTOM CSS
# Light theme — clean, modern, professional
# Font: Plus Jakarta Sans
# Palette: White base, indigo accent, soft grays
# =========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:           #F4F6FB;
    --sidebar-bg:   #FFFFFF;
    --card-bg:      #FFFFFF;
    --input-bg:     #F7F8FC;
    --border:       #E4E8F0;
    --border-mid:   #D0D6E4;

    --indigo:       #4F46E5;
    --indigo-light: #EEF2FF;
    --indigo-mid:   #C7D2FE;
    --indigo-hover: #4338CA;

    --text-primary:   #111827;
    --text-secondary: #6B7280;
    --text-muted:     #9CA3AF;

    --green:  #10B981;
    --red:    #EF4444;
    --orange: #F97316;
    --amber:  #F59E0B;
    --purple: #8B5CF6;

    --radius-sm:  6px;
    --radius-md:  10px;
    --radius-lg:  14px;
    --radius-xl:  18px;
    --radius-pill:999px;

    --shadow-sm:   0 1px 4px rgba(0,0,0,0.06);
    --shadow-card: 0 2px 12px rgba(0,0,0,0.07);
    --shadow-btn:  0 4px 14px rgba(79,70,229,0.30);

    --font: 'Plus Jakarta Sans', sans-serif;
}

/* ── App Shell ── */
.stApp {
    background: var(--bg) !important;
    font-family: var(--font) !important;
    color: var(--text-primary) !important;
}

#MainMenu, footer, header         { visibility: hidden; }
.stDeployButton                   { display: none !important; }
[data-testid="stToolbar"]         { display: none !important; }

.main .block-container {
    padding: 2.2rem 2.8rem 4rem !important;
    max-width: 1400px !important;
}

[data-testid="column"]            { padding: 0 8px !important; }
[data-testid="stHorizontalBlock"] { gap: 0 !important; }

hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.6rem 0 !important;
}

::-webkit-scrollbar               { width: 5px; height: 5px; }
::-webkit-scrollbar-track         { background: #F1F3F7; }
::-webkit-scrollbar-thumb         { background: #D1D5DB; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover   { background: #9CA3AF; }

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child,
section[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
    width: 260px !important;
}
[data-testid="stSidebar"] ul,
[data-testid="stSidebar"] nav,
[data-testid="stSidebar"] section,
[data-testid="stSidebar"] .css-1d391kg,
[data-testid="stSidebar"] .css-163ttbj,
[data-testid="stSidebar"] .css-1wrcr25,
ul[data-testid="stSidebarNavItems"],
nav { background: transparent !important; }

/* Sidebar logo */
.sidebar-logo {
    padding: 28px 20px 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}
.logo-mark {
    width: 52px;
    height: 52px;
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35);
}
.logo-name {
    font-family: var(--font);
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.logo-sub {
    font-size: 11.5px;
    color: var(--text-muted);
    margin-top: 2px;
    font-weight: 400;
}

/* Nav links */
.nav-link {
    font-family: var(--font) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-md) !important;
    margin: 2px 10px !important;
    padding: 11px 14px !important;
    transition: all 0.15s ease !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    min-height: 46px !important;
    border: 1px solid transparent !important;
}
.nav-link:hover {
    background: var(--indigo-light) !important;
    color: var(--indigo) !important;
}
.nav-link.active,
.nav-link-selected {
    background: var(--indigo-light) !important;
    border: 1px solid var(--indigo-mid) !important;
    color: var(--indigo) !important;
    font-weight: 700 !important;
}
.nav-link span { overflow: hidden !important; text-overflow: ellipsis !important; }
.nav-link i    { font-size: 14px !important; min-width: 16px !important; }

/* Sidebar footer */
.sidebar-status {
    padding: 16px 20px 24px;
    border-top: 1px solid var(--border);
    margin-top: 8px;
}
.sidebar-status-label {
    font-size: 10px;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.sidebar-status-item {
    font-size: 12px;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 8px;
    line-height: 2;
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    display: inline-block;
    flex-shrink: 0;
}

/* ─────────────────────────────────────────
   PAGE HEADER
───────────────────────────────────────── */
.page-header {
    margin-bottom: 1.8rem;
    padding-bottom: 1.4rem;
    border-bottom: 1px solid var(--border);
}
.page-header-icon {
    font-size: 2.4rem;
    margin-bottom: 4px;
    line-height: 1;
}
.page-header-title {
    font-family: var(--font);
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.page-header-sub {
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 5px;
    font-weight: 400;
}

/* ─────────────────────────────────────────
   SECTION TITLE (inline, like screenshot)
───────────────────────────────────────── */
.section-heading {
    font-family: var(--font);
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 9px;
    letter-spacing: -0.01em;
}

/* ─────────────────────────────────────────
   FORM SECTION (white card)
───────────────────────────────────────── */
.form-section {
    background: transparent;
    padding: 0;
    margin-bottom: 1.6rem;
}

/* Divider line between sections */
.section-divider {
    height: 1px;
    background: var(--border);
    margin: 1.8rem 0;
}

/* ─────────────────────────────────────────
   INPUT LABEL
───────────────────────────────────────── */
.input-label {
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 6px;
    display: block;
}

/* ─────────────────────────────────────────
   CARD (white surface)
───────────────────────────────────────── */
.rapiq-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-card);
}

/* ─────────────────────────────────────────
   METRIC CARDS
───────────────────────────────────────── */
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.3rem 1.4rem;
    box-shadow: var(--shadow-sm);
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: var(--font);
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1;
    letter-spacing: -0.03em;
}
.metric-sub {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 5px;
}

/* ─────────────────────────────────────────
   PREDICTION RESULT
───────────────────────────────────────── */
.prediction-result-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 1.8rem 2rem;
    margin: 1.2rem 0;
    box-shadow: var(--shadow-card);
}
.prediction-tag {
    font-size: 10.5px;
    font-weight: 700;
    color: var(--indigo);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.prediction-category {
    font-family: var(--font);
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.25;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
}
.prediction-desc {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 1.3rem;
    line-height: 1.65;
}

/* ─────────────────────────────────────────
   TAGS
───────────────────────────────────────── */
.tag {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: var(--radius-sm);
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.tag-blue   { background: var(--indigo-light); color: var(--indigo); }
.tag-purple { background: #F5F3FF; color: var(--purple); }
.tag-green  { background: #ECFDF5; color: var(--green); }

/* ─────────────────────────────────────────
   ABOUT — INFO BLOCK ROW
───────────────────────────────────────── */
.info-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 11px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13.5px;
}
.info-row:last-child { border-bottom: none; }
.info-row-label {
    color: var(--text-secondary);
    font-weight: 400;
}
.info-row-value {
    color: var(--text-primary);
    font-weight: 700;
    text-align: right;
}

/* ─────────────────────────────────────────
   OUTPUT CLASS ROWS
───────────────────────────────────────── */
.class-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    background: var(--card-bg);
    margin-bottom: 6px;
    box-shadow: var(--shadow-sm);
    transition: border-color 0.15s, box-shadow 0.15s;
}
.class-row:hover {
    border-color: var(--indigo-mid);
    box-shadow: 0 2px 8px rgba(79,70,229,0.10);
}
.class-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.class-row-name {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-primary);
}
.class-row-num {
    margin-left: auto;
    font-size: 10px;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ─────────────────────────────────────────
   DISCLAIMER BOX (yellow tint)
───────────────────────────────────────── */
.disclaimer-box {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: var(--radius-lg);
    padding: 1.1rem 1.4rem;
    font-size: 12.5px;
    color: #92400E;
    line-height: 1.75;
}
.disclaimer-box p { margin-bottom: 4px; }
.disclaimer-box p:last-child { margin-bottom: 0; }

/* ─────────────────────────────────────────
   FORM CONTROLS
───────────────────────────────────────── */

/* Selectbox */
[data-baseweb="select"] > div {
    background: var(--input-bg) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: var(--font) !important;
    font-size: 14px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
[data-baseweb="select"] > div:hover,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.10) !important;
}
[data-baseweb="popover"] {
    background: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12) !important;
}
[role="option"] {
    background: #FFFFFF !important;
    color: var(--text-secondary) !important;
    font-size: 13.5px !important;
    font-family: var(--font) !important;
}
[role="option"]:hover {
    background: var(--indigo-light) !important;
    color: var(--indigo) !important;
}

/* Number input */
[data-testid="stNumberInput"] input {
    background: var(--input-bg) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: var(--font) !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.10) !important;
    outline: none !important;
}
[data-testid="stNumberInput"] button {
    background: var(--input-bg) !important;
    border-color: var(--border-mid) !important;
    color: var(--text-secondary) !important;
    transition: color 0.15s !important;
}
[data-testid="stNumberInput"] button:hover {
    color: var(--indigo) !important;
}

/* Radio */
[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 12px !important;
}
[data-testid="stRadio"] label {
    background: var(--input-bg) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-md) !important;
    padding: 10px 22px !important;
    color: var(--text-secondary) !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    min-width: 110px !important;
    white-space: nowrap !important;
    font-family: var(--font) !important;
}
[data-testid="stRadio"] label:hover {
    border-color: var(--indigo) !important;
    color: var(--indigo) !important;
    background: var(--indigo-light) !important;
}
[data-testid="stRadio"] label[data-checked="true"] {
    border-color: var(--indigo) !important;
    background: var(--indigo-light) !important;
    color: var(--indigo) !important;
    font-weight: 700 !important;
}

/* Primary Button */
.stButton > button {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    color: #FFFFFF !important;
    font-family: var(--font) !important;
    font-weight: 700 !important;
    font-size: 14.5px !important;
    letter-spacing: 0.01em !important;
    border: none !important;
    border-radius: var(--radius-pill) !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    height: 54px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: var(--shadow-btn) !important;
}
.stButton > button:hover {
    opacity: 0.90 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79,70,229,0.38) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    opacity: 1 !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: var(--input-bg) !important;
    border: 1px solid var(--border-mid) !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-md) !important;
    font-size: 13px !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-sm) !important;
    height: auto !important;
    padding: 8px 16px !important;
    transition: all 0.15s !important;
    width: auto !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--indigo) !important;
    color: var(--indigo) !important;
    background: var(--indigo-light) !important;
    transform: none !important;
    box-shadow: var(--shadow-sm) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--input-bg) !important;
    border: 1.5px dashed var(--border-mid) !important;
    border-radius: var(--radius-lg) !important;
    padding: 10px !important;
    transition: all 0.15s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--indigo) !important;
    background: var(--indigo-light) !important;
}
section[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-family: var(--font) !important;
}
[data-testid="stFileUploader"] button {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-mid) !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-md) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    font-family: var(--font) !important;
    transition: all 0.15s !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stFileUploader"] button:hover {
    border-color: var(--indigo) !important;
    color: var(--indigo) !important;
    background: var(--indigo-light) !important;
}

/* Progress */
[data-testid="stProgress"] > div {
    background: #E5E7EB !important;
    border-radius: var(--radius-pill) !important;
    height: 6px !important;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--indigo), var(--purple)) !important;
    border-radius: var(--radius-pill) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}
.stDataFrame th {
    background: #F9FAFB !important;
    color: var(--text-muted) !important;
    font-size: 10.5px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--border) !important;
}
.stDataFrame td {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    padding: 9px 14px !important;
    border-bottom: 1px solid rgba(228,232,240,0.5) !important;
}

/* Info box */
.info-box {
    background: #EFF6FF;
    border-left: 3px solid var(--indigo);
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #1E40AF;
    font-weight: 500;
}

/* Spinner */
[data-testid="stSpinner"] { color: var(--indigo) !important; }

/* Hide default alert */
.stAlert { display: none !important; }

/* Footer */
.page-footer {
    text-align: center;
    padding-top: 2rem;
    border-top: 1px solid var(--border);
    margin-top: 2.5rem;
    font-size: 12px;
    color: var(--text-muted);
}
</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL & SCALER
# =========================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_mlp_iq.pkl")
        scaler = joblib.load("scaler_iq.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        st.stop()

model, scaler = load_model()

# =========================================
# MAPPING
# =========================================

edu_map = {
    "SD / SMP": 0,
    "Kejuruan": 1,
    "SMA": 2,
    "Perguruan Tinggi": 3
}

gender_map = {"Laki-laki": 1, "Perempuan": 0}

reverse_edu_map = {
    "primary or lower secondary": 0,
    "vocational": 1,
    "secondary": 2,
    "higher": 3
}

reverse_gender_map = {"male": 1, "female": 0}

iq_labels = {
    0: "Moderate intellectual disability (35-54)",
    1: "Mild intellectual disability (55-69)",
    2: "Below average intelligence (70-84)",
    3: "Average intelligence (85-114)",
    4: "Above-average intelligence (>114)"
}

iq_ranges = {
    0: "35–54",
    1: "55–69",
    2: "70–84",
    3: "85–114",
    4: ">114"
}

iq_colors = {
    "Moderate intellectual disability (35-54)":   "#EF4444",
    "Mild intellectual disability (55-69)":       "#F97316",
    "Below average intelligence (70-84)": "#F59E0B",
    "Average intelligence (85-114)":       "#4F46E5",
    "Above-average intelligence (>114)": "#8B5CF6"
}

iq_descriptions = {
    "Moderate intellectual disability (35-54)":   "Model mengindikasikan kemungkinan adanya kebutuhan dukungan perkembangan yang signifikan.",
    "Mild intellectual disability (55-69)":       "Model mengindikasikan pola perkembangan kecerdasan ringan.",
    "Below average intelligence (70-84)": "Model mengindikasikan performa kognitif sedikit di bawah rata-rata sebaya.",
    "Average intelligence (85-114)":       "Model memprediksi perkembangan kognitif yang tipikal untuk kelompok usia tersebut.",
    "Above-average intelligence (>114)": "Model memprediksi kemampuan adaptasi kognitif tinggi berdasarkan profil sosiodemografi."
}

required_columns = ["education_mother", "education_father", "age_years", "gender"]

# =========================================
# SIDEBAR
# =========================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-mark">🧠</div>
        <div class="logo-name">RapIQ</div>
        <div class="logo-sub">Klasifikasi Kecerdasan Berbasis MLP</div>
    </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Prediksi Tunggal", "Prediksi Massal", "Tentang Model"],
        icons=["grid-1x2-fill", "table", "info-circle-fill"],
        default_index=0,
        styles={
            "container": {
                "padding": "8px 0",
                "background-color": "transparent",
            },
            "icon": {
                "color": "#9CA3AF",
                "font-size": "14px",
            },
            "nav-link": {
                "font-family": "'Plus Jakarta Sans', sans-serif",
                "font-size": "13.5px",
                "color": "#6B7280",
                "border-radius": "10px",
                "margin": "2px 10px",
                "padding": "11px 14px",
                "--hover-color": "#EEF2FF",
            },
            "nav-link-selected": {
                "background": "#EEF2FF",
                "color": "#4F46E5",
                "border": "1px solid #C7D2FE",
                "font-weight": "700",
            },
        }
    )

    st.markdown("<div style='height:1px; background:var(--border); margin:12px 0'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-status">
        <div class="sidebar-status-label">Sistem</div>
        <div class="sidebar-status-item"><span class="status-dot"></span> Model Siap Digunakan</div>
        <div class="sidebar-status-item"><span class="status-dot"></span> MLP Classifier</div>
        <div class="sidebar-status-item"><span class="status-dot"></span> StandardScaler</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================
# PLOTLY THEME
# =========================================

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#6B7280"),
    margin=dict(l=20, r=20, t=36, b=20),
    xaxis=dict(
        gridcolor="#F3F4F6",
        zerolinecolor="#E5E7EB",
        tickfont=dict(size=11, color="#9CA3AF"),
    ),
    yaxis=dict(
        gridcolor="#F3F4F6",
        zerolinecolor="#E5E7EB",
        tickfont=dict(size=11, color="#9CA3AF"),
    ),
    hoverlabel=dict(
        bgcolor="#FFFFFF",
        bordercolor="#E5E7EB",
        font=dict(family="Plus Jakarta Sans", size=12, color="#111827"),
    ),
)

# =========================================
# PAGE: DASHBOARD
# =========================================

if selected == "Prediksi Tunggal":

    # Page header — matches screenshot style
    st.markdown("""
    <div class="page-header">
        <div class="page-header-icon">🧠</div>
        <div class="page-header-title">RapIQ</div>
        <div class="page-header-sub">Klasifikasi Kecerdasan Berbasis Artificial Intelligence menggunakan Multilayer Perceptron (MLP)</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Family + Child in two columns (no card box, flat like screenshot) ──
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="section-heading">👨‍👩‍👦 Informasi Keluarga</div>', unsafe_allow_html=True)

        st.markdown('<span class="input-label">Pendidikan Ibu</span>', unsafe_allow_html=True)
        education_mother = st.selectbox("_", list(edu_map.keys()), key="edu_mother", label_visibility="collapsed")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<span class="input-label">Pendidikan Ayah</span>', unsafe_allow_html=True)
        education_father = st.selectbox("_", list(edu_map.keys()), key="edu_father", label_visibility="collapsed")

    with col_right:
        st.markdown('<div class="section-heading">👶 Informasi Anak</div>', unsafe_allow_html=True)

        st.markdown('<span class="input-label">Usia (Tahun)</span>', unsafe_allow_html=True)
        age = st.number_input("_", min_value=1, max_value=18, value=10, key="age_input", label_visibility="collapsed")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<span class="input-label">Jenis Kelamin</span>', unsafe_allow_html=True)
        gender = st.radio("_", list(gender_map.keys()), horizontal=True, key="gender_input", label_visibility="collapsed")

    # ── Predict button — full width, gradient, pill ──
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🚀  Prediksi Kategori IQ", key="predict_btn")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Prediction output ──
    if predict_btn:
        try:
            input_data = pd.DataFrame({
                "education_mother": [edu_map[education_mother]],
                "education_father": [edu_map[education_father]],
                "age_years": [age],
                "gender": [gender_map[gender]]
            })

            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)[0]
            probabilities = model.predict_proba(scaled_data)[0]
            confidence = float(np.max(probabilities) * 100)
            predicted_label = iq_labels[prediction]
            predicted_iq_range = iq_ranges[prediction]
            pred_color = iq_colors.get(predicted_label, "#4F46E5")
            pred_desc = iq_descriptions.get(predicted_label, "")

            r_col, c_col = st.columns([1.1, 3], gap="medium")

            with r_col:
                st.markdown(f"""
                <div class="prediction-result-card">
                    <div class="prediction-tag">Hasil Prediksi</div>
                    <div class="prediction-category" style="color:{pred_color}">{predicted_label}</div>
                    <div class="prediction-desc">{pred_desc}</div>
                    <div style="margin-bottom:1rem">
                        <div style="font-size:11px;color:var(--text-muted);font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">Perkiraan Rentang IQ</div>
                        <div style="font-size:1.6rem;font-weight:800;color:{pred_color};letter-spacing:-0.02em">{predicted_iq_range}</div>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                        <span style="font-size:11px;color:var(--text-muted);font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Tingkat Keyakinan Model</span>
                        <span style="font-size:14px;color:{pred_color};font-weight:800">{confidence:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(confidence / 100)

            with c_col:
                prob_df = pd.DataFrame({
                    "Category": list(iq_labels.values()),
                    "Probability": probabilities * 100,
                    "Color": [iq_colors[l] for l in iq_labels.values()]
                })

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=prob_df["Category"],
                    y=prob_df["Probability"],
                    marker=dict(
                        color=prob_df["Color"],
                        opacity=0.85,
                        line=dict(color="rgba(0,0,0,0)", width=0)
                    ),
                    text=[f"{v:.1f}%" for v in prob_df["Probability"]],
                    textposition="outside",
                    textfont=dict(size=10, color="#9CA3AF"),
                    hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
                ))
                fig.update_layout(
                    paper_bgcolor=PLOT_LAYOUT["paper_bgcolor"],
                    plot_bgcolor=PLOT_LAYOUT["plot_bgcolor"],
                    font=PLOT_LAYOUT["font"],
                    margin=PLOT_LAYOUT["margin"],
                    xaxis=dict(
                        gridcolor="#F3F4F6",
                        zerolinecolor="#E5E7EB",
                        tickfont=dict(size=10, color="#9CA3AF"),
                        tickangle=-35
                    ),
                    yaxis=dict(
                        gridcolor="#F3F4F6",
                        zerolinecolor="#E5E7EB",
                        tickfont=dict(size=11, color="#9CA3AF"),
                        range=[0, 110]
                    ),
                    hoverlabel=PLOT_LAYOUT["hoverlabel"],
                    title=dict(
                        text="Distribusi Probabilitas Prediksi",
                        font=dict(size=12, color="#9CA3AF"),
                        x=0
                    ),
                    height=500,
                    bargap=0.38,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        except Exception as e:
            st.markdown(f"""
            <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:10px;padding:12px 16px;color:#991B1B;font-size:13px">
                ⚠️ Kesalahan Prediksi: {e}
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="page-footer">© 2025 RapIQ — Sistem Klasifikasi Kecerdasan</div>
    """, unsafe_allow_html=True)

# =========================================
# PAGE: BULK PREDICTION
# =========================================

elif selected == "Prediksi Massal":

    # Page header
    st.markdown("""
    <div class="page-header">
        <div class="page-header-icon">📁</div>
        <div class="page-header-title">Prediksi Massal</div>
        <div class="page-header-sub">Unggah dataset CSV dan lakukan prediksi untuk banyak data sekaligus.</div>
    </div>
    """, unsafe_allow_html=True)

    # Download template section
    st.markdown('<div class="section-heading">📥 Unduh Template</div>', unsafe_allow_html=True)

    template_df = pd.DataFrame({
        "education_mother": ["secondary", "higher", "vocational"],
        "education_father": ["vocational", "higher", "secondary"],
        "age_years": [10, 15, 8],
        "gender": ["male", "female", "male"]
    })
    template_csv = template_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Unduh Template CSV", data=template_csv,
                       file_name="template_input_iq.csv", mime="text/csv")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Upload section
    st.markdown('<span class="input-label">Unggah File CSV</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Unggah File CSV", type=["csv"], label_visibility="collapsed")

    # Upload hint / preview
    if not uploaded_file:
        st.markdown("""
        <div class="info-box" style="margin-top:12px">
            Unggah file CSV untuk memulai prediksi massal.
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine="python", decimal=",")
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-heading">👁️ Pratinjau Dataset</div>', unsafe_allow_html=True)
            st.dataframe(df.head(5), use_container_width=True, hide_index=True)
            st.markdown(f"""
            <div style="display:flex;gap:8px;margin-top:8px">
                <span class="tag tag-blue">{len(df)} Baris</span>
                <span class="tag tag-purple">{len(df.columns)} Kolom</span>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:10px;padding:12px 16px;color:#991B1B;font-size:13px;margin-top:10px">
                Kesalahan Membaca File: {e}
            </div>
            """, unsafe_allow_html=True)

    # Run button + results
    if uploaded_file:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        run_col = st.columns([2, 1, 2])[1]
        with run_col:
            run_btn = st.button("▶  Jalankan Analisis", key="run_bulk")

        if run_btn:
            try:
                df_encoded = df.copy()
                df_encoded = df_encoded.replace(r'^\s*$', np.nan, regex=True)
                jumlah_sebelum = len(df_encoded)
                df_encoded = df_encoded.dropna(subset=required_columns)
                jumlah_sesudah = len(df_encoded)
                jumlah_terhapus = jumlah_sebelum - jumlah_sesudah

                if len(df_encoded) == 0:
                    st.error("Seluruh data dihapus karena terdapat nilai yang kosong.")
                    st.stop()

                for col in ["education_mother", "education_father"]:
                    df_encoded[col] = df_encoded[col].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
                df_encoded["gender"] = df_encoded["gender"].astype(str).str.strip().str.lower()
                df_encoded["age_years"] = pd.to_numeric(df_encoded["age_years"], errors="coerce")
                df_encoded["education_mother"] = df_encoded["education_mother"].map(reverse_edu_map)
                df_encoded["education_father"] = df_encoded["education_father"].map(reverse_edu_map)
                df_encoded["gender"] = df_encoded["gender"].map(reverse_gender_map)

                errors = []
                if df_encoded["education_mother"].isnull().any(): errors.append("education_mother")
                if df_encoded["education_father"].isnull().any(): errors.append("education_father")
                if df_encoded["gender"].isnull().any(): errors.append("gender")
                if errors:
                    st.error(f"Nilai kategori tidak valid pada: {', '.join(errors)}")
                    st.stop()

                X = df_encoded[required_columns]
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)
                confidence_scores = np.max(probabilities, axis=1) * 100
                predicted_labels = [iq_labels[p] for p in predictions]

                result_df = pd.DataFrame({
                    "education_mother": df.loc[df_encoded.index, "education_mother"],
                    "education_father": df.loc[df_encoded.index, "education_father"],
                    "age_years": df.loc[df_encoded.index, "age_years"],
                    "gender": df.loc[df_encoded.index, "gender"],
                    "predicted_iq_category": predicted_labels,
                    "confidence_score (%)": [f"{x:.2f}%" for x in confidence_scores]
                })

                avg_conf = float(confidence_scores.mean())
                cat_counts = pd.Series(predicted_labels).value_counts()
                top_cat = cat_counts.index[0] if len(cat_counts) > 0 else "—"

                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                s1, s2, s3, s4 = st.columns(4, gap="medium")
                with s1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">📊 Total Data</div>
                        <div class="metric-value">{len(result_df):,}</div>
                        <div class="metric-sub">{jumlah_terhapus} baris diabaikan</div>
                    </div>""", unsafe_allow_html=True)
                with s2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">🎯 Rata-rata Keyakinan</div>
                        <div class="metric-value" style="color:var(--indigo)">{avg_conf:.1f}%</div>
                        <div class="metric-sub">Dari seluruh prediksi</div>
                    </div>""", unsafe_allow_html=True)
                with s3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">🏷️ Kategori Terbanyak</div>
                        <div class="metric-value" style="font-size:1.25rem">{top_cat}</div>
                        <div class="metric-sub">{cat_counts.iloc[0]} data</div>
                    </div>""", unsafe_allow_html=True)
                with s4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">✅ Status</div>
                        <div class="metric-value" style="color:var(--green);font-size:1.25rem">Selesai</div>
                        <div class="metric-sub">Proses selesai</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                chart_col, res_col = st.columns([1, 1.2], gap="medium")

                with chart_col:
                    st.markdown('<div class="section-heading">📈 Distribusi Kategori IQ</div>', unsafe_allow_html=True)
                    dist_counts = pd.Series(predicted_labels).value_counts().reindex(list(iq_labels.values()), fill_value=0)
                    colors = [iq_colors.get(c, "#4F46E5") for c in dist_counts.index]

                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=dist_counts.index,
                        y=dist_counts.values,
                        marker=dict(color=colors, opacity=0.85),
                        text=dist_counts.values,
                        textposition="outside",
                        textfont=dict(size=11, color="#9CA3AF"),
                        hovertemplate="<b>%{x}</b><br>%{y} data<extra></extra>",
                    ))
                    fig2.update_layout(
                        paper_bgcolor=PLOT_LAYOUT["paper_bgcolor"],
                        plot_bgcolor=PLOT_LAYOUT["plot_bgcolor"],
                        font=PLOT_LAYOUT["font"],
                        margin=PLOT_LAYOUT["margin"],
                        xaxis=PLOT_LAYOUT["xaxis"],
                        yaxis=dict(
                            gridcolor="#F3F4F6",
                            zerolinecolor="#E5E7EB",
                            tickfont=dict(size=11, color="#9CA3AF"),
                            range=[0, 100],
                            tickmode="linear",
                            tick0=0,
                            dtick=20
                        ),
                        hoverlabel=PLOT_LAYOUT["hoverlabel"],
                        height=280,
                        bargap=0.32,
                        showlegend=False,
                    )
                    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

                with res_col:
                    st.markdown('<div class="section-heading">📋 Hasil Analisis</div>', unsafe_allow_html=True)
                    st.dataframe(result_df[["age_years", "gender", "predicted_iq_category", "confidence_score (%)"]].head(8),
                                 use_container_width=True, hide_index=True)
                    csv_out = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Unduh Hasil Lengkap", data=csv_out,
                                       file_name="prediction_result.csv", mime="text/csv")

            except Exception as e:
                st.markdown(f"""
                <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:10px;padding:12px 16px;color:#991B1B;font-size:13px">
                    ERROR: {e}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
    <div class="page-footer">© 2025 RapIQ — Sistem Klasifikasi Kecerdasan</div>
    """, unsafe_allow_html=True)

# =========================================
# PAGE: ABOUT MODEL
# =========================================

elif selected == "Tentang Model":

    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
        <span style="font-size:1.1rem">ℹ️</span>
        <span style="font-family:var(--font);font-size:1.5rem;font-weight:800;color:var(--text-primary);letter-spacing:-0.025em">Tentang RapIQ</span>
    </div>
    <div style="font-size:12.5px;color:var(--text-secondary);margin-bottom:1.6rem">
        Sistem Prediksi Klasifikasi Kecerdasan
    </div>
    <div class="section-divider"></div>
    """, unsafe_allow_html=True)

    # Overview
    st.markdown("""
    <div style="margin-bottom:1.6rem">
        <div style="font-size:1rem;font-weight:700;color:var(--text-primary);margin-bottom:10px;display:flex;align-items:center;gap:8px">
            🔍 Ringkasan
        </div>
        <div style="font-size:13.5px;color:var(--text-secondary);line-height:1.8">
            RapIQ adalah sistem klasifikasi kecerdasan yang dikembangkan menggunakan teknik Machine Learning untuk memprediksi kategori IQ anak berdasarkan karakteristik demografis dan keluarga.<br><br>
            Model prediksi didasarkan pada jaringan saraf Multilayer Perceptron (MLP) yang dilatih menggunakan variabel pendidikan dan demografis.
        </div>
    </div>
    <div class="section-divider"></div>
    """, unsafe_allow_html=True)

    # Model Summary
    st.markdown("""
    <div style="font-size:1rem;font-weight:700;color:var(--text-primary);margin-bottom:14px;display:flex;align-items:center;gap:8px">
        📊 Ringkasan Model
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Akurasi</div>
            <div class="metric-value">28.12%</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Weighted F1</div>
            <div class="metric-value">26.01%</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Ukuran Dataset</div>
            <div class="metric-value">80K+</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Waktu Prediksi</div>
            <div class="metric-value">&lt;20 ms</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Model Information — table style like screenshot
    st.markdown("""
    <div style="font-size:1rem;font-weight:700;color:var(--text-primary);margin-bottom:14px;display:flex;align-items:center;gap:8px">
        ⚙️ Informasi Model
    </div>
    <div class="rapiq-card" style="padding:0;overflow:hidden">
        <div class="info-row" style="padding:11px 16px">
            <span class="info-row-label">Algoritma</span>
            <span class="info-row-value">MLP Classifier</span>
        </div>
        <div class="info-row" style="padding:11px 16px">
            <span class="info-row-label">Hidden Layer</span>
            <span class="info-row-value">(10, 6)</span>
        </div>
        <div class="info-row" style="padding:11px 16px">
            <span class="info-row-label">Fungsi Aktivasi</span>
            <span class="info-row-value">ReLU</span>
        </div>
        <div class="info-row" style="padding:11px 16px">
            <span class="info-row-label">Optimizer</span>
            <span class="info-row-value">Adam</span>
        </div>
        <div class="info-row" style="padding:11px 16px">
            <span class="info-row-label">Normalisasi Fitur</span>
            <span class="info-row-value">StandardScaler</span>
        </div>
        <div class="info-row" style="padding:11px 16px">
            <span class="info-row-label">Penyeimbangan Data</span>
            <span class="info-row-value">SMOTE</span>
        </div>
    </div>
    <div class="section-divider"></div>
    """, unsafe_allow_html=True)

    # Input Features
    st.markdown("""
    <div style="font-size:1rem;font-weight:700;color:var(--text-primary);margin-bottom:12px;display:flex;align-items:center;gap:8px">
        🔢 Fitur Input
    </div>
    <ul style="font-size:13.5px;color:var(--text-secondary);line-height:2.1;padding-left:1.3rem;margin-bottom:0">
        <li>Pendidikan ibu</li>
        <li>Pendidikan ayah</li>
        <li>Usia (tahun)</li>
        <li>Jenis kelamin</li>
    </ul>
    <div class="section-divider"></div>
    """, unsafe_allow_html=True)

    # Output Classes
    st.markdown("""
    <div style="font-size:1rem;font-weight:700;color:var(--text-primary);margin-bottom:12px;display:flex;align-items:center;gap:8px">
        🏷️ Kelas Output
    </div>
    """, unsafe_allow_html=True)

    class_colors_list = ["#EF4444", "#F97316", "#F59E0B", "#4F46E5", "#8B5CF6"]
    for i, (key, label) in enumerate(iq_labels.items()):
        color = class_colors_list[i]
        st.markdown(f"""
        <div class="class-row">
            <div class="class-dot" style="background:{color}"></div>
            <div class="class-row-name">{label}</div>
            <div class="class-row-num">Class {key+1:02d}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Dataset
    st.markdown("""
    <div style="font-size:1rem;font-weight:700;color:var(--text-primary);margin-bottom:10px;display:flex;align-items:center;gap:8px">
        🗄️ Dataset
    </div>
    <div style="font-size:13.5px;color:var(--text-secondary);line-height:1.8;margin-bottom:4px">
        Model dilatih menggunakan dataset penilaian kecerdasan berskala besar yang berisi data latar belakang demografis dan pendidikan.
    </div>
    <div style="font-size:13.5px;color:var(--text-secondary);line-height:1.8">
        Dataset dibagi menjadi tiga bagian yang digunakan untuk pelatihan, validasi, dan pengujian.
    </div>
    <div class="section-divider"></div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div style="font-size:1rem;font-weight:700;color:var(--text-primary);margin-bottom:10px;display:flex;align-items:center;gap:8px">
        ⚠️ Disclaimer
    </div>
    <div class="disclaimer-box">
        <p>Aplikasi ini ditujukan hanya untuk keperluan edukasi, akademis, dan penelitian.</p>
        <p>Hasil prediksi yang dihasilkan oleh RapIQ tidak boleh diartikan sebagai penilaian psikologis profesional atau diagnosis klinis.</p>
        <p>Setiap keputusan penting harus ditinjau oleh para ahli atau profesional yang berkualifikasi.</p>
    </div>
    <div class="section-divider"></div>
    """, unsafe_allow_html=True)

    # Developer Information
    st.markdown("""
    <div style="font-size:1rem;font-weight:700;color:var(--text-primary);margin-bottom:10px;display:flex;align-items:center;gap:8px">
        👨‍💻 Informasi Pengembang
    </div>
    <div style="font-size:13.5px;color:var(--text-secondary);line-height:1.8">
        RapIQ dikembangkan sebagai proyek machine learning untuk penelitian klasifikasi kecerdasan menggunakan algoritma Multilayer Perceptron (MLP) dan framework deployment Streamlit.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="page-footer">© 2025 RapIQ — Sistem Klasifikasi Kecerdasan</div>
    """, unsafe_allow_html=True)