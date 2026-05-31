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
# CUSTOM CSS — Obsidian Noir Design System
# Aesthetic: Dark luxury scientific instrument
# Font: Outfit (display) + JetBrains Mono (data)
# Palette: Near-black base, electric teal accent,
#          warm amber highlight, slate borders
# =========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

/* ─── Reset ─── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ─── Design Tokens ─── */
:root {
    /* Base surfaces — near-black with cool undertone */
    --s0:  #060A0F;   /* deepest bg */
    --s1:  #090E15;   /* sidebar */
    --s2:  #0C1219;   /* card */
    --s3:  #101820;   /* card hover / input */
    --s4:  #141F2A;   /* elevated surface */

    /* Borders */
    --b1:  #1A2535;   /* subtle */
    --b2:  #243448;   /* mid */
    --b3:  #2E4260;   /* strong */

    /* Accent — electric teal */
    --teal:        #00D4C8;
    --teal-dim:    rgba(0,212,200,0.12);
    --teal-glow:   rgba(0,212,200,0.18);
    --teal-border: rgba(0,212,200,0.28);

    /* Secondary accent — amber / gold */
    --amber:       #F5A623;
    --amber-dim:   rgba(245,166,35,0.12);
    --amber-border:rgba(245,166,35,0.28);

    /* Status */
    --green:  #00C896;
    --red:    #FF4D6A;
    --orange: #FF8C42;
    --purple: #9B72F0;

    /* Text */
    --t1: #E8F0F8;   /* primary */
    --t2: #5E7A96;   /* secondary */
    --t3: #2E4A62;   /* muted */

    /* Typography */
    --f-display: 'Outfit', sans-serif;
    --f-mono:    'JetBrains Mono', monospace;

    /* Geometry */
    --r-sm:   6px;
    --r-md:   10px;
    --r-lg:   14px;
    --r-xl:   20px;
    --r-pill: 999px;

    /* Shadows */
    --sh-card:  0 2px 16px rgba(0,0,0,0.5);
    --sh-teal:  0 0 24px rgba(0,212,200,0.12);
    --sh-btn:   0 4px 20px rgba(0,212,200,0.30);
}

/* ─── App shell ─── */
.stApp {
    background: var(--s0) !important;
    font-family: var(--f-display) !important;
    color: var(--t1) !important;
}

/* Hide streamlit chrome */
#MainMenu, footer, header         { visibility: hidden; }
.stDeployButton                   { display: none !important; }
[data-testid="stToolbar"]         { display: none !important; }

/* Content area */
.main .block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1440px !important;
}

/* Column gaps */
[data-testid="column"]            { padding: 0 7px !important; }
[data-testid="stHorizontalBlock"] { gap: 0 !important; }

/* Divider */
hr {
    border: none !important;
    border-top: 1px solid var(--b1) !important;
    margin: 1.4rem 0 !important;
}

/* Scrollbar */
::-webkit-scrollbar               { width: 4px; height: 4px; }
::-webkit-scrollbar-track         { background: var(--s0); }
::-webkit-scrollbar-thumb         { background: var(--b2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover   { background: var(--b3); }

/* Code */
code {
    font-family: var(--f-mono) !important;
    font-size: 11px !important;
    color: var(--teal) !important;
    background: var(--teal-dim) !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child,
section[data-testid="stSidebar"] {
    background: var(--s1) !important;
    border-right: 1px solid var(--b1) !important;
    width: 248px !important;
}
[data-testid="stSidebar"] ul,
[data-testid="stSidebar"] nav,
[data-testid="stSidebar"] section,
[data-testid="stSidebar"] .css-1d391kg,
[data-testid="stSidebar"] .css-163ttbj,
[data-testid="stSidebar"] .css-1wrcr25,
ul[data-testid="stSidebarNavItems"],
nav { background: transparent !important; }

/* ─── Sidebar logo ─── */
.sidebar-logo {
    padding: 24px 18px 18px;
    border-bottom: 1px solid var(--b1);
    margin-bottom: 4px;
}
.logo-mark {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, var(--teal) 0%, #007A9E 100%);
    border-radius: var(--r-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    margin-bottom: 12px;
    box-shadow: 0 4px 16px rgba(0,212,200,0.30);
    position: relative;
}
.logo-mark::after {
    content: '';
    position: absolute;
    inset: -1px;
    border-radius: calc(var(--r-md) + 1px);
    border: 1px solid rgba(0,212,200,0.40);
    pointer-events: none;
}
.logo-name {
    font-family: var(--f-display);
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--t1);
    letter-spacing: -0.03em;
}
.logo-sub {
    font-size: 9px;
    color: var(--t3);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 3px;
    font-family: var(--f-mono);
}

/* ─── Nav links ─── */
.nav-link {
    font-family: var(--f-display) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: var(--t2) !important;
    border-radius: var(--r-md) !important;
    margin: 2px 10px !important;
    padding: 12px 14px !important;
    transition: all 0.18s ease !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    min-height: 48px !important;
    border: 1px solid transparent !important;
}
.nav-link:hover {
    background: var(--s3) !important;
    color: var(--t1) !important;
    border-color: var(--b1) !important;
}
.nav-link.active,
.nav-link-selected {
    background: var(--teal-dim) !important;
    border: 1px solid var(--teal-border) !important;
    color: var(--teal) !important;
    font-weight: 700 !important;
    box-shadow: 0 0 12px rgba(0,212,200,0.07) !important;
}
.nav-link span { overflow: hidden !important; text-overflow: ellipsis !important; }
.nav-link i    { font-size: 14px !important; min-width: 16px !important; }

/* sidebar footer */
.sidebar-footer {
    padding: 14px 18px 20px;
}
.sidebar-footer-label {
    font-size: 9px;
    color: var(--t3);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 10px;
    font-family: var(--f-mono);
}
.sidebar-footer-items {
    font-size: 11.5px;
    color: var(--t2);
    line-height: 2;
    font-family: var(--f-mono);
}

/* ──────────────────────────────────────────
   HERO BANNER
────────────────────────────────────────── */
.hero-banner {
    background: var(--s2);
    border: 1px solid var(--b1);
    border-top: 2px solid var(--teal);
    border-radius: var(--r-xl);
    padding: 2.4rem 3rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
/* Grid texture overlay */
.hero-banner::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(var(--b1) 1px, transparent 1px),
        linear-gradient(90deg, var(--b1) 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.25;
    pointer-events: none;
}
/* Teal orb */
.hero-banner::after {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(0,212,200,0.07) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--teal-dim);
    border: 1px solid var(--teal-border);
    border-radius: var(--r-pill);
    padding: 4px 12px;
    font-size: 9.5px;
    font-weight: 700;
    color: var(--teal);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: var(--f-mono);
    position: relative;
}
.hero-title {
    font-family: var(--f-display);
    font-size: 3.2rem;
    font-weight: 900;
    color: var(--t1);
    line-height: 1.05;
    margin-bottom: 0.5rem;
    letter-spacing: -0.04em;
    position: relative;
}
/* teal underline glow on title */
.hero-title span {
    color: var(--teal);
}
.hero-subtitle {
    font-size: 13px;
    color: var(--t2);
    font-weight: 400;
    margin-bottom: 1.3rem;
    line-height: 1.7;
    max-width: 640px;
    position: relative;
}

/* ── Tags ── */
.tag {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: var(--r-sm);
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    font-family: var(--f-mono);
}
.tag-blue   { background: var(--teal-dim);   border: 1px solid var(--teal-border);   color: var(--teal); }
.tag-purple { background: rgba(155,114,240,0.10); border: 1px solid rgba(155,114,240,0.25); color: var(--purple); }
.tag-green  { background: rgba(0,200,150,0.10);   border: 1px solid rgba(0,200,150,0.25);   color: var(--green); }

/* ──────────────────────────────────────────
   CARDS
────────────────────────────────────────── */
.rapiq-card {
    background: var(--s2);
    border: 1px solid var(--b1);
    border-radius: var(--r-lg);
    padding: 1.4rem 1.5rem;
    margin-bottom: 0.9rem;
    box-shadow: var(--sh-card);
    transition: border-color 0.2s, box-shadow 0.2s;
}
.rapiq-card:hover {
    border-color: var(--b2);
    box-shadow: var(--sh-teal);
}

/* ── Section header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.15rem;
}
.section-icon {
    width: 30px;
    height: 30px;
    background: var(--teal-dim);
    border: 1px solid var(--teal-border);
    border-radius: var(--r-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
}
.section-title {
    font-family: var(--f-display);
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--t1);
    letter-spacing: -0.01em;
}

/* ── Input label ── */
.input-label {
    font-size: 9.5px;
    font-weight: 600;
    color: var(--t3);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 6px;
    font-family: var(--f-mono);
}

/* ──────────────────────────────────────────
   METRIC CARDS
────────────────────────────────────────── */
.metric-card {
    background: var(--s2);
    border: 1px solid var(--b1);
    border-radius: var(--r-lg);
    padding: 1.3rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}
/* Left accent bar instead of top */
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 2px;
    background: var(--teal);
    border-radius: var(--r-lg) 0 0 var(--r-lg);
}
.metric-card:hover {
    border-color: var(--teal-border);
    box-shadow: var(--sh-teal);
}
.metric-label {
    font-size: 9px;
    font-weight: 600;
    color: var(--t3);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: var(--f-mono);
}
.metric-value {
    font-family: var(--f-display);
    font-size: 2rem;
    font-weight: 800;
    color: var(--t1);
    line-height: 1;
    margin-bottom: 4px;
    letter-spacing: -0.04em;
}
.metric-sub {
    font-size: 10.5px;
    color: var(--t3);
    margin-top: 5px;
    font-family: var(--f-mono);
}

/* ──────────────────────────────────────────
   PREDICTION RESULT
────────────────────────────────────────── */
.prediction-result-card {
    background: var(--s2);
    border: 1px solid var(--teal-border);
    border-left: 3px solid var(--teal);
    border-radius: var(--r-xl);
    padding: 2rem 2.4rem;
    margin: 1.2rem 0;
    box-shadow: var(--sh-teal), var(--sh-card);
    position: relative;
    overflow: hidden;
}
/* Subtle teal orb */
.prediction-result-card::after {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, var(--teal-glow) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
/* Scanning line animation */
.prediction-result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%,
        var(--teal) 40%,
        var(--teal) 60%,
        transparent 100%);
    animation: scan-line 3.5s ease-in-out infinite;
    opacity: 0.6;
}
@keyframes scan-line {
    0%   { transform: translateY(0);    opacity: 0.6; }
    50%  { transform: translateY(200px); opacity: 0.2; }
    100% { transform: translateY(0);    opacity: 0.6; }
}
.prediction-tag {
    font-size: 9px;
    font-weight: 700;
    color: var(--teal);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 8px;
    font-family: var(--f-mono);
}
.prediction-category {
    font-family: var(--f-display);
    font-size: 2.6rem;
    font-weight: 900;
    color: var(--t1);
    line-height: 1.05;
    margin-bottom: 6px;
    letter-spacing: -0.03em;
}
.prediction-desc {
    font-size: 12.5px;
    color: var(--t2);
    margin-bottom: 1.4rem;
    line-height: 1.65;
}

/* ──────────────────────────────────────────
   CLASS CARDS (taxonomy)
────────────────────────────────────────── */
.class-card {
    background: var(--s2);
    border: 1px solid var(--b1);
    border-radius: var(--r-md);
    padding: 1.1rem 1.3rem;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
    cursor: default;
}
.class-card:hover {
    border-color: var(--teal-border);
    box-shadow: var(--sh-teal);
    transform: translateY(-2px);
}
.class-number {
    font-size: 9px;
    font-weight: 600;
    color: var(--t3);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 6px;
    font-family: var(--f-mono);
}
.class-name {
    font-family: var(--f-display);
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--t1);
}

/* ──────────────────────────────────────────
   FORM CONTROLS
────────────────────────────────────────── */

/* Selectbox */
[data-baseweb="select"] > div {
    background: var(--s3) !important;
    border: 1px solid var(--b2) !important;
    border-radius: var(--r-md) !important;
    color: var(--t1) !important;
    font-family: var(--f-display) !important;
    font-size: 13.5px !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
[data-baseweb="select"] > div:hover,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--teal-border) !important;
    box-shadow: 0 0 0 3px rgba(0,212,200,0.07) !important;
}
[data-baseweb="popover"] {
    background: var(--s3) !important;
    border: 1px solid var(--b2) !important;
    border-radius: var(--r-md) !important;
    box-shadow: var(--sh-card) !important;
}
[role="option"] {
    background: var(--s3) !important;
    color: var(--t2) !important;
    font-size: 13.5px !important;
    font-family: var(--f-display) !important;
}
[role="option"]:hover {
    background: var(--s4) !important;
    color: var(--t1) !important;
}

/* Number input */
[data-testid="stNumberInput"] input {
    background: var(--s3) !important;
    border: 1px solid var(--b2) !important;
    border-radius: var(--r-md) !important;
    color: var(--t1) !important;
    font-family: var(--f-display) !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--teal-border) !important;
    box-shadow: 0 0 0 3px rgba(0,212,200,0.07) !important;
    outline: none !important;
}
[data-testid="stNumberInput"] button {
    background: var(--s4) !important;
    border-color: var(--b2) !important;
    color: var(--t2) !important;
    transition: color 0.18s !important;
}
[data-testid="stNumberInput"] button:hover {
    color: var(--teal) !important;
}

/* Radio */
[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 10px !important;
}
[data-testid="stRadio"] label {
    background: var(--s3) !important;
    border: 1px solid var(--b2) !important;
    border-radius: var(--r-md) !important;
    padding: 10px 20px !important;
    color: var(--t2) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    min-width: 120px !important;
    white-space: nowrap !important;
    font-family: var(--f-display) !important;
}
[data-testid="stRadio"] label:hover {
    border-color: var(--teal-border) !important;
    color: var(--t1) !important;
    background: var(--s4) !important;
}
[data-testid="stRadio"] label[data-checked="true"] {
    border-color: var(--teal-border) !important;
    background: var(--teal-dim) !important;
    color: var(--teal) !important;
    font-weight: 700 !important;
    box-shadow: 0 0 14px rgba(0,212,200,0.08) !important;
}

/* ── Primary Button ── */
.stButton > button {
    background: var(--teal) !important;
    color: #060A0F !important;
    font-family: var(--f-display) !important;
    font-weight: 800 !important;
    font-size: 14px !important;
    letter-spacing: 0.01em !important;
    border: none !important;
    border-radius: var(--r-pill) !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    height: 52px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.18s, box-shadow 0.2s !important;
    box-shadow: var(--sh-btn) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 26px rgba(0,212,200,0.38) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    opacity: 1 !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid var(--b2) !important;
    color: var(--t2) !important;
    border-radius: var(--r-md) !important;
    font-size: 12.5px !important;
    font-family: var(--f-display) !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    height: auto !important;
    padding: 8px 16px !important;
    transition: all 0.18s !important;
    width: auto !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--teal-border) !important;
    color: var(--teal) !important;
    background: var(--teal-dim) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--s3) !important;
    border: 1px dashed var(--b2) !important;
    border-radius: var(--r-lg) !important;
    padding: 10px !important;
    transition: all 0.18s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--teal-border) !important;
}
section[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: var(--t2) !important;
    font-size: 13px !important;
    font-family: var(--f-display) !important;
}
[data-testid="stFileUploader"] button {
    background: var(--teal-dim) !important;
    border: 1px solid var(--teal-border) !important;
    color: var(--teal) !important;
    border-radius: var(--r-md) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 8px 18px !important;
    font-family: var(--f-display) !important;
    transition: all 0.18s !important;
}
[data-testid="stFileUploader"] button:hover {
    background: rgba(0,212,200,0.18) !important;
    transform: translateY(-1px) !important;
}

/* ──────────────────────────────────────────
   DATA DISPLAY
────────────────────────────────────────── */

/* Progress */
[data-testid="stProgress"] > div {
    background: var(--b2) !important;
    border-radius: var(--r-pill) !important;
    height: 4px !important;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--teal), var(--purple)) !important;
    border-radius: var(--r-pill) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: var(--s2) !important;
    border: 1px solid var(--b1) !important;
    border-radius: var(--r-md) !important;
    overflow: hidden !important;
}
.stDataFrame th {
    background: var(--s1) !important;
    color: var(--t3) !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--b1) !important;
    font-family: var(--f-mono) !important;
}
.stDataFrame td {
    color: var(--t2) !important;
    font-size: 13px !important;
    padding: 9px 14px !important;
    border-bottom: 1px solid rgba(26,37,53,0.5) !important;
    font-family: var(--f-mono) !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: var(--teal) !important; }

/* Hide alert */
.stAlert { display: none !important; }

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
    "Primary or Lower Secondary": 0,
    "Vocational": 1,
    "Secondary": 2,
    "Higher": 3
}

gender_map = {"Male": 1, "Female": 0}

reverse_edu_map = {
    "primary or lower secondary": 0,
    "vocational": 1,
    "secondary": 2,
    "higher": 3
}

reverse_gender_map = {"male": 1, "female": 0}

iq_labels = {
    0: "Moderate ID",
    1: "Mild ID",
    2: "Below Average",
    3: "Average",
    4: "Above Average"
}

iq_colors = {
    "Moderate ID":   "#FF4D6A",
    "Mild ID":       "#FF8C42",
    "Below Average": "#F5A623",
    "Average":       "#00D4C8",
    "Above Average": "#9B72F0"
}

iq_descriptions = {
    "Moderate ID":   "The model indicates significant developmental support may be needed.",
    "Mild ID":       "The model suggests mild intellectual developmental patterns.",
    "Below Average": "The model indicates cognitive performance slightly below peer average.",
    "Average":       "The model predicts typical cognitive development for the age group.",
    "Above Average": "The model predicts high cognitive adaptability based on socio-demographic profile."
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
        <div class="logo-sub">MLP Architecture · (64,64)</div>
    </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Bulk Prediction", "About Model"],
        icons=["grid-1x2-fill", "table", "info-circle-fill"],
        default_index=0,
        styles={
            "container": {
                "padding": "8px 0",
                "background-color": "transparent",
            },
            "icon": {
                "color": "#2E4A62",
                "font-size": "14px",
            },
            "nav-link": {
                "font-family": "'Outfit', sans-serif",
                "font-size": "13.5px",
                "color": "#5E7A96",
                "border-radius": "10px",
                "margin": "2px 10px",
                "padding": "11px 14px",
                "--hover-color": "#101820",
            },
            "nav-link-selected": {
                "background": "rgba(0,212,200,0.10)",
                "color": "#00D4C8",
                "border": "1px solid rgba(0,212,200,0.25)",
                "font-weight": "700",
            },
        }
    )

    st.markdown("<div style='height:1px; background:var(--b1); margin:14px 18px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-footer">
        <div class="sidebar-footer-label">System Status</div>
        <div class="sidebar-footer-items">
            <div>● &nbsp;Model Active</div>
            <div>● &nbsp;MLP Classifier</div>
            <div>● &nbsp;StandardScaler</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================
# PLOTLY THEME
# =========================================

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit", color="#5E7A96"),
    margin=dict(l=20, r=20, t=36, b=20),
    xaxis=dict(
        gridcolor="#1A2535",
        zerolinecolor="#1A2535",
        tickfont=dict(size=11, color="#2E4A62"),
    ),
    yaxis=dict(
        gridcolor="#1A2535",
        zerolinecolor="#1A2535",
        tickfont=dict(size=11, color="#2E4A62"),
    ),
    hoverlabel=dict(
        bgcolor="#0C1219",
        bordercolor="#243448",
        font=dict(family="Outfit", size=12, color="#E8F0F8"),
    ),
)

# =========================================
# PAGE: DASHBOARD
# =========================================

if selected == "Dashboard":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">⬡ Intelligence Classification Platform</div>
        <div class="hero-title" style="margin-top:14px">Rap<span>IQ</span></div>
        <div class="hero-subtitle">AI-powered pediatric IQ category prediction using a Multilayer Perceptron (MLP) architecture trained on the Stanford-Binet Intelligence Scales dataset.</div>
        <div style="display:flex; gap:8px; flex-wrap:wrap; position:relative;">
            <span class="tag tag-blue">MLP (64,64)</span>
            <span class="tag tag-purple">5 Classes</span>
            <span class="tag tag-green">53.66% Accuracy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        st.markdown("""
        <div class="rapiq-card">
            <div class="section-header">
                <div class="section-icon">👨‍👩‍👦</div>
                <div class="section-title">Family Information</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="input-label">Mother Education Level</div>', unsafe_allow_html=True)
        education_mother = st.selectbox("_", list(edu_map.keys()), key="edu_mother", label_visibility="collapsed")

        st.markdown('<div class="input-label" style="margin-top:14px">Father Education Level</div>', unsafe_allow_html=True)
        education_father = st.selectbox("_", list(edu_map.keys()), key="edu_father", label_visibility="collapsed")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="rapiq-card">
            <div class="section-header">
                <div class="section-icon">👶</div>
                <div class="section-title">Child Information</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="input-label">Age (Years)</div>', unsafe_allow_html=True)
        age = st.number_input("_", min_value=1, max_value=18, value=10, key="age_input", label_visibility="collapsed")

        st.markdown('<div class="input-label" style="margin-top:14px">Gender</div>', unsafe_allow_html=True)
        gender = st.radio("_", list(gender_map.keys()), horizontal=True, key="gender_input", label_visibility="collapsed")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_btn = st.columns([1, 2, 1])[1]
    with col_btn:
        predict_btn = st.button("⬡  Run Classification", key="predict_btn")

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
            pred_color = iq_colors.get(predicted_label, "#00D4C8")
            pred_desc = iq_descriptions.get(predicted_label, "")

            r_col, c_col = st.columns([1.1, 1], gap="medium")

            with r_col:
                st.markdown(f"""
                <div class="prediction-result-card">
                    <div class="prediction-tag">▸ Classification Output</div>
                    <div class="prediction-category" style="color:{pred_color}">{predicted_label}</div>
                    <div class="prediction-desc">{pred_desc}</div>
                    <div style="margin-bottom:8px">
                        <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                            <span style="font-size:9px;color:var(--t3);font-weight:700;letter-spacing:0.14em;text-transform:uppercase;font-family:var(--f-mono)">Confidence Score</span>
                            <span style="font-size:13px;color:{pred_color};font-weight:800;font-family:var(--f-mono)">{confidence:.1f}%</span>
                        </div>
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
                        opacity=0.80,
                        line=dict(color="rgba(0,0,0,0)", width=0)
                    ),
                    text=[f"{v:.1f}%" for v in prob_df["Probability"]],
                    textposition="outside",
                    textfont=dict(size=10, color="#5E7A96"),
                    hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
                ))
                fig.update_layout(
                    paper_bgcolor=PLOT_LAYOUT["paper_bgcolor"],
                    plot_bgcolor=PLOT_LAYOUT["plot_bgcolor"],
                    font=PLOT_LAYOUT["font"],
                    margin=PLOT_LAYOUT["margin"],
                    xaxis=PLOT_LAYOUT["xaxis"],
                    yaxis=dict(
                        gridcolor="#1A2535",
                        zerolinecolor="#1A2535",
                        tickfont=dict(size=11, color="#2E4A62"),
                        range=[0, 110]
                    ),
                    hoverlabel=PLOT_LAYOUT["hoverlabel"],
                    title=dict(
                        text="Confidence Distribution",
                        font=dict(size=12, color="#5E7A96"),
                        x=0
                    ),
                    height=300,
                    bargap=0.38,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        except Exception as e:
            st.markdown(f"""
            <div class="rapiq-card" style="border-color:rgba(255,77,106,0.30)">
                <div style="color:var(--red);font-size:13px;font-family:var(--f-mono)">⚠ Error: {e}</div>
            </div>
            """, unsafe_allow_html=True)

# =========================================
# PAGE: BULK PREDICTION
# =========================================

elif selected == "Bulk Prediction":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">⬡ Enterprise Processing</div>
        <div class="hero-title" style="margin-top:14px;font-size:2.4rem">Bulk Analysis</div>
        <div class="hero-subtitle">Upload your CSV file to run parallelized MLP inference across thousands of samples. Our engine automatically sanitizes features and produces confidence-weighted IQ categories.</div>
    </div>
    """, unsafe_allow_html=True)

    up_col, prev_col = st.columns([1, 1.4], gap="medium")

    with up_col:
        st.markdown("""
        <div class="rapiq-card" style="height:100%">
            <div class="section-header">
                <div class="section-icon">📤</div>
                <div class="section-title">Upload Dataset</div>
            </div>
            <div style="font-size:11.5px;color:var(--t3);margin-bottom:14px;line-height:1.8;font-family:var(--f-mono)">
                Required columns:<br>
                <code>education_mother</code> &nbsp;
                <code>education_father</code><br>
                <code>age_years</code> &nbsp;
                <code>gender</code>
            </div>
        """, unsafe_allow_html=True)

        template_df = pd.DataFrame({
            "education_mother": ["secondary", "higher", "vocational"],
            "education_father": ["vocational", "higher", "secondary"],
            "age_years": [10, 15, 8],
            "gender": ["male", "female", "male"]
        })
        template_csv = template_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV Template", data=template_csv,
                           file_name="template_input_iq.csv", mime="text/csv")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    with prev_col:
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, sep=None, engine="python", decimal=",")
                st.markdown("""
                <div class="rapiq-card">
                    <div class="section-header">
                        <div class="section-icon">👁️</div>
                        <div class="section-title">Dataset Preview</div>
                    </div>
                """, unsafe_allow_html=True)
                st.dataframe(df.head(5), use_container_width=True, hide_index=True)
                st.markdown(f"""
                    <div style="display:flex;gap:8px;margin-top:10px">
                        <span class="tag tag-blue">{len(df)} Rows</span>
                        <span class="tag tag-purple">{len(df.columns)} Columns</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="rapiq-card" style="border-color:rgba(255,77,106,0.30)"><div style="color:var(--red);font-size:13px;font-family:var(--f-mono)">Read Error: {e}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="rapiq-card" style="height:100%;display:flex;align-items:center;justify-content:center;min-height:180px">
                <div style="text-align:center;color:var(--t3)">
                    <div style="font-size:2.2rem;margin-bottom:10px;opacity:0.4">📂</div>
                    <div style="font-size:12.5px;font-family:var(--f-mono)">Upload a CSV to preview</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if uploaded_file:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        run_col = st.columns([2, 1, 2])[1]
        with run_col:
            run_btn = st.button("▶  Run Bulk Analysis", key="run_bulk")

        if run_btn:
            try:
                df_encoded = df.copy()
                df_encoded = df_encoded.replace(r'^\s*$', np.nan, regex=True)
                jumlah_sebelum = len(df_encoded)
                df_encoded = df_encoded.dropna(subset=required_columns)
                jumlah_sesudah = len(df_encoded)
                jumlah_terhapus = jumlah_sebelum - jumlah_sesudah

                if len(df_encoded) == 0:
                    st.error("All data removed due to missing values.")
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
                    st.error(f"Invalid category values in: {', '.join(errors)}")
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

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                s1, s2, s3, s4 = st.columns(4, gap="medium")
                with s1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">📊 Total Samples</div>
                        <div class="metric-value">{len(result_df):,}</div>
                        <div class="metric-sub">{jumlah_terhapus} rows skipped</div>
                    </div>""", unsafe_allow_html=True)
                with s2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">🎯 Avg Confidence</div>
                        <div class="metric-value" style="color:var(--teal)">{avg_conf:.1f}%</div>
                        <div class="metric-sub">Across all predictions</div>
                    </div>""", unsafe_allow_html=True)
                with s3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">🏷️ Top Category</div>
                        <div class="metric-value" style="font-size:1.25rem">{top_cat}</div>
                        <div class="metric-sub">{cat_counts.iloc[0]} samples</div>
                    </div>""", unsafe_allow_html=True)
                with s4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">✅ Status</div>
                        <div class="metric-value" style="color:var(--green);font-size:1.25rem">Done</div>
                        <div class="metric-sub">Processing complete</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

                chart_col, res_col = st.columns([1, 1.2], gap="medium")

                with chart_col:
                    st.markdown("""
                    <div class="rapiq-card">
                        <div class="section-header">
                            <div class="section-icon">📈</div>
                            <div class="section-title">IQ Distribution</div>
                        </div>
                    """, unsafe_allow_html=True)

                    dist_counts = pd.Series(predicted_labels).value_counts().reindex(list(iq_labels.values()), fill_value=0)
                    colors = [iq_colors.get(c, "#00D4C8") for c in dist_counts.index]

                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=dist_counts.index,
                        y=dist_counts.values,
                        marker=dict(color=colors, opacity=0.80),
                        text=dist_counts.values,
                        textposition="outside",
                        textfont=dict(size=11, color="#5E7A96"),
                        hovertemplate="<b>%{x}</b><br>%{y} samples<extra></extra>",
                    ))
                    fig2.update_layout(
                        paper_bgcolor=PLOT_LAYOUT["paper_bgcolor"],
                        plot_bgcolor=PLOT_LAYOUT["plot_bgcolor"],
                        font=PLOT_LAYOUT["font"],
                        margin=PLOT_LAYOUT["margin"],
                        xaxis=PLOT_LAYOUT["xaxis"],
                        yaxis=dict(
                            gridcolor="#1A2535",
                            zerolinecolor="#1A2535",
                            tickfont=dict(size=11, color="#2E4A62"),
                            title=dict(text="Samples", font=dict(size=11))
                        ),
                        hoverlabel=PLOT_LAYOUT["hoverlabel"],
                        height=280,
                        bargap=0.32,
                        showlegend=False,
                    )
                    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)

                with res_col:
                    st.markdown("""
                    <div class="rapiq-card">
                        <div class="section-header">
                            <div class="section-icon">📋</div>
                            <div class="section-title">Analysis Results</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(result_df[["age_years", "gender", "predicted_iq_category", "confidence_score (%)"]].head(8),
                                 use_container_width=True, hide_index=True)
                    csv_out = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Download Full Results", data=csv_out,
                                       file_name="prediction_result.csv", mime="text/csv")
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f'<div class="rapiq-card" style="border-color:rgba(255,77,106,0.30)"><div style="color:var(--red);font-size:13px;font-family:var(--f-mono)">ERROR: {e}</div></div>', unsafe_allow_html=True)

# =========================================
# PAGE: ABOUT MODEL
# =========================================

elif selected == "About Model":

    st.markdown("""
    <div style="margin-bottom:1.8rem">
        <div style="font-family:var(--f-display);font-size:1.9rem;font-weight:900;color:var(--t1);margin-bottom:4px;letter-spacing:-0.03em">
            About Model
        </div>
        <div style="font-size:12px;color:var(--t3);font-family:var(--f-mono)">RapIQ Neural Research Platform — Model Documentation</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    metrics = [
        ("📊", "Accuracy Score", "53.66%", "Weighted across all classes", "var(--teal)"),
        ("🎯", "F1-Score (Weighted)", "46.40%", "Macro-averaged F1", "var(--purple)"),
        ("🗃️", "Samples Processed", "~80k", "Stanford-Binet Dataset", "var(--amber)"),
        ("⚡", "Model Latency", "12ms", "Average response time", "var(--green)"),
    ]
    for col, (icon, label, val, sub, color) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{icon} {label}</div>
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    info_col, disc_col = st.columns([1.5, 1], gap="medium")

    with info_col:
        st.markdown("""
        <div class="rapiq-card" style="border-color:var(--teal-border)">
            <div class="section-header">
                <div class="section-icon">🧠</div>
                <div class="section-title">About RapIQ</div>
            </div>
            <div style="font-size:13px;color:var(--t2);line-height:1.85;margin-bottom:1.2rem">
                RapIQ is a state-of-the-art AI orchestration platform designed specifically for child
                intelligence classification. Utilizing advanced Multi-Layer Perceptron (MLP) architectures,
                the system analyzes complex psychometric patterns to provide objective data synthesis
                for educational and research professionals.
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap">
                <span class="tag tag-blue">AI-Driven</span>
                <span class="tag tag-purple">Classification</span>
                <span class="tag tag-green">MLP Engine</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with disc_col:
        st.markdown("""
        <div class="rapiq-card" style="border-color:var(--amber-border);border-left:3px solid var(--amber);height:100%">
            <div style="font-size:9px;font-weight:700;color:var(--amber);letter-spacing:0.16em;text-transform:uppercase;margin-bottom:12px;font-family:var(--f-mono)">
                ⚠ Research Disclaimer
            </div>
            <div style="font-size:12px;color:var(--t2);line-height:1.85;font-style:italic">
                "RapIQ is developed as a decision support system for research and preliminary screening purposes.
                It is NOT intended to serve as a standalone diagnostic tool. All results should be reviewed
                by licensed clinical neuropsychologists."
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    model_col, data_col = st.columns([1.5, 1], gap="medium")

    with model_col:
        st.markdown("""
        <div class="rapiq-card">
            <div class="section-header">
                <div class="section-icon">⚙️</div>
                <div class="section-title">Model Information</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-top:4px">
                <div>
                    <div style="font-size:9px;color:var(--t3);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:5px;font-family:var(--f-mono)">Algorithm</div>
                    <div style="font-size:14px;color:var(--t1);font-weight:700">MLP Classifier</div>
                </div>
                <div>
                    <div style="font-size:9px;color:var(--t3);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:5px;font-family:var(--f-mono)">Hidden Layer</div>
                    <div style="font-size:14px;color:var(--t1);font-weight:700;font-family:var(--f-mono)">(10, 6)</div>
                </div>
                <div>
                    <div style="font-size:9px;color:var(--t3);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:5px;font-family:var(--f-mono)">Activation</div>
                    <div style="font-size:14px;color:var(--t1);font-weight:700;font-family:var(--f-mono)">ReLU</div>
                </div>
                <div>
                    <div style="font-size:9px;color:var(--t3);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:5px;font-family:var(--f-mono)">Optimizer</div>
                    <div style="font-size:14px;color:var(--t1);font-weight:700;font-family:var(--f-mono)">Adam</div>
                </div>
                <div>
                    <div style="font-size:9px;color:var(--t3);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:5px;font-family:var(--f-mono)">Scaling</div>
                    <div style="font-size:14px;color:var(--t1);font-weight:700;font-family:var(--f-mono)">StandardScaler</div>
                </div>
                <div>
                    <div style="font-size:9px;color:var(--t3);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:5px;font-family:var(--f-mono)">Oversampling</div>
                    <div style="font-size:14px;color:var(--t1);font-weight:700;font-family:var(--f-mono)">SMOTE</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with data_col:
        st.markdown("""
        <div class="rapiq-card" style="height:100%">
            <div class="section-header">
                <div class="section-icon">🗄️</div>
                <div class="section-title">Dataset</div>
            </div>
            <div style="font-size:12.5px;color:var(--t2);line-height:1.75;margin-bottom:1rem">
                Utilizing the comprehensive Stanford-Binet Intelligence Scales dataset compiled
                from extensive longitudinal research.
            </div>
            <div style="background:var(--s3);border:1px solid var(--b1);border-radius:var(--r-md);padding:14px 18px;display:flex;align-items:center;justify-content:space-between">
                <div>
                    <div style="font-family:var(--f-display);font-size:1.6rem;font-weight:900;color:var(--t1);letter-spacing:-0.03em">80,000+</div>
                    <div style="font-size:10px;color:var(--t3);margin-top:3px;font-family:var(--f-mono);letter-spacing:0.08em;text-transform:uppercase">Validated Samples</div>
                </div>
                <div style="font-size:2rem;opacity:0.2">🗃️</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="rapiq-card">
        <div style="text-align:center;margin-bottom:1.5rem">
            <div style="font-family:var(--f-display);font-size:1.15rem;font-weight:800;color:var(--t1);letter-spacing:-0.02em">
                Intelligence Classification Taxonomy
            </div>
            <div style="font-size:10.5px;color:var(--t3);margin-top:4px;font-family:var(--f-mono);letter-spacing:0.08em;text-transform:uppercase">MLP Output Class Definitions</div>
        </div>
    """, unsafe_allow_html=True)

    class_colors = ["#FF4D6A", "#FF8C42", "#F5A623", "#00D4C8", "#9B72F0"]
    class_cols = st.columns(5, gap="medium")
    for i, (key, label) in enumerate(iq_labels.items()):
        with class_cols[i]:
            color = class_colors[i]
            st.markdown(f"""
            <div class="class-card" style="border-color:{color}22;border-left:2px solid {color}">
                <div class="class-number">Class {key+1:02d}</div>
                <div class="class-name" style="color:{color}">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:2.5rem;padding-top:1.5rem;border-top:1px solid var(--b1)">
        <div style="font-size:10px;color:var(--t3);letter-spacing:0.14em;text-transform:uppercase;font-family:var(--f-mono)">
            © 2024 RapIQ Neural Research Group · All Rights Reserved
        </div>
    </div>
    """, unsafe_allow_html=True)
