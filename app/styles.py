"""Scientific theme styling, color tokens, and custom CSS for HiggsLens dashboard.

Implements a cohesive dark graphite and navy palette with restrained cyan and
electric blue accents, crisp typography, Material Icons, and scientific data tables.
Material Icons are loaded from Google Fonts CDN and used via <span class="material-icons">...</span>.
"""

from __future__ import annotations

# Color Palette Tokens
SURFACE_BASE = "#0a0e17"        # Deep dark navy-graphite background
SURFACE_CARD = "#111827"        # Card / container background
SURFACE_CARD_HOVER = "#172033"  # Hover container state
SURFACE_BORDER = "#1f293d"      # Subtle border tone
TEXT_PRIMARY = "#f8fafc"        # High-contrast crisp white
TEXT_SECONDARY = "#94a3b8"      # Muted slate secondary text
TEXT_MUTED = "#64748b"          # Faded caption text

ACCENT_CYAN = "#00e5ff"         # Primary scientific cyan accent
ACCENT_BLUE = "#38bdf8"         # Sky / electric blue accent
ACCENT_TEAL = "#14b8a6"         # Verification teal
ACCENT_AMBER = "#f59e0b"        # Warning / incomplete amber
ACCENT_RED = "#ef4444"          # Error / malformed red
ACCENT_GREEN = "#34d399"        # Success green
ACCENT_VIOLET = "#a78bfa"       # Violet accent

# Matplotlib Scientific Colors
MPL_COLORS = [
    "#00e5ff",  # Cyan
    "#38bdf8",  # Sky Blue
    "#818cf8",  # Indigo
    "#34d399",  # Emerald
    "#f472b6",  # Rose
    "#fbbf24",  # Amber
    "#a78bfa",  # Violet
]

MATERIAL_ICONS_CSS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">'
    '<style>'
    '.material-icons { font-family: "Material Symbols Outlined"; font-weight: 400; font-size: 1.1rem; vertical-align: middle; }'
    '.icon-sm { font-size: 0.95rem; }'
    '.icon-md { font-size: 1.25rem; }'
    '</style>'
)

CUSTOM_CSS = """
<style>
/* Main Page Dark Surface */
.stApp {
    background-color: #0a0e17;
    color: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid #1f293d;
}

section[data-testid="stSidebar"] div.stRadio > label {
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
}

/* Metric card containers */
div[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #1f293d;
    border-radius: 8px;
    padding: 14px 18px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}

div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #00e5ff !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 1.6rem !important;
    font-weight: 700;
}

/* Headers */
h1, h2, h3, h4 {
    color: #f8fafc !important;
    letter-spacing: -0.02em;
    font-weight: 700;
}

h1 {
    font-size: 2.1rem !important;
    border-bottom: 1px solid #1f293d;
    padding-bottom: 0.6rem;
    margin-bottom: 1.2rem;
}

h2 {
    font-size: 1.45rem !important;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
    color: #e2e8f0 !important;
}

h3 {
    font-size: 1.15rem !important;
    color: #cbd5e1 !important;
}

/* Section header with icon */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
}

/* Styled containers */
.scientific-panel {
    background-color: #111827;
    border: 1px solid #1f293d;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
}

.scientific-panel-cyan {
    background-color: #0c182b;
    border: 1px solid #0284c7;
    border-left: 4px solid #00e5ff;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}

.scientific-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: ui-monospace, monospace;
    letter-spacing: 0.04em;
    background-color: #1e293b;
    color: #38bdf8;
    border: 1px solid #334155;
}

.scientific-badge-cyan {
    background-color: #082f49;
    color: #38bdf8;
    border: 1px solid #0284c7;
}

.scientific-badge-green {
    background-color: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
}

.scientific-badge-amber {
    background-color: #451a03;
    color: #fbbf24;
    border: 1px solid #d97706;
}

/* Streamlit Tabs */
button[data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-weight: 500;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #00e5ff !important;
    border-bottom-color: #00e5ff !important;
}

/* Streamlit DataFrames */
div[data-testid="stDataFrame"] {
    border: 1px solid #1f293d;
    border-radius: 6px;
}

/* Code block adjustments */
pre code {
    background-color: #090d16 !important;
    border: 1px solid #1f293d;
    border-radius: 6px;
    font-size: 0.85rem !important;
}

/* Sliders */
div[data-testid="stSlider"] label {
    color: #cbd5e1 !important;
    font-weight: 600;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    border: 1px dashed #1f293d;
    border-radius: 8px;
    padding: 1rem;
}

/* Charts container */
.chart-container {
    background-color: #111827;
    border: 1px solid #1f293d;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1.25rem;
}

/* HiggsLens v2 evidence-first interface */
.hero-shell {
    position: relative;
    overflow: hidden;
    padding: 2rem 2.2rem 1.8rem;
    margin: 0 0 1.4rem;
    border: 1px solid #25435f;
    border-radius: 18px;
    background:
        radial-gradient(circle at 85% 15%, rgba(0,229,255,0.18), transparent 32%),
        linear-gradient(135deg, #101d33 0%, #0b1221 58%, #111a2c 100%);
    box-shadow: 0 18px 50px rgba(0,0,0,0.24), inset 0 1px 0 rgba(255,255,255,0.04);
}
.hero-shell::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -90px;
    bottom: -150px;
    border: 1px solid rgba(0,229,255,0.20);
    border-radius: 50%;
    box-shadow: 0 0 0 22px rgba(0,229,255,0.035), 0 0 0 44px rgba(0,229,255,0.025);
}
.hero-kicker {
    color: #67e8f9;
    font: 700 0.72rem/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}
.hero-title {
    margin: 0.35rem 0 0.45rem;
    color: #f8fafc;
    font-size: clamp(2rem, 4vw, 3.3rem);
    line-height: 1.02;
    letter-spacing: -0.05em;
    font-weight: 800;
}
.hero-copy { max-width: 780px; color: #b8c7d9; font-size: 1rem; line-height: 1.6; }
.proof-strip { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-top: 1.1rem; }
.proof-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.42rem 0.7rem;
    border: 1px solid #294666;
    border-radius: 999px;
    background: rgba(7,15,29,0.58);
    color: #d5e8f4;
    font-size: 0.78rem;
}
.proof-pill strong { color: #67e8f9; }
.evidence-card {
    min-height: 112px;
    padding: 1rem 1.05rem;
    border: 1px solid #223853;
    border-radius: 13px;
    background: linear-gradient(145deg, #111d31, #0d1627);
}
.evidence-card .eyebrow {
    color: #7dd3fc;
    font: 700 0.68rem/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}
.evidence-card .value { margin-top: 0.45rem; color: #f8fafc; font-size: 1.35rem; font-weight: 800; }
.evidence-card .detail { margin-top: 0.25rem; color: #94a3b8; font-size: 0.78rem; }
.section-rule { height: 1px; margin: 1.6rem 0 1.1rem; background: linear-gradient(90deg, #2dd4bf, transparent); }
.stButton > button {
    border: 1px solid #2b597b !important;
    border-radius: 9px !important;
    background: linear-gradient(135deg, #123554, #0d2137) !important;
    color: #dff8ff !important;
    font-weight: 700 !important;
}
.stButton > button:hover { border-color: #22d3ee !important; box-shadow: 0 0 18px rgba(34,211,238,0.16); }
div[data-testid="stMetric"] { transition: transform 160ms ease, border-color 160ms ease; }
div[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #2b6a8c; }
</style>
"""

def apply_theme() -> None:
    """Inject custom CSS and Material Icons into Streamlit application."""
    import streamlit as st
    st.markdown(MATERIAL_ICONS_CSS, unsafe_allow_html=True)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
