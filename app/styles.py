"""Scientific theme styling, color tokens, and custom CSS for HiggsLens dashboard.

Implements a cohesive dark graphite and navy palette with restrained cyan and
electric blue accents, crisp typography, and scientific data tables.
"""

from __future__ import annotations

# Color Palette Tokens
SURFACE_BASE = "#0a0e17"       # Deep dark navy-graphite background
SURFACE_CARD = "#111827"       # Card / container background
SURFACE_CARD_HOVER = "#172033" # Hover container state
SURFACE_BORDER = "#1f293d"     # Subtle border tone
TEXT_PRIMARY = "#f8fafc"       # High-contrast crisp white
TEXT_SECONDARY = "#94a3b8"     # Muted slate secondary text
TEXT_MUTED = "#64748b"         # Faded caption text

ACCENT_CYAN = "#00e5ff"        # Primary scientific cyan accent
ACCENT_BLUE = "#38bdf8"        # Sky / electric blue accent
ACCENT_TEAL = "#14b8a6"        # Verification teal
ACCENT_AMBER = "#f59e0b"       # Warning / incomplete amber
ACCENT_RED = "#ef4444"         # Error / malformed red

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
</style>
"""


def apply_theme() -> None:
    """Inject custom CSS into Streamlit application."""
    import streamlit as st

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
