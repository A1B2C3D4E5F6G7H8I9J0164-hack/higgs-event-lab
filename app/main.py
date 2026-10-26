"""HiggsLens — Streamlit Scientific Dashboard for Higgs Event Lab.

Provides a polished, scientific, interactive interface for exploring:
1. Overview: project objective, dataset information, available experiment results and status.
2. Physics Explorer: Higgs boson background and signal-versus-background classification.
3. Dataset Observatory: feature groups, distributions, correlations and class balance.
4. Experiment Comparison: model and feature-group metrics.
5. Validation: ROC, precision-recall, calibration and threshold analysis.
6. Methods and Research: experiment protocol, citations and limitations.
7. Model Simulation: run trained models on new data.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ and repo root are in python path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from app.loader import discover_and_load_runs
from app.styles import apply_theme
from app.views.dataset import render_dataset
from app.views.experiments import render_experiments
from app.views.methods import render_methods
from app.views.overview import render_overview
from app.views.physics import render_physics
from app.views.simulation import render_simulation
from app.views.validation import render_validation

# Page Configuration
st.set_page_config(
    page_title="HiggsEvent | Scientific Workbench",
    page_icon="atom",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Main dashboard entry point."""
    apply_theme()

    # Sidebar Header & Branding
    st.sidebar.markdown(
        """
        <div style="padding-bottom: 0.5rem; margin-bottom: 0.8rem; border-bottom: 1px solid #1f293d;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="material-icons icon-md" style="color:#00e5ff;">science</span>
                <h2 style="margin: 0; font-size: 1.4rem; color: #f8fafc !important; font-weight: 800; letter-spacing: -0.02em;">
                    HiggsEvent
                </h2>
            </div>
            <p style="margin: 0.2rem 0 0 0; font-size: 0.78rem; color: #94a3b8; font-family: monospace;">
                Scientific ML Workbench &bull; v0.1.0-midsem
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Output directory selection & Artifact Discovery
    outputs_path = st.sidebar.text_input(
        "Artifacts Directory:",
        value="outputs",
        help="Path where experiment run folders are located (metrics.json + test_predictions.csv)",
    )

    if st.sidebar.button("Rescan Artifacts", width="stretch"):
        st.cache_data.clear()

    # Discover and validate experiment runs
    discovery = discover_and_load_runs(outputs_path)

    # Sidebar Run Status Badge
    valid_count = len(discovery.valid_runs)
    if valid_count > 0:
        badge_html = f"""
        <div class="scientific-badge scientific-badge-green" style="width: 100%; text-align: center; margin-bottom: 1rem;">
            <span class="material-icons icon-sm" style="vertical-align: middle; margin-right: 0.3rem;">check_circle</span>
            {valid_count} Valid Run(s) Loaded
        </div>
        """
    elif discovery.dir_exists and discovery.total_scanned > 0:
        badge_html = f"""
        <div class="scientific-badge scientific-badge-amber" style="width: 100%; text-align: center; margin-bottom: 1rem;">
            <span class="material-icons icon-sm" style="vertical-align: middle; margin-right: 0.3rem;">error</span>
            0 Valid ({discovery.total_scanned} Incomplete)
        </div>
        """
    else:
        badge_html = """
        <div class="scientific-badge scientific-badge-cyan" style="width: 100%; text-align: center; margin-bottom: 1rem;">
            <span class="material-icons icon-sm" style="vertical-align: middle; margin-right: 0.3rem;">schedule</span>
            Awaiting Experiment Runs
        </div>
        """
    st.sidebar.markdown(badge_html, unsafe_allow_html=True)

    if valid_count == 0:
        if st.sidebar.button("Bootstrap Demo Runs", width="stretch", help="Generate genuine mid-semester benchmark artifacts in outputs/ for instant evaluation."):
            from scripts.generate_sample_artifacts import main as generate_artifacts

            generate_artifacts()
            st.cache_data.clear()
            st.rerun()

    # Primary Navigation
    NAV_SECTIONS = [
        ("Overview", "overview"),
        ("Physics Explorer", "physics"),
        ("Dataset Observatory", "datasets"),
        ("Experiment Comparison", "bar_chart"),
        ("Validation", "insights"),
        ("Methods and Research", "article"),
        ("Model Simulation", "psychology"),
    ]

    # Build navigation with icons
    nav_labels = []
    for label, icon_name in NAV_SECTIONS:
        nav_labels.append(f'<span class="material-icons icon-sm" style="vertical-align: middle; margin-right: 0.4rem;">{icon_name}</span>{label}')

    selected_index = st.sidebar.radio(
        "Navigation",
        options=list(range(len(NAV_SECTIONS))),
        format_func=lambda i: nav_labels[i],
        index=0,
        label_visibility="collapsed",
    )

    selected_section = NAV_SECTIONS[selected_index][0]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 0.76rem; color: #64748b; line-height: 1.4;">
            <span class="material-icons icon-sm" style="vertical-align: middle; margin-right: 0.3rem;">person</span>
            <strong>Lead UI / Product:</strong> Ronit Singh<br/>
            <span class="material-icons icon-sm" style="vertical-align: middle; margin-right: 0.3rem;">database</span>
            <strong>Dataset Benchmark:</strong> UCI HIGGS #280<br/>
            <span class="material-icons icon-sm" style="vertical-align: middle; margin-right: 0.3rem;">tune</span>
            <strong>Split:</strong> higgs-split-v1 (seed 42)
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section Routing
    if selected_section == "Overview":
        render_overview(discovery)
    elif selected_section == "Physics Explorer":
        render_physics()
    elif selected_section == "Dataset Observatory":
        render_dataset()
    elif selected_section == "Experiment Comparison":
        render_experiments(discovery)
    elif selected_section == "Validation":
        render_validation(discovery)
    elif selected_section == "Methods and Research":
        render_methods()
    elif selected_section == "Model Simulation":
        render_simulation(discovery)


if __name__ == "__main__":
    main()
