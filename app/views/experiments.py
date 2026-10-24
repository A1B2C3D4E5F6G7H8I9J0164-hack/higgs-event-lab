"""Experiment Comparison section for HiggsLens Streamlit application.

Displays comprehensive model leaderboards, multi-group benchmark comparisons,
and compute efficiency trade-offs across low-level, high-level, and all features.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.loader import RunDiscoveryResult
from app.visualizations import (
    plot_feature_group_comparison,
    plot_runtime_vs_auc,
)


def render_experiments(discovery: RunDiscoveryResult) -> None:
    """Render Experiment Comparison view."""
    st.markdown("## Experiment Comparison: Model & Feature-Group Metrics")
    st.markdown(
        "Evaluate comparative performance across models (Logistic Regression, compact MLP) "
        "and feature configurations (`low_level`, `high_level`, `all`). All evaluations derive "
        "strictly from held-out test predictions under the `higgs-split-v1` partition."
    )

    if not discovery.has_valid_runs:
        st.warning(
            "No valid experiment artifacts are currently loaded from `outputs/`. "
            "Please run the model training scripts shown in the Overview section."
        )
        return

    runs = discovery.valid_runs

    # Tabbed views for Leaderboard and Visual Comparative Analytics
    tab_table, tab_groups, tab_compute = st.tabs([
        "1. Comparison Leaderboard",
        "2. Feature-Group Comparison",
        "3. Compute Efficiency Trade-Off",
    ])

    with tab_table:
        st.markdown("### Held-Out Evaluation Summary")

        records = []
        for r in runs:
            records.append({
                "Run ID": r.run_id,
                "Model": r.model_name,
                "Family": r.model_family,
                "Group": r.feature_group,
                "Features": r.feature_count,
                "Split Seed": r.split.get("seed", 42),
                "Runtime (s)": round(r.training_seconds, 2) if r.training_seconds is not None else None,
                "Val ROC-AUC": round(r.validation_metrics.get("roc_auc", 0.0), 4) if "roc_auc" in r.validation_metrics else None,
                "Test ROC-AUC": round(r.test_roc_auc, 4) if r.test_roc_auc is not None else None,
                "Test PR-AUC": round(r.test_pr_auc, 4) if r.test_pr_auc is not None else None,
                "Brier Score": round(r.test_brier_score, 4) if r.test_brier_score is not None else None,
            })

        df_leaderboard = pd.DataFrame(records)

        # Sort options
        sort_col = st.selectbox(
            "Sort Leaderboard By:",
            options=["Test ROC-AUC", "Test PR-AUC", "Brier Score", "Runtime (s)", "Group"],
            index=0,
        )
        ascending = sort_col in ("Brier Score", "Runtime (s)")
        df_sorted = df_leaderboard.sort_values(by=sort_col, ascending=ascending, na_position="last")

        st.dataframe(df_sorted, use_container_width=True, hide_index=True)

        st.markdown(
            "> **Note on Brier Score:** Lower values indicate superior probabilistic calibration, "
            "where 0.0 is perfect certainty and 0.25 represents uninformative 50/50 guessing."
        )

    with tab_groups:
        st.markdown("### Representation Learning: Low-Level (21) vs High-Level (7) vs All (28)")
        st.markdown(
            "Does the neural network learn equivalent or superior representations from raw kinematic "
            "variables alone, or do human-engineered invariant masses remain necessary?"
        )

        fig_groups = plot_feature_group_comparison(runs)
        st.pyplot(fig_groups)

        st.markdown(
            """
            <div class="scientific-panel">
                <h4 style="color:#00e5ff; margin-top:0;">Key Scientific Takeaways</h4>
                <ul style="font-size:0.9rem; color:#cbd5e1; margin-bottom:0;">
                    <li><strong>Classical Linear Baseline:</strong> Logistic Regression achieves markedly higher
                    performance on <code>high_level</code> (7 invariant masses) than on <code>low_level</code> (21 kinematics),
                    confirming that linear hyperplanes cannot easily model relativistic Lorentz invariants.</li>
                    <li><strong>Cross-Feature Synergy:</strong> Combining both groups into <code>all</code> (28 features)
                    yields maximal discrimination for both model families.</li>
                    <li><strong>Representation Gap:</strong> The MLP narrows the performance deficit on low-level features
                    due to nonlinear activation and hidden representations, but still benefits from explicit invariant masses.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_compute:
        st.markdown("### Compute Efficiency: Training Wall-Clock Time vs Performance")
        st.markdown(
            "Compare the trade-off between computational budget (GPU/CPU wall-clock seconds) "
            "and discrimination power (held-out test ROC-AUC)."
        )

        fig_runtime = plot_runtime_vs_auc(runs)
        st.pyplot(fig_runtime)
