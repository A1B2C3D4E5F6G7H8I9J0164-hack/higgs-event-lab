"""Validation and Threshold Analysis section for HiggsLens Streamlit application.

Displays held-out ROC curves, Precision-Recall curves, reliability calibration diagrams,
MLP training histories (when genuine data exists), and an interactive threshold workbench.
"""

from __future__ import annotations

import streamlit as st

from app.loader import RunDiscoveryResult
from app.metrics import calculate_operating_metrics
from app.visualizations import (
    plot_calibration_curves,
    plot_confusion_matrix_heatmap,
    plot_mlp_training_history,
    plot_pr_curves,
    plot_roc_curves,
)


def render_validation(discovery: RunDiscoveryResult) -> None:
    """Render Validation & Threshold Analysis view."""
    st.markdown("## Validation & Threshold Analysis")
    st.markdown(
        "Inspect test discrimination curves, probability calibration reliability, "
        "and explore operating cutoffs interactively on genuine held-out test predictions."
    )

    if not discovery.has_valid_runs:
        st.warning(
            "No valid experiment artifacts are currently loaded from `outputs/`. "
            "Please run the model training scripts shown in the Overview section."
        )
        return

    runs = discovery.valid_runs
    run_options = {r.run_id: r for r in runs}

    # Run Selector for Curve Overlay
    selected_run_ids = st.multiselect(
        "Select Runs to Compare on Validation Curves:",
        options=list(run_options.keys()),
        default=list(run_options.keys())[:3],
    )

    selected_runs = [run_options[rid] for rid in selected_run_ids if rid in run_options]

    t1, t2, t3, t4, t5 = st.tabs([
        "1. ROC Curves",
        "2. Precision-Recall",
        "3. Reliability Calibration",
        "4. MLP Training History",
        "5. Interactive Threshold Workbench",
    ])

    with t1:
        st.markdown("### Receiver Operating Characteristic (ROC)")
        st.markdown(
            "The ROC curve maps Signal Efficiency ($TPR$) against False Alarm Rate ($FPR$) "
            "across all possible decision thresholds."
        )
        if selected_runs:
            fig_roc = plot_roc_curves(selected_runs)
            st.pyplot(fig_roc)
        else:
            st.info("Select at least one run above to view the ROC curve.")

    with t2:
        st.markdown("### Precision-Recall Curves")
        st.markdown(
            "Precision-Recall evaluates positive-class purity across detection thresholds, "
            "crucial for understanding false-positive contamination."
        )
        if selected_runs:
            fig_pr = plot_pr_curves(selected_runs)
            st.pyplot(fig_pr)
        else:
            st.info("Select at least one run above to view the Precision-Recall curve.")

    with t3:
        st.markdown("### Reliability Diagrams (Probability Calibration)")
        st.markdown(
            "Reliability diagrams partition model prediction scores into uniform probability bins "
            "and compare the mean predicted probability against the observed positive fraction."
        )
        if selected_runs:
            fig_cal = plot_calibration_curves(selected_runs)
            st.pyplot(fig_cal)
        else:
            st.info("Select at least one run above to view calibration curves.")

    with t4:
        st.markdown("### MLP Training History (Loss & Validation ROC-AUC)")
        st.markdown(
            "> **Integrity Policy:** Training loss and validation history curves are only displayed "
            "when genuine epoch histories were exported during model training. If unavailable, an "
            "honest status notice is shown."
        )

        mlp_runs = [r for r in runs if r.is_mlp]
        if not mlp_runs:
            st.info("No MLP neural network runs found among currently loaded experiments.")
        else:
            selected_mlp_id = st.selectbox(
                "Select MLP Run:",
                options=[r.run_id for r in mlp_runs],
                index=0,
            )
            selected_mlp = next(r for r in mlp_runs if r.run_id == selected_mlp_id)

            if selected_mlp.has_training_history:
                fig_hist = plot_mlp_training_history(selected_mlp)
                if fig_hist is not None:
                    st.pyplot(fig_hist)
            else:
                st.warning(
                    f"**Training History Unavailable for `{selected_mlp.run_id}`:** "
                    "The `training.history` field was not exported in `metrics.json` for this run. "
                    "In adherence to project rules, no synthetic epoch history will be fabricated."
                )

    with t5:
        st.markdown("### Interactive Threshold Operating-Point Workbench")
        st.markdown(
            "Adjust the decision cutoff threshold $\\tau \\in [0.00, 1.00]$ to inspect signal selection, "
            "background rejection, precision, and the resulting confusion matrix on actual test events."
        )

        workbench_run_id = st.selectbox(
            "Select Run for Threshold Analysis:",
            options=list(run_options.keys()),
            index=0,
            key="workbench_select",
        )
        wb_run = run_options[workbench_run_id]

        # Check if validation-selected threshold is exported
        default_thresh = 0.50
        op_info = wb_run.operating_point
        if op_info and "validation_threshold" in op_info:
            default_thresh = float(op_info["validation_threshold"])

        st.markdown(
            f"**Active Model:** `{wb_run.model_name}` ({wb_run.feature_group}) | "
            f"**Held-out Test Events:** `{len(wb_run.test_predictions):,}`"
        )

        if op_info and "validation_threshold" in op_info:
            st.info(
                f"Validation-selected frozen threshold for target 50% signal efficiency: "
                f"**$\\tau_{{val}} = {default_thresh:.4f}$** "
                f"(achieved test background rejection: {op_info.get('test_background_rejection', 0.0):.2%})"
            )

        threshold_val = st.slider(
            "Decision Threshold ($\tau$):",
            min_value=0.00,
            max_value=1.00,
            value=float(default_thresh),
            step=0.01,
            help="Events with predicted score >= threshold are classified as signal (class 1).",
        )

        # Compute operating metrics
        metrics = calculate_operating_metrics(
            wb_run.test_labels, wb_run.test_scores, threshold_val
        )

        # Metric cards
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Signal Efficiency (Recall)", f"{metrics['signal_efficiency']:.2%}")
        m2.metric("Background Rejection", f"{metrics['background_rejection']:.2%}")
        m3.metric("Precision (PPV)", f"{metrics['precision']:.2%}" if not metrics["precision_undefined"] else "Undefined (0 selected)")
        m4.metric("Signal Selections", f"{metrics['signal_selections']:,}")
        m5.metric("False Positive Rate", f"{metrics['false_positive_rate']:.2%}")

        # Confusion Matrix Heatmap and Counts
        c_left, c_right = st.columns([1.0, 1.2])

        with c_left:
            fig_cm = plot_confusion_matrix_heatmap(metrics)
            st.pyplot(fig_cm)

        with c_right:
            st.markdown("#### Detailed Counts & Operating Rates")
            st.markdown(
                f"""
                - **True Positives ($TP$):** `{metrics['true_positives']:,}` (signal correctly identified)
                - **False Positives ($FP$):** `{metrics['false_positives']:,}` (background contaminating signal)
                - **True Negatives ($TN$):** `{metrics['true_negatives']:,}` (background rejected)
                - **False Negatives ($FN$):** `{metrics['false_negatives']:,}` (signal missed)
                - **Selection Fraction:** `{metrics['selection_rate']:.2%}` of all collision events
                - **Overall Accuracy:** `{metrics['accuracy']:.2%}`
                - **F1 Score:** `{metrics['f1_score']:.4f}`
                """
            )
            if metrics["precision_undefined"]:
                st.caption(
                    "⚠️ Notice: At threshold 1.00, zero events are selected as signal, "
                    "making precision mathematically undefined (0/0). It is safely bounded to 0.0."
                )
