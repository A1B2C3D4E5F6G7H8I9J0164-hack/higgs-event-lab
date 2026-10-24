"""Overview section for HiggsLens Streamlit application.

Displays project objectives, dataset context, artifact discovery status,
and verified commands when runs are missing.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.loader import RunDiscoveryResult


def render_overview(discovery: RunDiscoveryResult) -> None:
    """Render Overview view."""
    st.markdown("## Project Overview & Artifact Registry")
    st.markdown(
        "HiggsLens provides an interactive scientific interface for the **Higgs Event Lab** "
        "benchmark on the simulated UCI HIGGS dataset (Baldi et al., 2014). "
        "The objective is rigorous comparative evaluation between classical linear models "
        "and compact neural networks across low-level kinematic and expert-derived high-level features."
    )

    # Core Research Indicators
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Dataset Events",
            value="11,000,000",
            help="Full simulated Monte Carlo collision benchmark from Baldi et al. (2014)",
        )
    with c2:
        st.metric(
            label="Input Features",
            value="28 (21 + 7)",
            help="21 low-level kinematic variables and 7 high-level invariant masses",
        )
    with c3:
        st.metric(
            label="Split Protocol",
            value="higgs-split-v1",
            help="Stratified 70% train / 15% validation / 15% test with fixed seed 42",
        )
    with c4:
        st.metric(
            label="Usable Runs",
            value=f"{len(discovery.valid_runs)}",
            help=f"Discovered in '{discovery.base_dir}' with valid schema",
        )

    st.markdown("---")

    # Artifact Discovery Status Section
    st.markdown("### Artifact Registry Status")

    if not discovery.dir_exists:
        st.warning(
            f"Directory `{discovery.base_dir}` does not exist yet. "
            "No experiment runs have been generated locally."
        )
    elif discovery.total_scanned == 0:
        st.info(
            f"Directory `{discovery.base_dir}` exists but contains no run subdirectories."
        )
    else:
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Successfully Loaded", f"{len(discovery.valid_runs)} runs")
        with stat_col2:
            st.metric("Incomplete Runs", f"{len(discovery.incomplete_runs)} runs")
        with stat_col3:
            st.metric("Malformed Runs", f"{len(discovery.malformed_runs)} runs")

    # Show Diagnostic Warnings if there are incomplete or malformed runs
    if discovery.incomplete_runs:
        with st.expander(f"⚠️ {len(discovery.incomplete_runs)} Incomplete Run(s) Detected", expanded=False):
            st.markdown(
                "An experiment folder is considered **incomplete** if it lacks either `metrics.json` "
                "or `test_predictions.csv`."
            )
            for inc in discovery.incomplete_runs:
                st.markdown(
                    f"- **`{inc.run_id}`** in `{inc.run_dir}`: {inc.reason} "
                    f"(missing: `{', '.join(inc.missing_files)}`)"
                )

    if discovery.malformed_runs:
        with st.expander(f"❌ {len(discovery.malformed_runs)} Malformed Run(s) Detected", expanded=False):
            st.markdown(
                "A run is **malformed** if its JSON schema violates `higgs-experiment-v1`, "
                "or if predictions contain non-finite numbers or invalid labels."
            )
            for mal in discovery.malformed_runs:
                st.markdown(f"- **`{mal.run_id}`**: {mal.reason}")

    # Available Valid Experiments Table
    if discovery.has_valid_runs:
        st.markdown("### Loaded Benchmark Experiments")
        table_data = []
        for run in discovery.valid_runs:
            table_data.append(
                {
                    "Run ID": run.run_id,
                    "Model": run.model_name,
                    "Family": run.model_family,
                    "Feature Group": run.feature_group,
                    "Features": run.feature_count,
                    "Seed": run.split.get("seed", 42),
                    "Train Time (s)": f"{run.training_seconds:.2f}" if run.training_seconds is not None else "N/A",
                    "Val ROC-AUC": f"{run.validation_metrics.get('roc_auc', 0.0):.4f}" if "roc_auc" in run.validation_metrics else "N/A",
                    "Test ROC-AUC": f"{run.test_roc_auc:.4f}" if run.test_roc_auc is not None else "N/A",
                    "Test PR-AUC": f"{run.test_pr_auc:.4f}" if run.test_pr_auc is not None else "N/A",
                    "Brier Score": f"{run.test_brier_score:.4f}" if run.test_brier_score is not None else "N/A",
                }
            )
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    else:
        st.markdown("### Honest Empty State: Reproducing Experiment Runs")
        st.markdown(
            "> **Integrity Policy:** HiggsLens will never invent, simulate, or display mock benchmark "
            "metrics without genuine artifacts. In accordance with project rule 4 (`README.md`), all "
            "scores must come from executed runs adhering to `higgs-experiment-v1`."
        )

        st.markdown(
            "To generate actual experiment runs for both classical baselines and neural networks, "
            "execute these verified CLI commands in your terminal:"
        )

        st.markdown("#### 1. Extract a Documented Subset (250,000 events)")
        st.code(
            "python -m higgs_event_lab.data \\\n"
            "    --source data/raw/HIGGS.csv.gz \\\n"
            "    --output data/processed/higgs_sample_250k.csv.gz \\\n"
            "    --rows 250000",
            language="bash",
        )

        st.markdown("#### 2. Train Logistic Regression Baselines Across All Feature Groups")
        st.code(
            "python -m higgs_event_lab.classical \\\n"
            "    --data data/processed/higgs_sample_250k.csv.gz \\\n"
            "    --all-groups \\\n"
            "    --output-dir outputs",
            language="bash",
        )

        st.markdown("#### 3. Train Compact MLP Neural Networks Across All Feature Groups")
        st.code(
            "python -m higgs_event_lab.neural \\\n"
            "    --data data/processed/higgs_sample_250k.csv.gz \\\n"
            "    --all-groups \\\n"
            "    --output-dir outputs",
            language="bash",
        )

        st.info(
            "Once training completes, reload this dashboard to inspect curves, "
            "reliability diagrams, and threshold operating points."
        )
