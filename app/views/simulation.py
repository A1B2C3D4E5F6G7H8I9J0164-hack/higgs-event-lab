"""Model Simulation section for HiggsLens Streamlit application.

Lets users run the trained benchmark models on their own data:
- Upload a CSV of events and get per-model prediction probabilities.
- Inspect a single held-out event and compare model scores side by side.
- View prediction distributions as interactive histograms.

Gracefully falls back to a read-only demo mode (using the test predictions
already stored under ``outputs/``) when the saved model weights are not
present on the current machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app.loader import ExperimentArtifact, RunDiscoveryResult
from app.styles import ACCENT_CYAN, ACCENT_TEAL, TEXT_MUTED, TEXT_PRIMARY
from app.visualizations import MPL_COLORS, SURFACE_BASE, SURFACE_CARD, _configure_dark_ax

try:  # torch is optional at import time so the dashboard stays importable
    import torch
    HAS_TORCH = True
except Exception:  # pragma: no cover - environment dependent
    HAS_TORCH = False

import joblib

ROOT_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = ROOT_DIR / "outputs" / "model_weights"

FEATURE_LABELS = {
    "lepton_p_t": "Lepton pT",
    "lepton_eta": "Lepton eta",
    "lepton_phi": "Lepton phi",
    "missing_energy_magnitude": "Missing energy magnitude",
    "missing_energy_phi": "Missing energy phi",
    "jet_1_pt": "Jet 1 pT",
    "jet_1_eta": "Jet 1 eta",
    "jet_1_phi": "Jet 1 phi",
    "jet_1_btag": "Jet 1 b-tag",
    "jet_2_pt": "Jet 2 pT",
    "jet_2_eta": "Jet 2 eta",
    "jet_2_phi": "Jet 2 phi",
    "jet_2_btag": "Jet 2 b-tag",
    "jet_3_pt": "Jet 3 pT",
    "jet_3_eta": "Jet 3 eta",
    "jet_3_phi": "Jet 3 phi",
    "jet_3_btag": "Jet 3 b-tag",
    "jet_4_pt": "Jet 4 pT",
    "jet_4_eta": "Jet 4 eta",
    "jet_4_phi": "Jet 4 phi",
    "jet_4_btag": "Jet 4 b-tag",
    "m_jj": "Mass jj",
    "m_jjb": "Mass jjb",
    "m_bb": "Mass bb",
    "m_wbj": "Mass Wbj",
    "m_wbj2": "Mass Wbj2",
    "m_wwbj": "Mass Wwbj",
    "m_wwbb": "Mass Wwbb",
}


def _friendly(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def load_models() -> dict[str, Any]:
    """Load saved models and the shared scaler if present on disk."""
    models: dict[str, Any] = {}
    scaler = None

    logreg_path = WEIGHTS_DIR / "logistic_regression_tuned.joblib"
    xgb_path = WEIGHTS_DIR / "xgboost_tuned.joblib"
    scaler_path = WEIGHTS_DIR / "feature_scaler.joblib"
    mlp_path = WEIGHTS_DIR / "wide_mlp_tuned.pt"

    if logreg_path.exists():
        models["logistic_regression_tuned"] = joblib.load(logreg_path)
    if xgb_path.exists():
        models["xgboost_tuned"] = joblib.load(xgb_path)
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    if mlp_path.exists() and HAS_TORCH:
        try:
            models["wide_mlp_tuned"] = torch.load(mlp_path, map_location="cpu")
        except Exception:
            models["wide_mlp_tuned"] = torch.load(
                mlp_path, map_location="cpu", weights_only=False
            )

    return models, scaler


def predict_scores(models: dict[str, Any], scaler: Any, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return predicted probability scores per model for a feature DataFrame."""
    from higgs_event_lab.features import ALL_FEATURES

    feat_cols = list(ALL_FEATURES)
    missing = [c for c in feat_cols if c not in X.columns]
    if missing:
        raise ValueError(f"Uploaded data missing feature columns: {missing}")

    Xf = X[feat_cols].copy()
    if scaler is not None:
        Xf = pd.DataFrame(scaler.transform(Xf.values), columns=feat_cols, index=Xf.index)

    scores: dict[str, np.ndarray] = {}
    for name, model in models.items():
        if name.startswith("wide_mlp") and HAS_TORCH:
            import torch

            model.eval()
            with torch.no_grad():
                batch = torch.as_tensor(Xf.values, dtype=torch.float32)
                prob = torch.sigmoid(model(batch)).squeeze(1).cpu().numpy()
            scores[name] = prob
        elif hasattr(model, "predict_proba"):
            scores[name] = model.predict_proba(Xf.values)[:, 1]
        else:
            scores[name] = model.predict(Xf.values).astype(np.float64)
    return scores


def render_simulation(discovery: RunDiscoveryResult) -> None:
    """Render the Model Simulation view."""
    st.markdown("## Model Simulation: Run Trained Models on Your Data")
    st.markdown(
        "Upload a CSV of collision events (28 feature columns) to get prediction "
        "probabilities from the saved logistic regression, XGBoost, and wide MLP models. "
        "You can also inspect a single event and compare all models side by side."
    )

    models, scaler = load_models()
    from higgs_event_lab.features import ALL_FEATURES

    feature_cols = list(ALL_FEATURES)

    if not models:
        st.info(
            "Saved model weights were not found under `outputs/model_weights/`. "
            "Run the full Colab notebook (`notebooks/03_full/00_full_higgs_colab.ipynb`) "
            "to train and save the models, then reopen this page."
        )
        st.markdown("### Demo: held-out test score distributions")
        _render_demo_distributions(discovery)
        return

    st.success(f"{len(models)} model(s) loaded and ready: {', '.join(sorted(models))}")

    tab_upload, tab_single, tab_dist = st.tabs(
        ["1. Batch CSV Upload", "2. Single-Event Prediction", "3. Score Distributions"]
    )

    with tab_upload:
        _render_batch_upload(models, scaler, feature_cols)

    with tab_single:
        _render_single_event(models, scaler, feature_cols)

    with tab_dist:
        _render_score_distributions(discovery, models, scaler, feature_cols)


def _render_batch_upload(models: dict[str, Any], scaler: Any, feature_cols: list[str]) -> None:
    st.markdown("### Batch CSV Upload")
    st.markdown(
        "Upload a CSV containing the 28 feature columns (no label column needed). "
        "Each row is scored by every loaded model."
    )
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="sim_batch_csv")
    if uploaded is None:
        st.caption("Example: a CSV with columns such as lepton_p_t, lepton_eta, ..., m_wwbb.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    try:
        scores = predict_scores(models, scaler, df)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    out = df.copy()
    for name, probs in scores.items():
        out[f"{name}_score"] = probs

    st.markdown(f"### Predictions for {len(df)} event(s)")
    st.dataframe(out.head(20), use_container_width=True)

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions CSV",
        data=csv_bytes,
        file_name="higgs_simulation_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Average score per model bar chart
    avg_df = pd.DataFrame(
        {
            "model": list(scores.keys()),
            "average_signal_score": [float(np.mean(p)) for p in scores.values()],
        }
    )
    st.bar_chart(avg_df, x="model", y="average_signal_score", use_container_width=True)


def _render_single_event(models: dict[str, Any], scaler: Any, feature_cols: list[str]) -> None:
    st.markdown("### Single-Event Prediction")
    st.markdown("Adjust feature values and compare every model's signal probability.")

    inputs: dict[str, float] = {}
    cols = st.columns(4)
    for idx, feat in enumerate(feature_cols):
        with cols[idx % 4]:
            inputs[feat] = st.slider(
                _friendly(feat),
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.1,
                key=f"sim_{feat}",
            )

    if st.button("Score this event", key="sim_score_btn", use_container_width=True):
        row = pd.DataFrame([inputs], columns=feature_cols)
        try:
            scores = predict_scores(models, scaler, row)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        st.markdown("#### Signal probability by model")
        score_rows = []
        for name in sorted(scores):
            score_rows.append({"Model": name.replace("_", " ").title(), "Signal Probability": float(scores[name][0])})
        score_df = pd.DataFrame(score_rows)
        st.dataframe(score_df, use_container_width=True, hide_index=True)

        best = max(score_rows, key=lambda r: r["Signal Probability"])
        st.markdown(
            f"> Highest signal probability: **{best['Model']}** at **{best['Signal Probability']:.3f}**"
        )

        # Simple horizontal bar of scores
        chart_df = score_df.sort_values("Signal Probability")
        st.bar_chart(chart_df, x="Model", y="Signal Probability", use_container_width=True)


def _render_demo_distributions(discovery: RunDiscoveryResult) -> None:
    if not discovery.has_valid_runs:
        st.warning("No valid experiment artifacts found to demonstrate scores.")
        return
    run = discovery.valid_runs[0]
    scores = run.test_scores
    fig_hist, ax_hist = __import__("matplotlib.pyplot").pyplot.subplots(figsize=(7, 3.6))
    ax_hist.hist(scores[run.test_labels == 1], bins=30, alpha=0.7, label="Signal", color="#00e5ff")
    ax_hist.hist(scores[run.test_labels == 0], bins=30, alpha=0.7, label="Background", color="#f472b6")
    ax_hist.set_xlabel("Predicted signal probability")
    ax_hist.set_ylabel("Event count")
    ax_hist.set_title(f"Demo score distribution — {run.run_id}")
    ax_hist.legend()
    st.pyplot(fig_hist)


def _render_score_distributions(
    discovery: RunDiscoveryResult, models: dict[str, Any], scaler: Any, feature_cols: list[str]
) -> None:
    st.markdown("### Score Distributions")
    st.markdown(
        "Compare how each model spreads predicted probabilities across the held-out test set."
    )
    if not discovery.has_valid_runs:
        st.info("Load valid experiment artifacts to view distributions.")
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for idx, run in enumerate(discovery.valid_runs[:4]):
        color = MPL_COLORS[idx % len(MPL_COLORS)]
        ax.hist(run.test_scores, bins=30, alpha=0.5, label=run.run_id, color=color)
    ax.set_xlabel("Predicted signal probability")
    ax.set_ylabel("Event count")
    ax.set_title("Held-out test score distribution per run")
    ax.legend(fontsize=8)
    st.pyplot(fig)
