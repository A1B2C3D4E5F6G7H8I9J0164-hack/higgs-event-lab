"""High-fidelity scientific visualizations for the HiggsLens dashboard.

Uses Matplotlib with high DPI, dark surfaces, and cyan/blue accents to render:
- Held-out ROC curves
- Precision-Recall curves with no-skill reference
- Reliability calibration diagrams
- Low-level vs high-level vs all feature comparisons
- MLP genuine training history (loss and validation ROC-AUC)
- Confusion matrix heatmaps for threshold operating points
- Distribution and correlation visualizers
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from app.loader import ExperimentArtifact
from app.metrics import (
    compute_calibration_points,
    compute_pr_points,
    compute_roc_points,
)
from app.styles import (
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_CYAN,
    ACCENT_RED,
    ACCENT_TEAL,
    MPL_COLORS,
    SURFACE_BASE,
    SURFACE_CARD,
    SURFACE_BORDER,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


def _configure_dark_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply unified dark scientific aesthetic to a matplotlib Axes."""
    ax.set_facecolor(SURFACE_CARD)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=11, labelpad=8)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color(SURFACE_BORDER)
        spine.set_linewidth(1.0)
    ax.grid(True, linestyle="--", alpha=0.25, color=SURFACE_BORDER)


def plot_roc_curves(runs: list[ExperimentArtifact]) -> Figure:
    """Render ROC curves for selected runs derived from held-out predictions."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150, facecolor=SURFACE_BASE)
    _configure_dark_ax(
        ax,
        title="Receiver Operating Characteristic (ROC)",
        xlabel="False Positive Rate (1 - Background Rejection)",
        ylabel="True Positive Rate (Signal Efficiency)",
    )

    # Reference diagonal (random guessing)
    ax.plot([0, 1], [0, 1], linestyle=":", color=TEXT_MUTED, linewidth=1.2, label="Random Guess (AUC = 0.500)")

    for idx, run in enumerate(runs):
        color = MPL_COLORS[idx % len(MPL_COLORS)]
        try:
            fpr, tpr, auc_score = compute_roc_points(run.test_labels, run.test_scores)
            label = f"{run.model_name} [{run.feature_group}] (AUC = {auc_score:.4f})"
            ax.plot(fpr, tpr, color=color, linewidth=2.0, label=label)
        except Exception as exc:
            ax.text(
                0.05,
                0.95 - (idx * 0.05),
                f"{run.run_id}: evaluation error ({exc})",
                color=ACCENT_RED,
                fontsize=9,
            )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    legend = ax.legend(loc="lower right", facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    return fig


def plot_pr_curves(runs: list[ExperimentArtifact]) -> Figure:
    """Render Precision-Recall curves derived from held-out predictions."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150, facecolor=SURFACE_BASE)
    _configure_dark_ax(
        ax,
        title="Precision-Recall Curves (PR)",
        xlabel="Recall (Signal Efficiency)",
        ylabel="Precision (Positive Predictive Value)",
    )

    # Calculate baseline signal fraction from first run if available
    if runs:
        first_labels = runs[0].test_labels
        pos_frac = float(np.mean(first_labels == 1))
        ax.axhline(
            y=pos_frac,
            linestyle=":",
            color=TEXT_MUTED,
            linewidth=1.2,
            label=f"Class Balance Baseline ({pos_frac:.1%})",
        )

    for idx, run in enumerate(runs):
        color = MPL_COLORS[idx % len(MPL_COLORS)]
        try:
            prec, rec, ap = compute_pr_points(run.test_labels, run.test_scores)
            label = f"{run.model_name} [{run.feature_group}] (PR-AUC = {ap:.4f})"
            ax.plot(rec, prec, color=color, linewidth=2.0, label=label)
        except Exception as exc:
            ax.text(
                0.05,
                0.95 - (idx * 0.05),
                f"{run.run_id}: evaluation error ({exc})",
                color=ACCENT_RED,
                fontsize=9,
            )

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.02)
    legend = ax.legend(loc="lower left", facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    return fig


def plot_calibration_curves(runs: list[ExperimentArtifact], n_bins: int = 10) -> Figure:
    """Render reliability diagrams (calibration curve) for selected runs."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150, facecolor=SURFACE_BASE)
    _configure_dark_ax(
        ax,
        title="Calibration Reliability Diagram",
        xlabel="Mean Predicted Probability",
        ylabel="Observed Signal Frequency",
    )

    # Perfect calibration 45-degree line
    ax.plot([0, 1], [0, 1], linestyle="--", color=TEXT_MUTED, linewidth=1.2, label="Perfect Calibration")

    for idx, run in enumerate(runs):
        color = MPL_COLORS[idx % len(MPL_COLORS)]
        try:
            frac_pos, mean_pred, brier = compute_calibration_points(
                run.test_labels, run.test_scores, n_bins=n_bins
            )
            label = f"{run.model_name} [{run.feature_group}] (Brier = {brier:.4f})"
            ax.plot(mean_pred, frac_pos, marker="s", markersize=5, color=color, linewidth=1.8, label=label)
        except Exception as exc:
            ax.text(
                0.05,
                0.95 - (idx * 0.05),
                f"{run.run_id}: calibration error ({exc})",
                color=ACCENT_RED,
                fontsize=9,
            )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    legend = ax.legend(loc="upper left", facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    return fig


def plot_feature_group_comparison(runs: list[ExperimentArtifact]) -> Figure:
    """Compare low_level, high_level, and all feature configurations across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150, facecolor=SURFACE_BASE)

    _configure_dark_ax(ax1, "Test ROC-AUC by Feature Group", "Feature Group", "ROC-AUC Score")
    _configure_dark_ax(ax2, "Test PR-AUC by Feature Group", "Feature Group", "PR-AUC Score")

    group_order = ["low_level", "high_level", "all"]
    models = sorted({r.model_name for r in runs})

    width = 0.35
    x = np.arange(len(group_order))

    for m_idx, model in enumerate(models):
        m_runs = {r.feature_group: r for r in runs if r.model_name == model}
        roc_vals = [m_runs[g].test_roc_auc if g in m_runs and m_runs[g].test_roc_auc is not None else 0.0 for g in group_order]
        pr_vals = [m_runs[g].test_pr_auc if g in m_runs and m_runs[g].test_pr_auc is not None else 0.0 for g in group_order]

        color = ACCENT_CYAN if m_idx == 0 else ACCENT_BLUE
        offset = (m_idx - (len(models) - 1) / 2) * width

        bars1 = ax1.bar(x + offset, roc_vals, width, label=model, color=color, alpha=0.85, edgecolor=SURFACE_BORDER)
        bars2 = ax2.bar(x + offset, pr_vals, width, label=model, color=color, alpha=0.85, edgecolor=SURFACE_BORDER)

        # Value annotations
        for bar, val in zip(bars1, roc_vals):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", color=TEXT_PRIMARY, fontsize=8.5)

        for bar, val in zip(bars2, pr_vals):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", color=TEXT_PRIMARY, fontsize=8.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Low-level (21)", "High-level (7)", "All (28)"])
    ax1.set_ylim(0.4, 0.9)
    leg1 = ax1.legend(facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=9)
    for text in leg1.get_texts():
        text.set_color(TEXT_SECONDARY)

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Low-level (21)", "High-level (7)", "All (28)"])
    ax2.set_ylim(0.4, 0.9)
    leg2 = ax2.legend(facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=9)
    for text in leg2.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    return fig


def plot_mlp_training_history(run: ExperimentArtifact) -> Figure | None:
    """Plot genuine MLP loss and validation ROC-AUC learning curves.

    Returns None if no genuine training history is present in the artifact.
    """
    if not run.has_training_history:
        return None

    history = run.training_history
    epochs = [h.get("epoch", idx + 1) for idx, h in enumerate(history)]
    train_loss = [h.get("train_loss", 0.0) for h in history]
    val_auc = [h.get("validation_roc_auc", 0.0) for h in history]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6), dpi=150, facecolor=SURFACE_BASE)
    _configure_dark_ax(
        ax1,
        title=f"MLP Training History: {run.run_id}",
        xlabel="Epoch",
        ylabel="Training BCE Loss",
    )

    line1 = ax1.plot(epochs, train_loss, color=ACCENT_CYAN, linewidth=2.0, marker="o", markersize=4, label="Train Loss")
    ax1.set_ylabel("Training Loss", color=ACCENT_CYAN, fontsize=10.5)
    ax1.tick_params(axis="y", labelcolor=ACCENT_CYAN)

    ax2 = ax1.twinx()
    ax2.set_facecolor("none")
    line2 = ax2.plot(epochs, val_auc, color=ACCENT_TEAL, linewidth=2.0, marker="s", markersize=4, linestyle="--", label="Val ROC-AUC")
    ax2.set_ylabel("Validation ROC-AUC", color=ACCENT_TEAL, fontsize=10.5)
    ax2.tick_params(axis="y", labelcolor=ACCENT_TEAL)
    for spine in ax2.spines.values():
        spine.set_color(SURFACE_BORDER)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    leg = ax1.legend(lines, labels, loc="center right", facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=9)
    for text in leg.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    return fig


def plot_confusion_matrix_heatmap(cm_dict: dict[str, int]) -> Figure:
    """Plot confusion matrix counts in a crisp 2x2 scientific heatmap."""
    fig, ax = plt.subplots(figsize=(4.8, 3.8), dpi=150, facecolor=SURFACE_BASE)
    _configure_dark_ax(
        ax,
        title=f"Confusion Matrix (threshold = {cm_dict['threshold']:.2f})",
        xlabel="Predicted Class",
        ylabel="True Label",
    )

    tp = cm_dict["true_positives"]
    fp = cm_dict["false_positives"]
    tn = cm_dict["true_negatives"]
    fn = cm_dict["false_negatives"]

    matrix = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(matrix, cmap="Blues", alpha=0.85)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (Background)", "1 (Signal)"])
    ax.set_yticklabels(["0 (Background)", "1 (Signal)"])

    labels = [
        [f"TN\n{tn:,}", f"FP\n{fp:,}"],
        [f"FN\n{fn:,}", f"TP\n{tp:,}"],
    ]

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            txt_color = "black" if val > matrix.max() * 0.55 else TEXT_PRIMARY
            ax.text(j, i, labels[i][j], ha="center", va="center", color=txt_color, fontsize=11, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_runtime_vs_auc(runs: list[ExperimentArtifact]) -> Figure:
    """Scatter plot comparing training runtime in seconds versus held-out test ROC-AUC."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150, facecolor=SURFACE_BASE)
    _configure_dark_ax(
        ax,
        title="Compute Efficiency: Runtime vs Test Performance",
        xlabel="Training Time (seconds)",
        ylabel="Held-out Test ROC-AUC",
    )

    for idx, run in enumerate(runs):
        runtime = run.training_seconds
        auc = run.test_roc_auc
        if runtime is not None and auc is not None:
            color = MPL_COLORS[idx % len(MPL_COLORS)]
            marker = "o" if run.is_mlp else "^"
            ax.scatter(runtime, auc, color=color, s=120, edgecolors=TEXT_PRIMARY, linewidth=1.5, marker=marker, label=f"{run.model_name} ({run.feature_group})")
            ax.annotate(
                f"{run.run_id}",
                (runtime, auc),
                textcoords="offset points",
                xytext=(8, 4),
                ha="left",
                color=TEXT_SECONDARY,
                fontsize=8.5,
            )

    # Only show legend if there are labeled artists to avoid empty-legend warnings
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(loc="lower right", facecolor=SURFACE_CARD, edgecolor=SURFACE_BORDER, fontsize=8.5)
        for text in legend.get_texts():
            text.set_color(TEXT_SECONDARY)

    fig.tight_layout()
    return fig
