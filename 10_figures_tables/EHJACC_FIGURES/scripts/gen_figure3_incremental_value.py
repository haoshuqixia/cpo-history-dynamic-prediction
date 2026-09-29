#!/usr/bin/env python3
"""Generate Figure 3: incremental value of recent CPO history.

The figure uses locked temporal-evaluation predictions and paired bootstrap
results. It writes a 300-DPI PNG and an editable vector PDF without fitting or
recalibrating either model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]
FIGURES = ROOT / "figures"
STAGE1B_RESULTS = BASE / "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908" / "results"

INK = "#171717"
MODEL_A = "#7A7A7A"
MODEL_B = "#0072B2"
REFERENCE = "#B8B8B8"
GRID = "#E8E8E8"
WHITE = "#FFFFFF"

METRIC_SPECS = [
    ("brier", "Brier score", 4, 6),
    ("auroc", "AUROC", 3, 4),
    ("average_precision", "Average precision", 3, 4),
    ("log_loss", "Log loss", 4, 5),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#4D4D4D",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.fontsize": 7.3,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.facecolor": WHITE,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )


def load_inputs():
    predictions = pd.read_csv(STAGE1B_RESULTS / "temporal_predictions.csv.gz")
    metrics = pd.read_csv(STAGE1B_RESULTS / "temporal_metrics.csv")
    deltas = pd.read_csv(STAGE1B_RESULTS / "paired_temporal_deltas.csv")
    decision = pd.read_csv(STAGE1B_RESULTS / "temporal_decision_curve.csv")
    return predictions, metrics, deltas, decision


def metric_row(metrics: pd.DataFrame, model: str, metric: str) -> pd.Series:
    rows = metrics.loc[(metrics["model"] == model) & (metrics["metric"] == metric)]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for Model {model}, metric {metric}; found {len(rows)}")
    return rows.iloc[0]


def delta_row(deltas: pd.DataFrame, metric: str) -> pd.Series:
    rows = deltas.loc[(deltas["comparison"] == "B-A") & (deltas["metric"] == metric)]
    if len(rows) != 1:
        raise ValueError(f"Expected one B-A row for metric {metric}; found {len(rows)}")
    return rows.iloc[0]


def validate_inputs(predictions, metrics, deltas, decision) -> None:
    required_prediction_columns = {
        "subject_id",
        "primary_outcome_6h",
        "ipow_stabilized_truncated",
        "prediction_A",
        "prediction_B",
    }
    missing = required_prediction_columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if len(predictions) != 3_416 or predictions["subject_id"].nunique() != 265:
        raise ValueError("Temporal cohort no longer matches the locked 3,416 landmarks / 265 patients")
    for metric, _, _, _ in METRIC_SPECS:
        metric_row(metrics, "A", metric)
        metric_row(metrics, "B", metric)
        delta_row(deltas, metric)
    thresholds = sorted(decision.loc[decision["model"] == "B", "threshold"].tolist())
    if not np.allclose(thresholds, [0.02, 0.03, 0.05, 0.075, 0.10, 0.15]):
        raise ValueError(f"Unexpected decision thresholds: {thresholds}")


def weighted_calibration(y, prediction, weight, groups=10) -> pd.DataFrame:
    order = np.argsort(prediction)
    rows = []
    for index, members in enumerate(np.array_split(order, groups), start=1):
        rows.append(
            {
                "group": index,
                "predicted": np.average(prediction[members], weights=weight[members]),
                "observed": np.average(y[members], weights=weight[members]),
            }
        )
    return pd.DataFrame(rows)


def panel_title(ax, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", pad=9, color=INK)


def signed(value: float, digits: int) -> str:
    if value > 0:
        return f"+{value:.{digits}f}"
    if value < 0:
        return f"−{abs(value):.{digits}f}"
    return f"{value:.{digits}f}"


def draw_incremental_panel(ax, metrics: pd.DataFrame, deltas: pd.DataFrame) -> None:
    panel_title(ax, "a", "Incremental predictive performance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_metric, x_a, x_b, x_delta = 0.015, 0.365, 0.575, 0.690
    ax.text(x_metric, 0.925, "Metric", ha="left", va="center", fontsize=7.2, color="#555555", weight="bold")
    ax.text((x_a + x_b) / 2, 0.925, "Model A  →  Model B", ha="center", va="center", fontsize=6.65, color="#555555", weight="bold")
    ax.text(x_delta, 0.925, "Difference (95% CI)", ha="left", va="center", fontsize=7.2, color="#555555", weight="bold")
    ax.plot([0.01, 0.99], [0.875, 0.875], color="#777777", lw=0.7, clip_on=False)

    y_positions = [0.755, 0.555, 0.355, 0.155]
    for index, ((metric, label, model_digits, delta_digits), y) in enumerate(zip(METRIC_SPECS, y_positions)):
        a = metric_row(metrics, "A", metric)
        b = metric_row(metrics, "B", metric)
        delta = delta_row(deltas, metric)

        ax.text(x_metric, y + 0.025, label, ha="left", va="center", fontsize=8.0, color=INK,
                weight="bold" if metric == "brier" else "normal")
        better_direction = "lower is better" if metric in {"brier", "log_loss"} else "higher is better"
        direction_label = f"Primary · {better_direction}" if metric == "brier" else better_direction
        ax.text(
            x_metric,
            y - 0.037,
            direction_label,
            ha="left",
            va="center",
            fontsize=5.45,
            color=MODEL_B if metric == "brier" else "#777777",
            weight="bold" if metric == "brier" else "normal",
        )

        ax.annotate(
            "",
            xy=(x_b - 0.012, y),
            xytext=(x_a + 0.012, y),
            arrowprops=dict(arrowstyle="-|>", color="#B5B5B5", lw=1.45, mutation_scale=8),
            zorder=1,
        )
        ax.scatter([x_a], [y], s=42, color=MODEL_A, edgecolor=WHITE, linewidth=0.7, zorder=4)
        ax.scatter([x_b], [y], s=48, color=MODEL_B, edgecolor=WHITE, linewidth=0.7, zorder=5)
        ax.text(x_a, y + 0.058, f"A  {a.estimate:.{model_digits}f}", ha="center", va="center", fontsize=6.45, color=MODEL_A)
        ax.text(x_b, y + 0.058, f"B  {b.estimate:.{model_digits}f}", ha="center", va="center", fontsize=6.45, color=MODEL_B, weight="bold")
        ax.text(
            x_delta,
            y,
            f"{signed(delta.estimate, delta_digits)}\n"
            f"({signed(delta.ci_low, delta_digits)} to {signed(delta.ci_high, delta_digits)})",
            ha="left",
            va="center",
            fontsize=6.75,
            color=INK,
            linespacing=1.25,
        )
        if index < len(y_positions) - 1:
            ax.plot([0.01, 0.99], [y - 0.10, y - 0.10], color=GRID, lw=0.65, clip_on=False)


def clean_axes(ax, *, grid_axis="both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def draw_roc_panel(ax, predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    panel_title(ax, "b", "Discrimination")
    y = predictions["primary_outcome_6h"].to_numpy(dtype=int)
    weight = predictions["ipow_stabilized_truncated"].to_numpy(dtype=float)
    for model, color, linestyle in [("A", MODEL_A, "--"), ("B", MODEL_B, "-")]:
        prediction = predictions[f"prediction_{model}"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, prediction, sample_weight=weight)
        row = metric_row(metrics, model, "auroc")
        label = f"Model {model}: AUROC {row.estimate:.3f} ({row.ci_low:.3f}–{row.ci_high:.3f})"
        ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=1.8, label=label, zorder=3)
    ax.plot([0, 1], [0, 1], color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.0, label="Chance", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    clean_axes(ax)
    ax.legend(loc="lower right", handlelength=2.6, borderaxespad=0.4)


def draw_calibration_panel(ax, predictions: pd.DataFrame) -> None:
    panel_title(ax, "c", "Calibration")
    y = predictions["primary_outcome_6h"].to_numpy(dtype=int)
    weight = predictions["ipow_stabilized_truncated"].to_numpy(dtype=float)
    for model, color, linestyle, marker in [
        ("A", MODEL_A, "--", "o"),
        ("B", MODEL_B, "-", "o"),
    ]:
        prediction = predictions[f"prediction_{model}"].to_numpy(dtype=float)
        calibration = weighted_calibration(y, prediction, weight)
        ax.plot(
            calibration["predicted"],
            calibration["observed"],
            color=color,
            linestyle=linestyle,
            linewidth=1.55,
            marker=marker,
            markersize=3.6,
            markeredgewidth=0,
            label=f"Model {model}",
            zorder=3,
        )
    ax.plot([0, 0.30], [0, 0.30], color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.0, label="Ideal", zorder=1)
    ax.set_xlim(0, 0.30)
    ax.set_ylim(0, 0.30)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed risk")
    ax.xaxis.set_major_locator(MultipleLocator(0.10))
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    clean_axes(ax)
    ax.legend(loc="upper left", handlelength=2.5)


def draw_decision_panel(ax, decision: pd.DataFrame) -> None:
    panel_title(ax, "d", "Decision curve analysis")
    styles = {
        "A": (MODEL_A, "--", 1.55, "Model A", "o"),
        "B": (MODEL_B, "-", 1.8, "Model B", "o"),
        "treat_all": (REFERENCE, ":", 1.6, "Treat all", None),
        "treat_none": (INK, "-", 0.9, "Treat none", None),
    }
    for model in ["A", "B", "treat_all", "treat_none"]:
        color, linestyle, linewidth, label, marker = styles[model]
        rows = decision.loc[decision["model"] == model].sort_values("threshold")
        ax.plot(
            rows["threshold"],
            rows["net_benefit"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=3.5 if marker else 0,
            markeredgewidth=0,
            label=label,
            zorder=3 if model in {"A", "B"} else 2,
        )
    ax.set_xlim(0.015, 0.155)
    ax.set_ylim(-0.12, 0.055)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    clean_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.03), handlelength=2.5)


def build_figure():
    configure_style()
    predictions, metrics, deltas, decision = load_inputs()
    validate_inputs(predictions, metrics, deltas, decision)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.9), facecolor=WHITE)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.085, top=0.955, wspace=0.30, hspace=0.45)
    draw_incremental_panel(axes[0, 0], metrics, deltas)
    draw_roc_panel(axes[0, 1], predictions, metrics)
    draw_calibration_panel(axes[1, 0], predictions)
    draw_decision_panel(axes[1, 1], decision)
    return fig


def generate_figure(output_dir: Path = FIGURES, stem: str = "figure3_temporal_performance", dpi: int = 300):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, facecolor=WHITE)
    figure.savefig(png_path, dpi=dpi, facecolor=WHITE)
    plt.close(figure)
    print(pdf_path)
    print(png_path)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    parser.add_argument("--stem", default="figure3_temporal_performance")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_figure(args.output_dir, args.stem, args.dpi)


if __name__ == "__main__":
    main()
