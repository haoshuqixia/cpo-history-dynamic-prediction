#!/usr/bin/env python3
"""Generate Figure 5: frozen-model multicentre external validation in eICU.

The script uses locked eICU predictions and two-level bootstrap results. It
does not refit, tune, or recalibrate either MIMIC-derived model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external_validation"
FIGURES = ROOT / "figures"
SOURCE = ROOT.parents[1] / "CPO_EICU_EXTERNAL_DISCHARGE_BOUNDARY_R1_V1_0" / "results"

INK = "#171717"
MODEL_A = "#7A7A7A"
MODEL_B = "#0072B2"
REFERENCE = "#B8B8B8"
GRID = "#E8E8E8"
WHITE = "#FFFFFF"

METRIC_SPECS = [
    ("Brier score", "brier", "delta_brier_B_minus_A", -1),
    ("AUROC", "auroc", "delta_auroc_B_minus_A", 1),
    ("Log loss", "log_loss", "delta_log_loss_B_minus_A", -1),
    ("Average precision", "average_precision", "delta_average_precision_B_minus_A", 1),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.3,
            "axes.titlesize": 9.7,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.7,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#4D4D4D",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.3,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.fontsize": 6.8,
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


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))


def calibration_data(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ["A", "B"]:
        prediction_column = f"prediction_{model}"
        bins = pd.qcut(predictions[prediction_column], q=10, duplicates="drop")
        binned = predictions.assign(_bin=bins)
        for order, (_, group) in enumerate(binned.groupby("_bin", observed=True), start=1):
            rows.append(
                {
                    "model": model,
                    "decile": order,
                    "n": len(group),
                    "mean_predicted": weighted_mean(group[prediction_column], group["weight_primary"]),
                    "observed_fraction": weighted_mean(group["event_6h"].astype(float), group["weight_primary"]),
                }
            )
    return pd.DataFrame(rows)


def comparison_data() -> pd.DataFrame:
    absolute = pd.read_csv(EXTERNAL / "stage_e2_absolute_metrics.csv")
    absolute = absolute.loc[absolute["analysis"].eq("primary_ipow_floor_and_q01q99")]
    absolute = absolute.pivot(index="metric", columns="model", values="estimate")
    intervals = pd.read_csv(EXTERNAL / "stage_e2_primary_comparison_ci.csv").set_index("metric")
    rows = []
    for label, metric, delta_name, favourable_sign in METRIC_SPECS:
        interval = intervals.loc[delta_name]
        model_a = float(absolute.loc[metric, "A"])
        absolute_difference = float(interval["estimate"])
        absolute_low = float(interval["ci95_low"])
        absolute_high = float(interval["ci95_high"])
        relative_change = favourable_sign * absolute_difference / model_a * 100
        if favourable_sign > 0:
            relative_low = absolute_low / model_a * 100
            relative_high = absolute_high / model_a * 100
        else:
            relative_low = -absolute_high / model_a * 100
            relative_high = -absolute_low / model_a * 100
        rows.append(
            {
                "metric": label,
                "metric_key": metric,
                "model_A": model_a,
                "model_B": float(absolute.loc[metric, "B"]),
                "difference_B_minus_A": absolute_difference,
                "ci95_low": absolute_low,
                "ci95_high": absolute_high,
                "relative_change_favouring_B_percent": relative_change,
                "relative_ci95_low_percent": relative_low,
                "relative_ci95_high_percent": relative_high,
            }
        )
    return pd.DataFrame(rows)


def calibration_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "weighted_observed_event_rate": weighted_mean(predictions["event_6h"].astype(float), predictions["weight_primary"]),
                "model_A_mean_predicted_risk": weighted_mean(predictions["prediction_A"], predictions["weight_primary"]),
                "model_B_mean_predicted_risk": weighted_mean(predictions["prediction_B"], predictions["weight_primary"]),
            }
        ]
    )


def validate_inputs(predictions, comparison, hospitals, summary) -> None:
    if len(predictions) != 7_753:
        raise ValueError(f"Expected 7,753 eICU landmarks; found {len(predictions):,}")
    if comparison["metric"].tolist() != [item[0] for item in METRIC_SPECS]:
        raise ValueError("External metric order or content changed")
    indexed = comparison.set_index("metric")
    if not (indexed.loc["Brier score", "ci95_high"] < 0):
        raise ValueError("Primary Brier interval no longer lies entirely below zero")
    if not (indexed.loc["Average precision", "ci95_low"] > 0):
        raise ValueError("Average-precision interval no longer lies entirely above zero")
    for metric in ["AUROC", "Log loss"]:
        if not (indexed.loc[metric, "ci95_low"] <= 0 <= indexed.loc[metric, "ci95_high"]):
            raise ValueError(f"{metric} interval no longer crosses zero")
    relative = indexed[["relative_ci95_low_percent", "relative_ci95_high_percent"]]
    if not (relative.loc["Brier score", "relative_ci95_low_percent"] > 0):
        raise ValueError("Relative Brier interval no longer favours Model B")
    if not (relative.loc["Average precision", "relative_ci95_low_percent"] > 0):
        raise ValueError("Relative average-precision interval no longer favours Model B")
    if len(hospitals) != 11 or int((hospitals["delta_brier_B_minus_A"] < 0).sum()) != 10:
        raise ValueError("Hospital-specific Brier directions no longer match the locked 10-of-11 result")
    row = summary.iloc[0]
    if not np.isclose(row["weighted_observed_event_rate"], 0.05626721033428403):
        raise ValueError("Weighted eICU event rate changed")


def panel_title(ax, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", pad=8, color=INK)


def draw_incremental_panel(ax, comparison: pd.DataFrame) -> None:
    panel_title(ax, "a", "Incremental performance in external validation")
    y = np.arange(len(comparison))[::-1]
    point = comparison["relative_change_favouring_B_percent"].to_numpy(dtype=float)
    lower = comparison["relative_ci95_low_percent"].to_numpy(dtype=float)
    upper = comparison["relative_ci95_high_percent"].to_numpy(dtype=float)
    ax.errorbar(
        point,
        y,
        xerr=np.vstack([point - lower, upper - point]),
        fmt="o",
        color=MODEL_B,
        ecolor=MODEL_B,
        markerfacecolor=MODEL_B,
        markeredgecolor=WHITE,
        markeredgewidth=0.65,
        markersize=5.8,
        elinewidth=1.55,
        capsize=2.8,
        capthick=1.2,
        zorder=3,
    )
    ax.axvline(0, color="#3F3F3F", linewidth=1.0, zorder=2)
    ax.set_xlim(-5.2, 52.0)
    ax.set_ylim(-0.55, len(comparison) - 0.45)
    ax.set_yticks([])
    for row, yi in zip(comparison.itertuples(index=False), y):
        ax.text(-0.035, yi + (0.08 if row.metric == "Brier score" else 0), row.metric,
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=7.4, weight="bold", color=INK, clip_on=False)
        if row.metric == "Brier score":
            ax.text(-0.035, yi - 0.20, "Primary", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=5.8, weight="bold",
                    color=MODEL_B, clip_on=False)
    ax.set_xlabel("Relative change favouring Model B (%)")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    clean_axes(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.99, 0.98, "Positive values favour Model B", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.2, color="#666666")


def clean_axes(ax, *, grid_axis="both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def draw_calibration_panel(ax, calibration: pd.DataFrame) -> None:
    panel_title(ax, "b", "External calibration")
    limit = 0.30
    ax.plot([0, limit], [0, limit], color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.0, label="Ideal", zorder=1)
    for model, color, linestyle in [("A", MODEL_A, "--"), ("B", MODEL_B, "-")]:
        group = calibration.loc[calibration["model"].eq(model)]
        ax.plot(
            group["mean_predicted"],
            group["observed_fraction"],
            color=color,
            linestyle=linestyle,
            marker="o",
            markeredgewidth=0,
            linewidth=1.6,
            markersize=3.7,
            label=f"Model {model}",
            zorder=3,
        )
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Weighted observed fraction")
    ax.xaxis.set_major_locator(MultipleLocator(0.10))
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    clean_axes(ax)
    ax.legend(loc="upper left", handlelength=2.4)


def draw_hospital_panel(ax, hospitals: pd.DataFrame) -> None:
    panel_title(ax, "c", "Hospital specific Brier score differences")
    ordered = hospitals.sort_values("delta_brier_B_minus_A").reset_index(drop=True)
    y = np.arange(len(ordered))
    values = ordered["delta_brier_B_minus_A"].to_numpy(dtype=float)
    colors = [MODEL_B if value < 0 else "#E69F00" for value in values]
    ax.barh(y, values, height=0.60, color=colors, edgecolor=WHITE, linewidth=0.45, zorder=3)
    ax.axvline(0, color=INK, linewidth=0.9, zorder=1)
    labels = [f"Hospital {int(hospital)} ({int(events)})" for hospital, events in zip(ordered["hospitalid"], ordered["event_patients"])]
    ax.set_yticks(y, labels)
    ax.set_xlim(values.min() - 0.0006, values.max() + 0.0007)
    ax.set_xlabel("Brier score difference (Model B − Model A)")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    clean_axes(ax, grid_axis="x")
    ax.text(0.995, 1.035, "Negative values favour Model B",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color="#666666", clip_on=False)


def build_figure(output_dir: Path):
    configure_style()
    predictions = pd.read_csv(SOURCE / "stage_e2_external_predictions.csv.gz")
    calibration = calibration_data(predictions)
    comparison = comparison_data()
    hospitals = pd.read_csv(EXTERNAL / "stage_e2_hospital_heterogeneity.csv")
    summary = calibration_summary(predictions)
    validate_inputs(predictions, comparison, hospitals, summary)

    comparison.to_csv(output_dir / "figure5_external_comparison_data.csv", index=False)
    calibration.to_csv(output_dir / "figure5_external_calibration_data.csv", index=False)
    summary.to_csv(output_dir / "figure5_external_calibration_summary.csv", index=False)
    hospitals.sort_values("delta_brier_B_minus_A").to_csv(output_dir / "figure5_external_hospital_data.csv", index=False)

    fig = plt.figure(figsize=(7.09, 5.75), facecolor=WHITE)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.38, 1.0], height_ratios=[1.0, 1.08])
    ax_increment = fig.add_subplot(grid[0, 0])
    ax_calibration = fig.add_subplot(grid[0, 1])
    ax_hospital = fig.add_subplot(grid[1, :])
    fig.subplots_adjust(left=0.175, right=0.985, bottom=0.095, top=0.945, wspace=0.38, hspace=0.47)
    draw_incremental_panel(ax_increment, comparison)
    draw_calibration_panel(ax_calibration, calibration)
    draw_hospital_panel(ax_hospital, hospitals)
    return fig


def generate_figure(output_dir: Path = FIGURES, stem: str = "figure5_external_validation", dpi: int = 300):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = build_figure(output_dir)
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
    parser.add_argument("--stem", default="figure5_external_validation")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_figure(args.output_dir, args.stem, args.dpi)


if __name__ == "__main__":
    main()
