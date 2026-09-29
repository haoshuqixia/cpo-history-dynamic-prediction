#!/usr/bin/env python3
"""Generate Figure 4: recent CPO history, predicted risk, and alert timing.

The figure is descriptive and uses locked temporal predictions. It selects one
illustrative landmark per patient at a similar current CPO, reconstructs the
preceding four-hour CPO history, and summarizes alert timing at the prespecified
0.05 threshold. No model fitting or hypothesis testing is performed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]
FIGURES = ROOT / "figures"
STAGE0 = BASE / "CV_PAC_CPO_STAGE0_MECHANISM_20260908"
STAGE1A = BASE / "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908"
STAGE1B = BASE / "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908"
STAGE1D = BASE / "CV_PAC_CPO_STAGE1D_ALERT_UTILITY_20260908"
PREVIOUS = BASE / "STAGE_MINUS1A_CV_OBSERVABILITY_V1" / "work"

sys.path.insert(0, str(STAGE0 / "code"))
from run_stage0_mechanism_audit import build_axis_times  # noqa: E402


INK = "#171717"
HISTORY_HIGHER = "#7A7A7A"
HISTORY_LOWER = "#E69F00"
MODEL_B = "#0072B2"
MODEL_B_LIGHT = "#56B4E9"
THRESHOLD = "#A94442"
REFERENCE = "#9C9C9C"
GRID = "#E8E8E8"
WHITE = "#FFFFFF"
INFO_FILL = "#F6F6F6"

GROUP_ORDER = ["Higher recent CPO history", "Lower recent CPO history"]
GROUP_COLORS = {
    "Higher recent CPO history": HISTORY_HIGHER,
    "Lower recent CPO history": HISTORY_LOWER,
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.3,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.7,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#4D4D4D",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
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


def panel_title(ax, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", pad=8, color=INK)


def clean_axes(ax, *, grid_axis="both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def select_landmarks():
    predictions = pd.read_csv(
        STAGE1B / "results" / "temporal_predictions.csv.gz",
        dtype={"stay_id": str},
    )
    features = pd.read_csv(
        STAGE1A / "results" / "LANDMARK_FEATURES_V1.csv.gz",
        dtype={"stay_id": str},
    )
    keys = ["subject_id", "stay_id", "landmark_h", "interval_conservative_split"]
    selected_columns = ["cpo_current", "cpo_mean_4h", "cpo_min_4h", "cpo_max_4h"]
    merged = predictions.merge(
        features[keys + selected_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    candidates = merged.loc[
        merged["cpo_current"].between(0.70, 0.74) & merged["cpo_min_4h"].ge(0.60)
    ].copy()
    candidates["distance_to_0_72"] = (candidates["cpo_current"] - 0.72).abs()
    candidates = (
        candidates.sort_values(["subject_id", "distance_to_0_72", "stay_id", "landmark_h"])
        .drop_duplicates("subject_id")
    )
    lower_cut = candidates["cpo_mean_4h"].quantile(0.25)
    upper_cut = candidates["cpo_mean_4h"].quantile(0.75)
    candidates["history_group"] = np.select(
        [
            candidates["cpo_mean_4h"].le(lower_cut),
            candidates["cpo_mean_4h"].ge(upper_cut),
        ],
        ["Lower recent CPO history", "Higher recent CPO history"],
        default="Middle",
    )
    selected = candidates.loc[candidates["history_group"].isin(GROUP_ORDER)].copy()
    counts = selected.groupby("history_group")["subject_id"].nunique().to_dict()
    if counts != {"Higher recent CPO history": 23, "Lower recent CPO history": 23}:
        raise ValueError(f"Unexpected descriptive group sizes: {counts}")
    audit = {
        "purpose": "illustrative subset with similar latest CPO",
        "latest_cpo_lower_w": 0.70,
        "latest_cpo_upper_w": 0.74,
        "minimum_prior_cpo_w": 0.60,
        "per_patient_selection": "landmark closest to 0.72 W; ties by stay_id then landmark_h",
        "patients_after_one_landmark_per_patient": candidates["subject_id"].nunique(),
        "grouping_variable": "cpo_mean_4h",
        "lower_quartile_cut_w": lower_cut,
        "upper_quartile_cut_w": upper_cut,
        "higher_history_patients": counts["Higher recent CPO history"],
        "lower_history_patients": counts["Lower recent CPO history"],
        "prediction_or_outcome_used_for_selection": False,
    }
    return selected, audit


def reconstruct_histories(selected: pd.DataFrame):
    stays = pd.read_csv(
        PREVIOUS / "adult_stays.csv",
        dtype={"stay_id": str},
        parse_dates=["intime", "outtime"],
    )
    stays = stays.loc[stays["stay_id"].isin(set(selected["stay_id"]))]
    cpo, _, _ = build_axis_times(stays)
    by_stay = {stay_id: rows.sort_values("hours") for stay_id, rows in cpo.groupby("stay_id")}

    history_rows = []
    for row in selected.itertuples(index=False):
        observed = by_stay[row.stay_id]
        observed = observed.loc[
            observed["hours"].gt(row.landmark_h - 4) & observed["hours"].le(row.landmark_h),
            ["hours", "cpo"],
        ].copy()
        observed["relative_h"] = observed["hours"] - row.landmark_h
        for measurement in observed.itertuples(index=False):
            history_rows.append(
                {
                    "subject_id": row.subject_id,
                    "stay_id": row.stay_id,
                    "history_group": row.history_group,
                    "relative_h": measurement.relative_h,
                    "cpo": measurement.cpo,
                    "current_point": False,
                }
            )
        history_rows.append(
            {
                "subject_id": row.subject_id,
                "stay_id": row.stay_id,
                "history_group": row.history_group,
                "relative_h": 0.0,
                "cpo": row.cpo_current,
                "current_point": True,
            }
        )
    histories = pd.DataFrame(history_rows).sort_values(["history_group", "subject_id", "relative_h"])

    prior = histories.loc[~histories["current_point"]].copy()
    prior["time_bin"] = pd.cut(
        prior["relative_h"],
        bins=[-4.000001, -3, -2, -1, 0],
        labels=[-3.5, -2.5, -1.5, -0.5],
        include_lowest=True,
    )
    patient_bins = (
        prior.dropna(subset=["time_bin"])
        .groupby(["history_group", "subject_id", "time_bin"], observed=True)["cpo"]
        .median()
        .reset_index()
        .rename(columns={"time_bin": "relative_h"})
    )
    patient_bins["relative_h"] = patient_bins["relative_h"].astype(float)
    current = selected[["history_group", "subject_id", "cpo_current"]].rename(columns={"cpo_current": "cpo"})
    current["relative_h"] = 0.0
    patient_summary_source = pd.concat(
        [patient_bins, current[["history_group", "subject_id", "relative_h", "cpo"]]],
        ignore_index=True,
    )
    summary = (
        patient_summary_source.groupby(["history_group", "relative_h"])["cpo"]
        .agg(
            n="size",
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    return histories, summary


def load_alert_timing():
    details = pd.read_csv(STAGE1D / "results" / "event_pair_alert_details.csv.gz")
    timing = details.loc[
        details["model"].eq("B")
        & np.isclose(details["threshold"], 0.05)
        & details["detected"]
    ].drop_duplicates(["subject_id", "stay_id", "first_low_h", "confirming_low_h"])
    utility = pd.read_csv(STAGE1D / "results" / "alert_utility.csv")
    utility_row = utility.loc[utility["model"].eq("B") & np.isclose(utility["threshold"], 0.05)]
    if len(utility_row) != 1 or len(timing) != 68:
        raise ValueError("Locked Model B alert-timing results no longer match the expected threshold-0.05 analysis")
    return timing, utility_row.iloc[0]


def export_plot_data(selected, selection_audit, histories, history_summary, timing, utility_row, output_dir: Path) -> None:
    selected.to_csv(output_dir / "figure4_selected_landmarks.csv", index=False)
    pd.DataFrame([selection_audit]).to_csv(output_dir / "figure4_subset_selection_audit.csv", index=False)
    histories.to_csv(output_dir / "figure4_individual_history_data.csv", index=False)
    history_summary.to_csv(output_dir / "figure4_history_summary.csv", index=False)
    risk_summary = (
        selected.groupby("history_group")["prediction_B"]
        .agg(
            n="size",
            q1=lambda values: values.quantile(0.25),
            median="median",
            q3=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    risk_summary.to_csv(output_dir / "figure4_risk_summary.csv", index=False)
    timing.to_csv(output_dir / "figure4_alert_timing_data.csv", index=False)
    timing_summary = pd.DataFrame(
        [
            {
                "observed_endpoint": "First observed low CPO",
                "n_detected_event_pairs": len(timing),
                "q1_h": timing["lead_to_first_low_h"].quantile(0.25),
                "median_h": timing["lead_to_first_low_h"].median(),
                "q3_h": timing["lead_to_first_low_h"].quantile(0.75),
            },
            {
                "observed_endpoint": "Confirmed/repeated episode",
                "n_detected_event_pairs": len(timing),
                "q1_h": timing["lead_to_confirmation_h"].quantile(0.25),
                "median_h": timing["lead_to_confirmation_h"].median(),
                "q3_h": timing["lead_to_confirmation_h"].quantile(0.75),
            },
        ]
    )
    timing_summary.to_csv(output_dir / "figure4_alert_timing_summary.csv", index=False)
    pd.DataFrame([utility_row]).to_csv(output_dir / "figure4_alert_utility_threshold_005.csv", index=False)


def draw_histories(ax, selected: pd.DataFrame, histories: pd.DataFrame, summary: pd.DataFrame) -> None:
    panel_title(ax, "a", "Recent CPO histories at similar current CPO")
    ax.text(
        0.01,
        1.005,
        "Illustrative subset: one landmark per patient",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.15,
        color="#666666",
    )
    for group in GROUP_ORDER:
        color = GROUP_COLORS[group]
        group_histories = histories.loc[histories["history_group"].eq(group)]
        for _, patient_history in group_histories.groupby("subject_id"):
            ax.plot(
                patient_history["relative_h"],
                patient_history["cpo"],
                color=color,
                linewidth=0.55,
                alpha=0.12,
                zorder=1,
            )
        group_summary = summary.loc[summary["history_group"].eq(group)].sort_values("relative_h")
        x = group_summary["relative_h"].to_numpy(dtype=float)
        median = group_summary["median"].to_numpy(dtype=float)
        q1 = group_summary["q1"].to_numpy(dtype=float)
        q3 = group_summary["q3"].to_numpy(dtype=float)
        n = selected.loc[selected["history_group"].eq(group), "subject_id"].nunique()
        ax.fill_between(x, q1, q3, color=color, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(x, median, color=color, linewidth=2.0, marker="o", markersize=3.5, label=f"{group} (n={n})", zorder=4)
        current = group_summary.loc[np.isclose(group_summary["relative_h"], 0)]
        ax.scatter(current["relative_h"], current["median"], s=42, color=color, edgecolor=WHITE, linewidth=0.8, zorder=6)

    ax.axhline(0.60, color=THRESHOLD, linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
    ax.text(-3.95, 0.615, "Low CPO threshold, 0.60 W", color=THRESHOLD, fontsize=6.4, ha="left", va="bottom")
    ax.axvline(0, color="#555555", linestyle=(0, (3, 3)), linewidth=1.0, zorder=2)
    ax.text(-0.055, 1.425, "Landmark", color="#555555", fontsize=6.5, ha="right", va="bottom")
    ax.set_xlim(-4.05, 0.12)
    ax.set_ylim(0.56, 1.46)
    ax.set_xticks([-4, -3, -2, -1, 0])
    ax.yaxis.set_major_locator(MultipleLocator(0.20))
    ax.set_xlabel("Time relative to landmark (h)")
    ax.set_ylabel("CPO (W)")
    clean_axes(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.995), handlelength=2.5)


def draw_risk(ax, selected: pd.DataFrame) -> None:
    panel_title(ax, "b", "Predicted risk at similar current CPO")
    rng = np.random.default_rng(20260910)
    for index, group in enumerate(GROUP_ORDER):
        color = GROUP_COLORS[group]
        values = selected.loc[selected["history_group"].eq(group), "prediction_B"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.085, 0.085, size=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=17,
            color=color,
            alpha=0.62,
            edgecolor=WHITE,
            linewidth=0.4,
            zorder=3,
        )
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        ax.plot([index, index], [q1, q3], color=color, linewidth=3.0, solid_capstyle="round", zorder=5)
        ax.scatter([index], [median], marker="D", s=36, color=color, edgecolor=WHITE, linewidth=0.75, zorder=6)

    bracket_y = 0.345
    ax.plot([0, 0, 1, 1], [bracket_y - 0.008, bracket_y, bracket_y, bracket_y - 0.008], color=REFERENCE, linewidth=0.8)
    ax.text(
        0.5,
        bracket_y + 0.012,
        "Illustrative subset · similar latest CPO: 0.70–0.74 W",
        ha="center",
        va="bottom",
        fontsize=6.1,
        color="#666666",
    )
    ax.set_xlim(-0.48, 1.48)
    ax.set_ylim(0, 0.38)
    ax.set_xticks(
        [0, 1],
        ["Higher recent\nCPO history\n(n=23)", "Lower recent\nCPO history\n(n=23)"],
    )
    ax.set_ylabel("Model B predicted 6 h risk")
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    clean_axes(ax, grid_axis="y")
    summary_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=4.0, markerfacecolor="#777777", markeredgecolor=WHITE, label="Patient"),
        Line2D([], [], color="#777777", linewidth=3.0, label="IQR"),
        Line2D([], [], linestyle="none", marker="D", markersize=4.6, markerfacecolor="#777777", markeredgecolor=WHITE, label="Median"),
    ]
    ax.legend(handles=summary_handles, loc="upper left", bbox_to_anchor=(0.015, 0.86), fontsize=5.8, handlelength=1.5)


def draw_alert_timing(ax, timing: pd.DataFrame, utility: pd.Series) -> None:
    panel_title(ax, "c", "Alert timing relative to low CPO")
    rng = np.random.default_rng(20260910)
    rows = [
        ("First observed\nlow CPO", "lead_to_first_low_h", 1.0, MODEL_B_LIGHT),
        ("Confirmed/repeated\nepisode", "lead_to_confirmation_h", 0.0, MODEL_B),
    ]
    for label, column, y, color in rows:
        values = timing[column].to_numpy(dtype=float)
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(values, np.full(len(values), y) + jitter, s=10, color=color, alpha=0.18, edgecolor="none", zorder=2)
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        ax.plot([0, median], [y, y], color=color, linewidth=1.25, zorder=3)
        ax.plot([q1, q3], [y, y], color=color, linewidth=4.0, solid_capstyle="round", zorder=4)
        ax.scatter([median], [y], s=54, color=color, edgecolor=WHITE, linewidth=0.8, zorder=5)
        label_y = y - 0.18 if column == "lead_to_first_low_h" else y + 0.16
        label_va = "top" if column == "lead_to_first_low_h" else "bottom"
        ax.text(median, label_y, f"{median:.2f} h", color=color, fontsize=7.4, weight="bold", ha="center", va=label_va)

    ax.axvline(0, color=INK, linewidth=1.1, zorder=3)
    ax.text(0.03, 1.48, "First alert", fontsize=6.8, color=INK, weight="bold", ha="left", va="top")
    info = (
        "Prespecified threshold = 0.05\n"
        f"Event-pair sensitivity: {100 * utility.event_pair_sensitivity:.1f}%\n"
        f"Event-free landmark alert fraction: {100 * utility.event_free_landmark_alert_fraction:.1f}%"
    )
    ax.text(
        0.985,
        0.97,
        info,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.15,
        color="#4F4F4F",
        linespacing=1.28,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=INFO_FILL, edgecolor="#BDBDBD", linewidth=0.6),
        zorder=7,
    )
    ax.text(
        0.99,
        0.04,
        "Observed measurement times; not physiological onset",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#6A6A6A",
    )
    ax.set_xlim(-0.05, 6.20)
    ax.set_ylim(-0.48, 1.58)
    ax.set_yticks([1, 0], ["First observed\nlow CPO", "Confirmed/repeated\nepisode"])
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlabel("Time from first alert (h)")
    clean_axes(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    timing_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=3.5, markerfacecolor=MODEL_B_LIGHT, alpha=0.35, markeredgecolor="none", label="Observed event pair"),
        Line2D([], [], color="#777777", linewidth=4.0, label="IQR"),
        Line2D([], [], linestyle="none", marker="o", markersize=5.5, markerfacecolor="#777777", markeredgecolor=WHITE, label="Median"),
    ]
    ax.legend(handles=timing_handles, loc="upper left", bbox_to_anchor=(0.10, 0.995), ncol=3, fontsize=6.1, handlelength=1.8, columnspacing=1.2)


def build_figure(output_dir: Path):
    configure_style()
    selected, selection_audit = select_landmarks()
    histories, history_summary = reconstruct_histories(selected)
    timing, utility = load_alert_timing()
    export_plot_data(selected, selection_audit, histories, history_summary, timing, utility, output_dir)

    fig = plt.figure(figsize=(7.6, 5.55), facecolor=WHITE)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.22, 0.78], height_ratios=[1.08, 0.92])
    ax_history = fig.add_subplot(grid[0, 0])
    ax_risk = fig.add_subplot(grid[0, 1])
    ax_timing = fig.add_subplot(grid[1, :])
    fig.subplots_adjust(left=0.092, right=0.985, bottom=0.095, top=0.945, wspace=0.34, hspace=0.48)
    draw_histories(ax_history, selected, histories, history_summary)
    draw_risk(ax_risk, selected)
    draw_alert_timing(ax_timing, timing, utility)
    return fig


def generate_figure(output_dir: Path = FIGURES, stem: str = "figure4_trajectory_and_lead_time", dpi: int = 300):
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
    parser.add_argument("--stem", default="figure4_trajectory_and_lead_time")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_figure(args.output_dir, args.stem, args.dpi)


if __name__ == "__main__":
    main()
