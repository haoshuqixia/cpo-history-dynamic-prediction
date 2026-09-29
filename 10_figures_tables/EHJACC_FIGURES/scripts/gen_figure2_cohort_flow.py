#!/usr/bin/env python3
"""Generate Figure 2: dual-panel cohort and landmark selection flow.

The counts are locked study outputs and are validated internally before the
figure is drawn. The script writes a 300-DPI PNG and an editable vector PDF.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

INK = "#111111"
TEXT = "#1A1A1A"
MUTED = "#555555"
WHITE = "#FFFFFF"
LINE = "#5F5F5F"
LIGHT_GRAY = "#F6F6F6"


MIMIC_STAGES = [
    ("Eligible hourly landmarks", 2_056, 2_104, 38_613),
    ("After major MCS exclusion", 2_025, 2_071, 37_471),
    ("Common haemodynamic-history risk set", 1_992, 2_035, 36_851),
    ("6 h outcome observed", 1_891, 1_932, 26_530),
]

MIMIC_SPLITS = [
    ("Development ≤2019", 1_544, 1_572, 21_619),
    ("Temporal ≥2020", 265, 268, 3_416),
    ("Boundary ambiguous", 82, 92, 1_495),
]

EICU_STAGES = [
    ("Valid nurse-charted cardiac output", ["4,070 patients | 4,291 ICU stays", "74 hospitals | 104,789 CO records"]),
    ("MAP paired and CPO derived", ["100,173 CPO records"]),
    ("Strict eligible hourly landmarks", ["1,751 patients | 1,780 ICU stays", "32 hospitals | 22,136 landmarks"]),
    ("Final eligible risk set", ["1,179 patients | 32 hospitals", "12,971 landmarks"]),
    ("", ["840 patients | 27 hospitals", "7,753 landmarks"]),
    ("Confirmed/repeated low-CPO outcome", ["469 event-positive landmarks", "157 patients"]),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )


def validate_counts() -> None:
    final_patients, final_stays, final_landmarks = MIMIC_STAGES[-1][1:]
    assert sum(x[1] for x in MIMIC_SPLITS) == final_patients
    assert sum(x[2] for x in MIMIC_SPLITS) == final_stays
    assert sum(x[3] for x in MIMIC_SPLITS) == final_landmarks
    for previous, current in zip(MIMIC_STAGES, MIMIC_STAGES[1:]):
        assert current[1] <= previous[1]
        assert current[2] <= previous[2]
        assert current[3] <= previous[3]


def rounded(ax, x, y, w, h, *, face=WHITE, edge=LINE, lw=0.9, radius=0.012, zorder=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.003,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=MUTED, lw=1.0, mutation=10, zorder=3):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, *, size=8, weight="normal", color=TEXT, ha="center", va="center", linespacing=1.15, zorder=5):
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        linespacing=linespacing,
        zorder=zorder,
    )


def panel_frame(ax, key, title_value):
    rounded(ax, 0.008, 0.008, 0.984, 0.984, face=WHITE, edge=LINE, lw=0.9, radius=0.015)
    text(ax, 0.035, 0.952, key, size=14, weight="bold", color=INK, ha="left")
    text(ax, 0.092, 0.952, title_value, size=10.7, weight="bold", color=INK, ha="left")
    ax.plot([0.030, 0.970], [0.910, 0.910], color=LINE, lw=0.65, zorder=2)


def stage_box(ax, x, y, w, h, title_value, count_lines, *, edge=LINE, face=WHITE, title_size=8.2, body_size=7.5):
    rounded(ax, x, y, w, h, face=face, edge=edge, lw=0.95, radius=0.018)
    body = "\n".join(count_lines)
    if title_value:
        text(ax, x + w / 2, y + h * 0.72, title_value, size=title_size, weight="bold", linespacing=1.10)
        text(ax, x + w / 2, y + h * 0.31, body, size=body_size, color=TEXT, linespacing=1.18)
    else:
        text(ax, x + w / 2, y + h * 0.50, body, size=body_size, color=TEXT, linespacing=1.18)


def mimic_count_lines(stage):
    _, patients, stays, landmarks = stage
    return [f"{patients:,} patients | {stays:,} ICU stays", f"{landmarks:,} landmarks"]


def draw_mimic_panel(ax) -> None:
    panel_frame(ax, "a", "MIMIC-IV cohort and landmark selection")
    x, w, h = 0.16, 0.68, 0.112
    ys = [0.765, 0.585, 0.405, 0.225]
    for index, (stage, y) in enumerate(zip(MIMIC_STAGES, ys)):
        # After the initial cohort label, the arrow states the filtering rule;
        # retained-count boxes deliberately contain numbers only.
        stage_box(ax, x, y, w, h, stage[0] if index == 0 else "", mimic_count_lines(stage), edge=LINE, face=WHITE)

    arrow_labels = [
        "Exclude major MCS active\nat landmark",
        "Require common mPAP and SvO$_2$ history*",
        "Require observed 6 h outcome",
    ]
    for upper_y, lower_y, label_value in zip(ys, ys[1:], arrow_labels):
        gap_mid = (upper_y + lower_y + h) / 2
        arrow(ax, (0.50, upper_y), (0.50, lower_y + h), color=MUTED, lw=1.0, mutation=9)
        text(ax, 0.535, gap_mid, label_value, size=6.9, color=MUTED, ha="left", linespacing=1.10)

    branch_y = 0.174
    centers = [0.175, 0.500, 0.825]
    ax.plot([0.50, 0.50], [ys[-1], branch_y], color=MUTED, lw=0.9, zorder=2)
    ax.plot([centers[0], centers[-1]], [branch_y, branch_y], color=MUTED, lw=0.9, zorder=2)
    for center in centers:
        arrow(ax, (center, branch_y), (center, 0.145), color=MUTED, lw=0.9, mutation=8)

    split_specs = [
        (0.025, "Development\n≤2019", MIMIC_SPLITS[0], WHITE, None),
        (0.350, "Temporal\n≥2020", MIMIC_SPLITS[1], WHITE, None),
        (0.675, "Boundary ambiguous", MIMIC_SPLITS[2], LIGHT_GRAY, "Excluded from primary split"),
    ]
    for sx, title_value, split, face, note in split_specs:
        rounded(ax, sx, 0.026, 0.300, 0.116, face=face, edge=LINE, lw=0.9, radius=0.012)
        text(ax, sx + 0.150, 0.112, title_value, size=7.25, weight="bold", linespacing=1.05)
        text(ax, sx + 0.150, 0.067, f"{split[1]:,} patients\n{split[3]:,} landmarks", size=6.65, linespacing=1.10)
        if note:
            text(ax, sx + 0.150, 0.039, note, size=5.85, color=MUTED)


def draw_eicu_panel(ax) -> None:
    panel_frame(ax, "b", "eICU external validation cohort")
    x, w, h = 0.13, 0.74, 0.095
    ys = [0.790, 0.655, 0.520, 0.375, 0.230, 0.085]
    for (title_value, count_lines), y in zip(EICU_STAGES, ys):
        stage_box(
            ax,
            x,
            y,
            w,
            h,
            title_value,
            count_lines,
            edge=LINE,
            face=WHITE,
            title_size=7.75,
            body_size=6.85,
        )
    arrow_labels = [None, None, "Apply prior-episode and MCS rules", "Require observed 6 h outcome", None]
    for upper_y, lower_y, label_value in zip(ys, ys[1:], arrow_labels):
        arrow(ax, (0.50, upper_y), (0.50, lower_y + h), color=MUTED, lw=1.0, mutation=9)
        if label_value:
            gap_mid = (upper_y + lower_y + h) / 2
            text(ax, 0.535, gap_mid, label_value, size=6.45, color=MUTED, ha="left")


def build_figure():
    validate_counts()
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 7.6), facecolor=WHITE)
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.012, top=0.988, wspace=0.025)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("auto")
        ax.axis("off")
    draw_mimic_panel(axes[0])
    draw_eicu_panel(axes[1])
    return fig


def generate_figure(output_dir: Path = FIGURES, stem: str = "figure2_cohort_flow", dpi: int = 300):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, facecolor=WHITE)
    fig.savefig(png_path, dpi=dpi, facecolor=WHITE)
    plt.close(fig)
    print(pdf_path)
    print(png_path)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    parser.add_argument("--stem", default="figure2_cohort_flow")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_figure(args.output_dir, args.stem, args.dpi)


if __name__ == "__main__":
    main()
