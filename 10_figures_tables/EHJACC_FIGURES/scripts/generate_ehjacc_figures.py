#!/usr/bin/env python3
"""Generate the EHJ-ACVC graphical abstract and four locked main figures.

This script does not refit a model or alter any study estimate.  It imports the
frozen plotting modules, changes only the panel letters to upper case, and
exports submission-ready PDF, PNG, and TIFF files.  The graphical abstract is
drawn deterministically from the locked manuscript results.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE = PACKAGE_DIR.parents[1]
SOURCE_PACKAGE = (
    WORKSPACE
    / "PAC_CPO_VSCODE_RERUN_20260909"
    / "PAC_CPO_COMPLETE_FIGURE_TABLE_PACKAGE_20260909"
)
SOURCE_SCRIPTS = SCRIPT_DIR
OUTPUT_DIR = PACKAGE_DIR
SOURCE_DATA_DIR = PACKAGE_DIR / "source_data"
MPLCONFIGDIR = PACKAGE_DIR / ".mplconfig"

MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


WHITE = "#FFFFFF"
INK = "#14213D"
TEXT = "#263238"
MUTED = "#5F6B75"
MODEL_A = "#7A7A7A"
MODEL_B = "#0072B2"
AMBER = "#E69F00"
BLUE_BG = "#EAF4FB"
BLUE_EDGE = "#82BFE3"
GRAY_BG = "#F3F5F6"
GRAY_EDGE = "#B8C2C8"
PINK_BG = "#FCEEF2"
PINK_EDGE = "#E8A0B4"
PURPLE_BG = "#F1ECFA"
PURPLE_EDGE = "#B9A4DF"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.unicode_minus": True,
        }
    )


def load_module(name: str, filename: str):
    path = SOURCE_SCRIPTS / filename
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load plotting module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rounded(ax, x, y, w, h, *, face=WHITE, edge=GRAY_EDGE, lw=1.0, radius=0.018, zorder=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=MUTED, lw=1.2, mutation=11, zorder=4):
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


def label(ax, x, y, value, *, size=9, weight="normal", color=TEXT, ha="center", va="center", linespacing=1.18, zorder=5):
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


def export_figure(
    fig,
    stem: str,
    *,
    png_dpi: int = 600,
    tiff_dpi: int = 600,
    exact_canvas: bool = False,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bbox = None if exact_canvas else "tight"
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches=bbox, facecolor=WHITE)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=png_dpi, bbox_inches=bbox, facecolor=WHITE)
    fig.savefig(
        OUTPUT_DIR / f"{stem}.tiff",
        dpi=tiff_dpi,
        bbox_inches=bbox,
        facecolor=WHITE,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def build_graphical_abstract():
    """Draw an 18.0 cm by 12.5 cm horizontal graphical abstract."""
    width_in = 18.0 / 2.54
    height_in = 12.5 / 2.54
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    label(
        ax,
        0.5,
        0.955,
        "Recent CPO history improves short-term prediction beyond the latest CPO",
        size=12.3,
        weight="bold",
        color=INK,
    )
    ax.plot([0.035, 0.965], [0.915, 0.915], color="#D6DEE3", lw=0.9)

    y0, h = 0.205, 0.675
    left = (0.025, y0, 0.285, h)
    middle = (0.335, y0, 0.315, h)
    right = (0.675, y0, 0.300, h)
    rounded(ax, *left, face=BLUE_BG, edge=BLUE_EDGE, lw=1.0)
    rounded(ax, *middle, face=WHITE, edge=GRAY_EDGE, lw=1.0)
    rounded(ax, *right, face=PURPLE_BG, edge=PURPLE_EDGE, lw=1.0)

    # Left: clinical setting and the central physiological question.
    label(ax, 0.1675, 0.825, "Clinical setting\nand question", size=9.2, weight="bold", color=INK)
    label(ax, 0.1675, 0.752, "PAC-monitored ICU patients", size=7.0, weight="bold")
    rounded(ax, 0.060, 0.665, 0.215, 0.060, face=WHITE, edge=BLUE_EDGE, lw=0.9, radius=0.012)
    label(ax, 0.1675, 0.695, "Latest CPO ≥ 0.60 W", size=8.3, weight="bold", color=MODEL_B)

    # Mini-history plot: two histories converge at the same current CPO.
    xs = [0.075, 0.125, 0.180, 0.235, 0.272]
    higher = [0.625, 0.590, 0.606, 0.535, 0.505]
    lower = [0.460, 0.470, 0.452, 0.485, 0.505]
    ax.plot(xs, higher, color=MODEL_A, lw=2.0, marker="o", ms=3.2, zorder=5)
    ax.plot(xs, lower, color=AMBER, lw=2.0, marker="o", ms=3.2, zorder=5)
    ax.axvline(0.272, ymin=0.315, ymax=0.635, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
    label(ax, 0.073, 0.646, "−4 h", size=8.0, color=MUTED, ha="left")
    label(ax, 0.272, 0.646, "Landmark", size=8.0, color=MUTED, ha="right")
    label(ax, 0.1675, 0.405, "Different recent histories\ncan converge on a similar\ncurrent CPO", size=7.1, weight="bold")
    rounded(ax, 0.055, 0.230, 0.225, 0.125, face=WHITE, edge=BLUE_EDGE, lw=0.9, radius=0.012)
    label(ax, 0.1675, 0.292, "Does the preceding 4 h add\ninformation beyond the\nlatest CPO?", size=7.0, weight="bold", color=INK)

    arrow(ax, (0.312, 0.545), (0.330, 0.545), color=MUTED, lw=1.3)

    # Middle: direct nested comparison and outcome.
    label(ax, 0.4925, 0.825, "Nested model\ncomparison", size=9.2, weight="bold", color=INK)
    rounded(ax, 0.370, 0.685, 0.245, 0.105, face=GRAY_BG, edge=GRAY_EDGE, lw=1.0, radius=0.014)
    label(ax, 0.392, 0.752, "Model A", size=8.8, weight="bold", color=MODEL_A, ha="left")
    label(ax, 0.392, 0.704, "Latest CPO + clinical and\nobservation context", size=7.7, ha="left")
    label(ax, 0.4925, 0.650, "versus", size=8.2, weight="bold", color=MUTED)
    rounded(ax, 0.370, 0.510, 0.245, 0.115, face=BLUE_BG, edge=BLUE_EDGE, lw=1.1, radius=0.014)
    label(ax, 0.392, 0.585, "Model B", size=8.8, weight="bold", color=MODEL_B, ha="left")
    label(ax, 0.392, 0.542, "Model A + recent 4 h\nCPO history", size=7.7, ha="left")
    arrow(ax, (0.4925, 0.500), (0.4925, 0.440), color=MODEL_B, lw=1.4)
    rounded(ax, 0.365, 0.285, 0.255, 0.145, face=PINK_BG, edge=PINK_EDGE, lw=1.0, radius=0.014)
    label(ax, 0.4925, 0.390, "Future 6 h outcome", size=8.7, weight="bold", color=INK)
    label(ax, 0.4925, 0.338, "Two CPO values < 0.60 W\nseparated by 0.5–3 h", size=7.8)

    arrow(ax, (0.652, 0.545), (0.670, 0.545), color=MUTED, lw=1.3)

    # Right: locked results and transportability.
    label(ax, 0.825, 0.825, "Key findings", size=9.4, weight="bold", color=INK)
    rounded(ax, 0.710, 0.590, 0.230, 0.190, face=WHITE, edge=BLUE_EDGE, lw=1.0, radius=0.014)
    label(ax, 0.730, 0.746, "Temporal MIMIC-IV", size=8.8, weight="bold", color=MODEL_B, ha="left")
    label(ax, 0.730, 0.700, "265 patients\n3,416 landmarks", size=7.1, ha="left")
    label(ax, 0.730, 0.660, "ΔBrier  −0.0017", size=8.4, weight="bold", color=INK, ha="left")
    label(ax, 0.730, 0.620, "ΔAUROC  +0.044", size=8.4, weight="bold", color=INK, ha="left")

    rounded(ax, 0.710, 0.330, 0.230, 0.215, face=WHITE, edge=PURPLE_EDGE, lw=1.0, radius=0.014)
    label(ax, 0.730, 0.510, "Frozen-model eICU", size=8.8, weight="bold", color="#6B4CB3", ha="left")
    label(ax, 0.730, 0.468, "external validation", size=8.1, weight="bold", color="#6B4CB3", ha="left")
    label(ax, 0.730, 0.425, "840 patients\n27 hospitals", size=7.1, ha="left")
    label(ax, 0.730, 0.380, "ΔBrier  −0.00135", size=8.4, weight="bold", color=INK, ha="left")
    label(ax, 0.730, 0.345, "No retraining/recalibration", size=6.2, color=MUTED, ha="left")

    # Take-home band.
    rounded(ax, 0.025, 0.055, 0.950, 0.105, face=MODEL_B, edge=MODEL_B, lw=0, radius=0.016)
    label(
        ax,
        0.5,
        0.108,
        "Recent CPO history adds short-term information beyond the latest CPO\nand reproduces across databases.",
        size=8.4,
        weight="bold",
        color=WHITE,
    )
    return fig


def patch_uppercase_panel_labels(module, function_name: str) -> None:
    original = getattr(module, function_name)

    def uppercase(ax, letter, title):
        return original(ax, str(letter).upper(), title)

    setattr(module, function_name, uppercase)


def build_main_figures() -> None:
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    cohort = load_module("ehj_cohort", "gen_figure2_cohort_flow.py")
    patch_uppercase_panel_labels(cohort, "panel_frame")
    export_figure(cohort.build_figure(), "Figure_1")

    temporal = load_module("ehj_temporal", "gen_figure3_incremental_value.py")
    patch_uppercase_panel_labels(temporal, "panel_title")
    export_figure(temporal.build_figure(), "Figure_2")

    interpretation = load_module("ehj_interpretation", "gen_figure4_clinical_interpretability.py")
    patch_uppercase_panel_labels(interpretation, "panel_title")
    export_figure(interpretation.build_figure(SOURCE_DATA_DIR), "Figure_3")

    external = load_module("ehj_external", "gen_figure5_external_validation.py")
    patch_uppercase_panel_labels(external, "panel_title")
    export_figure(external.build_figure(SOURCE_DATA_DIR), "Figure_4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["all", "graphical-abstract", "main-figures"],
        default="all",
        help="Choose which figure set to generate.",
    )
    args = parser.parse_args()

    configure_style()
    if args.only in {"all", "graphical-abstract"}:
        export_figure(
            build_graphical_abstract(),
            "Graphical_Abstract",
            png_dpi=600,
            tiff_dpi=600,
            exact_canvas=True,
        )
    if args.only in {"all", "main-figures"}:
        build_main_figures()
    print(f"EHJ-ACVC figures generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
