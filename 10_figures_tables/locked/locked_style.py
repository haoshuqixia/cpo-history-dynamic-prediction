#!/usr/bin/env python3
"""Reference-inspired figure previews; frozen estimates are never refitted.

Run with the existing benchmark Python environment. Input modules and private
data remain in the independently reproduced, relocated work directory. This
file writes only new figures and aggregate/layout QA into this new directory.
No patient-level CSV is exported. Reference artwork is not copied.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import zipfile

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ["CPO_WORK_DIR"]).expanduser().resolve()
PACKAGE = WORK / "CPO_SUBMISSION_SUPPLEMENT_20260917" / "EHJACC_FIGURES"
REPORT = Path(os.environ["CPO_REPORT_DIR"]).resolve()
OUTPUT = REPORT / "png"
VECTOR = REPORT / "vector"
PREVIEW = REPORT / "previews"
os.environ.setdefault("MPLCONFIGDIR", str(REPORT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

WHITE = "#FFFFFF"
INK = "#202A30"
SLATE = "#3B5565"
PROCESS = "#648A9B"
BLUE = "#236E99"
GREY = "#737E85"
AMBER = "#C28B43"
MAROON = "#7C2745"
ROSE = "#A96570"
SAGE = "#5D8469"
PALE = "#EDF3F5"
LINE = "#9DAAB0"
QA = {}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 10,
        "axes.titleweight": "normal", "axes.edgecolor": "#646E73",
        "axes.linewidth": 0.75, "axes.labelcolor": INK,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "legend.fontsize": 7.4, "legend.frameon": False,
        "figure.facecolor": WHITE, "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE, "savefig.bbox": None,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": True, "axes.spines.top": True,
        "axes.spines.right": True,
    })


def module(name, path, plotting_only=False):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if plotting_only:
        # The empirical figure is drawn from previously frozen plot sources.
        # Its optional reconstruction import needs raw-data environment vars;
        # omit that unused import rather than re-accessing raw patient files.
        tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
        tree.body=[node for node in tree.body if not (isinstance(node,ast.ImportFrom)
            and node.module=="run_stage0_mechanism_audit")]
        exec(compile(tree,str(path),"exec"),mod.__dict__)
    else:
        spec.loader.exec_module(mod)
    return mod


def box(ax, x, y, w, h, face=WHITE, edge=LINE, radius=2, lw=0.7):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=1)
    ax.add_patch(p)
    return p


def txt(ax, x, y, value, size=10, color=INK, weight="normal", ha="center", va="center"):
    value=str(value).replace("SvO₂", "SvO$_2$")
    return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, linespacing=1.17, zorder=6)


def arrow(ax, p, q, color=SLATE, lw=1.3, scale=10):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=scale,
        color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=3))


def database_icon(ax, cx, cy, color, w=10, h=8):
    from matplotlib.patches import Ellipse
    ax.add_patch(Rectangle((cx-w/2, cy-h/2), w, h, facecolor=color, edgecolor=WHITE, lw=.6, zorder=4))
    for yy in [cy-h/2, cy, cy+h/2]:
        ax.add_patch(Ellipse((cx, yy), w, 2.5, facecolor=color, edgecolor=WHITE, lw=.6, zorder=5))


def monitor_icon(ax, cx, cy):
    box(ax, cx-5, cy-4, 10, 8, face=WHITE, edge=WHITE, radius=.6)
    ax.plot(np.array([-4,-2,-1,0,1,2,4])+cx,
            np.array([0,0,2,-2,1,0,0])+cy, color=BLUE, lw=1.2, zorder=5)


def signed(v, precision):
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):.{precision}f}"




def panel_title(ax, letter, title):
    title = title.replace("Hospital specific", "Hospital-specific")
    # A separate bold letter avoids making long titles visually heavy.
    ax.text(-.08,1.055,str(letter).upper(),transform=ax.transAxes,
            fontsize=12,fontweight="bold",ha="left",va="bottom",color=INK,clip_on=False)
    ax.set_title(title,loc="left",fontsize=9.3,fontweight="normal",pad=12,color=INK)


def clean_axes(ax, *, grid_axis="both"):
    # Reference statistical panels use full rectangular axes and no grid.
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(.7); s.set_color("#687178")
    ax.grid(False); ax.set_axisbelow(True)
    ax.tick_params(direction="out",length=2.6,width=.65)


def style_module(mod):
    mod.configure_style=style
    for attr,value in {"MODEL_A":GREY,"MODEL_B":BLUE,"MODEL_B_LIGHT":"#89B5CA",
        "INK":INK,"REFERENCE":"#A7AFB4","GRID":"#E4E9EC"}.items():
        if hasattr(mod,attr): setattr(mod,attr,value)
    if hasattr(mod,"panel_title"): mod.panel_title=panel_title
    if hasattr(mod,"clean_axes"): mod.clean_axes=clean_axes


def cohort_figure():
    mod=module("ref_cohort",PACKAGE/"scripts/gen_figure2_cohort_flow.py")
    mod.configure_style=style
    # Straight grey boxes, no overall panel frames: like the reference flow.
    def straight(ax,x,y,w,h,**kw):
        p=Rectangle((x,y),w,h,facecolor=kw.get("face",WHITE),
                    edgecolor=kw.get("edge",LINE),linewidth=.65,zorder=kw.get("zorder",1))
        ax.add_patch(p); return p
    def header(ax,key,title):
        mod.text(ax,.02,.958,str(key).upper(),size=14,weight="bold",ha="left",color=INK)
        mod.text(ax,.078,.958,title,size=8.6,weight="normal",ha="left",color=INK)
    mod.rounded=straight; mod.panel_frame=header
    original_text=mod.text
    def flow_text(ax,x,y,value,**kw):
        if value=="Require common mPAP and SvO$_2$ history*": value="Require common mPAP\nand SvO$_2$ history*"
        if value=="Apply prior-episode and MCS rules": value="Apply prior-episode\nand MCS rules"
        if value=="Require observed 6 h outcome": value="Require observed\n6 h outcome"
        if value=="82 patients\n1,495 landmarks": y+=.014
        if value=="Excluded from primary split":
            value="Excluded from\nprimary split"; y+=.006; kw["size"]=6.5
        original_text(ax,x,y,value,**kw)
    mod.text=flow_text
    fig=mod.build_figure()
    fig.set_size_inches(7.25,7.25)
    QA["cohort_counts"]={"external_landmarks":7753,"development_landmarks":21619,"temporal_landmarks":3416}
    return fig


def temporal_figure():
    mod=module("ref_temporal",PACKAGE/"scripts/gen_figure3_incremental_value.py")
    style_module(mod)
    fig=mod.build_figure()
    # A/B points are paired values in a display table, not a common metric axis.
    QA["temporal_panel_A"]="Labelled A-to-B display; connector lengths do not encode effect size. Raw paired deltas and CIs retained."
    return fig


def interpretation_figure():
    mod=module("ref_interpretation",PACKAGE/"scripts/gen_figure4_clinical_interpretability.py",plotting_only=True)
    style_module(mod)
    mod.HISTORY_HIGHER=GREY; mod.HISTORY_LOWER=AMBER
    mod.GROUP_COLORS={mod.GROUP_ORDER[0]:GREY,mod.GROUP_ORDER[1]:AMBER}
    mod.THRESHOLD=MAROON
    # Reuse the already reproduced private plot sources. Selection is unchanged.
    source=PACKAGE/"source_data"
    selected=pd.read_csv(source/"figure4_selected_landmarks.csv")
    histories=pd.read_csv(source/"figure4_individual_history_data.csv")
    summary=pd.read_csv(source/"figure4_history_summary.csv")
    timing=pd.read_csv(source/"figure4_alert_timing_data.csv")
    utility=pd.read_csv(source/"figure4_alert_utility_threshold_005.csv").iloc[0]
    style()
    fig=plt.figure(figsize=(7.7,5.65))
    grid=fig.add_gridspec(2,2,width_ratios=[1.22,.78],height_ratios=[1.08,.92])
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])]
    fig.subplots_adjust(left=.10,right=.982,bottom=.095,top=.935,wspace=.38,hspace=.55)
    mod.draw_histories(axes[0],selected,histories,summary)
    mod.draw_risk(axes[1],selected)
    for text in axes[1].texts:
        if "Illustrative subset" in text.get_text():
            text.set_text("Illustrative subset: latest CPO 0.70–0.74 W")
    mod.draw_alert_timing(axes[2],timing,utility)
    QA["illustrative_subset"]={"rows":len(selected),"groups":selected.groupby("history_group").size().to_dict(),
        "rule":"Unchanged frozen selection; no selection by predicted risk.",
        "timing_medians":timing[["lead_to_first_low_h","lead_to_confirmation_h"]].median().to_dict()}
    return fig


def external_figure():
    mod=module("ref_external",PACKAGE/"scripts/gen_figure5_external_validation.py")
    style_module(mod); style()
    predictions=pd.read_csv(mod.SOURCE/"stage_e2_external_predictions.csv.gz")
    calibration=mod.calibration_data(predictions); comparison=mod.comparison_data()
    hospitals=pd.read_csv(mod.EXTERNAL/"stage_e2_hospital_heterogeneity.csv")
    summary=mod.calibration_summary(predictions)
    mod.validate_inputs(predictions,comparison,hospitals,summary)
    fig=plt.figure(figsize=(7.5,5.9))
    grid=fig.add_gridspec(2,2,width_ratios=[1.38,1],height_ratios=[1,1.08])
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])]
    fig.subplots_adjust(left=.17,right=.983,bottom=.095,top=.94,wspace=.44,hspace=.50)
    mod.draw_incremental_panel(axes[0],comparison)
    mod.draw_calibration_panel(axes[1],calibration)
    mod.draw_hospital_panel(axes[2],hospitals)
    # Descriptive reverse-direction centre uses muted amber, not red.
    for p in axes[2].patches:
        if p.get_width()>0: p.set_facecolor(AMBER)
    QA["external"]={"landmarks":len(predictions),"favourable_hospitals":int((hospitals.delta_brier_B_minus_A<0).sum()),
        "hospital_count":len(hospitals),"relative_interval":"Paired absolute CI rescaled by the fixed Model A point estimate; not a bootstrap CI of the ratio."}
    return fig


def table_canvas(w,h):
    style(); fig=plt.figure(figsize=(w/25.4,h/25.4))
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,w),ylim=(0,h)); ax.axis("off")
    return fig,ax


def table_one():
    data=pd.read_csv(WORK/"submission_crosscheck/Table_1_recomputed.csv",dtype=str).fillna("")
    fig,ax=table_canvas(235,208)
    txt(ax,7,199,"Table 1",11,MAROON,"bold",ha="left")
    txt(ax,29,199,"Study sample and baseline characteristics",11,INK,"bold",ha="left")
    ax.plot([7,228],[193,193],color=MAROON,lw=1)
    xs=[7,137,179,221]
    txt(ax,7,187,"Characteristic",9,INK,"bold",ha="left")
    for x,v in zip(xs[1:],["MIMIC-IV\ndevelopment","MIMIC-IV\ntemporal","eICU\nexternal"]):
        txt(ax,x,186,v,9,INK,"bold",ha="right")
    ax.plot([7,228],[179,179],color=MAROON,lw=.8,ls=(0,(1,2)))
    y=174.5
    for row in data.itertuples(index=False,name=None):
        if row[0]=="__SECTION__":
            txt(ax,7,y,row[1],9,MAROON,"bold",ha="left")
        else:
            txt(ax,7,y,row[0],8.7,INK,ha="left")
            for x,val in zip(xs[1:],row[1:]): txt(ax,x,y,str(val),8.7,INK,ha="right")
        y-=5.3
    ax.plot([7,228],[y+2,y+2],color=MAROON,lw=1)
    txt(ax,7,y-2,"Continuous variables are median (IQR); categorical variables are n (%).",8,INK,ha="left",va="top")
    txt(ax,7,y-6.5,"Patient and landmark denominators are stated separately.",8,INK,ha="left",va="top")
    txt(ax,7,y-11,"CPO, cardiac power output; IABP, intra-aortic balloon pump; ICU, intensive care unit;",8,INK,ha="left",va="top")
    txt(ax,7,y-15.5,"mPAP, mean pulmonary artery pressure; SvO₂, mixed venous oxygen saturation.",8,INK,ha="left",va="top")
    QA["table_1"]={"rows":len(data),"source":"Table_1_recomputed.csv","manuscript_modified":False}
    return fig


def table_two():
    data=pd.read_csv(WORK/"submission_crosscheck/Table_2_recomputed.csv",dtype=str).fillna("")
    fig,ax=table_canvas(210,147)
    txt(ax,7,138,"Table 2",11,MAROON,"bold",ha="left")
    txt(ax,29,138,"Temporal and external performance",11,INK,"bold",ha="left")
    ax.plot([7,203],[132,132],color=MAROON,lw=1)
    y=126
    for cohort,start in [("Temporal MIMIC-IV evaluation",1),("Frozen-model eICU external validation",4)]:
        txt(ax,7,y,cohort,10,MAROON,"bold",ha="left")
        y-=8
        xs=[7,74,101,119]
        for x,head in zip(xs,["Metric","Model A","Model B","Difference (B − A), 95% CI"]):
            txt(ax,x,y,head,8.5,INK,"bold",ha="right" if x in xs[1:3] else "left")
        ax.plot([7,203],[y-4,y-4],color=MAROON,lw=.8,ls=(0,(1,2)))
        y-=9
        for row in data.itertuples(index=False,name=None):
            label=str(row[0]).replace("‡"," (primary)")
            txt(ax,7,y,label,8.7,INK,"bold" if "Brier" in label else "normal",ha="left")
            txt(ax,74,y,str(row[start]),8.7,INK,ha="right")
            txt(ax,101,y,str(row[start+1]),8.7,INK,ha="right")
            txt(ax,119,y,str(row[start+2]),8.1,INK,ha="left")
            y-=5.5
        y-=4
    ax.plot([7,203],[y+3,y+3],color=MAROON,lw=1)
    txt(ax,7,y,"Differences and paired 95% CIs are from frozen outputs; no new model fitting or bootstrap.",8,INK,ha="left",va="top")
    txt(ax,7,y-4.5,"Negative differences favour Model B for Brier score and log loss; positive values favour Model B",8,INK,ha="left",va="top")
    txt(ax,7,y-9,"for AUROC and average precision.",8,INK,ha="left",va="top")
    QA["table_2"]={"source":"Table_2_recomputed.csv","external_values":"Final discharge-boundary R1, not older manuscript rounded values","manuscript_modified":False}
    return fig


def supplement_figures():
    mod=module("ref_supp",WORK/"CPO_SUBMISSION_SUPPLEMENT_20260917/build_submission_supplement_r1.py")
    mod.BLUE=BLUE; mod.GREY=GREY; mod.PURPLE="#647D8D"
    def set_supp_style(): style(); return plt
    mod.set_mpl=set_supp_style
    def save_supp(fig,stem):
        # Correct letters, not data, and make clean full reference-style axes.
        for ax in fig.axes:
            clean_axes(ax)
            if "S1" in stem:
                if "Predicted probability" in ax.get_xlabel():
                    ax.set_xlabel("Predicted probability of\noutcome observation",fontsize=7.7)
                else:
                    ax.set_xlabel("Stabilised observation weight\n(log scale)",fontsize=7.7)
            if "S3" in stem:
                for text in ax.texts:
                    if "\n(" in text.get_text():
                        text.set_bbox(dict(facecolor=WHITE,edgecolor="none",pad=1))
            for text in ax.texts:
                if text.get_text() in list("abcdef"):
                    text.set_text(text.get_text().upper()); text.set_fontsize(11)
        if "S1" in stem:
            ax=fig.axes[-1]
            ax.legend(loc="upper right",bbox_to_anchor=(.99,.83),fontsize=7,frameon=False)
        if "S3" in stem:
            for letter,ax in zip("ABCD",fig.axes):
                ax.text(-.10,1.08,letter,transform=ax.transAxes,fontsize=11,weight="bold",clip_on=False)
        short={"figure_S1_observation_weight_diagnostics":"Figure_S1",
            "figure_S2_robustness_forest":"Figure_S2",
            "figure_S3_model_c_incremental_value":"Figure_S3"}[stem]
        export(fig,short)
        return OUTPUT/f"{short}.png"
    mod.save_figure=save_supp
    mod.fig_s1(mod.prepare_weight_data()); mod.fig_s2(mod.sensitivity_rows()); mod.fig_s3()


def export(fig,stem,exact=False):
    for d in [OUTPUT,VECTOR,PREVIEW]: d.mkdir(parents=True,exist_ok=True)
    # Retain legibility at the nominal double-column width rather than letting
    # inherited tiny annotation sizes survive the visual restyling.
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text() and text.get_fontsize()<7:
            text.set_fontsize(7)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    # GA/table canvas must be exact; statistical figures retain outside labels.
    bbox=None if exact else "tight"
    outside=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text(): continue
        b=t.get_window_extent(renderer=renderer)
        if exact and (b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5):
            outside.append(t.get_text())
    if outside: raise ValueError(f"Text outside {stem}: {outside}")
    fig.savefig(OUTPUT/f"{stem}.png",dpi=600,bbox_inches=bbox,facecolor=WHITE)
    fig.savefig(VECTOR/f"{stem}.pdf",bbox_inches=bbox,facecolor=WHITE)
    fig.savefig(VECTOR/f"{stem}.svg",bbox_inches=bbox,facecolor=WHITE)
    plt.close(fig)
    with Image.open(OUTPUT/f"{stem}.png") as im:
        QA[stem]={"pixels":list(im.size),"dpi":list(im.info.get("dpi",[])),"sha256":hashlib.sha256((OUTPUT/f"{stem}.png").read_bytes()).hexdigest(),"out_of_canvas_text":outside}
        im.thumbnail((1900,1600)); im.save(PREVIEW/f"{stem}.png")
    print(f"Generated {stem}",flush=True)






def main():
    export(cohort_figure(), "Figure_1")
    export(temporal_figure(), "Figure_2")
    export(interpretation_figure(), "Figure_3")
    export(external_figure(), "Figure_4")
    export(table_one(), "Table_1", exact=True)
    export(table_two(), "Table_2", exact=True)
    supplement_figures()
    (REPORT / "figures_tables_QA.json").write_text(json.dumps(QA, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__": main()
