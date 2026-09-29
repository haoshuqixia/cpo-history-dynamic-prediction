#!/usr/bin/env python3
"""Data-driven graphical abstract with real temporal ROC and external Brier.

Adds two result charts, using frozen predictions and paired intervals only.
No fitting, recalibration, bootstrap, or patient-level export is performed.
V1 figures and the manuscript are not changed.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os

HERE=Path(__file__).resolve().parent
V1=HERE
WORK=Path(os.environ["CPO_WORK_DIR"]).expanduser().resolve()
os.environ.setdefault("MPLCONFIGDIR",str(Path(os.environ["CPO_REPORT_DIR"])/".mplconfig"))
spec=importlib.util.spec_from_file_location("reference_style",V1/"locked_style.py")
r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)

import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import pandas as pd
from PIL import Image


def axes_mm(fig,x,y,w,h):
    return fig.add_axes([x/180,y/125,w/180,h/125])


def miniature_axes(ax):
    for spine in ax.spines.values():
        spine.set_color("#87969D"); spine.set_linewidth(.6)
    ax.tick_params(labelsize=8,direction="out",length=2,width=.6,pad=2)
    ax.grid(False)


def build():
    r.style()
    fig=plt.figure(figsize=(18/2.54,12.5/2.54))
    canvas=fig.add_axes([0,0,1,1]); canvas.set(xlim=(0,180),ylim=(0,125)); canvas.axis("off")
    r.txt(canvas,90,118,"Recent CPO history improves six-hour prediction\nbeyond the latest CPO",13,r.INK,"bold")

    # The compact top row keeps the nested comparison unambiguous.
    r.box(canvas,5,89,39,22,r.SLATE,r.SLATE,3,0)
    r.monitor_icon(canvas,11.5,103)
    r.txt(canvas,30,104,"PAC-monitored\nICU patients",8.4,r.WHITE,"bold")
    r.txt(canvas,24.5,93.7,"Latest CPO ≥0.60 W",9,r.WHITE)
    r.arrow(canvas,(44,100),(51,100))
    r.box(canvas,51,89,57,22,"#EEF0F1","#B3BFC5",2,.7)
    r.txt(canvas,79.5,106,"Model A",10.5,r.GREY,"bold")
    r.txt(canvas,79.5,97.1,"Latest CPO + clinical\nand observation context",9.6,r.INK)
    r.arrow(canvas,(108,100),(115,100),r.BLUE)
    r.box(canvas,115,89,60,22,r.PROCESS,r.PROCESS,3,0)
    r.txt(canvas,145,106,"Model B",10.5,r.WHITE,"bold")
    r.txt(canvas,145,97.1,"Model A + recent 4 h\nCPO history",10,r.WHITE,"bold")
    r.txt(canvas,90,85,"Developed in MIMIC-IV: 1,544 patients | 21,619 landmarks",8.5,r.GREY)

    r.box(canvas,5,71,170,10,r.SLATE,r.SLATE,2,0)
    r.txt(canvas,47,76,"Future 6 h outcome",10.2,r.WHITE,"bold")
    r.txt(canvas,122,76,"Two CPO values <0.60 W, separated by 0.5–3 h",9,r.WHITE)

    # Real result panels are redrawn from the same frozen inputs as Fig. 2/4.
    r.box(canvas,5,22,82,47,r.WHITE,r.PROCESS,3,.9)
    r.box(canvas,93,22,82,47,r.WHITE,r.PROCESS,3,.9)
    temporal_root=WORK/"CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908/results"
    predictions=pd.read_csv(temporal_root/"temporal_predictions.csv.gz")
    metrics=pd.read_csv(temporal_root/"temporal_metrics.csv")
    deltas=pd.read_csv(temporal_root/"paired_temporal_deltas.csv")
    td=deltas[(deltas.comparison=="B-A") & (deltas.metric=="brier")].iloc[0]
    assert len(predictions)==3416 and predictions.subject_id.nunique()==265
    r.txt(canvas,46,66,"Temporal MIMIC-IV",10.5,r.BLUE,"bold")
    r.txt(canvas,46,61.7,"265 patients | 3,416 landmarks",8.4,r.GREY)
    r.txt(canvas,46,57.4,f"Primary ΔBrier: {r.signed(td.estimate,6)}",9.4,r.BLUE,"bold")
    roc=axes_mm(fig,16,31,63,22.5)
    y=predictions.primary_outcome_6h.to_numpy(dtype=int)
    weights=predictions.ipow_stabilized_truncated.to_numpy(dtype=float)
    auc_values={}
    for model,color,ls in [("A",r.GREY,"--"),("B",r.BLUE,"-")]:
        p=predictions[f"prediction_{model}"].to_numpy(dtype=float)
        fpr,tpr,_=roc_curve(y,p,sample_weight=weights)
        value=metrics[(metrics.model==model)&(metrics.metric=="auroc")].iloc[0].estimate
        assert np.isclose(value,roc_auc_score(y,p,sample_weight=weights),atol=1e-12)
        auc_values[model]=float(value)
        roc.plot(fpr,tpr,color=color,ls=ls,lw=1.6,label=f"{model}: AUROC {value:.3f}",zorder=3)
    roc.plot([0,1],[0,1],color="#AAB6BC",ls=(0,(3,2)),lw=.7,zorder=1)
    roc.set(xlim=(0,1),ylim=(0,1.02))
    roc.set_xlabel("1 − specificity",fontsize=8,labelpad=2)
    roc.set_ylabel("Sensitivity",fontsize=8,labelpad=2)
    roc.set_xticks([0,.5,1]); roc.set_yticks([0,.5,1])
    miniature_axes(roc)
    roc.legend(loc="lower right",fontsize=8,handlelength=1.6,borderaxespad=.2,
        frameon=True,facecolor=r.WHITE,edgecolor="none",framealpha=.96,labelspacing=.22)

    external_root=WORK/"CPO_EICU_EXTERNAL_DISCHARGE_BOUNDARY_R1_V1_0/results"
    absolute=pd.read_csv(external_root/"stage_e2_absolute_metrics.csv")
    absolute=absolute[(absolute.analysis=="primary_ipow_floor_and_q01q99")&(absolute.metric=="brier")].set_index("model")
    intervals=pd.read_csv(external_root/"stage_e2_primary_comparison_ci.csv").set_index("metric")
    ed=intervals.loc["delta_brier_B_minus_A"]
    a=float(absolute.loc["A","estimate"]); b=float(absolute.loc["B","estimate"])
    assert np.isclose(b-a,float(ed.estimate),atol=1e-12)
    assert ed.ci95_low < ed.estimate < ed.ci95_high < 0
    r.txt(canvas,134,66,"Frozen-model eICU validation",10.1,r.BLUE,"bold")
    r.txt(canvas,134,61.7,"840 patients | 27 hospitals | 7,753 landmarks",8.1,r.GREY)
    r.txt(canvas,134,57.4,f"Primary ΔBrier: {r.signed(ed.estimate,5)}",9.4,r.BLUE,"bold")
    bars=axes_mm(fig,112,43,56,10.5)
    bars.barh([1,0],[a,b],height=.54,color=[r.GREY,r.BLUE])
    for value,yy in [(a,1),(b,0)]:
        bars.text(value-.002,yy,f"{value:.5f}",ha="right",va="center",color=r.WHITE,fontsize=8,weight="bold")
    bars.set(xlim=(0,.055),ylim=(-.5,1.5))
    bars.set_yticks([1,0],["Model A","Model B"])
    bars.set_xticks([0,.025,.05]); bars.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:.3f}"))
    bars.set_xlabel("Brier score",fontsize=8,labelpad=1.5)
    miniature_axes(bars)
    r.txt(canvas,134,33.6,f"ΔBrier 95% CI: {r.signed(ed.ci95_low,5)} to {r.signed(ed.ci95_high,5)}",8,r.INK)
    interval=axes_mm(fig,112,26,56,5)
    point=float(ed.estimate); lo=float(ed.ci95_low); hi=float(ed.ci95_high)
    interval.errorbar(point,0,xerr=[[point-lo],[hi-point]],fmt="o",color=r.BLUE,
        markersize=4.4,elinewidth=1.3,capsize=2.6,capthick=1.0,zorder=3)
    interval.axvline(0,color=r.SLATE,lw=.8,zorder=1)
    interval.set(xlim=(-.003,.0005),ylim=(-.8,.8)); interval.set_yticks([])
    interval.set_xticks([-.003,-.002,-.001,0])
    interval.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:"0" if np.isclose(v,0) else f"{v:.3f}"))
    miniature_axes(interval)
    interval.spines["left"].set_visible(False); interval.spines["right"].set_visible(False)
    interval.spines["top"].set_visible(False)
    r.txt(canvas,90,20,"Δ = Model B − Model A. Direct eICU application without retraining or recalibration.",8,r.GREY)
    r.box(canvas,5,3,170,15.5,r.SLATE,r.SLATE,3,0)
    r.txt(canvas,90,14.7,"Conclusion",10.5,r.WHITE,"bold")
    r.txt(canvas,90,8.6,"Recent CPO history adds information beyond the latest CPO;\nBrier improvement is reproduced in frozen-model external validation.",10.2,r.WHITE)
    qa={"temporal_landmarks":len(predictions),"temporal_patients":int(predictions.subject_id.nunique()),
        "temporal_auroc":auc_values,"temporal_delta_brier":float(td.estimate),
        "external_landmarks":7753,"external_patients":840,"external_hospitals":27,
        "external_brier_A":a,"external_brier_B":b,"external_delta_brier":point,
        "external_delta_brier_CI":[lo,hi],"plots":"Frozen weighted ROC; absolute external Brier; paired absolute Brier CI",
        "models_refitted":False,"V1_or_manuscript_modified":False}
    return fig,qa



