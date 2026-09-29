#!/usr/bin/env python3
"""Blue/gold graphical abstract with reviewed text and approved final R1 data.

Uses the frozen V2 numerical renderer without fitting or recalibration.
The reviewer-requested old Brier rounding was rejected after user confirmation.
Earlier images and Word are left unchanged.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil

HERE=Path(__file__).resolve().parent
V2=HERE
REPORT=Path(os.environ["CPO_REPORT_DIR"]).resolve()
os.environ.setdefault("MPLCONFIGDIR",str(REPORT/".mplconfig"))
spec=importlib.util.spec_from_file_location("frozen_ga",V2/"ga_base.py")
v2=importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.text import Text
import numpy as np
from PIL import Image

NAVY="#12243B"
BLUE="#0868D8"
AMBER="#F1B648"
PALE_BLUE="#EAF2FD"
WHITE="#FFFFFF"


def card_at(ax,x,y):
    matches=[p for p in ax.patches if isinstance(p,FancyBboxPatch)
             and np.isclose(p.get_x(),x) and np.isclose(p.get_y(),y)]
    assert len(matches)==1
    return matches[0]


def tiny_history_clock(ax,x,y):
    # Non-quantitative clock only: no invented CPO traces or implied effects.
    ax.add_patch(Circle((x,y),2.25,facecolor=BLUE,edgecolor=WHITE,linewidth=.85,zorder=4))
    ax.plot([x,x,x+1.15],[y+1.3,y,y-.65],color=WHITE,lw=.85,
            solid_capstyle="round",zorder=5)


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    r=v2.r; r.INK=NAVY; r.SLATE=NAVY; r.BLUE=BLUE; r.PROCESS=BLUE
    fig,qa=v2.build()
    canvas,roc,bars,interval=fig.axes
    frozen_lines={id(ax):[(np.asarray(l.get_xdata()).copy(),np.asarray(l.get_ydata()).copy())
                         for l in ax.lines] for ax in (roc,interval)}
    frozen_widths=[p.get_width() for p in bars.patches]
    replacements={
        "Latest CPO + clinical\nand observation context":
            ("Latest CPO + shared\nclinical/observation context",(79.5,97.1),9.4,NAVY),
        "Model A + recent 4 h\nCPO history":
            ("Model A + recent 4-h\nCPO history",(145,97.1),9.7,WHITE),
        "Future 6 h outcome":("Future 6-h outcome",(43,76),9.5,NAVY),
        "Two CPO values <0.60 W, separated by 0.5–3 h":
            ("Confirmed/repeated low-CPO episode:\ntwo CPO values <0.60 W, 0.5–3 h apart",(125,76),8.7,NAVY),
        "Temporal MIMIC-IV":("Temporal validation in MIMIC-IV",(46,66),9.4,WHITE),
        "Frozen-model eICU validation":("Frozen-model eICU external validation",(134,66.6),9.0,WHITE),
        "840 patients | 27 hospitals | 7,753 landmarks":
            ("840 patients | 27 hospitals | 7,753 landmarks",(134,59.7),8.0,r.GREY),
        "Primary ΔBrier: −0.00135":("Primary ΔBrier: −0.00135",(134,55.4),9.4,NAVY),
        "ΔBrier 95% CI: −0.00246 to −0.00025":
            ("ΔBrier (B − A) 95% CI: −0.00246 to −0.00025",(134,33.6),8.0,NAVY),
        "Conclusion":("Conclusion",(90,16.2),10.5,WHITE),
        "Recent CPO history adds information beyond the latest CPO;\nBrier improvement is reproduced in frozen-model external validation.":
            ("Recent CPO history improves short-term prediction beyond the latest CPO;\nthe primary Brier-score improvement is reproduced in frozen-model external validation.",
             (90,8.9),9.3,WHITE)
    }
    for t in list(canvas.texts):
        s=t.get_text()
        if s=="Recent CPO history improves six-hour prediction\nbeyond the latest CPO":
            t.set_visible(False)
            r.txt(canvas,90,120.5,"Recent CPO history improves 6-h prediction",13,NAVY,"bold")
            r.txt(canvas,90,115.4,"beyond the latest CPO",13,BLUE,"bold")
        elif s.startswith("Δ = Model B − Model A."):
            t.set_visible(False)
        elif s in replacements:
            value,pos,size,color=replacements[s]
            t.set_text(value); t.set_position(pos); t.set_fontsize(size); t.set_color(color)
            if s.startswith("Recent CPO history adds"):
                t.set_linespacing(1.35)
        elif s.startswith("Primary ΔBrier:"):
            t.set_color(NAVY)
    subtitle=r.txt(canvas,134,63.4,"No retraining or recalibration",8.1,WHITE)
    subtitle.set_fontstyle("italic")
    outcome=card_at(canvas,5,71)
    outcome.set_facecolor(PALE_BLUE); outcome.set_edgecolor(PALE_BLUE)
    for x,y,h in [(5,64,5),(93,62,7)]:
        outer=card_at(canvas,x,22); outer.set_edgecolor(BLUE)
        header=Rectangle((x,y),82,h,facecolor=NAVY,edgecolor="none",zorder=2)
        header.set_clip_path(outer); canvas.add_patch(header)
    for cx,cy in [(46,57.4),(134,55.4)]:
        r.box(canvas,cx-29.5,cy-2.25,59,4.5,AMBER,AMBER,.8,0).set_zorder(2)
    tiny_history_clock(canvas,168,106)
    # Compact the right chart without changing bar widths or reported values.
    bars.set_position([112/180,43/125,56/180,8.5/125])
    for p in bars.patches:
        middle=p.get_y()+p.get_height()/2
        p.set_height(.75); p.set_y(middle-.375)
    bars.set_xlabel("Brier score",fontsize=8,labelpad=1.5)
    bottom=card_at(canvas,5,3); bottom.set_bounds(5,2.5,170,18.5)
    stripe=Rectangle((5,20.35),170,.65,facecolor=AMBER,edgecolor="none",zorder=2)
    stripe.set_clip_path(bottom); canvas.add_patch(stripe)
    for p in canvas.patches:
        if isinstance(p,FancyArrowPatch) and np.allclose(p.get_facecolor(),to_rgba("#3B5565")):
            p.set_color(NAVY)

    # Keep all frozen statistics, ROC/CI data coordinates and bar values exact.
    prior=json.loads((HERE/"expected_ga_values.json").read_text())
    for key in qa: assert qa[key]==prior[key],key
    for ax in (roc,interval):
        for l,(xx,yy) in zip(ax.lines,frozen_lines[id(ax)]):
            np.testing.assert_array_equal(l.get_xdata(),xx)
            np.testing.assert_array_equal(l.get_ydata(),yy)
    assert [p.get_width() for p in bars.patches]==frozen_widths
    assert [t.get_text() for t in bars.texts]==["0.04882","0.04747"]
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); outside=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text(): continue
        assert t.get_fontsize()>=8,t.get_text()
        b=t.get_window_extent(renderer)
        if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height: outside.append(t.get_text())
    assert not outside,outside
    stem=REPORT/"png/Graphical_Abstract"
    for ext in ("png","pdf","svg"):
        destination=stem.with_suffix(".png") if ext=="png" else REPORT/"vector"/f"Graphical_Abstract.{ext}"
        fig.savefig(destination,dpi=600,bbox_inches=None,facecolor=WHITE)
    plt.close(fig)
    with Image.open(stem.with_suffix(".png")) as im:
        qa["pixels"]=list(im.size); qa["dpi"]=list(im.info["dpi"])
        im.thumbnail((1900,1600)); im.save(REPORT/"previews/Graphical_Abstract.png")
    qa.update({"user_approved_final_R1_rounding":["0.04882","0.04747"],
        "review_old_rounding_not_applied":True,
        "frozen_numeric_values_ROC_CI_and_bar_widths_unchanged":True,
        "minimum_font_pt":8,"outside_canvas_text":outside,
        "sha256":hashlib.sha256(stem.with_suffix(".png").read_bytes()).hexdigest()})
    (REPORT/"graphical_abstract_QA.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
    print(stem.with_suffix(".png"))


if __name__=="__main__": main()
