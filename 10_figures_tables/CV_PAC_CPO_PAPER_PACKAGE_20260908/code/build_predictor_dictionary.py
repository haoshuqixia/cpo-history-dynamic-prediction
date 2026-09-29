#!/usr/bin/env python3
"""Create the TRIPOD+AI predictor dictionary and missingness table."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
STAGE1A = BASE / "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908"
STAGE1B = BASE / "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908"
OUT = ROOT / "tables"
sys.path.insert(0, str(STAGE1B / "code"))
from run_stage1b import BLOCKS, CATEGORICAL, DEV, TEMP, DRUG_COLS, SHARED_NUMERIC  # noqa: E402


def markdown(frame: pd.DataFrame) -> str:
    vals = frame.fillna("").astype(str)
    head = "| " + " | ".join(vals.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(vals.columns)) + " |"
    rows = ["| " + " | ".join(r.replace("|", "\\|") for r in row) + " |" for row in vals.to_numpy()]
    return "\n".join([head, rule, *rows]) + "\n"


def describe(name: str) -> tuple[str, str, str]:
    if name == "age": return "Age at ICU admission", "years", "MIMIC-IV derived age"
    if name == "gender": return "Recorded administrative sex", "category", "hosp/patients"
    if name == "first_careunit": return "First ICU care unit", "category", "icu/icustays"
    if name == "landmark_h": return "Elapsed time from ICU admission to landmark", "hours", "derived from ICU intime"
    if name == "landmark_h_sq": return "Squared landmark hour", "hours squared", "derived"
    if name.startswith("n_") and name.endswith("_history_4h"):
        raw_axis = name[2:].replace("_history_4h", "")
        axis = {"cpo": "CPO", "mpap": "mPAP", "svo2": "SvO2"}[raw_axis]
        return f"Number of {axis} measurements in (L-4 h, L]", "count", "derived from chart events"
    if name.endswith("_recency_h"):
        raw_axis = name.replace("_recency_h", "")
        axis = {"cpo": "CPO", "mpap": "mPAP", "svo2": "SvO2"}[raw_axis]
        return f"Time since most recent {axis} measurement at L", "hours", "derived from chart events"
    if name == "iabp_active_at_landmark": return "Intra-aortic balloon pump active at L", "binary", "procedure/device evidence available by L"
    if name in DRUG_COLS:
        drug = name.removeprefix("drug_").removesuffix("_active_at_landmark")
        return f"{drug.capitalize()} infusion active at L", "binary", "ICU input events"
    axis, suffix = name.split("_", 1)
    axis_label = {"cpo": "CPO", "mpap": "mPAP", "svo2": "SvO2"}.get(axis, axis)
    units = {"cpo": "W", "mpap": "mmHg", "svo2": "%"}.get(axis, "")
    labels = {
        "current": f"Most recent {axis_label} in (L-4 h, L]; CPO eligibility additionally requires recency <=1 h",
        "mean_4h": f"Mean {axis_label} in (L-4 h, L]",
        "min_4h": f"Minimum {axis_label} in (L-4 h, L]",
        "max_4h": f"Maximum {axis_label} in (L-4 h, L]",
        "sd_4h": f"Population standard deviation of {axis_label} in (L-4 h, L]",
        "slope_4h": f"Linear time slope of {axis_label} in (L-4 h, L]",
        "delta_4h": f"Last minus first {axis_label} in (L-4 h, L]",
        "span_4h": f"Elapsed time from first to last {axis_label} in (L-4 h, L]",
    }
    if suffix in {"slope_4h"}: units += "/h"
    if suffix == "span_4h": units = "hours"
    source = {
        "cpo": "CCO item 224842 and concurrent MAP items 220052/225312/220181; CPO=CCO×MAP/451",
        "mpap": "mPAP item 220061",
        "svo2": "SvO2 item 223772",
    }.get(axis, "derived")
    return labels.get(suffix, name), units, source


def main() -> None:
    f = pd.read_csv(STAGE1A / "results/LANDMARK_FEATURES_V1.csv.gz", dtype={"stay_id": str})
    common = f.primary_eligible_at_landmark.astype(bool) & f.primary_outcome_observed_6h.astype(bool) & f.model_bc_common_risk_set.astype(bool)
    dev = f.loc[common & f.interval_conservative_split.eq(DEV)]
    temp = f.loc[common & f.interval_conservative_split.eq(TEMP)]
    block_lookup = {}
    for name in BLOCKS["A"] + CATEGORICAL: block_lookup[name] = "shared/Model A"
    for name in set(BLOCKS["B"]) - set(BLOCKS["A"]): block_lookup[name] = "Model B addition"
    for name in set(BLOCKS["C"]) - set(BLOCKS["B"]): block_lookup[name] = "Model C addition"
    rows = []
    for name in BLOCKS["C"] + CATEGORICAL:
        definition, unit, source = describe(name)
        rows.append({
            "predictor": name,
            "block": block_lookup[name],
            "definition_and_timing": definition,
            "unit": unit,
            "source_or_derivation": source,
            "development_missing": f"{dev[name].isna().sum()} ({dev[name].isna().mean()*100:.1f}%)",
            "temporal_missing": f"{temp[name].isna().sum()} ({temp[name].isna().mean()*100:.1f}%)",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tableS3_predictor_dictionary_and_missingness.csv", index=False)
    (OUT / "tableS3_predictor_dictionary_and_missingness.md").write_text(markdown(out), encoding="utf-8")


if __name__ == "__main__":
    main()
