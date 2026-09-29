#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
import re
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"
WORK = ROOT / "work"
sys.path.insert(0, str(ROOT / "code"))
from run_eicu_stage_eminus1 import normalize_drug, open_inner  # noqa: E402

STOP_RE = re.compile(r"\b(?:off|stop(?:ped)?|hold|held|discontinu(?:e|ed))\b", re.I)


def state(drug_rate: str, infusion_rate: str) -> str:
    text = f"{drug_rate} {infusion_rate}".strip()
    if STOP_RE.search(text):
        return "inactive_explicit"
    values = []
    for value in [drug_rate, infusion_rate]:
        try: values.append(float(value))
        except (TypeError, ValueError): pass
    if not values:
        return "unparseable"
    return "active" if any(v > 0 for v in values) else "inactive_numeric"


def main() -> None:
    legal = pd.read_csv(RESULTS / "stage_e0_landmarks.csv.gz", usecols=["sid"])
    sids = set(legal.sid.astype(int))
    rows = []; raw_names = Counter(); parsing = Counter()
    with open_inner("infusionDrug") as handle:
        for row in csv.DictReader(handle):
            try: sid = int(row["patientunitstayid"]); offset = int(float(row["infusionoffset"]))
            except (TypeError, ValueError): continue
            if sid not in sids or not (0 <= offset <= 4320): continue
            drug = normalize_drug(row["drugname"])
            if drug is None: continue
            s = state(row["drugrate"], row["infusionrate"])
            parsing[s] += 1; raw_names[row["drugname"]] += 1
            rows.append((sid, drug, offset, s, row["drugname"], row["drugrate"], row["infusionrate"]))
    raw = pd.DataFrame(rows, columns=["sid", "drug", "offset", "state", "raw_name", "drug_rate", "infusion_rate"])
    raw.to_csv(WORK / "formal_cohort_infusion_records.csv.gz", index=False, compression="gzip")

    parseable = raw[raw.state.ne("unparseable")].copy()
    grouped = parseable.groupby(["sid", "drug", "offset"]).state.agg(lambda x: set(x))
    conflicts = grouped.map(lambda x: any(v == "active" for v in x) and any(v.startswith("inactive") for v in x))
    resolved = []
    for (sid, drug, offset), states in grouped.items():
        if "active" in states and not any(v.startswith("inactive") for v in states): value = "active"
        elif any(v.startswith("inactive") for v in states) and "active" not in states: value = "inactive"
        else: value = "unknown_conflict"
        resolved.append((sid, drug, offset, value))
    resolved = pd.DataFrame(resolved, columns=["sid", "drug", "offset", "state"]).sort_values(["sid", "drug", "offset"])
    resolved.to_csv(WORK / "formal_cohort_infusion_states.csv.gz", index=False, compression="gzip")
    resolved["gap_to_next_min"] = resolved.groupby(["sid", "drug"]).offset.shift(-1) - resolved.offset
    active = resolved[resolved.state.eq("active") & resolved.gap_to_next_min.notna()].copy()

    qlevels = [0, .25, .5, .75, .9, .95, .99, 1]
    summaries = []
    for drug, frame in [("ALL", active), *list(active.groupby("drug"))]:
        q = frame.gap_to_next_min.quantile(qlevels) if len(frame) else pd.Series(index=qlevels, dtype=float)
        summaries.append({
            "drug": drug, "active_records_with_next": int(len(frame)),
            **{f"gap_q{int(level*100):02d}_min": float(q.loc[level]) if level in q.index and pd.notna(q.loc[level]) else np.nan for level in qlevels},
            "fraction_next_within_60m": float((frame.gap_to_next_min <= 60).mean()) if len(frame) else np.nan,
            "fraction_next_within_120m": float((frame.gap_to_next_min <= 120).mean()) if len(frame) else np.nan,
            "fraction_next_within_240m": float((frame.gap_to_next_min <= 240).mean()) if len(frame) else np.nan,
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / "stage_e1a_medication_gap_audit.csv", index=False)
    overall_p90 = float(summary.loc[summary.drug.eq("ALL"), "gap_q90_min"].iloc[0])
    candidates = [x for x in [60, 120, 240] if x >= overall_p90]
    cap = min(candidates) if candidates else 240
    limited = not bool(candidates)
    decision = {
        "selected_carry_forward_cap_min": cap,
        "selection_rule": "smallest of 60/120/240 minutes at least overall p90 active-to-next-parseable-record gap",
        "overall_p90_gap_min": overall_p90,
        "limited_state_identifiability": limited,
        "mapped_rows": int(len(raw)), "parseable_rows": int(len(parseable)),
        "simultaneous_conflicting_offsets": int(conflicts.sum()),
        "parsing_states": dict(parsing), "distinct_raw_names": len(raw_names),
        "outcomes_read": False, "predictions_generated": False,
    }
    (AUDIT / "stage_e1a_medication_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    report = f"""# eICU Stage E1A 药物状态记录机制审计

- 映射记录：{len(raw):,}
- 可解析状态记录：{len(parseable):,}
- 同offset矛盾状态：{int(conflicts.sum()):,}
- active到下一条可解析记录间隔P90：{overall_p90:.0f}分钟
- 冻结carry-forward上限：**{cap}分钟**
- 状态识别受限标志：{limited}

选择只依据输注记录时间结构，没有读取低CPO结局或任何模型性能。
"""
    (ROOT / "REPORT_STAGE_E1A_ZH.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
