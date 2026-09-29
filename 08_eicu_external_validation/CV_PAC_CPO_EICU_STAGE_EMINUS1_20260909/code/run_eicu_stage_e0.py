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
WORK = ROOT / "work"
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"
sys.path.insert(0, str(ROOT / "code"))
from run_eicu_stage_eminus1 import open_inner  # noqa: E402

MCS_RE = re.compile(
    r"\becmo\b|extracorporeal membrane|\bimpella\b|ventricular assist|\blvad\b|\brvad\b|tandem.?heart",
    re.I,
)


def event_time(times: np.ndarray, values: np.ndarray, left: float, right: float) -> float | None:
    low = times[(times > left) & (times <= right) & (values < 0.60)]
    if len(low) < 2:
        return None
    gaps = np.diff(low)
    pos = np.flatnonzero((gaps >= 0.5) & (gaps <= 3.0))
    return float(low[pos[0] + 1]) if len(pos) else None


def first_mcs_offsets(sids: set[int]) -> tuple[dict[int, float], dict]:
    first = {}; raw = Counter(); scanned = Counter()
    with open_inner("treatment") as handle:
        for row in csv.DictReader(handle):
            scanned["treatment"] += 1
            try: sid = int(row["patientunitstayid"]); offset = float(row["treatmentoffset"])
            except (ValueError, TypeError): continue
            if sid not in sids: continue
            text = row["treatmentstring"]
            if MCS_RE.search(text):
                first[sid] = min(first.get(sid, np.inf), offset / 60)
                raw[f"treatment|{text}"] += 1
    with open_inner("note") as handle:
        for row in csv.DictReader(handle):
            scanned["note"] += 1
            try: sid = int(row["patientunitstayid"]); offset = float(row["noteoffset"])
            except (ValueError, TypeError): continue
            if sid not in sids: continue
            text = " ".join([row["notetype"], row["notepath"], row["notevalue"], row["notetext"]])
            if MCS_RE.search(text):
                first[sid] = min(first.get(sid, np.inf), offset / 60)
                raw["note|MCS match"] += 1
    return first, {"rows_scanned": dict(scanned), "matched_records": int(sum(raw.values())), "matched_raw_terms": dict(raw), "stays_with_mcs_evidence": len(first)}


def main() -> None:
    structural = pd.read_csv(WORK / "structural_legal_landmarks.csv.gz")
    legal = structural[structural.analysis.eq("nurse__physiological_pac")].copy()
    cpo = pd.read_csv(WORK / "cpo_nurse.csv.gz")
    patients = pd.read_csv(WORK / "candidate_patient_metadata.csv.gz")
    sids = set(legal.sid.astype(int))
    first_mcs, mcs_audit = first_mcs_offsets(sids)
    cpo_by = {
        int(sid): (g.offset.to_numpy(float) / 60, g.available.to_numpy(float) / 60, g.cpo.to_numpy(float))
        for sid, g in cpo[cpo.sid.isin(sids)].sort_values("offset").groupby("sid")
    }

    rows = []
    for r in legal.itertuples(index=False):
        times, available, values = cpo_by[int(r.sid)]
        landmark = float(r.landmark_h)
        prior = event_time(times, values, -np.inf, landmark)
        if prior is not None:
            continue
        mcs = first_mcs.get(int(r.sid), np.inf)
        if mcs <= landmark:
            continue
        end = min(landmark + 6, mcs)
        event = event_time(times, values, landmark, end)
        future = times[(times > landmark) & (times <= landmark + 6)]
        dense_negative = False
        if event is None and mcs >= landmark + 6 and len(future) >= 4 and future[-1] >= landmark + 5:
            safe_hist = times[(times > landmark - 4) & (times <= landmark) & (available <= landmark)]
            chain = np.r_[safe_hist[-1], future]
            dense_negative = bool(np.diff(chain).max() <= 2)
        observed = event is not None or dense_negative
        rows.append({
            "sid": int(r.sid), "uniquepid": r.uniquepid, "hospitalid": int(r.hospitalid),
            "landmark_h": int(r.landmark_h), "outcome_observed_6h": observed,
            "event_6h": event is not None, "event_time_h": event,
            "future_mcs_censor_h": mcs if landmark < mcs < landmark + 6 else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "stage_e0_landmarks.csv.gz", index=False, compression="gzip")
    observed = out[out.outcome_observed_6h].copy()
    events = out[out.event_6h].copy()
    event_by_hospital = events.groupby("hospitalid").agg(
        event_landmarks=("event_6h", "size"), event_patients=("uniquepid", "nunique"), event_stays=("sid", "nunique")
    ).reset_index().sort_values("event_patients", ascending=False)
    event_by_hospital.to_csv(RESULTS / "stage_e0_event_distribution_by_hospital.csv", index=False)
    total_event_patients = int(events.uniquepid.nunique())
    patient_hospital = events[["uniquepid", "hospitalid"]].drop_duplicates()
    hospital_patient_counts = patient_hospital.groupby("hospitalid").uniquepid.nunique().sort_values(ascending=False)
    event_hospitals = int(len(hospital_patient_counts))
    if total_event_patients >= 100 and event_hospitals >= 5:
        verdict = "STRONG_GO_FORMAL_EXTERNAL_VALIDATION"
    elif total_event_patients >= 50 and event_hospitals >= 5:
        verdict = "GO_FORMAL_EXTERNAL_VALIDATION"
    elif total_event_patients >= 30 and event_hospitals >= 3:
        verdict = "SUPPORTING_TRANSPORT_ONLY"
    else:
        verdict = "NO_GO_EXTERNAL_VALIDATION"
    summary = {
        "decision": verdict,
        "legal_landmarks_after_prior_event_and_active_mcs_exclusion": int(len(out)),
        "legal_stays": int(out.sid.nunique()), "legal_patients": int(out.uniquepid.nunique()),
        "legal_hospitals": int(out.hospitalid.nunique()),
        "observed_landmarks": int(len(observed)), "observed_fraction": float(out.outcome_observed_6h.mean()),
        "event_landmarks": int(len(events)), "event_stays": int(events.sid.nunique()),
        "event_patients": total_event_patients, "event_hospitals": event_hospitals,
        "hospitals_with_ge5_event_patients": int((hospital_patient_counts >= 5).sum()),
        "top1_event_patient_share": float(hospital_patient_counts.iloc[0] / total_event_patients) if total_event_patients else None,
        "top2_event_patient_share": float(hospital_patient_counts.iloc[:2].sum() / total_event_patients) if total_event_patients else None,
        "landmarks_censored_by_future_mcs": int(out.future_mcs_censor_h.notna().sum()),
    }
    (RESULTS / "stage_e0_decision.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (AUDIT / "stage_e0_mcs_audit.json").write_text(json.dumps(mcs_audit, indent=2, ensure_ascii=False), encoding="utf-8")
    report = f"""# eICU Stage E0 事件与结局可观察性门

**裁决：{verdict}**

- 排除既往事件和地标时 major MCS 后合法地标：{summary['legal_landmarks_after_prior_event_and_active_mcs_exclusion']:,}
- 患者/医院：{summary['legal_patients']:,}/{summary['legal_hospitals']}
- 结局可观察地标：{summary['observed_landmarks']:,}（{summary['observed_fraction']:.1%}）
- 事件地标：{summary['event_landmarks']:,}
- 事件患者：{summary['event_patients']:,}
- 事件医院：{summary['event_hospitals']}
- 至少5名事件患者的医院：{summary['hospitals_with_ge5_event_patients']}
- 最大一家/两家医院的事件患者占比：{summary['top1_event_patient_share']:.1%}/{summary['top2_event_patient_share']:.1%}
- 随访期 major MCS 截尾地标：{summary['landmarks_censored_by_future_mcs']}

本阶段未运行任何模型预测。通过事件门只代表样本量与医院分布足以启动冻结模型外部验证；CO测量机制异质性和共享预测变量跨库映射仍需在性能分析前冻结。
"""
    (ROOT / "REPORT_STAGE_E0_ZH.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
