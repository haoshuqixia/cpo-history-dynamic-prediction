#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
DEFENSE = BASE / "CV_PAC_CPO_REVIEWER_DEFENSE_20260908"
TABLES = ROOT / "tables"


def fmt_delta(row: pd.Series, digits: int) -> str:
    return f"{row.estimate:+.{digits}f} ({row.ci_low:+.{digits}f}, {row.ci_high:+.{digits}f})"


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.astype(str)
    head = "| " + " | ".join(values.columns) + " |"
    rule = "|" + "|".join(["---"] * len(values.columns)) + "|"
    body = ["| " + " | ".join(row) + " |" for row in values.to_numpy().tolist()]
    return "\n".join([head, rule, *body]) + "\n"


def robustness_table() -> None:
    r1_delta = pd.read_csv(DEFENSE / "stage_r1_storetime/results/paired_temporal_deltas.csv")
    r1_decision = json.loads((DEFENSE / "stage_r1_storetime/results/decision.json").read_text())
    r2_delta = pd.read_csv(DEFENSE / "stage_r2_robustness/results/paired_temporal_deltas.csv")
    r2_counts = pd.read_csv(DEFENSE / "stage_r2_robustness/results/analysis_counts.csv").set_index("analysis")
    analyses = [
        ("Primary common risk set", "charttime", "IPOW", "locked_common_weighted"),
        ("Expanded CPO-eligible risk set", "charttime", "IPOW", "expanded_cpo_only_weighted"),
        ("Primary common risk set, unweighted", "charttime", "Unit", "locked_common_unweighted"),
    ]
    rows = []
    for label, clock, weighting, key in analyses:
        count = r2_counts.loc[key]
        q = r2_delta[r2_delta.analysis.eq(key)].set_index("metric")
        rows.append({
            "analysis": label, "information clock": clock, "weighting": weighting,
            "landmarks": int(count.landmarks), "patients": int(count.patients),
            "event landmarks": int(count.event_landmarks), "event patients": int(count.event_patients),
            "B-A Brier (95% CI)": fmt_delta(q.loc["brier"], 6),
            "B-A AUROC (95% CI)": fmt_delta(q.loc["auroc"], 4),
            "B-A average precision (95% CI)": fmt_delta(q.loc["average_precision"], 4),
            "B-A log loss (95% CI)": fmt_delta(q.loc["log_loss"], 6),
        })
    q = r1_delta.set_index("metric")
    rows.append({
        "analysis": "Strict stored-haemodynamic common risk set", "information clock": "storetime <= L", "weighting": "Re-estimated IPOW",
        "landmarks": r1_decision["temporal_landmarks"], "patients": r1_decision["temporal_patients"],
        "event landmarks": r1_decision["temporal_event_landmarks"], "event patients": r1_decision["temporal_event_patients"],
        "B-A Brier (95% CI)": fmt_delta(q.loc["brier"], 6),
        "B-A AUROC (95% CI)": fmt_delta(q.loc["auroc"], 4),
        "B-A average precision (95% CI)": fmt_delta(q.loc["average_precision"], 4),
        "B-A log loss (95% CI)": fmt_delta(q.loc["log_loss"], 6),
    })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "tableS5_robustness_analyses.csv", index=False)
    (TABLES / "tableS5_robustness_analyses.md").write_text(markdown_table(out), encoding="utf-8")


def measurement_table() -> None:
    lag = json.loads((DEFENSE / "results/source_storage_lag_summary.json").read_text())
    device = json.loads((DEFENSE / "results/device_and_native_cpo_audit.json").read_text())
    retention = pd.read_csv(DEFENSE / "results/landmark_timesafe_retention.csv")
    rows = []
    labels = {"derived_cpo": "Derived CPO", "mpap": "mPAP", "svo2": "SvO2"}
    for key, label in labels.items():
        x = lag[key]
        rows.append({"domain": "Storage lag", "measure": f"{label}: median [IQR], min", "result": f"{x['q50']:.0f} [{x['q25']:.0f}, {x['q75']:.0f}]"})
        rows.append({"domain": "Storage lag", "measure": f"{label}: >60 min", "result": f"{x['fraction_gt_60_min'] * 100:.1f}%"})
    rows += [
        {"domain": "Charting process", "measure": "CPO chart interval: median [IQR], h", "result": f"{device['chart_interval_h_quantiles']['0.5']:.2f} [{device['chart_interval_h_quantiles']['0.25']:.2f}, {device['chart_interval_h_quantiles']['0.75']:.2f}]"},
        {"domain": "Charting process", "measure": "Consecutive identical CCO values", "result": f"{device['consecutive_identical_cco_fraction'] * 100:.2f}%"},
        {"domain": "Charting process", "measure": "Consecutive identical derived CPO values", "result": f"{device['consecutive_identical_cpo_fraction'] * 100:.2f}%"},
        {"domain": "Native CPO item 229896", "measure": "Stays in complete source", "result": str(device['native_cpo_total_stays'])},
        {"domain": "Native CPO item 229896", "measure": "Overlap with final-analysis stays", "result": str(device['native_cpo_final_analysis_overlap_stays'])},
    ]
    for r in retention.itertuples(index=False):
        label = "Development" if "development" in r.split else "Temporal evaluation"
        rows.append({"domain": "Strict information clock", "measure": f"{label}: retained landmarks", "result": f"{r.timesafe_rows:,}/{r.locked_rows:,} ({r.timesafe_retention * 100:.1f}%)"})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "tableS6_measurement_information_clock_qa.csv", index=False)
    (TABLES / "tableS6_measurement_information_clock_qa.md").write_text(markdown_table(out), encoding="utf-8")


def protocol_table() -> None:
    count = pd.read_csv(DEFENSE / "stage_r3_sensitivity_counts/corrected_landmark_sensitivity_counts.csv")
    q = count[(count.population == "primary_eligible_common_axes") & (count.split == "temporal")]
    lookup = {(r.threshold_w, r.horizon_h): r for r in q.itertuples(index=False)}
    rows = []
    for threshold, horizon in [(0.60, 6), (0.60, 12), (0.50, 6), (0.50, 12)]:
        r = lookup[(threshold, horizon)]
        status = "Primary endpoint count" if (threshold, horizon) == (0.60, 6) else "Prespecified sensitivity count"
        rows.append({
            "declared item": f"Repeated CPO <{threshold:.2f} W within {horizon} h",
            "original scope": status,
            "corrected temporal legal-landmark result": f"{r.event_positive_landmarks} event landmarks; {r.event_positive_patients} event patients among {r.eligible_landmarks} eligible landmarks",
            "model performance fitted": "Yes, primary A/B/C" if (threshold, horizon) == (0.60, 6) else "No; counts only as declared",
        })
    rows += [
        {"declared item": "Low CPO with concurrent SvO2 <60%", "original scope": "Prespecified sensitivity count", "corrected temporal legal-landmark result": "Not operationalised: no time-matching rule was prespecified", "model performance fitted": "No"},
        {"declared item": "2-h and 4-h landmark grids", "original scope": "Prespecified grid sensitivity", "corrected temporal legal-landmark result": "Completed during landmark audit", "model performance fitted": "No; feasibility/observation counts only"},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "tableS7_protocol_evolution.csv", index=False)
    (TABLES / "tableS7_protocol_evolution.md").write_text(markdown_table(out), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    robustness_table()
    measurement_table()
    protocol_table()


if __name__ == "__main__":
    main()
