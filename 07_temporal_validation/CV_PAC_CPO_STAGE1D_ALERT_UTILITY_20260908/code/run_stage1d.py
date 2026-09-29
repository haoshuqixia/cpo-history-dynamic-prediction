#!/usr/bin/env python3
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
STAGE0 = BASE / "CV_PAC_CPO_STAGE0_MECHANISM_20260908"
STAGE1B = BASE / "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908"
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"
THRESHOLDS = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15]


def alert_episodes(df, pred_col, threshold):
    n = 0
    for _, g in df.sort_values(["stay_id", "landmark_h"]).groupby("stay_id"):
        alert = g[pred_col].to_numpy(float) >= threshold
        hours = g.landmark_h.to_numpy(float)
        start = alert & np.r_[True, (~alert[:-1]) | (np.diff(hours) > 1)]
        n += int(start.sum())
    return n


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(STAGE1B / "results/temporal_predictions.csv.gz", dtype={"stay_id": str})
    pairs = pd.read_csv(STAGE0 / "results/endpoint_pair_qa_landmarks.csv", dtype={"stay_id": str})
    pairs = pairs.merge(
        p[["subject_id", "stay_id", "landmark_h", "prediction_A", "prediction_B"]],
        on=["subject_id", "stay_id", "landmark_h"], how="inner", validate="one_to_one",
    )
    pair_keys = ["subject_id", "stay_id", "first_low_h", "confirming_low_h"]
    event_patients = set(p.loc[p.primary_outcome_6h.eq(1), "subject_id"])
    no_event = p.loc[~p.subject_id.isin(event_patients)]
    rows = []
    pair_detail = []
    for model in ["A", "B"]:
        col = f"prediction_{model}"
        for threshold in THRESHOLDS:
            pair_rows = []
            for key, g in pairs.groupby(pair_keys):
                alerted = g[g[col] >= threshold]
                detected = len(alerted) > 0
                first_alert = float(alerted.landmark_h.min()) if detected else np.nan
                pair_rows.append({
                    **dict(zip(pair_keys, key)), "model": model, "threshold": threshold,
                    "detected": detected, "first_alert_landmark_h": first_alert,
                    "lead_to_first_low_h": float(key[2] - first_alert) if detected else np.nan,
                    "lead_to_confirmation_h": float(key[3] - first_alert) if detected else np.nan,
                })
            q = pd.DataFrame(pair_rows)
            pair_detail.append(q)
            patient_detected = q.groupby("subject_id").detected.max()
            all_alert = p[col].ge(threshold)
            neg = p.primary_outcome_6h.eq(0)
            no_event_patient_alert = no_event.groupby("subject_id")[col].max().ge(threshold)
            episodes = alert_episodes(p, col, threshold)
            leads_first = q.loc[q.detected, "lead_to_first_low_h"]
            leads_confirm = q.loc[q.detected, "lead_to_confirmation_h"]
            rows.append({
                "model": model, "threshold": threshold,
                "event_pairs": len(q), "detected_event_pairs": int(q.detected.sum()),
                "event_pair_sensitivity": float(q.detected.mean()),
                "event_patients": len(patient_detected), "detected_event_patients": int(patient_detected.sum()),
                "event_patient_sensitivity": float(patient_detected.mean()),
                "median_lead_to_first_low_h": float(leads_first.median()) if len(leads_first) else np.nan,
                "q1_lead_to_first_low_h": float(leads_first.quantile(.25)) if len(leads_first) else np.nan,
                "q3_lead_to_first_low_h": float(leads_first.quantile(.75)) if len(leads_first) else np.nan,
                "median_lead_to_confirmation_h": float(leads_confirm.median()) if len(leads_confirm) else np.nan,
                "all_landmark_alert_fraction": float(all_alert.mean()),
                "event_free_landmark_alert_fraction": float(all_alert[neg].mean()),
                "alert_episodes": episodes,
                "alert_episodes_per_100_landmarks": float(episodes / len(p) * 100),
                "no_event_patients": int(no_event.subject_id.nunique()),
                "no_event_patients_with_any_alert": int(no_event_patient_alert.sum()),
                "no_event_patient_any_alert_fraction": float(no_event_patient_alert.mean()),
            })
    utility = pd.DataFrame(rows)
    utility.to_csv(RESULTS / "alert_utility.csv", index=False)
    pd.concat(pair_detail, ignore_index=True).to_csv(RESULTS / "event_pair_alert_details.csv.gz", index=False, compression="gzip")
    decision = {
        "status": "LOCKED_PREDICTIONS_TRANSLATED_NO_THRESHOLD_SELECTED",
        "temporal_landmarks": len(p), "temporal_patients": int(p.subject_id.nunique()),
        "event_pairs": int(pairs[pair_keys].drop_duplicates().shape[0]),
        "event_patients": int(pairs.subject_id.nunique()),
        "thresholds": THRESHOLDS,
        "new_model_fitted": False, "temporal_recalibration": False,
    }
    (RESULTS / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False))
    (AUDIT / "stage1d_audit.json").write_text(json.dumps({
        "prediction_source": str(STAGE1B / "results/temporal_predictions.csv.gz"),
        "pair_source": str(STAGE0 / "results/endpoint_pair_qa_landmarks.csv"),
        "threshold_selection_performed": False,
        "models": ["A", "B"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
