#!/usr/bin/env python3
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PREV = ROOT.parent / "CV_PAC_CPO_LANDMARK_AUDIT_20260908" / "results"


def main():
    z = pd.read_csv(RESULTS / "landmark_mechanism_audit.csv.gz", dtype={"stay_id": str})
    p = pd.read_csv(PREV / "patient_summary.csv")[["subject_id", "patient_era_split", "interval_conservative_split"]]
    z = z.merge(p, on="subject_id", how="left", validate="many_to_one")

    z["unresolved_fewer_than_4_future_cpo"] = z["n_future_cpo_6h"] < 4
    z["unresolved_last_cpo_before_l_plus_5"] = z["last_future_cpo_offset_h"].fillna(-1) < 5
    z["unresolved_gap_over_2h"] = z["max_followup_gap_h"].fillna(np.inf) > 2

    z["primary_eligible_at_landmark"] = ~z["major_mcs_active_at_landmark"].astype(bool)
    mcs_t = pd.to_numeric(z["major_mcs_first_procedure_offset_h"], errors="coerce")
    event_t = pd.to_numeric(z["lead_time_h"], errors="coerce")
    mcs_before_event = mcs_t.notna() & (event_t.isna() | (mcs_t < event_t))
    z["major_mcs_competing_censor_6h"] = mcs_before_event

    observed_event = z["event_6h"].astype(bool) & ~mcs_before_event
    observed_negative = z["dense_observed_negative_6h"].astype(bool) & ~mcs_before_event
    z["primary_outcome_observed_6h"] = observed_event | observed_negative
    outcome = pd.Series(pd.NA, index=z.index, dtype="Int64")
    outcome.loc[observed_negative] = 0
    outcome.loc[observed_event] = 1
    z["primary_outcome_6h"] = outcome

    censor_reason = np.select(
        [
            ~z["primary_eligible_at_landmark"],
            mcs_before_event,
            observed_event,
            observed_negative,
            z["observation_state_6h"].eq("icu_exit_before_horizon"),
            z["observation_state_6h"].eq("isolated_cpo_recording_cessation"),
            z["observation_state_6h"].eq("joint_pac_axis_cessation"),
        ],
        [
            "major_mcs_active_at_landmark_exclusion",
            "major_mcs_started_before_outcome",
            "confirmed_low_cpo_event",
            "dense_observed_event_free",
            "icu_exit",
            "isolated_cpo_cessation",
            "joint_pac_axis_cessation",
        ],
        default="sparse_or_gapped_cpo_followup",
    )
    z["primary_status_6h"] = censor_reason

    last_cpo = pd.to_numeric(z["last_future_cpo_offset_h"], errors="coerce").fillna(0).clip(0, 6)
    exit_t = pd.to_numeric(z["icu_exit_offset_h"], errors="coerce").clip(0, 6)
    followup = pd.Series(6.0, index=z.index)
    unknown = ~z["primary_outcome_observed_6h"]
    followup.loc[unknown] = last_cpo.loc[unknown]
    exit_mask = unknown & z["icu_exit_before_horizon"].astype(bool)
    followup.loc[exit_mask] = np.minimum(followup.loc[exit_mask], exit_t.loc[exit_mask])
    followup.loc[mcs_before_event] = np.minimum(followup.loc[mcs_before_event], mcs_t.loc[mcs_before_event])
    followup.loc[observed_event] = event_t.loc[observed_event]
    z["primary_followup_time_h"] = followup
    z["model_bc_common_risk_set"] = z["primary_eligible_at_landmark"] & z["full_mpap_svo2_history"].astype(bool)

    safe_cols = [
        "subject_id", "hadm_id", "stay_id", "landmark_h", "grid_1h", "grid_2h", "grid_4h",
        "patient_era_split", "interval_conservative_split",
        "current_cpo", "current_cpo_time_h", "n_cpo_history_4h",
        "n_mpap_history_4h", "n_svo2_history_4h", "cpo_recency_h", "mpap_recency_h", "svo2_recency_h",
        "full_mpap_svo2_history", "major_mcs_active_at_landmark", "iabp_active_at_landmark",
        "mcs_active_types", "drug_history_4h_any", "drug_active_at_landmark_any",
        "drug_history_4h_agents", "drug_active_agents",
    ]
    safe_cols += [c for c in z.columns if c.startswith("drug_") and c.endswith("_active_at_landmark")]
    outcome_cols = [
        "event_6h", "event_time_6h", "lead_time_h", "dense_observed_negative_6h",
        "outcome_observed_6h", "observation_state_6h", "primary_eligible_at_landmark",
        "major_mcs_competing_censor_6h", "primary_outcome_observed_6h", "primary_outcome_6h",
        "primary_status_6h", "primary_followup_time_h", "model_bc_common_risk_set",
        "unresolved_fewer_than_4_future_cpo", "unresolved_last_cpo_before_l_plus_5", "unresolved_gap_over_2h",
    ]
    dataset = z[safe_cols + outcome_cols].copy()
    dataset.to_csv(RESULTS / "LANDMARK_DATASET_V1.csv.gz", index=False, compression="gzip")

    rows = []
    for split, g in dataset.groupby("interval_conservative_split"):
        eligible = g["primary_eligible_at_landmark"].astype(bool)
        obs = eligible & g["primary_outcome_observed_6h"].astype(bool)
        bc = eligible & g["model_bc_common_risk_set"].astype(bool)
        bc_obs = bc & g["primary_outcome_observed_6h"].astype(bool)
        rows.append({
            "interval_conservative_split": split,
            "patients": int(g.subject_id.nunique()), "landmarks": len(g),
            "primary_eligible_patients": int(g.loc[eligible, "subject_id"].nunique()),
            "primary_eligible_landmarks": int(eligible.sum()),
            "primary_observed_landmarks": int(obs.sum()),
            "primary_event_landmarks": int((obs & g["primary_outcome_6h"].eq(1)).sum()),
            "primary_event_patients": int(g.loc[obs & g["primary_outcome_6h"].eq(1), "subject_id"].nunique()),
            "bc_common_landmarks": int(bc.sum()), "bc_common_observed_landmarks": int(bc_obs.sum()),
            "bc_common_event_landmarks": int((bc_obs & g["primary_outcome_6h"].eq(1)).sum()),
            "bc_common_event_patients": int(g.loc[bc_obs & g["primary_outcome_6h"].eq(1), "subject_id"].nunique()),
        })
    counts = pd.DataFrame(rows)
    counts.to_csv(RESULTS / "landmark_dataset_v1_counts.csv", index=False)

    assert len(dataset) == 38613
    assert not dataset.duplicated(["stay_id", "landmark_h"]).any()
    assert dataset.loc[dataset["primary_outcome_6h"].notna(), "primary_outcome_observed_6h"].all()
    assert not dataset.loc[~dataset["primary_eligible_at_landmark"], "model_bc_common_risk_set"].any()
    assert dataset["primary_followup_time_h"].between(0, 6).all()

    summary = {
        "status": "FROZEN_FOR_FEATURE_ENGINEERING_NO_MODEL_FIT",
        "rows": int(len(dataset)),
        "patients": int(dataset.subject_id.nunique()),
        "primary_eligible_rows": int(dataset.primary_eligible_at_landmark.sum()),
        "primary_eligible_patients": int(dataset.loc[dataset.primary_eligible_at_landmark, "subject_id"].nunique()),
        "primary_observed_rows": int((dataset.primary_eligible_at_landmark & dataset.primary_outcome_observed_6h).sum()),
        "primary_event_rows": int((dataset.primary_eligible_at_landmark & dataset.primary_outcome_6h.eq(1)).sum()),
        "primary_event_patients": int(dataset.loc[dataset.primary_eligible_at_landmark & dataset.primary_outcome_6h.eq(1), "subject_id"].nunique()),
        "major_mcs_competing_censor_rows": int((dataset.primary_eligible_at_landmark & dataset.major_mcs_competing_censor_6h).sum()),
        "future_treatment_predictors_in_dataset": False,
    }
    (RESULTS / "landmark_dataset_v1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
