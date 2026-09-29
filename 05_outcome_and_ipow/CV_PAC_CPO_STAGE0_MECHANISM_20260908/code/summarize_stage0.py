#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def main():
    z = pd.read_csv(R / "landmark_mechanism_audit.csv.gz", dtype={"stay_id": str})
    z["landmark_period_h"] = pd.cut(
        z["landmark_h"], bins=[5, 18, 30, 42, 54, 66],
        labels=["06-18", "19-30", "31-42", "43-54", "55-66"], include_lowest=True,
    )
    tab = z.groupby(["landmark_period_h", "observation_state_6h"], observed=True).agg(
        landmarks=("stay_id", "size"), patients=("subject_id", "nunique")
    ).reset_index()
    totals = tab.groupby("landmark_period_h")["landmarks"].transform("sum")
    tab["within_period_fraction"] = tab["landmarks"] / totals
    tab.to_csv(R / "observation_state_by_landmark_period.csv", index=False)

    u = z.loc[z["observation_state_6h"].eq("other_unresolved_observation_loss")].copy()
    u["unresolved_primary_reason"] = np.select(
        [u["n_future_cpo_6h"].lt(4), u["last_future_cpo_offset_h"].fillna(-1).lt(5), u["max_followup_gap_h"].fillna(np.inf).gt(2)],
        ["fewer_than_4_future_cpo", "last_cpo_before_L_plus_5", "gap_over_2h"],
        default="other",
    )
    ur = u.groupby("unresolved_primary_reason").agg(
        landmarks=("stay_id", "size"), patients=("subject_id", "nunique")
    ).reset_index()
    ur["fraction_of_other_unresolved"] = ur["landmarks"] / len(u)
    ur.to_csv(R / "unresolved_reason_counts.csv", index=False)

    e = z.loc[z["observation_state_6h"].eq("icu_exit_before_horizon")].copy()
    last_offsets = []
    for r in e.itertuples(index=False):
        vals = []
        for future_name, recency_name in [
            ("last_future_cpo_offset_h", "cpo_recency_h"),
            ("last_future_mpap_offset_h", "mpap_recency_h"),
            ("last_future_svo2_offset_h", "svo2_recency_h"),
        ]:
            future = getattr(r, future_name)
            recency = getattr(r, recency_name)
            vals.append(float(future) if pd.notna(future) else -float(recency))
        last_offsets.append(max(vals))
    e["last_any_pac_offset_h"] = last_offsets
    e["hours_from_last_pac_to_icu_exit"] = e["icu_exit_offset_h"] - e["last_any_pac_offset_h"]
    e[["subject_id", "stay_id", "landmark_h", "icu_exit_offset_h", "last_any_pac_offset_h", "hours_from_last_pac_to_icu_exit"]].to_csv(
        R / "pac_to_icu_exit_landmarks.csv", index=False
    )
    s = e["hours_from_last_pac_to_icu_exit"].describe(percentiles=[0.25, 0.5, 0.75]).rename("value")
    s.to_csv(R / "pac_to_icu_exit_summary.csv", header=True)


if __name__ == "__main__":
    main()
