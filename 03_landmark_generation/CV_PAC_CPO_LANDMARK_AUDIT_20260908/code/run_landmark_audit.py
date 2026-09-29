from __future__ import annotations
import os

from pathlib import Path
import csv
import gzip
import json
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT.parent / "STAGE_MINUS1A_CV_OBSERVABILITY_V1" / "work"
MIMIC = Path(os.environ["CPO_MIMIC_DIR"])
OUT = ROOT / "results"


def raw(itemid: str, low: float, high: float, low_open: bool = False) -> pd.DataFrame:
    x = pd.read_csv(
        PREV / f"raw_{itemid}.csv.gz",
        usecols=["stay_id", "charttime", "valuenum", "valueuom"],
        dtype={"stay_id": str},
        parse_dates=["charttime"],
    )
    x["v"] = pd.to_numeric(x["valuenum"], errors="coerce")
    keep = np.isfinite(x["v"]) & (x["v"] <= high)
    keep &= x["v"] > low if low_open else x["v"] >= low
    x = x.loc[keep, ["stay_id", "charttime", "v"]]
    conflicts = x.groupby(["stay_id", "charttime"])["v"].nunique()
    conflicts = conflicts[conflicts > 1].index
    if len(conflicts):
        keys = pd.MultiIndex.from_frame(x[["stay_id", "charttime"]])
        x = x.loc[~keys.isin(conflicts)]
    return x.drop_duplicates(["stay_id", "charttime"])


def maps() -> pd.DataFrame:
    parts = []
    for priority, itemid in enumerate(["220052", "225312", "220181"]):
        x = pd.read_pickle(PREV / f"pair_{itemid}.pkl")
        x = x.loc[x["pair_valid"] & x["hours"].between(0, 72), ["stay_id", "charttime", "v"]].copy()
        x["stay_id"] = x["stay_id"].astype(str)
        x["priority"] = priority
        parts.append(x)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["stay_id", "charttime", "priority"])
        .drop_duplicates(["stay_id", "charttime"])
        .rename(columns={"v": "map"})
    )


def event_times(times: np.ndarray, values: np.ndarray, low: float = 0.60) -> np.ndarray:
    t = times[values < low]
    if len(t) < 2:
        return np.array([], dtype=float)
    gap = np.diff(t)
    return t[1:][(gap >= 0.5) & (gap <= 3.0)]


def window_event(times: np.ndarray, values: np.ndarray, left: float, right: float) -> float | None:
    keep = (times > left) & (times <= right) & (values < 0.60)
    t = times[keep]
    if len(t) < 2:
        return None
    gap = np.diff(t)
    idx = np.flatnonzero((gap >= 0.5) & (gap <= 3.0))
    return float(t[idx[0] + 1]) if len(idx) else None


def aligned_era_metadata(stays: pd.DataFrame) -> pd.DataFrame:
    patients = {}
    with gzip.open(MIMIC / "hosp/patients.csv.gz", "rt", newline="") as handle:
        for r in csv.DictReader(handle):
            patients[int(r["subject_id"])] = (int(r["anchor_year"]), r["anchor_year_group"])
    admissions = {}
    with gzip.open(MIMIC / "hosp/admissions.csv.gz", "rt", newline="") as handle:
        for r in csv.DictReader(handle):
            admissions[int(r["hadm_id"])] = pd.Timestamp(r["admittime"])
    rows = []
    for r in stays.itertuples(index=False):
        anchor_year, group = patients[int(r.subject_id)]
        admittime = admissions[int(r.hadm_id)]
        lo, hi = [int(z.strip()) for z in group.split("-")]
        delta = admittime.year - anchor_year
        approx = (lo + hi) // 2 + delta
        rows.append({
            "stay_id": str(r.stay_id),
            "anchor_year_group": group,
            "admittime": admittime,
            "delta_year": delta,
            "real_year_lower": lo + delta,
            "real_year_upper": hi + delta,
            "approx_real_year": approx,
        })
    return pd.DataFrame(rows)


def main() -> None:
    stays = pd.read_csv(
        PREV / "adult_stays.csv",
        dtype={"stay_id": str},
        parse_dates=["intime", "outtime"],
    )
    era = aligned_era_metadata(stays)
    stays = stays.merge(era, on="stay_id", validate="one_to_one")

    cco = raw("224842", 0, 30, low_open=True).rename(columns={"v": "cco"})
    cpo = cco.merge(maps(), on=["stay_id", "charttime"], how="inner")
    cpo = cpo.merge(
        stays[["subject_id", "hadm_id", "stay_id", "intime", "outtime", "approx_real_year", "real_year_lower", "real_year_upper"]],
        on="stay_id",
        how="inner",
        validate="many_to_one",
    )
    cpo = cpo.loc[cpo["charttime"].between(cpo["intime"], cpo["outtime"])].copy()
    cpo["hours"] = (cpo["charttime"] - cpo["intime"]).dt.total_seconds() / 3600.0
    cpo = cpo.loc[cpo["hours"].between(0, 72)].copy()
    cpo["cpo"] = cpo["cco"] * cpo["map"] / 451.0

    mpap = raw("220061", 0, 150).merge(stays[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
    svo2 = raw("223772", 0, 100).merge(stays[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
    for z in (mpap, svo2):
        z.drop(z.index[~z["charttime"].between(z["intime"], z["outtime"])], inplace=True)
        z["hours"] = (z["charttime"] - z["intime"]).dt.total_seconds() / 3600.0
        z.drop(z.index[~z["hours"].between(0, 72)], inplace=True)
    mh = {k: np.sort(v["hours"].to_numpy()) for k, v in mpap.groupby("stay_id")}
    sh = {k: np.sort(v["hours"].to_numpy()) for k, v in svo2.groupby("stay_id")}

    landmark_rows = []
    stay_rows = []
    for stay_id, g in cpo.sort_values("hours").groupby("stay_id", sort=False):
        times = g["hours"].to_numpy(dtype=float)
        values = g["cpo"].to_numpy(dtype=float)
        all_events = event_times(times, values)
        first_prior_event = float(all_events[0]) if len(all_events) else math.inf
        n_landmarks = n_pos6 = n_pos12_common = n_pos6_common = n_supported_pos = n_observed = 0
        first_event_from_landmark = math.nan
        for landmark in range(6, 67):
            if first_prior_event <= landmark:
                continue
            hist = (times > landmark - 4) & (times <= landmark)
            current = (times > landmark - 1) & (times <= landmark)
            if current.sum() == 0:
                continue
            current_idx = np.flatnonzero(current)[-1]
            if values[current_idx] < 0.60:
                continue
            ht = times[hist]
            if len(ht) < 3 or ht[-1] - ht[0] < 2:
                continue

            m = mh.get(stay_id, np.array([], dtype=float))
            s = sh.get(stay_id, np.array([], dtype=float))
            full_axes = int(((m > landmark - 4) & (m <= landmark)).sum()) >= 2 and int(((s > landmark - 4) & (s <= landmark)).sum()) >= 2
            ev6 = window_event(times, values, landmark, landmark + 6)
            ev12 = window_event(times, values, landmark, landmark + 12) if landmark <= 60 else None
            future = times[(times > landmark) & (times <= landmark + 6)]
            if len(future):
                chain = np.r_[times[current_idx], future]
                max_gap = float(np.diff(chain).max()) if len(chain) > 1 else math.inf
                dense_negative = ev6 is None and len(future) >= 4 and future[-1] >= landmark + 5 and max_gap <= 2
            else:
                max_gap = math.inf
                dense_negative = False
            observed = ev6 is not None or dense_negative
            lead = ev6 - landmark if ev6 is not None else math.nan
            landmark_rows.append({
                "subject_id": int(g["subject_id"].iloc[0]),
                "hadm_id": int(g["hadm_id"].iloc[0]),
                "stay_id": stay_id,
                "landmark_h": landmark,
                "grid_1h": True,
                "grid_2h": (landmark - 6) % 2 == 0,
                "grid_4h": (landmark - 6) % 4 == 0,
                "current_cpo": values[current_idx],
                "current_cpo_time_h": times[current_idx],
                "n_cpo_history_4h": int(hist.sum()),
                "full_mpap_svo2_history": full_axes,
                "event_6h": ev6 is not None,
                "event_time_6h": ev6,
                "lead_time_h": lead,
                "event_12h_common": ev12 is not None if landmark <= 60 else np.nan,
                "dense_observed_negative_6h": dense_negative,
                "outcome_observed_6h": observed,
                "n_future_cpo_6h": len(future),
                "last_future_cpo_offset_h": float(future[-1] - landmark) if len(future) else np.nan,
                "max_followup_gap_h": max_gap if np.isfinite(max_gap) else np.nan,
                "approx_real_year": int(g["approx_real_year"].iloc[0]),
                "real_year_lower": int(g["real_year_lower"].iloc[0]),
                "real_year_upper": int(g["real_year_upper"].iloc[0]),
            })
            n_landmarks += 1
            n_pos6 += ev6 is not None
            if landmark <= 60:
                n_pos6_common += ev6 is not None
                n_pos12_common += ev12 is not None
            n_supported_pos += ev6 is not None and full_axes
            n_observed += observed
            if ev6 is not None and not np.isfinite(first_event_from_landmark):
                first_event_from_landmark = ev6
        if n_landmarks:
            stay_rows.append({
                "subject_id": int(g["subject_id"].iloc[0]),
                "hadm_id": int(g["hadm_id"].iloc[0]),
                "stay_id": stay_id,
                "n_landmarks": n_landmarks,
                "has_event_positive_6h_landmark": n_pos6 > 0,
                "has_full_axes_event_positive_6h_landmark": n_supported_pos > 0,
                "n_event_positive_6h_landmarks": n_pos6,
                "n_event_positive_6h_common_landmarks": n_pos6_common,
                "n_event_positive_12h_common_landmarks": n_pos12_common,
                "n_observed_6h_landmarks": n_observed,
                "first_event_from_landmark_h": first_event_from_landmark,
                "approx_real_year": int(g["approx_real_year"].iloc[0]),
                "real_year_lower": int(g["real_year_lower"].iloc[0]),
                "real_year_upper": int(g["real_year_upper"].iloc[0]),
            })

    landmarks = pd.DataFrame(landmark_rows)
    stay_summary = pd.DataFrame(stay_rows)
    landmarks.to_csv(OUT / "rolling_landmarks.csv.gz", index=False, compression="gzip")
    stay_summary.to_csv(OUT / "stay_summary.csv", index=False)

    patient = stay_summary.groupby("subject_id").agg(
        n_stays=("stay_id", "nunique"),
        n_landmarks=("n_landmarks", "sum"),
        has_event_positive_6h_landmark=("has_event_positive_6h_landmark", "max"),
        has_full_axes_event_positive_6h_landmark=("has_full_axes_event_positive_6h_landmark", "max"),
        min_approx_real_year=("approx_real_year", "min"),
        max_approx_real_year=("approx_real_year", "max"),
        min_real_year_lower=("real_year_lower", "min"),
        max_real_year_upper=("real_year_upper", "max"),
    ).reset_index()
    patient["patient_era_split"] = np.select(
        [patient["max_approx_real_year"] <= 2019, patient["min_approx_real_year"] >= 2020],
        ["development_to_2019", "temporal_2020_plus"],
        default="cross_era_exclude",
    )
    patient["interval_conservative_split"] = np.select(
        [patient["max_real_year_upper"] <= 2019, patient["min_real_year_lower"] >= 2020],
        ["definite_development_to_2019", "definite_temporal_2020_plus"],
        default="boundary_ambiguous",
    )
    patient.to_csv(OUT / "patient_summary.csv", index=False)

    grid_rows = []
    for step, flag in [(1, "grid_1h"), (2, "grid_2h"), (4, "grid_4h")]:
        z = landmarks.loc[landmarks[flag]].copy()
        people = z.groupby("subject_id").agg(
            has_event=("event_6h", "max"),
            has_supported_event=("event_6h", lambda q: False),
        )
        supported_ids = set(z.loc[z["event_6h"] & z["full_mpap_svo2_history"], "subject_id"])
        grid_rows.append({
            "grid_hours": step,
            "patients": int(z["subject_id"].nunique()),
            "stays": int(z["stay_id"].nunique()),
            "landmarks": len(z),
            "event_positive_landmarks_6h": int(z["event_6h"].sum()),
            "event_positive_patients_6h": int(people["has_event"].sum()),
            "full_axes_event_positive_patients_6h": len(supported_ids),
            "observed_landmarks_6h": int(z["outcome_observed_6h"].sum()),
            "observed_fraction_6h": float(z["outcome_observed_6h"].mean()),
        })
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "grid_sensitivity.csv", index=False)

    common = landmarks.loc[landmarks["landmark_h"] <= 60]
    horizon = {
        "common_landmarks": int(len(common)),
        "event_positive_landmarks_6h": int(common["event_6h"].sum()),
        "event_positive_landmarks_12h": int(common["event_12h_common"].sum()),
        "event_positive_patients_6h": int(common.loc[common["event_6h"], "subject_id"].nunique()),
        "event_positive_patients_12h": int(common.loc[common["event_12h_common"] == True, "subject_id"].nunique()),
    }

    temporal = patient.groupby("patient_era_split").agg(
        patients=("subject_id", "size"),
        event_patients=("has_event_positive_6h_landmark", "sum"),
        full_axes_event_patients=("has_full_axes_event_positive_6h_landmark", "sum"),
    ).reset_index()
    temporal.to_csv(OUT / "patient_aligned_temporal_split.csv", index=False)
    interval_temporal = patient.groupby("interval_conservative_split").agg(
        patients=("subject_id", "size"),
        event_patients=("has_event_positive_6h_landmark", "sum"),
        full_axes_event_patients=("has_full_axes_event_positive_6h_landmark", "sum"),
    ).reset_index()
    interval_temporal.to_csv(OUT / "patient_interval_conservative_temporal_split.csv", index=False)

    approx = stay_summary.groupby("approx_real_year").agg(
        stays=("stay_id", "size"),
        patients=("subject_id", "nunique"),
        event_stays=("has_event_positive_6h_landmark", "sum"),
    ).reset_index()
    approx.to_csv(OUT / "stay_aligned_approx_year.csv", index=False)

    primary = grid.loc[grid["grid_hours"] == 1].iloc[0]
    survival = {
        "patients_ge_500": bool(primary["patients"] >= 500),
        "event_patients_ge_100": bool(primary["event_positive_patients_6h"] >= 100),
        "full_axes_event_patients_ge_80": bool(primary["full_axes_event_positive_patients_6h"] >= 80),
        "observed_fraction_ge_0p70": bool(primary["observed_fraction_6h"] >= 0.70),
    }
    late = temporal.loc[temporal["patient_era_split"] == "temporal_2020_plus"]
    if len(late):
        late_n = int(late["patients"].iloc[0]); late_e = int(late["event_patients"].iloc[0])
    else:
        late_n = late_e = 0
    if late_n >= 150 and late_e >= 60:
        temporal_gate = "GO"
    elif late_n >= 100 and late_e >= 40:
        temporal_gate = "CONDITIONAL"
    else:
        temporal_gate = "NO-GO"
    definite_late = interval_temporal.loc[
        interval_temporal["interval_conservative_split"] == "definite_temporal_2020_plus"
    ]
    if len(definite_late):
        definite_late_n = int(definite_late["patients"].iloc[0])
        definite_late_e = int(definite_late["event_patients"].iloc[0])
    else:
        definite_late_n = definite_late_e = 0
    if definite_late_n >= 150 and definite_late_e >= 60:
        conservative_temporal_gate = "GO"
    elif definite_late_n >= 100 and definite_late_e >= 40:
        conservative_temporal_gate = "CONDITIONAL"
    else:
        conservative_temporal_gate = "NO-GO"
    verdict = "SURVIVES_LANDMARK_AUDIT" if all([survival["patients_ge_500"], survival["event_patients_ge_100"], survival["full_axes_event_patients_ge_80"]]) else "NO-GO"
    result = {
        "verdict": verdict,
        "observation_framework": "binary_complete_case_possible" if survival["observed_fraction_ge_0p70"] else "censoring_or_observation_model_required",
        "temporal_validation_gate": temporal_gate,
        "interval_conservative_temporal_validation_gate": conservative_temporal_gate,
        "primary_grid": grid_rows[0],
        "horizon_comparison_on_common_landmarks": horizon,
        "survival_gates": survival,
        "temporal_split": temporal.to_dict(orient="records"),
        "interval_conservative_temporal_split": interval_temporal.to_dict(orient="records"),
        "future_informed_landmark_used": False,
        "models_fit": False,
        "patient_outcomes_read": False,
    }
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
