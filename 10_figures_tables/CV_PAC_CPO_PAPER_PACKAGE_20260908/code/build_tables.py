#!/usr/bin/env python3
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
STAGE0 = BASE / "CV_PAC_CPO_STAGE0_MECHANISM_20260908"
STAGE1A = BASE / "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908"
PREV = BASE / "STAGE_MINUS1A_CV_OBSERVABILITY_V1" / "work"
TABLES = ROOT / "tables"
AUDIT = ROOT / "audit"
DEV = "definite_development_to_2019"
TEMP = "definite_temporal_2020_plus"

sys.path.insert(0, str(STAGE0 / "code"))
from run_stage0_mechanism_audit import build_axis_times  # noqa: E402


def smd_cont(x1, x2, w1=None, w2=None):
    def mv(x, w):
        x = np.asarray(x, float)
        if w is None:
            return np.nanmean(x), np.nanstd(x, ddof=1)
        w = np.asarray(w, float)
        keep = np.isfinite(x) & np.isfinite(w)
        x, w = x[keep], w[keep]
        w = w / w.sum()
        m = np.sum(w * x)
        v = np.sum(w * (x - m) ** 2)
        return m, np.sqrt(v)
    m1, s1 = mv(x1, w1)
    m2, s2 = mv(x2, w2)
    den = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
    return (m2 - m1) / den if den > 0 else np.nan


def weighted_quantile(x, q, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    keep = np.isfinite(x) & np.isfinite(w)
    x, w = x[keep], w[keep]
    order = np.argsort(x)
    x, w = x[order], w[order]
    c = np.cumsum(w) - 0.5 * w
    c = c / w.sum()
    return float(np.interp(q, c, x))


def cont_summary(x, w=None):
    x = pd.to_numeric(x, errors="coerce").to_numpy(float)
    if w is None:
        q = np.nanquantile(x, [.25, .5, .75])
    else:
        q = [weighted_quantile(x, z, w) for z in [.25, .5, .75]]
    return f"{q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}]"


def binary_summary(x, w=None):
    x = pd.Series(x).astype(bool).to_numpy()
    if w is None:
        return f"{int(x.sum())} ({100*x.mean():.1f}%)"
    w = np.asarray(w, float)
    return f"{100*np.sum(w*x)/np.sum(w):.1f}%"


def binary_smd(x1, x2, w1=None, w2=None):
    def prop(x, w):
        x = pd.Series(x).astype(bool).to_numpy(float)
        return x.mean() if w is None else np.average(x, weights=w)
    p1, p2 = prop(x1, w1), prop(x2, w2)
    den = np.sqrt((p1*(1-p1)+p2*(1-p2))/2)
    return (p2-p1)/den if den > 0 else np.nan


def markdown_table(df):
    cols = list(df.columns)
    def clean(value):
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines += ["| " + " | ".join(clean(row[c]) for c in cols) + " |" for _, row in df.iterrows()]
    return "\n".join(lines) + "\n"


def add_current_components(d):
    stays = pd.read_csv(PREV / "adult_stays.csv", dtype={"stay_id": str}, parse_dates=["intime", "outtime"])
    stays = stays[stays.stay_id.isin(set(d.stay_id))]
    cpo, _, _ = build_axis_times(stays)
    groups = {k: g.sort_values("hours") for k, g in cpo.groupby("stay_id")}
    maps, ccos = [], []
    for r in d[["stay_id", "landmark_h"]].itertuples(index=False):
        g = groups[r.stay_id]
        q = g[(g.hours > r.landmark_h - 1) & (g.hours <= r.landmark_h)].iloc[-1]
        maps.append(float(q["map"]))
        ccos.append(float(q["cco"]))
    d = d.copy()
    d["current_map"] = maps
    d["current_cco"] = ccos
    return d


def patient_table(model_rows, stays):
    z = model_rows.merge(stays[["stay_id", "intime"]], on="stay_id", how="left", validate="many_to_one")
    z["landmark_datetime"] = z.intime + pd.to_timedelta(z.landmark_h, unit="h")
    agg = z.groupby("subject_id").agg(
        n_model_landmarks=("stay_id", "size"),
        event_patient=("primary_outcome_6h", "max"),
    ).reset_index()
    first = z.sort_values(["subject_id", "landmark_datetime"]).drop_duplicates("subject_id")
    first = first.merge(agg, on="subject_id", how="left", validate="one_to_one")
    first["female"] = first.gender.eq("F")
    first["cvicu"] = first.first_careunit.eq("Cardiac Vascular Intensive Care Unit (CVICU)")
    first["any_drug_active"] = first[[c for c in first if c.startswith("drug_") and c.endswith("_active_at_landmark")]].astype(bool).any(axis=1)
    dev, temp = first[first.interval_conservative_split.eq(DEV)], first[first.interval_conservative_split.eq(TEMP)]
    rows = []
    continuous = [
        ("Age, years", "age"), ("Eligible model landmarks per patient", "n_model_landmarks"),
        ("First eligible landmark, ICU hour", "landmark_h"), ("Current CPO, W", "cpo_current"),
        ("CPO slope over 4 h, W/h", "cpo_slope_4h"), ("Current mPAP, mmHg", "mpap_current"),
        ("Current SvO2, %", "svo2_current"), ("CPO measurements in prior 4 h", "n_cpo_history_4h"),
    ]
    binary = [
        ("Female sex", "female"),
        ("IABP active at first landmark", "iabp_active_at_landmark"),
        ("Any vasoactive/inotropic infusion active", "any_drug_active"),
        ("Patient with >=1 event landmark", "event_patient"),
    ]
    rows.append({"variable": "Patients", "development": str(len(dev)), "temporal_validation": str(len(temp)), "smd": np.nan})
    for label, col in continuous:
        rows.append({"variable": label, "development": cont_summary(dev[col]), "temporal_validation": cont_summary(temp[col]), "smd": smd_cont(dev[col], temp[col])})
    for label, col in binary:
        rows.append({"variable": label, "development": binary_summary(dev[col]), "temporal_validation": binary_summary(temp[col]), "smd": binary_smd(dev[col], temp[col])})
    careunits = sorted(set(dev.first_careunit) | set(temp.first_careunit))
    for level in careunits:
        rows.append({
            "variable": f"First care unit: {level}",
            "development": binary_summary(dev.first_careunit.eq(level)),
            "temporal_validation": binary_summary(temp.first_careunit.eq(level)),
            "smd": binary_smd(dev.first_careunit.eq(level), temp.first_careunit.eq(level)),
        })
    return pd.DataFrame(rows), first


def landmark_table(risk):
    z = risk.copy()
    z["patient_weight"] = 1 / z.groupby(["interval_conservative_split", "subject_id"])["stay_id"].transform("size")
    dev, temp = z[z.interval_conservative_split.eq(DEV)], z[z.interval_conservative_split.eq(TEMP)]
    rows = [{"variable": "At-risk landmarks", "development": str(len(dev)), "temporal_validation": str(len(temp)), "smd": np.nan}]
    continuous = [
        ("Landmark ICU hour", "landmark_h"), ("Current CPO, W", "cpo_current"),
        ("Current MAP, mmHg", "current_map"), ("Current CCO, L/min", "current_cco"),
        ("Current mPAP, mmHg", "mpap_current"), ("Current SvO2, %", "svo2_current"),
        ("CPO slope over 4 h, W/h", "cpo_slope_4h"), ("CPO SD over 4 h, W", "cpo_sd_4h"),
        ("CPO measurements in prior 4 h", "n_cpo_history_4h"),
        ("mPAP measurements in prior 4 h", "n_mpap_history_4h"),
        ("SvO2 measurements in prior 4 h", "n_svo2_history_4h"),
        ("CPO recency, h", "cpo_recency_h"),
    ]
    binary = [
        ("Outcome observable over 6 h", "primary_outcome_observed_6h"),
        ("IABP active at landmark", "iabp_active_at_landmark"),
        ("Norepinephrine active", "drug_norepinephrine_active_at_landmark"),
        ("Epinephrine active", "drug_epinephrine_active_at_landmark"),
        ("Vasopressin active", "drug_vasopressin_active_at_landmark"),
        ("Phenylephrine active", "drug_phenylephrine_active_at_landmark"),
        ("Dobutamine active", "drug_dobutamine_active_at_landmark"),
        ("Milrinone active", "drug_milrinone_active_at_landmark"),
    ]
    for label, col in continuous:
        rows.append({
            "variable": label,
            "development": cont_summary(dev[col], dev.patient_weight),
            "temporal_validation": cont_summary(temp[col], temp.patient_weight),
            "smd": smd_cont(dev[col], temp[col], dev.patient_weight, temp.patient_weight),
        })
    for label, col in binary:
        rows.append({
            "variable": label,
            "development": binary_summary(dev[col], dev.patient_weight),
            "temporal_validation": binary_summary(temp[col], temp.patient_weight),
            "smd": binary_smd(dev[col], temp[col], dev.patient_weight, temp.patient_weight),
        })
    dev_obs = dev.loc[dev.primary_outcome_observed_6h.astype(bool)].copy()
    temp_obs = temp.loc[temp.primary_outcome_observed_6h.astype(bool)].copy()
    dev_obs["observed_patient_weight"] = 1 / dev_obs.groupby("subject_id")["stay_id"].transform("size")
    temp_obs["observed_patient_weight"] = 1 / temp_obs.groupby("subject_id")["stay_id"].transform("size")
    event_row = {
        "variable": "Confirmed/repeated low-CPO event among observed horizons",
        "development": binary_summary(dev_obs.event_6h, dev_obs.observed_patient_weight),
        "temporal_validation": binary_summary(temp_obs.event_6h, temp_obs.observed_patient_weight),
        "smd": binary_smd(dev_obs.event_6h, temp_obs.event_6h, dev_obs.observed_patient_weight, temp_obs.observed_patient_weight),
    }
    rows.insert(14, event_row)
    return pd.DataFrame(rows)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(STAGE1A / "results/LANDMARK_FEATURES_V1.csv.gz", dtype={"stay_id": str})
    base = pd.read_csv(STAGE0 / "results/LANDMARK_DATASET_V1.csv.gz", dtype={"stay_id": str})
    keys = ["subject_id", "hadm_id", "stay_id", "landmark_h", "interval_conservative_split"]
    extra = base[keys + ["event_6h", "major_mcs_competing_censor_6h"]]
    d = f.merge(extra, on=keys, how="left", validate="one_to_one")
    d = add_current_components(d)
    stays = pd.read_csv(PREV / "adult_stays.csv", dtype={"stay_id": str}, parse_dates=["intime"])

    model = d.primary_eligible_at_landmark.astype(bool) & d.primary_outcome_observed_6h.astype(bool) & d.model_bc_common_risk_set.astype(bool)
    table1, patient_data = patient_table(d.loc[model], stays)
    table1.to_csv(TABLES / "table1_patient_level.csv", index=False)
    (TABLES / "table1_patient_level.md").write_text(markdown_table(table1))

    risk = d.primary_eligible_at_landmark.astype(bool) & d.model_bc_common_risk_set.astype(bool)
    drift = landmark_table(d.loc[risk])
    drift.to_csv(TABLES / "tableS1_landmark_temporal_drift.csv", index=False)
    (TABLES / "tableS1_landmark_temporal_drift.md").write_text(markdown_table(drift))

    flow_rows = []
    stages = [
        ("Legal hourly landmarks", pd.Series(True, index=d.index)),
        ("After exclusion of major MCS active at L", d.primary_eligible_at_landmark.astype(bool)),
        ("Model B/C common risk set", risk),
        ("Outcome-observed common risk set", model),
    ]
    for stage, mask in stages:
        g = d.loc[mask]
        flow_rows.append({"stage": stage, "patients": g.subject_id.nunique(), "stays": g.stay_id.nunique(), "landmarks": len(g)})
    for split in [DEV, TEMP, "boundary_ambiguous"]:
        g = d.loc[model & d.interval_conservative_split.eq(split)]
        flow_rows.append({"stage": f"Final model rows: {split}", "patients": g.subject_id.nunique(), "stays": g.stay_id.nunique(), "landmarks": len(g)})
    flow = pd.DataFrame(flow_rows)
    flow.to_csv(TABLES / "cohort_flow_counts.csv", index=False)
    (TABLES / "cohort_flow_counts.md").write_text(markdown_table(flow))

    patient_data.to_csv(AUDIT / "patient_level_analysis_data.csv.gz", index=False, compression="gzip")
    d.loc[risk].to_csv(AUDIT / "landmark_drift_analysis_data.csv.gz", index=False, compression="gzip")
    assert table1.loc[table1.variable.eq("Patients"), "development"].iloc[0] == "1544"
    assert table1.loc[table1.variable.eq("Patients"), "temporal_validation"].iloc[0] == "265"
    assert len(d.loc[model & d.interval_conservative_split.eq(DEV)]) == 21619
    assert len(d.loc[model & d.interval_conservative_split.eq(TEMP)]) == 3416


if __name__ == "__main__":
    main()
