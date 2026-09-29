#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
STAGE0 = BASE / "CV_PAC_CPO_STAGE0_MECHANISM_20260908"
STAGE1A = BASE / "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908"
STAGE1B = BASE / "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908"
PREV = BASE / "STAGE_MINUS1A_CV_OBSERVABILITY_V1" / "work"
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"
DEV = "definite_development_to_2019"
TEMP = "definite_temporal_2020_plus"
SEED = 20260908

sys.path.insert(0, str(STAGE0 / "code"))
from run_stage0_mechanism_audit import build_axis_times  # noqa: E402
sys.path.insert(0, str(STAGE1B / "code"))
from run_stage1b import (  # noqa: E402
    BLOCKS, CATEGORICAL, make_model, weighted_calibration, apply_calibration,
    metric_set, percentile_interval,
)


def add_prior_low_count(d):
    stays = pd.read_csv(PREV / "adult_stays.csv", dtype={"stay_id": str}, parse_dates=["intime", "outtime"])
    stays = stays.loc[stays.stay_id.isin(set(d.stay_id))]
    cpo, _, _ = build_axis_times(stays)
    groups = {k: (g.hours.to_numpy(float), g.cpo.to_numpy(float)) for k, g in cpo.sort_values("hours").groupby("stay_id")}
    counts = []
    for r in d[["stay_id", "landmark_h"]].itertuples(index=False):
        t, v = groups[r.stay_id]
        keep = (t > r.landmark_h - 4) & (t <= r.landmark_h)
        counts.append(int(np.sum(v[keep] < .60)))
    d = d.copy()
    d["n_prior_low_cpo_4h"] = counts
    d["clean_cpo_history_4h"] = d["n_prior_low_cpo_4h"].eq(0)
    return d


def fit_ab_clean(d, dev, temp):
    y = d.primary_outcome_6h.astype(int)
    preds = {}
    maps = []
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name in ["A", "B"]:
        numeric = BLOCKS[name]
        cols = numeric + CATEGORICAL
        xdev, ydev = d.loc[dev, cols], y.loc[dev]
        groups = d.loc[dev, "subject_id"]
        sw = d.loc[dev, "ipow_stabilized_truncated"].to_numpy(float)
        oof = np.full(len(xdev), np.nan)
        for tr, va in cv.split(xdev, ydev, groups=groups):
            m = make_model(numeric)
            m.fit(xdev.iloc[tr], ydev.iloc[tr], model__sample_weight=sw[tr])
            oof[va] = m.predict_proba(xdev.iloc[va])[:, 1]
        cal = weighted_calibration(ydev.to_numpy(), oof, sw)
        final = make_model(numeric)
        final.fit(xdev, ydev, model__sample_weight=sw)
        raw = final.predict_proba(d.loc[temp, cols])[:, 1]
        preds[name] = apply_calibration(raw, cal)
        maps.append({"model": name, "development_oof_intercept": cal[0], "development_oof_slope": cal[1]})
    return preds, pd.DataFrame(maps)


def paired_bootstrap(t, preds, n=1000):
    y = t.primary_outcome_6h.to_numpy(int)
    w = t.ipow_stabilized_truncated.to_numpy(float)
    patients = t.subject_id.unique()
    positions = {pid: np.flatnonzero(t.subject_id.to_numpy() == pid) for pid in patients}
    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(n):
        counts = pd.Series(rng.choice(patients, size=len(patients), replace=True)).value_counts()
        mult = np.zeros(len(t), float)
        for pid, count in counts.items():
            mult[positions[pid]] = count
        wb = w * mult
        if wb[y == 1].sum() == 0 or wb[y == 0].sum() == 0:
            continue
        ma, mb = metric_set(y, preds["A"], wb), metric_set(y, preds["B"], wb)
        rows.append({"replicate": b, **{f"A_{k}": v for k, v in ma.items()}, **{f"B_{k}": v for k, v in mb.items()}})
    return pd.DataFrame(rows)


def subset_metrics(t, mask, label):
    g = t.loc[mask]
    y = g.primary_outcome_6h.to_numpy(int)
    w = g.ipow_stabilized_truncated.to_numpy(float)
    rows = []
    for name in ["A", "B"]:
        m = metric_set(y, g[f"prediction_{name}"].to_numpy(float), w)
        rows.append({"subset": label, "model": name, "landmarks": len(g), "patients": g.subject_id.nunique(), "event_landmarks": int(y.sum()), "event_patients": g.loc[g.primary_outcome_6h.eq(1), "subject_id"].nunique(), **m})
    return rows


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(STAGE1A / "results/LANDMARK_FEATURES_V1.csv.gz", dtype={"stay_id": str})
    weights = pd.read_csv(STAGE1A / "results/observation_weights_v1.csv.gz", dtype={"stay_id": str})
    base = pd.read_csv(STAGE0 / "results/LANDMARK_DATASET_V1.csv.gz", dtype={"stay_id": str})
    keys = ["subject_id", "stay_id", "landmark_h", "interval_conservative_split"]
    d = f.merge(weights[keys + ["ipow_stabilized_truncated"]], on=keys, how="left", validate="one_to_one")
    d = d.merge(base[["subject_id", "stay_id", "landmark_h", "lead_time_h", "event_time_6h"]], on=["subject_id", "stay_id", "landmark_h"], how="left", validate="one_to_one")
    d = add_prior_low_count(d)
    common = d.primary_eligible_at_landmark.astype(bool) & d.primary_outcome_observed_6h.astype(bool) & d.model_bc_common_risk_set.astype(bool)
    d = d.loc[common].copy()

    prevalence = d.groupby(["interval_conservative_split", "primary_outcome_6h"]).agg(
        landmarks=("stay_id", "size"), patients=("subject_id", "nunique"),
        any_prior_low_landmarks=("n_prior_low_cpo_4h", lambda x: int((x > 0).sum())),
        median_prior_low_count=("n_prior_low_cpo_4h", "median"),
    ).reset_index()
    prevalence["any_prior_low_fraction"] = prevalence.any_prior_low_landmarks / prevalence.landmarks
    prevalence.to_csv(RESULTS / "prior_low_cpo_prevalence.csv", index=False)

    temporal_events = d[d.interval_conservative_split.eq(TEMP) & d.primary_outcome_6h.eq(1)].copy()
    unique_events = temporal_events.groupby(["subject_id", "stay_id", "event_time_6h"]).agg(
        earliest_warning_lead_h=("lead_time_h", "max"),
        nearest_warning_lead_h=("lead_time_h", "min"),
        contributing_landmarks=("landmark_h", "size"),
    ).reset_index()
    unique_events.to_csv(RESULTS / "temporal_unique_event_lead_times.csv", index=False)
    lead_summary = []
    for level, g, col in [
        ("landmark", temporal_events, "lead_time_h"),
        ("unique_event_earliest_legal_warning", unique_events, "earliest_warning_lead_h"),
    ]:
        x = g[col]
        lead_summary.append({
            "level": level, "n": len(g), "patients": g.subject_id.nunique(),
            "median_h": float(x.median()), "q1_h": float(x.quantile(.25)), "q3_h": float(x.quantile(.75)),
            "n_over_1h": int((x > 1).sum()), "n_over_2h": int((x > 2).sum()), "n_over_3h": int((x > 3).sum()),
            "patients_over_2h": int(g.loc[x > 2, "subject_id"].nunique()),
        })
    pd.DataFrame(lead_summary).to_csv(RESULTS / "lead_time_summary.csv", index=False)

    clean = d.clean_cpo_history_4h
    dev = clean & d.interval_conservative_split.eq(DEV)
    temp = clean & d.interval_conservative_split.eq(TEMP)
    preds, maps = fit_ab_clean(d, dev, temp)
    maps.to_csv(RESULTS / "clean_history_development_calibration_maps.csv", index=False)
    tclean = d.loc[temp, keys + ["primary_outcome_6h", "ipow_stabilized_truncated", "cpo_current", "lead_time_h"]].copy()
    for name in ["A", "B"]:
        tclean[f"prediction_{name}"] = preds[name]
    tclean.to_csv(RESULTS / "clean_history_temporal_predictions.csv.gz", index=False, compression="gzip")
    boot = paired_bootstrap(tclean, preds)
    boot.to_csv(RESULTS / "clean_history_patient_bootstrap.csv.gz", index=False, compression="gzip")
    clean_rows = []
    yclean = tclean.primary_outcome_6h.to_numpy(int)
    wclean = tclean.ipow_stabilized_truncated.to_numpy(float)
    point = {}
    for name in ["A", "B"]:
        point[name] = metric_set(yclean, preds[name], wclean)
        for metric, estimate in point[name].items():
            lo, hi = percentile_interval(boot[f"{name}_{metric}"])
            clean_rows.append({"model": name, "metric": metric, "estimate": estimate, "ci_low": lo, "ci_high": hi})
    for metric in point["A"]:
        vals = boot[f"B_{metric}"] - boot[f"A_{metric}"]
        lo, hi = percentile_interval(vals)
        clean_rows.append({"model": "B-A", "metric": metric, "estimate": point["B"][metric] - point["A"][metric], "ci_low": lo, "ci_high": hi})
    pd.DataFrame(clean_rows).to_csv(RESULTS / "clean_history_temporal_metrics.csv", index=False)

    locked = pd.read_csv(STAGE1B / "results/temporal_predictions.csv.gz", dtype={"stay_id": str})
    locked = locked.merge(d[["subject_id", "stay_id", "landmark_h", "cpo_current", "lead_time_h", "n_prior_low_cpo_4h"]], on=["subject_id", "stay_id", "landmark_h"], how="left", validate="one_to_one")
    subset_rows = []
    subset_rows += subset_metrics(locked, locked.cpo_current.ge(.65), "current_cpo_ge_0.65")
    subset_rows += subset_metrics(locked, locked.cpo_current.ge(.70), "current_cpo_ge_0.70")
    for cut in [1, 2]:
        mask = locked.primary_outcome_6h.eq(0) | locked.lead_time_h.gt(cut)
        subset_rows += subset_metrics(locked, mask, f"exclude_events_confirmed_le_{cut}h")
    subset = pd.DataFrame(subset_rows)
    subset.to_csv(RESULTS / "locked_prediction_interpretation_subsets.csv", index=False)

    clean_brier_delta = point["B"]["brier"] - point["A"]["brier"]
    clean_auc_delta = point["B"]["auroc"] - point["A"]["auroc"]
    event_patients_over2 = int(unique_events.loc[unique_events.earliest_warning_lead_h.gt(2), "subject_id"].nunique())
    viable = clean_brier_delta < 0 and clean_auc_delta > 0 and event_patients_over2 >= 30
    decision = {
        "decision": "VIABLE_DYNAMIC_CPO_EARLY_WARNING" if viable else "EARLY_WARNING_INTERPRETATION_NOT_SUPPORTED",
        "clean_history_development_landmarks": int(dev.sum()),
        "clean_history_temporal_landmarks": int(temp.sum()),
        "clean_history_temporal_event_landmarks": int(tclean.primary_outcome_6h.sum()),
        "clean_history_temporal_event_patients": int(tclean.loc[tclean.primary_outcome_6h.eq(1), "subject_id"].nunique()),
        "clean_history_b_minus_a_brier": clean_brier_delta,
        "clean_history_b_minus_a_auroc": clean_auc_delta,
        "temporal_unique_events": int(len(unique_events)),
        "temporal_event_patients_with_more_than_2h_legal_warning": event_patients_over2,
        "new_model_or_feature_search_performed": False,
    }
    (RESULTS / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False))
    audit = {
        "same_locked_A_B_estimator": True,
        "model_C_rescue_attempted": False,
        "future_predictors_used": False,
        "clean_history_definition": "zero CPO values <0.60 W in (L-4,L]",
        "subsets_prespecified": ["current CPO >=0.65", "current CPO >=0.70", "exclude event lead <=1h", "exclude event lead <=2h"],
        "clean_history_bootstrap_replicates": len(boot),
    }
    (AUDIT / "stage1c_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
