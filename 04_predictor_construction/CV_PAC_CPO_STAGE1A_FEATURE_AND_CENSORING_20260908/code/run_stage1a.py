#!/usr/bin/env python3
from __future__ import annotations
import os

from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
STAGE0 = BASE / "CV_PAC_CPO_STAGE0_MECHANISM_20260908"
PREV = BASE / "STAGE_MINUS1A_CV_OBSERVABILITY_V1" / "work"
MIMIC = Path(os.environ["CPO_MIMIC_DIR"])
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"

sys.path.insert(0, str(STAGE0 / "code"))
from run_stage0_mechanism_audit import build_axis_times  # noqa: E402


DEV = "definite_development_to_2019"
TEMP = "definite_temporal_2020_plus"

DRUG_COLS = [
    "drug_norepinephrine_active_at_landmark",
    "drug_epinephrine_active_at_landmark",
    "drug_vasopressin_active_at_landmark",
    "drug_phenylephrine_active_at_landmark",
    "drug_dopamine_active_at_landmark",
    "drug_dobutamine_active_at_landmark",
    "drug_milrinone_active_at_landmark",
]


def history_features(times: np.ndarray, values: np.ndarray, landmark: float, prefix: str) -> dict:
    keep = (times > landmark - 4) & (times <= landmark)
    t = times[keep]
    v = values[keep]
    if not len(v):
        return {f"{prefix}_{x}": math.nan for x in ["current", "mean_4h", "min_4h", "max_4h", "sd_4h", "slope_4h", "delta_4h", "span_4h"]}
    slope = math.nan
    if len(v) >= 2 and np.ptp(t) > 0:
        slope = float(np.polyfit(t - t.mean(), v, 1)[0])
    return {
        f"{prefix}_current": float(v[-1]),
        f"{prefix}_mean_4h": float(np.mean(v)),
        f"{prefix}_min_4h": float(np.min(v)),
        f"{prefix}_max_4h": float(np.max(v)),
        f"{prefix}_sd_4h": float(np.std(v, ddof=0)),
        f"{prefix}_slope_4h": slope,
        f"{prefix}_delta_4h": float(v[-1] - v[0]),
        f"{prefix}_span_4h": float(t[-1] - t[0]),
    }


def load_baseline(stays: pd.DataFrame) -> pd.DataFrame:
    icu = pd.read_csv(
        MIMIC / "icu/icustays.csv.gz", dtype={"stay_id": str},
        usecols=["stay_id", "first_careunit", "last_careunit"], keep_default_na=False,
    )
    patients = pd.read_csv(
        MIMIC / "hosp/patients.csv.gz", usecols=["subject_id", "gender"], keep_default_na=False,
    )
    return (
        stays[["subject_id", "stay_id", "age"]]
        .merge(icu, on="stay_id", how="left", validate="one_to_one")
        .merge(patients, on="subject_id", how="left", validate="many_to_one")
    )


def feature_matrix() -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(STAGE0 / "results/LANDMARK_DATASET_V1.csv.gz", dtype={"stay_id": str})
    stays = pd.read_csv(PREV / "adult_stays.csv", dtype={"stay_id": str}, parse_dates=["intime", "outtime"])
    stays = stays.loc[stays["stay_id"].isin(set(d["stay_id"]))].copy()
    cpo, mpap, svo2 = build_axis_times(stays)
    groups = {}
    for name, frame, value in [("cpo", cpo, "cpo"), ("mpap", mpap, "v"), ("svo2", svo2, "v")]:
        groups[name] = {
            k: (g["hours"].to_numpy(float), g[value].to_numpy(float))
            for k, g in frame.sort_values("hours").groupby("stay_id")
        }

    feat_rows = []
    for r in d[["stay_id", "landmark_h"]].itertuples(index=False):
        row = {"stay_id": r.stay_id, "landmark_h": int(r.landmark_h)}
        for name in ["cpo", "mpap", "svo2"]:
            t, v = groups[name].get(r.stay_id, (np.array([], dtype=float), np.array([], dtype=float)))
            row.update(history_features(t, v, float(r.landmark_h), name))
        feat_rows.append(row)
    f = pd.DataFrame(feat_rows)
    out = d.merge(f, on=["stay_id", "landmark_h"], how="left", validate="one_to_one")
    out = out.merge(load_baseline(stays), on=["subject_id", "stay_id"], how="left", validate="many_to_one")
    out["landmark_h_sq"] = out["landmark_h"] ** 2

    current_diff = np.abs(out["current_cpo"] - out["cpo_current"])
    common = out["model_bc_common_risk_set"].astype(bool)
    c_cols = [c for c in out.columns if c.startswith("mpap_") or c.startswith("svo2_")]
    c_dynamic = [c for c in c_cols if c.endswith(("current", "mean_4h", "min_4h", "max_4h", "sd_4h", "slope_4h", "delta_4h", "span_4h"))]
    audit = {
        "rows": int(len(out)),
        "unique_landmark_key": bool(~out.duplicated(["stay_id", "landmark_h"]).any()),
        "max_current_cpo_reconstruction_difference": float(current_diff.max()),
        "bc_common_rows": int(common.sum()),
        "bc_common_rows_with_any_missing_c_features": int(out.loc[common, c_dynamic].isna().any(axis=1).sum()),
        "future_named_columns_in_feature_blocks": [],
    }
    return out, audit


def make_pipeline(numeric: list[str], categorical: list[str]):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(C=1.0, penalty="l2", max_iter=3000, solver="lbfgs"))])


def calibration_table(y: np.ndarray, p: np.ndarray, split: str) -> pd.DataFrame:
    bins = pd.qcut(pd.Series(p), q=10, duplicates="drop")
    z = pd.DataFrame({"y": y, "p": p, "bin": bins})
    return z.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")).reset_index().assign(split=split)


def probability_summary(y: np.ndarray, p: np.ndarray, split: str) -> dict:
    q = np.quantile(p, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return {
        "split": split, "n": int(len(y)), "observed_fraction": float(np.mean(y)),
        "brier_observation": float(brier_score_loss(y, p)), "log_loss_observation": float(log_loss(y, p)),
        **{f"p_q{int(k*100):02d}": float(v) for k, v in zip([0, .01, .05, .25, .5, .75, .95, .99, 1], q)},
        "n_p_below_0_05": int(np.sum(p < .05)), "n_p_below_0_10": int(np.sum(p < .10)),
    }


def weight_summary(weights: np.ndarray, split: str, kind: str) -> dict:
    q = np.quantile(weights, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    ess = float(weights.sum() ** 2 / np.square(weights).sum())
    return {
        "split": split, "kind": kind, "n_observed": int(len(weights)),
        **{f"w_q{int(k*100):02d}": float(v) for k, v in zip([0, .01, .05, .25, .5, .75, .95, .99, 1], q)},
        "ess": ess, "ess_fraction": ess / len(weights),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    d, audit = feature_matrix()

    shared_numeric = [
        "age", "landmark_h", "landmark_h_sq",
        "n_cpo_history_4h", "n_mpap_history_4h", "n_svo2_history_4h",
        "cpo_recency_h", "mpap_recency_h", "svo2_recency_h",
        "iabp_active_at_landmark", *DRUG_COLS,
    ]
    model_a = ["cpo_current"]
    model_b_add = [
        "cpo_mean_4h", "cpo_min_4h", "cpo_max_4h", "cpo_sd_4h",
        "cpo_slope_4h", "cpo_delta_4h", "cpo_span_4h",
    ]
    model_c_add = []
    for axis in ["mpap", "svo2"]:
        model_c_add += [
            f"{axis}_current", f"{axis}_mean_4h", f"{axis}_min_4h", f"{axis}_max_4h",
            f"{axis}_sd_4h", f"{axis}_slope_4h", f"{axis}_delta_4h", f"{axis}_span_4h",
        ]
    categorical = ["gender", "first_careunit"]
    numeric = shared_numeric + model_a + model_b_add + model_c_add
    audit["future_named_columns_in_feature_blocks"] = [c for c in numeric + categorical if "future" in c or "event" in c or "outcome" in c]
    audit["feature_blocks"] = {
        "shared_numeric": shared_numeric, "shared_categorical": categorical,
        "model_a": model_a, "model_b_add": model_b_add, "model_c_add": model_c_add,
    }

    eligible = d["primary_eligible_at_landmark"].astype(bool)
    dev = eligible & d["interval_conservative_split"].eq(DEV)
    temp = eligible & d["interval_conservative_split"].eq(TEMP)
    y = d["primary_outcome_observed_6h"].astype(int)
    pred = pd.Series(np.nan, index=d.index, dtype=float)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260908)
    Xdev, ydev, gdev = d.loc[dev, numeric + categorical], y.loc[dev], d.loc[dev, "subject_id"]
    for train_pos, test_pos in cv.split(Xdev, ydev, groups=gdev):
        pipe = make_pipeline(numeric, categorical)
        pipe.fit(Xdev.iloc[train_pos], ydev.iloc[train_pos])
        pred.loc[Xdev.index[test_pos]] = pipe.predict_proba(Xdev.iloc[test_pos])[:, 1]

    full = make_pipeline(numeric, categorical)
    full.fit(Xdev, ydev)
    for mask in [temp, eligible & ~dev & ~temp]:
        if mask.any():
            pred.loc[mask] = full.predict_proba(d.loc[mask, numeric + categorical])[:, 1]
    assert pred.loc[eligible].notna().all()
    d["p_outcome_observed_6h"] = pred

    numerator = float(y.loc[dev].mean())
    observed = eligible & d["primary_outcome_observed_6h"].astype(bool)
    d["ipow_raw"] = np.where(observed, 1 / d["p_outcome_observed_6h"], np.nan)
    d["ipow_stabilized"] = np.where(observed, numerator / d["p_outcome_observed_6h"], np.nan)
    dev_obs_weights = d.loc[dev & observed, "ipow_stabilized"]
    lo, hi = [float(x) for x in dev_obs_weights.quantile([.01, .99])]
    d["ipow_stabilized_truncated"] = d["ipow_stabilized"].clip(lo, hi)

    prob_rows, cal_parts, weight_rows = [], [], []
    for split, mask in [("development_oof", dev), ("temporal_locked", temp)]:
        prob_rows.append(probability_summary(y.loc[mask].to_numpy(), pred.loc[mask].to_numpy(), split))
        cal_parts.append(calibration_table(y.loc[mask].to_numpy(), pred.loc[mask].to_numpy(), split))
        obs_mask = mask & observed
        weight_rows.append(weight_summary(d.loc[obs_mask, "ipow_stabilized"].to_numpy(), split, "stabilized_raw"))
        weight_rows.append(weight_summary(d.loc[obs_mask, "ipow_stabilized_truncated"].to_numpy(), split, "stabilized_truncated"))

    pd.DataFrame(prob_rows).to_csv(RESULTS / "observation_probability_summary.csv", index=False)
    pd.concat(cal_parts, ignore_index=True).to_csv(RESULTS / "observation_calibration_deciles.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(RESULTS / "observation_weight_summary.csv", index=False)
    d[[
        "subject_id", "stay_id", "landmark_h", "interval_conservative_split",
        "primary_eligible_at_landmark", "primary_outcome_observed_6h",
        "p_outcome_observed_6h", "ipow_raw", "ipow_stabilized", "ipow_stabilized_truncated",
    ]].to_csv(RESULTS / "observation_weights_v1.csv.gz", index=False, compression="gzip")

    feature_output_cols = [
        "subject_id", "hadm_id", "stay_id", "landmark_h", "interval_conservative_split",
        "primary_eligible_at_landmark", "primary_outcome_observed_6h", "primary_outcome_6h",
        "primary_followup_time_h", "model_bc_common_risk_set",
        *numeric, *categorical,
    ]
    d[feature_output_cols].to_csv(RESULTS / "LANDMARK_FEATURES_V1.csv.gz", index=False, compression="gzip")

    transformed_names = full.named_steps["pre"].get_feature_names_out()
    coef = full.named_steps["model"].coef_[0]
    pd.DataFrame({"feature": transformed_names, "coefficient": coef}).sort_values("coefficient").to_csv(
        RESULTS / "observation_model_coefficients.csv", index=False
    )

    audit["development_observation_fraction"] = numerator
    audit["truncation_lower_development_q01"] = lo
    audit["truncation_upper_development_q99"] = hi
    audit["clinical_outcome_used_for_observation_model"] = False
    audit["future_treatment_used_as_predictor"] = False
    audit["estimator"] = "L2 logistic; 5-fold stratified patient-group cross-fitting"
    (AUDIT / "stage1a_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    ps = pd.DataFrame(prob_rows)
    ws = pd.DataFrame(weight_rows)
    temp_prob = ps.loc[ps["split"].eq("temporal_locked")].iloc[0]
    temp_w = ws.loc[(ws["split"].eq("temporal_locked")) & (ws["kind"].eq("stabilized_truncated"))].iloc[0]
    missing_bc = audit["bc_common_rows_with_any_missing_c_features"]
    pass_gate = (
        not audit["future_named_columns_in_feature_blocks"]
        and audit["max_current_cpo_reconstruction_difference"] < 1e-10
        and missing_bc == 0
        and int(temp_prob["n_p_below_0_05"]) == 0
        and float(temp_w["ess_fraction"]) >= 0.50
    )
    decision = {
        "decision": "PASS_STAGE1A_UNLOCK_PRESPECIFIED_CLINICAL_MODELING" if pass_gate else "HOLD_CLINICAL_MODELING",
        "gate_components": {
            "no_feature_clock_leakage": not bool(audit["future_named_columns_in_feature_blocks"]),
            "cpo_reconstruction_exact": audit["max_current_cpo_reconstruction_difference"] < 1e-10,
            "complete_model_c_features_on_common_set": missing_bc == 0,
            "temporal_rows_with_p_below_0_05": int(temp_prob["n_p_below_0_05"]),
            "temporal_truncated_weight_ess_fraction": float(temp_w["ess_fraction"]),
        },
        "clinical_models_fitted_in_stage1a": False,
    }
    (RESULTS / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
