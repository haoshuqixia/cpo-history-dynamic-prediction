#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import math
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"
WORK = ROOT / "work"
MODELS = BASE / "CV_PAC_CPO_PAPER_PACKAGE_20260908" / "models"
MIMIC_FEATURES = BASE / "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908" / "results" / "LANDMARK_FEATURES_V1.csv.gz"
SEED = 20260908

sys.path.insert(0, str(ROOT / "code"))
from run_eicu_stage_eminus1 import open_inner, read_nurse_targets, resolve  # noqa: E402


UNIT_MAP = {
    "CCU-CTICU": "Cardiac Vascular Intensive Care Unit (CVICU)",
    "CTICU": "Cardiac Vascular Intensive Care Unit (CVICU)",
    "CSICU": "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Cardiac ICU": "Coronary Care Unit (CCU)",
    "SICU": "Surgical Intensive Care Unit (SICU)",
    "Neuro ICU": "Neuro Surgical Intensive Care Unit (Neuro SICU)",
    "MICU": "Medical Intensive Care Unit (MICU)",
    "Med-Surg ICU": "Medical/Surgical Intensive Care Unit (MICU/SICU)",
}
GENDER_MAP = {"Male": "M", "Female": "F"}
DRUGS = ["norepinephrine", "epinephrine", "vasopressin", "phenylephrine", "dopamine", "dobutamine", "milrinone"]
IABP_ACTIVE = {
    "cardiovascular|non-operative procedures|intraaortic balloon pump",
    "cardiovascular|shock|intraaortic balloon pump",
}
IABP_REMOVAL = "cardiovascular|non-operative procedures|intraaortic balloon pump removal"


def history_features(times: np.ndarray, available: np.ndarray, values: np.ndarray, landmark: float, prefix: str) -> dict:
    keep = (times > landmark - 4) & (times <= landmark) & (available <= landmark)
    t = times[keep]
    v = values[keep]
    if not len(v):
        out = {f"{prefix}_{x}": math.nan for x in ["current", "mean_4h", "min_4h", "max_4h", "sd_4h", "slope_4h", "delta_4h", "span_4h"]}
        out[f"n_{prefix}_history_4h"] = 0
        out[f"{prefix}_recency_h"] = math.nan
        return out
    order = np.argsort(t, kind="stable")
    t = t[order]
    v = v[order]
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
        f"n_{prefix}_history_4h": int(len(v)),
        f"{prefix}_recency_h": float(landmark - t[-1]),
    }


def group_series(frame: pd.DataFrame, value_col: str, available_col: str) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    for sid, g in frame.sort_values(["sid", "offset", available_col]).groupby("sid"):
        out[int(sid)] = (
            g.offset.to_numpy(float) / 60,
            g[available_col].to_numpy(float) / 60,
            g[value_col].to_numpy(float),
        )
    return out


def medication_groups(sids: set[int]) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    states = pd.read_csv(WORK / "formal_cohort_infusion_states.csv.gz")
    states = states[states.sid.isin(sids)].sort_values(["sid", "drug", "offset"])
    return {
        (int(sid), str(drug)): (g.offset.to_numpy(float), g.state.to_numpy(str))
        for (sid, drug), g in states.groupby(["sid", "drug"])
    }


def drug_at_landmark(group: tuple[np.ndarray, np.ndarray] | None, landmark_min: float, cap: float = 60) -> float:
    if group is None:
        return 0.0
    times, states = group
    pos = np.searchsorted(times, landmark_min, side="right") - 1
    if pos < 0:
        return 0.0
    age = landmark_min - times[pos]
    if states[pos] == "unknown_conflict" and age <= cap:
        return math.nan
    return float(states[pos] == "active" and age <= cap)


def iabp_groups(sids: set[int]) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict]:
    raw: dict[tuple[int, int], set[str]] = defaultdict(set)
    exact_counts = defaultdict(int)
    with open_inner("treatment") as handle:
        for row in csv.DictReader(handle):
            try:
                sid = int(row["patientunitstayid"])
                offset = int(float(row["treatmentoffset"]))
            except (TypeError, ValueError):
                continue
            if sid not in sids:
                continue
            text = row["treatmentstring"]
            if text in IABP_ACTIVE:
                raw[(sid, offset)].add("active")
                exact_counts[text] += 1
            elif text == IABP_REMOVAL:
                raw[(sid, offset)].add("inactive")
                exact_counts[text] += 1
    resolved = defaultdict(list)
    conflicts = 0
    for (sid, offset), values in raw.items():
        if len(values) == 1:
            state = next(iter(values))
        else:
            state = "unknown_conflict"
            conflicts += 1
        resolved[sid].append((offset, state))
    groups = {}
    for sid, rows in resolved.items():
        rows.sort()
        groups[int(sid)] = (np.array([r[0] for r in rows], float), np.array([r[1] for r in rows], str))
    audit = {"exact_path_counts": dict(exact_counts), "stays_with_iabp_state": len(groups), "simultaneous_conflicts": conflicts}
    return groups, audit


def iabp_at_landmark(group: tuple[np.ndarray, np.ndarray] | None, landmark_min: float) -> float:
    if group is None:
        return 0.0
    times, states = group
    pos = np.searchsorted(times, landmark_min, side="right") - 1
    if pos < 0:
        return 0.0
    if states[pos] == "unknown_conflict":
        return math.nan
    return float(states[pos] == "active")


def make_observation_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(C=1.0, penalty="l2", max_iter=3000, solver="lbfgs"))])


def quantile_row(values: np.ndarray, prefix: str = "") -> dict:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    qs = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    q = np.quantile(values, qs) if len(values) else np.full(len(qs), np.nan)
    return {f"{prefix}q{int(k*100):02d}": float(v) for k, v in zip(qs, q)}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    a_art = joblib.load(MODELS / "model_A_locked.joblib")
    b_art = joblib.load(MODELS / "model_B_locked.joblib")
    c_art = joblib.load(MODELS / "model_C_locked.joblib")
    numeric_a = list(a_art["numeric_predictors"])
    numeric_b = list(b_art["numeric_predictors"])
    numeric_c = list(c_art["numeric_predictors"])
    categorical = list(c_art["categorical_predictors"])

    # Deliberately exclude event_6h and event_time_h from the read boundary.
    keys = ["sid", "uniquepid", "hospitalid", "landmark_h", "outcome_observed_6h"]
    landmarks = pd.read_csv(RESULTS / "stage_e0_landmarks.csv.gz", usecols=keys)
    landmarks["outcome_observed_6h"] = landmarks.outcome_observed_6h.astype(bool)
    sids = set(landmarks.sid.astype(int))

    patients = pd.read_csv(WORK / "candidate_patient_metadata.csv.gz")
    patients = patients[patients.sid.isin(sids)][["sid", "age", "gender", "unittype"]].drop_duplicates("sid")
    patients["gender"] = patients.gender.map(GENDER_MAP)
    patients["first_careunit"] = patients.unittype.map(UNIT_MAP).fillna(patients.unittype)
    d = landmarks.merge(patients.drop(columns="unittype"), on="sid", how="left", validate="many_to_one")
    d["landmark_h_sq"] = d.landmark_h.astype(float) ** 2

    cpo = pd.read_csv(WORK / "cpo_nurse.csv.gz", usecols=["sid", "offset", "available", "cpo"])
    cpo = cpo[cpo.sid.isin(sids)].copy()
    cpo_groups = group_series(cpo, "cpo", "available")

    nurse, nurse_audit = read_nurse_targets(WORK / "nurse_cardiac_targets.csv.gz")
    pamean, pamean_conflicts = resolve(nurse["pamean"], 0, 150, entry=True)
    svo2, svo2_conflicts = resolve(nurse["svo2"], 0, 100, entry=True)
    pamean = pamean[pamean.sid.isin(sids)].copy()
    svo2 = svo2[svo2.sid.isin(sids)].copy()
    mpap_groups = group_series(pamean, "value", "entry")
    svo2_groups = group_series(svo2, "value", "entry")
    drug_groups = medication_groups(sids)
    iabp, iabp_audit = iabp_groups(sids)

    empty = (np.array([], float), np.array([], float), np.array([], float))
    rows = []
    for r in d[["sid", "landmark_h"]].itertuples(index=False):
        sid = int(r.sid)
        landmark = float(r.landmark_h)
        row = {"sid": sid, "landmark_h": int(r.landmark_h)}
        for prefix, groups in [("cpo", cpo_groups), ("mpap", mpap_groups), ("svo2", svo2_groups)]:
            row.update(history_features(*groups.get(sid, empty), landmark, prefix))
        landmark_min = landmark * 60
        row["iabp_active_at_landmark"] = iabp_at_landmark(iabp.get(sid), landmark_min)
        for drug in DRUGS:
            row[f"drug_{drug}_active_at_landmark"] = drug_at_landmark(drug_groups.get((sid, drug)), landmark_min)
        rows.append(row)
    f = pd.DataFrame(rows)
    d = d.merge(f, on=["sid", "landmark_h"], how="left", validate="one_to_one")

    predictors = numeric_c + categorical
    future_named = [c for c in predictors if any(x in c.lower() for x in ["future", "event", "outcome"])]
    missing_a = [c for c in numeric_a + categorical if c not in d.columns]
    missing_b = [c for c in numeric_b + categorical if c not in d.columns]

    # Frozen model compatibility check only. Probabilities remain in memory and are discarded.
    dry_run = {}
    for name, art in [("A", a_art), ("B", b_art)]:
        cols = art["numeric_predictors"] + art["categorical_predictors"]
        p = art["pipeline"].predict_proba(d[cols])[:, 1]
        dry_run[name] = {
            "n": int(len(p)), "all_finite": bool(np.isfinite(p).all()),
            "all_in_unit_interval": bool(((p >= 0) & (p <= 1)).all()),
        }
        del p

    y_obs = d.outcome_observed_6h.astype(int).to_numpy()
    groups = d.hospitalid.to_numpy()
    x = d[predictors]
    p_obs = np.full(len(d), np.nan)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_audit = []
    for fold, (train, test) in enumerate(cv.split(x, y_obs, groups=groups), start=1):
        pipe = make_observation_pipeline(numeric_c, categorical)
        pipe.fit(x.iloc[train], y_obs[train])
        p_obs[test] = pipe.predict_proba(x.iloc[test])[:, 1]
        fold_audit.append({
            "fold": fold, "train_n": int(len(train)), "test_n": int(len(test)),
            "train_hospitals": int(pd.Series(groups[train]).nunique()),
            "test_hospitals": int(pd.Series(groups[test]).nunique()),
            "train_observed_fraction": float(y_obs[train].mean()),
            "test_observed_fraction": float(y_obs[test].mean()),
        })
    assert np.isfinite(p_obs).all()
    numerator = float(y_obs.mean())
    observed = y_obs == 1
    stabilized = numerator / p_obs[observed]
    lo, hi = np.quantile(stabilized, [.01, .99])
    truncated = np.clip(stabilized, lo, hi)
    ess = float(truncated.sum() ** 2 / np.square(truncated).sum())
    ess_fraction = ess / len(truncated)

    weights = d[keys].copy()
    weights["p_outcome_observed_6h"] = p_obs
    weights["ipow_stabilized"] = np.where(observed, numerator / p_obs, np.nan)
    weights["ipow_stabilized_truncated"] = np.nan
    weights.loc[observed, "ipow_stabilized_truncated"] = truncated
    weights.to_csv(RESULTS / "stage_e1b_observation_weights.csv.gz", index=False, compression="gzip")

    feature_cols = ["sid", "uniquepid", "hospitalid", "landmark_h", *predictors]
    d[feature_cols].to_csv(WORK / "stage_e1b_external_features.csv.gz", index=False, compression="gzip")

    probability_summary = {
        "n": int(len(d)), "observed_n": int(observed.sum()), "observed_fraction": numerator,
        **quantile_row(p_obs, "p_"),
        "n_p_below_0_05": int((p_obs < .05).sum()),
    }
    weight_summary = {
        "n_observed": int(observed.sum()), "truncation_q01": float(lo), "truncation_q99": float(hi),
        **quantile_row(truncated, "w_"), "ess": ess, "ess_fraction": ess_fraction,
    }
    pd.DataFrame([probability_summary]).to_csv(RESULTS / "stage_e1b_observation_probability_summary.csv", index=False)
    pd.DataFrame([weight_summary]).to_csv(RESULTS / "stage_e1b_observation_weight_summary.csv", index=False)
    pd.DataFrame(fold_audit).to_csv(RESULTS / "stage_e1b_observation_folds.csv", index=False)

    feature_rows = []
    mimic_cols = list(dict.fromkeys([*numeric_c, *categorical, "interval_conservative_split", "primary_eligible_at_landmark", "model_bc_common_risk_set"]))
    mimic = pd.read_csv(MIMIC_FEATURES, usecols=mimic_cols)
    mimic_dev = mimic[
        mimic.interval_conservative_split.eq("definite_development_to_2019")
        & mimic.primary_eligible_at_landmark.astype(bool)
        & mimic.model_bc_common_risk_set.astype(bool)
    ]
    for col in numeric_c:
        for cohort, frame in [("MIMIC_development", mimic_dev), ("eICU_external", d)]:
            vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
            feature_rows.append({
                "feature": col, "cohort": cohort, "n": int(len(vals)),
                "missing_fraction": float(np.mean(~np.isfinite(vals))), **quantile_row(vals),
            })
    pd.DataFrame(feature_rows).to_csv(RESULTS / "stage_e1b_numeric_feature_transport.csv", index=False)
    category_rows = []
    for col in categorical:
        for cohort, frame in [("MIMIC_development", mimic_dev), ("eICU_external", d)]:
            counts = frame[col].fillna("<MISSING>").astype(str).value_counts(dropna=False)
            category_rows.extend({"feature": col, "cohort": cohort, "value": value, "n": int(n), "fraction": float(n / len(frame))} for value, n in counts.items())
    pd.DataFrame(category_rows).to_csv(RESULTS / "stage_e1b_categorical_feature_transport.csv", index=False)

    passed = (
        not d.duplicated(["sid", "landmark_h"]).any()
        and not missing_a and not missing_b and not future_named
        and all(v["all_finite"] and v["all_in_unit_interval"] for v in dry_run.values())
        and probability_summary["n_p_below_0_05"] == 0
        and ess_fraction >= .50
    )
    audit = {
        "protocol": "PROTOCOL_STAGE_E1B_FEATURE_TRANSPORT_LOCKED.md",
        "rows": int(len(d)), "patients": int(d.uniquepid.nunique()), "hospitals": int(d.hospitalid.nunique()),
        "unique_landmark_key": bool(~d.duplicated(["sid", "landmark_h"]).any()),
        "missing_model_A_columns": missing_a, "missing_model_B_columns": missing_b,
        "future_event_outcome_named_predictors": future_named,
        "clinical_event_columns_read": False,
        "clinical_predictions_saved": False,
        "dry_run_predictions_discarded": True,
        "dry_run": dry_run,
        "nurse_pamean_conflicts_all_eicu": int(pamean_conflicts),
        "nurse_svo2_conflicts_all_eicu": int(svo2_conflicts),
        "nurse_label_audit_available": bool(nurse_audit),
        "iabp": iabp_audit,
        "drug_carry_forward_cap_min": 60,
        "observation_model": "L2 logistic; five-fold StratifiedGroupKFold grouped by hospitalid",
        "observation_probability": probability_summary,
        "observation_weight": weight_summary,
    }
    (AUDIT / "stage_e1b_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    decision = {
        "decision": "PASS_E1B_UNLOCK_FROZEN_EXTERNAL_VALIDATION" if passed else "HOLD_EXTERNAL_VALIDATION",
        "gate_components": {
            "unique_landmark_key": audit["unique_landmark_key"],
            "all_A_B_columns_present": not missing_a and not missing_b,
            "no_future_event_outcome_predictors": not bool(future_named),
            "frozen_pipeline_dry_run_valid": all(v["all_finite"] and v["all_in_unit_interval"] for v in dry_run.values()),
            "rows_with_observation_probability_below_0_05": probability_summary["n_p_below_0_05"],
            "truncated_weight_ess_fraction": ess_fraction,
        },
        "clinical_performance_computed": False,
    }
    (RESULTS / "stage_e1b_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    report = f"""# eICU Stage E1B 特征运输与观测模型

**裁决：{decision['decision']}**

- 合法地标/患者/医院：{len(d):,}/{d.uniquepid.nunique():,}/{d.hospitalid.nunique()}
- A/B 所需列完整：{not missing_a and not missing_b}
- 冻结流水线兼容性 dry-run：{decision['gate_components']['frozen_pipeline_dry_run_valid']}
- 结局可观察比例：{numerator:.1%}
- 观测概率最小值/P1：{p_obs.min():.3f}/{np.quantile(p_obs, .01):.3f}
- 观测概率低于 0.05：{probability_summary['n_p_below_0_05']}
- 截尾权重 q01/q99：{lo:.3f}/{hi:.3f}
- 截尾后 ESS：{ess:.1f}（{ess_fraction:.1%}）
- 药物 carry-forward：60 分钟

本阶段没有读取低 CPO 事件列，没有保存 A/B 临床预测，也没有计算外部性能。
"""
    (ROOT / "REPORT_STAGE_E1B_ZH.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
