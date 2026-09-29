#!/usr/bin/env python3
"""Rerun only the eICU external-validation chain after ICU-exit boundary correction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
OLD = BASE / "CV_PAC_CPO_EICU_STAGE_EMINUS1_20260909"
ORIGINAL_CODE = OLD / "code"
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"
WORK = ROOT / "work"
MODELS = BASE / "CV_PAC_CPO_PAPER_PACKAGE_20260908" / "models"
TEMPORAL = BASE / "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908" / "results" / "temporal_predictions.csv.gz"
KEYS = ["sid", "uniquepid", "hospitalid", "landmark_h"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_original(name: str):
    path = ORIGINAL_CODE / f"{name}.py"
    sys.path.insert(0, str(ORIGINAL_CODE))
    spec = importlib.util.spec_from_file_location(f"r1_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_hashes() -> dict[str, str]:
    files = {
        "model_A_locked": MODELS / "model_A_locked.joblib",
        "model_B_locked": MODELS / "model_B_locked.joblib",
        "model_C_locked": MODELS / "model_C_locked.joblib",
        "MIMIC_temporal_predictions": TEMPORAL,
    }
    return {name: sha256(path) for name, path in files.items()}


def prepare_work() -> None:
    for directory in [RESULTS, AUDIT, WORK]:
        directory.mkdir(parents=True, exist_ok=True)
    for name in [
        "candidate_patient_metadata.csv.gz",
        "cpo_nurse.csv.gz",
        "nurse_cardiac_targets.csv.gz",
        "formal_cohort_infusion_states.csv.gz",
    ]:
        link = WORK / name
        target = OLD / "work" / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"Unexpected existing input link: {link}")
        elif link.exists():
            raise RuntimeError(f"Refusing to replace existing input: {link}")
        else:
            link.symlink_to(target)


def build_corrected_labels() -> dict:
    original = pd.read_csv(OLD / "results" / "stage_e0_landmarks.csv.gz")
    if len(original) != 12_971 or original.duplicated(["sid", "landmark_h"]).any():
        raise RuntimeError("Original eICU risk-set count or key does not match the protocol")
    patients = pd.read_csv(WORK / "candidate_patient_metadata.csv.gz", usecols=["sid", "los_min"])
    cpo = pd.read_csv(WORK / "cpo_nurse.csv.gz", usecols=["sid", "offset", "available", "cpo"])
    relevant_sids = set(original.sid.astype(int))
    cpo = cpo[cpo.sid.isin(relevant_sids)].merge(patients, on="sid", validate="many_to_one")
    post_discharge_records = int(cpo.offset.gt(cpo.los_min).sum())
    cpo = cpo[cpo.offset.le(cpo.los_min)].sort_values(["sid", "offset", "available"])
    groups = {
        int(sid): (
            group.offset.to_numpy(float) / 60,
            group.available.to_numpy(float) / 60,
            group.cpo.to_numpy(float),
        )
        for sid, group in cpo.groupby("sid", sort=False)
    }
    source = load_original("run_eicu_stage_e0")
    rows = []
    for old in original.itertuples(index=False):
        times, available, values = groups[int(old.sid)]
        landmark = float(old.landmark_h)
        mcs = float(old.future_mcs_censor_h) if pd.notna(old.future_mcs_censor_h) else np.inf
        event = source.event_time(times, values, landmark, min(landmark + 6, mcs))
        future = times[(times > landmark) & (times <= landmark + 6)]
        dense_negative = False
        if event is None and mcs >= landmark + 6 and len(future) >= 4 and future[-1] >= landmark + 5:
            safe_hist = times[(times > landmark - 4) & (times <= landmark) & (available <= landmark)]
            if not len(safe_hist):
                raise RuntimeError(f"No available current CPO for {old.sid}, L={landmark}")
            dense_negative = bool(np.diff(np.r_[safe_hist[-1], future]).max() <= 2)
        rows.append({
            **{key: getattr(old, key) for key in KEYS},
            "outcome_observed_6h": event is not None or dense_negative,
            "event_6h": event is not None,
            "event_time_h": event,
            "future_mcs_censor_h": old.future_mcs_censor_h,
            "unitDischargeOffset": float(patients.set_index("sid").loc[int(old.sid), "los_min"]),
            "last_CPO_before_discharge_h": float(times[-1]) if len(times) else np.nan,
            "last_future_CPO_before_discharge_h": float(future[-1]) if len(future) else np.nan,
        })
    updated = pd.DataFrame(rows)
    labels = updated[[*KEYS, "outcome_observed_6h", "event_6h", "event_time_h", "future_mcs_censor_h"]]
    labels.to_csv(RESULTS / "stage_e0_landmarks.csv.gz", index=False, compression="gzip")
    labels.to_csv(RESULTS / "EICU_R1_LANDMARK_LABELS.csv", index=False)
    audit = original.merge(updated, on=KEYS, suffixes=("_old", "_R1"), validate="one_to_one")
    changed = audit[
        audit.outcome_observed_6h_old.ne(audit.outcome_observed_6h_R1)
        | audit.event_6h_old.ne(audit.event_6h_R1)
    ].copy()
    old_cpo = pd.read_csv(OLD / "work" / "cpo_nurse.csv.gz", usecols=["sid", "offset"])
    old_future_last = {}
    for row in changed.itertuples(index=False):
        t = old_cpo.loc[old_cpo.sid.eq(row.sid), "offset"].to_numpy(float) / 60
        f = t[(t > row.landmark_h) & (t <= row.landmark_h + 6)]
        old_future_last[(row.sid, row.landmark_h)] = float(f[-1]) if len(f) else np.nan
    report = pd.DataFrame({
        "landmark_id": changed.sid.astype(str) + "_L" + changed.landmark_h.astype(str),
        "sid": changed.sid,
        "hospitalid": changed.hospitalid,
        "landmark_h": changed.landmark_h,
        "old_label": np.where(changed.event_6h_old, "event", np.where(changed.outcome_observed_6h_old, "event_free", "unobserved")),
        "new_label": np.where(changed.event_6h_R1, "event", np.where(changed.outcome_observed_6h_R1, "event_free", "unobserved")),
        "old_observed": changed.outcome_observed_6h_old,
        "new_observed": changed.outcome_observed_6h_R1,
        "unitDischargeOffset_min": changed.unitDischargeOffset,
        "last_CPO_before_discharge_h": changed.last_future_CPO_before_discharge_h,
        "old_last_CPO_used_h": [old_future_last[(r.sid, r.landmark_h)] for r in changed.itertuples(index=False)],
        "reason": "Post-ICU-discharge CPO used for dense-negative ascertainment",
    })
    report.to_csv(RESULTS / "EICU_R1_LABEL_CHANGE_AUDIT.csv", index=False)
    summary = {
        "risk_set_landmarks": len(labels),
        "old_observed": int(original.outcome_observed_6h.sum()),
        "R1_observed": int(labels.outcome_observed_6h.sum()),
        "old_events": int(original.event_6h.sum()),
        "R1_events": int(labels.event_6h.sum()),
        "changed_labels": len(changed),
        "post_discharge_cpo_records_in_risk_set_stays": post_discharge_records,
        "all_outcome_used_cpo_at_or_before_discharge": bool(cpo.offset.le(cpo.los_min).all()),
    }
    pd.DataFrame([summary]).to_csv(RESULTS / "EICU_R1_OUTCOME_OBSERVATION_SUMMARY.csv", index=False)
    if not (
        summary["risk_set_landmarks"] == 12_971
        and summary["old_observed"] == 7_754
        and summary["R1_observed"] == 7_753
        and summary["old_events"] == summary["R1_events"] == 469
        and summary["changed_labels"] == 1
        and report.iloc[0]["old_label"] == "event_free"
        and report.iloc[0]["new_label"] == "unobserved"
    ):
        raise RuntimeError(f"R1 label gate failed: {summary}")
    return summary


def configure(module):
    module.ROOT = ROOT
    module.RESULTS = RESULTS
    module.AUDIT = AUDIT
    if hasattr(module, "WORK"):
        module.WORK = WORK
    if hasattr(module, "MODELS"):
        module.MODELS = MODELS
    return module


def copy_named_outputs() -> None:
    sources = {
        "EICU_R1_OBSERVATION_WEIGHTS.csv": RESULTS / "stage_e1b_observation_weights_final.csv.gz",
        "EICU_R1_EXTERNAL_PERFORMANCE.csv": RESULTS / "stage_e2_absolute_metrics.csv",
        "EICU_R1_EXTERNAL_BOOTSTRAP_CI.csv": RESULTS / "stage_e2_primary_comparison_ci.csv",
        "EICU_R1_HOSPITAL_BRIER.csv": RESULTS / "stage_e2_hospital_heterogeneity.csv",
    }
    for target_name, source in sources.items():
        pd.read_csv(source).to_csv(RESULTS / target_name, index=False)
    pd.read_csv(RESULTS / "stage_e2_model_comparisons.csv").to_csv(
        RESULTS / "EICU_R1_EXTERNAL_COMPARISONS.csv", index=False
    )
    comparison = pd.read_csv(RESULTS / "stage_e2_model_comparisons.csv")
    comparison[comparison.analysis.eq("sensitivity_unweighted")].to_csv(
        RESULTS / "EICU_R1_UNWEIGHTED_SENSITIVITY.csv", index=False
    )
    plot = load_original("run_eicu_stage_e2")
    predictions = pd.read_csv(RESULTS / "stage_e2_external_predictions.csv.gz")
    calibration = []
    for model in ["A", "B"]:
        pred = predictions[f"prediction_{model}"]
        for decile, (_, frame) in enumerate(predictions.assign(_bin=pd.qcut(pred, 10, duplicates="drop")).groupby("_bin", observed=True), 1):
            w = frame.weight_primary.to_numpy(float)
            calibration.append({
                "model": model, "decile": decile, "n": len(frame),
                "mean_predicted": float(np.average(frame[f"prediction_{model}"], weights=w)),
                "observed_fraction": float(np.average(frame.event_6h.astype(float), weights=w)),
            })
    pd.DataFrame(calibration).to_csv(RESULTS / "EICU_R1_CALIBRATION.csv", index=False)


def old_vs_new() -> None:
    old_abs = pd.read_csv(OLD / "results" / "stage_e2_absolute_metrics.csv")
    new_abs = pd.read_csv(RESULTS / "stage_e2_absolute_metrics.csv")
    old_comp = pd.read_csv(OLD / "results" / "stage_e2_model_comparisons.csv")
    new_comp = pd.read_csv(RESULTS / "stage_e2_model_comparisons.csv")
    old_abs["quantity"] = old_abs.model + "_" + old_abs.metric
    new_abs["quantity"] = new_abs.model + "_" + new_abs.metric
    old_comp["quantity"] = old_comp.metric
    new_comp["quantity"] = new_comp.metric
    left = pd.concat([old_abs, old_comp])[["analysis", "quantity", "estimate"]].rename(columns={"estimate": "old"})
    right = pd.concat([new_abs, new_comp])[["analysis", "quantity", "estimate"]].rename(columns={"estimate": "R1"})
    comparison = left.merge(right, on=["analysis", "quantity"], validate="one_to_one")
    comparison["delta_R1_minus_old"] = comparison.R1 - comparison.old
    comparison.to_csv(RESULTS / "EICU_R1_OLD_VS_NEW_METRICS.csv", index=False)


def main() -> None:
    prepare_work()
    before = source_hashes()
    summary = build_corrected_labels()
    configure(load_original("run_eicu_stage_e1b")).main()
    configure(load_original("run_eicu_stage_e1b_remediation")).main()
    configure(load_original("run_eicu_stage_e2")).main()
    copy_named_outputs()
    old_vs_new()
    after = source_hashes()
    hashes = {key: {"before": before[key], "after": after[key], "unchanged": before[key] == after[key]} for key in before}
    (AUDIT / "EICU_R1_HASH_AUDIT.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    if not all(row["unchanged"] for row in hashes.values()):
        raise RuntimeError("A frozen MIMIC artifact changed during R1")
    decision = json.loads((RESULTS / "stage_e2_decision.json").read_text())
    (ROOT / "R1_RUN_SUMMARY.json").write_text(json.dumps({"label_summary": summary, "external_decision": decision}, indent=2), encoding="utf-8")
    print(json.dumps({"label_summary": summary, "external_decision": decision}, indent=2))


if __name__ == "__main__":
    main()
