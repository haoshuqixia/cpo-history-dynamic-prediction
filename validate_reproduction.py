"""Read-only numerical and fold QA; write summary reports only in private work.

Reference directories must be authorized local research outputs. This audit
does not refit models, change estimates, or export individual observations.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import ast
import csv
import importlib.util
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


STAGES = [
    "CV_PAC_CPO_LANDMARK_AUDIT_20260908",
    "CV_PAC_CPO_STAGE0_MECHANISM_20260908",
    "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908",
    "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908",
    "CV_PAC_CPO_STAGE1C_EARLY_WARNING_AUDIT_20260908",
    "CV_PAC_CPO_STAGE1D_ALERT_UTILITY_20260908",
    "CV_PAC_CPO_REVIEWER_DEFENSE_20260908",
]


def compare_csv(reference, actual):
    if not actual.is_file():
        return "MISSING", 0, None
    left, right = pd.read_csv(reference), pd.read_csv(actual)
    if left.shape != right.shape or left.columns.tolist() != right.columns.tolist():
        return "SHAPE_OR_COLUMNS_DIFFER", len(right), None
    maximum = 0.0
    for column in left:
        a, b = left[column], right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            if pd.api.types.is_integer_dtype(a) and pd.api.types.is_integer_dtype(b):
                ok = np.array_equal(av, bv, equal_nan=True)
            else:
                ok = np.allclose(av, bv, rtol=1e-10, atol=1e-12, equal_nan=True)
            finite = np.isfinite(av) & np.isfinite(bv)
            if finite.any():
                maximum = max(maximum, float(np.max(np.abs(av[finite] - bv[finite]))))
        else:
            ok = a.fillna("__MISSING__").astype(str).equals(b.fillna("__MISSING__").astype(str))
        if not ok:
            return f"DIFFERENCE:{column}", len(right), maximum
    return "PASS", len(right), maximum


def load_module(path, name):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fold_checks(work):
    path = work / "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908/results/LANDMARK_FEATURES_V1.csv.gz"
    features = pd.read_csv(path)
    eligible = features.primary_eligible_at_landmark.astype(bool)
    common = eligible & features.primary_outcome_observed_6h.astype(bool) & features.model_bc_common_risk_set.astype(bool)
    dev = features.interval_conservative_split.eq("definite_development_to_2019")
    temporal = features.interval_conservative_split.eq("definite_temporal_2020_plus")
    assert not set(features.loc[dev, "subject_id"]) & set(features.loc[temporal, "subject_id"])
    rows = []
    for name, selected, target in [
        ("MIMIC observation patient folds", dev & eligible, "primary_outcome_observed_6h"),
        ("MIMIC clinical-model patient folds", dev & common, "primary_outcome_6h"),
    ]:
        frame = features.loc[selected]
        groups = frame.subject_id.to_numpy()
        y = frame[target].astype(int).to_numpy()
        covered = np.zeros(len(frame), dtype=int)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260908)
        for train, test in cv.split(frame, y, groups=groups):
            assert not set(groups[train]) & set(groups[test])
            covered[test] += 1
        assert np.all(covered == 1)
        rows.append({"check": name, "status": "PASS", "rows": len(frame), "groups": len(set(groups)), "folds": 5})
    eicu = pd.read_csv(work / "CPO_EICU_EXTERNAL_DISCHARGE_BOUNDARY_R1_V1_0/results/stage_e0_landmarks.csv.gz")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260908)
    covered = np.zeros(len(eicu), dtype=int)
    groups = eicu.hospitalid.to_numpy()
    for train, test in cv.split(eicu, eicu.outcome_observed_6h.astype(int), groups=groups):
        assert not set(groups[train]) & set(groups[test])
        covered[test] += 1
    assert np.all(covered == 1)
    rows.append({"check": "eICU observation hospital folds", "status": "PASS", "rows": len(eicu), "groups": len(set(groups)), "folds": 5})
    # Verify the pipeline is created and fitted inside each fold, before OOF
    # probabilities are used for development-only calibration.
    model = work / "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908/code/run_stage1b.py"
    tree = ast.parse(model.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    loops = [node for node in ast.walk(main) if isinstance(node, ast.For) and ast.unparse(node.target) == "(train_pos, test_pos)"]
    assert len(loops) == 1
    body = ast.unparse(loops[0])
    assert "model = make_model(numeric)" in body and "xdev.iloc[train_pos]" in body
    assert "ydev.iloc[train_pos]" in body and "xdev.iloc[test_pos]" in body
    assert "weighted_calibration(ydev.to_numpy(), oof, sw)" in ast.unparse(main)
    rows.append({"check": "Clinical preprocessing inside folds and development-only OOF calibration (source audit)", "status": "PASS"})
    return rows


def history_clock_checks(work):
    """Independent summaries from timestamp-selected haemodynamic records.

    The primary clock is charttime. Source derivation/mapping is shared with
    the frozen pipeline; history-window summary arithmetic is independent.
    This is not a claim of proven real-time storetime availability.
    """
    stage0 = load_module(work / "CV_PAC_CPO_STAGE0_MECHANISM_20260908/code/run_stage0_mechanism_audit.py", "qa_axes")
    features = pd.read_csv(work / "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908/results/LANDMARK_FEATURES_V1.csv.gz", dtype={"stay_id": str})
    stays = pd.read_csv(work / "STAGE_MINUS1A_CV_OBSERVABILITY_V1/work/adult_stays.csv", dtype={"stay_id": str}, parse_dates=["intime", "outtime"])
    stays = stays[stays.stay_id.isin(set(features.stay_id))]
    axes = stage0.build_axis_times(stays)
    total = 0
    maximum = 0.0
    for axis, frame, value in zip(("cpo", "mpap", "svo2"), axes, ("cpo", "v", "v")):
        groups = {key: (g.hours.to_numpy(float), g[value].to_numpy(float))
                  for key, g in frame.sort_values("hours").groupby("stay_id")}
        for row in features.itertuples(index=False):
            t, v = groups.get(row.stay_id, (np.array([]), np.array([])))
            included = (t > row.landmark_h - 4) & (t <= row.landmark_h)
            times, values = t[included], v[included]
            if not len(values):
                assert pd.isna(getattr(row, f"{axis}_current"))
                continue
            assert times.max() <= row.landmark_h and times.min() > row.landmark_h - 4
            centered = times - times.mean()
            slope = float(np.dot(centered, values - values.mean()) / np.dot(centered, centered)) if np.dot(centered, centered) > 0 else np.nan
            summary = {"current": values[-1], "mean_4h": values.mean(), "min_4h": values.min(),
                       "max_4h": values.max(), "sd_4h": values.std(ddof=0), "slope_4h": slope,
                       "delta_4h": values[-1] - values[0], "span_4h": times[-1] - times[0]}
            for suffix, expected in summary.items():
                actual = getattr(row, f"{axis}_{suffix}")
                # Polyfit can return ~1e-11 instead of zero for a constant
                # SvO2 series spanning only five minutes. Closed-form OLS
                # removes that cancellation artifact. This tolerance applies
                # only to the independent arithmetic check, not reference CSVs.
                atol = 1e-10 if suffix == "slope_4h" else 1e-12
                assert np.isclose(expected, actual, rtol=1e-10, atol=atol, equal_nan=True), axis + "_" + suffix
                if np.isfinite(expected) and np.isfinite(actual):
                    maximum = max(maximum, abs(float(expected) - float(actual)))
                total += 1
    return {"check": "MIMIC primary-clock history summaries independently reconstructed from (L-4,L]", "status": "PASS", "numeric_cells": total, "max_abs_difference": maximum, "slope_absolute_arithmetic_tolerance": 1e-10}


def external_history_clock_checks(work):
    stage = work / "CV_PAC_CPO_EICU_STAGE_EMINUS1_20260909"
    source = load_module(stage / "code/run_eicu_stage_eminus1.py", "qa_external_mapping")
    features = pd.read_csv(work / "CPO_EICU_EXTERNAL_DISCHARGE_BOUNDARY_R1_V1_0/work/stage_e1b_external_features.csv.gz")
    nurse, _ = source.read_nurse_targets(stage / "work/nurse_cardiac_targets.csv.gz")
    mpap, _ = source.resolve(nurse["pamean"], 0, 150, entry=True)
    svo2, _ = source.resolve(nurse["svo2"], 0, 100, entry=True)
    cpo = pd.read_csv(stage / "work/cpo_nurse.csv.gz", usecols=["sid", "offset", "available", "cpo"])
    total = 0
    for axis, frame, value, availability in [("cpo", cpo, "cpo", "available"),
                                            ("mpap", mpap, "value", "entry"),
                                            ("svo2", svo2, "value", "entry")]:
        groups = {int(key): (g.offset.to_numpy(float) / 60, g[availability].to_numpy(float) / 60, g[value].to_numpy(float))
                  for key, g in frame.sort_values(["offset", availability]).groupby("sid")}
        for row in features.itertuples(index=False):
            times, available, values = groups.get(int(row.sid), (np.array([]), np.array([]), np.array([])))
            included = (times > row.landmark_h - 4) & (times <= row.landmark_h) & (available <= row.landmark_h)
            t, v = times[included], values[included]
            assert int(getattr(row, f"n_{axis}_history_4h")) == len(v)
            if not len(v):
                assert pd.isna(getattr(row, f"{axis}_current"))
                continue
            centered = t - t.mean()
            denominator = np.dot(centered, centered)
            summary = {"current": v[-1], "mean_4h": v.mean(), "min_4h": v.min(), "max_4h": v.max(),
                       "sd_4h": v.std(ddof=0), "slope_4h": np.dot(centered, v - v.mean()) / denominator if denominator > 0 else np.nan,
                       "delta_4h": v[-1] - v[0], "span_4h": t[-1] - t[0]}
            for suffix, expected in summary.items():
                atol = 1e-10 if suffix == "slope_4h" else 1e-12
                assert np.isclose(expected, getattr(row, f"{axis}_{suffix}"), rtol=1e-10, atol=atol, equal_nan=True), axis + "_" + suffix
                total += 1
    return {"check": "eICU history summaries independently reconstructed using measurement and availability clocks <=L", "status": "PASS", "numeric_cells": total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-mimic", type=Path)
    parser.add_argument("--reference-eicu", type=Path)
    args = parser.parse_args()
    work = Path(os.environ["CPO_WORK_DIR"]).resolve()
    output = work / "reproduction_qa"
    output.mkdir(exist_ok=True)
    checks = []
    if bool(args.reference_mimic) != bool(args.reference_eicu):
        parser.error("Provide both authorized reference directories, or neither")
    reference_sets = [(args.reference_mimic, STAGES), (args.reference_eicu.parent, [args.reference_eicu.name])] if args.reference_mimic else []
    for reference_base, groups in reference_sets:
        for stage in groups:
            reference_stage = reference_base / stage
            for reference in sorted(reference_stage.rglob("*.csv*")):
                relative = reference.relative_to(reference_base)
                if "results" not in relative.parts:
                    continue
                if reference.name in {"EICU_R1_INDEPENDENT_QA.csv", "EICU_R1_HASH_AUDIT.csv"}:
                    # Historical QA attestations and hashes are provenance,
                    # not numerical outputs of the R1 statistical runner.
                    continue
                status, rows, maximum = compare_csv(reference, work / relative)
                checks.append({"file": str(relative), "status": status, "rows": rows, "max_abs_difference": maximum})
    pd.DataFrame(checks).to_csv(output / "numerical_equivalence.csv", index=False)
    audits = fold_checks(work)
    audits.append(history_clock_checks(work))
    audits.append(external_history_clock_checks(work))
    predictions = pd.read_csv(work / "CPO_EICU_EXTERNAL_DISCHARGE_BOUNDARY_R1_V1_0/results/stage_e2_external_predictions.csv.gz")
    counts = {"landmarks": len(predictions), "patients": predictions.uniquepid.nunique(),
              "stays": predictions.sid.nunique(), "hospitals": predictions.hospitalid.nunique(),
              "event_landmarks": int(predictions.event_6h.sum()),
              "event_patients": predictions.loc[predictions.event_6h.astype(bool), "uniquepid"].nunique()}
    assert counts == {"landmarks": 7753, "patients": 840, "stays": 850, "hospitals": 27, "event_landmarks": 469, "event_patients": 157}
    audits.append({"check": "Final eICU R1 denominators", "status": "PASS", **counts})
    summary = {"reference_comparison": "COMPLETED" if reference_sets else "NOT_RUN_NO_REFERENCES",
               "numerical_files": len(checks), "passed": sum(row["status"] == "PASS" for row in checks),
               "failed": [row for row in checks if row["status"] != "PASS"], "audits": audits}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=int))
    if reference_sets:
        print(f"Numerical equivalence: {summary['passed']}/{summary['numerical_files']} files PASS")
    else:
        print("Frozen-reference numerical comparison NOT RUN: reference outputs were not supplied.")
    print("Fold and primary-clock history audits completed; patient-level records not exported.")
    if summary["failed"]:
        raise RuntimeError("Result differences found. Inspect private QA; no technical seal may be asserted.")


if __name__ == "__main__":
    main()
