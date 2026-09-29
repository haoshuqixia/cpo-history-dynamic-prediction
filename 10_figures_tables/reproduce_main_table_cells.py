"""Reproduce final main-table numeric cells as CSV, without editing a manuscript.

Table-summary functions are projected from the existing main-table generator.
Only input locations, the final eICU R1 denominator assertion and two labels
are updated. No new statistical analysis is defined.
"""

from __future__ import annotations

from pathlib import Path

import argparse, csv, os

from zipfile import ZipFile

from xml.etree import ElementTree as ET

import pandas as pd

ROOT = Path(os.environ["CPO_WORK_DIR"]).resolve()

BASE = ROOT

MIMIC_FEATURES = BASE / "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908" / "results" / "LANDMARK_FEATURES_V1.csv.gz"

MIMIC_STAYS = BASE / "STAGE_MINUS1A_CV_OBSERVABILITY_V1" / "work" / "adult_stays.csv"

TEMPORAL_METRICS = BASE / "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908" / "results" / "temporal_metrics.csv"

TEMPORAL_DELTAS = BASE / "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908" / "results" / "paired_temporal_deltas.csv"

EICU_FEATURES = BASE / "CPO_EICU_EXTERNAL_DISCHARGE_BOUNDARY_R1_V1_0" / "work" / "stage_e1b_external_features.csv.gz"

EICU_PREDICTIONS = BASE / "CPO_EICU_EXTERNAL_DISCHARGE_BOUNDARY_R1_V1_0" / "results" / "stage_e2_external_predictions.csv.gz"

EICU_METRICS = BASE / "CPO_EICU_EXTERNAL_DISCHARGE_BOUNDARY_R1_V1_0" / "results" / "stage_e2_absolute_metrics.csv"

EICU_DELTAS = BASE / "CPO_EICU_EXTERNAL_DISCHARGE_BOUNDARY_R1_V1_0" / "results" / "stage_e2_primary_comparison_ci.csv"

DEV = "definite_development_to_2019"

TEMP = "definite_temporal_2020_plus"

EM_DASH = "—"

VASOPRESSOR_COLUMNS = [
    "drug_norepinephrine_active_at_landmark",
    "drug_epinephrine_active_at_landmark",
    "drug_vasopressin_active_at_landmark",
    "drug_phenylephrine_active_at_landmark",
    "drug_dopamine_active_at_landmark",
]

INOTROPE_COLUMNS = [
    "drug_epinephrine_active_at_landmark",
    "drug_dopamine_active_at_landmark",
    "drug_dobutamine_active_at_landmark",
    "drug_milrinone_active_at_landmark",
]

def qsummary(series: pd.Series, digits: int) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    q1, median, q3 = values.quantile([0.25, 0.50, 0.75])
    return f"{median:.{digits}f} ({q1:.{digits}f}–{q3:.{digits}f})"

def nsummary(mask: pd.Series, denominator: int | None = None) -> str:
    values = mask.fillna(False).astype(bool)
    total = len(values) if denominator is None else denominator
    count = int(values.sum())
    return f"{count:,} ({100 * count / total:.1f})"

def count_percent(count: int, total: int) -> str:
    return f"{count:,} ({100 * count / total:.1f})"

def add_drug_groups(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["any_vasopressor"] = result[VASOPRESSOR_COLUMNS].fillna(0).astype(bool).any(axis=1)
    result["any_inotrope"] = result[INOTROPE_COLUMNS].fillna(0).astype(bool).any(axis=1)
    return result

def add_icu_groups(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    care = result["first_careunit"].fillna("")
    result["cardiac_icu"] = care.isin(
        ["Cardiac Vascular Intensive Care Unit (CVICU)", "Coronary Care Unit (CCU)"]
    )
    result["medical_icu"] = care.isin(
        ["Medical Intensive Care Unit (MICU)", "Intensive Care Unit (ICU)"]
    )
    result["surgical_icu"] = care.isin(
        [
            "Surgical Intensive Care Unit (SICU)",
            "Trauma SICU (TSICU)",
            "Neuro Surgical Intensive Care Unit (Neuro SICU)",
        ]
    )
    result["other_icu"] = ~(result["cardiac_icu"] | result["medical_icu"] | result["surgical_icu"])
    return result

def load_cohorts() -> dict[str, dict[str, pd.DataFrame]]:
    mimic = pd.read_csv(MIMIC_FEATURES, dtype={"stay_id": str})
    eligible = (
        mimic["primary_eligible_at_landmark"].astype(bool)
        & mimic["primary_outcome_observed_6h"].astype(bool)
        & mimic["model_bc_common_risk_set"].astype(bool)
    )
    mimic = add_icu_groups(add_drug_groups(mimic.loc[eligible].copy()))

    stays = pd.read_csv(MIMIC_STAYS, dtype={"stay_id": str}, parse_dates=["intime"])
    stays = stays[["stay_id", "intime"]].drop_duplicates("stay_id")

    mimic_groups: dict[str, dict[str, pd.DataFrame]] = {}
    for label, split in [("development", DEV), ("temporal", TEMP)]:
        landmarks = mimic.loc[mimic["interval_conservative_split"].eq(split)].copy()
        timed = landmarks.merge(stays, on="stay_id", how="left", validate="many_to_one")
        timed["landmark_datetime"] = timed["intime"] + pd.to_timedelta(timed["landmark_h"], unit="h")
        events = timed.groupby("subject_id")["primary_outcome_6h"].max().rename("event_patient")
        patients = (
            timed.sort_values(["subject_id", "landmark_datetime", "stay_id"])
            .drop_duplicates("subject_id")
            .merge(events, on="subject_id", how="left", validate="one_to_one")
        )
        mimic_groups[label] = {"landmarks": landmarks, "patients": patients}

    eicu_predictions = pd.read_csv(EICU_PREDICTIONS)
    eicu_features = pd.read_csv(EICU_FEATURES)
    keys = ["sid", "uniquepid", "hospitalid", "landmark_h"]
    eicu = eicu_predictions.merge(eicu_features, on=keys, how="left", validate="one_to_one")
    if eicu["cpo_current"].isna().any():
        raise ValueError("External feature merge produced missing CPO values")
    eicu = add_icu_groups(add_drug_groups(eicu))
    eicu_events = eicu.groupby("uniquepid")["event_6h"].max().rename("event_patient")
    eicu_patients = (
        eicu.sort_values(["uniquepid", "sid", "landmark_h"])
        .drop_duplicates("uniquepid")
        .merge(eicu_events, on="uniquepid", how="left", validate="one_to_one")
    )

    cohorts = {
        **mimic_groups,
        "external": {"landmarks": eicu, "patients": eicu_patients},
    }
    assert len(cohorts["development"]["landmarks"]) == 21_619
    assert len(cohorts["temporal"]["landmarks"]) == 3_416
    assert len(cohorts["external"]["landmarks"]) == 7_753
    assert cohorts["development"]["patients"]["subject_id"].nunique() == 1_544
    assert cohorts["temporal"]["patients"]["subject_id"].nunique() == 265
    assert cohorts["external"]["patients"]["uniquepid"].nunique() == 840
    return cohorts

def table1_rows(cohorts: dict[str, dict[str, pd.DataFrame]]) -> list[tuple[str, str, str, str]]:
    dev_l = cohorts["development"]["landmarks"]
    temp_l = cohorts["temporal"]["landmarks"]
    ext_l = cohorts["external"]["landmarks"]
    dev_p = cohorts["development"]["patients"]
    temp_p = cohorts["temporal"]["patients"]
    ext_p = cohorts["external"]["patients"]

    def patient_rows(label: str, data: pd.DataFrame) -> str:
        if label == "age":
            return qsummary(data["age"], 1)
        if label == "female":
            return nsummary(data["gender"].eq("F"))
        return nsummary(data[label])

    def landmark_rows(label: str, data: pd.DataFrame, digits: int | None = None) -> str:
        if digits is not None:
            return qsummary(data[label], digits)
        return nsummary(data[label])

    return [
        ("__SECTION__", "Study sample", "", ""),
        ("Patients, n", f"{len(dev_p):,}", f"{len(temp_p):,}", f"{len(ext_p):,}"),
        ("ICU stays, n", f"{dev_l['stay_id'].nunique():,}", f"{temp_l['stay_id'].nunique():,}", f"{ext_l['sid'].nunique():,}"),
        ("Hospitals, n", EM_DASH, EM_DASH, f"{ext_l['hospitalid'].nunique():,}"),
        ("Outcome-observed landmarks, n", f"{len(dev_l):,}", f"{len(temp_l):,}", f"{len(ext_l):,}"),
        (
            "Event-positive landmarks, n (%)",
            count_percent(int(dev_l["primary_outcome_6h"].sum()), len(dev_l)),
            count_percent(int(temp_l["primary_outcome_6h"].sum()), len(temp_l)),
            count_percent(int(ext_l["event_6h"].sum()), len(ext_l)),
        ),
        (
            "Patients with ≥1 event, n (%)",
            nsummary(dev_p["event_patient"]),
            nsummary(temp_p["event_patient"]),
            nsummary(ext_p["event_patient"]),
        ),
        ("__SECTION__", "Patient characteristics", "", ""),
        ("Age, years", patient_rows("age", dev_p), patient_rows("age", temp_p), patient_rows("age", ext_p)),
        ("Female sex, n (%)", patient_rows("female", dev_p), patient_rows("female", temp_p), patient_rows("female", ext_p)),
        ("Medical ICU, n (%)", patient_rows("medical_icu", dev_p), patient_rows("medical_icu", temp_p), patient_rows("medical_icu", ext_p)),
        ("Cardiac ICU, n (%)", patient_rows("cardiac_icu", dev_p), patient_rows("cardiac_icu", temp_p), patient_rows("cardiac_icu", ext_p)),
        ("Surgical ICU, n (%)", patient_rows("surgical_icu", dev_p), patient_rows("surgical_icu", temp_p), patient_rows("surgical_icu", ext_p)),
        ("Other or mixed ICU, n (%)", patient_rows("other_icu", dev_p), patient_rows("other_icu", temp_p), patient_rows("other_icu", ext_p)),
        ("IABP active at first landmark, n (%)", nsummary(dev_p["iabp_active_at_landmark"]), nsummary(temp_p["iabp_active_at_landmark"]), nsummary(ext_p["iabp_active_at_landmark"])),
        ("Any vasopressor active, n (%)", nsummary(dev_p["any_vasopressor"]), nsummary(temp_p["any_vasopressor"]), nsummary(ext_p["any_vasopressor"])),
        ("Any inotrope active, n (%)", nsummary(dev_p["any_inotrope"]), nsummary(temp_p["any_inotrope"]), nsummary(ext_p["any_inotrope"])),
        ("__SECTION__", "Landmark characteristics", "", ""),
        ("Landmark time from ICU admission, h", landmark_rows("landmark_h", dev_l, 1), landmark_rows("landmark_h", temp_l, 1), landmark_rows("landmark_h", ext_l, 1)),
        ("Latest CPO, W", landmark_rows("cpo_current", dev_l, 2), landmark_rows("cpo_current", temp_l, 2), landmark_rows("cpo_current", ext_l, 2)),
        ("CPO measurements in preceding 4 h, n", landmark_rows("n_cpo_history_4h", dev_l, 1), landmark_rows("n_cpo_history_4h", temp_l, 1), landmark_rows("n_cpo_history_4h", ext_l, 1)),
        ("CPO observation span in preceding 4 h, h", landmark_rows("cpo_span_4h", dev_l, 1), landmark_rows("cpo_span_4h", temp_l, 1), landmark_rows("cpo_span_4h", ext_l, 1)),
        ("Latest mPAP in preceding 4 h available, n (%)", nsummary(dev_l["mpap_current"].notna()), nsummary(temp_l["mpap_current"].notna()), nsummary(ext_l["mpap_current"].notna())),
        ("Latest SvO₂ in preceding 4 h available, n (%)", nsummary(dev_l["svo2_current"].notna()), nsummary(temp_l["svo2_current"].notna()), nsummary(ext_l["svo2_current"].notna())),
        ("IABP active at landmark, n (%)", nsummary(dev_l["iabp_active_at_landmark"]), nsummary(temp_l["iabp_active_at_landmark"]), nsummary(ext_l["iabp_active_at_landmark"])),
        ("Any vasopressor active at landmark, n (%)", nsummary(dev_l["any_vasopressor"]), nsummary(temp_l["any_vasopressor"]), nsummary(ext_l["any_vasopressor"])),
        ("Any inotrope active at landmark, n (%)", nsummary(dev_l["any_inotrope"]), nsummary(temp_l["any_inotrope"]), nsummary(ext_l["any_inotrope"])),
    ]

def table2_rows() -> list[tuple[str, str, str, str, str, str, str]]:
    temporal_metrics = pd.read_csv(TEMPORAL_METRICS).set_index(["model", "metric"])
    temporal_deltas = pd.read_csv(TEMPORAL_DELTAS)
    temporal_deltas = temporal_deltas.loc[temporal_deltas["comparison"].eq("B-A")].set_index("metric")

    eicu_metrics = pd.read_csv(EICU_METRICS)
    eicu_metrics = eicu_metrics.loc[eicu_metrics["analysis"].eq("primary_ipow_floor_and_q01q99")].set_index(["model", "metric"])
    eicu_deltas = pd.read_csv(EICU_DELTAS).set_index("metric")

    specs = [
        ("Brier score‡", "brier", "delta_brier_B_minus_A", 5, 6, 5),
        ("AUROC", "auroc", "delta_auroc_B_minus_A", 4, 4, 4),
        ("Average precision", "average_precision", "delta_average_precision_B_minus_A", 4, 4, 4),
        ("Log loss", "log_loss", "delta_log_loss_B_minus_A", 5, 5, 5),
        ("Calibration intercept", "calibration_intercept", None, 3, None, 3),
        ("Calibration slope", "calibration_slope", None, 3, None, 3),
    ]

    def number(value: float, digits: int) -> str:
        return f"{value:.{digits}f}"

    def signed(value: float, digits: int) -> str:
        if value > 0:
            return f"+{value:.{digits}f}"
        if value < 0:
            return f"−{abs(value):.{digits}f}"
        return number(value, digits)

    rows = []
    for label, metric, eicu_delta_key, metric_digits, temporal_delta_digits, eicu_delta_digits in specs:
        temporal_a = float(temporal_metrics.loc[("A", metric), "estimate"])
        temporal_b = float(temporal_metrics.loc[("B", metric), "estimate"])
        eicu_a = float(eicu_metrics.loc[("A", metric), "estimate"])
        eicu_b = float(eicu_metrics.loc[("B", metric), "estimate"])
        if temporal_delta_digits is None:
            temporal_difference = EM_DASH
            eicu_difference = EM_DASH
        else:
            td = temporal_deltas.loc[metric]
            temporal_difference = (
                f"{signed(float(td['estimate']), temporal_delta_digits)} "
                f"({signed(float(td['ci_low']), temporal_delta_digits)} to "
                f"{signed(float(td['ci_high']), temporal_delta_digits)})"
            )
            ed = eicu_deltas.loc[eicu_delta_key]
            eicu_difference = (
                f"{signed(float(ed['estimate']), eicu_delta_digits)} "
                f"({signed(float(ed['ci95_low']), eicu_delta_digits)} to "
                f"{signed(float(ed['ci95_high']), eicu_delta_digits)})"
            )
        rows.append(
            (
                label,
                number(temporal_a, metric_digits),
                number(temporal_b, metric_digits),
                temporal_difference,
                number(eicu_a, metric_digits),
                number(eicu_b, metric_digits),
                eicu_difference,
            )
        )
    return rows


def manuscript_rows(path, table_index):
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
    table = document.findall(".//w:tbl", namespace)[table_index]
    return [["".join(text.text or "" for text in cell.findall(".//w:t", namespace))
             for cell in row.findall("w:tc", namespace)]
            for row in table.findall("w:tr", namespace)]


def crosscheck(expected, actual, label):
    findings = []
    for row in expected:
        if row[0] == "__SECTION__":
            continue
        matches = [candidate for candidate in actual if candidate and candidate[0] == row[0]]
        if len(matches) != 1:
            findings.append([label, row[0], "row", "NOT_FOUND", "", ""])
            continue
        for column, expected_value in enumerate(row[1:], 1):
            actual_value = matches[0][column] if len(matches[0]) > column else ""
            findings.append([label, row[0], column,
                             "PASS" if expected_value == actual_value else "DIFFERENCE",
                             expected_value, actual_value])
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path)
    args = parser.parse_args()
    output = ROOT / "submission_crosscheck"
    output.mkdir(exist_ok=True)
    table1 = table1_rows(load_cohorts())
    table2 = table2_rows()
    for name, headers, rows in [
        ("Table_1", ["Characteristic", "MIMIC development", "MIMIC temporal", "eICU external"], table1),
        ("Table_2", ["Metric", "Temporal A", "Temporal B", "Temporal B-A", "External A", "External B", "External B-A"], table2),
    ]:
        with (output / f"{name}_recomputed.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)
    if args.manuscript:
        checks = crosscheck(table1, manuscript_rows(args.manuscript, 0), "Table 1")
        checks += crosscheck(table2, manuscript_rows(args.manuscript, 1), "Table 2")
        with (output / "manuscript_table_crosscheck.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["table", "row", "column", "status", "recomputed", "manuscript"])
            writer.writerows(checks)
        for name in ("Table 1", "Table 2"):
            selected = [check for check in checks if check[0] == name]
            print(f"{name}: {sum(check[3] == 'PASS' for check in selected)}/{len(selected)} cells match")
    print("Summary-only tables generated; manuscript unchanged.")


if __name__ == "__main__":
    main()
