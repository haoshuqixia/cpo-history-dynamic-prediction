#!/usr/bin/env python3
from __future__ import annotations
import os

from collections import defaultdict
from pathlib import Path
import csv
import gzip
import json
import math
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
PREV = BASE / "STAGE_MINUS1A_CV_OBSERVABILITY_V1" / "work"
LANDMARK_ROOT = BASE / "CV_PAC_CPO_LANDMARK_AUDIT_20260908"
MIMIC = Path(os.environ["CPO_MIMIC_DIR"])
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"

LOW_CPO = 0.60

DRUGS = {
    "221906": "norepinephrine",
    "221289": "epinephrine",
    "229617": "epinephrine",
    "222315": "vasopressin",
    "221749": "phenylephrine",
    "229630": "phenylephrine",
    "229631": "phenylephrine",
    "229632": "phenylephrine",
    "221662": "dopamine",
    "221653": "dobutamine",
    "221986": "milrinone",
}

MCS_PROCEDURES = {
    "224272": "iabp",
    "228169": "impella",
    "228201": "vad_tandemheart",
    "228202": "vad_tandemheart",
    "229529": "ecmo",
    "229530": "ecmo",
}

MCS_CHART = {
    "220120": "iabp", "224309": "iabp", "224310": "iabp",
    "224311": "iabp", "224322": "iabp", "224652": "iabp",
    "227980": "iabp",
    "228154": "impella", "228156": "impella", "229671": "impella",
    "229675": "impella",
    "224660": "ecmo", "229270": "ecmo", "229842": "ecmo",
    "220128": "vad", "223775": "vad", "224363": "vad",
    "228189": "vad_tandemheart", "228195": "vad_tandemheart",
    "228198": "vad_tandemheart", "229254": "vad", "229255": "vad",
    "229262": "vad", "229263": "vad", "229823": "vad", "229829": "vad",
}

MAJOR_MCS = {"ecmo", "impella", "vad", "vad_tandemheart"}


def finite_number(value: object) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def raw(itemid: str, low: float, high: float, low_open: bool = False) -> pd.DataFrame:
    x = pd.read_csv(
        PREV / f"raw_{itemid}.csv.gz",
        usecols=["stay_id", "charttime", "valuenum"],
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


def map_values() -> pd.DataFrame:
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


def build_axis_times(stays: pd.DataFrame):
    keep = stays[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]].copy()
    keep["stay_id"] = keep["stay_id"].astype(str)
    cco = raw("224842", 0, 30, low_open=True).rename(columns={"v": "cco"})
    cpo = cco.merge(map_values(), on=["stay_id", "charttime"], how="inner")
    cpo = cpo.merge(keep, on="stay_id", how="inner", validate="many_to_one")
    cpo = cpo.loc[cpo["charttime"].between(cpo["intime"], cpo["outtime"])].copy()
    cpo["hours"] = (cpo["charttime"] - cpo["intime"]).dt.total_seconds() / 3600
    cpo = cpo.loc[cpo["hours"].between(0, 72)]
    cpo["cpo"] = cpo["cco"] * cpo["map"] / 451.0

    axes = {}
    for name, itemid, lo, hi in [("mpap", "220061", 0, 150), ("svo2", "223772", 0, 100)]:
        z = raw(itemid, lo, hi).merge(keep[["stay_id", "intime", "outtime"]], on="stay_id", how="inner")
        z = z.loc[z["charttime"].between(z["intime"], z["outtime"])].copy()
        z["hours"] = (z["charttime"] - z["intime"]).dt.total_seconds() / 3600
        axes[name] = z.loc[z["hours"].between(0, 72)]
    return cpo, axes["mpap"], axes["svo2"]


def filtered_rows(relative: str, ids: set[str], usecols: list[str], chunksize: int = 250_000):
    path = MIMIC / relative
    with gzip.open(path, "rt", newline="") as f:
        header = next(csv.reader(f))
    idx = header.index("itemid")
    pattern = r"^(?:[^,]*,){" + str(idx) + r"}(?:" + "|".join(sorted(ids)) + r"),"
    dec = subprocess.Popen(["gzip", "-cd", str(path)], stdout=subprocess.PIPE)
    sel = subprocess.Popen(["rg", "-a", pattern], stdin=dec.stdout, stdout=subprocess.PIPE)
    assert dec.stdout is not None and sel.stdout is not None
    dec.stdout.close()
    try:
        yield from pd.read_csv(
            sel.stdout, names=header, header=None, usecols=usecols, dtype=str,
            keep_default_na=False, chunksize=chunksize,
        )
    except pd.errors.EmptyDataError:
        pass
    finally:
        sel.stdout.close()
        rc = sel.wait()
        dc = dec.wait()
        if rc not in (0, 1) or dc != 0:
            raise RuntimeError((relative, rc, dc))


def item_contract() -> pd.DataFrame:
    ids = set(DRUGS) | set(MCS_PROCEDURES) | set(MCS_CHART) | {"224654"}
    d = pd.read_csv(MIMIC / "icu/d_items.csv.gz", dtype=str, keep_default_na=False)
    d = d.loc[d["itemid"].isin(ids), ["itemid", "label", "category", "unitname", "param_type"]].copy()
    d["role"] = d["itemid"].map(DRUGS).fillna(d["itemid"].map(MCS_PROCEDURES)).fillna(d["itemid"].map(MCS_CHART))
    d.loc[d["itemid"] == "224654", "role"] = "excluded_from_mcs: PAEDP is not device evidence"
    return d.sort_values("itemid")


def load_mcs(stay_ids: set[str]):
    proc = defaultdict(list)
    proc_audit = defaultdict(int)
    for chunk in filtered_rows(
        "icu/procedureevents.csv.gz", set(MCS_PROCEDURES),
        ["stay_id", "starttime", "endtime", "itemid", "statusdescription"],
    ):
        for r in chunk.itertuples(index=False):
            proc_audit["mapped_rows"] += 1
            if r.stay_id not in stay_ids:
                continue
            proc_audit["cohort_rows"] += 1
            if str(r.statusdescription).strip().lower() == "rewritten":
                proc_audit["rewritten_excluded"] += 1
                continue
            start, end = pd.to_datetime(r.starttime, errors="coerce"), pd.to_datetime(r.endtime, errors="coerce")
            if pd.isna(start):
                proc_audit["invalid_start_excluded"] += 1
                continue
            if pd.isna(end) or end < start:
                end = start
            proc[r.stay_id].append((start, end, MCS_PROCEDURES[r.itemid], r.itemid))

    chart = defaultdict(list)
    chart_audit = defaultdict(int)
    numeric_items = {
        "224309", "224310", "224311", "224322", "224652", "228154", "229671",
        "224660", "229270", "229842", "220128", "223775", "224363", "228189",
        "228195", "228198", "229254", "229255", "229262", "229263", "229823", "229829",
    }
    for chunk in filtered_rows(
        "icu/chartevents.csv.gz", set(MCS_CHART),
        ["stay_id", "charttime", "itemid", "value", "valuenum"],
    ):
        for r in chunk.itertuples(index=False):
            chart_audit["mapped_rows"] += 1
            if r.stay_id not in stay_ids:
                continue
            chart_audit["cohort_rows"] += 1
            t = pd.to_datetime(r.charttime, errors="coerce")
            if pd.isna(t):
                chart_audit["invalid_time_excluded"] += 1
                continue
            if r.itemid in numeric_items:
                val = finite_number(r.valuenum)
                if val is None or val <= 0:
                    chart_audit["nonpositive_numeric_excluded"] += 1
                    continue
            elif not str(r.value).strip():
                chart_audit["empty_text_excluded"] += 1
                continue
            chart[r.stay_id].append((t, MCS_CHART[r.itemid], r.itemid))
    return proc, chart, {"procedure": dict(proc_audit), "chart": dict(chart_audit)}


def load_drugs(stay_ids: set[str]):
    intervals = defaultdict(list)
    audit = defaultdict(int)
    cols = ["stay_id", "starttime", "endtime", "itemid", "rate", "rateuom", "statusdescription"]
    for chunk in filtered_rows("icu/inputevents.csv.gz", set(DRUGS), cols):
        for r in chunk.itertuples(index=False):
            audit["mapped_rows"] += 1
            if r.stay_id not in stay_ids:
                continue
            audit["cohort_rows"] += 1
            if str(r.statusdescription).strip().lower() == "rewritten":
                audit["rewritten_excluded"] += 1
                continue
            rate = finite_number(r.rate)
            if rate is None or rate <= 0:
                audit["nonpositive_rate_excluded"] += 1
                continue
            start, end = pd.to_datetime(r.starttime, errors="coerce"), pd.to_datetime(r.endtime, errors="coerce")
            if pd.isna(start) or pd.isna(end) or end <= start:
                audit["invalid_interval_excluded"] += 1
                continue
            intervals[r.stay_id].append((start, end, DRUGS[r.itemid], rate, str(r.rateuom).strip(), r.itemid))
    return intervals, dict(audit)


def count_recency(times: np.ndarray, landmark: float) -> tuple[int, float]:
    hist = times[(times > landmark - 4) & (times <= landmark)]
    return len(hist), float(landmark - hist[-1]) if len(hist) else math.nan


def first_pair_qa(times: np.ndarray, values: np.ndarray, left: float, right: float):
    in_window = (times > left) & (times <= right)
    wt, wv = times[in_window], values[in_window]
    low_idx = np.flatnonzero(wv < LOW_CPO)
    if len(low_idx) < 2:
        return None
    low_times = wt[low_idx]
    gaps = np.diff(low_times)
    hits = np.flatnonzero((gaps >= 0.5) & (gaps <= 3.0))
    if not len(hits):
        return None
    j = int(hits[0])
    i1, i2 = int(low_idx[j]), int(low_idx[j + 1])
    between = wv[i1 + 1:i2]
    return low_times[j], low_times[j + 1], bool(np.any(between >= LOW_CPO)), int(np.sum(between >= LOW_CPO))


def qsummary(x: pd.Series) -> dict:
    z = pd.to_numeric(x, errors="coerce").dropna()
    if not len(z):
        return {"n": 0, "median": math.nan, "q1": math.nan, "q3": math.nan}
    return {
        "n": int(len(z)), "median": float(z.median()),
        "q1": float(z.quantile(0.25)), "q3": float(z.quantile(0.75)),
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    landmarks = pd.read_csv(
        LANDMARK_ROOT / "results/rolling_landmarks.csv.gz",
        dtype={"stay_id": str},
    )
    stays = pd.read_csv(PREV / "adult_stays.csv", dtype={"stay_id": str}, parse_dates=["intime", "outtime"])
    stays = stays.loc[stays["stay_id"].isin(set(landmarks["stay_id"]))].copy()
    stay_lookup = stays.set_index("stay_id")
    cpo, mpap, svo2 = build_axis_times(stays)
    cpo_groups = {k: (g["hours"].to_numpy(float), g["cpo"].to_numpy(float)) for k, g in cpo.sort_values("hours").groupby("stay_id")}
    mpap_groups = {k: np.sort(g["hours"].to_numpy(float)) for k, g in mpap.groupby("stay_id")}
    svo2_groups = {k: np.sort(g["hours"].to_numpy(float)) for k, g in svo2.groupby("stay_id")}

    contract = item_contract()
    contract.to_csv(AUDIT / "item_contract.csv", index=False)
    stay_ids = set(landmarks["stay_id"])
    mcs_proc, mcs_chart, mcs_audit = load_mcs(stay_ids)
    drugs, drug_audit = load_drugs(stay_ids)

    rows = []
    pair_rows = []
    for r in landmarks.itertuples(index=False):
        stay_id = r.stay_id
        L = float(r.landmark_h)
        st = stay_lookup.loc[stay_id]
        lt = st.intime + pd.Timedelta(hours=L)
        horizon = lt + pd.Timedelta(hours=6)
        ct, cv = cpo_groups[stay_id]
        mt = mpap_groups.get(stay_id, np.array([], dtype=float))
        svt = svo2_groups.get(stay_id, np.array([], dtype=float))

        n_cpo, rec_cpo = count_recency(ct, L)
        n_mpap, rec_mpap = count_recency(mt, L)
        n_svo2, rec_svo2 = count_recency(svt, L)
        cpo_late = bool(np.any((ct > L + 4) & (ct <= L + 6)))
        mpap_late = bool(np.any((mt > L + 4) & (mt <= L + 6)))
        svo2_late = bool(np.any((svt > L + 4) & (svt <= L + 6)))
        future_cpo = ct[(ct > L) & (ct <= L + 6)]
        last_cpo = float(future_cpo[-1]) if len(future_cpo) else L
        future_mpap = mt[(mt > L) & (mt <= L + 6)]
        future_svo2 = svt[(svt > L) & (svt <= L + 6)]
        other_axis_after_last_cpo = bool(np.any((mt > last_cpo) & (mt <= L + 6)) or np.any((svt > last_cpo) & (svt <= L + 6)))
        icu_exit = bool(st.outtime <= horizon)

        if bool(r.event_6h):
            state = "confirmed_low_cpo_event"
        elif bool(r.dense_observed_negative_6h):
            state = "dense_observed_event_free"
        elif icu_exit:
            state = "icu_exit_before_horizon"
        elif (not cpo_late) and other_axis_after_last_cpo:
            state = "isolated_cpo_recording_cessation"
        elif (not cpo_late) and (not mpap_late) and (not svo2_late):
            state = "joint_pac_axis_cessation"
        else:
            state = "other_unresolved_observation_loss"

        proc = mcs_proc.get(stay_id, [])
        chart = mcs_chart.get(stay_id, [])
        active_proc_types = {typ for a, b, typ, _ in proc if a <= lt <= b}
        active_chart_types = {typ for t, typ, _ in chart if lt - pd.Timedelta(hours=2) < t <= lt}
        active_types = active_proc_types | active_chart_types
        future_proc_types = {typ for a, _, typ, _ in proc if lt < a <= horizon}
        prior_chart_types = {typ for t, typ, _ in chart if t <= lt}
        first_chart_types = {typ for t, typ, _ in chart if lt < t <= horizon and typ not in prior_chart_types}
        major_proc_future_times = [a for a, _, typ, _ in proc if typ in MAJOR_MCS and lt < a <= horizon]
        major_chart_future_times = [t for t, typ, _ in chart if typ in MAJOR_MCS and lt < t <= horizon and typ not in prior_chart_types]
        major_active = bool(active_types & MAJOR_MCS)
        iabp_active = "iabp" in active_types
        major_new_proc = bool(future_proc_types & MAJOR_MCS)
        iabp_new_proc = "iabp" in future_proc_types

        ivals = drugs.get(stay_id, [])
        hist_agents = {d for a, b, d, _, _, _ in ivals if a <= lt and b > lt - pd.Timedelta(hours=4)}
        active = [(d, rate, uom) for a, b, d, rate, uom, _ in ivals if a <= lt < b]
        active_agents = {d for d, _, _ in active}
        new_agents = {d for a, _, d, _, _, _ in ivals if lt < a <= horizon and d not in active_agents}
        dose_increase_agents = set()
        for d, rate, uom in active:
            if any(d2 == d and u2 == uom and lt < a <= horizon and rate2 > rate
                   for a, _, d2, rate2, u2, _ in ivals):
                dose_increase_agents.add(d)

        row = r._asdict()
        row.update({
            "observation_state_6h": state,
            "icu_exit_before_horizon": icu_exit,
            "cpo_recorded_late_horizon": cpo_late,
            "mpap_recorded_late_horizon": mpap_late,
            "svo2_recorded_late_horizon": svo2_late,
            "other_axis_after_last_future_cpo": other_axis_after_last_cpo,
            "icu_exit_offset_h": float((st.outtime - lt).total_seconds() / 3600),
            "last_future_mpap_offset_h": float(future_mpap[-1] - L) if len(future_mpap) else math.nan,
            "last_future_svo2_offset_h": float(future_svo2[-1] - L) if len(future_svo2) else math.nan,
            "n_mpap_history_4h": n_mpap,
            "n_svo2_history_4h": n_svo2,
            "cpo_recency_h": rec_cpo,
            "mpap_recency_h": rec_mpap,
            "svo2_recency_h": rec_svo2,
            "major_mcs_active_at_landmark": major_active,
            "iabp_active_at_landmark": iabp_active,
            "major_mcs_procedure_start_future_6h": major_new_proc,
            "iabp_procedure_start_future_6h": iabp_new_proc,
            "major_mcs_first_charted_future_6h": bool(first_chart_types & MAJOR_MCS),
            "iabp_first_charted_future_6h": "iabp" in first_chart_types,
            "mcs_active_types": ";".join(sorted(active_types)),
            "mcs_active_procedure_types": ";".join(sorted(active_proc_types)),
            "mcs_active_chart_types": ";".join(sorted(active_chart_types)),
            "mcs_future_procedure_types": ";".join(sorted(future_proc_types)),
            "major_mcs_first_procedure_offset_h": float((min(major_proc_future_times) - lt).total_seconds() / 3600) if major_proc_future_times else math.nan,
            "major_mcs_first_chart_offset_h": float((min(major_chart_future_times) - lt).total_seconds() / 3600) if major_chart_future_times else math.nan,
            "drug_history_4h_any": bool(hist_agents),
            "drug_active_at_landmark_any": bool(active_agents),
            "drug_new_agent_future_6h_any": bool(new_agents),
            "drug_dose_increase_future_6h_any": bool(dose_increase_agents),
            "drug_history_4h_agents": ";".join(sorted(hist_agents)),
            "drug_active_agents": ";".join(sorted(active_agents)),
            "drug_new_future_agents": ";".join(sorted(new_agents)),
            "drug_dose_increase_future_agents": ";".join(sorted(dose_increase_agents)),
        })
        for d in sorted(set(DRUGS.values())):
            row[f"drug_{d}_active_at_landmark"] = d in active_agents
            row[f"drug_{d}_new_future_6h"] = d in new_agents
        rows.append(row)

        if bool(r.event_6h):
            pair = first_pair_qa(ct, cv, L, L + 6)
            if pair is None:
                raise AssertionError(f"positive landmark without pair: {stay_id} {L}")
            pair_rows.append({
                "subject_id": r.subject_id, "stay_id": stay_id, "landmark_h": L,
                "first_low_h": pair[0], "confirming_low_h": pair[1],
                "intervening_normal": pair[2], "n_intervening_normal": pair[3],
            })

    out = pd.DataFrame(rows)
    pairs = pd.DataFrame(pair_rows)
    out.to_csv(RESULTS / "landmark_mechanism_audit.csv.gz", index=False, compression="gzip")
    pairs.to_csv(RESULTS / "endpoint_pair_qa_landmarks.csv", index=False)
    unique_pairs = pairs.drop_duplicates(["stay_id", "first_low_h", "confirming_low_h"])
    unique_pairs.to_csv(RESULTS / "endpoint_pair_qa_unique_pairs.csv", index=False)

    state_counts = out.groupby("observation_state_6h").agg(
        landmarks=("stay_id", "size"), patients=("subject_id", "nunique")
    ).reset_index()
    state_counts["fraction"] = state_counts["landmarks"] / len(out)
    state_counts.to_csv(RESULTS / "observation_state_counts.csv", index=False)

    measurement_rows = []
    for state, g in out.groupby("observation_state_6h"):
        for variable in ["n_cpo_history_4h", "n_mpap_history_4h", "n_svo2_history_4h", "cpo_recency_h", "mpap_recency_h", "svo2_recency_h"]:
            z = qsummary(g[variable])
            measurement_rows.append({"observation_state_6h": state, "variable": variable, **z})
    pd.DataFrame(measurement_rows).to_csv(RESULTS / "measurement_process_summary.csv", index=False)

    mcs_cols = [
        "major_mcs_active_at_landmark", "iabp_active_at_landmark",
        "major_mcs_procedure_start_future_6h", "iabp_procedure_start_future_6h",
        "major_mcs_first_charted_future_6h", "iabp_first_charted_future_6h",
    ]
    mcs_rows = []
    for name in mcs_cols:
        q = out[name].astype(bool)
        mcs_rows.append({
            "indicator": name, "landmarks": int(q.sum()),
            "patients": int(out.loc[q, "subject_id"].nunique()),
            "event_landmarks": int(out.loc[q, "event_6h"].sum()),
            "unknown_landmarks": int((q & ~out["outcome_observed_6h"].astype(bool)).sum()),
        })
    pd.DataFrame(mcs_rows).to_csv(RESULTS / "mcs_summary.csv", index=False)

    treatment_rows = []
    for indicator in ["drug_history_4h_any", "drug_active_at_landmark_any", "drug_new_agent_future_6h_any", "drug_dose_increase_future_6h_any"]:
        for state, g in out.groupby("observation_state_6h"):
            treatment_rows.append({
                "indicator": indicator, "observation_state_6h": state,
                "landmarks": len(g), "positive": int(g[indicator].astype(bool).sum()),
                "fraction": float(g[indicator].astype(bool).mean()),
            })
    for d in sorted(set(DRUGS.values())):
        for suffix in ["active_at_landmark", "new_future_6h"]:
            col = f"drug_{d}_{suffix}"
            q = out[col].astype(bool)
            treatment_rows.append({
                "indicator": col, "observation_state_6h": "ALL",
                "landmarks": len(out), "positive": int(q.sum()), "fraction": float(q.mean()),
            })
    pd.DataFrame(treatment_rows).to_csv(RESULTS / "treatment_summary.csv", index=False)

    patients = pd.read_csv(LANDMARK_ROOT / "results/patient_summary.csv")[["subject_id", "interval_conservative_split"]]
    den = out.merge(patients, on="subject_id", how="left", validate="many_to_one")
    den_rows = []
    for split, g in den.groupby("interval_conservative_split"):
        eligible = g["full_mpap_svo2_history"].astype(bool)
        observed = g["outcome_observed_6h"].astype(bool)
        e = g.loc[eligible]
        den_rows.append({
            "interval_conservative_split": split,
            "all_patients": int(g["subject_id"].nunique()), "all_landmarks": len(g),
            "full_axes_patients": int(e["subject_id"].nunique()), "full_axes_landmarks": len(e),
            "full_axes_fraction": float(eligible.mean()),
            "full_axes_observed_landmarks": int((eligible & observed).sum()),
            "full_axes_event_landmarks": int(e["event_6h"].sum()),
            "full_axes_event_patients": int(e.loc[e["event_6h"].astype(bool), "subject_id"].nunique()),
        })
    pd.DataFrame(den_rows).to_csv(RESULTS / "full_axes_common_denominator.csv", index=False)

    summary = {
        "landmarks": int(len(out)),
        "patients": int(out["subject_id"].nunique()),
        "observation_states": {r.observation_state_6h: int(r.landmarks) for r in state_counts.itertuples()},
        "unknown_landmarks": int((~out["outcome_observed_6h"].astype(bool)).sum()),
        "unknown_fraction": float((~out["outcome_observed_6h"].astype(bool)).mean()),
        "unique_endpoint_pairs": int(len(unique_pairs)),
        "unique_endpoint_pairs_with_intervening_normal": int(unique_pairs["intervening_normal"].sum()),
        "endpoint_wording": "confirmed/repeated low CPO episode" if bool(unique_pairs["intervening_normal"].any()) else "persistent low cardiac power state is consistent with pair QA",
        "major_mcs_active_landmarks": int(out["major_mcs_active_at_landmark"].sum()),
        "major_mcs_active_patients": int(out.loc[out["major_mcs_active_at_landmark"], "subject_id"].nunique()),
        "major_mcs_future_procedure_landmarks": int(out["major_mcs_procedure_start_future_6h"].sum()),
        "major_mcs_future_procedure_patients": int(out.loc[out["major_mcs_procedure_start_future_6h"], "subject_id"].nunique()),
        "mcs_exclusion_note": "itemid 224654 excluded because label is PAEDP, not device evidence",
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (AUDIT / "extraction_audit.json").write_text(json.dumps({"mcs": mcs_audit, "drugs": drug_audit}, indent=2, ensure_ascii=False))
    sources = []
    for p in [
        LANDMARK_ROOT / "results/rolling_landmarks.csv.gz", PREV / "adult_stays.csv",
        PREV / "raw_224842.csv.gz", PREV / "raw_220061.csv.gz", PREV / "raw_223772.csv.gz",
        MIMIC / "icu/d_items.csv.gz", MIMIC / "icu/procedureevents.csv.gz",
        MIMIC / "icu/chartevents.csv.gz", MIMIC / "icu/inputevents.csv.gz",
    ]:
        sources.append({"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns})
    (AUDIT / "source_files.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
