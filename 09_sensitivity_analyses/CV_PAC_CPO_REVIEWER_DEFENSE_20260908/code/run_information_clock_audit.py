#!/usr/bin/env python3
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
PREV = BASE / "STAGE_MINUS1A_CV_OBSERVABILITY_V1" / "work"
STAGE0 = BASE / "CV_PAC_CPO_STAGE0_MECHANISM_20260908"
STAGE1A = BASE / "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908"
OUT = ROOT / "results"
DEV = "definite_development_to_2019"
TEMP = "definite_temporal_2020_plus"
sys.path.insert(0, str(STAGE0 / "code"))
from run_stage0_mechanism_audit import build_axis_times  # noqa: E402


def raw_store(item: str, stays: set[str]) -> pd.DataFrame:
    d = pd.read_csv(
        PREV / f"raw_{item}.csv.gz",
        usecols=["stay_id", "charttime", "storetime", "valuenum"],
        dtype={"stay_id": str}, parse_dates=["charttime", "storetime"],
    )
    d = d[d.stay_id.isin(stays)].copy()
    d["v"] = pd.to_numeric(d.valuenum, errors="coerce")
    return d


def first_store(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["stay_id", "charttime"], as_index=False).storetime.min()


def add_store_times(cpo: pd.DataFrame, stays: set[str]) -> pd.DataFrame:
    cco = raw_store("224842", stays)
    cco = cco[np.isfinite(cco.v) & cco.v.gt(0) & cco.v.le(30)]
    cpo = cpo.merge(first_store(cco).rename(columns={"storetime": "cco_storetime"}), on=["stay_id", "charttime"], how="left")
    map_parts = []
    for priority, item in enumerate(["220052", "225312", "220181"]):
        x = first_store(raw_store(item, stays))
        x["priority"] = priority
        map_parts.append(x)
    maps = pd.concat(map_parts, ignore_index=True).rename(columns={"storetime": "map_storetime"})
    cpo = cpo.merge(maps, on=["stay_id", "charttime", "priority"], how="left", validate="one_to_one")
    cpo["available_time"] = cpo[["cco_storetime", "map_storetime"]].max(axis=1)
    cpo["available_h"] = (cpo.available_time - cpo.intime).dt.total_seconds() / 3600
    cpo["storage_lag_min"] = (cpo.available_time - cpo.charttime).dt.total_seconds() / 60
    return cpo


def attach_store(axis: pd.DataFrame, item: str, stays: set[str]) -> pd.DataFrame:
    stores = first_store(raw_store(item, stays))
    axis = axis.merge(stores.rename(columns={"storetime": "available_time"}), on=["stay_id", "charttime"], how="left", validate="one_to_one")
    axis["available_h"] = (axis.available_time - axis.intime).dt.total_seconds() / 3600
    axis["storage_lag_min"] = (axis.available_time - axis.charttime).dt.total_seconds() / 60
    return axis


def qdict(x: pd.Series) -> dict:
    x = x.dropna().astype(float)
    q = x.quantile([0, .25, .5, .75, .9, .95, .99, 1])
    return {f"q{int(k*100):02d}": float(v) for k, v in q.items()} | {
        "n": int(len(x)), "fraction_gt_60_min": float((x > 60).mean()), "fraction_negative": float((x < 0).mean())
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    landmarks = pd.read_csv(STAGE0 / "results/LANDMARK_DATASET_V1.csv.gz", dtype={"stay_id": str})
    stays_all = pd.read_csv(PREV / "adult_stays.csv", dtype={"stay_id": str}, parse_dates=["intime", "outtime"])
    wanted = set(landmarks.stay_id)
    stays = stays_all[stays_all.stay_id.isin(wanted)].copy()
    cpo, mpap, svo2 = build_axis_times(stays)
    cpo = add_store_times(cpo, wanted)
    mpap = attach_store(mpap, "220061", wanted)
    svo2 = attach_store(svo2, "223772", wanted)
    groups = {name: {k: g.sort_values("hours") for k, g in frame.groupby("stay_id")} for name, frame in [("cpo", cpo), ("mpap", mpap), ("svo2", svo2)]}

    rows = []
    for r in landmarks[["stay_id", "landmark_h"]].itertuples(index=False):
        out = {"stay_id": r.stay_id, "landmark_h": r.landmark_h}
        for name in ["cpo", "mpap", "svo2"]:
            g = groups[name].get(r.stay_id)
            if g is None:
                out[f"{name}_history_n_original"] = out[f"{name}_history_n_timesafe"] = 0
                out[f"{name}_any_future_stored"] = False
                continue
            hist = g[(g.hours > r.landmark_h - 4) & (g.hours <= r.landmark_h)]
            safe = hist[hist.available_h <= r.landmark_h]
            out[f"{name}_history_n_original"] = int(len(hist))
            out[f"{name}_history_n_timesafe"] = int(len(safe))
            out[f"{name}_any_future_stored"] = bool((hist.available_h > r.landmark_h).any())
            if name == "cpo":
                out["cpo_original_current"] = float(hist.cpo.iloc[-1]) if len(hist) else np.nan
                out["cpo_timesafe_current"] = float(safe.cpo.iloc[-1]) if len(safe) else np.nan
                out["cpo_timesafe_current_recency_h"] = float(r.landmark_h - safe.hours.iloc[-1]) if len(safe) else np.nan
                out["cpo_timesafe_span_h"] = float(safe.hours.iloc[-1] - safe.hours.iloc[0]) if len(safe) else np.nan
        out["timesafe_cpo_eligible"] = (
            out.get("cpo_history_n_timesafe", 0) >= 3
            and out.get("cpo_timesafe_span_h", -np.inf) >= 2
            and out.get("cpo_timesafe_current_recency_h", np.inf) <= 1
            and out.get("cpo_timesafe_current", -np.inf) >= .60
        )
        out["timesafe_common_axes"] = out.get("mpap_history_n_timesafe", 0) >= 2 and out.get("svo2_history_n_timesafe", 0) >= 2
        rows.append(out)
    audit = pd.DataFrame(rows)
    keys = ["stay_id", "landmark_h"]
    z = landmarks.merge(audit, on=keys, how="left", validate="one_to_one")
    original_final = z.primary_eligible_at_landmark.astype(bool) & z.primary_outcome_observed_6h.astype(bool) & z.model_bc_common_risk_set.astype(bool)
    summary = []
    for split in [DEV, TEMP]:
        q = z[original_final & z.interval_conservative_split.eq(split)].copy()
        retained = q.timesafe_cpo_eligible & q.timesafe_common_axes
        changed = (q.cpo_original_current - q.cpo_timesafe_current).abs().gt(1e-12)
        summary.append({
            "split": split, "locked_rows": len(q), "timesafe_rows": int(retained.sum()),
            "timesafe_retention": float(retained.mean()),
            "any_future_stored_cpo_history": int(q.cpo_any_future_stored.sum()),
            "any_future_stored_mpap_history": int(q.mpap_any_future_stored.sum()),
            "any_future_stored_svo2_history": int(q.svo2_any_future_stored.sum()),
            "current_cpo_changed_or_unavailable": int((changed | q.cpo_timesafe_current.isna()).sum()),
        })
    pd.DataFrame(summary).to_csv(OUT / "landmark_timesafe_retention.csv", index=False)
    z.loc[original_final, keys + ["interval_conservative_split", "timesafe_cpo_eligible", "timesafe_common_axes",
                                  "cpo_any_future_stored", "mpap_any_future_stored", "svo2_any_future_stored",
                                  "cpo_original_current", "cpo_timesafe_current"]].to_csv(OUT / "locked_row_information_clock_flags.csv.gz", index=False, compression="gzip")

    final_stays = set(z.loc[original_final, "stay_id"])
    source = {}
    for name, frame in [("derived_cpo", cpo), ("mpap", mpap), ("svo2", svo2)]:
        source[name] = qdict(frame.loc[frame.stay_id.isin(final_stays), "storage_lag_min"])
    (OUT / "source_storage_lag_summary.json").write_text(json.dumps(source, indent=2), encoding="utf-8")

    qc = cpo[cpo.stay_id.isin(final_stays)].sort_values(["stay_id", "hours"]).copy()
    qc["gap_h"] = qc.groupby("stay_id").hours.diff()
    qc["identical_cco"] = qc.cco.eq(qc.groupby("stay_id").cco.shift())
    qc["identical_cpo"] = qc.cpo.eq(qc.groupby("stay_id").cpo.shift())
    device = {
        "cpo_records": int(len(qc)), "stays": int(qc.stay_id.nunique()),
        "chart_interval_h_quantiles": {str(k): float(v) for k, v in qc.gap_h.dropna().quantile([.01,.25,.5,.75,.9,.95,.99]).items()},
        "consecutive_identical_cco_fraction": float(qc.identical_cco.mean()),
        "consecutive_identical_cpo_fraction": float(qc.identical_cpo.mean()),
    }
    native = pd.read_csv(PREV / "raw_229896.csv.gz", dtype={"stay_id": str})
    device["native_cpo_total_stays"] = int(native.stay_id.nunique())
    device["native_cpo_final_analysis_overlap_stays"] = int(native.loc[native.stay_id.isin(final_stays), "stay_id"].nunique())
    (OUT / "device_and_native_cpo_audit.json").write_text(json.dumps(device, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

