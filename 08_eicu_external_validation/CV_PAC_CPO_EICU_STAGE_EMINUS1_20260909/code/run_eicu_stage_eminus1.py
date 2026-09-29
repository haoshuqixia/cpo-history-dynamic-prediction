#!/usr/bin/env python3
from __future__ import annotations
import os

from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import gzip
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EICU_ZIP = Path(os.environ["CPO_EICU_ZIP"])
PREFIX = "eicu-collaborative-research-database-2.0"
WORK = ROOT / "work"
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"

NURSE_NAMES = [
    "nursingchartid", "patientunitstayid", "nursingchartoffset", "nursingchartentryoffset",
    "nursingchartcelltypecat", "nursingchartcelltypevallabel", "nursingchartcelltypevalname", "nursingchartvalue",
]
PAC_RE = re.compile(r"pulmonary artery catheter|pulmonary arterial catheter|swan[- ]?ganz|right heart catheter", re.I)
IABP_RE = re.compile(r"\biabp\b|intra.?aortic.*balloon", re.I)


@contextmanager
def open_inner(table: str, binary: bool = False):
    with zipfile.ZipFile(EICU_ZIP) as archive:
        with archive.open(f"{PREFIX}/{table}.csv.gz") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
                if binary:
                    yield gz
                else:
                    with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                        yield text


def number(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def ensure_nurse_targets() -> Path:
    out = WORK / "nurse_cardiac_targets.csv.gz"
    if out.exists() and out.stat().st_size > 100:
        return out
    rg = shutil.which("rg")
    unzip = shutil.which("unzip")
    gzip_bin = shutil.which("gzip")
    if not all([rg, unzip, gzip_bin]):
        raise RuntimeError("unzip, gzip, and rg are required")
    member = f"{PREFIX}/nurseCharting.csv.gz"
    patterns = [
        ",CO,CO,", ",PA,PA Mean,", ",SVO2,SVO2,", ",Invasive BP,Invasive BP Mean,",
        ",MAP (mmHg),Value,", ",Arterial Line MAP (mmHg),Value,",
    ]
    p1 = subprocess.Popen([unzip, "-p", str(EICU_ZIP), member], stdout=subprocess.PIPE)
    p2 = subprocess.Popen([gzip_bin, "-dc"], stdin=p1.stdout, stdout=subprocess.PIPE)
    assert p1.stdout is not None
    p1.stdout.close()
    cmd = [rg, "-a", "-F"]
    for pattern in patterns:
        cmd += ["-e", pattern]
    p3 = subprocess.Popen(cmd, stdin=p2.stdout, stdout=subprocess.PIPE)
    assert p2.stdout is not None
    p2.stdout.close()
    with gzip.open(out, "wb", compresslevel=3) as handle:
        assert p3.stdout is not None
        shutil.copyfileobj(p3.stdout, handle, length=1024 * 1024)
    codes = [p3.wait(), p2.wait(), p1.wait()]
    if codes[0] not in (0, 1) or any(c != 0 for c in codes[1:]):
        raise RuntimeError(f"nurse target extraction failed: {codes}")
    return out


def read_nurse_targets(path: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    rows = defaultdict(list)
    raw_labels = Counter()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for values in csv.reader(handle):
            if len(values) != 8:
                continue
            row = dict(zip(NURSE_NAMES, values))
            sid = number(row["patientunitstayid"]); offset = number(row["nursingchartoffset"])
            entry = number(row["nursingchartentryoffset"]); value = number(row["nursingchartvalue"])
            if None in (sid, offset, entry, value):
                continue
            label = row["nursingchartcelltypevallabel"]
            name = row["nursingchartcelltypevalname"]
            raw_labels[(label, name)] += 1
            key = None
            if label == "CO" and name == "CO": key = "co"
            elif label == "PA" and name == "PA Mean": key = "pamean"
            elif label == "SVO2" and name == "SVO2": key = "svo2"
            elif (label, name) in {
                ("Invasive BP", "Invasive BP Mean"), ("MAP (mmHg)", "Value"),
                ("Arterial Line MAP (mmHg)", "Value"),
            }: key = "map"
            if key:
                rows[key].append((int(sid), int(offset), int(entry), float(value), f"{label}|{name}"))
    frames = {key: pd.DataFrame(value, columns=["sid", "offset", "entry", "value", "raw_source"]) for key, value in rows.items()}
    for key in ["co", "pamean", "svo2", "map"]:
        frames.setdefault(key, pd.DataFrame(columns=["sid", "offset", "entry", "value", "raw_source"]))
    return frames, {"raw_label_name_counts": {f"{k[0]}|{k[1]}": v for k, v in raw_labels.items()}}


def extract_aperiodic_co() -> Path:
    out = WORK / "vitalaperiodic_co.csv.gz"
    if out.exists() and out.stat().st_size > 100:
        return out
    with open_inner("vitalAperiodic", binary=True) as src, gzip.open(out, "wt", newline="", compresslevel=3) as dst:
        writer = csv.writer(dst); writer.writerow(["sid", "offset", "cardiacoutput"])
        next(src)
        scanned = 0
        for line in src:
            scanned += 1
            parts = line.rstrip(b"\r\n").split(b",")
            if len(parts) >= 8 and parts[7]:
                writer.writerow([parts[1].decode(), parts[2].decode(), parts[7].decode()])
            if scanned % 10_000_000 == 0:
                print(f"vitalAperiodic CO scan {scanned:,}", file=sys.stderr, flush=True)
    return out


def scan_candidate_vitals(candidate_sids: set[int]) -> tuple[Path, Path, dict]:
    aper_out = WORK / "vitalaperiodic_candidate.csv.gz"
    periodic_out = WORK / "vitalperiodic_candidate.csv.gz"
    counts = {}
    if not aper_out.exists() or aper_out.stat().st_size <= 100:
        with open_inner("vitalAperiodic", binary=True) as src, gzip.open(aper_out, "wt", newline="", compresslevel=3) as dst:
            writer = csv.writer(dst); writer.writerow(["sid", "offset", "noninvasivemean", "cardiacoutput"])
            next(src); scanned = kept = 0
            for line in src:
                scanned += 1
                parts = line.rstrip(b"\r\n").split(b",")
                if len(parts) < 8:
                    continue
                try: sid = int(parts[1])
                except ValueError: continue
                if sid in candidate_sids and (parts[5] or parts[7]):
                    writer.writerow([sid, parts[2].decode(), parts[5].decode(), parts[7].decode()]); kept += 1
            counts["vitalAperiodic_rows_scanned"] = scanned; counts["vitalAperiodic_candidate_rows"] = kept
    if not periodic_out.exists() or periodic_out.stat().st_size <= 100:
        with open_inner("vitalPeriodic", binary=True) as src, gzip.open(periodic_out, "wt", newline="", compresslevel=3) as dst:
            writer = csv.writer(dst); writer.writerow(["sid", "offset", "systemicmean", "pamean"])
            next(src); scanned = kept = 0
            for line in src:
                scanned += 1
                parts = line.rstrip(b"\r\n").split(b",")
                if len(parts) < 14:
                    continue
                try: sid = int(parts[1])
                except ValueError: continue
                if sid in candidate_sids and (parts[10] or parts[13]):
                    writer.writerow([sid, parts[2].decode(), parts[10].decode(), parts[13].decode()]); kept += 1
                if scanned % 25_000_000 == 0:
                    print(f"vitalPeriodic candidate scan {scanned:,}", file=sys.stderr, flush=True)
            counts["vitalPeriodic_rows_scanned"] = scanned; counts["vitalPeriodic_candidate_rows"] = kept
    return aper_out, periodic_out, counts


def resolve(frame: pd.DataFrame, low: float, high: float, entry: bool) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        cols = ["sid", "offset", "value"] + (["entry"] if entry else [])
        return pd.DataFrame(columns=cols), 0
    frame = frame.copy()
    for col in ["sid", "offset", "value"] + (["entry"] if entry else []):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[np.isfinite(frame.value) & frame.value.between(low, high, inclusive="both")].copy()
    conflict = frame.groupby(["sid", "offset"]).value.nunique().gt(1)
    conflict_keys = set(conflict[conflict].index)
    if conflict_keys:
        idx = pd.MultiIndex.from_frame(frame[["sid", "offset"]])
        frame = frame[~idx.isin(conflict_keys)].copy()
    agg = {"value": "first"}
    if entry: agg["entry"] = "min"
    frame = frame.sort_values(["sid", "offset"] + (["entry"] if entry else [])).groupby(["sid", "offset"], as_index=False).agg(agg)
    frame[["sid", "offset"]] = frame[["sid", "offset"]].astype(int)
    return frame, len(conflict_keys)


def load_patients(candidate_sids: set[int]) -> pd.DataFrame:
    rows = []
    with open_inner("patient") as handle:
        for row in csv.DictReader(handle):
            sid = int(row["patientunitstayid"])
            if sid not in candidate_sids:
                continue
            age = 90 if row["age"].strip() == "> 89" else number(row["age"])
            los = number(row["unitdischargeoffset"])
            rows.append({
                "sid": sid, "patienthealthsystemstayid": int(row["patienthealthsystemstayid"]),
                "uniquepid": row["uniquepid"], "hospitalid": int(row["hospitalid"]),
                "unittype": row["unittype"], "gender": row["gender"], "age": age, "los_min": los,
            })
    return pd.DataFrame(rows)


def normalize_drug(name: str) -> str | None:
    text = name.lower()
    if "norepinephrine" in text or "levophed" in text: return "norepinephrine"
    if re.search(r"(?:^|[^a-z])epinephrine(?:[^a-z]|$)", text): return "epinephrine"
    if "vasopressin" in text: return "vasopressin"
    if "phenylephrine" in text or "neosynephrine" in text or "neo-synephrine" in text: return "phenylephrine"
    if "dopamine" in text: return "dopamine"
    if "dobutamine" in text or "dobutrex" in text: return "dobutamine"
    if "milrinone" in text: return "milrinone"
    return None


def scan_support_and_pac(candidate_sids: set[int]) -> tuple[dict, set[int], set[int]]:
    drug_stays = defaultdict(set); drug_rows = Counter(); raw_drugs = Counter()
    with open_inner("infusionDrug") as handle:
        for row in csv.DictReader(handle):
            try: sid = int(row["patientunitstayid"]); offset = int(float(row["infusionoffset"]))
            except (ValueError, TypeError): continue
            if sid not in candidate_sids or not (0 <= offset <= 4320): continue
            drug = normalize_drug(row["drugname"])
            if drug:
                drug_rows[drug] += 1; drug_stays[drug].add(sid); raw_drugs[row["drugname"]] += 1
    explicit_pac = set(); iabp = set(); matched_terms = Counter()
    with open_inner("treatment") as handle:
        for row in csv.DictReader(handle):
            try: sid = int(row["patientunitstayid"])
            except ValueError: continue
            if sid not in candidate_sids: continue
            text = row["treatmentstring"]
            if PAC_RE.search(text): explicit_pac.add(sid); matched_terms[f"treatment PAC|{text}"] += 1
            if IABP_RE.search(text): iabp.add(sid); matched_terms[f"treatment IABP|{text}"] += 1
    with open_inner("note") as handle:
        for row in csv.DictReader(handle):
            try: sid = int(row["patientunitstayid"])
            except ValueError: continue
            if sid not in candidate_sids: continue
            text = " ".join([row["notetype"], row["notepath"], row["notevalue"], row["notetext"]])
            if PAC_RE.search(text): explicit_pac.add(sid); matched_terms["note PAC match"] += 1
            if IABP_RE.search(text): iabp.add(sid); matched_terms["note IABP match"] += 1
    audit = {
        "mapped_drug_rows_0_72h": dict(drug_rows),
        "mapped_drug_stays_0_72h": {k: len(v) for k, v in drug_stays.items()},
        "distinct_mapped_raw_drug_names": len(raw_drugs),
        "explicit_pac_stays": len(explicit_pac), "iabp_evidence_stays": len(iabp),
        "matched_term_counts": dict(matched_terms),
    }
    return audit, explicit_pac, iabp


def pair_cpo(co: pd.DataFrame, nurse_map: pd.DataFrame, periodic: pd.DataFrame, aper: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict]:
    nurse_exact = {(int(r.sid), int(r.offset)): (float(r.value), int(r.entry)) for r in nurse_map.itertuples(index=False)}
    aper_mean = {}
    for r in aper.itertuples(index=False):
        val = number(r.noninvasivemean)
        if val is not None and 20 <= val <= 250:
            aper_mean[(int(r.sid), int(r.offset))] = val
    periodic_by = {}
    for sid, g in periodic.groupby("sid"):
        z = g.copy(); z["value"] = pd.to_numeric(z.systemicmean, errors="coerce")
        z = z[np.isfinite(z.value) & z.value.between(20, 250)].sort_values("offset")
        periodic_by[int(sid)] = (z.offset.to_numpy(int), z.value.to_numpy(float))
    rows = []; map_sources = Counter()
    for r in co.itertuples(index=False):
        sid, offset = int(r.sid), int(r.offset)
        entry = int(r.entry) if hasattr(r, "entry") else offset
        value = float(r.value); map_value = None; map_available = offset; map_source = None
        exact = nurse_exact.get((sid, offset))
        if exact is not None:
            map_value, map_available = exact; map_source = "nurse_invasive_exact"
        else:
            arr = periodic_by.get(sid)
            if arr is not None and len(arr[0]):
                pos = np.searchsorted(arr[0], offset, side="right") - 1
                if pos >= 0 and arr[0][pos] >= offset - 5:
                    map_value = float(arr[1][pos]); map_available = int(arr[0][pos]); map_source = "periodic_invasive_prior5"
            if map_value is None and (sid, offset) in aper_mean:
                map_value = aper_mean[(sid, offset)]; map_available = offset; map_source = "aperiodic_nibp_exact"
        if map_value is not None:
            map_sources[map_source] += 1
            rows.append((sid, offset, max(entry, map_available), value, map_value, value * map_value / 451, source, map_source))
    out = pd.DataFrame(rows, columns=["sid", "offset", "available", "co", "map", "cpo", "co_source", "map_source"])
    return out, {"co_records": int(len(co)), "paired_cpo_records": int(len(out)), "pair_fraction": len(out) / len(co) if len(co) else 0, "map_sources": dict(map_sources)}


def union_cpo(nurse: pd.DataFrame, aper: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    both = pd.concat([nurse, aper], ignore_index=True)
    conflicts = both.groupby(["sid", "offset"]).cpo.nunique().gt(1)
    conflict_keys = set(conflicts[conflicts].index)
    if conflict_keys:
        idx = pd.MultiIndex.from_frame(both[["sid", "offset"]]); both = both[~idx.isin(conflict_keys)]
    both = both.sort_values(["sid", "offset", "available"]).drop_duplicates(["sid", "offset", "cpo"])
    return both, len(conflict_keys)


def landmark_audit(cpo: pd.DataFrame, patients: pd.DataFrame, eligible_sids: set[int], label: str) -> dict:
    meta = patients.set_index("sid").to_dict("index")
    rows = []
    for sid, g in cpo[cpo.sid.isin(eligible_sids)].sort_values("offset").groupby("sid"):
        info = meta.get(int(sid))
        if not info or info["age"] is None or info["age"] < 18 or info["los_min"] is None:
            continue
        limit = min(float(info["los_min"]), 4320)
        g = g[(g.offset >= 0) & (g.offset <= limit)].copy()
        times = g.offset.to_numpy(float) / 60
        available = g.available.to_numpy(float) / 60
        values = g.cpo.to_numpy(float)
        for landmark in range(6, 67):
            if landmark * 60 > limit:
                break
            keep = (times > landmark - 4) & (times <= landmark) & (available <= landmark)
            pos = np.flatnonzero(keep)
            if len(pos) < 3:
                continue
            ht = times[pos]
            if ht[-1] - ht[0] < 2 or landmark - ht[-1] > 1 or values[pos[-1]] < 0.60:
                continue
            future = times[(times > landmark) & (times <= landmark + 6)]
            dense = False
            if len(future) >= 4 and future[-1] >= landmark + 5:
                chain = np.r_[ht[-1], future]
                dense = bool(np.diff(chain).max() <= 2)
            rows.append((int(sid), info["uniquepid"], int(info["hospitalid"]), landmark, dense))
    frame = pd.DataFrame(rows, columns=["sid", "uniquepid", "hospitalid", "landmark_h", "timestamp_dense_6h"])
    if frame.empty:
        return {"analysis": label, "legal_landmarks": 0, "stays": 0, "patients": 0, "hospitals": 0, "timestamp_dense_landmarks": 0, "timestamp_dense_fraction": 0, "hospitals_ge10_patients": 0}, frame
    patient_hospital = frame[["uniquepid", "hospitalid"]].drop_duplicates()
    hosp_counts = patient_hospital.groupby("hospitalid").uniquepid.nunique()
    summary = {
        "analysis": label, "legal_landmarks": int(len(frame)), "stays": int(frame.sid.nunique()),
        "patients": int(frame.uniquepid.nunique()), "hospitals": int(frame.hospitalid.nunique()),
        "timestamp_dense_landmarks": int(frame.timestamp_dense_6h.sum()),
        "timestamp_dense_fraction": float(frame.timestamp_dense_6h.mean()),
        "hospitals_ge10_patients": int((hosp_counts >= 10).sum()),
        "top_hospital_patient_share": float(hosp_counts.max() / hosp_counts.sum()),
    }
    return summary, frame


def source_summary(frame: pd.DataFrame, patients: pd.DataFrame, label: str) -> dict:
    if frame.empty:
        return {"source": label, "records": 0, "stays": 0, "patients": 0, "hospitals": 0}
    m = frame.merge(patients[["sid", "uniquepid", "hospitalid"]], on="sid", how="left")
    gaps = frame.sort_values(["sid", "offset"]).groupby("sid").offset.diff() / 60
    return {
        "source": label, "records": int(len(frame)), "stays": int(frame.sid.nunique()),
        "patients": int(m.uniquepid.nunique()), "hospitals": int(m.hospitalid.nunique()),
        "records_per_stay_median": float(frame.groupby("sid").size().median()),
        "chart_gap_h_median": float(gaps.median()) if gaps.notna().any() else None,
        "chart_gap_h_p75": float(gaps.quantile(.75)) if gaps.notna().any() else None,
    }


def main() -> None:
    for path in [WORK, RESULTS, AUDIT]: path.mkdir(parents=True, exist_ok=True)
    nurse_path = ensure_nurse_targets()
    nurse, nurse_audit = read_nurse_targets(nurse_path)
    aper_co_path = extract_aperiodic_co()
    aper_co_raw = pd.read_csv(aper_co_path).rename(columns={"cardiacoutput": "value"})
    aper_co, aper_co_conflicts = resolve(aper_co_raw, 0.000001, 30, entry=False)
    nurse_co, nurse_co_conflicts = resolve(nurse["co"], 0.000001, 30, entry=True)
    nurse_map, nurse_map_conflicts = resolve(nurse["map"], 20, 250, entry=True)
    nurse_pamean, nurse_pamean_conflicts = resolve(nurse["pamean"], 0, 150, entry=True)
    nurse_svo2, nurse_svo2_conflicts = resolve(nurse["svo2"], 0, 100, entry=True)
    candidate_sids = set(nurse_co.sid.astype(int)) | set(aper_co.sid.astype(int))
    patients = load_patients(candidate_sids)
    candidate_sids = set(patients.loc[patients.age.ge(18) & patients.los_min.notna(), "sid"].astype(int))
    aper_path, periodic_path, scan_counts = scan_candidate_vitals(candidate_sids)
    aper = pd.read_csv(aper_path)
    periodic = pd.read_csv(periodic_path)
    periodic["sid"] = pd.to_numeric(periodic.sid, errors="coerce"); periodic["offset"] = pd.to_numeric(periodic.offset, errors="coerce")
    periodic = periodic[periodic.sid.notna() & periodic.offset.notna()].copy(); periodic[["sid", "offset"]] = periodic[["sid", "offset"]].astype(int)
    support_audit, explicit_pac, iabp = scan_support_and_pac(candidate_sids)

    nurse_cpo, nurse_pair = pair_cpo(nurse_co, nurse_map, periodic, aper, "nurse_CO")
    aper_cpo, aper_pair = pair_cpo(aper_co, nurse_map, periodic, aper, "vitalAperiodic_CO")
    union, union_conflicts = union_cpo(nurse_cpo, aper_cpo)
    for name, frame in [("nurse", nurse_cpo), ("aperiodic", aper_cpo), ("union", union)]:
        frame.to_csv(WORK / f"cpo_{name}.csv.gz", index=False, compression="gzip")

    limit_map = patients.set_index("sid").los_min.to_dict()
    def in_window_sids(frame: pd.DataFrame, value_col: str, low: float, high: float) -> set[int]:
        z = frame.copy()
        z[value_col] = pd.to_numeric(z[value_col], errors="coerce")
        z = z[np.isfinite(z[value_col]) & z[value_col].between(low, high)].copy()
        return {
            int(r.sid) for r in z.itertuples(index=False)
            if int(r.sid) in limit_map and 0 <= float(r.offset) <= min(float(limit_map[int(r.sid)]), 4320)
        }
    periodic_pamean_sids = in_window_sids(periodic, "pamean", 0, 150)
    nurse_pamean_sids = in_window_sids(nurse_pamean, "value", 0, 150)
    nurse_svo2_sids = in_window_sids(nurse_svo2, "value", 0, 100)
    phys_pac = nurse_pamean_sids | nurse_svo2_sids | periodic_pamean_sids
    evidence_sets = {
        "physiological_pac": phys_pac,
        "explicit_pac": explicit_pac,
        "physiological_or_explicit_pac": phys_pac | explicit_pac,
        "co_only": candidate_sids,
    }
    tier_frames = {"nurse": nurse_cpo, "aperiodic": aper_cpo, "union": union}
    landmark_summaries = []; landmark_outputs = []
    for source_name, frame in tier_frames.items():
        for evidence_name, sids in evidence_sets.items():
            summary, lm = landmark_audit(frame, patients, sids, f"{source_name}__{evidence_name}")
            landmark_summaries.append(summary)
            if not lm.empty:
                lm["analysis"] = summary["analysis"]; landmark_outputs.append(lm)
    pd.DataFrame(landmark_summaries).to_csv(RESULTS / "structural_landmark_gate.csv", index=False)
    if landmark_outputs:
        pd.concat(landmark_outputs, ignore_index=True).to_csv(WORK / "structural_legal_landmarks.csv.gz", index=False, compression="gzip")

    source_summaries = [
        source_summary(nurse_co, patients, "valid nurse CO"),
        source_summary(aper_co, patients, "valid vitalAperiodic CO"),
        source_summary(nurse_cpo, patients, "paired nurse-derived CPO"),
        source_summary(aper_cpo, patients, "paired aperiodic-derived CPO"),
        source_summary(union, patients, "paired union CPO"),
    ]
    pd.DataFrame(source_summaries).to_csv(RESULTS / "source_coverage.csv", index=False)
    primary = next(x for x in landmark_summaries if x["analysis"] == "nurse__physiological_pac")
    if primary["patients"] >= 500 and primary["hospitals"] >= 10 and primary["legal_landmarks"] >= 1000 and primary["timestamp_dense_fraction"] >= .50:
        verdict = "STRONG_GO"
    elif primary["patients"] >= 200 and primary["hospitals"] >= 5 and primary["legal_landmarks"] >= 500 and primary["timestamp_dense_fraction"] >= .30:
        verdict = "GO"
    elif primary["patients"] >= 100 and primary["hospitals"] >= 3 and primary["timestamp_dense_fraction"] >= .30:
        verdict = "SUPPORTING_TRANSPORT_ONLY"
    else:
        verdict = "NO_GO_PRIMARY_NURSE_CO"
    best = max(landmark_summaries, key=lambda x: (x["patients"], x["legal_landmarks"]))

    audit = {
        "decision": verdict,
        "primary_structural_gate": primary,
        "largest_prespecified_tier": best,
        "source_pairing": {"nurse": nurse_pair, "aperiodic": aper_pair, "union_cross_source_conflict_offsets": union_conflicts},
        "conflicting_offsets_excluded": {
            "nurse_co": nurse_co_conflicts, "aperiodic_co": aper_co_conflicts, "nurse_map": nurse_map_conflicts,
            "nurse_pamean": nurse_pamean_conflicts, "nurse_svo2": nurse_svo2_conflicts,
        },
        "pac_evidence_stays": {k: len(v) for k, v in evidence_sets.items()},
        "nurse_target_audit": nurse_audit, "support_and_device_audit": support_audit,
        "scan_counts_if_uncached": scan_counts,
        "outcomes_read": False, "model_predictions_generated": False,
    }
    (AUDIT / "stage_eminus1_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    patients.to_csv(WORK / "candidate_patient_metadata.csv.gz", index=False, compression="gzip")

    report = [
        "# eICU Stage E−1 外部验证可行性审计", "", f"**裁决：{verdict}**", "",
        "本阶段没有读取未来低 CPO 结局，也没有生成任何模型预测。", "", "## 主要结构门", "",
        f"- 合法地标：{primary['legal_landmarks']:,}", f"- ICU stays：{primary['stays']:,}",
        f"- 患者：{primary['patients']:,}", f"- 医院：{primary['hospitals']:,}",
        f"- 未来6小时仅按时间戳可密集判定：{primary['timestamp_dense_landmarks']:,}/{primary['legal_landmarks']:,}（{primary['timestamp_dense_fraction']:.1%}）",
        f"- 至少10名患者的医院：{primary['hospitals_ge10_patients']}", "", "## 最大预设来源层", "",
        f"`{best['analysis']}`：{best['patients']:,}名患者、{best['hospitals']}家医院、{best['legal_landmarks']:,}个合法地标，时间戳密集可判定率{best['timestamp_dense_fraction']:.1%}。", "",
        "## 解释", "",
        "Primary 使用 nurseCharting 的通用 CO，并要求 PA mean 或 SvO2 作为生理 PAC 证据。eICU 无法证明这些 CO 与 MIMIC 的 continuous-thermodilution CCO 测量机制相同，因此即使通过，也只能称 transported cardiac-output measurement 下的跨库外部验证。",
    ]
    (ROOT / "REPORT_STAGE_EMINUS1_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print((ROOT / "REPORT_STAGE_EMINUS1_ZH.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
