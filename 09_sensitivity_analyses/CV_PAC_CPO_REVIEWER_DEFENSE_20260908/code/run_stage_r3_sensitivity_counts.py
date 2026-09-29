#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
PREV = BASE / "STAGE_MINUS1A_CV_OBSERVABILITY_V1" / "work"
STAGE0 = BASE / "CV_PAC_CPO_STAGE0_MECHANISM_20260908"
OUT = ROOT / "stage_r3_sensitivity_counts"
DEV = "definite_development_to_2019"
TEMP = "definite_temporal_2020_plus"

sys.path.insert(0, str(STAGE0 / "code"))
from run_stage0_mechanism_audit import build_axis_times  # noqa: E402


def has_event(times: np.ndarray, values: np.ndarray, landmark: float, horizon: float, threshold: float) -> bool:
    low_times = times[(times > landmark) & (times <= landmark + horizon) & (values < threshold)]
    if len(low_times) < 2:
        return False
    gaps = np.diff(low_times)
    return bool(np.any((gaps >= 0.5) & (gaps <= 3.0)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    landmarks = pd.read_csv(STAGE0 / "results/LANDMARK_DATASET_V1.csv.gz", dtype={"stay_id": str})
    stays = pd.read_csv(PREV / "adult_stays.csv", dtype={"stay_id": str}, parse_dates=["intime", "outtime"])
    stays = stays[stays.stay_id.isin(set(landmarks.stay_id))].copy()
    cpo, _, _ = build_axis_times(stays)
    groups = {
        stay_id: (g.hours.to_numpy(float), g.cpo.to_numpy(float))
        for stay_id, g in cpo.sort_values("hours").groupby("stay_id")
    }

    rows = []
    for r in landmarks[["stay_id", "landmark_h"]].itertuples(index=False):
        times, values = groups.get(r.stay_id, (np.array([], float), np.array([], float)))
        row = {"stay_id": r.stay_id, "landmark_h": int(r.landmark_h)}
        for threshold, threshold_name in [(0.60, "0p60"), (0.50, "0p50")]:
            row[f"event_{threshold_name}_6h"] = has_event(times, values, float(r.landmark_h), 6.0, threshold)
            row[f"event_{threshold_name}_12h"] = (
                has_event(times, values, float(r.landmark_h), 12.0, threshold)
                if r.landmark_h <= 60 else np.nan
            )
        rows.append(row)
    flags = pd.DataFrame(rows)
    d = landmarks.merge(flags, on=["stay_id", "landmark_h"], how="left", validate="one_to_one")
    eligible = d.primary_eligible_at_landmark.astype(bool)
    reconstruction_errors = int((d.loc[eligible, "event_0p60_6h"].astype(bool) != d.loc[eligible, "event_6h"].astype(bool)).sum())
    if reconstruction_errors:
        raise RuntimeError(f"Existing 0.60-W six-hour outcome reconstruction mismatch: {reconstruction_errors}")

    count_rows = []
    populations = {
        "primary_eligible": eligible,
        "primary_eligible_common_axes": eligible & d.model_bc_common_risk_set.astype(bool),
    }
    splits = {"all": pd.Series(True, index=d.index), "development": d.interval_conservative_split.eq(DEV), "temporal": d.interval_conservative_split.eq(TEMP)}
    for population, pop_mask in populations.items():
        for split, split_mask in splits.items():
            base = pop_mask & split_mask
            for horizon in [6, 12]:
                horizon_mask = base & (d.landmark_h <= 60 if horizon == 12 else True)
                for threshold_name in ["0p60", "0p50"]:
                    event_col = f"event_{threshold_name}_{horizon}h"
                    event = horizon_mask & d[event_col].fillna(False).astype(bool)
                    count_rows.append({
                        "population": population,
                        "split": split,
                        "threshold_w": float(threshold_name.replace("p", ".")),
                        "horizon_h": horizon,
                        "eligible_landmarks": int(horizon_mask.sum()),
                        "eligible_patients": int(d.loc[horizon_mask, "subject_id"].nunique()),
                        "event_positive_landmarks": int(event.sum()),
                        "event_positive_patients": int(d.loc[event, "subject_id"].nunique()),
                    })
    counts = pd.DataFrame(count_rows)
    counts.to_csv(OUT / "corrected_landmark_sensitivity_counts.csv", index=False)
    d[["subject_id", "stay_id", "landmark_h", "event_0p60_6h", "event_0p60_12h", "event_0p50_6h", "event_0p50_12h"]].to_csv(
        OUT / "landmark_sensitivity_flags.csv.gz", index=False, compression="gzip"
    )

    key = counts[(counts.population == "primary_eligible_common_axes") & (counts.split == "temporal")]
    report = [
        "# Stage R3 预先声明终点敏感性计数", "",
        "本阶段只重建事件计数，没有拟合或评价预测模型。", "",
        "## 时间验证期共同风险集", "",
        "| 阈值 | 预测窗 | 合格地标 | 事件地标 | 事件患者 |", "|---:|---:|---:|---:|---:|",
    ]
    for row in key.sort_values(["threshold_w", "horizon_h"], ascending=[False, True]).itertuples(index=False):
        report.append(f"| <{row.threshold_w:.2f} W | {row.horizon_h} h | {row.eligible_landmarks:,} | {row.event_positive_landmarks:,} | {row.event_positive_patients:,} |")
    report += [
        "", "## 协议解释", "",
        "0.50 W 和 12 小时在最初候选筛查协议中仅被声明为可行性/敏感性计数，因此这里不补做模型性能分析。早期筛查使用过未来信息选择最后正常时点，不能作为预测性能证据；本表改用纠正后的合法滚动地标。",
        "", "最初协议还提到低 CPO 同时 SvO2 <60%，但没有预先定义两个测量的时间匹配规则。为避免看过数据后选择匹配窗，该分析不运行，并在协议演化表中公开记录。",
    ]
    (OUT / "REPORT_STAGE_R3_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    audit = {
        "existing_0p60_6h_reconstruction_errors": reconstruction_errors,
        "models_fitted": False,
        "concurrent_svo2_lt60_run": False,
        "concurrent_svo2_lt60_reason": "Initial protocol did not operationally define a CPO-SvO2 time-matching rule.",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print((OUT / "REPORT_STAGE_R3_ZH.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
