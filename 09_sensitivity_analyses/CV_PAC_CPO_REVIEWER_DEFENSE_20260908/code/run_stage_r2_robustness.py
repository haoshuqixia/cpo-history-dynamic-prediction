#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
STAGE1A = BASE / "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908"
STAGE1B = BASE / "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908"
PAPER = BASE / "CV_PAC_CPO_PAPER_PACKAGE_20260908"
OUT = ROOT / "stage_r2_robustness"
RESULTS = OUT / "results"
TEMP = "definite_temporal_2020_plus"
SEED = 20260908

sys.path.insert(0, str(STAGE1B / "code"))
from run_stage1b import apply_calibration, metric_set, percentile_interval  # noqa: E402


def predict(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    cols = artifact["numeric_predictors"] + artifact["categorical_predictors"]
    raw = artifact["pipeline"].predict_proba(frame[cols])[:, 1]
    cal = (artifact["development_calibration_intercept"], artifact["development_calibration_slope"])
    return apply_calibration(raw, cal)


def evaluate(label: str, frame: pd.DataFrame, weight_col: str) -> tuple[list[dict], list[dict], pd.DataFrame]:
    y = frame.primary_outcome_6h.to_numpy(int)
    w = frame[weight_col].to_numpy(float)
    metrics = {name: metric_set(y, frame[f"prediction_{name}"].to_numpy(float), w) for name in ["A", "B"]}
    patients = frame.subject_id.unique()
    patient_values = frame.subject_id.to_numpy()
    patient_pos = {pid: np.flatnonzero(patient_values == pid) for pid in patients}
    rng = np.random.default_rng(SEED)
    boot_rows = []
    for replicate in range(1000):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        counts = pd.Series(sampled).value_counts()
        multiplier = np.zeros(len(frame), float)
        for pid, count in counts.items():
            multiplier[patient_pos[pid]] = count
        wb = w * multiplier
        if wb[y == 1].sum() == 0 or wb[y == 0].sum() == 0:
            continue
        one = {"analysis": label, "replicate": replicate}
        for name in ["A", "B"]:
            for metric, estimate in metric_set(y, frame[f"prediction_{name}"].to_numpy(float), wb).items():
                one[f"{name}_{metric}"] = estimate
        boot_rows.append(one)
    boot = pd.DataFrame(boot_rows)
    metric_rows, delta_rows = [], []
    for name in ["A", "B"]:
        for metric, estimate in metrics[name].items():
            lo, hi = percentile_interval(boot[f"{name}_{metric}"])
            metric_rows.append({"analysis": label, "model": name, "metric": metric, "estimate": estimate, "ci_low": lo, "ci_high": hi})
    for metric in metrics["A"]:
        estimate = metrics["B"][metric] - metrics["A"][metric]
        lo, hi = percentile_interval(boot[f"B_{metric}"] - boot[f"A_{metric}"])
        delta_rows.append({"analysis": label, "comparison": "B-A", "metric": metric, "estimate": estimate, "ci_low": lo, "ci_high": hi})
    return metric_rows, delta_rows, boot


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(STAGE1A / "results/LANDMARK_FEATURES_V1.csv.gz", dtype={"stay_id": str})
    weights = pd.read_csv(STAGE1A / "results/observation_weights_v1.csv.gz", dtype={"stay_id": str})
    keys = ["subject_id", "stay_id", "landmark_h", "interval_conservative_split"]
    d = f.merge(weights[keys + ["ipow_stabilized_truncated"]], on=keys, how="left", validate="one_to_one")
    temp_observed = (
        d.interval_conservative_split.eq(TEMP)
        & d.primary_eligible_at_landmark.astype(bool)
        & d.primary_outcome_observed_6h.astype(bool)
    )
    expanded = d.loc[temp_observed].copy()
    expanded["unit_weight"] = 1.0
    common = expanded[expanded.model_bc_common_risk_set.astype(bool)].copy()

    for name in ["A", "B"]:
        artifact = joblib.load(PAPER / "models" / f"model_{name}_locked.joblib")
        expanded[f"prediction_{name}"] = predict(artifact, expanded)
    common = expanded[expanded.model_bc_common_risk_set.astype(bool)].copy()

    locked = pd.read_csv(STAGE1B / "results/temporal_predictions.csv.gz", dtype={"stay_id": str})
    check = common.merge(locked[keys + ["prediction_A", "prediction_B"]], on=keys, how="left", validate="one_to_one", suffixes=("", "_locked"))
    max_difference = {
        name: float(np.max(np.abs(check[f"prediction_{name}"] - check[f"prediction_{name}_locked"])))
        for name in ["A", "B"]
    }
    if max(max_difference.values()) > 1e-12:
        raise RuntimeError(f"Serialized model predictions do not reproduce the locked set: {max_difference}")

    metric_rows, delta_rows, boot_parts = [], [], []
    analyses = [
        ("expanded_cpo_only_weighted", expanded, "ipow_stabilized_truncated"),
        ("locked_common_weighted", common, "ipow_stabilized_truncated"),
        ("locked_common_unweighted", common, "unit_weight"),
    ]
    for label, frame, weight_col in analyses:
        m, delta, boot = evaluate(label, frame, weight_col)
        metric_rows.extend(m)
        delta_rows.extend(delta)
        boot_parts.append(boot)

    metrics = pd.DataFrame(metric_rows)
    deltas = pd.DataFrame(delta_rows)
    metrics.to_csv(RESULTS / "temporal_metrics.csv", index=False)
    deltas.to_csv(RESULTS / "paired_temporal_deltas.csv", index=False)
    pd.concat(boot_parts, ignore_index=True).to_csv(RESULTS / "patient_cluster_bootstrap.csv.gz", index=False, compression="gzip")
    expanded[keys + ["primary_outcome_6h", "model_bc_common_risk_set", "ipow_stabilized_truncated", "prediction_A", "prediction_B"]].to_csv(
        RESULTS / "expanded_temporal_predictions.csv.gz", index=False, compression="gzip"
    )

    counts = []
    for label, frame, _ in analyses:
        counts.append({
            "analysis": label,
            "landmarks": int(len(frame)),
            "patients": int(frame.subject_id.nunique()),
            "event_landmarks": int(frame.primary_outcome_6h.sum()),
            "event_patients": int(frame.loc[frame.primary_outcome_6h.eq(1), "subject_id"].nunique()),
        })
    pd.DataFrame(counts).to_csv(RESULTS / "analysis_counts.csv", index=False)

    def delta(label: str, metric: str) -> pd.Series:
        return deltas[(deltas.analysis == label) & (deltas.metric == metric)].iloc[0]

    report_lines = [
        "# Stage R2 扩展风险集与观察权重稳健性", "",
        "本阶段没有重新拟合、调参或重新校准临床模型。模型 A/B 均为已序列化的锁定模型。", "",
        "## 样本", "",
    ]
    for row in counts:
        report_lines.append(f"- `{row['analysis']}`：{row['landmarks']:,} 个地标，{row['patients']} 名患者，{row['event_landmarks']} 个事件地标，{row['event_patients']} 名事件患者。")
    report_lines += ["", "## B-A 配对差异", "", "| 分析 | Brier差（95% CI） | AUROC差（95% CI） | Log loss差（95% CI） |", "|---|---:|---:|---:|"]
    for label, _, _ in analyses:
        b = delta(label, "brier")
        a = delta(label, "auroc")
        l = delta(label, "log_loss")
        report_lines.append(
            f"| {label} | {b.estimate:+.6f}（{b.ci_low:+.6f}, {b.ci_high:+.6f}） | "
            f"{a.estimate:+.4f}（{a.ci_low:+.4f}, {a.ci_high:+.4f}） | "
            f"{l.estimate:+.6f}（{l.ci_low:+.6f}, {l.ci_high:+.6f}） |"
        )
    report_lines += [
        "", "## 解释", "",
        "扩展风险集用于回答 mPAP/SvO2 完整历史限制是否驱动了主结果；未加权复核用于判断结果是否主要依赖观察权重。二者均为支持性分析。",
    ]
    (OUT / "REPORT_STAGE_R2_ZH.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    audit = {
        "serialized_models_reproduced_locked_predictions": True,
        "maximum_absolute_prediction_difference": max_difference,
        "models_refit": False,
        "models_recalibrated": False,
        "bootstrap_replicates_per_analysis": 1000,
        "seed": SEED,
        "counts": counts,
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print((OUT / "REPORT_STAGE_R2_ZH.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
