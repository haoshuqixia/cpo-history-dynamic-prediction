#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
PREV = BASE / "STAGE_MINUS1A_CV_OBSERVABILITY_V1" / "work"
STAGE0 = BASE / "CV_PAC_CPO_STAGE0_MECHANISM_20260908"
STAGE1A = BASE / "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908"
STAGE1B = BASE / "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908"
OUT = ROOT / "stage_r1_storetime"
RESULTS = OUT / "results"
AUDIT = OUT / "audit"
DEV = "definite_development_to_2019"
TEMP = "definite_temporal_2020_plus"
SEED = 20260908
THRESHOLDS = np.array([0.02, 0.03, 0.05, 0.075, 0.10, 0.15])

sys.path.insert(0, str(STAGE0 / "code"))
sys.path.insert(0, str(STAGE1A / "code"))
sys.path.insert(0, str(STAGE1B / "code"))

from run_stage0_mechanism_audit import build_axis_times  # noqa: E402
from run_stage1a import history_features, make_pipeline, calibration_table, probability_summary, weight_summary  # noqa: E402
from run_stage1b import (  # noqa: E402
    BLOCKS, CATEGORICAL, MODEL_C, make_model, weighted_calibration,
    apply_calibration, metric_set, net_benefit, percentile_interval,
)
from run_information_clock_audit import add_store_times, attach_store  # noqa: E402


def make_timesafe_features() -> tuple[pd.DataFrame, dict]:
    base = pd.read_csv(STAGE0 / "results/LANDMARK_DATASET_V1.csv.gz", dtype={"stay_id": str})
    original = pd.read_csv(STAGE1A / "results/LANDMARK_FEATURES_V1.csv.gz", dtype={"stay_id": str})
    static_cols = [
        "subject_id", "stay_id", "landmark_h", "age", "landmark_h_sq",
        "gender", "first_careunit",
    ]
    original = original[static_cols].copy()
    stays_all = pd.read_csv(PREV / "adult_stays.csv", dtype={"stay_id": str}, parse_dates=["intime", "outtime"])
    wanted = set(base.stay_id)
    stays = stays_all[stays_all.stay_id.isin(wanted)].copy()

    cpo, mpap, svo2 = build_axis_times(stays)
    cpo = add_store_times(cpo, wanted)
    mpap = attach_store(mpap, "220061", wanted)
    svo2 = attach_store(svo2, "223772", wanted)
    groups = {}
    for name, frame, value in [("cpo", cpo, "cpo"), ("mpap", mpap, "v"), ("svo2", svo2, "v")]:
        groups[name] = {
            stay_id: (
                g["hours"].to_numpy(float),
                g[value].to_numpy(float),
                g["available_h"].to_numpy(float),
            )
            for stay_id, g in frame.sort_values("hours").groupby("stay_id")
        }

    rows = []
    for r in base[["stay_id", "landmark_h"]].itertuples(index=False):
        landmark = float(r.landmark_h)
        row = {"stay_id": r.stay_id, "landmark_h": int(r.landmark_h)}
        safe_arrays = {}
        for name in ["cpo", "mpap", "svo2"]:
            times, values, available = groups[name].get(
                r.stay_id,
                (np.array([], float), np.array([], float), np.array([], float)),
            )
            keep = (times > landmark - 4) & (times <= landmark) & np.isfinite(available) & (available <= landmark)
            t, v = times[keep], values[keep]
            safe_arrays[name] = (t, v)
            row.update(history_features(t, v, landmark, name))
            row[f"n_{name}_history_4h"] = int(len(v))
            row[f"{name}_recency_h"] = float(landmark - t[-1]) if len(t) else np.nan
        c_t, c_v = safe_arrays["cpo"]
        row["timesafe_cpo_eligible"] = bool(
            len(c_v) >= 3
            and (c_t[-1] - c_t[0]) >= 2
            and landmark - c_t[-1] <= 1
            and c_v[-1] >= 0.60
        )
        row["timesafe_common_axes"] = bool(
            row["n_mpap_history_4h"] >= 2 and row["n_svo2_history_4h"] >= 2
        )
        rows.append(row)

    dynamic = pd.DataFrame(rows)
    keys = ["stay_id", "landmark_h"]
    out = base.merge(dynamic, on=keys, how="left", validate="one_to_one", suffixes=("_original", ""))
    out = out.merge(original, on=["subject_id", *keys], how="left", validate="one_to_one")
    primary = out["primary_eligible_at_landmark"].astype(bool)
    out["timesafe_primary_eligible"] = primary & out["timesafe_cpo_eligible"] & out["timesafe_common_axes"]

    dyn_cols = [c for c in MODEL_C if c.startswith(("cpo_", "mpap_", "svo2_"))] + [
        "n_cpo_history_4h", "n_mpap_history_4h", "n_svo2_history_4h"
    ]
    eligible = out["timesafe_primary_eligible"]
    missing = out.loc[eligible, sorted(set(dyn_cols))].isna().any(axis=1)
    audit = {
        "landmarks_rebuilt": int(len(out)),
        "unique_landmark_key": bool(~out.duplicated(keys).any()),
        "timesafe_eligible_rows": int(eligible.sum()),
        "timesafe_eligible_rows_with_any_missing_dynamic_predictor": int(missing.sum()),
        "baseline_columns_reused_from_locked_stage1a": [c for c in static_cols if c not in ["subject_id", "stay_id", "landmark_h"]],
        "treatment_columns_reused_from_locked_stage0": [
            "iabp_active_at_landmark",
            "drug_norepinephrine_active_at_landmark", "drug_epinephrine_active_at_landmark",
            "drug_vasopressin_active_at_landmark", "drug_phenylephrine_active_at_landmark",
            "drug_dopamine_active_at_landmark", "drug_dobutamine_active_at_landmark",
            "drug_milrinone_active_at_landmark",
        ],
        "haemodynamic_axes_rebuilt_with_storetime": ["derived CPO", "mPAP item 220061", "SvO2 item 223772"],
    }
    return out, audit


def fit_observation_model(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    eligible = d["timesafe_primary_eligible"].astype(bool)
    dev = eligible & d["interval_conservative_split"].eq(DEV)
    temp = eligible & d["interval_conservative_split"].eq(TEMP)
    y = d["primary_outcome_observed_6h"].astype(int)
    pred = pd.Series(np.nan, index=d.index, dtype=float)
    numeric = MODEL_C
    cols = numeric + CATEGORICAL
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    xdev = d.loc[dev, cols]
    ydev = y.loc[dev]
    groups = d.loc[dev, "subject_id"]
    for train_pos, test_pos in cv.split(xdev, ydev, groups=groups):
        model = make_pipeline(numeric, CATEGORICAL)
        model.fit(xdev.iloc[train_pos], ydev.iloc[train_pos])
        pred.loc[xdev.index[test_pos]] = model.predict_proba(xdev.iloc[test_pos])[:, 1]
    final = make_pipeline(numeric, CATEGORICAL)
    final.fit(xdev, ydev)
    if temp.any():
        pred.loc[temp] = final.predict_proba(d.loc[temp, cols])[:, 1]
    other = eligible & ~dev & ~temp
    if other.any():
        pred.loc[other] = final.predict_proba(d.loc[other, cols])[:, 1]
    assert pred.loc[eligible].notna().all()

    numerator = float(y.loc[dev].mean())
    observed = eligible & d["primary_outcome_observed_6h"].astype(bool)
    d["p_outcome_observed_6h_timesafe"] = pred
    d["ipow_timesafe_raw"] = np.where(observed, 1 / pred, np.nan)
    d["ipow_timesafe_stabilized"] = np.where(observed, numerator / pred, np.nan)
    lo, hi = [float(x) for x in d.loc[dev & observed, "ipow_timesafe_stabilized"].quantile([0.01, 0.99])]
    d["ipow_timesafe_stabilized_truncated"] = d["ipow_timesafe_stabilized"].clip(lo, hi)

    prob_rows, weight_rows, cal_parts = [], [], []
    for label, mask in [("development_oof", dev), ("temporal_locked", temp)]:
        yy = y.loc[mask].to_numpy()
        pp = pred.loc[mask].to_numpy()
        prob_rows.append(probability_summary(yy, pp, label))
        cal_parts.append(calibration_table(yy, pp, label))
        obs = mask & d["primary_outcome_observed_6h"].astype(bool)
        for kind, col in [
            ("stabilized_raw", "ipow_timesafe_stabilized"),
            ("stabilized_truncated", "ipow_timesafe_stabilized_truncated"),
        ]:
            weight_rows.append(weight_summary(d.loc[obs, col].to_numpy(float), label, kind))
    pd.DataFrame(prob_rows).to_csv(RESULTS / "observation_probability_summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(RESULTS / "observation_weight_summary.csv", index=False)
    pd.concat(cal_parts, ignore_index=True).to_csv(RESULTS / "observation_calibration_deciles.csv", index=False)
    return d, {
        "development_observation_fraction": numerator,
        "development_weight_truncation_q01": lo,
        "development_weight_truncation_q99": hi,
        "observation_model": "L2 logistic; five-fold stratified patient-group cross-fitting",
    }


def fit_ab(d: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    common = d["timesafe_primary_eligible"].astype(bool) & d["primary_outcome_observed_6h"].astype(bool)
    dev = common & d["interval_conservative_split"].eq(DEV)
    temp = common & d["interval_conservative_split"].eq(TEMP)
    y = d["primary_outcome_6h"].astype("Int64")
    predictions = {}
    calibration_maps = []
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name in ["A", "B"]:
        numeric = BLOCKS[name]
        cols = numeric + CATEGORICAL
        xdev = d.loc[dev, cols]
        ydev = y.loc[dev].astype(int)
        groups = d.loc[dev, "subject_id"]
        sw = d.loc[dev, "ipow_timesafe_stabilized_truncated"].to_numpy(float)
        oof = np.full(len(xdev), np.nan)
        for train_pos, test_pos in cv.split(xdev, ydev, groups=groups):
            model = make_model(numeric)
            model.fit(xdev.iloc[train_pos], ydev.iloc[train_pos], model__sample_weight=sw[train_pos])
            oof[test_pos] = model.predict_proba(xdev.iloc[test_pos])[:, 1]
        cal = weighted_calibration(ydev.to_numpy(), oof, sw)
        calibration_maps.append({"model": name, "development_oof_intercept": cal[0], "development_oof_slope": cal[1]})
        final = make_model(numeric)
        final.fit(xdev, ydev, model__sample_weight=sw)
        predictions[name] = apply_calibration(final.predict_proba(d.loc[temp, cols])[:, 1], cal)

    keys = ["subject_id", "stay_id", "landmark_h", "interval_conservative_split"]
    t = d.loc[temp, keys + ["primary_outcome_6h", "ipow_timesafe_stabilized_truncated"]].copy()
    t["primary_outcome_6h"] = t["primary_outcome_6h"].astype(int)
    for name in ["A", "B"]:
        t[f"prediction_{name}"] = predictions[name]
    t.to_csv(RESULTS / "temporal_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(calibration_maps).to_csv(RESULTS / "development_calibration_maps.csv", index=False)

    yt = t["primary_outcome_6h"].to_numpy(int)
    wt = t["ipow_timesafe_stabilized_truncated"].to_numpy(float)
    metric_rows = []
    dca_rows = []
    for name in ["A", "B"]:
        for metric, estimate in metric_set(yt, predictions[name], wt).items():
            metric_rows.append({"model": name, "metric": metric, "estimate": estimate})
        for threshold in THRESHOLDS:
            dca_rows.append({"model": name, "threshold": threshold, "net_benefit": net_benefit(yt, predictions[name], wt, threshold)})

    patients = t["subject_id"].unique()
    patient_values = t["subject_id"].to_numpy()
    patient_pos = {pid: np.flatnonzero(patient_values == pid) for pid in patients}
    rng = np.random.default_rng(SEED)
    boot_rows = []
    for replicate in range(1000):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        counts = pd.Series(sampled).value_counts()
        multiplier = np.zeros(len(t), float)
        for pid, count in counts.items():
            multiplier[patient_pos[pid]] = count
        wb = wt * multiplier
        if wb[yt == 1].sum() == 0 or wb[yt == 0].sum() == 0:
            continue
        one = {"replicate": replicate}
        for name in ["A", "B"]:
            for metric, estimate in metric_set(yt, predictions[name], wb).items():
                one[f"{name}_{metric}"] = estimate
            for threshold in THRESHOLDS:
                one[f"{name}_nb_{threshold:g}"] = net_benefit(yt, predictions[name], wb, threshold)
        boot_rows.append(one)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(RESULTS / "patient_cluster_bootstrap.csv.gz", index=False, compression="gzip")

    metrics = pd.DataFrame(metric_rows)
    for i, row in metrics.iterrows():
        metrics.loc[i, ["ci_low", "ci_high"]] = percentile_interval(boot[f"{row.model}_{row.metric}"])
    metrics.to_csv(RESULTS / "temporal_metrics.csv", index=False)
    dca = pd.DataFrame(dca_rows)
    for i, row in dca.iterrows():
        dca.loc[i, ["ci_low", "ci_high"]] = percentile_interval(boot[f"{row.model}_nb_{row.threshold:g}"])
    dca.to_csv(RESULTS / "temporal_decision_curve.csv", index=False)

    delta_rows = []
    for metric in ["brier", "auroc", "average_precision", "log_loss", "calibration_intercept", "calibration_slope"]:
        est = metrics.loc[(metrics.model == "B") & (metrics.metric == metric), "estimate"].iloc[0] - metrics.loc[(metrics.model == "A") & (metrics.metric == metric), "estimate"].iloc[0]
        lo, hi = percentile_interval(boot[f"B_{metric}"] - boot[f"A_{metric}"])
        delta_rows.append({"comparison": "B-A", "metric": metric, "estimate": float(est), "ci_low": lo, "ci_high": hi})
    for threshold in THRESHOLDS:
        est_b = dca.loc[(dca.model == "B") & np.isclose(dca.threshold, threshold), "net_benefit"].iloc[0]
        est_a = dca.loc[(dca.model == "A") & np.isclose(dca.threshold, threshold), "net_benefit"].iloc[0]
        lo, hi = percentile_interval(boot[f"B_nb_{threshold:g}"] - boot[f"A_nb_{threshold:g}"])
        delta_rows.append({"comparison": "B-A", "metric": f"net_benefit_{threshold:g}", "estimate": float(est_b - est_a), "ci_low": lo, "ci_high": hi})
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(RESULTS / "paired_temporal_deltas.csv", index=False)

    brier = deltas[deltas.metric.eq("brier")].iloc[0]
    auroc = deltas[deltas.metric.eq("auroc")].iloc[0]
    if brier.ci_high < 0 and auroc.ci_low > 0:
        verdict = "ROBUST_STRICT_INFORMATION_CLOCK_SUPPORT"
    elif brier.estimate < 0 and auroc.estimate > 0:
        verdict = "DIRECTIONALLY_SUPPORTIVE_STRICT_INFORMATION_CLOCK"
    else:
        verdict = "NO_STRICT_INFORMATION_CLOCK_SUPPORT"
    decision = {
        "decision": verdict,
        "development_landmarks": int(dev.sum()),
        "development_patients": int(d.loc[dev, "subject_id"].nunique()),
        "development_event_landmarks": int(d.loc[dev, "primary_outcome_6h"].sum()),
        "development_event_patients": int(d.loc[dev & d.primary_outcome_6h.eq(1), "subject_id"].nunique()),
        "temporal_landmarks": int(temp.sum()),
        "temporal_patients": int(t.subject_id.nunique()),
        "temporal_event_landmarks": int(t.primary_outcome_6h.sum()),
        "temporal_event_patients": int(t.loc[t.primary_outcome_6h.eq(1), "subject_id"].nunique()),
        "bootstrap_replicates": int(len(boot)),
        "b_minus_a_brier": {"estimate": float(brier.estimate), "ci_low": float(brier.ci_low), "ci_high": float(brier.ci_high)},
        "b_minus_a_auroc": {"estimate": float(auroc.estimate), "ci_low": float(auroc.ci_low), "ci_high": float(auroc.ci_high)},
    }
    return decision, metrics, deltas


def write_report(decision: dict, metrics: pd.DataFrame, deltas: pd.DataFrame, observation: dict) -> None:
    def metric(model: str, name: str) -> pd.Series:
        return metrics[(metrics.model == model) & (metrics.metric == name)].iloc[0]

    brier = deltas[deltas.metric.eq("brier")].iloc[0]
    auroc = deltas[deltas.metric.eq("auroc")].iloc[0]
    ap = deltas[deltas.metric.eq("average_precision")].iloc[0]
    ll = deltas[deltas.metric.eq("log_loss")].iloc[0]
    ess = pd.read_csv(RESULTS / "observation_weight_summary.csv")
    temp_ess = ess[(ess.split == "temporal_locked") & (ess.kind == "stabilized_truncated")].iloc[0]
    text = f"""# Stage R1 严格信息时钟 A/B 敏感性分析

## 裁决

**{decision['decision']}**

本分析只允许使用在地标时点前已经写入 EHR 的 CPO、mPAP 和 SvO2 记录，并重新构建观察权重、模型 A 与模型 B。它用于检验原始结果是否可能由事后补录的血流动力学记录造成。

## 样本

- 开发集：{decision['development_landmarks']:,} 个地标，{decision['development_patients']:,} 名患者，{decision['development_event_landmarks']:,} 个事件地标，{decision['development_event_patients']:,} 名事件患者。
- 时间验证集：{decision['temporal_landmarks']:,} 个地标，{decision['temporal_patients']:,} 名患者，{decision['temporal_event_landmarks']:,} 个事件地标，{decision['temporal_event_patients']:,} 名事件患者。
- 时间验证集截尾稳定观察权重 ESS：{temp_ess.ess:.1f}/{int(temp_ess.n_observed):,}（{temp_ess.ess_fraction:.1%}）。

## 时间验证性能

| 指标 | 模型 A | 模型 B | B-A（95% CI） |
|---|---:|---:|---:|
| Brier | {metric('A','brier').estimate:.6f} | {metric('B','brier').estimate:.6f} | {brier.estimate:+.6f}（{brier.ci_low:+.6f}, {brier.ci_high:+.6f}） |
| AUROC | {metric('A','auroc').estimate:.4f} | {metric('B','auroc').estimate:.4f} | {auroc.estimate:+.4f}（{auroc.ci_low:+.4f}, {auroc.ci_high:+.4f}） |
| 平均精确率 | {metric('A','average_precision').estimate:.4f} | {metric('B','average_precision').estimate:.4f} | {ap.estimate:+.4f}（{ap.ci_low:+.4f}, {ap.ci_high:+.4f}） |
| Log loss | {metric('A','log_loss').estimate:.6f} | {metric('B','log_loss').estimate:.6f} | {ll.estimate:+.6f}（{ll.ci_low:+.6f}, {ll.ci_high:+.6f}） |

模型 A 为共享临床背景加当前 CPO；模型 B 在此基础上加入最近四小时 CPO 历史特征。所有模型结构、超参数、开发/验证划分和评价流程均与锁定主分析一致。

## 解释边界

该结果支持或限制“使用预测时已写入 EHR 的血流动力学数据”的主张。原始 `charttime` 主分析仍代表按测量发生时点构建的模型。药物与 IABP 指标保留原定义，因为它们表示地标时实际存在的治疗；本敏感性专门处理血流动力学测量的存储延迟。
"""
    (OUT / "REPORT_STAGE_R1_ZH.md").write_text(text, encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    d, feature_audit = make_timesafe_features()
    d, observation_audit = fit_observation_model(d)
    decision, metrics, deltas = fit_ab(d)

    save_cols = [
        "subject_id", "hadm_id", "stay_id", "landmark_h", "interval_conservative_split",
        "primary_outcome_observed_6h", "primary_outcome_6h", "timesafe_cpo_eligible",
        "timesafe_common_axes", "timesafe_primary_eligible", *BLOCKS["B"], *CATEGORICAL,
        "p_outcome_observed_6h_timesafe", "ipow_timesafe_stabilized_truncated",
    ]
    d[save_cols].to_csv(RESULTS / "LANDMARK_FEATURES_STORETIME_AB.csv.gz", index=False, compression="gzip")
    (RESULTS / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    audit = {
        **feature_audit,
        **observation_audit,
        "feature_blocks": {name: BLOCKS[name] + CATEGORICAL for name in ["A", "B"]},
        "estimator": "HistGradientBoostingClassifier fixed hyperparameters",
        "clinical_model_hyperparameter_tuning": False,
        "temporal_recalibration": False,
        "bootstrap_seed": SEED,
        "bootstrap_unit": "patient",
    }
    (AUDIT / "stage_r1_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(decision, metrics, deltas, observation_audit)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
