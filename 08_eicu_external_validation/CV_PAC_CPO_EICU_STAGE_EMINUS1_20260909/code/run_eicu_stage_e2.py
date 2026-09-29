#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"
WORK = ROOT / "work"
MODELS = BASE / "CV_PAC_CPO_PAPER_PACKAGE_20260908" / "models"
SEED = 20260908
N_BOOT = 1000


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, float), -35, 35)
    return 1 / (1 + np.exp(-x))


def apply_locked_model(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    cols = artifact["numeric_predictors"] + artifact["categorical_predictors"]
    raw = artifact["pipeline"].predict_proba(frame[cols])[:, 1]
    return expit(
        float(artifact["development_calibration_intercept"])
        + float(artifact["development_calibration_slope"]) * logit(raw)
    )


def weighted_calibration(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    x = np.column_stack([np.ones(len(p)), logit(p)])
    beta = np.array([0.0, 1.0])
    for _ in range(100):
        mu = expit(x @ beta)
        grad = x.T @ (w * (y - mu))
        h = x.T @ ((w * mu * (1 - mu))[:, None] * x)
        step = np.linalg.solve(h + np.eye(2) * 1e-8, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, float]:
    intercept, slope = weighted_calibration(y, p, w)
    return {
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
        "auroc": float(roc_auc_score(y, p, sample_weight=w)),
        "log_loss": float(log_loss(y, p, sample_weight=w)),
        "average_precision": float(average_precision_score(y, p, sample_weight=w)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def comparison_metrics(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, w: np.ndarray) -> dict[str, float]:
    return {
        "delta_brier_B_minus_A": float(brier_score_loss(y, pb, sample_weight=w) - brier_score_loss(y, pa, sample_weight=w)),
        "delta_auroc_B_minus_A": float(roc_auc_score(y, pb, sample_weight=w) - roc_auc_score(y, pa, sample_weight=w)),
        "delta_log_loss_B_minus_A": float(log_loss(y, pb, sample_weight=w) - log_loss(y, pa, sample_weight=w)),
        "delta_average_precision_B_minus_A": float(average_precision_score(y, pb, sample_weight=w) - average_precision_score(y, pa, sample_weight=w)),
    }


def two_level_bootstrap(frame: pd.DataFrame, y: np.ndarray, pa: np.ndarray, pb: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    hospitals = frame.hospitalid.unique()
    patient_ids = {h: frame.loc[frame.hospitalid.eq(h), "uniquepid"].unique() for h in hospitals}
    positions = {
        (h, pid): np.flatnonzero(frame.hospitalid.to_numpy() == h)[frame.loc[frame.hospitalid.eq(h), "uniquepid"].to_numpy() == pid]
        for h in hospitals for pid in patient_ids[h]
    }
    rng = np.random.default_rng(SEED)
    rows = []
    for replicate in range(N_BOOT):
        mult = np.zeros(len(frame), float)
        for h in rng.choice(hospitals, size=len(hospitals), replace=True):
            pats = patient_ids[h]
            for pid, count in Counter(rng.choice(pats, size=len(pats), replace=True)).items():
                mult[positions[(h, pid)]] += count
        wb = w * mult
        if wb[y == 1].sum() <= 0 or wb[y == 0].sum() <= 0:
            continue
        rows.append({"replicate": replicate, **comparison_metrics(y, pa, pb, wb)})
    return pd.DataFrame(rows)


def main() -> None:
    keys = ["sid", "uniquepid", "hospitalid", "landmark_h"]
    features = pd.read_csv(WORK / "stage_e1b_external_features.csv.gz")
    outcomes = pd.read_csv(
        RESULTS / "stage_e0_landmarks.csv.gz",
        usecols=[*keys, "outcome_observed_6h", "event_6h"],
    )
    final_weights = pd.read_csv(
        RESULTS / "stage_e1b_observation_weights_final.csv.gz",
        usecols=[*keys, "ipow_stabilized_truncated"],
    ).rename(columns={"ipow_stabilized_truncated": "weight_primary"})
    original_weights = pd.read_csv(
        RESULTS / "stage_e1b_observation_weights.csv.gz",
        usecols=[*keys, "ipow_stabilized_truncated"],
    ).rename(columns={"ipow_stabilized_truncated": "weight_original_sensitivity"})
    d = (
        features.merge(outcomes, on=keys, how="inner", validate="one_to_one")
        .merge(final_weights, on=keys, how="left", validate="one_to_one")
        .merge(original_weights, on=keys, how="left", validate="one_to_one")
    )
    d["outcome_observed_6h"] = d.outcome_observed_6h.astype(bool)
    d["event_6h"] = d.event_6h.astype(bool)
    t = d[d.outcome_observed_6h].reset_index(drop=True).copy()
    assert t.weight_primary.notna().all() and t.weight_original_sensitivity.notna().all()

    artifacts = {name: joblib.load(MODELS / f"model_{name}_locked.joblib") for name in ["A", "B"]}
    for name, artifact in artifacts.items():
        t[f"prediction_{name}"] = apply_locked_model(artifact, t)
    y = t.event_6h.astype(int).to_numpy()
    pa = t.prediction_A.to_numpy(float)
    pb = t.prediction_B.to_numpy(float)

    schemes = {
        "primary_ipow_floor_and_q01q99": t.weight_primary.to_numpy(float),
        "sensitivity_ipow_q01q99_only": t.weight_original_sensitivity.to_numpy(float),
        "sensitivity_unweighted": np.ones(len(t), float),
    }
    absolute_rows = []
    comparison_rows = []
    for scheme, w in schemes.items():
        for name, p in [("A", pa), ("B", pb)]:
            one = metrics(y, p, w)
            absolute_rows.extend({"analysis": scheme, "model": name, "metric": metric, "estimate": value} for metric, value in one.items())
        comparison_rows.extend({"analysis": scheme, "metric": metric, "estimate": value} for metric, value in comparison_metrics(y, pa, pb, w).items())
    absolute = pd.DataFrame(absolute_rows)
    comparisons = pd.DataFrame(comparison_rows)
    absolute.to_csv(RESULTS / "stage_e2_absolute_metrics.csv", index=False)
    comparisons.to_csv(RESULTS / "stage_e2_model_comparisons.csv", index=False)

    boot = two_level_bootstrap(t, y, pa, pb, schemes["primary_ipow_floor_and_q01q99"])
    boot.to_csv(RESULTS / "stage_e2_two_level_bootstrap.csv.gz", index=False, compression="gzip")
    ci_rows = []
    primary_points = comparisons[comparisons.analysis.eq("primary_ipow_floor_and_q01q99")].set_index("metric").estimate
    for metric in [c for c in boot.columns if c != "replicate"]:
        lo, hi = boot[metric].quantile([.025, .975])
        ci_rows.append({"metric": metric, "estimate": float(primary_points.loc[metric]), "ci95_low": float(lo), "ci95_high": float(hi), "valid_bootstrap_replicates": int(len(boot))})
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(RESULTS / "stage_e2_primary_comparison_ci.csv", index=False)

    hospital_rows = []
    for hospitalid, g in t.groupby("hospitalid"):
        event_patients = int(g.loc[g.event_6h, "uniquepid"].nunique())
        nonevent_patients = int(g.loc[~g.event_6h, "uniquepid"].nunique())
        if event_patients < 5:
            continue
        idx = g.index.to_numpy()
        yh = y[idx]
        wh = schemes["primary_ipow_floor_and_q01q99"][idx]
        ba = float(brier_score_loss(yh, pa[idx], sample_weight=wh))
        bb = float(brier_score_loss(yh, pb[idx], sample_weight=wh))
        auroc_a = auroc_b = np.nan
        if event_patients >= 10 and nonevent_patients >= 10 and len(np.unique(yh)) == 2:
            auroc_a = float(roc_auc_score(yh, pa[idx], sample_weight=wh))
            auroc_b = float(roc_auc_score(yh, pb[idx], sample_weight=wh))
        hospital_rows.append({
            "hospitalid": int(hospitalid), "observed_landmarks": int(len(g)),
            "event_landmarks": int(g.event_6h.sum()), "event_patients": event_patients,
            "nonevent_patients": nonevent_patients,
            "brier_A": ba, "brier_B": bb, "delta_brier_B_minus_A": bb - ba,
            "auroc_A": auroc_a, "auroc_B": auroc_b,
            "delta_auroc_B_minus_A": auroc_b - auroc_a if np.isfinite(auroc_a) else np.nan,
        })
    hospital = pd.DataFrame(hospital_rows).sort_values(["event_patients", "hospitalid"], ascending=[False, True])
    hospital.to_csv(RESULTS / "stage_e2_hospital_heterogeneity.csv", index=False)

    t[[*keys, "event_6h", "weight_primary", "weight_original_sensitivity", "prediction_A", "prediction_B"]].to_csv(
        RESULTS / "stage_e2_external_predictions.csv.gz", index=False, compression="gzip"
    )

    points = primary_points.to_dict()
    brier_ci = ci.set_index("metric").loc["delta_brier_B_minus_A"]
    direction = (
        points["delta_brier_B_minus_A"] < 0
        and points["delta_auroc_B_minus_A"] > 0
        and points["delta_log_loss_B_minus_A"] < 0
    )
    if direction and brier_ci.ci95_high < 0:
        decision = "STRONG_EXTERNAL_REPLICATION"
    elif direction:
        decision = "SUPPORTIVE_EXTERNAL_REPLICATION"
    else:
        decision = "NONREPLICATION"
    improved_hospitals = int((hospital.delta_brier_B_minus_A < 0).sum())
    summary = {
        "decision": decision,
        "validation_wording": "multicenter external validation under transported cardiac-output measurement",
        "observed_landmarks": int(len(t)), "patients": int(t.uniquepid.nunique()), "hospitals": int(t.hospitalid.nunique()),
        "event_landmarks": int(y.sum()), "event_patients": int(t.loc[t.event_6h, "uniquepid"].nunique()),
        "primary_comparison": points,
        "primary_delta_brier_ci95": [float(brier_ci.ci95_low), float(brier_ci.ci95_high)],
        "valid_bootstrap_replicates": int(len(boot)),
        "hospitals_with_ge5_event_patients": int(len(hospital)),
        "hospitals_with_brier_improvement": improved_hospitals,
        "fraction_hospitals_with_brier_improvement": float(improved_hospitals / len(hospital)) if len(hospital) else None,
        "model_retrained": False, "model_recalibrated": False, "threshold_tuned": False,
    }
    (RESULTS / "stage_e2_decision.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (AUDIT / "stage_e2_audit.json").write_text(json.dumps({
        "protocol": "PROTOCOL_STAGE_E2_FROZEN_EXTERNAL_VALIDATION_LOCKED.md",
        "seed": SEED, "bootstrap_requested": N_BOOT, "bootstrap_valid": int(len(boot)),
        "bootstrap_levels": ["hospitalid", "uniquepid within hospital"],
        "prediction_function_from_locked_artifacts": True,
        **summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    abs_primary = absolute[absolute.analysis.eq("primary_ipow_floor_and_q01q99")].pivot(index="metric", columns="model", values="estimate")
    report = f"""# eICU Stage E2 冻结模型多中心外部验证

**裁决：{decision}**

正式分析包含 {len(t):,} 个可判定地标、{t.uniquepid.nunique():,} 名患者、{t.hospitalid.nunique()} 家医院；事件地标 {int(y.sum()):,} 个，事件患者 {summary['event_patients']:,} 名。

| 指标 | Model A | Model B | B−A |
|---|---:|---:|---:|
| Brier | {abs_primary.loc['brier','A']:.5f} | {abs_primary.loc['brier','B']:.5f} | {points['delta_brier_B_minus_A']:.5f} |
| AUROC | {abs_primary.loc['auroc','A']:.4f} | {abs_primary.loc['auroc','B']:.4f} | {points['delta_auroc_B_minus_A']:.4f} |
| Log loss | {abs_primary.loc['log_loss','A']:.5f} | {abs_primary.loc['log_loss','B']:.5f} | {points['delta_log_loss_B_minus_A']:.5f} |
| Average precision | {abs_primary.loc['average_precision','A']:.4f} | {abs_primary.loc['average_precision','B']:.4f} | {points['delta_average_precision_B_minus_A']:.4f} |

- 主要 ΔBrier 95% CI：{brier_ci.ci95_low:.5f} 至 {brier_ci.ci95_high:.5f}
- 有效医院→患者两层 bootstrap：{len(boot)}/{N_BOOT}
- ≥5 名事件患者医院中，Brier 改善：{improved_hospitals}/{len(hospital)}（{improved_hospitals/len(hospital):.1%}）
- 未重训练、未重新校准、未调阈值。

该结果应表述为 transported cardiac-output measurement 下的外部验证；eICU 的 generic CO 不能被写成与 MIMIC 连续热稀释 CCO 完全等同。
"""
    (ROOT / "REPORT_STAGE_E2_ZH.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
