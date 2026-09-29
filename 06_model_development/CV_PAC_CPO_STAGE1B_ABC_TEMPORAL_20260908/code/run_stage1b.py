#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
STAGE1A = BASE / "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908"
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"
DEV = "definite_development_to_2019"
TEMP = "definite_temporal_2020_plus"
SEED = 20260908
THRESHOLDS = np.array([0.02, 0.03, 0.05, 0.075, 0.10, 0.15])

DRUG_COLS = [
    "drug_norepinephrine_active_at_landmark", "drug_epinephrine_active_at_landmark",
    "drug_vasopressin_active_at_landmark", "drug_phenylephrine_active_at_landmark",
    "drug_dopamine_active_at_landmark", "drug_dobutamine_active_at_landmark",
    "drug_milrinone_active_at_landmark",
]
SHARED_NUMERIC = [
    "age", "landmark_h", "landmark_h_sq", "n_cpo_history_4h", "n_mpap_history_4h",
    "n_svo2_history_4h", "cpo_recency_h", "mpap_recency_h", "svo2_recency_h",
    "iabp_active_at_landmark", *DRUG_COLS,
]
CATEGORICAL = ["gender", "first_careunit"]
MODEL_A = SHARED_NUMERIC + ["cpo_current"]
MODEL_B = MODEL_A + [
    "cpo_mean_4h", "cpo_min_4h", "cpo_max_4h", "cpo_sd_4h",
    "cpo_slope_4h", "cpo_delta_4h", "cpo_span_4h",
]
MODEL_C = MODEL_B + [
    "mpap_current", "mpap_mean_4h", "mpap_min_4h", "mpap_max_4h",
    "mpap_sd_4h", "mpap_slope_4h", "mpap_delta_4h", "mpap_span_4h",
    "svo2_current", "svo2_mean_4h", "svo2_min_4h", "svo2_max_4h",
    "svo2_sd_4h", "svo2_slope_4h", "svo2_delta_4h", "svo2_span_4h",
]
BLOCKS = {"A": MODEL_A, "B": MODEL_B, "C": MODEL_C}


def make_model(numeric):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15, max_depth=3,
        min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
        random_state=SEED,
    )
    return Pipeline([("pre", pre), ("model", model)])


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(x):
    x = np.clip(np.asarray(x, float), -35, 35)
    return 1 / (1 + np.exp(-x))


def weighted_calibration(y, p, w, start=(0.0, 1.0)):
    x = np.column_stack([np.ones(len(p)), logit(p)])
    beta = np.asarray(start, float)
    for _ in range(50):
        mu = expit(x @ beta)
        grad = x.T @ (w * (y - mu))
        h = x.T @ ((w * mu * (1 - mu))[:, None] * x)
        step = np.linalg.solve(h + np.eye(2) * 1e-8, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(beta[0]), float(beta[1])


def apply_calibration(p, params):
    return expit(params[0] + params[1] * logit(p))


def metric_set(y, p, w):
    intercept, slope = weighted_calibration(y, p, w)
    return {
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
        "auroc": float(roc_auc_score(y, p, sample_weight=w)),
        "average_precision": float(average_precision_score(y, p, sample_weight=w)),
        "log_loss": float(log_loss(y, p, sample_weight=w)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def net_benefit(y, p, w, threshold):
    positive = p >= threshold
    total = w.sum()
    tp = w[(positive) & (y == 1)].sum() / total
    fp = w[(positive) & (y == 0)].sum() / total
    return float(tp - fp * threshold / (1 - threshold))


def percentile_interval(values):
    q = np.quantile(np.asarray(values, float), [0.025, 0.975])
    return float(q[0]), float(q[1])


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(STAGE1A / "results/LANDMARK_FEATURES_V1.csv.gz", dtype={"stay_id": str})
    w = pd.read_csv(STAGE1A / "results/observation_weights_v1.csv.gz", dtype={"stay_id": str})
    keys = ["subject_id", "stay_id", "landmark_h", "interval_conservative_split"]
    d = f.merge(w[keys + ["ipow_stabilized_truncated"]], on=keys, how="left", validate="one_to_one")
    common = d["primary_eligible_at_landmark"].astype(bool) & d["primary_outcome_observed_6h"].astype(bool) & d["model_bc_common_risk_set"].astype(bool)
    dev = common & d["interval_conservative_split"].eq(DEV)
    temp = common & d["interval_conservative_split"].eq(TEMP)
    assert int(dev.sum()) == 21619 and int(temp.sum()) == 3416
    y = d["primary_outcome_6h"].astype("Int64")
    predictions = {}
    calibration_maps = []
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, numeric in BLOCKS.items():
        cols = numeric + CATEGORICAL
        xdev, ydev = d.loc[dev, cols], y.loc[dev].astype(int)
        groups = d.loc[dev, "subject_id"]
        sw = d.loc[dev, "ipow_stabilized_truncated"].to_numpy(float)
        oof = np.full(len(xdev), np.nan)
        for train_pos, test_pos in cv.split(xdev, ydev, groups=groups):
            model = make_model(numeric)
            model.fit(xdev.iloc[train_pos], ydev.iloc[train_pos], model__sample_weight=sw[train_pos])
            oof[test_pos] = model.predict_proba(xdev.iloc[test_pos])[:, 1]
        cal = weighted_calibration(ydev.to_numpy(), oof, sw)
        calibration_maps.append({"model": name, "development_oof_intercept": cal[0], "development_oof_slope": cal[1]})
        final = make_model(numeric)
        final.fit(xdev, ydev, model__sample_weight=sw)
        raw = final.predict_proba(d.loc[temp, cols])[:, 1]
        predictions[name] = apply_calibration(raw, cal)

    t = d.loc[temp, keys + ["primary_outcome_6h", "ipow_stabilized_truncated"]].copy()
    t["primary_outcome_6h"] = t["primary_outcome_6h"].astype(int)
    for name in BLOCKS:
        t[f"prediction_{name}"] = predictions[name]
    t.to_csv(RESULTS / "temporal_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(calibration_maps).to_csv(RESULTS / "development_calibration_maps.csv", index=False)

    yt = t["primary_outcome_6h"].to_numpy(int)
    wt = t["ipow_stabilized_truncated"].to_numpy(float)
    metric_rows = []
    for name in BLOCKS:
        m = metric_set(yt, predictions[name], wt)
        metric_rows += [{"model": name, "metric": k, "estimate": v} for k, v in m.items()]

    dca_rows = []
    for threshold in THRESHOLDS:
        for name in BLOCKS:
            dca_rows.append({"model": name, "threshold": threshold, "net_benefit": net_benefit(yt, predictions[name], wt, threshold)})
        dca_rows.append({"model": "treat_all", "threshold": threshold, "net_benefit": net_benefit(yt, np.ones(len(yt)), wt, threshold)})
        dca_rows.append({"model": "treat_none", "threshold": threshold, "net_benefit": 0.0})

    patients = t["subject_id"].unique()
    patient_pos = {pid: np.flatnonzero(t["subject_id"].to_numpy() == pid) for pid in patients}
    rng = np.random.default_rng(SEED)
    boot_rows = []
    for b in range(1000):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        counts = pd.Series(sampled).value_counts()
        mult = np.zeros(len(t), float)
        for pid, count in counts.items():
            mult[patient_pos[pid]] = count
        wb = wt * mult
        if wb[yt == 1].sum() == 0 or wb[yt == 0].sum() == 0:
            continue
        one = {"replicate": b}
        for name in BLOCKS:
            m = metric_set(yt, predictions[name], wb)
            for k, v in m.items():
                one[f"{name}_{k}"] = v
            for threshold in THRESHOLDS:
                one[f"{name}_nb_{threshold:g}"] = net_benefit(yt, predictions[name], wb, threshold)
        boot_rows.append(one)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(RESULTS / "patient_cluster_bootstrap.csv.gz", index=False, compression="gzip")

    metrics = pd.DataFrame(metric_rows)
    for i, r in metrics.iterrows():
        lo, hi = percentile_interval(boot[f"{r.model}_{r.metric}"])
        metrics.loc[i, "ci_low"] = lo
        metrics.loc[i, "ci_high"] = hi
    metrics.to_csv(RESULTS / "temporal_metrics.csv", index=False)

    dca = pd.DataFrame(dca_rows)
    for i, r in dca.loc[dca["model"].isin(BLOCKS)].iterrows():
        vals = boot[f"{r.model}_nb_{r.threshold:g}"]
        lo, hi = percentile_interval(vals)
        dca.loc[i, "ci_low"] = lo
        dca.loc[i, "ci_high"] = hi
    dca.to_csv(RESULTS / "temporal_decision_curve.csv", index=False)

    delta_rows = []
    for left, right in [("B", "A"), ("C", "B")]:
        for metric in ["brier", "auroc", "average_precision", "log_loss", "calibration_intercept", "calibration_slope"]:
            estimate = float(metrics.loc[(metrics.model == left) & (metrics.metric == metric), "estimate"].iloc[0] - metrics.loc[(metrics.model == right) & (metrics.metric == metric), "estimate"].iloc[0])
            vals = boot[f"{left}_{metric}"] - boot[f"{right}_{metric}"]
            lo, hi = percentile_interval(vals)
            delta_rows.append({"comparison": f"{left}-{right}", "metric": metric, "estimate": estimate, "ci_low": lo, "ci_high": hi})
        for threshold in THRESHOLDS:
            est_left = dca.loc[(dca.model == left) & np.isclose(dca.threshold, threshold), "net_benefit"].iloc[0]
            est_right = dca.loc[(dca.model == right) & np.isclose(dca.threshold, threshold), "net_benefit"].iloc[0]
            vals = boot[f"{left}_nb_{threshold:g}"] - boot[f"{right}_nb_{threshold:g}"]
            lo, hi = percentile_interval(vals)
            delta_rows.append({"comparison": f"{left}-{right}", "metric": f"net_benefit_{threshold:g}", "estimate": float(est_left-est_right), "ci_low": lo, "ci_high": hi})
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(RESULTS / "paired_temporal_deltas.csv", index=False)

    cb = delta[(delta.comparison == "C-B") & (delta.metric == "brier")].iloc[0]
    ca = delta[(delta.comparison == "C-B") & (delta.metric == "auroc")].iloc[0]
    cp = delta[(delta.comparison == "C-B") & (delta.metric == "average_precision")].iloc[0]
    nb = delta[(delta.comparison == "C-B") & delta.metric.str.startswith("net_benefit_")]
    n_nb_positive = int((nb.estimate > 0).sum())
    b_cal = metrics[(metrics.model == "B")].set_index("metric").estimate
    c_cal = metrics[(metrics.model == "C")].set_index("metric").estimate
    calibration_not_worse = abs(c_cal["calibration_slope"] - 1) <= abs(b_cal["calibration_slope"] - 1) + .10 and abs(c_cal["calibration_intercept"]) <= abs(b_cal["calibration_intercept"]) + .10
    if cb.ci_high < 0 and calibration_not_worse and n_nb_positive >= 2:
        verdict = "STRONG_MULTIAXIS_INCREMENTAL_SUPPORT"
    elif cb.estimate < 0 and (ca.estimate >= 0 or cp.estimate >= 0 or n_nb_positive >= 1):
        verdict = "SUGGESTIVE_MULTIAXIS_INCREMENTAL_SUPPORT"
    else:
        verdict = "NO_INCREMENTAL_MULTIAXIS_SUPPORT"
    decision = {
        "decision": verdict,
        "temporal_patients": int(t.subject_id.nunique()),
        "temporal_landmarks": int(len(t)),
        "temporal_event_patients": int(t.loc[t.primary_outcome_6h.eq(1), "subject_id"].nunique()),
        "temporal_event_landmarks": int(t.primary_outcome_6h.sum()),
        "bootstrap_replicates": int(len(boot)),
        "c_minus_b_brier": {"estimate": float(cb.estimate), "ci_low": float(cb.ci_low), "ci_high": float(cb.ci_high)},
        "c_minus_b_auroc": {"estimate": float(ca.estimate), "ci_low": float(ca.ci_low), "ci_high": float(ca.ci_high)},
        "c_minus_b_average_precision": {"estimate": float(cp.estimate), "ci_low": float(cp.ci_low), "ci_high": float(cp.ci_high)},
        "c_minus_b_positive_net_benefit_thresholds": n_nb_positive,
        "calibration_not_materially_worse": bool(calibration_not_worse),
    }
    (RESULTS / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False))

    audit = {
        "same_rows_all_models": True,
        "development_landmarks": int(dev.sum()), "development_patients": int(d.loc[dev, "subject_id"].nunique()),
        "temporal_landmarks": int(temp.sum()), "temporal_patients": int(d.loc[temp, "subject_id"].nunique()),
        "future_named_predictors": {k: [c for c in v + CATEGORICAL if "future" in c or "event" in c or "outcome" in c] for k, v in BLOCKS.items()},
        "feature_blocks": {k: v + CATEGORICAL for k, v in BLOCKS.items()},
        "observation_weight": "Stage1A truncated stabilized; development q01/q99 locked",
        "estimator": "HistGradientBoostingClassifier fixed hyperparameters",
        "clinical_model_hyperparameter_tuning": False,
        "temporal_recalibration": False,
    }
    (AUDIT / "stage1b_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
