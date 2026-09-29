#!/usr/bin/env python3
"""Format locked model, observation-weight, and alert-utility results for the paper."""

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
TABLES = ROOT / "tables"
S1A = BASE / "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908" / "results"
S1B = BASE / "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908" / "results"
S1D = BASE / "CV_PAC_CPO_STAGE1D_ALERT_UTILITY_20260908" / "results"


def markdown(frame: pd.DataFrame) -> str:
    vals = frame.fillna("").astype(str)
    return "\n".join([
        "| " + " | ".join(vals.columns) + " |",
        "| " + " | ".join(["---"] * len(vals.columns)) + " |",
        *("| " + " | ".join(x.replace("|", "\\|") for x in row) + " |" for row in vals.to_numpy()),
    ]) + "\n"


def ci(r, digits=4):
    return f"{r.estimate:.{digits}f} ({r.ci_low:.{digits}f} to {r.ci_high:.{digits}f})"


def main() -> None:
    m = pd.read_csv(S1B / "temporal_metrics.csv")
    labels = {"brier": "Brier score", "auroc": "AUROC", "average_precision": "Average precision",
              "log_loss": "Log loss", "calibration_intercept": "Calibration intercept", "calibration_slope": "Calibration slope"}
    rows = []
    for metric in labels:
        one = {"measure": labels[metric]}
        for model in ["A", "B", "C"]:
            r = m[(m.model == model) & (m.metric == metric)].iloc[0]
            one[f"Model {model}, estimate (95% CI)"] = ci(r)
        rows.append(one)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "table2_temporal_model_performance.csv", index=False)
    (TABLES / "table2_temporal_model_performance.md").write_text(markdown(out), encoding="utf-8")

    d = pd.read_csv(S1B / "paired_temporal_deltas.csv")
    keep = d[d.metric.isin(["brier", "auroc", "average_precision", "log_loss"])].copy()
    keep["measure"] = keep.metric.map(labels)
    keep["difference (95% CI)"] = keep.apply(ci, axis=1)
    keep = keep[["comparison", "measure", "difference (95% CI)"]]
    keep.to_csv(TABLES / "table3_paired_temporal_model_differences.csv", index=False)
    (TABLES / "table3_paired_temporal_model_differences.md").write_text(markdown(keep), encoding="utf-8")

    a = pd.read_csv(S1D / "alert_utility.csv")
    alert = pd.DataFrame({
        "model": a.model,
        "threshold": a.threshold.map(lambda x: f"{x:g}"),
        "event-pair sensitivity": a.event_pair_sensitivity.map(lambda x: f"{x*100:.1f}%"),
        "event-patient sensitivity": a.event_patient_sensitivity.map(lambda x: f"{x*100:.1f}%"),
        "event-free landmark alert fraction": a.event_free_landmark_alert_fraction.map(lambda x: f"{x*100:.1f}%"),
        "alert episodes": a.alert_episodes.astype(int),
        "alert episodes per 100 eligible patient-hours": a.alert_episodes_per_100_landmarks.map(lambda x: f"{x:.1f}"),
        "lead to first low, median [IQR], h": a.apply(lambda r: f"{r.median_lead_to_first_low_h:.2f} [{r.q1_lead_to_first_low_h:.2f}, {r.q3_lead_to_first_low_h:.2f}]", axis=1),
        "lead to confirmation, median, h": a.median_lead_to_confirmation_h.map(lambda x: f"{x:.2f}"),
        "no-event patients with any alert": a.apply(lambda r: f"{int(r.no_event_patients_with_any_alert)}/{int(r.no_event_patients)} ({r.no_event_patient_any_alert_fraction*100:.1f}%)", axis=1),
    })
    alert.to_csv(TABLES / "tableS2_alert_utility.csv", index=False)
    (TABLES / "tableS2_alert_utility.md").write_text(markdown(alert), encoding="utf-8")

    p = pd.read_csv(S1A / "observation_probability_summary.csv")
    w = pd.read_csv(S1A / "observation_weight_summary.csv")
    summary = []
    for split in ["development_oof", "temporal_locked"]:
        pr = p[p.split.eq(split)].iloc[0]
        wr = w[w.split.eq(split) & w.kind.eq("stabilized_truncated")].iloc[0]
        summary.append({
            "period": split,
            "eligible landmarks": int(pr.n),
            "outcome observed": f"{pr.observed_fraction*100:.1f}%",
            "minimum predicted observation probability": f"{pr.p_q00:.3f}",
            "truncated weight median [1st, 99th percentile]": f"{wr.w_q50:.3f} [{wr.w_q01:.3f}, {wr.w_q99:.3f}]",
            "effective sample size": f"{wr.ess:.0f}/{int(wr.n_observed)} ({wr.ess_fraction*100:.1f}%)",
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(TABLES / "tableS4_observation_weight_audit.csv", index=False)
    (TABLES / "tableS4_observation_weight_audit.md").write_text(markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
