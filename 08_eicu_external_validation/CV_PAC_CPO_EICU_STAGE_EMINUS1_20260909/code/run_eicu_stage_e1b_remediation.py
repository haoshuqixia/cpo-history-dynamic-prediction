#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"


def main() -> None:
    cols = ["sid", "uniquepid", "hospitalid", "landmark_h", "outcome_observed_6h", "p_outcome_observed_6h"]
    d = pd.read_csv(RESULTS / "stage_e1b_observation_weights.csv.gz", usecols=cols)
    d["outcome_observed_6h"] = d.outcome_observed_6h.astype(bool)
    d["p_outcome_observed_6h_original"] = d.p_outcome_observed_6h
    d["p_outcome_observed_6h"] = d.p_outcome_observed_6h.clip(lower=.05)
    numerator = float(d.outcome_observed_6h.mean())
    observed = d.outcome_observed_6h
    d["ipow_stabilized"] = np.where(observed, numerator / d.p_outcome_observed_6h, np.nan)
    lo, hi = d.loc[observed, "ipow_stabilized"].quantile([.01, .99])
    d["ipow_stabilized_truncated"] = d.ipow_stabilized.clip(lo, hi)
    w = d.loc[observed, "ipow_stabilized_truncated"].to_numpy(float)
    ess = float(w.sum() ** 2 / np.square(w).sum())
    ess_fraction = ess / len(w)
    passed = int((d.p_outcome_observed_6h < .05).sum()) == 0 and ess_fraction >= .50
    d.to_csv(RESULTS / "stage_e1b_observation_weights_final.csv.gz", index=False, compression="gzip")
    decision = {
        "decision": "PASS_E1B_R_UNLOCK_FROZEN_EXTERNAL_VALIDATION" if passed else "HOLD_EXTERNAL_VALIDATION",
        "rows": int(len(d)), "observed_rows": int(observed.sum()),
        "probabilities_floored_at_0_05": int((d.p_outcome_observed_6h_original < .05).sum()),
        "rows_below_0_05_after_floor": int((d.p_outcome_observed_6h < .05).sum()),
        "truncation_q01": float(lo), "truncation_q99": float(hi),
        "truncated_weight_ess": ess, "truncated_weight_ess_fraction": ess_fraction,
        "clinical_event_columns_read": False, "clinical_predictions_read": False,
    }
    (RESULTS / "stage_e1b_remediation_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    (AUDIT / "stage_e1b_remediation_audit.json").write_text(json.dumps({
        **decision,
        "protocol": "PROTOCOL_STAGE_E1B_POSITIVITY_REMEDIATION_LOCKED.md",
        "primary_weight_file": "results/stage_e1b_observation_weights_final.csv.gz",
        "sensitivity_weight_file": "results/stage_e1b_observation_weights.csv.gz",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    report = f"""# eICU Stage E1B-R positivity 修复

**裁决：{decision['decision']}**

- 概率下限截断地标：{decision['probabilities_floored_at_0_05']} / {len(d):,}
- 修复后 p<0.05：{decision['rows_below_0_05_after_floor']}
- observed 截尾权重 q01/q99：{lo:.3f}/{hi:.3f}
- 截尾权重 ESS：{ess:.1f}（{ess_fraction:.1%}）

修复只使用观测指示与观测概率，没有读取临床事件或 A/B 预测。
"""
    (ROOT / "REPORT_STAGE_E1B_REMEDIATION_ZH.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
