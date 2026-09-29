#!/usr/bin/env python3
"""Refit and serialize the already locked final models without changing the analysis."""

from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
STAGE1A = BASE / "CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908"
STAGE1B = BASE / "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908"
MODELS = ROOT / "models"
sys.path.insert(0, str(STAGE1B / "code"))
from run_stage1b import BLOCKS, CATEGORICAL, DEV, TEMP, apply_calibration, make_model  # noqa: E402


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(STAGE1A / "results/LANDMARK_FEATURES_V1.csv.gz", dtype={"stay_id": str})
    w = pd.read_csv(STAGE1A / "results/observation_weights_v1.csv.gz", dtype={"stay_id": str})
    keys = ["subject_id", "stay_id", "landmark_h", "interval_conservative_split"]
    d = f.merge(w[keys + ["ipow_stabilized_truncated"]], on=keys, how="left", validate="one_to_one")
    common = (
        d.primary_eligible_at_landmark.astype(bool)
        & d.primary_outcome_observed_6h.astype(bool)
        & d.model_bc_common_risk_set.astype(bool)
    )
    dev = common & d.interval_conservative_split.eq(DEV)
    temp = common & d.interval_conservative_split.eq(TEMP)
    locked = pd.read_csv(STAGE1B / "results/temporal_predictions.csv.gz", dtype={"stay_id": str})
    maps = pd.read_csv(STAGE1B / "results/development_calibration_maps.csv").set_index("model")
    manifest = {
        "status": "locked_models_refit_for_serialization_only",
        "clinical_model_hyperparameter_tuning": False,
        "temporal_recalibration": False,
        "development_landmarks": int(dev.sum()),
        "temporal_landmarks": int(temp.sum()),
        "models": {},
    }

    ydev = d.loc[dev, "primary_outcome_6h"].astype(int)
    sw = d.loc[dev, "ipow_stabilized_truncated"].to_numpy(float)
    for name, numeric in BLOCKS.items():
        cols = numeric + CATEGORICAL
        model = make_model(numeric)
        model.fit(d.loc[dev, cols], ydev, model__sample_weight=sw)
        cal = (
            float(maps.loc[name, "development_oof_intercept"]),
            float(maps.loc[name, "development_oof_slope"]),
        )
        check = apply_calibration(model.predict_proba(d.loc[temp, cols])[:, 1], cal)
        maximum_difference = float(np.max(np.abs(check - locked[f"prediction_{name}"].to_numpy(float))))
        if maximum_difference > 1e-12:
            raise RuntimeError(f"Model {name} does not reproduce locked predictions: {maximum_difference}")
        artifact = MODELS / f"model_{name}_locked.joblib"
        joblib.dump(
            {
                "model_name": name,
                "pipeline": model,
                "numeric_predictors": numeric,
                "categorical_predictors": CATEGORICAL,
                "development_calibration_intercept": cal[0],
                "development_calibration_slope": cal[1],
                "prediction_function": "expit(intercept + slope * logit(pipeline.predict_proba(X)[:, 1]))",
            },
            artifact,
            compress=3,
        )
        manifest["models"][name] = {
            "artifact": artifact.name,
            "sha256": __import__("hashlib").sha256(artifact.read_bytes()).hexdigest(),
            "maximum_absolute_difference_from_locked_temporal_predictions": maximum_difference,
            "numeric_predictors": numeric,
            "categorical_predictors": CATEGORICAL,
            "calibration_intercept": cal[0],
            "calibration_slope": cal[1],
        }

    packages = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "matplotlib"]
    manifest["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in packages},
    }
    (MODELS / "MODEL_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

