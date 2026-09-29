"""Regenerate the author-approved locked figures from private frozen outputs.

This reporting-only entry point never trains, recalibrates or bootstraps.
Use a NEW private output directory. No source data are bundled in this repo.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RENDERERS = ROOT / "10_figures_tables/locked"
NAMES = ["Graphical_Abstract", "Figure_1", "Figure_2", "Figure_3", "Figure_4",
         "Figure_S1", "Figure_S2", "Figure_S3", "Table_1", "Table_2"]


def private_path(path, label):
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents or path in ROOT.parents:
        raise ValueError(f"{label} must be outside (and not an ancestor of) the code repository")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=os.environ.get("CPO_WORK_DIR"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-lock", action="store_true",
                        help="Require exact PNG hashes in the recorded rendering environment")
    parser.add_argument("--plan", action="store_true", help="List outputs; no data access or writes")
    args = parser.parse_args()
    if args.plan:
        print("Locked set: GA V5; main Figures 1–4, S1–S3, Tables 1–2 reference style V1.")
        for name in NAMES:
            print(f"{name}: PNG (600 dpi), PDF (vector), SVG")
        print("Requires private final-R1 analysis outputs and legacy source-data preparation.")
        return
    if args.work_dir is None:
        parser.error("Supply --work-dir or set CPO_WORK_DIR to a private completed analysis tree")
    work = private_path(args.work_dir, "Private work")
    if not work.is_dir():
        raise FileNotFoundError("Private analysis work directory does not exist")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = private_path(args.output_dir or work / f"locked_figures_{stamp}", "Figure output")
    if output.exists():
        raise FileExistsError("Output exists; choose a NEW directory to preserve the locked originals")
    if output == work or output in work.parents:
        raise ValueError("Output must not replace the work tree or any ancestor")
    required = [
        "CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908/results/temporal_predictions.csv.gz",
        "CPO_EICU_EXTERNAL_DISCHARGE_BOUNDARY_R1_V1_0/results/stage_e2_external_predictions.csv.gz",
        "submission_crosscheck/Table_1_recomputed.csv",
        "submission_crosscheck/Table_2_recomputed.csv",
        "CPO_SUBMISSION_SUPPLEMENT_20260917/EHJACC_FIGURES/source_data/figure4_selected_landmarks.csv",
    ]
    missing = [name for name in required if not (work / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing private reporting inputs; run run_workflow.py --report first: "
                                + ", ".join(missing))
    # A private work tree may contain older/edited draw functions. Refuse these
    # rather than silently changing the supposedly locked submission figures.
    mapping = json.loads((ROOT / "documentation/runtime_map.json").read_text())
    for item in mapping:
        if not item["source"].startswith("10_figures_tables/"):
            continue
        staged = work / item["target"]
        approved = ROOT / item["source"]
        if not staged.is_file() or hashlib.sha256(staged.read_bytes()).digest() != hashlib.sha256(approved.read_bytes()).digest():
            raise RuntimeError(f"Private reporting code differs from bundled code: {item['target']}; "
                               "use a private tree assembled by this candidate's --prepare")
    for directory in (output, output / "png", output / "vector", output / "previews"):
        directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CPO_WORK_DIR"] = str(work)
    env["CPO_REPORT_DIR"] = str(output)
    env["MPLCONFIGDIR"] = str(output / ".mplconfig")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Separate interpreters prevent GA palette changes leaking into the main plots.
    for script in ("graphical_abstract.py", "locked_style.py"):
        subprocess.run([sys.executable, str(RENDERERS / script)], env=env, cwd=work, check=True)
    from PIL import Image
    lock = json.loads((ROOT / "documentation/FIGURE_LOCK.json").read_text())
    checks = []
    for name in NAMES:
        for folder, extension in (("png", "png"), ("vector", "pdf"), ("vector", "svg")):
            file = output / folder / f"{name}.{extension}"
            if not file.is_file() or file.stat().st_size == 0:
                raise RuntimeError(f"Missing output: {name}.{extension}")
        file = output / "png" / f"{name}.png"
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        with Image.open(file) as im:
            size = list(im.size)
            dpi = list(im.info.get("dpi", []))
            if len(dpi) != 2 or any(abs(value - 600) > .02 for value in dpi):
                raise ValueError(f"Incorrect DPI: {name}")
            im.verify()
        expected = lock["artifacts"][name]
        exact = digest == expected["files"]["png"]["sha256"]
        checks.append({"name": name, "pixels": size, "dpi": dpi, "sha256": digest,
                       "matches_locked_PNG": exact})
    report = {"figures": checks, "model_refitting": False, "bootstrap_repeated": False,
              "exact_png_matches": sum(row["matches_locked_PNG"] for row in checks),
              "statistical_analysis_rerun_this_command": False,
              "note": "Exact pixels require recorded software and fonts; PDFs have timestamp metadata."}
    (output / "LOCK_REPRODUCTION_QA.json").write_text(json.dumps(report, indent=2))
    print(f"Generated {len(checks)} locked figures/tables in {output}")
    print(f"Exact PNG matches: {report['exact_png_matches']}/{len(checks)}")
    if args.check_lock and report["exact_png_matches"] != len(checks):
        raise RuntimeError("PNG identity check failed. Inspect fonts/environment and private inputs; "
                           "do not replace the lock manifest with new hashes")


if __name__ == "__main__":
    main()
