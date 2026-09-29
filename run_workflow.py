"""Prepare and explicitly execute the frozen workflow in a private work tree."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import hashlib
import subprocess
import datetime

ROOT = Path(__file__).resolve().parent


def candidate_plan():
    steps = []
    extraction = "STAGE_MINUS1A_CV_OBSERVABILITY_V1/code"
    for name, arguments in [("common.py", []), ("cohort.py", []),
                            ("extract.py", ["chartevents"]), ("extract.py", ["outputevents"]),
                            ("extract.py", ["labevents"]), ("aggregate.py", []), ("pairing.py", [])]:
        steps.append((f"{extraction}/{name}", arguments))
    modules = [
        ("CV_PAC_CPO_LANDMARK_AUDIT_20260908", "run_landmark_audit.py"),
        ("CV_PAC_CPO_STAGE0_MECHANISM_20260908", "run_stage0_mechanism_audit.py"),
        ("CV_PAC_CPO_STAGE0_MECHANISM_20260908", "build_landmark_dataset_v1.py"),
        ("CV_PAC_CPO_STAGE0_MECHANISM_20260908", "summarize_stage0.py"),
        ("CV_PAC_CPO_STAGE1A_FEATURE_AND_CENSORING_20260908", "run_stage1a.py"),
        ("CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908", "run_stage1b.py"),
        ("CV_PAC_CPO_STAGE1C_EARLY_WARNING_AUDIT_20260908", "run_stage1c.py"),
        ("CV_PAC_CPO_STAGE1D_ALERT_UTILITY_20260908", "run_stage1d.py"),
        ("CV_PAC_CPO_PAPER_PACKAGE_20260908", "export_locked_models.py"),
        ("CV_PAC_CPO_REVIEWER_DEFENSE_20260908", "run_information_clock_audit.py"),
        ("CV_PAC_CPO_REVIEWER_DEFENSE_20260908", "run_storetime_ab_sensitivity.py"),
        ("CV_PAC_CPO_REVIEWER_DEFENSE_20260908", "run_stage_r2_robustness.py"),
        ("CV_PAC_CPO_REVIEWER_DEFENSE_20260908", "run_stage_r3_sensitivity_counts.py"),
    ]
    modules += [("CV_PAC_CPO_EICU_STAGE_EMINUS1_20260909", name) for name in
                ["run_eicu_stage_eminus1.py", "run_eicu_stage_e0.py", "run_eicu_stage_e1a_medication_audit.py",
                 "run_eicu_stage_e1b.py", "run_eicu_stage_e1b_remediation.py", "run_eicu_stage_e2.py"]]
    steps += [(f"{stage}/code/{name}", []) for stage, name in modules]
    # Baseline E2 is needed by the unchanged R1 old-versus-new audit.
    # Only the subsequent R1 outputs are the final scientific endpoint.
    steps.append(("CPO_EICU_EXTERNAL_DISCHARGE_BOUNDARY_R1_V1_0/run_eicu_r1.py", []))
    return steps


def prepare():
    required = ("CPO_WORK_DIR", "CPO_MIMIC_DIR", "CPO_EICU_ZIP")
    if not all(os.environ.get(name) for name in required):
        raise RuntimeError("Set CPO_WORK_DIR, CPO_MIMIC_DIR and CPO_EICU_ZIP first")
    work = Path(os.environ["CPO_WORK_DIR"]).expanduser().resolve()
    if work == ROOT or ROOT in work.parents:
        raise RuntimeError("Private work must be outside the public-repository tree")
    if work.exists():
        raise FileExistsError("Use a new empty private-work path; refusing to overwrite")
    if not Path(os.environ["CPO_MIMIC_DIR"]).is_dir() or not Path(os.environ["CPO_EICU_ZIP"]).is_file():
        raise FileNotFoundError("Configured source data are missing")
    work.mkdir(parents=True)
    mapping = json.loads((ROOT / "documentation/runtime_map.json").read_text())
    for item in mapping:
        target = work / item["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / item["source"], target)
        stage = target.parent.parent if target.parent.name == "code" else target.parent
        for folder in ("audit", "evidence", "results", "work", "models", "tables"):
            (stage / folder).mkdir(parents=True, exist_ok=True)
    (work / "execution_logs").mkdir()
    # The R1 supplement references rerun subdirectories; make a local-only alias.
    (work / "CPO_STATISTICS_RERUN_20260915").symlink_to(work, target_is_directory=True)
    print(f"Prepared private code tree: {work}")
    print("No data copied and no analysis executed. Use --report after completed analysis to generate the locked figures.")


def fingerprint(work, script, arguments):
    digest = hashlib.sha256()
    for path in sorted(work.rglob("*.py")):
        # pathlib does not traverse directory symlinks: the local legacy alias
        # is therefore not scanned recursively.
        digest.update(str(path.relative_to(work)).encode())
        digest.update(path.read_bytes())
    digest.update(json.dumps([script, arguments, sys.executable, sys.version]).encode())
    return digest.hexdigest()


def execute(resume=False):
    work = Path(os.environ["CPO_WORK_DIR"]).expanduser().resolve()
    if work == ROOT or ROOT in work.parents or not (work / "execution_logs").is_dir():
        raise RuntimeError("Use --prepare first with a private work directory")
    checkpoint = work / "execution_checkpoint.json"
    if checkpoint.exists() and not resume:
        raise RuntimeError("Existing execution checkpoint: use --resume; refusing silent overwrite")
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {"stages": {}}
    environment = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(variable, "1")
    environment.setdefault("MPLCONFIGDIR", str(work / "matplotlib_cache"))
    for index, (script, arguments) in enumerate(candidate_plan(), 1):
        key = f"{index:02d}"
        code_hash = fingerprint(work, script, arguments)
        previous = state["stages"].get(key, {})
        if resume and previous.get("status") == "PASS" and previous.get("fingerprint") == code_hash:
            print(f"[{index:02d}] Previously completed, unchanged: {script}", flush=True)
            continue
        log_path = work / "execution_logs" / f"{key}_{Path(script).stem}.log"
        if log_path.exists():
            suffix = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            log_path = log_path.with_name(f"{log_path.stem}_{suffix}.log")
        entry = {"script": script, "arguments": arguments, "fingerprint": code_hash,
                 "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "log": str(log_path.relative_to(work)), "status": "RUNNING"}
        state["stages"][key] = entry
        checkpoint.write_text(json.dumps(state, indent=2))
        print(f"[{index:02d}/{len(candidate_plan())}] Running {script}", flush=True)
        with log_path.open("w") as log:
            result = subprocess.run([sys.executable, str(work / script), *arguments],
                                    cwd=work, env=environment, stdout=log, stderr=subprocess.STDOUT)
        entry.update(status="PASS" if result.returncode == 0 else "FAIL", returncode=result.returncode,
                     finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        checkpoint.write_text(json.dumps(state, indent=2))
        print(f"[{index:02d}] {entry['status']}; private log: {log_path.name}", flush=True)
        if result.returncode:
            raise RuntimeError(f"Stage {index} failed. Inspect its private log; no release PASS is asserted.")
    print("Analysis execution completed. Numerical, reporting and privacy QA are still required.", flush=True)


def report(manuscript=None):
    work = Path(os.environ["CPO_WORK_DIR"]).expanduser().resolve()
    checkpoint = json.loads((work / "execution_checkpoint.json").read_text())
    if len(checkpoint["stages"]) != len(candidate_plan()) or any(
        entry["status"] != "PASS" for entry in checkpoint["stages"].values()
    ):
        raise RuntimeError("All analysis stages must succeed before reporting")
    figures = work / "CPO_SUBMISSION_SUPPLEMENT_20260917/EHJACC_FIGURES"
    external = figures / "external_validation"
    external.mkdir(exist_ok=True)
    for name in ("stage_e2_absolute_metrics.csv", "stage_e2_primary_comparison_ci.csv",
                 "stage_e2_hospital_heterogeneity.csv"):
        shutil.copyfile(work / "CPO_EICU_EXTERNAL_DISCHARGE_BOUNDARY_R1_V1_0/results" / name,
                        external / name)
    commands = [
        [work / "CV_PAC_CPO_PAPER_PACKAGE_20260908/code/build_predictor_dictionary.py"],
        [figures / "scripts/generate_ehjacc_figures.py", "--only", "all"],
        [work / "CPO_SUBMISSION_SUPPLEMENT_20260917/build_submission_supplement_r1.py", "--mode", "figures"],
        [ROOT / "10_figures_tables/reproduce_main_table_cells.py"],
    ]
    if manuscript:
        commands[-1] += ["--manuscript", str(manuscript)]
    for command in commands:
        print(f"Reporting: {Path(command[0]).name}", flush=True)
        subprocess.run([sys.executable, *map(str, command)], cwd=work, check=True)
    subprocess.run([sys.executable, str(ROOT / "regenerate_final_figures.py"),
                    "--work-dir", str(work)], cwd=work, check=True)
    print("Locked reporting complete. Result equivalence and visual QA are separate checks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--manuscript", type=Path, help="Read-only Word table cross-check")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.execute:
        execute(args.resume)
    if args.report:
        report(args.manuscript)
    if args.execute or args.report:
        return
    print("LOCAL DRAFT: execution requires explicit --execute.")
    for index, (script, arguments) in enumerate(candidate_plan(), 1):
        print(f"{index:02d}. python {script}" + (" " + " ".join(arguments) if arguments else ""))
    print("Locked V1.1 reporting is available with --report. See documentation/REPRODUCIBILITY.md.")


if __name__ == "__main__":
    main()
