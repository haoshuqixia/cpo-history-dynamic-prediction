"""Read-only local preflight for the MIT-authorized public code metadata.

Checks are scoped automated preflight, not independent legal clearance.
Does not read patient files, write data, run models, or publish anything.
Checks local metadata consistency, not live GitHub access or remote content.
"""
from pathlib import Path
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent


def main():
    files = sorted(path for path in ROOT.rglob("*") if path.is_file()
                   and not any(part in {"__pycache__", ".git"} for part in path.parts))
    allowed = {".py", ".md", ".txt", ".json", ".csv", ".cff"}
    for path in files:
        if path.is_symlink():
            raise ValueError("Symlinks are not permitted")
        if path.suffix not in allowed and path.name not in {".gitignore", "LICENSE"}:
            raise ValueError(f"Unreviewed file type: {path.relative_to(ROOT)}")
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            compile(text, str(path.relative_to(ROOT)), "exec")
        if path.suffix == ".csv" and path.name not in {
            "variable_dictionary.csv", "CPO_CODE_RELEASE_MANIFEST_V1_0.csv",
            "CPO_CODE_RELEASE_MANIFEST_V1_2.csv", "CPO_MANUSCRIPT_CODE_CROSSWALK_V1_0.csv"
        }:
            raise ValueError("No data CSVs are permitted in this code candidate")
        if re.search(r"/(Users|home)/[^\s'\"]+", text):
            raise ValueError(f"Personal absolute path: {path.relative_to(ROOT)}")
        if re.search(r"(?:gh[pousr]_|github_pat_|sk-)[A-Za-z0-9_-]{20,}", text):
            raise ValueError("Possible credential literal")
        if "-----BEGIN " + "PRIVATE KEY-----" in text:
            raise ValueError("Possible private key")
    analysis = json.loads((ROOT / "documentation/ANALYSIS_CODE_LOCK.json").read_text())
    for name, digest in analysis["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Locked statistical module changed: {name}")
    assert (ROOT / "LICENSE").is_file() and not (ROOT / "LICENSE_PENDING.md").exists()
    licence = (ROOT / "LICENSE").read_text()
    assert licence.startswith("MIT License") and "Copyright (c) 2026" in licence
    assert all(name in licence for name in ["Zhihao Lin", "Tangjiang Wan", "Wenyue Lv", "Shitong Shen", "Luyang Wu", "Yafei Li"])
    sums = (ROOT / "SHA256SUMS.txt").read_text()
    assert sums == (ROOT / "CPO_CODE_RELEASE_SHA256_V1_2.txt").read_text()
    recorded = set()
    for line in sums.splitlines():
        digest, name = line.split("  ", 1)
        assert not name.startswith("/") and ".." not in Path(name).parts
        assert name not in recorded
        recorded.add(name)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    expected = {str(path.relative_to(ROOT)) for path in files
                if path.name not in {"SHA256SUMS.txt", "CPO_CODE_RELEASE_SHA256_V1_2.txt"}}
    assert recorded == expected, "Checksum inventory must cover all non-checksum payload files"
    status = json.loads((ROOT / "documentation/RELEASE_STATUS.json").read_text())
    assert status["licence"] == "MIT" and status["author_approved"] is True
    assert status["publicly_published"] is True
    repository_url = "https://github.com/haoshuqixia/cpo-history-dynamic-prediction"
    assert status["github_url"] == repository_url and status["public_access_verified"] is True
    assert status["code_release"] == "SEALED_PUBLIC_CODE_RELEASED"
    assert status["formal_release_published"] is True and status["release_tag"] == "v1.0.0"
    assert status["release_url"] == f"{repository_url}/releases/tag/v1.0.0"
    assert status["existing_release_tag_preserved"] is True
    assert status["main_branch_metadata_checksum_status"] == "REGENERATED_LOCAL_CANDIDATE_PENDING_UPLOAD_VERIFICATION"
    assert status["zenodo_doi"] is None
    assert re.fullmatch(r"[0-9a-f]{40}", status["public_access_verified_commit"])
    provenance = json.loads((ROOT / "documentation/environment_provenance.json").read_text())
    assert provenance["public_repository"] == repository_url
    assert provenance["licence"] == status["licence"]
    assert provenance["status"] == status["code_release"]
    assert provenance["manuscript_title"] == status["current_manuscript_title"]
    citation = (ROOT / "CITATION.cff").read_text()
    assert f'repository-code: "{repository_url}"' in citation
    assert "license: MIT" in citation and 'version: "1.0.0"' in citation
    assert f'title: "{status["current_manuscript_title"]}"' in citation
    assert f'url: "{status["release_url"]}"' in citation
    readme = (ROOT / "README.md").read_text()
    assert repository_url in readme and "NOT YET UPLOADED" not in readme
    assert status["core_statistical_modules_unchanged"] == len(analysis["files"])
    model = ROOT / "06_model_development/CV_PAC_CPO_STAGE1B_ABC_TEMPORAL_20260908/code/run_stage1b.py"
    tree = ast.parse(model.read_text())
    blocks = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"MODEL_A", "MODEL_B", "MODEL_C"}:
                # Nested concatenations are validated by the original QA; hashes protect them here.
                blocks[name] = ast.dump(node, include_attributes=False)
    assert len(blocks) == 3
    print(f"PASS: {len(files)} reviewed text/code files; {len(analysis['files'])} unchanged statistical modules")
    print("PASS: syntax, recorded checksums and automated path/credential/data-file preflight")
    print("PASS: author-confirmed MIT licence and complete payload checksum inventory")
    print("PASS: consistent public URL, MIT, v1.0.0 release metadata and IJMI manuscript title")
    print(f"{status['code_release']}; LOCAL PREFLIGHT ONLY / LIVE METADATA SYNC NOT CHECKED")


if __name__ == "__main__":
    main()
