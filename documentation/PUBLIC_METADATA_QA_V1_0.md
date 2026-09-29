# Public repository metadata QA — post-upload V1.0

## Verified initial publication

- Repository: https://github.com/haoshuqixia/cpo-history-dynamic-prediction
- Verification date: 2026-09-29, unauthenticated GitHub API read-only requests.
- Public repository, default branch main.
- Initial commit: 1a4ea27865aad968f266788294b0f3193bf5ac90.
- Complete recursive tree: 74 files; no missing or extra payload files.
- All 74 Git blob hashes match the METADATA_FIXED sealed ZIP byte identities.
- GitHub tags and releases were both empty at verification time.

## Current metadata scope

README, CITATION, environment provenance, release-state metadata, publication
checklist and current reproducibility wording now use the actual public URL.
Public code availability is distinct from a formal tagged release:

```ini
INITIAL_PUBLIC_REPOSITORY_ACCESS = PASS
INITIAL_PUBLIC_PAYLOAD_IDENTITY = PASS_74_OF_74
CODE_RELEASE = SEALED_PUBLIC_CODE_RELEASE_PENDING_TAG
FORMAL_V1_0_0_TAG_RELEASE = NOT_CREATED
ZENODO_DOI = NOT_ASSIGNED
UPDATED_METADATA_REMOTE_SYNC = VERIFY_AFTER_AUTHOR_UPLOAD
```

The local validator checks metadata consistency and the refreshed payload
checksums. It does not perform network requests and cannot establish whether
this newer metadata has reached GitHub. A new remote-content check is required
after the author uploads this bundle and before creating v1.0.0.

## Unchanged scientific content and scope limits

- All 25 core statistical modules and all drawing modules are unchanged.
- Original/rerun environment and frozen numerical/figure evidence are preserved.
- The only Python change is the release-preflight validator's publication checks.
- No analyses, models, bootstraps, figures, tables or manuscripts are rerun/edited.
- Six-author order, author-team MIT licence and data restrictions are unchanged.
- No patient data, models, credentials, logs or figure images are added.
- Historical local-seal reports remain clearly labeled historical; their verdicts
  are not rewritten to imply they were originally public-repository checks.
- This is not whole-manuscript submission compliance or independent legal advice.
- No GitHub push, Tag, Release or Zenodo record is created by this update.

Current state: RELEASE_STATUS.json. Historical numerical evidence:
CPO_CODE_RELEASE_QA_V1_2.md and RELEASE_EVIDENCE_V1_2.json.
