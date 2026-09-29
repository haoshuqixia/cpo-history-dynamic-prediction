# Publication checklist — historical V1.1 gate snapshot

**Historical record, superseded by RELEASE_STATUS.json, the V1.2 QA report,
and PUBLICATION_CHECKLIST_V1_2.md. Unchecked items below describe the prior
draft, not the current author-authorized release.**

- [x] Final author names/order and English title confirmed; Yafei Li is corresponding author; no equal-contribution designation.
- [ ] Copyright ownership and software licence confirmed.
- [x] Trained models remain excluded from the local code candidate.
- [x] Verified rerun environment chosen and tested for statistical reproduction.
- [ ] Optional Word-generation environment pinned.
- [x] Relocated raw-data pipeline run in a fresh private output tree; 27/27 stages completed.
- [x] MIMIC temporal estimates match the frozen reference numerically.
- [x] eICU final R1 denominator = 7,753; 469 events; 840 patients; 27 hospitals.
- [x] Final eICU R1 metrics and uncertainty match the reference.
- [x] Clinical missing-value preprocessing/calibration fold boundaries inspected; patient/hospital folds disjoint.
- [x] Charttime primary and storetime sensitivity documented separately.
- [x] Haemodynamic history measurement/availability windows checked independently; no claim that primary charttime proves real-time availability.
- [ ] All current main/supplement tables and figures map to the final code.
- [x] Main Table 1 numeric formatter checked against the current manuscript: 72/72 cells match.
- [x] Reporting adapters use R1, not the baseline eICU dependency results.
- [x] Newly generated data/logs/models are outside the repository.
- [ ] Eight Table 2 external numeric differences resolved; author requested list only.
- [x] Approved V5 Graphical Abstract spacing and V1 supplementary panel lettering locked.
- [x] All 10 locked figure/table PNGs reproduced exactly by the portable V1.1 reporting endpoint.
- [x] 25 statistical modules match the verified V1.0 candidate byte-for-byte.
- [ ] eICU descriptive first-row ordering terminology clarified (sid is not verified chronology).
- [ ] Credential, path, restricted-data and artifact review rerun before upload.
- [ ] SHA-256 release manifest generated after all edits.
- [ ] Final QA PASS and SEALED recorded separately from source-analysis sealing.
- [ ] Public repository URL verified; optional release DOI resolves correctly.
- [ ] Manuscript Code availability updated only after public release exists.
