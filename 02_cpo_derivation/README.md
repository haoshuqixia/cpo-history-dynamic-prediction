# CPO derivation — shared implementations

MIMIC CO/MAP pairing is implemented in
`01_cohort_construction/STAGE_MINUS1A_CV_OBSERVABILITY_V1/code/pairing.py`.
Landmark CPO is derived in `03_landmark_generation/.../run_landmark_audit.py`
and the axis construction in `05_outcome_and_ipow/.../run_stage0_mechanism_audit.py`.
The formula is CO × MAP / 451 (watts).

eICU nursing-chart CO/MAP pairing is implemented in
`08_eicu_external_validation/.../run_eicu_stage_eminus1.py`.
No duplicate independent implementation was invented for this folder.
