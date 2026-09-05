# Validation v5.7.1

- Python compile check: PASS for `app.py`, `gams_ui.py`, `gams_compat.py`.
- Embedded SYN2 budget input: 540 × 5 (EP2, ATT, CMK, WMK, STE).
- Embedded SYN2 score input: 540 × 3.
- Embedded sector vector: 540 projects.
- Supplied status sets retained: GREEN 141, RED 295, GRAY 103; project 484 remains UNCLASSIFIED because it is absent from the supplied status text sets.
- Regional mapping UI no longer silently reuses the same numeric source column.
- Solve button is disabled until required preset regions are mapped uniquely.
- No earlier analytical module removed.
