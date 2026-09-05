# Validation v5.7.2

Validated locally against the retained v5.7.1 codebase.

- Python compilation: `mapping.py`, `gams_compat.py`, `gams_ui.py` passed.
- Original SYN2 540 reference inputs reconstructed successfully.
- Round 4 GAMS-compatible model solved OPTIMAL with HiGHS for DM1/DM2/DM3.
- Solver diagnostics exposed node count, dual bound, MIP gap, model/constraint counts and fixed-variable counts.
- Explicit geography crosswalk maps ATT→EL30, CMK→EL52, WMK→EL53 and STE→EL64; EP2 is correctly retained as non-geographic EPANEK2 and is not fabricated as a map region.
- Bundled NUTS-2/NUTS-3 Greece GeoJSON loads without a map API key.
- Detailed static map export generated successfully with NUTS-3 linework overlay.
- Existing modules were preserved; the release is additive.
