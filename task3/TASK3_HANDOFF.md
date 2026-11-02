# COMP5048 Task 3 handoff for D · streamlined page

This independent package was copied from the 2026-11-02 `COMP5048_Task3_D_handoff` package. At the start of this edit, its `task3_app.py` and `task3_analysis.py` matched `/Users/ruqili/5048/` byte for byte. Only this package's `task3_app.py` was changed for the page. The Mac project folder and the earlier handoff were not overwritten.

Run with Python 3.13 from this folder:

```bash
python -m pip install -r requirements-task3.txt
python task3_app.py
```

Open `http://127.0.0.1:8051`. The visible page now has three sections:

1. **Critical Moment:** Nov 1–8 by default, G1 blue (`#2563eb`) and G2 orange (`#f28e2b`) selected by default, with optional G3 purple (`#9b3fc4`). The timeline can show any of eight attributes. The second chart compares the eight observed attribute medians of A's five Nov 1 G2 and two Nov 8 G1 representative rows.
2. **Weather-only What-if:** Fixes A's Nov 8 13:00 G1 hour and selects one real, jointly observed T/RH/AH record from Oct 15–Nov 1 using the slider. Choose Weather Linear Regression or Weather Random Forest Regression. It shows the S4 bar comparison, eight-attribute chart, chosen values, held-out S4 error, and representative G2 ranges.
3. **Limitations:** Calls out the negative held-out S4 R², the representative-record meaning of G2 ranges, and the absence of causal or group-transition evidence.

The earlier lag-based What-if, repeated G1/G2/G3 charts and tables, distance chart, A/B ID button, and long explanations are absent from the visible page. Their calculation functions and hidden compatibility components remain in `task3_app.py` because the unchanged test suite calls those Dash endpoints. D should integrate the visible sections and leave any legacy endpoints out of the submitted interface. `task3_analysis.py`, input data, evaluation outputs, and all test files are unchanged from the prior handoff.

Validation in this package: `python -m pytest -q test_task3.py test_task3_final.py test_weather_task3.py` passed **118 tests**. The tests include actual Dash HTTP callbacks for date/group/attribute controls, both weather models, the S4/eight-attribute figures, and legacy compatibility. The local preview server could not bind a port in this workspace, so no live-browser screenshot is claimed. Run `python verify_task3.py` if D needs to regenerate the unchanged evaluation and audit files in `outputs/`.

The weather-only models use T, RH and AH to predict five sensors, with chronological 70%/15%/15% train/validation/test partitions. Rows missing required measurements are excluded only from fitting and evaluation; original NaNs remain untouched. The Nov 8 demonstration hour lies in training, so its local error is illustrative. Held-out S4 performance remains weak: Linear Regression MAE 293.95, R² −0.532; Random Forest MAE 223.31, R² −0.053. The G2 band is the min–max of A-selected representative rows, not a confirmed Tableau brush boundary. Scenario counts are descriptive, not assignments to a group. D should write the submitted report independently under course AI-use rules.
