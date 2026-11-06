COMP5048 REVISED INTERACTIVE VISUALISATIONS

The original sensor_response.xlsx, groups.csv and Group Data.xlsx are copied WITHOUT ANY CHANGES. Original uploaded Python files also remain untouched.

Task 2: python task2_app.py  -> http://127.0.0.1:8050
Task 3: python task3_app.py  -> http://127.0.0.1:8051

Open a terminal in this extracted folder and install: 
  pip install dash pandas numpy plotly openpyxl scikit-learn

Task 2 additions:
 - Explicit interpretation of grey: missing value in View 1 vs no grouped records in View 2.
 - Group counts in active month range, selected month-hour counts and small month-group warnings (n < 10).
 - Existing membership heatmap, linking, monthly medians, groups, dates and axes preserved.

Task 3 additions:
 - Clear labels for observed, baseline-model and scenario-model S4, with G2 range shaded.
 - A new five-sensor grouped comparison of four profiles (observed reference, actual-weather baseline, alternative-weather scenario, observed G2 median); raw measurements in hover tooltips.
 - Existing eight-attribute view preserved.
 - Small selected-sample warning and missing value note.
 - Existing regression models, split logic and files unchanged.

IMPORTANT: Weather what-if predictions are illustrative model projections, not causal evidence or proof of actual group membership.

Validation performed in this environment: Python source compilation, unchanged-data SHA256 comparisons, and input schema inspection. Live Dash runtime could not be tested because the environment does not provide the dash package and network installation was unavailable.
