"""Descriptive group checks and one-hour-ahead multivariate regression.

The source tables retain NaNs. Model evaluation uses only hours with all
required observed current and immediately previous measurements available.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGETS = (
    "PT08.S1(CO)", "C6H6(GT)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)",
)
WEATHER = ("T", "RH", "AH")
COMPARISON_ATTRIBUTES = TARGETS + WEATHER
LAG_FEATURES = tuple(f"previous_{target}" for target in TARGETS)
TIME_FEATURES = ("hour_sin", "hour_cos", "year_sin", "year_cos")
FEATURES = WEATHER + LAG_FEATURES + TIME_FEATURES
FOCUS = ("PT08.S4(NO2)", "T", "AH")
PERIODS = (
    ("Before · Nov 1–4", "2004-11-01", "2004-11-05"),
    ("Candidate · Nov 5–7", "2004-11-05", "2004-11-08"),
    ("After · Nov 8–11", "2004-11-08", "2004-11-12"),
)


def eight_attribute_period_profiles(source, attributes=COMPARISON_ATTRIBUTES):
    """Observed period medians and quartiles; NaNs remain absent from each count."""
    result = {}
    for label, start, end in PERIODS:
        subset = source.loc[(source.DateTime >= start) & (source.DateTime < end)]
        result[label] = {
            attribute: {
                "n": int(subset[attribute].count()),
                "median": float(subset[attribute].median()),
                "q25": float(subset[attribute].quantile(.25)),
                "q75": float(subset[attribute].quantile(.75)),
            } for attribute in attributes
        }
    return result


def weather_joint_support(training, weather):
    """Nearest observed training weather triplet in full-source IQR units."""
    frame = training[list(WEATHER)].dropna()
    scale = frame.quantile(.75) - frame.quantile(.25)
    values = np.array([weather[name] for name in WEATHER], dtype=float)
    distances = np.sqrt(np.mean(((frame.to_numpy(dtype=float) - values) /
                                 scale.to_numpy(dtype=float)) ** 2, axis=1))
    return float(distances.min())


def g3_attribute_directions(baseline, changed, profiles, source):
    """Signed movement of each predicted sensor relative to A's G3 median."""
    scale = source[list(TARGETS)].quantile(.75) - source[list(TARGETS)].quantile(.25)
    answer = {}
    for i, target in enumerate(TARGETS):
        g3 = profiles["G3"][target]["median"]
        before = abs(baseline[i] - g3) / scale[target]
        after = abs(changed[i] - g3) / scale[target]
        answer[target] = {"baseline_gap": float(before), "scenario_gap": float(after),
                          "change": "toward G3" if after < before - 1e-9 else
                                    "away from G3" if after > before + 1e-9 else "unchanged"}
    return answer


def period_summary(source):
    result = []
    for label, start, end in PERIODS:
        subset = source.loc[(source.DateTime >= start) & (source.DateTime < end)]
        row = {"period": label, "records": len(subset)}
        for attr in FOCUS:
            values = subset[attr].dropna()
            row[attr] = {"n": len(values), "median": float(values.median()),
                         "q25": float(values.quantile(.25)),
                         "q75": float(values.quantile(.75))}
        result.append(row)
    return result


def group_profiles(groups, source, attributes=FOCUS):
    """A's fixed labels only; full source sets a common numeric scale."""
    scale = source[list(attributes)].quantile(.75) - source[list(attributes)].quantile(.25)
    center = source[list(attributes)].median()
    result = {}
    for group, subset in groups.groupby("Group"):
        result[group] = {attr: {"n": int(subset[attr].count()),
                                "median": float(subset[attr].median()),
                                "q25": float(subset[attr].quantile(.25)),
                                "q75": float(subset[attr].quantile(.75)),
                                "scaled_median": float((subset[attr].median() - center[attr]) / scale[attr])}
                         for attr in attributes}
    return result


def model_table(source):
    """Use only a truly adjacent observed hour; never fill missing values."""
    source = source.sort_values("DateTime").reset_index(drop=True).copy()
    dates = pd.to_datetime(source.DateTime)
    for target, feature in zip(TARGETS, LAG_FEATURES):
        source[feature] = source[target].shift(1)
    source["hour_sin"] = np.sin(2 * np.pi * dates.dt.hour / 24)
    source["hour_cos"] = np.cos(2 * np.pi * dates.dt.hour / 24)
    source["year_sin"] = np.sin(2 * np.pi * dates.dt.dayofyear / 366)
    source["year_cos"] = np.cos(2 * np.pi * dates.dt.dayofyear / 366)
    adjacent = dates.diff().eq(pd.Timedelta(hours=1))
    eligible = adjacent & source[list(FEATURES) + list(TARGETS)].notna().all(axis=1)
    return source.loc[eligible].copy()


def fit_models(source):
    complete = model_table(source)
    n = len(complete)
    train_end, valid_end = int(.70 * n), int(.85 * n)
    splits = {"train": complete.iloc[:train_end],
              "validation": complete.iloc[train_end:valid_end],
              "test": complete.iloc[valid_end:]}
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=120, min_samples_leaf=3, max_features=.8,
            random_state=42, n_jobs=-1),
    }
    for model in models.values():
        model.fit(splits["train"][list(FEATURES)], splits["train"][list(TARGETS)])

    def metrics(actual, predicted):
        return {target: {
            "mae": float(mean_absolute_error(actual[:, col], predicted[:, col])),
            "rmse": float(np.sqrt(mean_squared_error(actual[:, col], predicted[:, col]))),
            "r2": float(r2_score(actual[:, col], predicted[:, col])),
        } for col, target in enumerate(TARGETS)}

    evaluations = {}
    for split_name, part in splits.items():
        actual = part[list(TARGETS)].to_numpy()
        evaluations[split_name] = {
            "n": len(part), "start": str(part.DateTime.iloc[0]),
            "end": str(part.DateTime.iloc[-1]),
            "models": {name: metrics(actual, model.predict(part[list(FEATURES)]))
                       for name, model in models.items()},
            "Previous hour persistence": metrics(actual, part[list(LAG_FEATURES)].to_numpy()),
        }
    report = {
        "task": "Predict five current-hour sensor responses from previous-hour sensor responses, current T/RH/AH, and clock/annual cycles",
        "targets": list(TARGETS), "features": list(FEATURES),
        "source_rows": len(source), "eligible_rows": n,
        "excluded_rows": len(source) - n,
        "split_rule": "chronological 70% / 15% / 15%; both models fit on the same train rows; no test refit",
        "model_settings": {"Linear Regression": "scikit-learn defaults",
                           "Random Forest Regression": "120 trees, min_samples_leaf=3, max_features=0.8, random_state=42"},
        "evaluation": evaluations,
        "training_limits": {feature: [float(splits["train"][feature].quantile(.01)),
                                     float(splits["train"][feature].quantile(.99))]
                            for feature in WEATHER},
    }
    return models, report, complete


def scenario_prediction(model, report, reference, weather):
    baseline = reference[list(FEATURES)].to_frame().T.astype(float)
    changed = baseline.copy()
    for feature in WEATHER:
        changed.loc[:, feature] = float(weather[feature])
    outside = [feature for feature in WEATHER
               if not report["training_limits"][feature][0] <= changed.iloc[0][feature]
               <= report["training_limits"][feature][1]]
    return model.predict(baseline)[0], model.predict(changed)[0], outside


def profile_distances(values, profiles, source):
    """Descriptive distances to A's five-sensor medians; never group labels."""
    scale = (source[list(TARGETS)].quantile(.75) -
             source[list(TARGETS)].quantile(.25)).to_numpy(dtype=float)
    vector = np.asarray(values, dtype=float)
    return {group: float(np.sqrt(np.mean(((vector - np.array(
        [profiles[group][target]["median"] for target in TARGETS])) / scale) ** 2)))
            for group in ("G1", "G2", "G3")}


def pair_comparison(groups, source):
    """Compare A's selected rows only; no grouping is inferred for other rows."""
    medians = groups.groupby("Group")[list(COMPARISON_ATTRIBUTES)].median()
    scale = source[list(COMPARISON_ATTRIBUTES)].quantile(.75) - source[list(COMPARISON_ATTRIBUTES)].quantile(.25)
    result = {}
    for other in ("G2", "G3"):
        sensor_differences = ((medians.loc[other, list(TARGETS)] -
                               medians.loc["G1", list(TARGETS)]) /
                              scale[list(TARGETS)]).to_numpy(dtype=float)
        result[f"{other}/G1"] = {
            "five_sensor_distance": float(np.sqrt(np.mean(sensor_differences ** 2))),
            "attributes": {attribute: {
                "G1_median": float(medians.loc["G1", attribute]),
                f"{other}_median": float(medians.loc[other, attribute]),
                "absolute_difference_in_full_data_IQR": float(abs(
                    (medians.loc[other, attribute] - medians.loc["G1", attribute]) /
                    scale[attribute])),
            } for attribute in COMPARISON_ATTRIBUTES},
        }
    return result


def check_groups_csv(path, selected_groups):
    """Compare B's CSV with A's selected IDs if the file exists; never relabel."""
    if not path.exists():
        return {"status": "unavailable", "message": "B's groups.csv is not present yet; A's labels remain the only group reference."}
    try:
        b = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as error:
        return {"status": "invalid", "message": f"Cannot read groups.csv: {error}"}
    if not {"RecordID", "Group"}.issubset(b.columns):
        return {"status": "invalid", "message": "groups.csv needs RecordID and Group columns."}
    b = b[["RecordID", "Group"]].copy()
    if b.isna().any().any() or not b.Group.isin(["G1", "G2", "G3"]).all():
        return {"status": "invalid", "message": "groups.csv needs non-null IDs and G1/G2/G3 labels."}
    if b.RecordID.duplicated().any():
        return {"status": "invalid", "message": "groups.csv has duplicate RecordIDs."}
    a = selected_groups[["RecordID", "Group"]].merge(b, on="RecordID", how="left",
                                                       suffixes=("_A", "_B"))
    common = int(a.Group_B.notna().sum())
    agreement = int((a.Group_A == a.Group_B).sum())
    return {"status": "compared", "A_selected_rows": len(a),
            "A_rows_in_B": common, "matching_labels": agreement,
            "conflicting_labels": common - agreement,
            "A_rows_missing_from_B": len(a) - common,
            "B_total_rows": len(b),
            "B_rows_not_in_A": int((~b.RecordID.isin(selected_groups.RecordID)).sum())}


def fit_weather_models(source):
    """Contemporaneous weather association, evaluated in chronological order."""
    complete = source.sort_values("DateTime").loc[
        lambda rows: rows[list(WEATHER) + list(TARGETS)].notna().all(axis=1)
    ].copy()
    n = len(complete)
    edges = (int(.70 * n), int(.85 * n))
    parts = {"train": complete.iloc[:edges[0]],
             "validation": complete.iloc[edges[0]:edges[1]],
             "test": complete.iloc[edges[1]:]}
    models = {
        "Weather Linear Regression": LinearRegression(),
        "Weather Random Forest": RandomForestRegressor(
            n_estimators=120, min_samples_leaf=3, max_features=1.0,
            random_state=42, n_jobs=-1),
    }
    for model in models.values():
        model.fit(parts["train"][list(WEATHER)], parts["train"][list(TARGETS)])
    report = {"features": list(WEATHER), "targets": list(TARGETS),
              "source_rows": len(source), "eligible_rows": n,
              "excluded_rows": len(source) - n,
              "split_rule": "chronological 70% / 15% / 15%; train only fit; complete observed rows; no imputation",
              "evaluation": {}}
    for split, part in parts.items():
        actual = part[list(TARGETS)].to_numpy(dtype=float)
        report["evaluation"][split] = {
            "n": len(part), "start": str(part.DateTime.iloc[0]),
            "end": str(part.DateTime.iloc[-1]), "models": {}}
        for name, model in models.items():
            predicted = model.predict(part[list(WEATHER)])
            report["evaluation"][split]["models"][name] = {
                target: {"mae": float(mean_absolute_error(actual[:, i], predicted[:, i])),
                         "rmse": float(np.sqrt(mean_squared_error(actual[:, i], predicted[:, i]))),
                         "r2": float(r2_score(actual[:, i], predicted[:, i]))}
                for i, target in enumerate(TARGETS)}
    return models, report, complete


def representative_ranges(groups, attributes=COMPARISON_ATTRIBUTES):
    """Observed min/max of A's selected rows; not Tableau brush boundaries."""
    return {group: {attribute: {"min": float(part[attribute].min()),
                                "max": float(part[attribute].max()),
                                "n": int(part[attribute].count())}
                    for attribute in attributes}
            for group, part in groups.groupby("Group")}


def observed_weather_pool(source):
    """Real, jointly observed late-October weather triplets, in time order."""
    return source.loc[(source.DateTime >= "2004-10-15") &
                      (source.DateTime < "2004-11-02") &
                      source[list(WEATHER)].notna().all(axis=1),
                      ["RecordID", "DateTime", *WEATHER]].sort_values("DateTime").reset_index(drop=True)


def weather_scenario_summary(models, pool, ranges):
    """Count model scenarios within selected G2 ranges for S4, T and AH."""
    g2 = ranges["G2"]
    weather_in_range = np.logical_and.reduce([
        pool[attr].between(g2[attr]["min"], g2[attr]["max"]).to_numpy()
        for attr in ("T", "AH")])
    result = {}
    for name, model in models.items():
        predicted = model.predict(pool[list(WEATHER)])[:, TARGETS.index("PT08.S4(NO2)")]
        s4_in_range = (predicted >= g2["PT08.S4(NO2)"]["min"]) & (
            predicted <= g2["PT08.S4(NO2)"]["max"])
        result[name] = {"n": len(pool), "within_selected_G2_S4_T_AH_ranges":
                        int(np.count_nonzero(weather_in_range & s4_in_range)),
                        "criterion": "S4 prediction and observed T/AH each within A-selected G2 min/max; descriptive only"}
    return result
