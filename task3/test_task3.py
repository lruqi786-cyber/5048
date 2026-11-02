"""Targeted checks for Task 3 data handling, evaluation and interaction."""

import numpy as np
import pandas as pd

import task3_app as app
from task3_analysis import (FEATURES, LAG_FEATURES, TARGETS, PERIODS, check_groups_csv,
                            model_table, scenario_prediction)


def test_fixed_groups_match_source_and_keep_nans():
    assert len(app.df) == 1205
    assert app.df.RecordID.is_unique
    merged = app.df.merge(app.source_df, on="RecordID", suffixes=("_group", "_source"))
    assert len(merged) == 1205
    for column in TARGETS:
        np.testing.assert_allclose(merged[f"{column}_group"], merged[f"{column}_source"])
    assert app.source_df[list(TARGETS)].isna().sum().eq(366).all()


def test_model_uses_previous_observed_hour_without_filling():
    assert not any(target in FEATURES for target in TARGETS)
    assert all(feature.startswith("previous_") for feature in LAG_FEATURES)
    assert app.model_report["eligible_rows"] == 8974
    assert app.model_report["excluded_rows"] == len(app.source_df) - 8974
    small = app.source_df.iloc[:5].copy()
    small.loc[small.index[2], TARGETS[0]] = np.nan
    table = model_table(small)
    assert small[TARGETS[0]].isna().sum() == 1
    assert small.iloc[2].RecordID not in set(table.RecordID)
    assert small.iloc[3].RecordID not in set(table.RecordID)


def test_identical_chronological_evaluation_splits():
    evaluation = app.model_report["evaluation"]
    assert sum(evaluation[name]["n"] for name in ("train", "validation", "test")) == 8974
    assert pd.Timestamp(evaluation["train"]["end"]) < pd.Timestamp(evaluation["validation"]["start"])
    assert pd.Timestamp(evaluation["validation"]["end"]) < pd.Timestamp(evaluation["test"]["start"])
    for split in evaluation.values():
        for method in app.models:
            assert set(split["models"][method]) == set(TARGETS)
            assert all(np.isfinite(list(values.values())).all()
                       for values in split["models"][method].values())


def test_scenario_and_group_comparison_are_descriptive():
    row = app.scenario_rows.iloc[0]
    weather = {name: row[name] for name in ("T", "RH", "AH")}
    for model in app.models.values():
        baseline, unchanged, outside = scenario_prediction(model, app.model_report, row, weather)
        np.testing.assert_allclose(baseline, unchanged)
        assert len(baseline) == 5 and outside == []
    result = app.update_scenario(row.RecordID, "Linear Regression",
                                 row["T"], row["RH"], row["AH"])
    assert len(result[0].data) == 6  # observation, two predictions, three A profiles
    assert len(result[1].data) == 2
    assert "Evidence insufficient" in result[3]
    assert app.pair_results["G3/G1"]["five_sensor_distance"] > app.pair_results["G2/G1"]["five_sensor_distance"]
    assert app.DEFAULT_RECORD == "AQ_005829"


def test_reverse_case_fixes_nov8_hour_and_compares_two_predictions():
    row = app.scenario_rows.loc[app.scenario_rows.RecordID == app.DEFAULT_RECORD].iloc[0]
    assert str(row.DateTime) == "2004-11-08 13:00:00"
    assert app.NOV5_REFERENCE.RecordID == "AQ_005762"
    for model in app.models.values():
        base, changed, outside = scenario_prediction(model, app.model_report, row,
                                                     app.NOV5_WEATHER)
        assert outside == []
        assert not np.allclose(base, changed)
        result = app.update_scenario(row.RecordID, next(k for k,v in app.models.items() if v is model),
                                     *[app.NOV5_WEATHER[k] for k in ("T", "RH", "AH")])
        np.testing.assert_allclose(result[0].data[1].customdata, base)
        np.testing.assert_allclose(result[0].data[2].customdata, changed)
        assert "Distance to A's G3" in result[3]
        assert "Held-out test MAE" in result[2]
        assert "Evidence insufficient" in result[3]


def test_eight_attribute_periods_use_complete_source_and_keep_missing_counts():
    assert len(app.make_eight_parallel("source").data) == 3
    assert len(app.make_eight_parallel("A").data) == 3
    for label, start, end in PERIODS:
        subset = app.source_df.loc[(app.source_df.DateTime >= start) &
                                   (app.source_df.DateTime < end)]
        for attribute in app.ATTRS:
            value = app.period_eight_profiles[label][attribute]
            assert value['n'] == subset[attribute].count()
            assert value['median'] == subset[attribute].median()


def test_b_groups_csv_check_when_available(tmp_path):
    absent = check_groups_csv(tmp_path / "groups.csv", app.df)
    assert absent["status"] == "unavailable"
    sample = app.df[["RecordID", "Group"]].iloc[:2].copy()
    sample.loc[sample.index[1], "Group"] = "G1" if sample.iloc[1].Group != "G1" else "G2"
    sample.to_csv(tmp_path / "groups.csv", index=False)
    result = check_groups_csv(tmp_path / "groups.csv", app.df)
    assert result["A_rows_in_B"] == 2
    assert result["matching_labels"] == 1
    assert result["conflicting_labels"] == 1


def test_dash_page_loads_with_existing_explorer():
    response = app.app.server.test_client().get("/_dash-layout")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    for chart in ("timeline", "comparison", "full-change", "scenario-chart", "distance-chart"):
        assert chart in body
