"""Acceptance checks for the observed-weather comparison."""
import numpy as np
import pandas as pd

import task3_app as app
from task3_analysis import TARGETS, WEATHER, observed_weather_pool


def test_weather_rows_and_time_order_without_imputation():
    table = app.weather_rows
    assert table.DateTime.is_monotonic_increasing
    assert table[list(WEATHER) + list(TARGETS)].notna().all().all()
    assert len(table) + app.weather_report['excluded_rows'] == len(app.source_df)
    splits = app.weather_report['evaluation']
    assert splits['train']['end'] < splits['validation']['start']
    assert splits['validation']['end'] < splits['test']['start']
    assert app.weather_report['features'] == list(WEATHER)


def test_pool_is_jointly_observed_and_g2_ranges_are_selected_values():
    pool = observed_weather_pool(app.source_df)
    pd.testing.assert_frame_equal(pool, app.weather_pool)
    assert len(pool) == 432
    assert pool[list(WEATHER)].notna().all().all()
    assert pool.DateTime.min() == pd.Timestamp('2004-10-15 00:00')
    assert pool.DateTime.max() == pd.Timestamp('2004-11-01 23:00')
    g2 = app.df.loc[app.df.Group == 'G2']
    for column in app.ATTRS:
        assert app.selected_ranges['G2'][column]['min'] == g2[column].min()
        assert app.selected_ranges['G2'][column]['max'] == g2[column].max()


def test_weather_case_matches_model_and_has_eight_attributes():
    index = app.WEATHER_DEFAULT_INDEX
    candidate = app.weather_pool.iloc[index]
    for name, model in app.weather_models.items():
        s4, parallel, case, range_info = app.update_weather_case(index, name)
        actual = pd.DataFrame([{k: app.NOV8_REFERENCE[k] for k in WEATHER}])
        alternative = pd.DataFrame([{k: candidate[k] for k in WEATHER}])
        expected = model.predict(alternative)[0][TARGETS.index('PT08.S4(NO2)')]
        np.testing.assert_allclose(s4.data[0].y[2], expected)
        assert len(parallel.data) == 5
        assert all(len(trace.x) == 8 for trace in parallel.data)
        assert candidate.RecordID in case
        assert 'Held-out S4 MAE' in case
        assert f"{app.weather_counts[name]['within_selected_G2_S4_T_AH_ranges']}/432" in range_info
        assert not np.allclose(model.predict(actual), model.predict(alternative))


def test_focus_case_uses_a_selected_nov1_g2_and_nov8_g1():
    assert (app.DEFAULT_START, app.DEFAULT_END) == ('2004-11-01', '2004-11-08')
    fig = app.make_local_g2_g1_chart()
    assert len(fig.data) == 2
    assert 'n=5' in fig.data[0].name
    assert 'n=2' in fig.data[1].name
    assert fig.data[0].line.color == app.COLORS['G2']
    assert fig.data[1].line.color == app.COLORS['G1']
    for group, trace, date in [('G2', fig.data[0], '2004-11-01'),
                               ('G1', fig.data[1], '2004-11-08')]:
        part = app.df.loc[(app.df.Group == group) & (app.df.DateTime.dt.date == pd.Timestamp(date).date())]
        np.testing.assert_allclose(trace.customdata, part[list(app.ATTRS)].median())


def test_weather_callback_and_layout_load():
    client = app.app.server.test_client()
    assert client.get('/_dash-layout').status_code == 200
    assert 'weather-index' in client.get('/_dash-layout').get_data(as_text=True)
    key = next(k for k in app.app.callback_map if k.startswith('..weather-s4-chart.'))
    outputs = app.app.callback_map[key]['output']
    response = client.post('/_dash-update-component', json={
        'output': key,
        'outputs': [{'id': x.component_id, 'property': x.component_property} for x in outputs],
        'inputs': [{'id': 'weather-index', 'property': 'value', 'value': 0},
                   {'id': 'weather-model', 'property': 'value', 'value': 'Weather Linear Regression'}],
        'state': [], 'changedPropIds': ['weather-index.value'],
    })
    assert response.status_code == 200, response.get_data(as_text=True)
    assert 'weather-s4-chart' in response.get_json()['response']
