"""Final acceptance: actual Dash HTTP callbacks and independent metrics."""
import itertools
import numpy as np
import pandas as pd
import pytest
import task3_app as a
from task3_analysis import FEATURES, TARGETS, LAG_FEATURES, WEATHER, check_groups_csv


def invoke(prefix, values, changed):
    key = next(k for k in a.app.callback_map if k.startswith(prefix))
    spec = a.app.callback_map[key]
    outputs = spec['output']
    def output(x):
        return {'id': x.component_id, 'property': x.component_property}
    response = a.app.server.test_client().post('/_dash-update-component', json={
        'output': key,
        'outputs': [output(x) for x in outputs] if isinstance(outputs, list) else output(outputs),
        'inputs': [dict(item, value=value) for item, value in zip(spec['inputs'], values)],
        'state': [], 'changedPropIds': [changed],
    })
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()['response']


@pytest.mark.parametrize('attribute', a.ATTRS)
@pytest.mark.parametrize('groups', [[], ['G1'], ['G2'], ['G3'], list(a.GROUPS)])
@pytest.mark.parametrize('context', [[], ['source']])
def test_explorer_controls(attribute, groups, context):
    response = invoke('..timeline.', [groups, attribute, '2004-11-05', '2004-11-08', context], 'attribute.value')
    fig = response['timeline']['figure']
    assert len(fig['data']) == len(groups) + bool(context)
    for trace in fig['data']:
        if trace['name'] in a.COLORS:
            assert trace['marker']['color'] == a.COLORS[trace['name']]


@pytest.mark.parametrize('button,start,end', [
    ('oct-nov','2004-10-01','2004-11-30'), ('october','2004-10-01','2004-10-31'),
    ('november','2004-11-01','2004-11-30'), ('all-dates',a.MIN_DATE.isoformat(),a.MAX_DATE.isoformat())])
def test_date_presets(button, start, end):
    r = invoke('..dates.', [1,1,1,1], button+'.n_clicks')['dates']
    assert (r['start_date'],r['end_date']) == (start,end)


@pytest.mark.parametrize('split', ['train','validation','test'])
def test_evaluation_callback(split):
    assert 'evaluation-table' in invoke('evaluation-table.', [split], 'evaluation-split.value')


@pytest.mark.parametrize('model', list(a.models))
def test_all_reference_hours_and_presets(model):
    for row in a.scenario_rows.itertuples():
        for button in ['scenario-record.value','scenario-reset.n_clicks',
                       'scenario-case.n_clicks','scenario-nov5.n_clicks']:
            inputs = invoke('..scenario-t.', [row.RecordID, 1, 1, 1], button)
            weather = [inputs[k]['value'] for k in ['scenario-t','scenario-rh','scenario-ah']]
            if button == 'scenario-nov5.n_clicks':
                np.testing.assert_allclose(weather, [a.NOV5_WEATHER[k] for k in WEATHER])
            r = invoke('..scenario-chart.', [row.RecordID, model, *weather], 'scenario-model.value')
            assert 'Evidence insufficient' in r['group-evidence']['children']
            assert len(r['scenario-chart']['figure']['data']) == 6


@pytest.mark.parametrize('weather', [(None,None,None), (25,50,1), (100,100,5), (-274,50,1), (25,-1,1), (25,101,1), (25,50,-1)])
def test_weather_boundaries(weather):
    r = invoke('..scenario-chart.', [a.DEFAULT_RECORD,'Linear Regression',*weather], 'scenario-t.value')
    invalid = any(v is not None and v < 0 for v in weather[1:]) or weather[0] == -274 or weather[1] == 101
    assert ('Invalid weather input' in r['scenario-note']['children']) == invalid
    if weather[0] == 100:
        assert 'Outside the training' in r['scenario-note']['children']


def test_metrics_independently():
    table = a.model_rows
    n = len(table)
    for name, part in zip(['train','validation','test'], [table.iloc[:int(.7*n)],table.iloc[int(.7*n):int(.85*n)],table.iloc[int(.85*n):]]):
        y = part[list(TARGETS)].to_numpy()
        for method in [*a.models,'Previous hour persistence']:
            p = a.models[method].predict(part[list(FEATURES)]) if method in a.models else part[list(LAG_FEATURES)].to_numpy()
            report = a.model_report['evaluation'][name]
            recorded = report['models'][method] if method in a.models else report[method]
            for i,t in enumerate(TARGETS):
                error = y[:,i]-p[:,i]
                actual = [np.abs(error).mean(), np.sqrt((error**2).mean()), 1-(error**2).sum()/((y[:,i]-y[:,i].mean())**2).sum()]
                np.testing.assert_allclose(actual, [recorded[t][k] for k in ['mae','rmse','r2']],rtol=1e-12)


def test_all_eight_attributes_and_labels():
    merged = a.df.merge(a.source_df,on='RecordID',suffixes=('_a','_s'),validate='one_to_one')
    for col in a.ATTRS:
        np.testing.assert_allclose(merged[col+'_a'],merged[col+'_s'],equal_nan=True,rtol=0,atol=0)
    result = check_groups_csv(a.ROOT/'groups.csv',a.df)
    assert result['matching_labels'] == 1205
    assert result['conflicting_labels'] == result['A_rows_missing_from_B'] == result['B_rows_not_in_A'] == 0
    assert '1205 labels match' in invoke('b-groups-status.',[1],'check-b-groups.n_clicks')['b-groups-status']['children']


@pytest.mark.parametrize('data', ['RecordID,Group\na,G9\n','RecordID,Group\na,\n','RecordID,Group\na,G1\na,G2\n','wrong,Group\na,G1\n',''])
def test_bad_csv(tmp_path,data):
    path=tmp_path/'groups.csv';path.write_text(data)
    assert check_groups_csv(path,a.df)['status']=='invalid'


def test_empty_window_and_nonfinite():
    r=invoke('..timeline.',[list(a.GROUPS),a.ATTRS[0],'2000-01-01','2000-01-02',[]],'dates.start_date')
    assert '(0 total)' in r['coverage']['children']
    for value in [np.nan,np.inf,'bad']:
        assert 'Invalid weather input' in a.update_scenario(a.DEFAULT_RECORD,'Linear Regression',value,50,1)[2]


def test_source_not_mutated():
    raw=pd.read_excel(a.SOURCE_FILE,sheet_name='Data').sort_values('DateTime')
    pd.testing.assert_frame_equal(raw,a.source_df,check_dtype=False)
