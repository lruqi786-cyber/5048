"""Reproduce technical audit artifacts without changing source workbooks."""
import hashlib
import json
import platform
from pathlib import Path
import importlib.metadata as metadata
import numpy as np
import pandas as pd
import task3_app as a
from task3_analysis import (TARGETS, WEATHER, scenario_prediction, profile_distances,
                            check_groups_csv, g3_attribute_directions, weather_joint_support)


def main():
    out = a.ROOT / 'outputs'
    out.mkdir(exist_ok=True)
    row = a.scenario_rows.loc[a.scenario_rows.RecordID == a.DEFAULT_RECORD].iloc[0]
    training = a.model_rows.iloc[:int(.70 * len(a.model_rows))]
    cases = {}
    for name, model in a.models.items():
        base, changed, outside = scenario_prediction(model,a.model_report,row,a.NOV5_WEATHER)
        cases[name] = {
            'actual_weather_prediction': dict(zip(TARGETS,base)),
            'nov5_weather_prediction': dict(zip(TARGETS,changed)),
            'prediction_delta': dict(zip(TARGETS,changed-base)),
            'actual_weather_prediction_abs_error': dict(zip(TARGETS,abs(row[list(TARGETS)].to_numpy(dtype=float)-base))),
            'held_out_test_mae':{target:a.model_report['evaluation']['test']['models'][name][target]['mae'] for target in TARGETS},
            'observed_distances': profile_distances(row[list(TARGETS)],a.sensor_profiles,a.source_df),
            'actual_weather_prediction_distances':profile_distances(base,a.sensor_profiles,a.source_df),
            'scenario_distances': profile_distances(changed,a.sensor_profiles,a.source_df),
            'g3_attribute_directions':g3_attribute_directions(base,changed,a.sensor_profiles,a.source_df),
            'nearest_training_weather_triplet_iqr':weather_joint_support(training,a.NOV5_WEATHER),
            'outside_marginal_training_range': outside,
        }
        fig = a.make_scenario_chart(row[list(TARGETS)],changed,base)
        fig.write_html(out / ('scenario_linear.html' if name=='Linear Regression' else 'scenario_forest.html'),include_plotlyjs=True)
    rows=[]
    for split,values in a.model_report['evaluation'].items():
        for method, metrics in {**values['models'],'Previous hour persistence':values['Previous hour persistence']}.items():
            for target, result in metrics.items():
                rows.append({'split':split,'n':values['n'],'method':method,'target':target,**result})
    pd.DataFrame(rows).to_csv(out/'model_metrics.csv',index=False)
    audit = {
        'source_files': {p.name: {'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [a.GROUP_FILE,a.SOURCE_FILE,a.ROOT/'groups.csv']},
        'python':platform.python_version(),
        'dependencies':{n:metadata.version(n) for n in ['dash','plotly','numpy','pandas','openpyxl','scikit-learn','pytest']},
        'group_mapping':check_groups_csv(a.ROOT/'groups.csv',a.df),
        'group_counts':{str(k):int(v) for k,v in a.df.Group.value_counts().items()},
        'source_rows':len(a.source_df),
        'source_nan_counts':{k:int(v) for k,v in a.source_df[list(a.ATTRS)].isna().sum().items()},
        'eight_attributes':list(a.ATTRS),
        'colors':a.COLORS,
        'candidate_periods':a.periods,
        'local_selected_counts':{str(k):int(v) for k,v in a.df.loc[(a.df.DateTime>='2004-11-05')&(a.df.DateTime<'2004-11-09')].Group.value_counts().items()},
        'focus_selected_counts': {
            'Nov 1 G2': int(((a.df.Group == 'G2') & (a.df.DateTime.dt.date == pd.Timestamp('2004-11-01').date())).sum()),
            'Nov 8 G1': int(((a.df.Group == 'G1') & (a.df.DateTime.dt.date == pd.Timestamp('2004-11-08').date())).sum()),
        },
        'reference_record':a.DEFAULT_RECORD,'reference_time':str(row.DateTime),
        'reference_observed':{k:float(row[k]) for k in [*TARGETS,*WEATHER]},
        'scenario_weather':a.NOV5_WEATHER,
        'scenario_weather_reference_record':a.NOV5_REFERENCE.RecordID,
        'scenario_weather_reference_time':str(a.NOV5_REFERENCE.DateTime),
        'complete_data_eight_attribute_periods':a.period_eight_profiles,
        'what_if':cases,
        'weather_only': {
            'evaluation': a.weather_report,
            'observed_pool_window': '2004-10-15 00:00 inclusive to 2004-11-02 00:00 exclusive',
            'observed_weather_triplets': len(a.weather_pool),
            'selected_G2_observed_ranges': a.selected_ranges['G2'],
            'scenario_counts': a.weather_counts,
            'nov8_reference_is_training': bool(a.NOV8_REFERENCE.DateTime <=
                pd.Timestamp(a.weather_report['evaluation']['train']['end'])),
            'default_weather_record': str(a.weather_pool.iloc[a.WEATHER_DEFAULT_INDEX].RecordID),
        },
        'evidence_verdict':'Insufficient evidence for preserved or changed group membership; descriptive profile resemblance only, no causal claim.',
        'limitations':[
            'No validated group-membership rule, decision boundary, or independent labelled transition outcomes.',
            'No calibrated scenario uncertainty interval or equivalence margin; nearest median is not a membership test.',
            'Selected reference hour is in training, not a held-out prospective forecast.',
            'Weather is contemporaneous; an operational one-hour-ahead forecast would need future weather inputs.',
            'Fixed lagged sensors can dominate apparent persistence; T/RH/AH can be mutually inconsistent.',
            'Scaling uses all-source IQR for descriptive displays; not a training-only validation selection scale.',
            'Only two G3 and two G1 representative rows occur in the candidate window; selected rows are not population samples.',
        ],
    }
    (out/'final_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (out/'task3_evaluation.json').write_text(json.dumps(a.model_report,indent=2),encoding='utf-8')
    (out/'task3_weather_evaluation.json').write_text(json.dumps(a.weather_report,indent=2),encoding='utf-8')
    (out/'weather_scenario_summary.json').write_text(json.dumps(audit['weather_only'],indent=2),encoding='utf-8')
    weather_rows=[]
    for split, values in a.weather_report['evaluation'].items():
        for method, results in values['models'].items():
            for target, metrics in results.items():
                weather_rows.append({'split':split,'n':values['n'],'method':method,'target':target,**metrics})
    pd.DataFrame(weather_rows).to_csv(out/'weather_model_metrics.csv',index=False)
    a.make_eight_parallel('source').write_html(out/'source_eight_profile.html',include_plotlyjs=True)
    a.make_eight_parallel('A').write_html(out/'a_eight_profile.html',include_plotlyjs=True)
    print(json.dumps({'mapping':audit['group_mapping'],'what_if':cases,'nan':audit['source_nan_counts']},indent=2))


if __name__=='__main__':
    main()
