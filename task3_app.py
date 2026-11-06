"""Explore visually selected Task 3 records without changing source values."""

from pathlib import Path
import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, ctx, dcc, html
from task3_analysis import (COMPARISON_ATTRIBUTES, FOCUS, TARGETS, WEATHER,
                            check_groups_csv, fit_models, group_profiles,
                            pair_comparison, period_summary, profile_distances,
                            scenario_prediction, eight_attribute_period_profiles,
                            weather_joint_support, g3_attribute_directions,
                            fit_weather_models, representative_ranges,
                            observed_weather_pool, weather_scenario_summary)


ROOT = Path(__file__).resolve().parent
GROUP_FILE = ROOT / "Group Data.xlsx"
SOURCE_FILE = ROOT / "sensor_response.xlsx"
GROUPS = ("G1", "G2", "G3")
COLORS = {"G1": "#2563eb", "G2": "#f28e2b", "G3": "#9b3fc4"}
ATTRS = (
    "PT08.S1(CO)", "C6H6(GT)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
)
COMPARE_ATTRS = ("PT08.S4(NO2)", "T", "AH")
DEFAULT_START, DEFAULT_END = "2004-11-01", "2004-11-08"


def load_data():
    selected = []
    for group in GROUPS:
        frame = pd.read_excel(GROUP_FILE, sheet_name=group, header=3)
        frame.columns = frame.columns.str.strip()
        frame["Group"] = group
        selected.append(frame)
    groups = pd.concat(selected, ignore_index=True)
    source = pd.read_excel(SOURCE_FILE, sheet_name="Data")
    for frame in (groups, source):
        required = {"RecordID", "DateTime", *ATTRS}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing required columns: {sorted(required - set(frame.columns))}")
        if frame.RecordID.isna().any() or frame.RecordID.duplicated().any():
            raise ValueError("RecordIDs must be non-null and unique")
        frame["DateTime"] = pd.to_datetime(frame["DateTime"], errors="raise")
        if frame.DateTime.isna().any():
            raise ValueError("DateTime must be non-null")
        frame.sort_values("DateTime", inplace=True)
    if groups["RecordID"].duplicated().any():
        raise ValueError("Group Data.xlsx contains duplicate RecordIDs")
    if not groups["RecordID"].isin(source["RecordID"]).all():
        raise ValueError("A selected RecordID is absent from sensor response.xlsx")
    if source.DateTime.duplicated().any():
        raise ValueError("Source DateTime must be unique for hourly lag modelling")
    left = groups.set_index("RecordID")[["DateTime", *ATTRS]].sort_index()
    right = source.set_index("RecordID").loc[left.index, ["DateTime", *ATTRS]]
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    return groups, source


df, source_df = load_data()
models, model_report, model_rows = fit_models(source_df)
weather_models, weather_report, weather_rows = fit_weather_models(source_df)
weather_pool = observed_weather_pool(source_df)
selected_ranges = representative_ranges(df)
weather_counts = weather_scenario_summary(weather_models, weather_pool, selected_ranges)
WEATHER_DEFAULT_INDEX = int((weather_pool.DateTime - pd.Timestamp("2004-10-31 12:00")).abs().idxmin())
periods = period_summary(source_df)
profiles = group_profiles(df, source_df)
sensor_profiles = group_profiles(df, source_df, TARGETS)
eight_profiles = group_profiles(df, source_df, ATTRS)
period_eight_profiles = eight_attribute_period_profiles(source_df, ATTRS)
pair_results = pair_comparison(df, source_df)
scenario_rows = model_rows.loc[(model_rows.DateTime >= "2004-11-05") &
                               (model_rows.DateTime < "2004-11-09")].copy()
SCENARIO_IDS = set(scenario_rows.RecordID)
local_g3 = df.loc[(df.Group == "G3") & (df.DateTime >= "2004-11-05") &
                  (df.DateTime < "2004-11-09")].sort_values("DateTime")
DEFAULT_RECORD = local_g3.iloc[-1].RecordID
assert DEFAULT_RECORD in SCENARIO_IDS
local_g1 = df.loc[(df.Group == "G1") & (df.DateTime >= "2004-11-08") &
                  (df.DateTime < "2004-11-09")]
LOCAL_G1_WEATHER = {feature: float(local_g1[feature].median()) for feature in WEATHER}
NOV5_REFERENCE = local_g3.iloc[-1]
NOV5_WEATHER = {feature: float(NOV5_REFERENCE[feature]) for feature in WEATHER}
NOV8_REFERENCE = df.loc[(df.Group == "G1") &
                        (df.DateTime == pd.Timestamp("2004-11-08 13:00"))].iloc[0]
assert NOV8_REFERENCE.RecordID in SCENARIO_IDS
DEFAULT_RECORD = NOV8_REFERENCE.RecordID
sensor_iqr = source_df[list(TARGETS)].quantile(.75) - source_df[list(TARGETS)].quantile(.25)
VALIDATION_SCALED_MAE = {
    name: sum(model_report["evaluation"]["validation"]["models"][name][target]["mae"] /
              sensor_iqr[target] for target in TARGETS) / len(TARGETS)
    for name in models
}
MIN_DATE = min(df["DateTime"].min(), source_df["DateTime"].min()).date()
MAX_DATE = max(df["DateTime"].max(), source_df["DateTime"].max()).date()


def in_window(frame, start_date, end_date):
    """Include every hour of the end date; leave measurement NaNs untouched."""
    start = pd.Timestamp(start_date or DEFAULT_START)
    end = pd.Timestamp(end_date or DEFAULT_END) + pd.Timedelta(days=1)
    return frame.loc[(frame["DateTime"] >= start) & (frame["DateTime"] < end)]


def selected_window(groups, start_date, end_date):
    return in_window(df, start_date, end_date).loc[
        lambda records: records["Group"].isin(groups or [])
    ]


def point_trace(records, attribute, group, *, legend=False):
    subset = records.loc[records["Group"] == group]
    return go.Scattergl(
        x=subset["DateTime"], y=subset[attribute], mode="markers",
        name=group, legendgroup=group, showlegend=legend,
        marker={"color": COLORS[group], "size": 7, "opacity": 0.78},
        customdata=subset[["RecordID"]].to_numpy(),
        hovertemplate=(
            "<b>%{customdata[0]}</b> · " + group +
            "<br>%{x|%Y-%m-%d %H:%M}<br>" + attribute +
            ": %{y}<extra></extra>"
        ),
    )


def context_trace(records, attribute):
    return go.Scattergl(
        x=records["DateTime"], y=records[attribute], mode="markers",
        name="All source records", legendgroup="source", showlegend=False,
        marker={"color": "#a9b0b8", "size": 4, "opacity": 0.25},
        hoverinfo="skip",
    )


def make_timeline(groups, attribute, start_date, end_date, show_context):
    start_date = start_date or DEFAULT_START
    end_date = end_date or DEFAULT_END
    chosen = selected_window(groups, start_date, end_date)
    fig = go.Figure()
    if show_context:
        fig.add_trace(context_trace(in_window(source_df, start_date, end_date), attribute))
    for group in GROUPS:
        if group in (groups or []):
            fig.add_trace(point_trace(chosen, attribute, group, legend=True))
    fig.update_layout(
        title=f"Selected records over time · {attribute}",
        height=520, template="plotly_white", hovermode="closest",
        margin={"l": 60, "r": 20, "t": 65, "b": 60},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.12},
    )
    fig.update_xaxes(title="Date and time", rangeslider={"visible": True},
                     range=[pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1)])
    fig.update_yaxes(title=attribute)
    return fig


def make_comparison(groups, start_date, end_date, show_context):
    start_date = start_date or DEFAULT_START
    end_date = end_date or DEFAULT_END
    chosen = selected_window(groups, start_date, end_date)
    context = in_window(source_df, start_date, end_date) if show_context else None
    fig = make_subplots(rows=1, cols=3, subplot_titles=COMPARE_ATTRS,
                        horizontal_spacing=0.075)
    for col, attribute in enumerate(COMPARE_ATTRS, start=1):
        if context is not None:
            fig.add_trace(context_trace(context, attribute), row=1, col=col)
        for group in GROUPS:
            if group in (groups or []):
                fig.add_trace(point_trace(chosen, attribute, group,
                                          legend=col == 1), row=1, col=col)
        fig.update_xaxes(row=1, col=col, title_text="Date",
                         range=[pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1)])
        fig.update_yaxes(row=1, col=col, title_text=attribute)
    fig.update_layout(
        height=470, template="plotly_white", hovermode="closest",
        margin={"l": 65, "r": 20, "t": 60, "b": 70},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.16},
        title={"text": "Three attributes, same time window", "y": 0.98},
    )
    return fig


def coverage_text(groups, start_date, end_date):
    chosen = selected_window(groups, start_date, end_date)
    source = in_window(source_df, start_date, end_date)
    counts = chosen.groupby("Group").size()
    parts = [f"{group}: {counts.get(group, 0)}" for group in GROUPS if group in (groups or [])]
    missing = source[list(COMPARE_ATTRS)].isna().sum()
    return (
        f"Selected records in view: {', '.join(parts) if parts else 'none'} "
        f"({len(chosen)} total). Source records in view: {len(source)}. "
        "Source missing values — " + ", ".join(f"{a}: {missing[a]}" for a in COMPARE_ATTRS) + ". " +
        ("SMALL SAMPLE warning: " + ", ".join(f"{g} n={counts.get(g, 0)}" for g in (groups or []) if 0 < counts.get(g, 0) < 10)
         if any(0 < counts.get(g, 0) < 10 for g in (groups or [])) else "")
    )


def make_full_data_change():
    """Daily observed medians and IQR; no inferred membership or interpolation."""
    window = source_df.loc[(source_df.DateTime >= "2004-11-01") &
                           (source_df.DateTime < "2004-11-12")].copy()
    window["Day"] = window.DateTime.dt.floor("D")
    fig = make_subplots(rows=1, cols=3, subplot_titles=FOCUS,
                        horizontal_spacing=.075)
    for col, attr in enumerate(FOCUS, start=1):
        days = window.groupby("Day")[attr].agg(
            n="count", median="median", q25=lambda x: x.quantile(.25),
            q75=lambda x: x.quantile(.75)).reset_index()
        fig.add_trace(go.Scatter(x=days.Day, y=days["median"], mode="lines+markers",
                                 name="Daily median", showlegend=col == 1,
                                 line={"color": "#2563eb"},
                                 customdata=days[["n", "q25", "q75"]].to_numpy(),
                                 hovertemplate="%{x|%b %d}: %{y:.2f}<br>n=%{customdata[0]}"
                                               "<br>IQR %{customdata[1]:.2f}–%{customdata[2]:.2f}<extra></extra>"),
                      row=1, col=col)
        fig.add_trace(go.Scatter(x=window.DateTime, y=window[attr], mode="markers",
                                 name="Observed hours", showlegend=col == 1,
                                 marker={"size": 3, "color": "#a9b0b8", "opacity": .35},
                                 hoverinfo="skip"), row=1, col=col)
        fig.add_vrect(x0="2004-11-05", x1="2004-11-08", fillcolor="#f7b267",
                      opacity=.16, line_width=0, row=1, col=col)
        fig.update_xaxes(title="November 2004", row=1, col=col)
    fig.update_layout(height=440, template="plotly_white",
                      title="Full data around the candidate window",
                      margin={"l": 55, "r": 20, "t": 65, "b": 60},
                      legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.16})
    return fig


def make_group_profile():
    fig = go.Figure()
    for group in GROUPS:
        fig.add_trace(go.Bar(name=group, x=list(FOCUS),
                             y=[profiles[group][attr]["scaled_median"] for attr in FOCUS],
                             marker_color=COLORS[group],
                             customdata=[profiles[group][attr]["median"] for attr in FOCUS],
                             hovertemplate="%{x}<br>Observed median: %{customdata:.3f}"
                                           "<br>Relative to full-data median: %{y:.2f} IQR<extra></extra>"))
    fig.update_layout(barmode="group", height=390, template="plotly_white",
                      title="A's fixed representative groups · three-attribute profiles",
                      yaxis_title="Difference from full-data median / full-data IQR")
    return fig


def make_eight_parallel(kind):
    """Eight normalized vertical attribute axes on a shared full-data scale."""
    center = source_df[list(ATTRS)].median()
    scale = source_df[list(ATTRS)].quantile(.75) - source_df[list(ATTRS)].quantile(.25)
    fig = go.Figure()
    if kind == "source":
        series = [(label, values, color) for (label, values), color in zip(
            period_eight_profiles.items(), ("#607d8b", "#dc7b21", "#007f73"))]
        title = "Complete data · before, candidate and after"
    else:
        series = [(f"A's {group}", eight_profiles[group], COLORS[group]) for group in GROUPS]
        title = "A's fixed G1/G2/G3 representative profiles"
    for label, values, color in series:
        raw = [values[attr]["median"] for attr in ATTRS]
        scaled = [(values[attr]["median"] - center[attr]) / scale[attr] for attr in ATTRS]
        counts = [values[attr]["n"] for attr in ATTRS]
        fig.add_trace(go.Scatter(x=list(ATTRS), y=scaled, mode="lines+markers",
                                 name=label, line={"color": color, "width": 3},
                                 marker={"size": 8}, customdata=list(zip(raw, counts)),
                                 hovertemplate="%{x}<br>Observed median: %{customdata[0]:.3f}"
                                               "<br>Valid n: %{customdata[1]}"
                                               "<br>Scaled: %{y:.2f} IQR<extra>" + label + "</extra>"))
    fig.update_layout(height=440, template="plotly_white", title=title,
                      yaxis_title="(Median − full-data median) / full-data IQR",
                      xaxis={"tickangle": -35}, legend={"orientation": "h", "y": -0.35},
                      margin={"l": 65, "r": 20, "t": 60, "b": 90})
    return fig


def make_group_scatter():
    fig = go.Figure()
    for group in GROUPS:
        subset = df.loc[df.Group == group]
        fig.add_trace(go.Scattergl(
            x=subset["PT08.S4(NO2)"], y=subset["T"], mode="markers",
            name=group, marker={"color": COLORS[group], "size": 5 + 5 * subset.AH,
                                "opacity": .55},
            customdata=subset[["AH", "RecordID", "DateTime"]].to_numpy(),
            hovertemplate="S4 %{x:.1f}<br>T %{y:.1f}<br>AH %{customdata[0]:.3f}"
                          "<br>%{customdata[1]} · %{customdata[2]}<extra>" + group + "</extra>"))
    fig.update_layout(height=460, template="plotly_white",
                      title="A's representative records · S4, T and AH",
                      xaxis_title="PT08.S4(NO2)", yaxis_title="T")
    return fig


def make_scenario_chart(observed, predicted, baseline=None):
    """All five sensors on the same source-IQR scale for comparison."""
    center = source_df[list(TARGETS)].median().to_numpy(dtype=float)
    scale = (source_df[list(TARGETS)].quantile(.75) -
             source_df[list(TARGETS)].quantile(.25)).to_numpy(dtype=float)
    fig = go.Figure()
    series = [("Observed reference hour", observed, "#263238", "solid")]
    if baseline is not None:
        series.append(("Actual-weather prediction", baseline, "#607d8b", "solid"))
    series.append(("Scenario-weather prediction", predicted, "#007f73", "solid"))
    series += [(f"A's {group} median", [sensor_profiles[group][t]["median"] for t in TARGETS],
                COLORS[group], "dot") for group in GROUPS]
    for name, values, color, dash in series:
        values = pd.Series(values, dtype=float).to_numpy()
        fig.add_trace(go.Scatter(x=list(TARGETS), y=(values - center) / scale,
                                 mode="lines+markers", name=name,
                                 line={"color": color, "dash": dash, "width": 2},
                                 customdata=values,
                                 hovertemplate="%{x}<br>Raw value: %{customdata:.2f}"
                                               "<br>Scaled value: %{y:.2f}<extra>" + name + "</extra>"))
    fig.update_layout(height=470, template="plotly_white",
                      title="Five-sensor comparison · fixed reference hour and two weather inputs",
                      yaxis_title="(Value − full-data median) / full-data IQR",
                      legend={"orientation": "h", "y": -0.27})
    return fig


def make_distance_chart(original_distances, scenario_distances):
    fig = go.Figure()
    for name, values, color in (("Actual-weather prediction", original_distances, "#607d8b"),
                                ("Scenario-weather prediction", scenario_distances, "#007f73")):
        fig.add_trace(go.Bar(name=name, x=list(GROUPS),
                             y=[values[group] for group in GROUPS], marker_color=color))
    fig.update_layout(barmode="group", height=330, template="plotly_white",
                      title="Descriptive distance to A's five-sensor medians · lower is closer",
                      yaxis_title="Root mean squared difference / full-data IQR")
    return fig


def make_weather_parallel(observed, actual_prediction, scenario_prediction_values, weather):
    """Eight axes for the fixed Nov 8 observation and two model inputs."""
    center = source_df[list(ATTRS)].median()
    scale = source_df[list(ATTRS)].quantile(.75) - source_df[list(ATTRS)].quantile(.25)
    actual_weather = {k: float(NOV8_REFERENCE[k]) for k in WEATHER}
    series = [
        ("Nov 8 observed", dict(zip(TARGETS, observed)) | actual_weather, "#263238", "solid"),
        ("Nov 8 weather · model", dict(zip(TARGETS, actual_prediction)) | actual_weather, "#607d8b", "solid"),
        ("Late October weather · model", dict(zip(TARGETS, scenario_prediction_values)) | weather, "#007f73", "solid"),
    ]
    series += [(f"A's {group} median", {k: eight_profiles[group][k]["median"] for k in ATTRS},
                COLORS[group], "dot") for group in ("G1", "G2")]
    fig = go.Figure()
    for name, values, color, dash in series:
        raw = [values[k] for k in ATTRS]
        fig.add_trace(go.Scatter(x=list(ATTRS),
                                 y=[(values[k] - center[k]) / scale[k] for k in ATTRS],
                                 customdata=raw, mode="lines+markers", name=name,
                                 line={"color": color, "dash": dash, "width": 2},
                                 hovertemplate="%{x}<br>Value: %{customdata:.3f}<extra>" + name + "</extra>"))
    fig.update_layout(template="plotly_white", height=460,
                      title="Eight attributes · fixed Nov 8 hour and observed late October weather",
                      yaxis_title="(Value − full-data median) / full-data IQR",
                      xaxis={"tickangle": -35}, legend={"orientation": "h", "y": -.32})
    return fig


def make_local_g2_g1_chart():
    """Observed eight-attribute medians of the seven fixed local G2/G1 rows."""
    center = source_df[list(ATTRS)].median()
    scale = source_df[list(ATTRS)].quantile(.75) - source_df[list(ATTRS)].quantile(.25)
    fig = go.Figure()
    for group, start, end in (("G2", "2004-11-01", "2004-11-02"),
                              ("G1", "2004-11-08", "2004-11-09")):
        part = df.loc[(df.Group == group) & (df.DateTime >= start) & (df.DateTime < end)]
        raw = part[list(ATTRS)].median()
        fig.add_trace(go.Scatter(x=list(ATTRS), y=[(raw[k]-center[k])/scale[k] for k in ATTRS],
                                 customdata=[raw[k] for k in ATTRS], mode="lines+markers",
                                 name=f"{group} · {start} · n={len(part)}",
                                 line={"color": COLORS[group], "width": 3},
                                 hovertemplate="%{x}<br>Observed median: %{customdata:.3f}<extra>"+
                                               f"{group} · {start}</extra>"))
    fig.update_layout(template="plotly_white", height=440,
                      title="A's Nov 1 G2 and Nov 8 G1 representative records · eight attributes",
                      yaxis_title="(Median − full-data median) / full-data IQR",
                      xaxis={"tickangle": -35})
    return fig


def make_weather_s4(observed, baseline, changed):
    s4 = "PT08.S4(NO2)"
    values = [observed[TARGETS.index(s4)], baseline[TARGETS.index(s4)],
              changed[TARGETS.index(s4)]]
    fig = go.Figure(go.Bar(x=["Nov 8 observed", "Nov 8 weather model", "Late October weather model"],
                           y=values, marker_color=["#263238", "#607d8b", "#007f73"],
                           text=[f"{v:.0f}" for v in values], textposition="outside"))
    low, high = (selected_ranges["G2"][s4][key] for key in ("min", "max"))
    fig.add_hrect(y0=low, y1=high, fillcolor=COLORS["G2"], opacity=.12,
                  line_width=0, annotation_text="A-selected G2 observed range")
    fig.update_layout(template="plotly_white", height=400,
                      title="S4: measured vs actual-weather baseline vs alternative weather",
                      yaxis_title=s4, showlegend=False)
    return fig


def make_weather_five_sensor_comparison(observed, baseline, scenario):
    """All five sensor values side-by-side; original values retained and visually labelled."""
    center = source_df[list(TARGETS)].median().to_numpy(dtype=float)
    spread = (source_df[list(TARGETS)].quantile(.75) - source_df[list(TARGETS)].quantile(.25)).to_numpy(dtype=float)
    fig = go.Figure()
    for label, values, color in (
        ("Observed Nov 8 hour", observed, "#263238"),
        ("Baseline model · actual weather", baseline, "#607d8b"),
        ("Alternative model · selected Oct weather", scenario, "#007f73"),
        ("Observed G2 selected-record median", [sensor_profiles["G2"][t]["median"] for t in TARGETS], COLORS["G2"]),
    ):
        values = pd.Series(values, dtype=float).to_numpy()
        fig.add_bar(x=list(TARGETS), y=(values-center)/spread, name=label,
                    marker_color=color, customdata=values,
                    hovertemplate="%{x}<br>Raw value: %{customdata:.2f}<br>Scaled: %{y:.2f} IQR<extra>"+label+"</extra>")
    fig.update_layout(barmode="group", template="plotly_white", height=480,
                      title="Five sensor responses · observed vs baseline vs alternative vs G2",
                      yaxis_title="(Value − full-data median) / full-data IQR",
                      xaxis={"tickangle": -25}, legend={"orientation": "h", "y": -0.34},
                      margin={"b": 135, "t": 65})
    return fig

def weather_evaluation_table():
    rows = []
    for name in weather_models:
        for target in TARGETS:
            metric = weather_report["evaluation"]["test"]["models"][name][target]
            rows.append(html.Tr([html.Td(value, style={"padding": "5px 10px"}) for value in
                                 (name, target, f"{metric['mae']:.1f}", f"{metric['rmse']:.1f}",
                                  f"{metric['r2']:.3f}")]))
    return html.Table([html.Thead(html.Tr([html.Th(x) for x in
                      ("Weather model", "Sensor", "Test MAE", "Test RMSE", "Test R²")])),
                      html.Tbody(rows)], style={"width": "100%"})


def evaluation_table(split):
    split = split if split in model_report["evaluation"] else "test"
    data = model_report["evaluation"][split]
    cell_style = {"padding": "6px 10px", "borderBottom": "1px solid #ddd", "textAlign": "left"}
    header = html.Tr([html.Th(value, style=cell_style) for value in
                      ("Sensor", "Method", "MAE", "RMSE", "R²")])
    rows = []
    for target in TARGETS:
        for method in (*models, "Previous hour persistence"):
            values = (data["models"].get(method) or data[method])[target]
            rows.append(html.Tr([html.Td(value, style=cell_style) for value in
                                 (target, method, f"{values['mae']:.2f}",
                                  f"{values['rmse']:.2f}", f"{values['r2']:.3f}")]))
    return html.Div([
        html.P(f"{split.title()}: {data['n']} eligible hours · {data['start']} to {data['end']}"),
        html.Table([html.Thead(header), html.Tbody(rows)],
                   style={"borderCollapse": "collapse", "width": "100%"})
    ])


def comparison_table():
    cell = {"padding": "6px 10px", "borderBottom": "1px solid #ddd", "textAlign": "left"}
    headers = ("Attribute", "G1 median", "G2 median", "G3 median",
               "G2/G1 gap ÷ IQR", "G3/G1 gap ÷ IQR")
    rows = []
    for attr in COMPARISON_ATTRIBUTES:
        g2 = pair_results["G2/G1"]["attributes"][attr]
        g3 = pair_results["G3/G1"]["attributes"][attr]
        values = (attr, f"{g2['G1_median']:.2f}", f"{g2['G2_median']:.2f}",
                  f"{g3['G3_median']:.2f}",
                  f"{g2['absolute_difference_in_full_data_IQR']:.2f}",
                  f"{g3['absolute_difference_in_full_data_IQR']:.2f}")
        rows.append(html.Tr([html.Td(value, style=cell) for value in values]))
    return html.Table([html.Thead(html.Tr([html.Th(value, style=cell) for value in headers])),
                       html.Tbody(rows)], style={"borderCollapse": "collapse", "width": "100%"})


app = Dash(__name__)
app.title = "Task 3 · Critical Moment"
app.layout = html.Div([
    html.H2("Task 3 · Critical Moment"),
    html.P("Explore A's selected representative records from Nov 1–8, then compare a fixed Nov 8 hour under observed weather conditions."),

    html.H3("Critical Moment · Nov 1–8"),
    html.Div([
        html.Div([html.Label("Representative groups"), dcc.Checklist(
            id="groups", options=[{"label": g, "value": g} for g in GROUPS],
            value=["G1", "G2"], inline=True)]),
        html.Div([html.Label("Attribute"), dcc.Dropdown(
            id="attribute", options=list(ATTRS), value="PT08.S4(NO2)", clearable=False)]),
        html.Div([html.Label("Date window"), dcc.DatePickerRange(
            id="dates", min_date_allowed=MIN_DATE, max_date_allowed=MAX_DATE,
            start_date=DEFAULT_START, end_date=DEFAULT_END)]),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))",
              "gap": "18px", "alignItems": "end"}),
    dcc.Checklist(id="show-context", options=[{
        "label": "Show all source hours in grey", "value": "source"
    }], value=[], style={"marginTop": "16px"}),
    html.P(id="coverage", style={"fontWeight": "bold"}),
    html.P("Warning: fewer than 10 selected records in any group/date window is a small sample. Missing sensor measurements are excluded without imputation.",
           style={"background": "#fff7ed", "padding": "10px", "borderLeft": "4px solid #ea580c"}),
    dcc.Graph(id="timeline", config={"displaylogo": False}),
    dcc.Graph(id="local-g2-g1-chart", figure=make_local_g2_g1_chart(),
              config={"displaylogo": False}),
    html.P("Eight attributes compare the observed medians of five Nov 1 G2 records and two Nov 8 G1 records selected by A."),

    html.H3("Weather-only What-if"),
    html.P("Fix A's Nov 8 13:00 G1 record. Choose a real, jointly observed T/RH/AH record from Oct 15–Nov 1; each model predicts the five sensor responses."),
    html.Label("Observed weather record"),
    dcc.Slider(id="weather-index", min=0, max=len(weather_pool)-1,
               value=WEATHER_DEFAULT_INDEX, step=1,
               marks={i: weather_pool.DateTime.iloc[i].strftime("%b %d")
                      for i in range(0, len(weather_pool), 72)},
               tooltip={"placement": "bottom"}),
    html.Label("Weather-only model"),
    dcc.Dropdown(id="weather-model", options=list(weather_models),
                 value="Weather Linear Regression", clearable=False,
                 style={"maxWidth": "350px"}),
    html.Div("Legend: dark = observed measurement; slate = baseline model under actual weather; teal = alternative-weather model; orange = G2 observed representative profile.",
             style={"padding": "10px", "background": "#edf6f5", "borderRadius": "6px"}),
    html.P(id="weather-case-info"),
    dcc.Graph(id="weather-s4-chart", config={"displaylogo": False}),
    dcc.Graph(id="weather-five-chart", config={"displaylogo": False}),
    html.P("Chart uses a common full-dataset median/IQR scale; hover over each bar for the original sensor value. The G2 line of comparison represents observed selected records, not a model classification."),
    dcc.Graph(id="weather-eight-chart", config={"displaylogo": False}),
    html.P(id="weather-range-info"),

    html.H3("Limitations"),
    html.P("Both weather-only models have negative held-out S4 R². The shaded G2 range comes from A's selected records, not confirmed Tableau boundaries. These model scenarios do not show a group transition or a causal weather effect."),

    # Preserve the previous callbacks and their test endpoints without showing
    # the older analysis panels in the simplified main page.
    html.Div([
        html.Button(id="oct-nov", n_clicks=0),
        html.Button(id="october", n_clicks=0),
        html.Button(id="november", n_clicks=0),
        html.Button(id="all-dates", n_clicks=0),
        dcc.Graph(id="comparison"),
        dcc.Graph(id="full-change"),
        dcc.Dropdown(id="evaluation-split", options=["train", "validation", "test"], value="test"),
        html.Div(id="evaluation-table"),
        html.Button(id="check-b-groups", n_clicks=0),
        html.Div(id="b-groups-status"),
        dcc.Dropdown(id="scenario-record", options=[{"label": str(r.RecordID), "value": r.RecordID}
                                                    for r in scenario_rows.itertuples()],
                     value=DEFAULT_RECORD),
        dcc.Dropdown(id="scenario-model", options=list(models), value="Linear Regression"),
        dcc.Input(id="scenario-t", type="number"),
        dcc.Input(id="scenario-rh", type="number"),
        dcc.Input(id="scenario-ah", type="number"),
        html.Button(id="scenario-reset", n_clicks=0),
        html.Button(id="scenario-case", n_clicks=0),
        html.Button(id="scenario-nov5", n_clicks=0),
        html.Div(id="scenario-note"),
        dcc.Graph(id="scenario-chart"),
        dcc.Graph(id="distance-chart"),
        html.Div(id="group-evidence"),
    ], style={"display": "none"}),
], style={"fontFamily": "Arial, sans-serif", "maxWidth": "1200px",
          "margin": "24px auto", "padding": "0 18px", "color": "#263238"})

@app.callback(Output("dates", "start_date"), Output("dates", "end_date"),
              Input("oct-nov", "n_clicks"), Input("october", "n_clicks"),
              Input("november", "n_clicks"), Input("all-dates", "n_clicks"),
              prevent_initial_call=True)
def set_date_preset(*_clicks):
    return {
        "oct-nov": ("2004-10-01", "2004-11-30"),
        "october": ("2004-10-01", "2004-10-31"),
        "november": ("2004-11-01", "2004-11-30"),
        "all-dates": (MIN_DATE.isoformat(), MAX_DATE.isoformat()),
    }[ctx.triggered_id]


@app.callback(Output("timeline", "figure"), Output("comparison", "figure"),
              Output("coverage", "children"), Input("groups", "value"),
              Input("attribute", "value"), Input("dates", "start_date"),
              Input("dates", "end_date"), Input("show-context", "value"))
def update_charts(groups, attribute, start_date, end_date, context):
    visible = "source" in (context or [])
    return (make_timeline(groups, attribute, start_date, end_date, visible),
            make_comparison(groups, start_date, end_date, visible),
            coverage_text(groups, start_date, end_date))


@app.callback(Output("evaluation-table", "children"), Input("evaluation-split", "value"))
def update_evaluation(split):
    return evaluation_table(split)


@app.callback(Output("b-groups-status", "children"), Input("check-b-groups", "n_clicks"))
def update_b_groups_status(_clicks):
    result = check_groups_csv(ROOT / "groups.csv", df)
    if result["status"] != "compared":
        return result["message"]
    return (f"B groups.csv: {result['A_rows_in_B']}/{result['A_selected_rows']} A-selected IDs found; "
            f"{result['matching_labels']} labels match, {result['conflicting_labels']} conflict, "
            f"{result['A_rows_missing_from_B']} are absent. A's labels remain unchanged.")


@app.callback(Output("scenario-t", "value"), Output("scenario-rh", "value"),
              Output("scenario-ah", "value"),
              Input("scenario-record", "value"), Input("scenario-reset", "n_clicks"),
              Input("scenario-case", "n_clicks"), Input("scenario-nov5", "n_clicks"))
def set_scenario_inputs(record_id, _reset, _case, _nov5):
    if ctx.triggered_id == "scenario-case":
        return tuple(LOCAL_G1_WEATHER[feature] for feature in WEATHER)
    if ctx.triggered_id == "scenario-nov5":
        return tuple(NOV5_WEATHER[feature] for feature in WEATHER)
    if record_id not in SCENARIO_IDS:
        record_id = DEFAULT_RECORD
    row = scenario_rows.loc[scenario_rows.RecordID == record_id].iloc[0]
    return tuple(float(row[feature]) for feature in WEATHER)


@app.callback(Output("scenario-chart", "figure"), Output("distance-chart", "figure"),
              Output("scenario-note", "children"), Output("group-evidence", "children"),
              Input("scenario-record", "value"), Input("scenario-model", "value"),
              Input("scenario-t", "value"), Input("scenario-rh", "value"),
              Input("scenario-ah", "value"))
def update_scenario(record_id, model_name, temperature, humidity, absolute_humidity):
    if record_id not in SCENARIO_IDS:
        record_id = DEFAULT_RECORD
    row = scenario_rows.loc[scenario_rows.RecordID == record_id].iloc[0]
    model_name = model_name if model_name in models else "Linear Regression"
    weather = dict(zip(WEATHER, (temperature, humidity, absolute_humidity)))
    for feature in WEATHER:
        if weather[feature] is None:
            weather[feature] = float(row[feature])
    try:
        weather = {key: float(value) for key, value in weather.items()}
        valid = all(math.isfinite(value) for value in weather.values())
        valid = valid and weather["T"] > -273.15 and 0 <= weather["RH"] <= 100 and weather["AH"] >= 0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        message = "Invalid weather input: use finite numbers, T > -273.15, RH from 0 to 100, and AH >= 0."
        return go.Figure(), go.Figure(), message, "Evidence insufficient: no valid scenario prediction."
    baseline, changed, outside = scenario_prediction(models[model_name], model_report, row, weather)
    observed = row[list(TARGETS)].to_numpy(dtype=float)
    original_distances = profile_distances(baseline, sensor_profiles, source_df)
    scenario_distances = profile_distances(changed, sensor_profiles, source_df)
    directions = g3_attribute_directions(baseline, changed, sensor_profiles, source_df)
    original_order = sorted(original_distances, key=original_distances.get)
    scenario_order = sorted(scenario_distances, key=scenario_distances.get)
    deltas = ", ".join(f"{target}: {delta:+.1f}" for target, delta in zip(TARGETS, changed-baseline))
    local_errors = ", ".join(f"{target}: {abs(value):.1f}" for target, value in
                             zip(TARGETS, observed-baseline))
    test_mae = ", ".join(
        f"{target}: {model_report['evaluation']['test']['models'][model_name][target]['mae']:.1f}"
        for target in TARGETS)
    training = model_rows.iloc[:int(.70 * len(model_rows))]
    nearest = weather_joint_support(training, weather)
    note = (f"Reference {record_id} · {row.DateTime:%Y-%m-%d %H:%M} · {model_name}. "
            f"Actual T/RH/AH = {row['T']:.1f}/{row['RH']:.1f}/{row['AH']:.4f}; "
            f"scenario = {weather['T']:.1f}/{weather['RH']:.1f}/{weather['AH']:.4f}. "
            f"Both predictions use this same hour and previous-hour readings. "
            f"Scenario minus actual-weather prediction — {deltas}. "
            f"Observed minus actual-weather prediction, absolute error at this training-period hour — {local_errors}. "
            f"Held-out test MAE — {test_mae}. "
            f"Nearest observed training T/RH/AH triplet: {nearest:.3f} training-IQR units; "
            "this checks joint weather support, not the plausibility of its pairing with fixed sensors.")
    if outside:
        note += " Outside the training 1st–99th percentile range: " + ", ".join(outside) + "."
    else:
        note += " All three weather values are within the training 1st–99th percentile ranges."
    direction_text = "; ".join(f"{target}: {details['change']}" for target, details in directions.items())
    evidence = (f"Distance to A's G3 five-sensor median: actual-weather prediction "
                f"{original_distances['G3']:.2f}, scenario-weather prediction "
                f"{scenario_distances['G3']:.2f} full-data IQR units. "
                f"Attribute directions relative to G3 — {direction_text}. "
                f"Closest A representative median: actual-weather prediction {original_order[0]} "
                f"({original_distances[original_order[0]]:.2f}), scenario {scenario_order[0]} "
                f"({scenario_distances[scenario_order[0]]:.2f}). "
                "Evidence insufficient to establish preserved or changed group membership. "
                "These are descriptive distances; the models do not consistently outperform "
                "previous-hour persistence across all five sensors on held-out data.")
    return (make_scenario_chart(observed, changed, baseline),
            make_distance_chart(original_distances, scenario_distances), note, evidence)


@app.callback(Output("weather-s4-chart", "figure"), Output("weather-eight-chart", "figure"),
              Output("weather-five-chart", "figure"),
              Output("weather-case-info", "children"), Output("weather-range-info", "children"),
              Input("weather-index", "value"), Input("weather-model", "value"))
def update_weather_case(index, model_name):
    index = WEATHER_DEFAULT_INDEX if index is None else max(0, min(int(index), len(weather_pool)-1))
    model_name = model_name if model_name in weather_models else "Weather Linear Regression"
    candidate = weather_pool.iloc[index]
    actual_weather = pd.DataFrame([{k: float(NOV8_REFERENCE[k]) for k in WEATHER}])
    scenario_weather = {k: float(candidate[k]) for k in WEATHER}
    model = weather_models[model_name]
    baseline = model.predict(actual_weather)[0]
    changed = model.predict(pd.DataFrame([scenario_weather]))[0]
    observed = NOV8_REFERENCE[list(TARGETS)].to_numpy(dtype=float)
    s4 = "PT08.S4(NO2)"
    s4_index = TARGETS.index(s4)
    metric = weather_report["evaluation"]["test"]["models"][model_name][s4]
    g2 = selected_ranges["G2"]
    count = weather_counts[model_name]["within_selected_G2_S4_T_AH_ranges"]
    case_info = (f"Nov 8 13:00 actual T/RH/AH: {actual_weather.iloc[0]['T']:.1f} / "
                 f"{actual_weather.iloc[0]['RH']:.1f} / {actual_weather.iloc[0]['AH']:.4f}. "
                 f"Alternative: {candidate.DateTime:%Y-%m-%d %H:%M} ({candidate.RecordID}), "
                 f"T/RH/AH {scenario_weather['T']:.1f} / {scenario_weather['RH']:.1f} / "
                 f"{scenario_weather['AH']:.4f}. S4 observed {observed[s4_index]:.1f}; "
                 f"model at actual weather {baseline[s4_index]:.1f}; alternative "
                 f"{changed[s4_index]:.1f}. Held-out S4 MAE {metric['mae']:.1f}, "
                 f"R² {metric['r2']:.3f}.")
    range_info = (f"A-selected G2 observed ranges: S4 {g2[s4]['min']:.1f}–{g2[s4]['max']:.1f}; "
                  f"T {g2['T']['min']:.1f}–{g2['T']['max']:.1f}; "
                  f"AH {g2['AH']['min']:.4f}–{g2['AH']['max']:.4f}. "
                  f"{count}/{len(weather_pool)} real weather triplets give model S4, T and AH "
                  "inside those selected-record ranges. This is a descriptive scenario count.")
    return (make_weather_s4(observed, baseline, changed),
            make_weather_parallel(observed, baseline, changed, scenario_weather),
            make_weather_five_sensor_comparison(observed, baseline, changed),
            case_info, range_info)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8051)