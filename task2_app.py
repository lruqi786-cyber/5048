"""
COMP5048 Group Assignment - Task 2: Temporal Analysis module (Plotly Dash)

Run:
    pip install dash pandas openpyxl
    python task2_app.py
Then open http://127.0.0.1:8050 in a browser.

Inputs
- DATA_PATH: A's cleaned sensor_response.xlsx (missing values already NaN, never imputed).
- GROUP_PATH (optional): CSV with columns RecordID, Group exported from Task 1 brushing.
  If it does not exist, a TEMPORARY rule-based grouping is used as a placeholder.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, dcc, html

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "sensor_response.xlsx"
GROUP_PATH = "groups.csv"  # from Task 1 (RecordID, Group); optional

ATTRS = ["PT08.S1(CO)", "C6H6(GT)", "PT08.S3(NOx)", "PT08.S4(NO2)",
         "PT08.S5(O3)", "T", "RH", "AH"]
INVERTED = {"PT08.S3(NOx)"}  # lower reading = more NOx, so flip this axis

# Temporary placeholder grouping until Task 1 groups arrive
TEMP_RULE_ATTR, TEMP_RULE_THRESHOLD = "PT08.S1(CO)", 1200

GROUP_COLOURS = {"G1": "#2563eb", "G2": "#f28e2b", "G3": "#9b3fc4",  # match Task 1 colours
                 "G4": "#9467bd", "G5": "#ff7f0e", "Unassigned": "#bbbbbb"}
MISSING_BG = "#d9d9d9"  # grey background = missing data shows through

# ---------------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA_PATH)
df["DateTime"] = pd.to_datetime(df["DateTime"])
df["Date"] = df["DateTime"].dt.date
df["YearMonth"] = df["DateTime"].dt.to_period("M").astype(str)  # 2004-03 ... 2005-04

if os.path.exists(GROUP_PATH):
    groups = pd.read_csv(GROUP_PATH)[["RecordID", "Group"]]
    df = df.merge(groups, on="RecordID", how="left")
    GROUP_SOURCE = "Task 1 brushing (groups.csv)"
else:
    df["Group"] = np.where(df[TEMP_RULE_ATTR] > TEMP_RULE_THRESHOLD, "G1", "G2")
    GROUP_SOURCE = f"TEMPORARY rule: {TEMP_RULE_ATTR} > {TEMP_RULE_THRESHOLD} = G1, else G2"
# Records with missing attribute values cannot be placed in a group; keep them visible as Unassigned
df.loc[df[ATTRS].isna().any(axis=1), "Group"] = np.nan
df["Group"] = df["Group"].fillna("Unassigned")

MONTHS = sorted(df["YearMonth"].unique())
GROUPS = sorted(g for g in df["Group"].unique() if g != "Unassigned")

# Global min-max normalisation (whole year, NOT per month) so panels are comparable
norm = df[ATTRS].copy()
for a in ATTRS:
    lo, hi = df[a].min(), df[a].max()
    norm[a] = (df[a] - lo) / (hi - lo)
    if a in INVERTED:
        norm[a] = 1 - norm[a]


SHORT = {"PT08.S1(CO)": "S1(CO)", "C6H6(GT)": "C6H6", "PT08.S3(NOx)": "S3(NOx)",
         "PT08.S4(NO2)": "S4(NO2)", "PT08.S5(O3)": "S5(O3)", "T": "T", "RH": "RH", "AH": "AH"}


def axis_label(a, short=False):
    name = SHORT[a] if short else a
    return f"{name} inv." if a in INVERTED else name


# ---------------------------------------------------------------------------
# 3. Figure builders
# ---------------------------------------------------------------------------
def calendar_heatmap(attr, months):
    """View 1 (overview): date x hour, colour = attribute value. Missing = grey gaps."""
    d = df[df["YearMonth"].isin(months)]
    grid = d.pivot_table(index="Hour", columns="Date", values=attr, aggfunc="first", dropna=False)
    grid = grid.reindex(range(24))
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=grid.columns, y=grid.index, colorscale="Viridis",
        colorbar=dict(title=attr),
        hovertemplate="%{x}<br>Hour %{y}:00<br>" + attr + " = %{z}<extra></extra>"))
    fig.update_layout(title=f"View 1 - Calendar heatmap of {attr} (grey = missing, not imputed)",
                      xaxis_title="Date", yaxis_title="Hour of day",
                      plot_bgcolor=MISSING_BG, height=360, margin=dict(t=50, b=40))
    return fig


def membership_heatmap(group, months):
    """View 2: month x hour, colour = share of hours in that cell belonging to the group."""
    d = df[df["YearMonth"].isin(months) & (df["Group"] != "Unassigned")]
    share = (d.assign(is_g=d["Group"] == group)
              .pivot_table(index="Hour", columns="YearMonth", values="is_g", aggfunc="mean")
              .reindex(index=range(24), columns=months))
    count = (d.pivot_table(index="Hour", columns="YearMonth", values="RecordID", aggfunc="count")
              .reindex(index=range(24), columns=months))
    fig = go.Figure(go.Heatmap(
        z=share.values * 100, x=share.columns, y=share.index,
        text=count.fillna(0).astype(int).astype(str).values,  # grouped hours per cell
        colorscale="Reds", zmin=0, zmax=100, colorbar=dict(title=f"% in {group}"),
        hovertemplate="%{x}, %{y}:00<br>%{z:.0f}% in " + group +
                      "<br>n = %{text} grouped hours<extra></extra>"))
    fig.update_layout(title=f"View 2 - Membership of {group} by month and hour (click a cell)",
                      xaxis_title="Month", yaxis_title="Hour of day", xaxis_type="category",
                      plot_bgcolor=MISSING_BG, height=420, margin=dict(t=50, b=40))
    return fig


def membership_bars(months):
    """View 2b: share of each group per month (stacked)."""
    d = df[df["YearMonth"].isin(months)]
    tab = pd.crosstab(d["YearMonth"], d["Group"], normalize="index").reindex(months) * 100
    fig = go.Figure()
    for g in GROUPS + ["Unassigned"]:
        if g in tab:
            fig.add_bar(x=tab.index, y=tab[g], name=g, marker_color=GROUP_COLOURS.get(g, "#555"),
                        hovertemplate="%{x}<br>" + g + ": %{y:.0f}%<extra></extra>")
    fig.update_layout(barmode="stack", title="View 2b - Group share per month",
                      xaxis_title="Month", yaxis_title="% of hours", xaxis_type="category",
                      height=300, margin=dict(t=50, b=40), legend_title="Group")
    return fig


def small_multiples(months, highlight=None, ncols=5):
    """View 3: one parallel-coordinates panel per month, lines coloured by group.
    Thin translucent lines = individual hours; thick line = group median.
    highlight = (YearMonth, Hour) clicked in View 2 -> those records drawn in black."""
    ncols = min(ncols, len(months))  # fewer months -> wider panels
    rows = int(np.ceil(len(months) / ncols))
    fig = make_subplots(rows=rows, cols=ncols, subplot_titles=months,
                        shared_yaxes=True, vertical_spacing=0.16, horizontal_spacing=0.025)
    x = list(range(len(ATTRS)))
    legend_done = set()
    for i, m in enumerate(months):
        r, c = i // ncols + 1, i % ncols + 1
        in_month = (df["YearMonth"] == m).values
        for g in GROUPS:
            sel = in_month & (df["Group"] == g).values
            block = norm[sel].values
            if len(block) == 0:
                continue
            # all lines of one group in a single trace, separated by None (fast)
            xs = np.tile(x + [None], len(block))
            ys = np.hstack([block, np.full((len(block), 1), np.nan)]).ravel()
            fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(width=0.5),
                                       opacity=0.06, marker_color=GROUP_COLOURS[g],
                                       hoverinfo="skip", showlegend=False), r, c)
            fig.add_trace(go.Scatter(x=x, y=np.nanmedian(block, axis=0), mode="lines",
                                     line=dict(width=3, color=GROUP_COLOURS[g]), name=g,
                                     legendgroup=g, showlegend=(g not in legend_done),
                                     hovertemplate=g + " median<br>%{text}: %{y:.2f}<extra></extra>",
                                     text=[axis_label(a) for a in ATTRS]), r, c)
            legend_done.add(g)
        if highlight and highlight[0] == m:
            sel = in_month & (df["Hour"] == highlight[1]).values & (df["Group"] != "Unassigned").values
            block = norm[sel].values
            if len(block):
                xs = np.tile(x + [None], len(block))
                ys = np.hstack([block, np.full((len(block), 1), np.nan)]).ravel()
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=1.2, color="black"),
                                         name=f"{m} {highlight[1]}:00", hoverinfo="skip"), r, c)
    fig.update_xaxes(tickvals=x, ticktext=[axis_label(a, short=True) for a in ATTRS], tickangle=-90,
                     tickfont=dict(size=9), showgrid=True, gridcolor="#999")
    fig.update_yaxes(range=[0, 1], showticklabels=False)
    fig.update_layout(title="View 3 - Monthly parallel coordinates (global min-max scale; "
                            "thick = group median, black = clicked month-hour)",
                      height=330 * rows, margin=dict(t=70, b=40), plot_bgcolor="white")
    return fig


# ---------------------------------------------------------------------------
# 4. Layout and interaction
# ---------------------------------------------------------------------------
app = Dash(__name__)
app.title = "Task 2 - Temporal Analysis"

app.layout = html.Div(style={"fontFamily": "Arial", "margin": "16px"}, children=[
    html.H2("Task 2 - Temporal Analysis of Pollution Groups"),
    html.P(f"Group source: {GROUP_SOURCE}", style={"color": "#a00"}),
    html.Div(style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}, children=[
        html.Div([html.Label("Colour attribute (View 1)"),
                  dcc.Dropdown(ATTRS, ATTRS[0], id="attr", clearable=False)], style={"width": "260px"}),
        html.Div([html.Label("Group (View 2)"),
                  dcc.Dropdown(GROUPS, GROUPS[0], id="group", clearable=False)], style={"width": "160px"}),
    ]),
    html.Label("Month range (filters all views)"),
    dcc.RangeSlider(0, len(MONTHS) - 1, 1, value=[0, len(MONTHS) - 1], id="months",
                    marks={i: m for i, m in enumerate(MONTHS)}),
    dcc.Graph(id="calendar"),
    html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
        dcc.Graph(id="membership", style={"flex": "1 1 600px"}),
        dcc.Graph(id="bars", style={"flex": "1 1 500px"}),
    ]),
    dcc.Graph(id="multiples"),
])


@app.callback(Output("calendar", "figure"), Output("membership", "figure"), Output("bars", "figure"),
              Input("attr", "value"), Input("group", "value"), Input("months", "value"))
def update_top(attr, group, rng):
    months = MONTHS[rng[0]:rng[1] + 1]
    return calendar_heatmap(attr, months), membership_heatmap(group, months), membership_bars(months)


@app.callback(Output("multiples", "figure"),
              Input("months", "value"), Input("membership", "clickData"))
def update_multiples(rng, click):
    months = MONTHS[rng[0]:rng[1] + 1]
    highlight = None
    if click:
        p = click["points"][0]
        highlight = (str(p["x"]), int(p["y"]))
    return small_multiples(months, highlight)


if __name__ == "__main__":
    app.run(debug=False)  # set True while developing; False hides the debug toolbar for screenshots
