# app_dash.py
import socket
from typing import Dict, List

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.express as px

import themes_core as core  # <-- your GUI-free logic

APP_TITLE = "CC Themes (Mobile)"
COLOR_SCALE = core.COLOR_SCALE  # keep visual consistency

app = Dash(__name__)
server = app.server  # for hosting later if desired

def _make_themes_figure(df: pd.DataFrame, days: int) -> "plotly.graph_objs._figure.Figure":
    """Build the top-level themes figure (bars, arrows, hover)."""
    fig = px.bar(
        df,
        y=df.index,
        x="avg_return",
        title=f"Themes ({days}-Day Avg % Change)",
        labels={"y": "Theme", "avg_return": f"{days}-Day Avg %"},
        color="avg_return",
        color_continuous_scale=COLOR_SCALE,
        orientation="h",
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=60, r=30, t=60, b=40),
        xaxis_title="Avg % Change",
        yaxis_title="Theme",
    )
    # Keep sort & place hottest at the TOP (your existing pattern)
    fig.update_yaxes(categoryorder="array", categoryarray=df.index.tolist())
    fig.update_yaxes(autorange="reversed")
    # Replace tick text with labels containing ▲/▼/→ and 🔶 NEW:
    fig.update_yaxes(
        tickmode="array",
        tickvals=df.index.tolist(),
        ticktext=df["label"].tolist(),
    )
    # Visual split at 0%
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    # Tooltip: 1-decimal bar value + your top-4 movers today
    fig.update_traces(
        customdata=df["hover"],
        hovertemplate="<b>%{y}</b><br>Avg: %{x:.1f}%<br>%{customdata}<extra></extra>",
    )
    return fig

def _make_stocks_figure(theme_key: str, rows: List[Dict], today_min: float, today_max: float):
    """Build the drilldown figure for one theme using today's % change."""
    # rows: [{ "ticker": T, "today": float, "avg_days": float|None }, ...]
    clean = [r for r in rows if isinstance(r.get("today"), (int, float))]
    clean.sort(key=lambda r: r["today"], reverse=True)
    if not clean:
        # Placeholder empty chart
        fig = px.bar(
            pd.DataFrame({"Today %": [0.0]}, index=["No intraday data"]),
            orientation="h",
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(l=60, r=30, t=20, b=40),
            xaxis_title="Today %",
            yaxis_title="Ticker",
            paper_bgcolor="#111",
            plot_bgcolor="#111",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        return fig

    y = [r["ticker"] for r in clean]
    x = [r["today"] for r in clean]
    hover = [f"<b>{r['ticker']}</b><br>Today: {r['today']:+.1f}%" for r in clean]

    fig = px.bar(
        x=x, y=y, orientation="h",
        color=x, color_continuous_scale=COLOR_SCALE,
        labels={"x": "Today %", "y": "Ticker"},
    )
    fig.update_traces(
        hovertext=hover,
        hoverinfo="text",
        marker_line_color="rgba(0,0,0,0)",
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=60, r=30, t=20, b=40),
        paper_bgcolor="#111",
        plot_bgcolor="#111",
    )
    fig.update_xaxes(tickformat=".1f")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=y,
        autorange="reversed",  # hottest at top
    )
    # Stabilize colors using global min/max from all tickers
    fig.update_traces(marker={"cmin": today_min, "cmax": today_max})
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(title_text=f"{theme_key} — Stocks (Today’s % Change)")
    return fig

# -------- Layout --------
app.layout = html.Div(
    style={"maxWidth": "980px", "margin": "0 auto", "padding": "12px"},
    children=[
        html.H2(APP_TITLE),
        html.Div(
            style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
            children=[
                html.Label("Days:"),
                dcc.Input(id="days", type="number", value=core.DEFAULT_DAYS,
                          min=2, max=30, step=1, style={"width": "90px"}),
                dcc.Checklist(
                    id="discover",
                    options=[{"label": " Include 🔶 NEW (auto-discovered)", "value": "on"}],
                    value=["on"],  # default ON
                    style={"marginLeft": "12px"},
                ),
                html.Button("Refresh", id="btn-refresh", n_clicks=0),
                html.Button("↩ Back to Themes", id="btn-back", n_clicks=0, disabled=True),
            ],
        ),
        html.Div(id="status", style={"fontSize": "12px", "opacity": 0.7, "marginTop": "6px"}),
        dcc.Graph(id="themes-chart", style={"height": "68vh"}),
        dcc.Graph(id="stocks-chart", style={"height": "68vh", "display": "none"}),

        # Client-side state
        dcc.Store(id="store-drilldown"),      # { theme: [ {ticker, today, avg_days}, ... ], ... }
        dcc.Store(id="store-today-minmax"),   # [min, max]
        dcc.Store(id="store-view", data="themes"),
        dcc.Store(id="store-selected-theme"),
    ],
)

# -------- Callbacks --------
@app.callback(
    Output("themes-chart", "figure"),
    Output("store-drilldown", "data"),
    Output("store-today-minmax", "data"),
    Output("status", "children"),
    Output("store-view", "data"),
    Output("store-selected-theme", "data"),
    Output("btn-back", "disabled"),
    Input("btn-refresh", "n_clicks"),
    State("days", "value"),
    State("discover", "value"),
    prevent_initial_call=False,
)
def refresh_data(n, days_val, discover_vals):
    """Load data via themes_core and render the top-level themes chart."""
    days = int(days_val or core.DEFAULT_DAYS)
    discover = ("on" in (discover_vals or []))
    df, themes, prices_2w, prices_curr, drilldown, today_minmax = core.get_hot_themes(
        days=days, discover=discover
    )
    if df.empty:
        # return an empty placeholder chart
        empty_fig = px.bar(pd.DataFrame({"avg_return": [0.0]}, index=["No data"]),
                           x="avg_return", y=["No data"], orientation="h",
                           title="No data")
        return empty_fig, {}, [0.0, 0.0], "No data", "themes", None, True

    fig = _make_themes_figure(df, days)
    status = f"Updated · days={days} · discover={'ON' if discover else 'OFF'}"
    return fig, drilldown, list(today_minmax), status, "themes", None, True

@app.callback(
    Output("store-selected-theme", "data"),
    Output("store-view", "data"),
    Output("btn-back", "disabled"),
    Input("themes-chart", "clickData"),
    State("store-view", "data"),
    prevent_initial_call=True,
)
def on_theme_click(clickData, view):
    """Switch to stocks view when a bar is clicked."""
    if not clickData or not clickData.get("points"):
        return no_update, no_update, no_update
    theme_key = clickData["points"][0]["y"]  # y is the index (theme key)
    return theme_key, "stocks", False

@app.callback(
    Output("stocks-chart", "figure"),
    Output("stocks-chart", "style"),
    Output("themes-chart", "style"),
    Input("store-view", "data"),
    State("store-selected-theme", "data"),
    State("store-drilldown", "data"),
    State("store-today-minmax", "data"),
    prevent_initial_call=True,
)
def update_stocks_view(view, theme_key, drilldown, minmax):
    """Show/hide the stocks chart and build it for the selected theme."""
    if view != "stocks":
        # Hide stocks, show themes
        return no_update, {"display": "none"}, {"height": "68vh", "display": "block"}

    rows = (drilldown or {}).get(theme_key, [])
    today_min, today_max = (minmax or [0.0, 0.0])
    fig = _make_stocks_figure(theme_key, rows, today_min, today_max)

    # Show stocks, hide themes
    return fig, {"height": "68vh", "display": "block"}, {"display": "none"}

@app.callback(
    Output("store-view", "data"),
    Output("btn-back", "disabled"),
    Input("btn-back", "n_clicks"),
    prevent_initial_call=True,
)
def go_back(n):
    """Back button: return to themes view."""
    if n:
        return "themes", True
    return no_update, no_update

if __name__ == "__main__":
    # Helpful LAN IP printout for phone access on same Wi-Fi
    try:
        hostname = socket.gethostname()
        lan_ip = socket.gethostbyname(hostname)
    except Exception:
        lan_ip = "0.0.0.0"

    print(f"\nOpen on this Mac:     http://127.0.0.1:8050")
    print(f"Open on your iPhone:  http://{lan_ip}:8050  (same Wi-Fi)\n")

    # Run web server
    app.run(host="0.0.0.0", port=8050, debug=False)
