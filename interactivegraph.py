"""
please run the foillowing in the terminal 

pip install dash plotly pandas
python interactivegraph.py

open locally @:  http://127.0.0.1:8050)

to end server ctrl c in temrinal 


"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, callback_context, dcc, html
from plotly.subplots import make_subplots

_DATA = Path(__file__).resolve().parent / "data"


def _first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"No file found among: {[str(p) for p in paths]}")


def load_prices(day: int) -> pd.DataFrame:
    path = _first_existing(
        [
            _DATA / f"prices_round_0_day_{day}.csv",
            _DATA / f"day_minus_{-day}_books.csv",
        ]
    )
    return pd.read_csv(path, sep=";")


def load_trades(day: int) -> pd.DataFrame | None:
    try:
        path = _first_existing(
            [
                _DATA / f"trades_round_0_day_{day}.csv",
                _DATA / f"day_minus_{-day}_trades.csv",
            ]
        )
    except FileNotFoundError:
        return None
    return pd.read_csv(path, sep=";")


def _product_column(trades: pd.DataFrame) -> str:
    if "symbol" in trades.columns:
        return "symbol"
    if "product" in trades.columns:
        return "product"
    raise ValueError("Trades file needs a 'symbol' or 'product' column.")


def decimate(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def build_figure(
    day: int,
    t_min: float,
    t_max: float,
    max_points: int,
    show_trades: bool,
    show_spread: bool,
    show_mid: bool,
    means_use_window: bool,
) -> go.Figure:
    book = load_prices(day)
    book = book.sort_values("timestamp")
    t_lo = book["timestamp"].min()
    t_hi = book["timestamp"].max()
    t_min = max(t_min, float(t_lo))
    t_max = min(t_max, float(t_hi))
    window = book[(book["timestamp"] >= t_min) & (book["timestamp"] <= t_max)]

    tomatoes = window[window["product"] == "TOMATOES"]
    emeralds = window[window["product"] == "EMERALDS"]

    t_full = book[book["product"] == "TOMATOES"]
    e_full = book[book["product"] == "EMERALDS"]
    mean_src_t = window if means_use_window else t_full
    mean_src_e = window if means_use_window else e_full

    t_bid_mean = mean_src_t["bid_price_1"].mean()
    t_ask_mean = mean_src_t["ask_price_1"].mean()
    e_bid_mean = mean_src_e["bid_price_1"].mean()
    e_ask_mean = mean_src_e["ask_price_1"].mean()

    tomatoes_p = decimate(tomatoes, max_points)
    emeralds_p = decimate(emeralds, max_points)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            f"TOMATOES — day {day} (max {max_points} pts / product in view)",
            f"EMERALDS — day {day}",
        ),
    )

    def add_order_book(
        row: int,
        dplot: pd.DataFrame,
        name_prefix: str,
        bid_mean: float,
        ask_mean: float,
    ) -> None:
        if dplot.empty:
            return
        ts = dplot["timestamp"]
        bid = dplot["bid_price_1"]
        ask = dplot["ask_price_1"]
        if show_spread:
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=ask,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=bid,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(99,110,250,0.12)",
                    line=dict(width=0),
                    name=f"{name_prefix} spread",
                    legendgroup=name_prefix,
                    hovertemplate="spread<extra></extra>",
                ),
                row=row,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=bid,
                mode="lines",
                name=f"{name_prefix} bid",
                legendgroup=name_prefix,
                line=dict(width=1.2),
                hovertemplate="t=%{x}<br>bid=%{y:.4f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=ask,
                mode="lines",
                name=f"{name_prefix} ask",
                legendgroup=name_prefix,
                line=dict(width=1.2),
                hovertemplate="t=%{x}<br>ask=%{y:.4f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        if show_mid and "mid_price" in dplot.columns:
            mid = decimate(dplot[["timestamp", "mid_price"]].dropna(), max_points)
            fig.add_trace(
                go.Scatter(
                    x=mid["timestamp"],
                    y=mid["mid_price"],
                    mode="lines",
                    name=f"{name_prefix} mid",
                    legendgroup=name_prefix,
                    line=dict(width=1, dash="dot"),
                    hovertemplate="t=%{x}<br>mid=%{y:.4f}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        fig.add_hline(
            y=bid_mean,
            line_dash="dash",
            line_color="gray",
            annotation_text="bid mean",
            annotation_position="right",
            row=row,
            col=1,
        )
        fig.add_hline(
            y=ask_mean,
            line_dash="dash",
            line_color="darkgray",
            annotation_text="ask mean",
            annotation_position="right",
            row=row,
            col=1,
        )

    add_order_book(1, tomatoes_p, "TOM", t_bid_mean, t_ask_mean)
    add_order_book(2, emeralds_p, "EME", e_bid_mean, e_ask_mean)

    trades = load_trades(day) if show_trades else None
    if trades is not None and not trades.empty:
        col = _product_column(trades)
        tw = trades[(trades["timestamp"] >= t_min) & (trades["timestamp"] <= t_max)]
        for row, prod, marker in (
            (1, "TOMATOES", "circle"),
            (2, "EMERALDS", "diamond"),
        ):
            sub = tw[tw[col] == prod]
            if sub.empty:
                continue
            qty = sub["quantity"].to_numpy().reshape(-1, 1) if "quantity" in sub.columns else None
            fig.add_trace(
                go.Scatter(
                    x=sub["timestamp"],
                    y=sub["price"],
                    mode="markers",
                    name=f"{prod[:3]} trades",
                    marker=dict(size=6, symbol=marker, opacity=0.75),
                    hovertemplate=(
                        "t=%{x}<br>px=%{y:.4f}<br>qty=%{customdata[0]}<extra></extra>"
                        if qty is not None
                        else "t=%{x}<br>px=%{y:.4f}<extra></extra>"
                    ),
                    customdata=qty,
                ),
                row=row,
                col=1,
            )

    fig.update_layout(
        height=820,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode="zoom",
    )
    fig.update_xaxes(title_text="timestamp", rangeslider=dict(visible=True), row=2, col=1)
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="price", row=2, col=1)

    return fig


def main() -> None:
    sample = load_prices(-1)
    t0 = float(sample["timestamp"].min())
    t1 = float(sample["timestamp"].max())

    app = Dash(__name__)
    app.title = "IMC Prosperity — interactive books"

    app.layout = html.Div(
        [
            html.H3("Order book explorer"),
            html.Div(
                [
                    html.Label("Day "),
                    dcc.Dropdown(
                        id="day",
                        options=[{"label": "Day -1", "value": -1}, {"label": "Day -2", "value": -2}],
                        value=-1,
                        clearable=False,
                        style={"width": "200px", "display": "inline-block", "verticalAlign": "middle"},
                    ),
                    html.Span(" — "),
                    html.Label("Max points / product (downsample for speed) "),
                    html.Div(
                        dcc.Slider(
                            id="max_points",
                            min=500,
                            max=8000,
                            step=500,
                            value=3500,
                            marks={500: "500", 3500: "3500", 8000: "8000"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        style={"width": "420px", "display": "inline-block", "verticalAlign": "middle"},
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Label("Timestamp window "),
                    dcc.RangeSlider(
                        id="t_range",
                        min=t0,
                        max=t1,
                        value=[t0, t1],
                        allowCross=False,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="flags",
                        options=[
                            {"label": " Overlay trades", "value": "trades"},
                            {"label": " Spread band (ask→bid fill)", "value": "spread"},
                            {"label": " Mid price (from file)", "value": "mid"},
                            {"label": " Means from visible window only", "value": "meanwin"},
                        ],
                        value=["trades", "spread"],
                        inline=True,
                    ),
                ],
                style={"marginBottom": "8px"},
            ),
            dcc.Graph(id="graph", figure=build_figure(-1, t0, t1, 3500, True, True, False, False)),
            html.P(
                "Tip: drag on the chart to zoom; double-click resets axes. "
                "Bottom range slider scrubs time. Legend clicks hide/show series.",
                style={"fontSize": "13px", "color": "#444"},
            ),
        ],
        style={"maxWidth": "1100px", "margin": "0 auto", "fontFamily": "system-ui, sans-serif"},
    )

    @callback(
        Output("graph", "figure"),
        Output("t_range", "min"),
        Output("t_range", "max"),
        Output("t_range", "value"),
        Input("day", "value"),
        Input("t_range", "value"),
        Input("max_points", "value"),
        Input("flags", "value"),
    )
    def update(day, t_range, max_points, flags):
        flags = flags or []
        book = load_prices(int(day))
        t_lo = float(book["timestamp"].min())
        t_hi = float(book["timestamp"].max())

        triggered = callback_context.triggered_id
        reset_window = triggered == "day" or triggered is None

        if reset_window or t_range is None or len(t_range) != 2:
            lo, hi = t_lo, t_hi
        else:
            lo, hi = float(t_range[0]), float(t_range[1])
            lo = max(lo, t_lo)
            hi = min(hi, t_hi)
            if hi < lo:
                lo, hi = t_lo, t_hi

        fig = build_figure(
            int(day),
            lo,
            hi,
            int(max_points),
            "trades" in flags,
            "spread" in flags,
            "mid" in flags,
            "meanwin" in flags,
        )
        return fig, t_lo, t_hi, [lo, hi]

    app.run(debug=True)


if __name__ == "__main__":
    main()
