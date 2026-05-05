"""Plotting helpers for research notebooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def apply_default_style() -> None:
    """Apply a readable default Matplotlib style for static notebook output."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (11, 5),
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.frameon": True,
        }
    )


def _prepare_plotly_frame(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    frame = data.to_frame() if isinstance(data, pd.Series) else data.copy()
    if frame.empty:
        raise ValueError("Cannot plot an empty time-series frame.")
    return frame.sort_index()


def _display_label(column: object, labels: Mapping[str, str] | None) -> str:
    name = str(column)
    return labels.get(name, name) if labels else name


def _add_start_date_slider(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(
        rangeslider={"visible": True, "thickness": 0.12},
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"count": 3, "label": "3Y", "step": "year", "stepmode": "backward"},
                {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ]
        },
    )
    return fig


def plot_time_series_with_start_slider(
    data: pd.Series | pd.DataFrame,
    *,
    title: str,
    yaxis_title: str,
    labels: Mapping[str, str] | None = None,
    horizontal_lines: Sequence[Mapping[str, object]] = (),
    height: int = 520,
) -> go.Figure:
    """Create an interactive Plotly time-series chart with a start-date slider."""
    frame = _prepare_plotly_frame(data)
    fig = go.Figure()
    for column in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                mode="lines",
                name=_display_label(column, labels),
            )
        )

    for line in horizontal_lines:
        fig.add_hline(
            y=line["y"],
            line_color=str(line.get("color", "black")),
            line_dash=str(line.get("dash", "dash")),
            line_width=float(line.get("width", 1)),
            annotation_text=str(line.get("label", "")),
            annotation_position=str(line.get("annotation_position", "top right")),
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        template="plotly_white",
        height=height,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 60, "r": 30, "t": 90, "b": 40},
    )
    return _add_start_date_slider(fig)


def plot_time_series_panels_with_start_slider(
    panels: Mapping[str, pd.Series | pd.DataFrame],
    *,
    title: str,
    yaxis_titles: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    height_per_panel: int = 300,
) -> go.Figure:
    """Create stacked time-series panels with a shared start-date slider."""
    if not panels:
        raise ValueError("Cannot plot time-series panels without any panels.")

    panel_items = [(panel_title, _prepare_plotly_frame(panel)) for panel_title, panel in panels.items()]
    fig = make_subplots(
        rows=len(panel_items),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[panel_title for panel_title, _ in panel_items],
        vertical_spacing=0.10,
    )

    for row, (panel_title, frame) in enumerate(panel_items, start=1):
        for column in frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame[column],
                    mode="lines",
                    name=_display_label(column, labels),
                    showlegend=len(frame.columns) > 1,
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(
            title_text=(yaxis_titles or {}).get(panel_title),
            row=row,
            col=1,
        )

    bottom_row = len(panel_items)
    fig.update_xaxes(
        rangeslider={"visible": True, "thickness": 0.08},
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"count": 3, "label": "3Y", "step": "year", "stepmode": "backward"},
                {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ]
        },
        row=bottom_row,
        col=1,
    )
    fig.update_layout(
        title=title,
        hovermode="x unified",
        template="plotly_white",
        height=max(420, height_per_panel * len(panel_items)),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 60, "r": 30, "t": 100, "b": 40},
    )
    return fig


def plot_stacked_area_with_start_slider(
    panels: Mapping[str, pd.DataFrame],
    *,
    title: str,
    yaxis_title: str,
    labels: Mapping[str, str] | None = None,
    height_per_panel: int = 340,
) -> go.Figure:
    """Create one or more stacked-area panels with a shared start-date slider."""
    if not panels:
        raise ValueError("Cannot plot stacked areas without any panels.")

    panel_items = [(panel_title, _prepare_plotly_frame(panel)) for panel_title, panel in panels.items()]
    fig = make_subplots(
        rows=len(panel_items),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[panel_title for panel_title, _ in panel_items],
        vertical_spacing=0.10,
    )

    for row, (_, frame) in enumerate(panel_items, start=1):
        for column in frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame[column],
                    mode="lines",
                    name=_display_label(column, labels),
                    stackgroup=f"panel-{row}",
                    hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>",
                    showlegend=row == 1,
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text=yaxis_title, row=row, col=1)

    bottom_row = len(panel_items)
    fig.update_xaxes(
        rangeslider={"visible": True, "thickness": 0.08},
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"count": 3, "label": "3Y", "step": "year", "stepmode": "backward"},
                {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ]
        },
        row=bottom_row,
        col=1,
    )
    fig.update_layout(
        title=title,
        hovermode="x unified",
        template="plotly_white",
        height=max(420, height_per_panel * len(panel_items)),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 60, "r": 30, "t": 100, "b": 40},
    )
    return fig
