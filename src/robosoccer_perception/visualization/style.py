"""Shared plot styling: one small palette, used by role.

Colours follow the entity, never the panel: ground truth is always blue, the estimate always
orange, observations always aqua. These three hues stay distinguishable under the common forms
of colour blindness; every plot also labels its series, so colour is never the only cue.
"""
from __future__ import annotations

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
FIELD_LINE = "#a9a8a2"          # painted lines, drawn at their true width
FIELD_TURF = "#f1f3ee"          # carpet, kept pale so the data stands out
TRUTH = "#2a78d6"
ESTIMATE = "#eb6834"
OBSERVATION = "#1baf7a"
PARTICLE = "#52514e"
REFERENCE = "#8b8a85"           # thresholds and event markers


def style_axes(ax, grid: bool = True) -> None:
    """Recessive axes: light grid, no top/right spines, secondary-ink ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=3)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def new_figure(width: float, height: float):
    """A Matplotlib figure that needs no GUI backend (works on headless machines)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), dpi=110, facecolor=SURFACE)
    FigureCanvasAgg(figure)
    return figure
