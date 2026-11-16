"""Figures for the vision evaluation reports (needs matplotlib).

Same visual language as the localization plots (`visualization.style`): one hue per series,
recessive axes, values written on the marks so no figure relies on colour alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robosoccer_perception.visualization import style


def _heatmap(ax, counts: np.ndarray, rows: list[str], columns: list[str], row_title: str,
             column_title: str, normalise_columns: bool = True) -> None:
    """Counts as a single-hue heat map (light = few, dark = many), each cell annotated."""
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=0, keepdims=True)
    share = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0) \
        if normalise_columns else counts / max(counts.max(), 1.0)
    ax.imshow(share, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    for (i, j), value in np.ndenumerate(counts):
        ax.text(j, i, f"{int(value)}", ha="center", va="center", fontsize=10,
                color="white" if share[i, j] > 0.55 else style.TEXT)
    ax.set_xticks(range(len(columns)), columns, rotation=20, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel(column_title)
    ax.set_ylabel(row_title)
    style.style_axes(ax, grid=False)
    ax.set_xticks(np.arange(-0.5, len(columns)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=style.SURFACE, linewidth=2)       # gaps between cells
    ax.tick_params(which="minor", length=0)


def _recall_bars(ax, rows: list[dict[str, Any]], title: str, colour: str) -> None:
    """Horizontal recall bars, one per bin, annotated with recall and counts."""
    labels = [row["bin"] for row in rows]
    values = [0.0 if row["recall"] != row["recall"] else row["recall"] for row in rows]
    positions = np.arange(len(rows))[::-1]
    ax.barh(positions, values, height=0.6, color=colour)
    for y, row, value in zip(positions, rows, values):
        ax.text(min(value + 0.02, 1.02), y, f"{100 * value:.0f} %  ({row['detected']}/{row['total']})",
                va="center", fontsize=8.5, color=style.TEXT_SECONDARY)
    ax.set_yticks(positions, labels, fontsize=9)
    ax.set_xlim(0.0, 1.45)
    ax.set_xticks([0.0, 0.5, 1.0], ["0", "50 %", "100 %"])
    ax.set_title(title, fontsize=10, loc="left")
    style.style_axes(ax)
    ax.grid(axis="y", visible=False)


def plot_detection_report(result: dict[str, Any], output_dir) -> list[Path]:
    """``confusion_matrix.png`` and one ``recall_<class>.png`` per class."""
    output = Path(output_dir)
    written = []
    confusion = result["confusion_matrix"]
    figure = style.new_figure(6.4, 4.6)
    ax = figure.subplots()
    _heatmap(ax, np.asarray(confusion["counts"]), confusion["rows_predicted"],
             confusion["columns_true"], "Predicted", "True (colour: share of each column)")
    ax.set_title(f"Confusion matrix at confidence >= {result['conf_threshold']}", fontsize=11,
                 loc="left")
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png")
    written.append(output / "confusion_matrix.png")

    for class_name in result["per_class"]:
        rows = [r for r in result["stratified_recall"]
                if r["class"] == class_name and r["breakdown"] not in ("all", "frame mode")]
        groups = list(dict.fromkeys(r["breakdown"] for r in rows))
        if not groups:
            continue
        heights = [0.35 * sum(r["breakdown"] == g for r in rows) + 0.6 for g in groups]
        figure = style.new_figure(6.8, sum(heights) + 0.5)
        axes = figure.subplots(len(groups), 1, squeeze=False,
                               gridspec_kw={"height_ratios": heights})[:, 0]
        for ax, group in zip(axes, groups):
            _recall_bars(ax, [r for r in rows if r["breakdown"] == group],
                         f"{class_name} recall by {group}", style.TRUTH)
        figure.tight_layout()
        path = output / f"recall_{class_name}.png"
        figure.savefig(path)
        written.append(path)
    return written


def plot_segmentation_report(result: dict[str, Any], output_dir) -> list[Path]:
    """``confusion_matrix.png`` (pixels) and ``line_by_distance.png``."""
    output = Path(output_dir)
    written = []
    names = result["classes"]
    figure = style.new_figure(5.6, 4.2)
    ax = figure.subplots()
    counts = np.asarray(result["confusion_matrix"]["counts"], dtype=float)
    totals = counts.sum(axis=0, keepdims=True)
    share = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    ax.imshow(share, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    for (i, j), value in np.ndenumerate(share):
        ax.text(j, i, f"{100 * value:.1f} %", ha="center", va="center", fontsize=10,
                color="white" if value > 0.55 else style.TEXT)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("True class (each column sums to 100 %)")
    ax.set_ylabel("Predicted class")
    ax.set_title("Pixel confusion matrix", fontsize=11, loc="left")
    style.style_axes(ax, grid=False)
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png")
    written.append(output / "confusion_matrix.png")

    rows = result.get("line_by_distance") or []
    if rows:
        figure = style.new_figure(6.8, 3.4)
        ax = figure.subplots()
        positions = np.arange(len(rows))
        width = 0.36
        for offset, key, colour, label in ((-width / 2 - 0.01, "recall", style.TRUTH, "Recall"),
                                           (width / 2 + 0.01, "precision", style.ESTIMATE, "Precision")):
            values = [0.0 if row[key] is None else row[key] for row in rows]
            ax.bar(positions + offset, values, width=width, color=colour, label=label)
            for x, value, row in zip(positions + offset, values, rows):
                if row[key] is not None:
                    ax.text(x, value + 0.015, f"{100 * value:.0f}", ha="center", fontsize=8.5,
                            color=style.TEXT_SECONDARY)
        ax.set_xticks(positions, [row["bin"] for row in rows])
        ax.set_ylim(0.0, 1.12)
        ax.set_yticks([0.0, 0.5, 1.0], ["0", "50 %", "100 %"])
        ax.set_xlabel("Distance of the line pixel from the camera, along the ground")
        ax.set_title("Field-line pixels: recall and precision by distance (values in %)",
                     fontsize=11, loc="left")
        style.style_axes(ax)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right", fontsize=9, frameon=False, ncols=2)
        figure.tight_layout()
        figure.savefig(output / "line_by_distance.png")
        written.append(output / "line_by_distance.png")
    return written


def plot_training_history(history: list[dict[str, Any]], path, series: dict[str, list[str]],
                          title: str) -> Path:
    """Small multiples over epochs: one panel per entry of ``series`` (panel title -> keys)."""
    colours = [style.TRUTH, style.ESTIMATE, style.OBSERVATION, style.PARTICLE]
    figure = style.new_figure(4.4 * len(series), 3.2)
    axes = figure.subplots(1, len(series), squeeze=False)[0]
    epochs = [row["epoch"] for row in history]
    for ax, (panel, keys) in zip(axes, series.items()):
        for key, colour in zip(keys, colours):
            values = [row.get(key) for row in history]
            ax.plot(epochs, values, color=colour, linewidth=2, label=key)
            if values and values[-1] is not None:
                ax.annotate(f"{values[-1]:.3f}", (epochs[-1], values[-1]), xytext=(4, 0),
                            textcoords="offset points", fontsize=8.5, va="center",
                            color=style.TEXT_SECONDARY)
        ax.set_title(panel, fontsize=10, loc="left")
        ax.set_xlabel("Epoch")
        style.style_axes(ax)
        if len(keys) > 1:
            ax.legend(fontsize=8.5, frameon=False)
        ax.margins(x=0.12)
    figure.suptitle(title, fontsize=11, x=0.01, ha="left", color=style.TEXT)
    figure.tight_layout()
    figure.savefig(path)
    return Path(path)
