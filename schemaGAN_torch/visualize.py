"""Figures for training diagnostics and prediction review.

Uses the Agg canvas directly so that plotting never needs a display.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def _as_grid(image) -> np.ndarray:
    """Reduce a tensor or array to the 2D grid that ``imshow`` expects."""
    array = np.asarray(image, dtype=float)
    while array.ndim > 2:
        array = array[0] if array.shape[0] == 1 else array.mean(axis=0)
    if array.ndim != 2:
        raise ValueError("expected an image that reduces to a 2D grid")
    return array


def _save(figure: Figure, path: str | Path) -> Path:
    """Render a figure to PNG, creating the parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_grids(
    panels: Sequence[tuple[str, object, float, float, str]],
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Stack ``(name, grid, vmin, vmax, cmap)`` panels into one figure.

    Returns:
        The path of the PNG that was written.
    """
    if not panels:
        raise ValueError("at least one panel is required")

    figure = Figure(figsize=(12, 2.1 * len(panels)), layout="constrained")
    axes = None
    for index, (name, grid, low, high, colormap) in enumerate(panels, start=1):
        axes = figure.add_subplot(len(panels), 1, index)
        image = axes.imshow(_as_grid(grid), cmap=colormap, vmin=low, vmax=high, aspect="auto")
        axes.set_title(name, fontsize=9)
        axes.set_ylabel("depth [px]", fontsize=8)
        figure.colorbar(image, ax=axes, fraction=0.02, pad=0.01)
    axes.set_xlabel("distance [px]", fontsize=8)
    if title:
        figure.suptitle(title)
    return _save(figure, path)


def plot_cross_sections(
    source,
    generated,
    target,
    path: str | Path,
    *,
    title: str | None = None,
    vmin: float = 0.0,
    vmax: float = 4.3,
    cmap: str = "viridis",
) -> Path:
    """Save an input / generated / target / absolute-error comparison figure.

    Args:
        source: The CPT-like input, in IC units.
        generated: The generator output, in IC units.
        target: The complete cross-section, in IC units.
        path: Destination PNG.
        title: Optional figure title.
        vmin: Lower bound of the colour scale.
        vmax: Upper bound of the colour scale.
        cmap: Colormap of the three IC panels.
    """
    source, generated, target = (_as_grid(x) for x in (source, generated, target))
    error = np.abs(target - generated)
    return plot_grids(
        [
            ("CPT-like input", source, vmin, vmax, cmap),
            ("Generated", generated, vmin, vmax, cmap),
            ("Target", target, vmin, vmax, cmap),
            ("Absolute error", error, 0.0, max(float(error.max()), 1e-6), "inferno"),
        ],
        path,
        title=title,
    )


def plot_history(
    history: Mapping[str, Sequence[float]],
    path: str | Path,
    *,
    keys: Sequence[str] | None = None,
    xlabel: str = "iteration",
) -> Path:
    """Plot the recorded loss/metric curves.

    Args:
        history: Mapping of metric name to the recorded values.
        path: Destination PNG.
        keys: Metrics to plot; defaults to every non-empty series.
        xlabel: Label of the shared horizontal axis.
    """
    keys = list(keys) if keys is not None else [k for k, v in history.items() if len(v)]
    if not keys:
        raise ValueError("history does not contain anything to plot")

    figure = Figure(figsize=(10, 2.5 * len(keys)), layout="constrained")
    for index, key in enumerate(keys, start=1):
        axes = figure.add_subplot(len(keys), 1, index)
        axes.plot(history[key], linewidth=1.0)
        axes.set_ylabel(key, fontsize=8)
        axes.grid(alpha=0.3)
    axes.set_xlabel(xlabel, fontsize=8)
    return _save(figure, path)


def plot_error_histogram(errors, path: str | Path, *, bins: int = 30, label: str = "MAE") -> Path:
    """Plot the distribution of a per-sample error metric.

    Args:
        errors: One error value per cross-section.
        path: Destination PNG.
        bins: Upper bound on the number of histogram bins.
        label: Name of the metric, used on the horizontal axis.
    """
    values = np.asarray(errors, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("no errors to plot")

    figure = Figure(figsize=(6, 4))
    axes = figure.add_subplot(1, 1, 1)
    axes.hist(values, bins=min(bins, max(values.size, 1)))
    axes.axvline(values.mean(), color="crimson", linestyle="--", label=f"mean = {values.mean():.4f}")
    axes.set_xlabel(label)
    axes.set_ylabel("count")
    axes.legend()
    return _save(figure, path)
