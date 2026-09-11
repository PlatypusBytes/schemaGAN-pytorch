"""Tests for the diagnostic figures."""

import numpy as np
import pytest
import torch

from schemaGAN_torch.visualize import (
    plot_cross_sections,
    plot_error_histogram,
    plot_grids,
    plot_history,
)

from conftest import HEIGHT, WIDTH


@pytest.fixture
def grid():
    """A random cross-section in IC units."""
    return np.random.default_rng(0).uniform(0.0, 4.3, (HEIGHT, WIDTH))


class TestPlotCrossSections:
    """The four-panel input/generated/target/error figure."""

    def test_writes_a_png(self, tmp_path, grid):
        path = plot_cross_sections(grid, grid, grid, tmp_path / "figures" / "cs.png", title="cs_0")
        assert path.exists() and path.stat().st_size > 0

    def test_accepts_batched_channel_first_tensors(self, tmp_path, grid):
        tensor = torch.from_numpy(grid)[None, None]
        assert plot_cross_sections(tensor, tensor, tensor, tmp_path / "cs.png").exists()

    def test_rejects_something_that_is_not_an_image(self, tmp_path):
        with pytest.raises(ValueError, match="2D grid"):
            plot_cross_sections(np.zeros(4), np.zeros(4), np.zeros(4), tmp_path / "cs.png")


class TestPlotGrids:
    """The generic stacked-panel figure."""

    def test_writes_one_panel_per_entry(self, tmp_path, grid):
        panels = [("input", grid, 0.0, 4.3, "viridis"), ("generated", grid, 0.0, 4.3, "viridis")]
        assert plot_grids(panels, tmp_path / "grids.png").exists()

    def test_rejects_an_empty_figure(self, tmp_path):
        with pytest.raises(ValueError, match="at least one panel"):
            plot_grids([], tmp_path / "grids.png")


class TestPlotHistory:
    """The loss and metric curves."""

    def test_plots_the_selected_curves(self, tmp_path):
        history = {"d_loss": [1.0, 0.8, 0.6], "g_loss": [5.0, 4.0, 3.0], "unused": []}
        assert plot_history(history, tmp_path / "history.png", keys=["g_loss"]).exists()

    def test_skips_empty_series_by_default(self, tmp_path):
        assert plot_history({"d_loss": [1.0], "unused": []}, tmp_path / "history.png").exists()

    def test_rejects_an_empty_history(self, tmp_path):
        with pytest.raises(ValueError, match="does not contain"):
            plot_history({"unused": []}, tmp_path / "history.png")


class TestPlotErrorHistogram:
    """The distribution of a per-sample error metric."""

    def test_writes_a_png(self, tmp_path):
        assert plot_error_histogram([0.1, 0.2, 0.3], tmp_path / "hist.png").exists()

    def test_rejects_an_empty_series(self, tmp_path):
        with pytest.raises(ValueError, match="no errors"):
            plot_error_histogram([], tmp_path / "hist.png")
