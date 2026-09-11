"""Tests for the reconstruction metrics."""

import pytest
import torch

from schemaGAN_torch.metrics import (
    absolute_error,
    mean_absolute_error,
    mean_squared_error,
    per_sample_metrics,
    root_mean_squared_error,
    root_squared_error,
    squared_error,
)


@pytest.fixture
def pair():
    """Two samples: the first is off by ``[1, 3]``, the second is exact."""
    prediction = torch.tensor([[[[1.0, 3.0]]], [[[0.0, 0.0]]]])
    target = torch.tensor([[[[0.0, 0.0]]], [[[0.0, 0.0]]]])
    return prediction, target


class TestPerSampleErrors:
    """Errors reported once per cross-section."""

    def test_absolute_error_averages_over_each_sample(self, pair):
        assert absolute_error(*pair).tolist() == [2.0, 0.0]

    def test_squared_error_averages_over_each_sample(self, pair):
        assert squared_error(*pair).tolist() == [5.0, 0.0]

    def test_root_squared_error_is_the_square_root_of_the_squared_error(self, pair):
        assert root_squared_error(*pair).tolist() == pytest.approx([5.0**0.5, 0.0])

    def test_per_sample_metrics_reports_one_value_per_sample(self, pair):
        metrics = per_sample_metrics(*pair)
        assert set(metrics) == {"mae", "mse", "rmse"}
        assert all(value.shape == (2,) for value in metrics.values())

    def test_a_perfect_prediction_has_no_error(self):
        x = torch.rand(3, 1, 8, 8)
        assert absolute_error(x, x).sum() == 0.0

    @pytest.mark.parametrize("function", [absolute_error, squared_error, root_squared_error])
    def test_rejects_mismatched_shapes(self, function):
        with pytest.raises(ValueError, match="shape mismatch"):
            function(torch.zeros(2, 1, 4, 4), torch.zeros(2, 1, 4, 8))


class TestAggregateErrors:
    """Errors reduced to a single number over the batch."""

    def test_mean_absolute_error_averages_the_batch(self, pair):
        assert mean_absolute_error(*pair) == pytest.approx(1.0)

    def test_mean_squared_error_averages_the_batch(self, pair):
        assert mean_squared_error(*pair) == pytest.approx(2.5)

    def test_root_mean_squared_error_is_the_root_of_the_mean(self, pair):
        assert root_mean_squared_error(*pair) == pytest.approx(2.5**0.5)

    def test_matches_a_plain_torch_computation(self):
        prediction, target = torch.rand(4, 1, 8, 16), torch.rand(4, 1, 8, 16)
        assert mean_absolute_error(prediction, target) == pytest.approx(
            (prediction - target).abs().mean().item(), rel=1e-6
        )
