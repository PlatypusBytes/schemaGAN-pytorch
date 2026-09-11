"""Reconstruction metrics used to score generated cross-sections."""

from __future__ import annotations

import torch


def _flatten(tensor: torch.Tensor) -> torch.Tensor:
    """Collapse everything but the batch dimension."""
    if tensor.dim() == 0:
        raise ValueError("expected at least a one-dimensional tensor")
    return tensor.reshape(tensor.shape[0], -1) if tensor.dim() > 1 else tensor[None]


def _check(prediction: torch.Tensor, target: torch.Tensor) -> None:
    """Raise if the two tensors cannot be compared element-wise."""
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")


def absolute_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error per sample."""
    _check(prediction, target)
    return (_flatten(prediction) - _flatten(target)).abs().mean(dim=1)


def squared_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error per sample."""
    _check(prediction, target)
    return (_flatten(prediction) - _flatten(target)).pow(2).mean(dim=1)


def root_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root mean squared error per sample."""
    return squared_error(prediction, target).sqrt()


def per_sample_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """MAE, MSE and RMSE for every sample of a batch."""
    mse = squared_error(prediction, target)
    return {
        "mae": absolute_error(prediction, target),
        "mse": mse,
        "rmse": mse.sqrt(),
    }


def mean_absolute_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error over the whole batch."""
    return absolute_error(prediction, target).mean().item()


def mean_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Mean squared error over the whole batch."""
    return squared_error(prediction, target).mean().item()


def root_mean_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Root of the batch mean squared error."""
    return mean_squared_error(prediction, target) ** 0.5
