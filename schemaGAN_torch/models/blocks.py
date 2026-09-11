"""Convolutional building blocks shared by the generator and the discriminator."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

Pair = tuple[int, int]


def same_padding(kernel_size: Pair, stride: Pair) -> tuple[int, int, int, int]:
    """Padding ``(left, right, top, bottom)`` reproducing TensorFlow's ``padding='same'``.

    Valid for input sizes that are a multiple of the stride, which is the case
    for every layer of schemaGAN.
    """
    pads = []
    for size, step in zip(kernel_size, stride):
        total = max(size - step, 0)
        pads.append((total // 2, total - total // 2))
    (top, bottom), (left, right) = pads
    return left, right, top, bottom


class SameConv2d(nn.Module):
    """Convolution whose output is the input size divided by the stride."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Pair = (4, 4), stride: Pair = (2, 2)):
        super().__init__()
        self.pad = nn.ZeroPad2d(same_padding(kernel_size, stride))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad the input as TensorFlow would, then convolve it."""
        return self.conv(self.pad(x))


class SameConvTranspose2d(nn.Module):
    """Transposed convolution whose output is the input size times the stride."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Pair = (4, 4), stride: Pair = (2, 2)):
        super().__init__()
        self.stride = stride
        left, _, top, _ = same_padding(kernel_size, stride)
        self.offset = (top, left)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample, then crop the border that TensorFlow would have dropped."""
        height, width = x.shape[-2:]
        y = self.deconv(x)
        top, left = self.offset
        return y[..., top : top + height * self.stride[0], left : left + width * self.stride[1]]


class NoiseDropout(nn.Module):
    """Dropout that can stay active at inference time.

    Pix2pix-style generators use dropout as their only source of noise and keep
    it enabled when generating, which is what ``always_on`` reproduces.
    """

    def __init__(self, p: float, always_on: bool = False):
        super().__init__()
        self.p = p
        self.always_on = always_on

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Drop activations while training, or always when ``always_on``."""
        return F.dropout(x, self.p, training=self.training or self.always_on)

    def extra_repr(self) -> str:
        """Show the rate and whether the layer survives ``eval()``."""
        return f"p={self.p}, always_on={self.always_on}"


def _batch_norm(channels: int, batch_stats_only: bool) -> nn.BatchNorm2d:
    """Batch norm that optionally normalises with batch statistics everywhere."""
    # ``track_running_stats=False`` makes the layer normalise with the statistics
    # of the current batch in eval mode too, mirroring Keras ``training=True``.
    return nn.BatchNorm2d(channels, track_running_stats=not batch_stats_only)


class EncoderBlock(nn.Module):
    """Downsampling convolution, optional batch norm and leaky ReLU.

    Args:
        in_channels: Channels of the input feature map.
        out_channels: Channels produced by the convolution.
        stride: Downsampling factor per axis; ``(1, 2)`` only shrinks the width.
        kernel_size: Convolution kernel.
        batchnorm: Add batch norm; the first block of a network omits it.
        batch_stats_only: Normalise with batch statistics in eval mode too.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Pair = (2, 2),
        kernel_size: Pair = (4, 4),
        batchnorm: bool = True,
        batch_stats_only: bool = False,
    ):
        super().__init__()
        self.conv = SameConv2d(in_channels, out_channels, kernel_size, stride)
        self.norm = _batch_norm(out_channels, batch_stats_only) if batchnorm else None
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample the feature map by the block stride."""
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.activation(x)


class DecoderBlock(nn.Module):
    """Upsampling convolution, batch norm, optional dropout, skip merge and ReLU.

    Args:
        in_channels: Channels of the input feature map.
        out_channels: Channels produced before the skip connection is merged.
        stride: Upsampling factor per axis; ``(1, 2)`` only grows the width.
        kernel_size: Convolution kernel.
        dropout: Dropout rate; ``0`` removes the layer.
        always_dropout: Keep dropout active outside training.
        batchnorm: Add batch norm.
        batch_stats_only: Normalise with batch statistics in eval mode too.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Pair = (2, 2),
        kernel_size: Pair = (4, 4),
        dropout: float = 0.0,
        always_dropout: bool = False,
        batchnorm: bool = True,
        batch_stats_only: bool = False,
    ):
        super().__init__()
        self.deconv = SameConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        self.norm = _batch_norm(out_channels, batch_stats_only) if batchnorm else None
        self.dropout = NoiseDropout(dropout, always_dropout) if dropout > 0.0 else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        """Upsample the feature map and concatenate the matching skip connection."""
        x = self.deconv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.activation(x)


def init_conv_weights(module: nn.Module, std: float = 0.02) -> nn.Module:
    """Initialise every convolution with ``N(0, std)`` weights, as in the paper.

    Returns:
        The module, so the call can be chained.
    """
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    return module
