"""Hooks for combining schemaGAN with an auto-encoder/decoder.

The schemaGAN generator works on ``32 x 512`` cross-sections. To condition it
on CPT profiles with a finer depth resolution, a cross-section of any size is
compressed by :class:`ConvAutoencoder` into a ``(latent_channels, 32, 512)``
code, the generator runs in that code space and the decoder expands the
generated code back to the full resolution. :class:`AutoencoderGenerator` wires
the three pieces into a single module that
:class:`~schemaGAN_torch.schemagan.SchemaGAN` accepts as a drop-in generator,
and :func:`train_autoencoder` pre-trains the auto-encoder on its own.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .blocks import Pair, SameConv2d, SameConvTranspose2d, init_conv_weights


def _prime_factors(value: int) -> list[int]:
    """Prime factors of ``value`` in increasing order; ``1`` gives an empty list."""
    factors: list[int] = []
    divisor = 2
    while value > 1:
        while value % divisor == 0:
            factors.append(divisor)
            value //= divisor
        divisor += 1
    return factors


def compression_strides(image_shape: Pair, latent_shape: Pair) -> list[Pair]:
    """Per-level ``(stride_h, stride_w)`` pairs shrinking ``image_shape`` to ``latent_shape``.

    The compression factor of every axis is split into its prime factors,
    smallest first, and the two axes are zipped level by level; the axis with
    fewer factors is padded with a stride of 1. ``(320, 512) -> (32, 512)`` gives
    ``[(2, 1), (5, 1)]`` and ``(3200, 51200) -> (32, 512)`` gives
    ``[(2, 2), (2, 2), (5, 5), (5, 5)]``.

    Raises:
        ValueError: If an axis of the image is not a multiple of the latent axis.
    """
    factors: list[list[int]] = []
    for size, target in zip(image_shape, latent_shape):
        if size <= 0 or target <= 0 or size % target:
            raise ValueError(f"image shape {tuple(image_shape)} is not a multiple of {tuple(latent_shape)}")
        factors.append(_prime_factors(size // target))
    levels = max(len(axis) for axis in factors)
    if levels == 0:
        return [(1, 1)]
    return [
        (factors[0][level] if level < len(factors[0]) else 1, factors[1][level] if level < len(factors[1]) else 1)
        for level in range(levels)
    ]


class ConvAutoencoder(nn.Module):
    """Convolutional auto-encoder that compresses cross-sections into a latent grid.

    Every level shrinks the feature map by its ``(stride_h, stride_w)`` pair, so
    the two axes can be compressed by different factors; the depth axis of a
    cross-section usually needs far more compression than the horizontal one.
    The latent code is bounded by ``tanh`` so it lives in the same ``[-1, 1]``
    range as the output of the schemaGAN generator, which lets the generator
    produce codes the decoder has been trained on.

    Args:
        in_channels: Channels of a cross-section.
        base_filters: Filters of the first encoder level; deeper levels double it.
        latent_channels: Channels of the latent feature map.
        strides: One ``(stride_h, stride_w)`` pair per level, see
            :func:`compression_strides` and :meth:`for_geometry`.
        column_wise: Never mix neighbouring columns on a level whose horizontal
            stride is 1, i.e. encode every column as an independent 1D profile.
            This keeps the empty columns of a CPT-like input from leaking into
            the measured ones, so one auto-encoder serves sparse and dense
            cross-sections alike.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        latent_channels: int = 64,
        strides: Sequence[Pair] = ((2, 2), (2, 2), (2, 2)),
        column_wise: bool = False,
    ) -> None:
        super().__init__()
        self.strides = [(int(stride[0]), int(stride[1])) for stride in strides]
        if not self.strides:
            raise ValueError("strides must hold at least one level")
        if any(s < 1 for stride in self.strides for s in stride):
            raise ValueError("every stride must be at least 1")
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.column_wise = column_wise

        levels = len(self.strides)
        encoder: list[nn.Module] = []
        channels = in_channels
        for level, stride in enumerate(self.strides):
            last = level == levels - 1
            out_channels = latent_channels if last else base_filters * 2**level
            encoder.append(SameConv2d(channels, out_channels, self._kernel(stride), stride))
            encoder.append(nn.Tanh() if last else nn.LeakyReLU(0.2, inplace=True))
            channels = out_channels
        self.encoder = nn.Sequential(*encoder)

        decoder: list[nn.Module] = []
        for level in reversed(range(levels)):
            stride = self.strides[level]
            out_channels = in_channels if level == 0 else base_filters * 2 ** (level - 1)
            decoder.append(SameConvTranspose2d(channels, out_channels, self._kernel(stride), stride))
            decoder.append(nn.Tanh() if level == 0 else nn.ReLU(inplace=True))
            channels = out_channels
        self.decoder = nn.Sequential(*decoder)

        init_conv_weights(self)

    @classmethod
    def for_geometry(
        cls, image_shape: Pair, latent_shape: Pair = (32, 512), **kwargs
    ) -> "ConvAutoencoder":
        """Build an auto-encoder mapping ``image_shape`` cross-sections to ``latent_shape`` codes.

        Extra keyword arguments are forwarded to the constructor.
        """
        return cls(strides=compression_strides(image_shape, latent_shape), **kwargs)

    def _kernel(self, stride: Pair) -> Pair:
        """Kernel of a level: twice the stride, or a 3-wide (1-wide when column-wise) kernel at stride 1."""
        return tuple(2 * s if s > 1 else (1 if self.column_wise else 3) for s in stride)  # type: ignore[return-value]

    @property
    def compression(self) -> Pair:
        """Overall ``(factor_h, factor_w)`` between a cross-section and its code."""
        factor_h = factor_w = 1
        for stride_h, stride_w in self.strides:
            factor_h *= stride_h
            factor_w *= stride_w
        return factor_h, factor_w

    def check_input_size(self, height: int, width: int) -> None:
        """Raise unless the geometry survives every level exactly."""
        factor_h, factor_w = self.compression
        if height % factor_h or width % factor_w:
            raise ValueError(f"input must be a multiple of {factor_h}x{factor_w}, got {height}x{width}")

    def latent_shape(self, height: int, width: int) -> Pair:
        """Spatial size of the code produced for a ``height x width`` cross-section."""
        self.check_input_size(height, width)
        factor_h, factor_w = self.compression
        return height // factor_h, width // factor_w

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compress a cross-section into its latent feature map, in ``[-1, 1]``."""
        self.check_input_size(*x.shape[-2:])
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Expand a latent feature map back to a cross-section in ``[-1, 1]``."""
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct a cross-section through the latent bottleneck."""
        return self.decode(self.encode(x))


class AutoencoderGenerator(nn.Module):
    """Compose an auto-encoder with a schemaGAN generator.

    ``mode='compress'`` encodes the input, runs the generator on the code and
    decodes the result, so the generator works at the auto-encoder's latent
    resolution while the losses and the discriminator see full-size
    cross-sections. The generator must be built with ``in_channels`` and
    ``out_channels`` equal to the latent channels.

    ``mode='latent'`` feeds the auto-encoder code into the generator bottleneck
    (the generator must have been built with matching ``latent_channels``), while
    ``mode='preprocess'`` runs the generator on the auto-encoder reconstruction.

    The result is a plain ``nn.Module`` that can be handed to
    :class:`~schemaGAN_torch.schemagan.SchemaGAN` as ``generator``.

    Args:
        autoencoder: Any module exposing ``encode``/``decode`` (compress mode),
            ``encode`` (latent mode) or ``forward`` (preprocess mode).
        generator: The schemaGAN generator to drive.
        mode: One of :attr:`MODES`.
        freeze_autoencoder: Keep the auto-encoder weights fixed and in eval mode,
            which is what a pre-trained auto-encoder usually needs.
    """

    MODES = ("compress", "latent", "preprocess")

    def __init__(
        self,
        autoencoder: nn.Module,
        generator: nn.Module,
        mode: str = "compress",
        freeze_autoencoder: bool = True,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if mode == "latent" and not hasattr(autoencoder, "encode"):
            raise TypeError("latent mode requires an auto-encoder exposing encode()")
        if mode == "compress" and not (hasattr(autoencoder, "encode") and hasattr(autoencoder, "decode")):
            raise TypeError("compress mode requires an auto-encoder exposing encode() and decode()")
        self.autoencoder = autoencoder
        self.generator = generator
        self.mode = mode
        self.freeze_autoencoder = freeze_autoencoder
        if freeze_autoencoder:
            self.autoencoder.requires_grad_(False)
            self.autoencoder.eval()

    def train(self, mode: bool = True) -> "AutoencoderGenerator":
        """Switch modes, keeping a frozen auto-encoder in eval."""
        super().train(mode)
        if self.freeze_autoencoder:
            self.autoencoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a cross-section with the auto-encoder in the loop."""
        if self.mode == "preprocess":
            return self.generator(self.autoencoder(x))
        latent = self.autoencoder.encode(x)
        if self.freeze_autoencoder:
            latent = latent.detach()
        if self.mode == "compress":
            return self.autoencoder.decode(self.generator(latent))
        return self.generator(x, latent=latent)


def train_autoencoder(
    autoencoder: nn.Module,
    data: Dataset | DataLoader,
    *,
    epochs: int = 10,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    betas: tuple[float, float] = (0.5, 0.999),
    include_sources: bool = True,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> list[float]:
    """Pre-train an auto-encoder to reconstruct cross-sections with an L1 loss.

    The dataset yields ``(source, target)`` pairs in ``[-1, 1]``, as
    :class:`~schemaGAN_torch.data.CrossSectionDataset` does. The auto-encoder is
    trained on the complete targets and, with ``include_sources``, on the
    CPT-like sources as well, so that its encoder can later be applied to both
    kinds of input.

    Args:
        autoencoder: The module to train; it is left unfrozen and in train mode.
        data: A ``Dataset`` or a ``DataLoader`` of ``(source, target)`` pairs.
        epochs: Passes over the data.
        batch_size: Cross-sections per step when ``data`` is a ``Dataset``.
        learning_rate: Adam step size.
        betas: Adam moment decays.
        include_sources: Also reconstruct the sparse sources.
        device: Where to train; ``None``/``'auto'`` picks CUDA when available.
        verbose: Print the mean loss of every epoch.

    Returns:
        The mean reconstruction loss of every epoch.
    """
    if device is None or device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if isinstance(data, DataLoader):
        loader = data
    else:
        if len(data) == 0:
            raise ValueError("cannot train an auto-encoder on an empty dataset")
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    autoencoder.to(device)
    autoencoder.requires_grad_(True)
    autoencoder.train()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, betas=betas)
    criterion = nn.L1Loss()

    history: list[float] = []
    for epoch in range(1, epochs + 1):
        total = 0.0
        batches = 0
        for source, target in loader:
            batch = torch.cat([source, target]) if include_sources else target
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(autoencoder(batch), batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        if batches == 0:
            raise ValueError("cannot train an auto-encoder on an empty dataset")
        history.append(total / batches)
        if verbose:
            print(f"autoencoder epoch {epoch}/{epochs} l1={history[-1]:.4f}")
    return history
