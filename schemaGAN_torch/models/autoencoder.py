"""Hooks for combining schemaGAN with an auto-encoder/decoder.

:class:`ConvAutoencoder` is a plain convolutional auto-encoder over the same
``(n, c, 32, 512)`` cross-sections; :class:`AutoencoderGenerator` plugs any such
model into a schemaGAN generator so it can be passed to
:class:`~schemaGAN_torch.schemagan.SchemaGAN` as a drop-in generator.
"""

from __future__ import annotations

import torch
from torch import nn

from .blocks import init_conv_weights


class ConvAutoencoder(nn.Module):
    """Symmetric convolutional auto-encoder used to learn a latent cross-section code.

    Args:
        in_channels: Channels of a cross-section.
        base_filters: Filters of the first encoder level.
        latent_channels: Channels of the latent feature map.
        depth: Number of stride-2 levels; the latent map is ``2**depth`` times
            smaller than the input along both axes.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        latent_channels: int = 64,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth
        self.latent_channels = latent_channels

        encoder: list[nn.Module] = []
        channels = in_channels
        for level in range(depth):
            out_channels = latent_channels if level == depth - 1 else base_filters * 2**level
            encoder += [
                nn.Conv2d(channels, out_channels, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels = out_channels
        self.encoder = nn.Sequential(*encoder)

        decoder: list[nn.Module] = []
        for level in reversed(range(depth)):
            out_channels = in_channels if level == 0 else base_filters * 2 ** (level - 1)
            decoder.append(nn.ConvTranspose2d(channels, out_channels, 4, stride=2, padding=1))
            decoder.append(nn.Tanh() if level == 0 else nn.ReLU(inplace=True))
            channels = out_channels
        self.decoder = nn.Sequential(*decoder)

        init_conv_weights(self)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compress a cross-section into its latent feature map."""
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Expand a latent feature map back to a cross-section in ``[-1, 1]``."""
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct a cross-section through the latent bottleneck."""
        return self.decode(self.encode(x))


class AutoencoderGenerator(nn.Module):
    """Compose an auto-encoder with a schemaGAN generator.

    ``mode='latent'`` feeds the auto-encoder code into the generator bottleneck
    (the generator must have been built with matching ``latent_channels``), while
    ``mode='preprocess'`` runs the generator on the auto-encoder reconstruction.

    The result is a plain ``nn.Module`` that can be handed to
    :class:`~schemaGAN_torch.schemagan.SchemaGAN` as ``generator``.

    Args:
        autoencoder: Any module exposing ``encode`` (latent mode) or ``forward``
            (preprocess mode).
        generator: The schemaGAN generator to drive.
        mode: One of :attr:`MODES`.
        freeze_autoencoder: Keep the auto-encoder weights fixed and in eval mode,
            which is what a pre-trained auto-encoder usually needs.
    """

    MODES = ("latent", "preprocess")

    def __init__(
        self,
        autoencoder: nn.Module,
        generator: nn.Module,
        mode: str = "latent",
        freeze_autoencoder: bool = True,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if mode == "latent" and not hasattr(autoencoder, "encode"):
            raise TypeError("latent mode requires an auto-encoder exposing encode()")
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
        return self.generator(x, latent=latent)
