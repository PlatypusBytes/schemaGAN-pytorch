"""U-Net generator of schemaGAN."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import DecoderBlock, EncoderBlock, SameConv2d, SameConvTranspose2d, init_conv_weights


class UNetGenerator(nn.Module):
    """Encoder/decoder with skip connections mapping a sparse cross-section to a dense one.

    The horizontal axis (512 px) is downsampled nine times while the vertical
    axis (32 px) is only downsampled five times, which is why half of the
    encoder/decoder blocks use an asymmetric ``(1, 2)`` stride.

    Inputs and outputs live in ``[-1, 1]`` and have shape ``(n, channels, 32, 512)``
    (or any multiple of that geometry).

    Args:
        in_channels: Channels of the sparse input cross-section.
        out_channels: Channels of the generated cross-section.
        base_filters: Filters of the first encoder block; deeper blocks are
            multiples of it.
        dropout: Dropout rate of the first four decoder blocks.
        stochastic_inference: Keep dropout and batch statistics active outside
            training, as pix2pix does.
        latent_channels: Channels of an external latent code fused into the
            bottleneck; ``None`` disables the fusion.
    """

    HEIGHT_FACTOR = 32
    WIDTH_FACTOR = 512

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
        dropout: float = 0.5,
        stochastic_inference: bool = True,
        latent_channels: int | None = None,
    ) -> None:
        super().__init__()
        f = base_filters
        stochastic = bool(stochastic_inference)
        self.latent_channels = latent_channels

        norm_kwargs = {"batch_stats_only": stochastic}
        drop_kwargs = {"dropout": dropout, "always_dropout": stochastic, **norm_kwargs}

        self.encoder = nn.ModuleList(
            [
                EncoderBlock(in_channels, f, (2, 2), batchnorm=False),
                EncoderBlock(f, f * 2, (2, 2), **norm_kwargs),
                EncoderBlock(f * 2, f * 4, (2, 2), **norm_kwargs),
                EncoderBlock(f * 4, f * 8, (2, 2), **norm_kwargs),
                EncoderBlock(f * 8, f * 8, (1, 2), **norm_kwargs),
                EncoderBlock(f * 8, f * 8, (1, 2), **norm_kwargs),
                EncoderBlock(f * 8, f * 8, (1, 2), **norm_kwargs),
                EncoderBlock(f * 8, f * 8, (1, 2), **norm_kwargs),
            ]
        )

        self.bottleneck = nn.Sequential(
            SameConv2d(f * 8, f * 8, (4, 4), (2, 2)),
            nn.ReLU(inplace=True),
        )

        self.latent_fusion = (
            nn.Conv2d(f * 8 + latent_channels, f * 8, kernel_size=1) if latent_channels else None
        )

        self.decoder = nn.ModuleList(
            [
                DecoderBlock(f * 8, f * 8, (2, 2), **drop_kwargs),
                DecoderBlock(f * 16, f * 8, (1, 2), **drop_kwargs),
                DecoderBlock(f * 16, f * 8, (1, 2), **drop_kwargs),
                DecoderBlock(f * 16, f * 8, (1, 2), **drop_kwargs),
                DecoderBlock(f * 16, f * 8, (1, 2), **norm_kwargs),
                DecoderBlock(f * 16, f * 4, (2, 2), **norm_kwargs),
                DecoderBlock(f * 8, f * 2, (2, 2), **norm_kwargs),
                DecoderBlock(f * 4, f, (2, 2), **norm_kwargs),
            ]
        )

        self.output = nn.Sequential(
            SameConvTranspose2d(f * 2, out_channels, (4, 4), (2, 2)),
            nn.Tanh(),
        )

        init_conv_weights(self)

    @classmethod
    def check_input_size(cls, height: int, width: int) -> None:
        """Raise unless the geometry survives every downsampling step."""
        if height % cls.HEIGHT_FACTOR or width % cls.WIDTH_FACTOR:
            raise ValueError(
                f"input must be a multiple of {cls.HEIGHT_FACTOR}x{cls.WIDTH_FACTOR}, got {height}x{width}"
            )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the encoder, returning the bottleneck and the skip connections."""
        self.check_input_size(*x.shape[-2:])
        skips: list[torch.Tensor] = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        return self.bottleneck(x), skips

    def fuse_latent(self, bottleneck: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Merge an external latent code into the bottleneck feature map.

        The code may be flat (``(n, c)``) or spatial (``(n, c, h, w)``); it is
        resized to the bottleneck resolution before being concatenated.
        """
        if self.latent_fusion is None:
            raise RuntimeError("generator was built without latent_channels")
        if latent.dim() == 2:
            latent = latent[:, :, None, None]
        if latent.dim() != 4:
            raise ValueError("latent must have shape (n, c) or (n, c, h, w)")
        if latent.shape[1] != self.latent_channels:
            raise ValueError(f"latent has {latent.shape[1]} channels, expected {self.latent_channels}")
        if latent.shape[-2:] != bottleneck.shape[-2:]:
            latent = F.interpolate(latent, size=bottleneck.shape[-2:], mode="nearest")
        return self.latent_fusion(torch.cat([bottleneck, latent], dim=1))

    def decode(
        self,
        bottleneck: torch.Tensor,
        skips: list[torch.Tensor],
        latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the decoder on a bottleneck and its skip connections."""
        if len(skips) != len(self.decoder):
            raise ValueError(f"expected {len(self.decoder)} skip connections, got {len(skips)}")
        if latent is not None:
            bottleneck = self.fuse_latent(bottleneck, latent)
        elif self.latent_fusion is not None:
            raise ValueError("this generator requires a latent code")

        x = bottleneck
        for block, skip in zip(self.decoder, reversed(skips)):
            x = block(x, skip)
        return self.output(x)

    def forward(self, x: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        """Turn a sparse cross-section into a dense one, in ``[-1, 1]``."""
        bottleneck, skips = self.encode(x)
        return self.decode(bottleneck, skips, latent)
