"""PatchGAN discriminator of schemaGAN."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import EncoderBlock, SameConv2d, init_conv_weights


class PatchDiscriminator(nn.Module):
    """Classifies overlapping patches of a (source, target) pair as real or fake.

    ``forward`` returns raw logits; the sigmoid of the original implementation is
    folded into :class:`torch.nn.BCEWithLogitsLoss` for numerical stability.
    For a ``32x512`` input the output is a ``16x16`` patch map.

    Args:
        in_channels: Channels of a single cross-section; the source and the
            target are concatenated, so the first layer sees twice as many.
        base_filters: Filters of the first block; deeper blocks are multiples
            of it.
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 64) -> None:
        super().__init__()
        f = base_filters
        self.model = nn.Sequential(
            EncoderBlock(in_channels * 2, f, (2, 2), batchnorm=False),
            EncoderBlock(f, f * 2, (1, 2)),
            EncoderBlock(f * 2, f * 4, (1, 2)),
            EncoderBlock(f * 4, f * 8, (1, 2)),
            EncoderBlock(f * 8, f * 8, (1, 2)),
            EncoderBlock(f * 8, f * 8, (1, 1)),
            SameConv2d(f * 8, 1, (4, 4), (1, 1)),
        )
        init_conv_weights(self)

    def forward(self, source: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        """Score a pair; ``source`` may already hold both images stacked."""
        x = source if target is None else torch.cat([source, target], dim=1)
        return self.model(x)

    @torch.no_grad()
    def patch_shape(self, height: int, width: int, channels: int = 1) -> torch.Size:
        """Shape of the patch map produced for an image of the given size."""
        device = next(self.parameters()).device
        dummy = torch.zeros(1, channels * 2, height, width, device=device)
        return self.model(dummy).shape
