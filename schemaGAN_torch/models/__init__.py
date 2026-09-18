"""Neural network components of the PyTorch schemaGAN."""

from __future__ import annotations

from ..config import ModelConfig
from .autoencoder import AutoencoderGenerator, ConvAutoencoder, compression_strides, train_autoencoder
from .blocks import (
    DecoderBlock,
    EncoderBlock,
    NoiseDropout,
    SameConv2d,
    SameConvTranspose2d,
    init_conv_weights,
    same_padding,
)
from .discriminator import PatchDiscriminator
from .generator import UNetGenerator

__all__ = [
    "AutoencoderGenerator",
    "ConvAutoencoder",
    "DecoderBlock",
    "EncoderBlock",
    "NoiseDropout",
    "PatchDiscriminator",
    "SameConv2d",
    "SameConvTranspose2d",
    "UNetGenerator",
    "build_discriminator",
    "build_generator",
    "compression_strides",
    "init_conv_weights",
    "same_padding",
    "train_autoencoder",
]


def build_generator(config: ModelConfig | None = None) -> UNetGenerator:
    """Build the default U-Net generator described by a model configuration."""
    config = config or ModelConfig()
    return UNetGenerator(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_filters=config.generator_filters,
        dropout=config.dropout,
        stochastic_inference=config.stochastic_inference,
        latent_channels=config.latent_channels,
    )


def build_discriminator(config: ModelConfig | None = None) -> PatchDiscriminator:
    """Build the default patch discriminator described by a model configuration."""
    config = config or ModelConfig()
    return PatchDiscriminator(
        in_channels=config.in_channels,
        base_filters=config.discriminator_filters,
    )
