"""PyTorch implementation of schemaGAN.

schemaGAN is a conditional GAN that reconstructs a full 2D subsoil
schematisation from a handful of CPT-like vertical soundings.

Typical use::

    from schemaGAN_torch import CrossSectionDataset, SchemaGAN, SchemaGANConfig

    config = SchemaGANConfig()
    dataset = CrossSectionDataset("synthetic_data/512x32/train", config.data)
    model = SchemaGAN(config)
    model.train(dataset, output_dir="results")
"""

from __future__ import annotations

from .config import DataConfig, ModelConfig, OptimConfig, SchemaGANConfig, TrainConfig
from .data import (
    CrossSectionDataset,
    apply_mask,
    cpt_like_mask,
    denormalize_ic,
    list_csv_files,
    load_cross_sections,
    normalize_ic,
    read_cross_section_csv,
    write_cross_section_csv,
)
from .models import (
    AutoencoderGenerator,
    ConvAutoencoder,
    PatchDiscriminator,
    UNetGenerator,
    build_discriminator,
    build_generator,
    train_autoencoder,
)
from .schemagan import History, SchemaGAN, ValidationResult, resolve_device, set_seed

__all__ = [
    "AutoencoderGenerator",
    "ConvAutoencoder",
    "CrossSectionDataset",
    "DataConfig",
    "History",
    "ModelConfig",
    "OptimConfig",
    "PatchDiscriminator",
    "SchemaGAN",
    "SchemaGANConfig",
    "TrainConfig",
    "UNetGenerator",
    "ValidationResult",
    "apply_mask",
    "build_discriminator",
    "build_generator",
    "cpt_like_mask",
    "denormalize_ic",
    "list_csv_files",
    "load_cross_sections",
    "normalize_ic",
    "read_cross_section_csv",
    "resolve_device",
    "set_seed",
    "train_autoencoder",
    "write_cross_section_csv",
]

__version__ = "0.1.0"
