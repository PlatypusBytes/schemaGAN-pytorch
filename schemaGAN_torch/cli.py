"""YAML settings shared by the training/validation/inference scripts.

Every entry point reads a single YAML file that holds both the model
configuration (the ``data``/``model``/``optim``/``train`` sections consumed by
:class:`~schemaGAN_torch.config.SchemaGANConfig`) and one section per script::

    python training_schemaGAN_torch.py configs/default.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .config import SchemaGANConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

_Settings = TypeVar("_Settings")


@dataclass
class TrainingSettings:
    """The ``training`` section.

    Attributes:
        data_dir: Directory with the training cross-sections.
        val_dir: Optional directory with validation cross-sections.
        output_dir: Where checkpoints, figures and metrics are written.
        resample_mask: Draw a new CPT layout on every access instead of a fixed
            one per cross-section.
        verbose: Print per-batch progress.
    """

    data_dir: str | None = None
    val_dir: str | None = None
    output_dir: str = "results/schemagan_torch"
    resample_mask: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        """Reject a configuration that does not say what to train on."""
        if not self.data_dir:
            raise ValueError("'training.data_dir' must be set in the configuration file")


@dataclass
class ValidationSettings:
    """The ``validation`` section.

    Attributes:
        data_dir: Directory with the labelled cross-sections to score.
        checkpoint: A ``.pt`` checkpoint or a directory of them.
        output_dir: Where the errors and the figures are written.
        batch_size: Cross-sections per forward pass.
        plots: Comparison figures per checkpoint; ``0`` disables them.
    """

    data_dir: str | None = None
    checkpoint: str | None = None
    output_dir: str = "results/schemagan_torch/validation"
    batch_size: int = 1
    plots: int = 3

    def __post_init__(self) -> None:
        """Reject a configuration without data or without a model to score."""
        if not self.data_dir:
            raise ValueError("'validation.data_dir' must be set in the configuration file")
        if not self.checkpoint:
            raise ValueError("'validation.checkpoint' must be set in the configuration file")


@dataclass
class InferenceSettings:
    """The ``inference`` section.

    Attributes:
        input: A CSV cross-section or a directory of them.
        checkpoint: The ``.pt`` checkpoint to generate with.
        output_dir: Where the generated cross-sections are written.
        batch_size: Cross-sections per forward pass.
        simulate_cpt: Treat the input as complete cross-sections and mask them.
        miss_rate: Overrides the masking rate stored in the checkpoint.
        min_distance: Overrides the CPT spacing stored in the checkpoint.
        plots: Write a figure next to every generated CSV file.
    """

    input: str | None = None
    checkpoint: str | None = None
    output_dir: str = "results/schemagan_torch/inference"
    batch_size: int = 1
    simulate_cpt: bool = False
    miss_rate: float | None = None
    min_distance: int | None = None
    plots: bool = True

    def __post_init__(self) -> None:
        """Reject a configuration without input or without a model."""
        if not self.input:
            raise ValueError("'inference.input' must be set in the configuration file")
        if not self.checkpoint:
            raise ValueError("'inference.checkpoint' must be set in the configuration file")


def read_config(path: str | Path | None = None) -> dict[str, Any]:
    """Parse a YAML configuration file into a mapping of sections.

    Args:
        path: The file to read; :data:`DEFAULT_CONFIG_PATH` when omitted.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the document is not a mapping.
    """
    path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"configuration file {path} does not exist")
    document = yaml.safe_load(path.read_text()) or {}
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a mapping of sections")
    return document


def script_settings(
    document: dict[str, Any], section: str, settings_type: type[_Settings]
) -> _Settings:
    """Build the settings of one script from its section of the document.

    Raises:
        ValueError: If the section is not a mapping or holds an unknown key.
    """
    values = document.get(section) or {}
    if not isinstance(values, dict):
        raise ValueError(f"section '{section}' must be a mapping")
    unknown = set(values) - {f.name for f in fields(settings_type)}
    if unknown:
        raise ValueError(f"unknown keys in section '{section}': {', '.join(sorted(unknown))}")
    return settings_type(**values)


def load_settings(
    argv: list[str] | None, description: str | None, section: str, settings_type: type[_Settings]
) -> tuple[SchemaGANConfig, _Settings]:
    """Read the configuration file named on the command line.

    Args:
        argv: Command line arguments; ``sys.argv`` is used when omitted.
        description: Help text of the calling script.
        section: Name of the script section of the YAML document.
        settings_type: Dataclass describing that section.

    Returns:
        The model configuration and the settings of the script.
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help=f"YAML settings file (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args(argv)
    document = read_config(args.config)
    return SchemaGANConfig.from_dict(document), script_settings(document, section, settings_type)
