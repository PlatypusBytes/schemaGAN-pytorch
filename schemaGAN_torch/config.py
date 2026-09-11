"""Configuration objects for the PyTorch implementation of schemaGAN."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any


@dataclass
class DataConfig:
    """Cross-section geometry and how the sparse CPT-like input is simulated.

    Attributes:
        image_height: Number of depth pixels of a cross-section.
        image_width: Number of horizontal pixels of a cross-section.
        miss_rate: Fraction of the columns that carry no measurement.
        min_distance: Minimum number of pixels between two kept columns.
        min_ic: Lower bound of the IC range, mapped to ``-1``.
        max_ic: Upper bound of the IC range, mapped to ``+1``.
        max_missing_depth_fraction: Largest share of a profile that can be cut
            away from the bottom of a kept column.
    """

    image_height: int = 32
    image_width: int = 512
    miss_rate: float = 0.99
    min_distance: int = 51
    min_ic: float = 0.0
    max_ic: float = 4.3
    # Fraction of the profile depth that can be cut away from the bottom of a CPT.
    max_missing_depth_fraction: float = 0.5

    def __post_init__(self) -> None:
        """Reject geometries and ranges that the data pipeline cannot honour."""
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image_height and image_width must be positive")
        if not 0.0 <= self.miss_rate < 1.0:
            raise ValueError("miss_rate must lie in [0, 1)")
        if self.min_distance < 0:
            raise ValueError("min_distance must be non-negative")
        if self.max_ic <= self.min_ic:
            raise ValueError("max_ic must be greater than min_ic")
        if not 0.0 <= self.max_missing_depth_fraction <= 1.0:
            raise ValueError("max_missing_depth_fraction must lie in [0, 1]")

    @property
    def ic_range(self) -> float:
        """Width of the IC range that is mapped onto ``[-1, 1]``."""
        return self.max_ic - self.min_ic

    @property
    def image_shape(self) -> tuple[int, int]:
        """Cross-section shape as ``(height, width)``."""
        return self.image_height, self.image_width


@dataclass
class ModelConfig:
    """Architecture of the generator and the discriminator.

    Attributes:
        in_channels: Channels of a source cross-section.
        out_channels: Channels produced by the generator.
        generator_filters: Filters of the first generator layer; later layers
            are multiples of it.
        discriminator_filters: Same, for the discriminator.
        dropout: Dropout rate of the first four decoder blocks.
        stochastic_inference: Keep dropout and batch statistics active outside
            training, as pix2pix does.
        latent_channels: Channels of an external latent code fused into the
            generator bottleneck; ``None`` disables the fusion.
    """

    in_channels: int = 1
    out_channels: int = 1
    generator_filters: int = 64
    discriminator_filters: int = 64
    dropout: float = 0.5
    # Pix2pix keeps dropout and batch statistics active at inference time; that is
    # the only source of stochasticity in the generator. Set to False for
    # deterministic predictions.
    stochastic_inference: bool = True
    # Number of channels of an external latent code (e.g. from an auto-encoder)
    # that is fused into the generator bottleneck. ``None`` disables the fusion.
    latent_channels: int | None = None

    def __post_init__(self) -> None:
        """Reject filter counts and dropout rates that cannot build a network."""
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.generator_filters <= 0 or self.discriminator_filters <= 0:
            raise ValueError("filter counts must be positive")


@dataclass
class OptimConfig:
    """Optimiser and loss weighting.

    Attributes:
        learning_rate: Adam step size, shared by both networks.
        beta1: Adam first moment decay; ``0.5`` as in the pix2pix paper.
        beta2: Adam second moment decay.
        lambda_l1: Weight of the L1 reconstruction term of the generator loss.
        discriminator_loss_weight: Weight applied to the summed real/fake
            discriminator loss.
    """

    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    # Weight of the L1 reconstruction term of the generator loss.
    lambda_l1: float = 100.0
    # Weight applied to the summed real/fake discriminator loss.
    discriminator_loss_weight: float = 0.5


@dataclass
class TrainConfig:
    """Training loop settings.

    Attributes:
        epochs: Number of passes over the training set.
        batch_size: Cross-sections per optimiser step.
        shuffle: Shuffle the training set between epochs.
        num_workers: Data loading workers; keep at ``0`` when resampling masks.
        device: ``'auto'``, ``'cpu'``, ``'cuda'`` or ``'cuda:N'``.
        seed: Seed applied when the model is built.
        checkpoint_every: Epochs between checkpoints; ``0`` disables them.
        sample_every: Epochs between sample figures; ``0`` disables them.
        log_every: Batches between progress lines; ``0`` disables them.
        denormalize_metrics: Report validation errors in IC units instead of in
            the normalised ``[-1, 1]`` range.
    """

    epochs: int = 10
    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 0
    device: str = "auto"
    seed: int | None = None
    checkpoint_every: int = 1
    sample_every: int = 1
    log_every: int = 1
    denormalize_metrics: bool = True

    def __post_init__(self) -> None:
        """Reject loop settings that would never produce an update."""
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass
class SchemaGANConfig:
    """Full configuration of a :class:`~schemaGAN_torch.schemagan.SchemaGAN`."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a nested dict of primitives, safe to store in a checkpoint."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaGANConfig":
        """Rebuild a configuration from :meth:`to_dict`.

        Missing sections fall back to their defaults, so older checkpoints keep
        loading after a new section is introduced.
        """
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            value = data.get(f.name)
            if value is None:
                continue
            kwargs[f.name] = _build(f.type, value) if not is_dataclass(value) else value
        return cls(**kwargs)


_SECTION_TYPES = {
    "DataConfig": DataConfig,
    "ModelConfig": ModelConfig,
    "OptimConfig": OptimConfig,
    "TrainConfig": TrainConfig,
}


def _build(type_hint: Any, value: dict[str, Any]):
    """Instantiate a config section, ignoring keys unknown to this version."""
    name = type_hint if isinstance(type_hint, str) else getattr(type_hint, "__name__", "")
    section = _SECTION_TYPES[name]
    known = {f.name for f in fields(section)}
    return section(**{k: v for k, v in value.items() if k in known})
