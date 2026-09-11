"""Shared fixtures for the schemaGAN PyTorch test-suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from schemaGAN_torch import CrossSectionDataset, SchemaGAN, SchemaGANConfig  # noqa: E402
from schemaGAN_torch.config import DataConfig, ModelConfig, OptimConfig, TrainConfig  # noqa: E402
from schemaGAN_torch.data import write_cross_section_csv  # noqa: E402

HEIGHT = 32
WIDTH = 512


def make_grids(n: int = 3, seed: int = 0) -> np.ndarray:
    """Smooth, IC-like cross-sections with layered structure."""
    rng = np.random.default_rng(seed)
    depth = np.linspace(0.0, 1.0, HEIGHT)[:, None]
    distance = np.linspace(0.0, 1.0, WIDTH)[None, :]
    grids = []
    for _ in range(n):
        offset = rng.uniform(0.5, 1.5)
        layers = 1.0 + 2.5 * depth + 0.4 * np.sin(6.0 * np.pi * distance + offset)
        grids.append((layers + rng.normal(0.0, 0.05, (HEIGHT, WIDTH))).clip(0.1, 4.2))
    return np.stack(grids).astype(np.float32)


@pytest.fixture(scope="session")
def grids() -> np.ndarray:
    """Three synthetic cross-sections in raw IC units."""
    return make_grids()


@pytest.fixture(scope="session")
def csv_dir(tmp_path_factory, grids) -> Path:
    """A directory holding the synthetic cross-sections as schemaGAN CSV files."""
    directory = tmp_path_factory.mktemp("cross_sections")
    for index, grid in enumerate(grids):
        write_cross_section_csv(directory / f"cs_{index}.csv", grid)
    return directory


def tiny_config(**train_overrides) -> SchemaGANConfig:
    """A configuration small enough to train within a test."""
    train = {
        "epochs": 1,
        "batch_size": 1,
        "device": "cpu",
        "seed": 1234,
        "log_every": 0,
        "shuffle": False,
    }
    train.update(train_overrides)
    return SchemaGANConfig(
        data=DataConfig(),
        model=ModelConfig(generator_filters=4, discriminator_filters=4),
        optim=OptimConfig(lambda_l1=10.0),
        train=TrainConfig(**train),
    )


@pytest.fixture
def config() -> SchemaGANConfig:
    """The small configuration used by most tests."""
    return tiny_config()


def write_config_file(path: Path, **sections: dict) -> Path:
    """Write a YAML settings file with the tiny model and the given script sections."""
    document = tiny_config().to_dict()
    document.update(sections)
    path.write_text(yaml.safe_dump(document))
    return path


@pytest.fixture
def deterministic_config() -> SchemaGANConfig:
    """Same, but without dropout or batch-statistics noise at inference."""
    cfg = tiny_config()
    cfg.model.stochastic_inference = False
    cfg.model.dropout = 0.0
    return cfg


@pytest.fixture
def dataset(grids, config) -> CrossSectionDataset:
    """A dataset with fixed CPT layouts, built from the synthetic grids."""
    return CrossSectionDataset(grids, config.data, seed=7)


@pytest.fixture
def model(config) -> SchemaGAN:
    """An untrained model on the CPU."""
    return SchemaGAN(config)


@pytest.fixture(scope="session")
def trained_checkpoint(tmp_path_factory, csv_dir) -> Path:
    """A one-epoch checkpoint reused by the CLI tests."""
    output_dir = tmp_path_factory.mktemp("trained")
    cfg = tiny_config()
    data = CrossSectionDataset(csv_dir, cfg.data, seed=11)
    SchemaGAN(cfg).train(data, output_dir=output_dir, verbose=False)
    return output_dir / "final_model.pt"
