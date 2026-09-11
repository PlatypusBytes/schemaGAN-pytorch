"""Tests for the configuration dataclasses and their serialisation."""

import pytest

from schemaGAN_torch.config import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    SchemaGANConfig,
    TrainConfig,
)


class TestDataConfig:
    """Cross-section geometry and CPT simulation settings."""

    def test_defaults_match_the_original_implementation(self):
        config = DataConfig()
        assert config.image_shape == (32, 512)
        assert config.miss_rate == 0.99
        assert config.min_distance == 51
        assert config.ic_range == pytest.approx(4.3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"image_height": 0},
            {"image_width": -1},
            {"miss_rate": 1.0},
            {"miss_rate": -0.1},
            {"min_distance": -1},
            {"min_ic": 5.0, "max_ic": 4.3},
            {"max_missing_depth_fraction": 1.5},
        ],
    )
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            DataConfig(**kwargs)


class TestModelConfig:
    """Architecture settings."""

    @pytest.mark.parametrize("kwargs", [{"dropout": 1.0}, {"dropout": -0.1}, {"generator_filters": 0}])
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            ModelConfig(**kwargs)

    def test_latent_channels_default_to_disabled(self):
        assert ModelConfig().latent_channels is None


class TestTrainConfig:
    """Training loop settings."""

    @pytest.mark.parametrize("kwargs", [{"epochs": -1}, {"batch_size": 0}])
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            TrainConfig(**kwargs)


class TestSchemaGANConfig:
    """Round-tripping the full configuration through a checkpoint payload."""

    def test_roundtrips_through_a_dict(self):
        config = SchemaGANConfig(
            data=DataConfig(miss_rate=0.9),
            model=ModelConfig(generator_filters=8, latent_channels=16),
            optim=OptimConfig(lambda_l1=50.0),
            train=TrainConfig(epochs=3, seed=7),
        )
        restored = SchemaGANConfig.from_dict(config.to_dict())
        assert restored == config

    def test_ignores_unknown_keys(self):
        payload = SchemaGANConfig().to_dict()
        payload["data"]["from_a_future_version"] = 1
        assert SchemaGANConfig.from_dict(payload).data == DataConfig()

    def test_missing_sections_fall_back_to_defaults(self):
        restored = SchemaGANConfig.from_dict({"train": {"epochs": 2}})
        assert restored.train.epochs == 2
        assert restored.model == ModelConfig()
