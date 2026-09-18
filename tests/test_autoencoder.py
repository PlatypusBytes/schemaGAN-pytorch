"""Tests for the auto-encoder hooks used to extend schemaGAN."""

import numpy as np
import pytest
import torch

from schemaGAN_torch import CrossSectionDataset, SchemaGAN
from schemaGAN_torch.models import (
    AutoencoderGenerator,
    ConvAutoencoder,
    UNetGenerator,
    compression_strides,
    train_autoencoder,
)

from conftest import HEIGHT, WIDTH, make_grids, tiny_config

TALL = 4 * HEIGHT  # a cross-section with a finer depth axis, compressed 4x to 32 px


class TestCompressionStrides:
    """Splitting a compression factor into per-level strides."""

    def test_depth_only_compression(self):
        assert compression_strides((320, 512), (32, 512)) == [(2, 1), (5, 1)]

    def test_both_axes(self):
        assert compression_strides((3200, 51200), (32, 512)) == [(2, 2), (2, 2), (5, 5), (5, 5)]

    def test_no_compression_is_a_single_level(self):
        assert compression_strides((32, 512), (32, 512)) == [(1, 1)]

    def test_rejects_a_non_integer_factor(self):
        with pytest.raises(ValueError, match="multiple"):
            compression_strides((100, 512), (32, 512))


class TestConvAutoencoder:
    """The stand-alone convolutional auto-encoder."""

    @pytest.fixture
    def autoencoder(self):
        """A narrow isotropic auto-encoder, fast enough for a test."""
        return ConvAutoencoder(base_filters=4, latent_channels=8)

    def test_reconstructs_the_input_geometry(self, autoencoder):
        x = torch.randn(2, 1, HEIGHT, WIDTH)
        assert autoencoder(x).shape == x.shape

    def test_latent_code_is_downsampled_by_two_per_level(self, autoencoder):
        latent = autoencoder.encode(torch.randn(2, 1, HEIGHT, WIDTH))
        assert latent.shape == (2, 8, HEIGHT // 8, WIDTH // 8)
        assert autoencoder.compression == (8, 8)
        assert autoencoder.latent_shape(HEIGHT, WIDTH) == (HEIGHT // 8, WIDTH // 8)

    def test_decode_inverts_the_latent_geometry(self, autoencoder):
        latent = autoencoder.encode(torch.randn(1, 1, HEIGHT, WIDTH))
        assert autoencoder.decode(latent).shape == (1, 1, HEIGHT, WIDTH)

    def test_code_and_reconstruction_are_bounded_by_tanh(self, autoencoder):
        x = torch.randn(1, 1, HEIGHT, WIDTH) * 10.0
        latent = autoencoder.encode(x)
        out = autoencoder.decode(latent)
        assert latent.min() >= -1.0 and latent.max() <= 1.0
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_anisotropic_strides_compress_the_depth_axis_only(self):
        autoencoder = ConvAutoencoder(base_filters=4, latent_channels=8, strides=[(2, 1), (5, 1)])
        latent = autoencoder.encode(torch.randn(1, 1, 320, 64))
        assert latent.shape == (1, 8, 32, 64)
        assert autoencoder.decode(latent).shape == (1, 1, 320, 64)

    def test_for_geometry_builds_the_matching_strides(self):
        autoencoder = ConvAutoencoder.for_geometry((320, 512), (32, 512), base_filters=4, latent_channels=8)
        assert autoencoder.strides == [(2, 1), (5, 1)]
        assert autoencoder.latent_shape(320, 512) == (32, 512)

    def test_rejects_a_geometry_that_does_not_divide(self):
        autoencoder = ConvAutoencoder(base_filters=4, latent_channels=8, strides=[(2, 1), (5, 1)])
        with pytest.raises(ValueError, match="multiple of 10x1"):
            autoencoder.encode(torch.randn(1, 1, 35, 64))

    def test_column_wise_encoder_keeps_columns_independent(self):
        autoencoder = ConvAutoencoder(
            base_filters=4, latent_channels=8, strides=[(2, 1), (2, 1)], column_wise=True
        )
        x = torch.randn(1, 1, HEIGHT, 16)
        perturbed = x.clone()
        perturbed[..., 5] += 1.0
        difference = (autoencoder.encode(perturbed) - autoencoder.encode(x)).abs().sum(dim=(0, 1, 2))
        assert difference[5] > 0.0
        assert torch.all(difference[torch.arange(16) != 5] == 0.0)

    def test_mixing_encoder_spreads_a_column_to_its_neighbours(self):
        autoencoder = ConvAutoencoder(base_filters=4, latent_channels=8, strides=[(2, 1), (2, 1)])
        x = torch.randn(1, 1, HEIGHT, 16)
        perturbed = x.clone()
        perturbed[..., 5] += 1.0
        difference = (autoencoder.encode(perturbed) - autoencoder.encode(x)).abs().sum(dim=(0, 1, 2))
        assert (difference > 0.0).sum() > 1

    def test_rejects_empty_or_degenerate_strides(self):
        with pytest.raises(ValueError, match="at least one"):
            ConvAutoencoder(strides=[])
        with pytest.raises(ValueError, match="at least 1"):
            ConvAutoencoder(strides=[(0, 2)])


class TestTrainAutoencoder:
    """Pre-training the auto-encoder on its own."""

    def test_reduces_the_reconstruction_loss(self, dataset):
        torch.manual_seed(0)
        autoencoder = ConvAutoencoder(base_filters=4, latent_channels=8, strides=[(2, 1), (2, 1)])
        history = train_autoencoder(autoencoder, dataset, epochs=4, learning_rate=1e-3, device="cpu", verbose=False)
        assert len(history) == 4
        assert history[-1] < history[0]
        assert all(p.requires_grad for p in autoencoder.parameters())

    def test_can_skip_the_sparse_sources(self, dataset):
        autoencoder = ConvAutoencoder(base_filters=4, latent_channels=8, strides=[(2, 1)])
        history = train_autoencoder(
            autoencoder, dataset, epochs=1, include_sources=False, device="cpu", verbose=False
        )
        assert len(history) == 1

    def test_rejects_an_empty_dataset(self):
        autoencoder = ConvAutoencoder(base_filters=4, latent_channels=8, strides=[(2, 1)])
        empty = torch.utils.data.TensorDataset(torch.empty(0, 1, HEIGHT, WIDTH), torch.empty(0, 1, HEIGHT, WIDTH))
        with pytest.raises(ValueError, match="empty"):
            train_autoencoder(autoencoder, empty, epochs=1, device="cpu", verbose=False)


class TestAutoencoderGenerator:
    """Composing an auto-encoder with the schemaGAN generator."""

    @pytest.fixture
    def autoencoder(self):
        """A narrow isotropic auto-encoder, fast enough for a test."""
        return ConvAutoencoder(base_filters=4, latent_channels=8)

    @pytest.fixture
    def compressor(self):
        """An auto-encoder mapping tall cross-sections onto the 32 x 512 generator grid."""
        return ConvAutoencoder.for_geometry((TALL, WIDTH), (HEIGHT, WIDTH), base_filters=4, latent_channels=8)

    def test_compress_mode_runs_the_generator_in_the_code_space(self, compressor):
        generator = UNetGenerator(in_channels=8, out_channels=8, base_filters=4)
        combined = AutoencoderGenerator(compressor, generator, mode="compress")
        seen: list[torch.Size] = []
        generator.register_forward_hook(lambda module, args, output: seen.append(args[0].shape))
        out = combined(torch.randn(2, 1, TALL, WIDTH))
        assert out.shape == (2, 1, TALL, WIDTH)
        assert seen == [torch.Size((2, 8, HEIGHT, WIDTH))]

    def test_compress_is_the_default_mode(self, compressor):
        combined = AutoencoderGenerator(compressor, UNetGenerator(in_channels=8, out_channels=8, base_filters=4))
        assert combined.mode == "compress"

    def test_latent_mode_feeds_the_generator_bottleneck(self, autoencoder):
        generator = UNetGenerator(base_filters=4, latent_channels=8)
        combined = AutoencoderGenerator(autoencoder, generator, mode="latent")
        assert combined(torch.randn(2, 1, HEIGHT, WIDTH)).shape == (2, 1, HEIGHT, WIDTH)

    def test_latent_fusion_averages_a_spatial_code(self):
        generator = UNetGenerator(base_filters=4, latent_channels=1)
        bottleneck = torch.zeros(1, 32, 1, 1)
        code = torch.arange(4 * 64, dtype=torch.float32).reshape(1, 1, 4, 64)
        fused = generator.fuse_latent(bottleneck, code)
        # The fusion is a 1x1 convolution: with a zero bottleneck it is linear in
        # the pooled code, so a corner pixel (0) and the mean give different results.
        corner = generator.fuse_latent(bottleneck, torch.zeros(1, 1))
        mean = generator.fuse_latent(bottleneck, torch.full((1, 1), code.mean().item()))
        assert torch.allclose(fused, mean)
        assert not torch.allclose(fused, corner)

    def test_preprocess_mode_generates_from_the_reconstruction(self, autoencoder):
        combined = AutoencoderGenerator(autoencoder, UNetGenerator(base_filters=4), mode="preprocess")
        assert combined(torch.randn(1, 1, HEIGHT, WIDTH)).shape == (1, 1, HEIGHT, WIDTH)

    def test_a_frozen_autoencoder_keeps_its_weights(self, compressor):
        combined = AutoencoderGenerator(
            compressor, UNetGenerator(in_channels=8, out_channels=8, base_filters=4), freeze_autoencoder=True
        )
        before = compressor.encoder[0].conv.weight.detach().clone()
        combined(torch.randn(1, 1, TALL, WIDTH)).sum().backward()
        assert all(not p.requires_grad for p in combined.autoencoder.parameters())
        assert compressor.encoder[0].conv.weight.grad is None
        assert torch.equal(before, compressor.encoder[0].conv.weight)
        assert combined.generator.encoder[0].conv.conv.weight.grad is not None

    def test_a_frozen_autoencoder_stays_in_eval_mode(self, autoencoder):
        combined = AutoencoderGenerator(
            autoencoder, UNetGenerator(base_filters=4, latent_channels=8), mode="latent", freeze_autoencoder=True
        )
        combined.train()
        assert combined.generator.training
        assert not combined.autoencoder.training

    def test_a_trainable_autoencoder_receives_gradients(self, compressor):
        combined = AutoencoderGenerator(
            compressor,
            UNetGenerator(in_channels=8, out_channels=8, base_filters=4),
            freeze_autoencoder=False,
        )
        combined(torch.randn(1, 1, TALL, WIDTH)).sum().backward()
        assert compressor.encoder[0].conv.weight.grad is not None

    def test_rejects_an_unknown_mode(self, autoencoder):
        with pytest.raises(ValueError, match="mode"):
            AutoencoderGenerator(autoencoder, UNetGenerator(base_filters=4), mode="latents")

    def test_latent_mode_requires_an_encoder(self):
        with pytest.raises(TypeError, match="encode"):
            AutoencoderGenerator(torch.nn.Identity(), UNetGenerator(base_filters=4), mode="latent")

    def test_compress_mode_requires_an_encoder_and_a_decoder(self):
        with pytest.raises(TypeError, match="decode"):
            AutoencoderGenerator(torch.nn.Identity(), UNetGenerator(base_filters=4), mode="compress")


class TestSchemaGANWithAnAutoencoder:
    """Driving the full model with an auto-encoder backed generator."""

    @pytest.fixture
    def tall_dataset(self):
        """Cross-sections four times deeper than the generator grid."""
        grids = make_grids(n=2, seed=3)
        grids = np.repeat(grids, TALL // HEIGHT, axis=1)
        config = tiny_config()
        config.data.image_height = TALL
        return CrossSectionDataset(grids, config.data, seed=7)

    def test_trains_at_a_finer_depth_resolution(self, tall_dataset, tmp_path):
        config = tiny_config()
        config.data.image_height = TALL
        compressor = ConvAutoencoder.for_geometry((TALL, WIDTH), (HEIGHT, WIDTH), base_filters=4, latent_channels=8)
        train_autoencoder(compressor, tall_dataset, epochs=1, device="cpu", verbose=False)
        generator = AutoencoderGenerator(compressor, UNetGenerator(in_channels=8, out_channels=8, base_filters=4))
        model = SchemaGAN(config, generator=generator)

        history = model.train(tall_dataset, output_dir=tmp_path, verbose=False)
        assert len(history.iterations["g_loss"]) == len(tall_dataset)
        assert model.predict(tall_dataset[0][0]).shape == (1, 1, TALL, WIDTH)
        assert model.validate(tall_dataset).summary["mae"] >= 0.0

    def test_trains_with_a_latent_fused_generator(self, dataset):
        config = tiny_config()
        config.model.latent_channels = 8
        generator = AutoencoderGenerator(
            ConvAutoencoder(base_filters=4, latent_channels=8),
            UNetGenerator(base_filters=4, latent_channels=8),
            mode="latent",
        )
        model = SchemaGAN(config, generator=generator)

        history = model.train(dataset, verbose=False)
        assert len(history.iterations["g_loss"]) == len(dataset)
        assert model.predict(dataset[0][0]).shape == (1, 1, HEIGHT, WIDTH)

    def test_reloads_a_custom_generator_when_it_is_supplied(self, tall_dataset, tmp_path):
        config = tiny_config()
        config.data.image_height = TALL
        make_generator = lambda: AutoencoderGenerator(
            ConvAutoencoder.for_geometry((TALL, WIDTH), (HEIGHT, WIDTH), base_filters=4, latent_channels=8),
            UNetGenerator(in_channels=8, out_channels=8, base_filters=4),
        )
        model = SchemaGAN(config, generator=make_generator())
        path = model.save(tmp_path / "ae_model.pt")

        restored = SchemaGAN.load(path, device="cpu", generator=make_generator())
        assert isinstance(restored.generator, AutoencoderGenerator)
        for saved, loaded in zip(model.generator.parameters(), restored.generator.parameters()):
            assert torch.equal(saved, loaded)
        assert restored.predict(tall_dataset[0][0]).shape == (1, 1, TALL, WIDTH)
