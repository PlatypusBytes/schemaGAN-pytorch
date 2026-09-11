"""Tests for the auto-encoder hooks used to extend schemaGAN."""

import pytest
import torch

from schemaGAN_torch import SchemaGAN
from schemaGAN_torch.models import AutoencoderGenerator, ConvAutoencoder, UNetGenerator

from conftest import HEIGHT, WIDTH, tiny_config


class TestConvAutoencoder:
    """The stand-alone convolutional auto-encoder."""

    @pytest.fixture
    def autoencoder(self):
        """A narrow auto-encoder, fast enough for a test."""
        return ConvAutoencoder(base_filters=4, latent_channels=8, depth=3)

    def test_reconstructs_the_input_geometry(self, autoencoder):
        x = torch.randn(2, 1, HEIGHT, WIDTH)
        assert autoencoder(x).shape == x.shape

    def test_latent_code_is_downsampled_by_two_per_level(self, autoencoder):
        latent = autoencoder.encode(torch.randn(2, 1, HEIGHT, WIDTH))
        assert latent.shape == (2, 8, HEIGHT // 8, WIDTH // 8)

    def test_decode_inverts_the_latent_geometry(self, autoencoder):
        latent = autoencoder.encode(torch.randn(1, 1, HEIGHT, WIDTH))
        assert autoencoder.decode(latent).shape == (1, 1, HEIGHT, WIDTH)

    def test_reconstruction_is_bounded_by_tanh(self, autoencoder):
        out = autoencoder(torch.randn(1, 1, HEIGHT, WIDTH) * 10.0)
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_rejects_a_degenerate_depth(self):
        with pytest.raises(ValueError, match="depth"):
            ConvAutoencoder(depth=0)


class TestAutoencoderGenerator:
    """Composing an auto-encoder with the schemaGAN generator."""

    @pytest.fixture
    def autoencoder(self):
        """A narrow auto-encoder, fast enough for a test."""
        return ConvAutoencoder(base_filters=4, latent_channels=8, depth=3)

    def test_latent_mode_feeds_the_generator_bottleneck(self, autoencoder):
        generator = UNetGenerator(base_filters=4, latent_channels=8)
        combined = AutoencoderGenerator(autoencoder, generator, mode="latent")
        assert combined(torch.randn(2, 1, HEIGHT, WIDTH)).shape == (2, 1, HEIGHT, WIDTH)

    def test_preprocess_mode_generates_from_the_reconstruction(self, autoencoder):
        combined = AutoencoderGenerator(autoencoder, UNetGenerator(base_filters=4), mode="preprocess")
        assert combined(torch.randn(1, 1, HEIGHT, WIDTH)).shape == (1, 1, HEIGHT, WIDTH)

    def test_a_frozen_autoencoder_keeps_its_weights(self, autoencoder):
        combined = AutoencoderGenerator(
            autoencoder, UNetGenerator(base_filters=4, latent_channels=8), freeze_autoencoder=True
        )
        before = autoencoder.encoder[0].weight.detach().clone()
        combined(torch.randn(1, 1, HEIGHT, WIDTH)).sum().backward()
        assert all(not p.requires_grad for p in combined.autoencoder.parameters())
        assert autoencoder.encoder[0].weight.grad is None
        assert torch.equal(before, autoencoder.encoder[0].weight)

    def test_a_frozen_autoencoder_stays_in_eval_mode(self, autoencoder):
        combined = AutoencoderGenerator(
            autoencoder, UNetGenerator(base_filters=4, latent_channels=8), freeze_autoencoder=True
        )
        combined.train()
        assert combined.generator.training
        assert not combined.autoencoder.training

    def test_a_trainable_autoencoder_receives_gradients(self, autoencoder):
        combined = AutoencoderGenerator(
            autoencoder,
            UNetGenerator(base_filters=4, latent_channels=8),
            freeze_autoencoder=False,
        )
        combined(torch.randn(1, 1, HEIGHT, WIDTH)).sum().backward()
        assert autoencoder.encoder[0].weight.grad is not None

    def test_rejects_an_unknown_mode(self, autoencoder):
        with pytest.raises(ValueError, match="mode"):
            AutoencoderGenerator(autoencoder, UNetGenerator(base_filters=4), mode="latents")

    def test_latent_mode_requires_an_encoder(self):
        with pytest.raises(TypeError, match="encode"):
            AutoencoderGenerator(torch.nn.Identity(), UNetGenerator(base_filters=4), mode="latent")


class TestSchemaGANWithAnAutoencoder:
    """Driving the full model with an auto-encoder backed generator."""

    def test_trains_with_an_autoencoder_backed_generator(self, dataset):
        config = tiny_config()
        config.model.latent_channels = 8
        generator = AutoencoderGenerator(
            ConvAutoencoder(base_filters=4, latent_channels=8, depth=3),
            UNetGenerator(base_filters=4, latent_channels=8),
        )
        model = SchemaGAN(config, generator=generator)

        history = model.train(dataset, verbose=False)
        assert len(history.iterations["g_loss"]) == len(dataset)
        assert model.predict(dataset[0][0]).shape == (1, 1, HEIGHT, WIDTH)

    def test_reloads_a_custom_generator_when_it_is_supplied(self, dataset, tmp_path):
        config = tiny_config()
        config.model.latent_channels = 8
        make_generator = lambda: AutoencoderGenerator(
            ConvAutoencoder(base_filters=4, latent_channels=8, depth=3),
            UNetGenerator(base_filters=4, latent_channels=8),
        )
        model = SchemaGAN(config, generator=make_generator())
        path = model.save(tmp_path / "ae_model.pt")

        restored = SchemaGAN.load(path, device="cpu", generator=make_generator())
        assert isinstance(restored.generator, AutoencoderGenerator)
        for saved, loaded in zip(model.generator.parameters(), restored.generator.parameters()):
            assert torch.equal(saved, loaded)
