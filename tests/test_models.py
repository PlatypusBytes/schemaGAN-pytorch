"""Tests for the convolutional blocks, the generator and the discriminator."""

import pytest
import torch
from torch import nn

from schemaGAN_torch.config import ModelConfig
from schemaGAN_torch.models import (
    NoiseDropout,
    PatchDiscriminator,
    SameConv2d,
    SameConvTranspose2d,
    UNetGenerator,
    build_discriminator,
    build_generator,
    init_conv_weights,
    same_padding,
)

from conftest import HEIGHT, WIDTH


class TestSamePadding:
    """TensorFlow ``padding='same'`` semantics, including the ``(1, 2)`` strides."""

    @pytest.mark.parametrize(
        "kernel, stride, expected",
        [
            ((4, 4), (2, 2), (1, 1, 1, 1)),
            ((4, 4), (1, 2), (1, 1, 1, 2)),
            ((4, 4), (1, 1), (1, 2, 1, 2)),
        ],
    )
    def test_matches_tensorflow_same_padding(self, kernel, stride, expected):
        assert same_padding(kernel, stride) == expected

    @pytest.mark.parametrize("stride", [(2, 2), (1, 2), (1, 1)])
    def test_convolution_divides_the_input_by_the_stride(self, stride):
        layer = SameConv2d(3, 5, (4, 4), stride)
        out = layer(torch.zeros(2, 3, HEIGHT, WIDTH))
        assert out.shape == (2, 5, HEIGHT // stride[0], WIDTH // stride[1])

    @pytest.mark.parametrize("stride", [(2, 2), (1, 2), (1, 1)])
    def test_transposed_convolution_multiplies_the_input_by_the_stride(self, stride):
        layer = SameConvTranspose2d(3, 5, (4, 4), stride)
        out = layer(torch.zeros(2, 3, 4, 8))
        assert out.shape == (2, 5, 4 * stride[0], 8 * stride[1])


class TestNoiseDropout:
    """Dropout that can survive ``eval()``, as pix2pix requires."""

    def test_is_disabled_in_eval_mode_by_default(self):
        layer = NoiseDropout(0.9).eval()
        x = torch.ones(1, 8, 4, 4)
        assert torch.equal(layer(x), x)

    def test_stays_active_in_eval_mode_when_always_on(self):
        layer = NoiseDropout(0.9, always_on=True).eval()
        x = torch.ones(1, 64, 8, 8)
        assert (layer(x) == 0).any()


class TestWeightInit:
    """The ``N(0, 0.02)`` initialisation of the paper."""

    def test_initialises_convolutions_with_a_small_normal(self):
        module = nn.Sequential(nn.Conv2d(1, 64, 4), nn.ConvTranspose2d(64, 1, 4))
        init_conv_weights(module, std=0.02)
        for layer in module:
            assert layer.weight.std().item() == pytest.approx(0.02, abs=5e-3)
            assert torch.equal(layer.bias, torch.zeros_like(layer.bias))


class TestUNetGenerator:
    """Shapes, bounds and stochasticity of the generator."""

    @pytest.fixture
    def generator(self):
        """A narrow generator, fast enough for a test."""
        return UNetGenerator(base_filters=4)

    def test_preserves_the_cross_section_geometry(self, generator):
        out = generator(torch.randn(2, 1, HEIGHT, WIDTH))
        assert out.shape == (2, 1, HEIGHT, WIDTH)

    def test_output_is_bounded_by_tanh(self, generator):
        out = generator(torch.randn(2, 1, HEIGHT, WIDTH) * 10.0)
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_encoder_produces_one_skip_per_decoder_block(self, generator):
        bottleneck, skips = generator.encode(torch.randn(1, 1, HEIGHT, WIDTH))
        assert bottleneck.shape == (1, 32, 1, 1)
        assert len(skips) == len(generator.decoder)
        assert [tuple(s.shape[-2:]) for s in skips] == [
            (16, 256), (8, 128), (4, 64), (2, 32), (2, 16), (2, 8), (2, 4), (2, 2)
        ]

    def test_encode_and_decode_compose_into_forward(self, deterministic_config):
        generator = build_generator(deterministic_config.model).eval()
        x = torch.randn(1, 1, HEIGHT, WIDTH)
        bottleneck, skips = generator.encode(x)
        assert torch.allclose(generator.decode(bottleneck, skips), generator(x), atol=1e-6)

    def test_rejects_sizes_that_do_not_fit_the_downsampling(self, generator):
        with pytest.raises(ValueError, match="multiple of"):
            generator(torch.randn(1, 1, 16, WIDTH))

    def test_rejects_the_wrong_number_of_skip_connections(self, generator):
        bottleneck, skips = generator.encode(torch.randn(1, 1, HEIGHT, WIDTH))
        with pytest.raises(ValueError, match="skip connections"):
            generator.decode(bottleneck, skips[:-1])

    def test_supports_more_than_one_channel(self):
        generator = UNetGenerator(in_channels=2, out_channels=3, base_filters=4)
        assert generator(torch.randn(1, 2, HEIGHT, WIDTH)).shape == (1, 3, HEIGHT, WIDTH)

    def test_is_deterministic_without_stochastic_inference(self):
        generator = UNetGenerator(base_filters=4, stochastic_inference=False).eval()
        x = torch.randn(1, 1, HEIGHT, WIDTH)
        with torch.no_grad():
            assert torch.equal(generator(x), generator(x))

    def test_keeps_dropout_alive_at_inference_by_default(self):
        generator = UNetGenerator(base_filters=4, stochastic_inference=True).eval()
        x = torch.randn(1, 1, HEIGHT, WIDTH)
        with torch.no_grad():
            assert not torch.equal(generator(x), generator(x))


class TestLatentFusion:
    """Injection of an external latent code into the generator bottleneck."""

    @pytest.fixture
    def generator(self):
        """A narrow generator expecting an eight-channel latent code."""
        return UNetGenerator(base_filters=4, latent_channels=8)

    def test_accepts_a_spatial_latent_code(self, generator):
        out = generator(torch.randn(2, 1, HEIGHT, WIDTH), latent=torch.randn(2, 8, 2, 4))
        assert out.shape == (2, 1, HEIGHT, WIDTH)

    def test_accepts_a_flat_latent_code(self, generator):
        out = generator(torch.randn(2, 1, HEIGHT, WIDTH), latent=torch.randn(2, 8))
        assert out.shape == (2, 1, HEIGHT, WIDTH)

    def test_requires_a_latent_code(self, generator):
        with pytest.raises(ValueError, match="requires a latent code"):
            generator(torch.randn(1, 1, HEIGHT, WIDTH))

    def test_rejects_a_latent_code_with_the_wrong_width(self, generator):
        with pytest.raises(ValueError, match="channels"):
            generator(torch.randn(1, 1, HEIGHT, WIDTH), latent=torch.randn(1, 3))

    def test_rejects_a_latent_code_with_the_wrong_rank(self, generator):
        with pytest.raises(ValueError, match="shape"):
            generator(torch.randn(1, 1, HEIGHT, WIDTH), latent=torch.randn(1, 8, 2))

    def test_plain_generators_cannot_fuse_a_latent_code(self):
        generator = UNetGenerator(base_filters=4)
        with pytest.raises(RuntimeError, match="latent_channels"):
            generator.fuse_latent(torch.randn(1, 32, 1, 1), torch.randn(1, 8))


class TestPatchDiscriminator:
    """The patch classifier and its 16x16 output map."""

    @pytest.fixture
    def discriminator(self):
        """A narrow discriminator, fast enough for a test."""
        return PatchDiscriminator(base_filters=4)

    def test_returns_a_16_by_16_patch_map(self, discriminator):
        logits = discriminator(torch.randn(2, 1, HEIGHT, WIDTH), torch.randn(2, 1, HEIGHT, WIDTH))
        assert logits.shape == (2, 1, 16, 16)

    def test_reports_its_patch_shape(self, discriminator):
        assert tuple(discriminator.patch_shape(HEIGHT, WIDTH)) == (1, 1, 16, 16)

    def test_returns_logits_rather_than_probabilities(self, discriminator):
        logits = discriminator(torch.randn(4, 1, HEIGHT, WIDTH), torch.randn(4, 1, HEIGHT, WIDTH))
        assert logits.min() < 0.0

    def test_accepts_a_pre_concatenated_pair(self, discriminator):
        pair = torch.randn(1, 2, HEIGHT, WIDTH)
        assert discriminator(pair).shape == (1, 1, 16, 16)


class TestFactories:
    """Building networks from a :class:`ModelConfig`."""

    def test_build_generator_follows_the_model_config(self):
        config = ModelConfig(generator_filters=8, dropout=0.25, latent_channels=4)
        generator = build_generator(config)
        assert generator.latent_channels == 4
        assert generator.encoder[0].conv.conv.out_channels == 8
        assert generator.decoder[0].dropout.p == 0.25

    def test_build_discriminator_follows_the_model_config(self):
        discriminator = build_discriminator(ModelConfig(discriminator_filters=8))
        assert discriminator.model[0].conv.conv.out_channels == 8

    def test_defaults_reproduce_the_published_architecture(self):
        generator, discriminator = build_generator(), build_discriminator()
        assert len(generator.encoder) == len(generator.decoder) == 8
        assert generator.encoder[0].norm is None
        assert discriminator.model[0].norm is None
        assert [block.dropout is not None for block in generator.decoder] == [
            True, True, True, True, False, False, False, False
        ]
