"""Tests for the SchemaGAN training, validation, inference and persistence API."""

import json

import numpy as np
import pytest
import torch

from schemaGAN_torch import CrossSectionDataset, SchemaGAN
from schemaGAN_torch.schemagan import History, ValidationResult, resolve_device, set_seed

from conftest import HEIGHT, WIDTH, tiny_config


class TestHistory:
    """The per-iteration and per-epoch metric log."""

    def test_collects_values_per_scope(self):
        history = History()
        history.append("iterations", {"g_loss": 1.0})
        history.append("iterations", {"g_loss": 2.0})
        history.append("epochs", {"g_loss": 1.5})
        assert history.iterations == {"g_loss": [1.0, 2.0]}
        assert history.epochs == {"g_loss": [1.5]}

    def test_rejects_an_unknown_scope(self):
        with pytest.raises(ValueError, match="scope"):
            History().append("batches", {"g_loss": 1.0})

    def test_roundtrips_through_a_dict(self):
        history = History()
        history.append("iterations", {"d_loss": 0.5})
        assert History.from_dict(history.to_dict()).iterations == history.iterations

    def test_writes_a_csv(self, tmp_path):
        history = History()
        for value in (1.0, 2.0):
            history.append("iterations", {"d_loss": value, "g_loss": value * 2})
        lines = history.to_csv(tmp_path / "h.csv").read_text().splitlines()
        assert lines == ["d_loss,g_loss", "1.0,2.0", "2.0,4.0"]


class TestValidationResult:
    """The per-sample error container returned by ``validate``."""

    @pytest.fixture
    def result(self):
        """Errors of two cross-sections."""
        return ValidationResult(
            mae=np.array([0.1, 0.3]), mse=np.array([0.01, 0.09]), rmse=np.array([0.1, 0.3])
        )

    def test_summarises_the_mean_of_each_metric(self, result):
        assert result.summary == pytest.approx({"mae": 0.2, "mse": 0.05, "rmse": 0.2})

    def test_reports_the_number_of_samples(self, result):
        assert len(result) == 2

    def test_writes_one_csv_row_per_sample(self, tmp_path, result):
        lines = result.to_csv(tmp_path / "errors.csv").read_text().splitlines()
        assert lines[0] == "sample,mae,mse,rmse"
        assert len(lines) == 3


class TestHelpers:
    """Device resolution and seeding."""

    def test_resolves_an_explicit_device(self):
        assert resolve_device("cpu") == torch.device("cpu")

    @pytest.mark.parametrize("value", [None, "auto"])
    def test_resolves_the_default_device(self, value):
        assert resolve_device(value).type in {"cpu", "cuda"}

    def test_seeding_makes_torch_reproducible(self):
        set_seed(42)
        first = torch.randn(4)
        set_seed(42)
        assert torch.equal(first, torch.randn(4))


class TestTrainStep:
    """A single adversarial update."""

    def test_reports_the_expected_metrics(self, model, dataset):
        metrics = model.train_step(*dataset[0])
        assert set(metrics) == {
            "d_loss", "d_loss_real", "d_loss_fake", "d_acc_real", "d_acc_fake",
            "g_loss", "g_adversarial", "g_l1",
        }
        assert all(np.isfinite(value) for value in metrics.values())
        assert 0.0 <= metrics["d_acc_real"] <= 1.0

    def test_updates_both_networks(self, model, dataset):
        before_g = model.generator.output[0].deconv.weight.detach().clone()
        before_d = model.discriminator.model[0].conv.conv.weight.detach().clone()
        model.train_step(*dataset[0])
        assert not torch.equal(before_g, model.generator.output[0].deconv.weight)
        assert not torch.equal(before_d, model.discriminator.model[0].conv.conv.weight)

    def test_discriminator_takes_two_steps_per_batch(self, model, dataset, monkeypatch):
        steps: list[int] = []
        original = model.optimizer_d.step
        monkeypatch.setattr(model.optimizer_d, "step", lambda *a, **k: (steps.append(1), original(*a, **k))[1])
        metrics = model.train_step(*dataset[0])
        assert len(steps) == 2
        expected = 0.5 * (metrics["d_loss_real"] + metrics["d_loss_fake"])
        assert metrics["d_loss"] == pytest.approx(expected, rel=1e-5)

    def test_the_l1_term_dominates_the_generator_loss(self, model, dataset):
        metrics = model.train_step(*dataset[0])
        expected = metrics["g_adversarial"] + model.config.optim.lambda_l1 * metrics["g_l1"]
        assert metrics["g_loss"] == pytest.approx(expected, rel=1e-5)


class TestTrain:
    """The training loop, its history and the artifacts it writes."""

    def test_records_one_history_entry_per_batch_and_epoch(self, model, dataset):
        history = model.train(dataset, epochs=2, verbose=False)
        assert len(history.iterations["g_loss"]) == 2 * len(dataset)
        assert len(history.epochs["g_loss"]) == 2
        assert model.epochs_trained == 2

    def test_adds_validation_metrics_to_the_epoch_history(self, model, dataset):
        history = model.train(dataset, val_data=dataset, verbose=False)
        assert {"val_mae", "val_mse", "val_rmse"} <= set(history.epochs)

    def test_accepts_a_plain_source_target_pair(self, model, dataset):
        sources, targets = dataset.as_arrays()
        history = model.train((sources, targets), verbose=False)
        assert len(history.iterations["g_loss"]) == len(dataset)

    def test_rejects_unsupported_training_data(self, model):
        with pytest.raises(TypeError, match="Dataset"):
            model.train(np.zeros((2, 1, HEIGHT, WIDTH), dtype=np.float32), verbose=False)

    def test_writes_the_expected_artifacts(self, model, dataset, tmp_path):
        model.train(dataset, output_dir=tmp_path, verbose=False)
        written = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file()}
        assert {
            "final_model.pt",
            "history.png",
            "history_per_epoch.csv",
            "history_per_iteration.csv",
            "checkpoints/schemagan_epoch_000001.pt",
            "samples/epoch_000001_000.png",
        } <= written

    def test_checkpoints_and_samples_can_be_switched_off(self, dataset, tmp_path):
        model = SchemaGAN(tiny_config(checkpoint_every=0, sample_every=0))
        model.train(dataset, output_dir=tmp_path, verbose=False)
        assert not (tmp_path / "checkpoints").exists()
        assert not (tmp_path / "samples").exists()


class TestPredict:
    """Inference on normalised input."""

    @pytest.mark.parametrize("shape", [(HEIGHT, WIDTH), (2, HEIGHT, WIDTH), (2, 1, HEIGHT, WIDTH)])
    def test_accepts_two_three_and_four_dimensional_input(self, model, shape):
        prediction = model.predict(np.zeros(shape, dtype=np.float32))
        assert prediction.shape[-3:] == (1, HEIGHT, WIDTH)

    def test_rejects_input_with_an_unsupported_rank(self, model):
        with pytest.raises(ValueError, match="dimensions"):
            model.predict(np.zeros((1, 1, 1, HEIGHT, WIDTH), dtype=np.float32))

    def test_returns_values_in_the_normalised_range(self, model, dataset):
        prediction = model.predict(dataset[0][0])
        assert prediction.min() >= -1.0 and prediction.max() <= 1.0

    def test_can_return_ic_units(self, model, dataset):
        prediction = model.predict(dataset[0][0], denormalize=True)
        data = model.config.data
        assert prediction.min() >= data.min_ic - 1e-5
        assert prediction.max() <= data.max_ic + 1e-5

    def test_batches_do_not_change_the_result(self, deterministic_config, dataset):
        model = SchemaGAN(deterministic_config)
        sources = torch.from_numpy(dataset.as_arrays()[0])
        assert torch.allclose(
            model.predict(sources, batch_size=1), model.predict(sources, batch_size=3), atol=1e-6
        )


class TestValidateAndTest:
    """Scoring a labelled dataset."""

    def test_returns_one_error_per_sample(self, model, dataset):
        result = model.validate(dataset)
        assert len(result) == len(dataset)
        assert result.rmse == pytest.approx(np.sqrt(result.mse), rel=1e-5)

    def test_errors_are_reported_in_ic_units_by_default(self, deterministic_config, dataset):
        model = SchemaGAN(deterministic_config)
        in_ic = model.validate(dataset).summary["mae"]
        normalised = model.validate(dataset, denormalize=False).summary["mae"]
        assert in_ic == pytest.approx(normalised * model.config.data.ic_range / 2.0, rel=1e-4)

    def test_test_returns_the_aggregate_summary(self, model, dataset):
        summary = model.test(dataset)
        assert set(summary) == {"mae", "mse", "rmse"}
        assert all(value >= 0.0 for value in summary.values())

    def test_rejects_an_empty_dataset(self, model, grids, config):
        empty = CrossSectionDataset(grids[:0].reshape(0, HEIGHT, WIDTH), config.data)
        with pytest.raises(ValueError, match="empty"):
            model.validate(empty)

    def test_does_not_leave_the_networks_in_training_mode(self, model, dataset):
        model.generator.train()
        model.validate(dataset)
        assert not model.generator.training


class TestPersistence:
    """Saving and reloading checkpoints."""

    def test_reloads_into_an_identical_model(self, deterministic_config, dataset, tmp_path):
        model = SchemaGAN(deterministic_config)
        model.train(dataset, verbose=False)
        path = model.save(tmp_path / "model.pt")

        restored = SchemaGAN.load(path, device="cpu")
        source = dataset[0][0]
        assert torch.allclose(model.predict(source), restored.predict(source), atol=1e-6)
        assert restored.config == model.config
        assert restored.epochs_trained == model.epochs_trained
        assert restored.history.iterations == model.history.iterations

    def test_optimizer_state_can_be_left_out(self, model, tmp_path):
        path = model.save(tmp_path / "weights.pt", include_optimizers=False)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert "optimizer_g" not in payload
        assert SchemaGAN.load(path, device="cpu").epochs_trained == 0

    def test_training_can_continue_after_a_reload(self, model, dataset, tmp_path):
        model.train(dataset, verbose=False)
        restored = SchemaGAN.load(model.save(tmp_path / "model.pt"), device="cpu")
        restored.train(dataset, verbose=False)
        assert restored.epochs_trained == 2

    def test_checkpoints_are_loaded_without_executing_code(self, model, tmp_path):
        path = model.save(tmp_path / "model.pt")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert json.loads(json.dumps(payload["config"])) == model.config.to_dict()


class TestDeviceHandling:
    """Placement of the two networks."""

    def test_stays_on_the_configured_device(self, model):
        assert model.device == torch.device("cpu")
        assert next(model.generator.parameters()).device.type == "cpu"

    def test_can_be_moved(self, model):
        model.to("cpu")
        assert next(model.discriminator.parameters()).device.type == "cpu"

    def test_has_a_readable_repr(self, model):
        assert "UNetGenerator" in repr(model)
