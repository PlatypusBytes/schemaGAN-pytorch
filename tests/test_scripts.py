"""End-to-end checks of the three command line entry points."""

import json

import pytest

import inference_schemaGAN_torch as inference
import training_schemaGAN_torch as training
import validation_schemaGAN_torch as validation
from schemaGAN_torch import SchemaGANConfig
from schemaGAN_torch.cli import (
    InferenceSettings,
    TrainingSettings,
    ValidationSettings,
    read_config,
    script_settings,
)

from conftest import HEIGHT, WIDTH, write_config_file


class TestTraining:
    """``training_schemaGAN_torch.py``."""

    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory, csv_dir):
        """Train for one epoch and return the exit code and output directory."""
        output_dir = tmp_path_factory.mktemp("training_cli")
        config_file = write_config_file(
            output_dir / "settings.yaml",
            training={
                "data_dir": str(csv_dir),
                "val_dir": str(csv_dir),
                "output_dir": str(output_dir),
                "verbose": False,
            },
        )
        return training.main([str(config_file)]), output_dir

    def test_exits_successfully(self, run):
        assert run[0] == 0

    def test_writes_the_model_and_its_history(self, run):
        output_dir = run[1]
        assert (output_dir / "final_model.pt").is_file()
        assert (output_dir / "history_per_iteration.csv").is_file()
        assert list((output_dir / "checkpoints").glob("*.pt"))

    def test_records_the_configuration_and_the_validation_summary(self, run):
        output_dir = run[1]
        config = json.loads((output_dir / "config.json").read_text())
        assert config["model"]["generator_filters"] == 4
        assert config["data"]["image_height"] == HEIGHT
        assert set(json.loads((output_dir / "validation_summary.json").read_text())) == {
            "mae", "mse", "rmse"
        }

    def test_requires_a_data_directory(self, tmp_path):
        config_file = write_config_file(tmp_path / "settings.yaml", training={})
        with pytest.raises(ValueError, match="training.data_dir"):
            training.main([str(config_file)])

    def test_reports_a_missing_settings_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            training.main([str(tmp_path / "nope.yaml")])

    def test_rejects_an_unknown_setting(self, tmp_path):
        config_file = write_config_file(tmp_path / "settings.yaml", training={"epoch": 1})
        with pytest.raises(ValueError, match="unknown keys"):
            training.main([str(config_file)])


class TestValidation:
    """``validation_schemaGAN_torch.py``."""

    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory, csv_dir, trained_checkpoint):
        """Score the shared checkpoint and return the exit code and output directory."""
        output_dir = tmp_path_factory.mktemp("validation_cli")
        config_file = write_config_file(
            output_dir / "settings.yaml",
            validation={
                "data_dir": str(csv_dir),
                "checkpoint": str(trained_checkpoint),
                "output_dir": str(output_dir),
                "plots": 1,
            },
        )
        return validation.main([str(config_file)]), output_dir

    def test_exits_successfully(self, run):
        assert run[0] == 0

    def test_writes_per_sample_and_aggregate_errors(self, run):
        output_dir = run[1]
        summary = (output_dir / "summary.csv").read_text().splitlines()
        assert summary[0] == "checkpoint,mae,mse,rmse"
        assert len(summary) == 2
        assert list(output_dir.glob("errors_*.csv"))
        assert list(output_dir.glob("mae_histogram_*.png"))
        assert list(output_dir.rglob("validation_000.png"))

    def test_accepts_a_directory_of_checkpoints(self, tmp_path, csv_dir, trained_checkpoint):
        checkpoints = tmp_path / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "epoch_1.pt").write_bytes(trained_checkpoint.read_bytes())
        assert validation.collect_checkpoints(checkpoints) == [checkpoints / "epoch_1.pt"]

    def test_reports_a_missing_checkpoint(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validation.collect_checkpoints(tmp_path / "nope.pt")

    def test_reports_a_directory_without_checkpoints(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no .pt checkpoints"):
            validation.collect_checkpoints(tmp_path)


class TestInference:
    """``inference_schemaGAN_torch.py``."""

    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory, csv_dir, trained_checkpoint):
        """Generate from the sample CSVs and return the exit code and output directory."""
        output_dir = tmp_path_factory.mktemp("inference_cli")
        config_file = write_config_file(
            output_dir / "settings.yaml",
            inference={
                "input": str(csv_dir),
                "checkpoint": str(trained_checkpoint),
                "output_dir": str(output_dir),
                "simulate_cpt": True,
            },
        )
        return inference.main([str(config_file)]), output_dir

    def test_exits_successfully(self, run):
        assert run[0] == 0

    def test_writes_one_cross_section_per_input(self, run, grids):
        output_dir = run[1]
        assert len(list(output_dir.glob("*_generated.csv"))) == len(grids)
        assert len(list(output_dir.glob("*_generated.png"))) == len(grids)

    def test_the_generated_csv_can_be_read_back(self, run):
        from schemaGAN_torch.data import read_cross_section_csv

        generated = sorted(run[1].glob("*_generated.csv"))[0]
        grid = read_cross_section_csv(generated, HEIGHT, WIDTH)
        assert grid.shape == (HEIGHT, WIDTH)
        assert grid.min() >= -1e-5 and grid.max() <= 4.3 + 1e-5

    def test_can_skip_the_figures(self, tmp_path, csv_dir, trained_checkpoint):
        config_file = write_config_file(
            tmp_path / "settings.yaml",
            inference={
                "input": str(next(csv_dir.glob("*.csv"))),
                "checkpoint": str(trained_checkpoint),
                "output_dir": str(tmp_path),
                "plots": False,
            },
        )
        inference.main([str(config_file)])
        assert list(tmp_path.glob("*_generated.csv"))
        assert not list(tmp_path.glob("*.png"))

    def test_reports_a_missing_input(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inference.collect_inputs(tmp_path / "nope.csv")

    def test_reports_an_input_directory_without_csv_files(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no CSV files"):
            inference.collect_inputs(tmp_path)


class TestDefaultSettingsFile:
    """``configs/default.yaml``, the file every script falls back to."""

    @pytest.fixture(scope="class")
    def document(self):
        return read_config()

    def test_rebuilds_the_model_configuration(self, document):
        assert SchemaGANConfig.from_dict(document) == SchemaGANConfig()

    @pytest.mark.parametrize(
        "section,settings_type",
        [
            ("training", TrainingSettings),
            ("validation", ValidationSettings),
            ("inference", InferenceSettings),
        ],
    )
    def test_every_script_section_is_complete(self, document, section, settings_type):
        script_settings(document, section, settings_type)
