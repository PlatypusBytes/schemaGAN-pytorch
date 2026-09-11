"""Tests for CSV loading, CPT-like masking, normalisation and the dataset."""

import numpy as np
import pytest
import torch

from schemaGAN_torch.config import DataConfig
from schemaGAN_torch.data import (
    CrossSectionDataset,
    apply_mask,
    cpt_like_mask,
    denormalize_ic,
    list_csv_files,
    load_cross_sections,
    max_columns_with_spacing,
    normalize_ic,
    read_cross_section_csv,
    select_columns,
    write_cross_section_csv,
)

from conftest import HEIGHT, WIDTH


class TestCsvIO:
    """Reading and writing the long-format cross-section CSV files."""

    def test_write_then_read_returns_the_same_grid(self, tmp_path, grids):
        path = write_cross_section_csv(tmp_path / "cs.csv", grids[0])
        assert np.allclose(read_cross_section_csv(path, HEIGHT, WIDTH), grids[0], atol=1e-6)

    def test_reading_uses_the_x_and_z_columns_not_the_row_order(self, tmp_path, grids):
        path = tmp_path / "shuffled.csv"
        rows = [
            (index, x, z, float(grids[0][z, x]))
            for index, (x, z) in enumerate((x, z) for x in range(WIDTH) for z in range(HEIGHT))
        ]
        rng = np.random.default_rng(0)
        rng.shuffle(rows)
        lines = [",x,z,IC"] + [f"{i},{x},{z},{v}" for i, x, z, v in rows]
        path.write_text("\n".join(lines) + "\n")
        assert np.allclose(read_cross_section_csv(path, HEIGHT, WIDTH), grids[0], atol=1e-6)

    def test_rejects_a_wrong_number_of_rows(self, tmp_path, grids):
        path = write_cross_section_csv(tmp_path / "cs.csv", grids[0][:16])
        with pytest.raises(ValueError, match="rows"):
            read_cross_section_csv(path, HEIGHT, WIDTH)

    def test_rejects_a_missing_column(self, tmp_path):
        (tmp_path / "bad.csv").write_text(",x,z,value\n0,0,0,1.0\n")
        with pytest.raises(ValueError, match="missing one of the columns"):
            read_cross_section_csv(tmp_path / "bad.csv", 1, 1)

    def test_rejects_an_empty_file(self, tmp_path):
        (tmp_path / "empty.csv").write_text("")
        with pytest.raises(ValueError, match="empty"):
            read_cross_section_csv(tmp_path / "empty.csv", 1, 1)

    def test_lists_files_in_natural_order(self, tmp_path):
        for name in ("cs_10.csv", "cs_2.csv", "cs_1.csv", "notes.txt"):
            (tmp_path / name).write_text("")
        assert [p.name for p in list_csv_files(tmp_path)] == ["cs_1.csv", "cs_2.csv", "cs_10.csv"]

    def test_loads_a_whole_directory(self, csv_dir, grids):
        loaded = load_cross_sections(csv_dir, HEIGHT, WIDTH)
        assert loaded.shape == grids.shape
        assert np.allclose(loaded, grids, atol=1e-6)

    def test_reports_an_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_cross_sections(tmp_path, HEIGHT, WIDTH)


class TestColumnSelection:
    """Placement of the columns that stand in for CPT soundings."""

    def test_capacity_matches_the_spacing_rule(self):
        assert max_columns_with_spacing(512, 51) == 10
        assert max_columns_with_spacing(10, 0) == 10

    def test_selected_columns_respect_the_minimum_distance(self):
        rng = np.random.default_rng(0)
        columns = select_columns(WIDTH, 6, 51, rng)
        assert len(columns) == 6
        assert np.all(np.diff(columns) > 51)
        assert columns.min() >= 0 and columns.max() < WIDTH

    def test_is_reproducible_for_a_given_seed(self):
        first = select_columns(WIDTH, 6, 51, np.random.default_rng(3))
        second = select_columns(WIDTH, 6, 51, np.random.default_rng(3))
        assert np.array_equal(first, second)

    def test_rejects_an_impossible_request(self):
        with pytest.raises(ValueError, match="cannot place"):
            select_columns(WIDTH, 20, 51, np.random.default_rng(0))

    def test_reports_when_sampling_does_not_converge(self):
        with pytest.raises(RuntimeError, match="failed to place"):
            select_columns(WIDTH, 10, 51, np.random.default_rng(0), max_attempts=5)


class TestCptLikeMask:
    """The boolean mask that turns a full cross-section into sparse input."""

    @pytest.fixture
    def mask(self):
        """A mask drawn with the published miss rate and spacing."""
        return cpt_like_mask(HEIGHT, WIDTH, 0.99, 51, np.random.default_rng(5))

    def test_keeps_the_expected_number_of_columns(self, mask):
        kept = np.flatnonzero(mask.any(axis=0))
        assert len(kept) == WIDTH - int(0.99 * WIDTH)
        assert np.all(np.diff(kept) > 51)

    def test_columns_are_measured_from_the_surface_downwards(self, mask):
        for column in np.flatnonzero(mask.any(axis=0)):
            profile = mask[:, column]
            assert profile[0]
            assert np.all(profile[: profile.sum()])

    def test_never_removes_more_than_the_allowed_depth(self, mask):
        for column in np.flatnonzero(mask.any(axis=0)):
            assert mask[:, column].sum() > HEIGHT / 2

    def test_full_depth_is_kept_when_depth_removal_is_disabled(self):
        mask = cpt_like_mask(
            HEIGHT, WIDTH, 0.99, 51, np.random.default_rng(1), max_missing_depth_fraction=0.0
        )
        kept = np.flatnonzero(mask.any(axis=0))
        assert mask[:, kept].all()

    def test_applying_a_mask_zeroes_the_unmeasured_pixels(self, grids, mask):
        masked = apply_mask(grids[0], mask)
        assert np.array_equal(masked[mask], grids[0][mask])
        assert not masked[~mask].any()

    def test_apply_mask_rejects_mismatched_shapes(self, grids):
        with pytest.raises(ValueError, match="does not match"):
            apply_mask(grids[0], np.ones((4, 4), dtype=bool))


class TestNormalization:
    """Mapping between IC units and the ``[-1, 1]`` model range."""

    def test_maps_the_ic_range_onto_minus_one_to_one(self):
        assert normalize_ic(np.array([0.0, 2.15, 4.3])) == pytest.approx([-1.0, 0.0, 1.0])

    def test_is_invertible(self, grids):
        assert np.allclose(denormalize_ic(normalize_ic(grids[0])), grids[0], atol=1e-5)

    def test_works_on_torch_tensors(self):
        values = torch.tensor([0.0, 4.3])
        assert torch.allclose(normalize_ic(values), torch.tensor([-1.0, 1.0]))

    @pytest.mark.parametrize("function", [normalize_ic, denormalize_ic])
    def test_rejects_an_empty_range(self, function):
        with pytest.raises(ValueError, match="max_ic"):
            function(np.zeros(1), min_ic=1.0, max_ic=1.0)


class TestCrossSectionDataset:
    """The dataset that pairs masked sources with complete targets."""

    def test_returns_normalised_source_and_target_pairs(self, dataset, grids):
        source, target = dataset[0]
        assert len(dataset) == len(grids)
        assert source.shape == (1, HEIGHT, WIDTH) == target.shape
        assert source.dtype is torch.float32
        assert float(target.min()) >= -1.0 and float(target.max()) <= 1.0

    def test_the_source_is_the_masked_target(self, dataset):
        source, target = dataset[0]
        measured = torch.from_numpy(dataset.masks[0])[None]
        assert torch.allclose(source[measured], target[measured])
        assert torch.allclose(source[~measured], torch.tensor(-1.0))

    def test_masks_are_stable_across_accesses(self, dataset):
        assert torch.equal(dataset[1][0], dataset[1][0])

    def test_resampling_draws_a_new_layout_every_time(self, grids, config):
        data = CrossSectionDataset(grids, config.data, resample_mask=True, seed=0)
        assert data.masks is None
        assert not torch.equal(data[0][0], data[0][0])

    def test_can_be_built_from_a_directory(self, csv_dir, config, grids):
        data = CrossSectionDataset(csv_dir, config.data, seed=1)
        assert len(data) == len(grids)

    def test_accepts_a_single_cross_section(self, grids, config):
        assert len(CrossSectionDataset(grids[0], config.data, seed=1)) == 1

    def test_rejects_arrays_with_the_wrong_geometry(self, config):
        with pytest.raises(ValueError, match="expected"):
            CrossSectionDataset(np.zeros((2, 8, 8), dtype=np.float32), config.data)

    def test_rejects_arrays_with_too_many_dimensions(self, config):
        with pytest.raises(ValueError, match="shape"):
            CrossSectionDataset(np.zeros((2, 1, HEIGHT, WIDTH), dtype=np.float32), config.data)

    def test_exposes_the_whole_dataset_as_arrays(self, dataset):
        sources, targets = dataset.as_arrays()
        assert sources.shape == targets.shape == (len(dataset), 1, HEIGHT, WIDTH)

    def test_honours_a_custom_ic_range(self, grids):
        config = DataConfig(max_ic=8.6)
        data = CrossSectionDataset(grids, config, seed=1)
        assert float(data[0][1].max()) < 0.0
