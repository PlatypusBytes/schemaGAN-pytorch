"""Loading of cross-section CSV files and simulation of CPT-like sparse input."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig

X_COLUMN = "x"
Z_COLUMN = "z"
VALUE_COLUMN = "IC"

_TRAILING_NUMBER = re.compile(r"(\d+)")


def list_csv_files(directory: str | Path, recursive: bool = True) -> list[Path]:
    """Return the CSV files in ``directory``, sorted naturally (cs_2 before cs_10).

    Args:
        directory: Folder to scan.
        recursive: Also descend into sub-folders.

    Raises:
        NotADirectoryError: If ``directory`` is not a folder.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    paths = directory.rglob("*.csv") if recursive else directory.glob("*.csv")

    def sort_key(path: Path):
        """Order by folder, then by the trailing number of the file name."""
        match = _TRAILING_NUMBER.findall(path.stem)
        return (path.parent.as_posix(), int(match[-1]) if match else -1, path.stem)

    return sorted(paths, key=sort_key)


def read_cross_section_csv(
    path: str | Path,
    height: int,
    width: int,
    *,
    x_column: str = X_COLUMN,
    z_column: str = Z_COLUMN,
    value_column: str = VALUE_COLUMN,
) -> np.ndarray:
    """Read a single cross-section into a ``(height, width)`` array of IC values.

    The CSV is expected in long format with one row per pixel and columns for the
    horizontal position, the depth and the value. Rows may appear in any order,
    since the grid is filled through the coordinate columns.

    Raises:
        ValueError: If the file is empty, lacks a column, has the wrong number of
            rows or does not cover the whole grid.
    """
    path = Path(path)
    with path.open("r", newline="") as handle:
        header = next(csv.reader(handle), None)
    if header is None:
        raise ValueError(f"{path} is empty")
    header = [name.strip() for name in header]
    try:
        columns = (header.index(x_column), header.index(z_column), header.index(value_column))
    except ValueError as exc:
        raise ValueError(f"{path} is missing one of the columns {(x_column, z_column, value_column)}") from exc

    raw = np.loadtxt(path, delimiter=",", skiprows=1, usecols=columns, ndmin=2)
    if raw.shape[0] != height * width:
        raise ValueError(f"{path} holds {raw.shape[0]} rows, expected {height * width}")

    x = raw[:, 0].astype(np.intp)
    z = raw[:, 1].astype(np.intp)
    if x.min() < 0 or x.max() >= width or z.min() < 0 or z.max() >= height:
        raise ValueError(f"{path} contains coordinates outside the {height}x{width} grid")

    grid = np.full((height, width), np.nan, dtype=np.float32)
    grid[z, x] = raw[:, 2]
    if np.isnan(grid).any():
        raise ValueError(f"{path} does not cover the full {height}x{width} grid")
    return grid


def load_cross_sections(
    source: str | Path | Sequence[str | Path],
    height: int,
    width: int,
    **kwargs,
) -> np.ndarray:
    """Load one directory, one file or a list of files into ``(n, height, width)``.

    Extra keyword arguments are forwarded to :func:`read_cross_section_csv`.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        paths = list_csv_files(path) if path.is_dir() else [path]
    else:
        paths = [Path(p) for p in source]
    if not paths:
        raise FileNotFoundError(f"no CSV files found in {source}")
    return np.stack([read_cross_section_csv(p, height, width, **kwargs) for p in paths])


def max_columns_with_spacing(width: int, min_distance: int) -> int:
    """Largest number of columns that can be picked with a spacing > ``min_distance``."""
    return (width - 1) // (min_distance + 1) + 1


def select_columns(
    width: int,
    n_keep: int,
    min_distance: int,
    rng: np.random.Generator,
    max_attempts: int = 10_000,
) -> np.ndarray:
    """Pick ``n_keep`` column indices that are more than ``min_distance`` apart.

    Uses rejection sampling, like the original implementation.

    Args:
        width: Number of columns to choose from.
        n_keep: Number of columns to keep.
        min_distance: Minimum gap between two kept columns.
        rng: Random generator, for reproducible layouts.
        max_attempts: Give up after this many draws.

    Raises:
        ValueError: If the request cannot be satisfied by any layout.
        RuntimeError: If sampling did not converge within ``max_attempts``.
    """
    if n_keep < 0:
        raise ValueError("n_keep must be non-negative")
    if n_keep > max_columns_with_spacing(width, min_distance):
        raise ValueError(
            f"cannot place {n_keep} columns in a width of {width} with a minimum distance of {min_distance}"
        )

    chosen: list[int] = []
    for _ in range(max_attempts):
        if len(chosen) == n_keep:
            break
        candidate = int(rng.integers(0, width))
        if all(abs(candidate - kept) > min_distance for kept in chosen):
            chosen.append(candidate)
    if len(chosen) != n_keep:
        raise RuntimeError(
            f"failed to place {n_keep} columns after {max_attempts} attempts; "
            "lower min_distance or miss_rate"
        )
    return np.array(sorted(chosen), dtype=np.intp)


def cpt_like_mask(
    height: int,
    width: int,
    miss_rate: float,
    min_distance: int,
    rng: np.random.Generator | None = None,
    max_missing_depth_fraction: float = 0.5,
) -> np.ndarray:
    """Build a boolean mask that keeps a few full-depth columns, as CPTs do.

    ``True`` marks a pixel that is measured; every other pixel is unknown. Each
    kept column additionally loses a random number of rows at its bottom, drawn
    from a triangular distribution biased towards keeping the full profile.

    Args:
        height: Depth pixels of the cross-section.
        width: Horizontal pixels of the cross-section.
        miss_rate: Fraction of the columns to drop.
        min_distance: Minimum gap between two kept columns.
        rng: Random generator; a fresh one is used when omitted.
        max_missing_depth_fraction: Largest share of a profile that can be cut
            away from the bottom of a kept column.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_keep = width - int(miss_rate * width)
    columns = select_columns(width, n_keep, min_distance, rng)

    mask = np.zeros((height, width), dtype=bool)
    mask[:, columns] = True
    max_removed = height * max_missing_depth_fraction
    for column in columns:
        n_rows = int(rng.triangular(0.0, 0.0, max_removed)) if max_removed > 0 else 0
        if n_rows > 0:
            mask[height - n_rows :, column] = False
    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out every unmeasured pixel of ``image``."""
    if image.shape[-2:] != mask.shape[-2:]:
        raise ValueError(f"image shape {image.shape} does not match mask shape {mask.shape}")
    return np.where(mask, image, 0.0).astype(image.dtype, copy=False)


def normalize_ic(data, min_ic: float = 0.0, max_ic: float = 4.3):
    """Scale IC values from ``[min_ic, max_ic]`` to ``[-1, 1]``."""
    if max_ic <= min_ic:
        raise ValueError("max_ic must be greater than min_ic")
    return 2.0 * (data - min_ic) / (max_ic - min_ic) - 1.0


def denormalize_ic(data, min_ic: float = 0.0, max_ic: float = 4.3):
    """Inverse of :func:`normalize_ic`."""
    if max_ic <= min_ic:
        raise ValueError("max_ic must be greater than min_ic")
    return (data + 1.0) * (max_ic - min_ic) / 2.0 + min_ic


class CrossSectionDataset(Dataset):
    """Pairs of (CPT-like source, complete target) cross-sections in ``[-1, 1]``.

    Items are returned as ``(source, target)`` float tensors of shape
    ``(channels, height, width)``.

    Args:
        source: A directory of CSV files, a list of files, or an array of shape
            ``(n, height, width)`` holding raw IC values.
        config: Geometry and masking settings; defaults to the published ones.
        resample_mask: Draw a new CPT layout on every access instead of a fixed
            one per cross-section. Keep ``num_workers=0`` when enabling this, so
            that workers do not replay the same random stream.
        seed: Seed of the mask generator.
        masks: Pre-computed masks, used instead of drawing new ones.
    """

    def __init__(
        self,
        source: str | Path | Sequence[str | Path] | np.ndarray,
        config: DataConfig | None = None,
        *,
        resample_mask: bool = False,
        seed: int | None = None,
        masks: np.ndarray | None = None,
    ) -> None:
        self.config = config or DataConfig()
        self.resample_mask = resample_mask
        self._rng = np.random.default_rng(seed)

        if isinstance(source, np.ndarray):
            targets = np.asarray(source, dtype=np.float32)
            if targets.ndim == 2:
                targets = targets[None]
            if targets.ndim != 3:
                raise ValueError("array input must have shape (n, height, width)")
            if targets.shape[1:] != self.config.image_shape:
                raise ValueError(
                    f"array input has shape {targets.shape[1:]}, expected {self.config.image_shape}"
                )
        else:
            targets = load_cross_sections(
                source, self.config.image_height, self.config.image_width
            )

        self.targets = targets
        self.masks = None if resample_mask else (masks if masks is not None else self._build_masks())

    def _build_masks(self) -> np.ndarray:
        """Draw one fixed CPT layout per cross-section."""
        if len(self.targets) == 0:
            return np.zeros((0, self.config.image_height, self.config.image_width), dtype=bool)
        return np.stack([self._new_mask() for _ in range(len(self.targets))])

    def _new_mask(self) -> np.ndarray:
        """Draw a single CPT layout from the configured settings."""
        return cpt_like_mask(
            self.config.image_height,
            self.config.image_width,
            self.config.miss_rate,
            self.config.min_distance,
            self._rng,
            self.config.max_missing_depth_fraction,
        )

    def __len__(self) -> int:
        """Number of cross-sections in the dataset."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the masked source and the complete target, both normalised."""
        target = self.targets[index]
        mask = self._new_mask() if self.masks is None else self.masks[index]
        source = apply_mask(target, mask)

        normalize = lambda a: normalize_ic(a, self.config.min_ic, self.config.max_ic)
        return (
            torch.from_numpy(np.ascontiguousarray(normalize(source)[None])).float(),
            torch.from_numpy(np.ascontiguousarray(normalize(target)[None])).float(),
        )

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the whole dataset as ``(sources, targets)`` of shape ``(n, 1, h, w)``."""
        sources, targets = zip(*(self[i] for i in range(len(self))))
        return (
            torch.stack(sources).numpy(),
            torch.stack(targets).numpy(),
        )


def write_cross_section_csv(
    path: str | Path,
    grid: np.ndarray,
    *,
    x_column: str = X_COLUMN,
    z_column: str = Z_COLUMN,
    value_column: str = VALUE_COLUMN,
) -> Path:
    """Write a ``(height, width)`` grid back to the long CSV format used by schemaGAN.

    Returns:
        The path that was written.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("grid must be two-dimensional")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = grid.shape
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", x_column, z_column, value_column])
        index = 0
        for x in range(width):
            for z in range(height):
                writer.writerow([index, x, z, float(grid[z, x])])
                index += 1
    return path
