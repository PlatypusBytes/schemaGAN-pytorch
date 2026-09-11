#!/usr/bin/env python
"""Generate complete cross-sections from CPT-like input with a trained schemaGAN.

The input CSVs use the same long format as the synthetic data (columns
``x``, ``z``, ``IC``). Unmeasured pixels are expected to be zero; set
``inference.simulate_cpt`` to derive the sparse input from complete
cross-sections.

The image geometry and the IC range are taken from the checkpoint, since the
generator can only consume the representation it was trained on. All other
settings are read from a YAML file; see ``configs/default.yaml``.

Example::

    python inference_schemaGAN_torch.py configs/default.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemaGAN_torch import SchemaGAN
from schemaGAN_torch.cli import InferenceSettings, load_settings
from schemaGAN_torch.data import (
    apply_mask,
    cpt_like_mask,
    list_csv_files,
    normalize_ic,
    read_cross_section_csv,
    write_cross_section_csv,
)
from schemaGAN_torch.visualize import plot_grids


def collect_inputs(path: str | Path) -> list[Path]:
    """Return the cross-sections to generate from, from a file or a directory.

    Raises:
        FileNotFoundError: If the path does not exist or holds no CSV file.
    """
    path = Path(path)
    if path.is_dir():
        files = list_csv_files(path)
        if not files:
            raise FileNotFoundError(f"no CSV files found in {path}")
        return files
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    return [path]


def main(argv: list[str] | None = None) -> int:
    """Generate a dense cross-section for every input file.

    Args:
        argv: Command line arguments; ``sys.argv`` is used when omitted.

    Returns:
        The process exit code.
    """
    config, settings = load_settings(argv, __doc__, "inference", InferenceSettings)
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SchemaGAN.load(settings.checkpoint, device=config.train.device)
    data = model.config.data
    print(f"device: {model.device}")

    files = collect_inputs(settings.input)
    rng = np.random.default_rng(config.train.seed)
    miss_rate = data.miss_rate if settings.miss_rate is None else settings.miss_rate
    min_distance = data.min_distance if settings.min_distance is None else settings.min_distance

    grids = []
    for file in files:
        grid = read_cross_section_csv(file, data.image_height, data.image_width)
        if settings.simulate_cpt:
            mask = cpt_like_mask(
                data.image_height,
                data.image_width,
                miss_rate,
                min_distance,
                rng,
                data.max_missing_depth_fraction,
            )
            grid = apply_mask(grid, mask)
        grids.append(grid)

    sources = np.stack(grids)
    predictions = model.predict(
        normalize_ic(sources, data.min_ic, data.max_ic),
        batch_size=settings.batch_size,
        denormalize=True,
    ).numpy()[:, 0]

    for file, source, prediction in zip(files, sources, predictions):
        write_cross_section_csv(output_dir / f"{file.stem}_generated.csv", prediction)
        if settings.plots:
            plot_grids(
                [
                    ("CPT-like input", source, data.min_ic, data.max_ic, "viridis"),
                    ("Generated", prediction, data.min_ic, data.max_ic, "viridis"),
                ],
                output_dir / f"{file.stem}_generated.png",
                title=file.stem,
            )
        print(f"{file.name} -> {file.stem}_generated.csv")

    print(f"{len(files)} cross-section(s) written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
