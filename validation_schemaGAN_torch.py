#!/usr/bin/env python
"""Validate one or more trained schemaGAN checkpoints on labelled cross-sections.

All settings are read from a YAML file; see ``configs/default.yaml``.

Example::

    python validation_schemaGAN_torch.py configs/default.yaml
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemaGAN_torch import CrossSectionDataset, SchemaGAN
from schemaGAN_torch.cli import ValidationSettings, load_settings
from schemaGAN_torch.visualize import plot_error_histogram


def collect_checkpoints(path: str | Path) -> list[Path]:
    """Return the checkpoints to score, from a file or a directory.

    Raises:
        FileNotFoundError: If the path does not exist or holds no checkpoint.
    """
    path = Path(path)
    if path.is_dir():
        checkpoints = sorted(path.glob("*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"no .pt checkpoints found in {path}")
        return checkpoints
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    return [path]


def main(argv: list[str] | None = None) -> int:
    """Score every checkpoint and write the per-sample and aggregate errors.

    Args:
        argv: Command line arguments; ``sys.argv`` is used when omitted.

    Returns:
        The process exit code.
    """
    config, settings = load_settings(argv, __doc__, "validation", ValidationSettings)
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CrossSectionDataset(settings.data_dir, config.data, seed=config.train.seed)
    print(f"validation cross-sections: {len(dataset)}")

    summaries: list[dict[str, float | str]] = []
    for checkpoint in collect_checkpoints(settings.checkpoint):
        model = SchemaGAN.load(checkpoint, device=config.train.device)
        result = model.validate(dataset, batch_size=settings.batch_size)

        name = checkpoint.stem
        result.to_csv(output_dir / f"errors_{name}.csv")
        plot_error_histogram(result.mae, output_dir / f"mae_histogram_{name}.png")
        if settings.plots > 0:
            model.save_samples(dataset, output_dir / name, prefix="validation", limit=settings.plots)

        summary = result.summary
        summaries.append({"checkpoint": name, **summary})
        print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in summary.items()))

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["checkpoint", "mae", "mse", "rmse"])
        writer.writeheader()
        writer.writerows(summaries)

    print(f"results written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
