#!/usr/bin/env python
"""Validate one or more trained schemaGAN checkpoints on labelled cross-sections.

Example::

    python validation_schemaGAN_torch.py \
        --data-dir synthetic_data/512x32/validation \
        --checkpoint results/torch_run/checkpoints \
        --output-dir results/torch_run/validation
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemaGAN_torch import CrossSectionDataset, SchemaGAN
from schemaGAN_torch.cli import add_data_arguments, add_runtime_arguments, data_config_from_args
from schemaGAN_torch.visualize import plot_error_histogram


def build_parser() -> argparse.ArgumentParser:
    """Define the validation command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="directory with the validation cross-sections")
    parser.add_argument("--checkpoint", required=True, help="a .pt checkpoint or a directory of them")
    parser.add_argument("--output-dir", default="results/schemagan_torch/validation")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--plots", type=int, default=3, help="number of comparison figures per model")
    add_data_arguments(parser)
    add_runtime_arguments(parser)
    return parser


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
    args = build_parser().parse_args(argv)
    data_config = data_config_from_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CrossSectionDataset(args.data_dir, data_config, seed=args.seed)
    print(f"validation cross-sections: {len(dataset)}")

    summaries: list[dict[str, float | str]] = []
    for checkpoint in collect_checkpoints(args.checkpoint):
        model = SchemaGAN.load(checkpoint, device=args.device)
        result = model.validate(dataset, batch_size=args.batch_size)

        name = checkpoint.stem
        result.to_csv(output_dir / f"errors_{name}.csv")
        plot_error_histogram(result.mae, output_dir / f"mae_histogram_{name}.png")
        if args.plots > 0:
            model.save_samples(dataset, output_dir / name, prefix="validation", limit=args.plots)

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
