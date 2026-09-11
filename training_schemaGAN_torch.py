#!/usr/bin/env python
"""Train a schemaGAN model on synthetic cross-sections.

Example::

    python training_schemaGAN_torch.py \
        --data-dir synthetic_data/512x32/train \
        --val-dir synthetic_data/512x32/validation \
        --output-dir results/torch_run --epochs 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemaGAN_torch import CrossSectionDataset, SchemaGAN, SchemaGANConfig
from schemaGAN_torch.cli import add_data_arguments, add_runtime_arguments, data_config_from_args
from schemaGAN_torch.config import ModelConfig, OptimConfig, TrainConfig


def build_parser() -> argparse.ArgumentParser:
    """Define the training command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="directory with the training cross-sections")
    parser.add_argument("--val-dir", default=None, help="optional directory with validation cross-sections")
    parser.add_argument("--output-dir", default="results/schemagan_torch", help="where to write artifacts")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0, help="weight of the L1 term")
    parser.add_argument("--generator-filters", type=int, default=64)
    parser.add_argument("--discriminator-filters", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1, help="0 disables checkpoints")
    parser.add_argument("--sample-every", type=int, default=1, help="0 disables sample figures")
    parser.add_argument(
        "--resample-mask",
        action="store_true",
        help="draw a new CPT layout on every access instead of a fixed one per cross-section",
    )
    parser.add_argument("--quiet", action="store_true", help="do not print per-batch progress")
    add_data_arguments(parser)
    add_runtime_arguments(parser)
    return parser


def build_config(args: argparse.Namespace) -> SchemaGANConfig:
    """Assemble the model configuration from the parsed arguments."""
    return SchemaGANConfig(
        data=data_config_from_args(args),
        model=ModelConfig(
            generator_filters=args.generator_filters,
            discriminator_filters=args.discriminator_filters,
            dropout=args.dropout,
        ),
        optim=OptimConfig(learning_rate=args.learning_rate, lambda_l1=args.lambda_l1),
        train=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
            checkpoint_every=args.checkpoint_every,
            sample_every=args.sample_every,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Train a model and write the checkpoints, figures and metrics.

    Args:
        argv: Command line arguments; ``sys.argv`` is used when omitted.

    Returns:
        The process exit code.
    """
    args = build_parser().parse_args(argv)
    config = build_config(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CrossSectionDataset(
        args.data_dir, config.data, resample_mask=args.resample_mask, seed=args.seed
    )
    val_dataset = (
        CrossSectionDataset(args.val_dir, config.data, seed=args.seed) if args.val_dir else None
    )
    print(f"training cross-sections: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"validation cross-sections: {len(val_dataset)}")

    model = SchemaGAN(config)
    print(f"device: {model.device}")
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    model.train(
        train_dataset,
        val_data=val_dataset,
        output_dir=output_dir,
        verbose=not args.quiet,
    )

    if val_dataset is not None:
        summary = model.test(val_dataset)
        print("validation:", ", ".join(f"{k}={v:.4f}" for k, v in summary.items()))
        (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
