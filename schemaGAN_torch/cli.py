"""Command line helpers shared by the training/validation/inference scripts."""

from __future__ import annotations

import argparse

from .config import DataConfig


def add_data_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the cross-section geometry and CPT simulation options."""
    group = parser.add_argument_group("data")
    group.add_argument("--image-height", type=int, default=32, help="rows of a cross-section")
    group.add_argument("--image-width", type=int, default=512, help="columns of a cross-section")
    group.add_argument("--miss-rate", type=float, default=0.99, help="fraction of columns removed")
    group.add_argument("--min-distance", type=int, default=51, help="minimum spacing between CPTs")
    group.add_argument("--min-ic", type=float, default=0.0, help="lower bound of the IC range")
    group.add_argument("--max-ic", type=float, default=4.3, help="upper bound of the IC range")
    return parser


def data_config_from_args(args: argparse.Namespace) -> DataConfig:
    """Build a :class:`DataConfig` from the parsed data arguments."""
    return DataConfig(
        image_height=args.image_height,
        image_width=args.image_width,
        miss_rate=args.miss_rate,
        min_distance=args.min_distance,
        min_ic=args.min_ic,
        max_ic=args.max_ic,
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add device and seed options."""
    group = parser.add_argument_group("runtime")
    group.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda' or 'cuda:N'")
    group.add_argument("--seed", type=int, default=None, help="seed for reproducible runs")
    return parser
