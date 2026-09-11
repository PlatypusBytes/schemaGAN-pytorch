#!/usr/bin/env python
"""Train a schemaGAN model on synthetic cross-sections.

All settings are read from a YAML file; see ``configs/default.yaml``.

Example::

    python training_schemaGAN_torch.py configs/default.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemaGAN_torch import CrossSectionDataset, SchemaGAN
from schemaGAN_torch.cli import TrainingSettings, load_settings


def main(argv: list[str] | None = None) -> int:
    """Train a model and write the checkpoints, figures and metrics.

    Args:
        argv: Command line arguments; ``sys.argv`` is used when omitted.

    Returns:
        The process exit code.
    """
    config, settings = load_settings(argv, __doc__, "training", TrainingSettings)

    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = config.train.seed
    train_dataset = CrossSectionDataset(
        settings.data_dir, config.data, resample_mask=settings.resample_mask, seed=seed
    )
    val_dataset = (
        CrossSectionDataset(settings.val_dir, config.data, seed=seed) if settings.val_dir else None
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
        verbose=settings.verbose,
    )

    if val_dataset is not None:
        summary = model.test(val_dataset)
        print("validation:", ", ".join(f"{k}={v:.4f}" for k, v in summary.items()))
        (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
