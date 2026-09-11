"""The :class:`SchemaGAN` model: training, validation, testing and inference."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .config import SchemaGANConfig
from .data import denormalize_ic
from .metrics import per_sample_metrics
from .models import build_discriminator, build_generator
from .visualize import plot_cross_sections, plot_history

__all__ = ["History", "SchemaGAN", "ValidationResult", "resolve_device", "set_seed"]


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Turn ``None``/``'auto'`` into the best available device."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class History:
    """Metrics recorded per training iteration and per epoch."""

    iterations: dict[str, list[float]] = field(default_factory=dict)
    epochs: dict[str, list[float]] = field(default_factory=dict)

    def append(self, scope: str, metrics: dict[str, float]) -> None:
        """Record one value per metric in ``'iterations'`` or ``'epochs'``."""
        if scope not in ("iterations", "epochs"):
            raise ValueError("scope must be 'iterations' or 'epochs'")
        target = getattr(self, scope)
        for key, value in metrics.items():
            target.setdefault(key, []).append(float(value))

    def to_dict(self) -> dict[str, dict[str, list[float]]]:
        """Return a plain dict, safe to store inside a checkpoint."""
        return {"iterations": self.iterations, "epochs": self.epochs}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "History":
        """Rebuild a history from :meth:`to_dict`; missing scopes stay empty."""
        return cls(
            iterations={k: list(v) for k, v in dict(data.get("iterations", {})).items()},
            epochs={k: list(v) for k, v in dict(data.get("epochs", {})).items()},
        )

    def to_csv(self, path: str | Path, scope: str = "iterations") -> Path:
        """Write one scope as a CSV with a column per metric."""
        records = getattr(self, scope)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = list(records)
        rows = zip(*(records[c] for c in columns)) if columns else []
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return path


@dataclass
class ValidationResult:
    """Per-sample reconstruction errors of a dataset, in IC units by default."""

    mae: np.ndarray
    mse: np.ndarray
    rmse: np.ndarray

    def __len__(self) -> int:
        """Number of scored cross-sections."""
        return int(self.mae.size)

    @property
    def summary(self) -> dict[str, float]:
        """Mean of every metric over the dataset."""
        return {
            "mae": float(self.mae.mean()),
            "mse": float(self.mse.mean()),
            "rmse": float(self.rmse.mean()),
        }

    def to_csv(self, path: str | Path) -> Path:
        """Write one row per cross-section with its three error metrics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample", "mae", "mse", "rmse"])
            for index, row in enumerate(zip(self.mae, self.mse, self.rmse)):
                writer.writerow([index, *(float(v) for v in row)])
        return path


class SchemaGAN:
    """Conditional GAN that turns sparse CPT-like cross-sections into dense ones.

    The generator and the discriminator can be replaced by any compatible
    ``nn.Module`` (see :mod:`schemaGAN_torch.models.autoencoder`), which is how an
    auto-encoder is plugged in.

    Args:
        config: Full model/data/training configuration; defaults to the published
            settings.
        generator: Optional replacement for the default :class:`UNetGenerator`.
        discriminator: Optional replacement for the default patch discriminator.
        device: Overrides ``config.train.device``.
    """

    def __init__(
        self,
        config: SchemaGANConfig | None = None,
        *,
        generator: nn.Module | None = None,
        discriminator: nn.Module | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or SchemaGANConfig()
        self.device = resolve_device(device if device is not None else self.config.train.device)
        if self.config.train.seed is not None:
            set_seed(self.config.train.seed)

        self.generator = (generator or build_generator(self.config.model)).to(self.device)
        self.discriminator = (discriminator or build_discriminator(self.config.model)).to(self.device)

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.L1Loss()
        self.optimizer_g, self.optimizer_d = self._build_optimizers()
        self.history = History()
        self.epochs_trained = 0

    # ------------------------------------------------------------------ setup

    def _build_optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """Create the generator and discriminator Adam optimisers."""
        optim = self.config.optim
        betas = (optim.beta1, optim.beta2)
        return (
            torch.optim.Adam(self.generator.parameters(), lr=optim.learning_rate, betas=betas),
            torch.optim.Adam(self.discriminator.parameters(), lr=optim.learning_rate, betas=betas),
        )

    def to(self, device: str | torch.device) -> "SchemaGAN":
        """Move both networks to ``device`` and return self."""
        self.device = resolve_device(device)
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        return self

    def _as_loader(self, data, batch_size: int | None = None, shuffle: bool = False) -> DataLoader:
        """Accept a ``DataLoader``, a ``Dataset`` or a ``(source, target)`` pair."""
        if isinstance(data, DataLoader):
            return data
        if isinstance(data, (tuple, list)) and len(data) == 2:
            data = TensorDataset(self._prepare(data[0]), self._prepare(data[1]))
        if not isinstance(data, Dataset):
            raise TypeError("expected a Dataset, a DataLoader or a (source, target) pair")
        return DataLoader(
            data,
            batch_size=batch_size or self.config.train.batch_size,
            shuffle=shuffle,
            num_workers=self.config.train.num_workers,
        )

    def _prepare(self, data) -> torch.Tensor:
        """Coerce arrays/tensors to a float ``(n, c, h, w)`` tensor."""
        tensor = data if torch.is_tensor(data) else torch.as_tensor(np.asarray(data))
        tensor = tensor.to(torch.float32)
        if tensor.dim() == 2:
            tensor = tensor[None, None]
        elif tensor.dim() == 3:
            tensor = tensor[:, None]
        elif tensor.dim() != 4:
            raise ValueError(f"expected 2, 3 or 4 dimensions, got {tensor.dim()}")
        return tensor

    # --------------------------------------------------------------- training

    def train_step(self, source: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """Update the discriminator and the generator on a single batch.

        Args:
            source: CPT-like input in ``[-1, 1]``.
            target: Matching complete cross-section in ``[-1, 1]``.

        Returns:
            The losses and discriminator accuracies of this batch.
        """
        source = self._prepare(source).to(self.device)
        target = self._prepare(target).to(self.device)
        optim = self.config.optim

        self.optimizer_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            detached_fake = self.generator(source)
        real_logits = self.discriminator(source, target)
        fake_logits = self.discriminator(source, detached_fake)
        d_real = self.adversarial_loss(real_logits, torch.ones_like(real_logits))
        d_fake = self.adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
        d_loss = optim.discriminator_loss_weight * (d_real + d_fake)
        d_loss.backward()
        self.optimizer_d.step()

        self.optimizer_g.zero_grad(set_to_none=True)
        fake = self.generator(source)
        logits = self.discriminator(source, fake)
        g_adversarial = self.adversarial_loss(logits, torch.ones_like(logits))
        g_l1 = self.reconstruction_loss(fake, target)
        g_loss = g_adversarial + optim.lambda_l1 * g_l1
        g_loss.backward()
        self.optimizer_g.step()

        with torch.no_grad():
            accuracy_real = (real_logits > 0).float().mean()
            accuracy_fake = (fake_logits <= 0).float().mean()

        return {
            "d_loss": d_loss.item(),
            "d_loss_real": d_real.item(),
            "d_loss_fake": d_fake.item(),
            "d_acc_real": accuracy_real.item(),
            "d_acc_fake": accuracy_fake.item(),
            "g_loss": g_loss.item(),
            "g_adversarial": g_adversarial.item(),
            "g_l1": g_l1.item(),
        }

    def train(
        self,
        train_data,
        *,
        val_data=None,
        output_dir: str | Path | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        verbose: bool = True,
    ) -> History:
        """Run the adversarial training loop and return the recorded history.

        Args:
            train_data: A ``Dataset``, a ``DataLoader`` or a ``(source, target)`` pair.
            val_data: Optional data scored after every epoch; its metrics are
                stored under ``val_*`` in the epoch history.
            output_dir: When given, checkpoints, sample figures and history CSVs
                are written there.
            epochs: Overrides ``config.train.epochs``.
            batch_size: Overrides ``config.train.batch_size``.
            verbose: Print per-batch progress.

        Returns:
            The history, which also stays available as ``self.history``.
        """
        loader = self._as_loader(train_data, batch_size, shuffle=self.config.train.shuffle)
        epochs = self.config.train.epochs if epochs is None else epochs
        settings = self.config.train
        directory = Path(output_dir) if output_dir is not None else None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            self.generator.train()
            self.discriminator.train()
            totals: dict[str, float] = {}
            batches = 0

            for step, (source, target) in enumerate(loader, start=1):
                metrics = self.train_step(source, target)
                self.history.append("iterations", metrics)
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                batches += 1
                if verbose and settings.log_every and step % settings.log_every == 0:
                    print(
                        f"epoch {epoch}/{epochs} batch {step}/{len(loader)} "
                        f"d_loss={metrics['d_loss']:.4f} g_loss={metrics['g_loss']:.4f} "
                        f"g_l1={metrics['g_l1']:.4f}"
                    )

            epoch_metrics = {key: value / max(batches, 1) for key, value in totals.items()}
            if val_data is not None:
                epoch_metrics.update(
                    {f"val_{k}": v for k, v in self.validate(val_data).summary.items()}
                )
            self.history.append("epochs", epoch_metrics)
            self.epochs_trained += 1

            if directory is not None:
                self._write_epoch_artifacts(directory, epoch, epochs, val_data or train_data)

        if directory is not None:
            self.save(directory / "final_model.pt")
            self.history.to_csv(directory / "history_per_iteration.csv", "iterations")
            self.history.to_csv(directory / "history_per_epoch.csv", "epochs")
        return self.history

    def _write_epoch_artifacts(self, directory: Path, epoch: int, epochs: int, sample_data) -> None:
        """Save the checkpoint, sample figures and loss curves of one epoch."""
        settings = self.config.train
        last = epoch == epochs
        if settings.checkpoint_every and (last or epoch % settings.checkpoint_every == 0):
            self.save(directory / "checkpoints" / f"schemagan_epoch_{epoch:06d}.pt")
        if settings.sample_every and (last or epoch % settings.sample_every == 0):
            self.save_samples(sample_data, directory / "samples", prefix=f"epoch_{epoch:06d}", limit=1)
        if self.history.iterations:
            plot_history(self.history.iterations, directory / "history.png", keys=["d_loss", "g_loss", "g_l1"])

    # ------------------------------------------------------- evaluation / use

    @torch.no_grad()
    def predict(self, source, *, batch_size: int | None = None, denormalize: bool = False) -> torch.Tensor:
        """Generate cross-sections for normalised sources.

        Args:
            source: Normalised input of shape ``(h, w)``, ``(n, h, w)`` or ``(n, c, h, w)``.
            batch_size: Number of cross-sections generated at once.
            denormalize: Return IC units instead of ``[-1, 1]``.

        Returns:
            A CPU tensor of shape ``(n, c, h, w)``.
        """
        self.generator.eval()
        tensor = self._prepare(source)
        size = batch_size or self.config.train.batch_size
        outputs = [
            self.generator(tensor[start : start + size].to(self.device)).cpu()
            for start in range(0, tensor.shape[0], size)
        ]
        prediction = torch.cat(outputs) if outputs else tensor.new_empty((0,) + tensor.shape[1:])
        if denormalize:
            prediction = denormalize_ic(prediction, self.config.data.min_ic, self.config.data.max_ic)
        return prediction

    @torch.no_grad()
    def validate(self, data, *, batch_size: int | None = None, denormalize: bool = True) -> ValidationResult:
        """Score a labelled dataset, one error per cross-section.

        Args:
            data: A ``Dataset``, a ``DataLoader`` or a ``(source, target)`` pair.
            batch_size: Number of cross-sections scored at once.
            denormalize: Report errors in IC units rather than in ``[-1, 1]``.
        """
        loader = self._as_loader(data, batch_size, shuffle=False)
        self.generator.eval()
        collected: dict[str, list[torch.Tensor]] = {"mae": [], "mse": [], "rmse": []}

        for source, target in loader:
            source = self._prepare(source).to(self.device)
            target = self._prepare(target).to(self.device)
            prediction = self.generator(source)
            if denormalize:
                low, high = self.config.data.min_ic, self.config.data.max_ic
                prediction = denormalize_ic(prediction, low, high)
                target = denormalize_ic(target, low, high)
            for key, value in per_sample_metrics(prediction, target).items():
                collected[key].append(value.cpu())

        if not collected["mae"]:
            raise ValueError("cannot validate on an empty dataset")
        return ValidationResult(**{k: torch.cat(v).numpy() for k, v in collected.items()})

    def test(self, data, **kwargs) -> dict[str, float]:
        """Mean MAE/MSE/RMSE over a held-out dataset.

        Thin wrapper around :meth:`validate` for when only the aggregate matters.
        """
        return self.validate(data, **kwargs).summary

    @torch.no_grad()
    def save_samples(
        self,
        data,
        directory: str | Path,
        *,
        prefix: str = "sample",
        limit: int = 4,
    ) -> list[Path]:
        """Write input/generated/target/error figures for the first cross-sections.

        Args:
            data: A ``Dataset``, a ``DataLoader`` or a ``(source, target)`` pair.
            directory: Where the PNG files are written.
            prefix: File name prefix, completed with the sample index.
            limit: Maximum number of figures to write.

        Returns:
            The paths of the figures that were written.
        """
        loader = self._as_loader(data, batch_size=1, shuffle=False)
        directory = Path(directory)
        low, high = self.config.data.min_ic, self.config.data.max_ic
        self.generator.eval()

        paths: list[Path] = []
        for index, (source, target) in enumerate(loader):
            if index >= limit:
                break
            source = self._prepare(source).to(self.device)
            prediction = self.generator(source)
            paths.append(
                plot_cross_sections(
                    denormalize_ic(source.cpu(), low, high),
                    denormalize_ic(prediction.cpu(), low, high),
                    denormalize_ic(self._prepare(target), low, high),
                    directory / f"{prefix}_{index:03d}.png",
                    vmin=low,
                    vmax=high,
                )
            )
        return paths

    # ------------------------------------------------------ persistence

    def save(self, path: str | Path, *, include_optimizers: bool = True) -> Path:
        """Serialise the configuration, the weights and the training history.

        Args:
            path: Destination ``.pt`` file; parent directories are created.
            include_optimizers: Store the Adam state so training can resume.

        Returns:
            The path that was written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "config": self.config.to_dict(),
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "history": self.history.to_dict(),
            "epochs_trained": self.epochs_trained,
        }
        if include_optimizers:
            payload["optimizer_g"] = self.optimizer_g.state_dict()
            payload["optimizer_d"] = self.optimizer_d.state_dict()
        torch.save(payload, path)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
        generator: nn.Module | None = None,
        discriminator: nn.Module | None = None,
        strict: bool = True,
    ) -> "SchemaGAN":
        """Rebuild a model from a checkpoint written by :meth:`save`.

        ``weights_only=True`` keeps loading safe: a checkpoint can never execute
        code.

        Args:
            path: The ``.pt`` file to read.
            device: Overrides the device stored in the checkpoint configuration.
            generator: Required when the checkpoint was produced with a custom
                generator, since only its weights are stored.
            discriminator: Same, for the discriminator.
            strict: Passed to ``load_state_dict``.
        """
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        config = SchemaGANConfig.from_dict(checkpoint["config"])
        model = cls(config=config, generator=generator, discriminator=discriminator, device=device)
        model.generator.load_state_dict(checkpoint["generator"], strict=strict)
        if "discriminator" in checkpoint:
            model.discriminator.load_state_dict(checkpoint["discriminator"], strict=strict)
        if "optimizer_g" in checkpoint:
            model.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        if "optimizer_d" in checkpoint:
            model.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        model.history = History.from_dict(checkpoint.get("history", {}))
        model.epochs_trained = int(checkpoint.get("epochs_trained", 0))
        return model

    def __repr__(self) -> str:
        """Summarise the device, the training progress and the two networks."""
        return (
            f"{type(self).__name__}(device={self.device}, epochs_trained={self.epochs_trained}, "
            f"generator={type(self.generator).__name__}, discriminator={type(self.discriminator).__name__})"
        )
