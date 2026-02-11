"""Tests for the Trainer class."""

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from munet.models.unet import UNet
from munet.training.trainer import Trainer


def _make_loader(n: int = 8, channels: int = 3, size: int = 32, batch_size: int = 4):
    """Create a small DataLoader for testing."""
    sim = torch.randn(n, channels, size, size)
    real = torch.randn(n, channels, size, size)
    return DataLoader(TensorDataset(sim, real), batch_size=batch_size)


def test_trainer_one_epoch():
    """Trainer should complete one epoch without errors."""
    model = UNet(in_channels=3, out_channels=3, base_features=16, bilinear=True)
    loader = _make_loader()
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            epochs=1,
            checkpoint_dir=str(Path(tmpdir) / "ckpt"),
            log_dir=str(Path(tmpdir) / "logs"),
        )
        trainer.train()
        assert (Path(tmpdir) / "ckpt" / "best_model.pt").exists()


def test_trainer_saves_and_loads_checkpoint():
    """Trainer should save and load checkpoints."""
    model = UNet(in_channels=3, out_channels=3, base_features=16, bilinear=True)
    loader = _make_loader()
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            epochs=1,
            checkpoint_dir=str(Path(tmpdir) / "ckpt"),
            log_dir=str(Path(tmpdir) / "logs"),
        )
        ckpt_path = str(Path(tmpdir) / "manual_ckpt.pt")
        trainer.save_checkpoint(ckpt_path)
        trainer.load_checkpoint(ckpt_path)
