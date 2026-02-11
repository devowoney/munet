"""Tests for xarray dataset."""

import tempfile
from pathlib import Path

import torch

from munet.data.dataset import SimRealDataset, create_dummy_dataset


def test_create_dummy_dataset():
    """Dummy dataset file should be created successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test.nc")
        create_dummy_dataset(path, n_samples=20, channels=3, height=32, width=32)
        assert Path(path).exists()


def test_dataset_splits():
    """Dataset splits should partition samples correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test.nc")
        create_dummy_dataset(path, n_samples=100, channels=3, height=16, width=16)

        train_ds = SimRealDataset(path, split="train", train_ratio=0.8, val_ratio=0.1)
        val_ds = SimRealDataset(path, split="val", train_ratio=0.8, val_ratio=0.1)
        test_ds = SimRealDataset(path, split="test", train_ratio=0.8, val_ratio=0.1)

        assert len(train_ds) == 80
        assert len(val_ds) == 10
        assert len(test_ds) == 10


def test_dataset_item_shape():
    """Each dataset item should return (sim, real) tensors with correct shapes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test.nc")
        create_dummy_dataset(path, n_samples=10, channels=3, height=16, width=16)

        ds = SimRealDataset(path, split="train", train_ratio=0.8, val_ratio=0.1)
        sim, real = ds[0]
        assert isinstance(sim, torch.Tensor)
        assert isinstance(real, torch.Tensor)
        assert sim.shape == (3, 16, 16)
        assert real.shape == (3, 16, 16)
