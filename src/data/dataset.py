"""xarray-based dataset for sim-to-real training pairs."""

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class SimRealDataset(Dataset):
    """Dataset that loads sim/real image pairs from an xarray NetCDF file.

    The expected xarray Dataset structure::

        <xarray.Dataset>
        Dimensions:  (sample: N, channel: C, height: H, width: W)
        Data variables:
            sim   (sample, channel, height, width) float32
            real  (sample, channel, height, width) float32

    Args:
        path: Path to the NetCDF file containing the xarray dataset.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        train_ratio: Fraction of samples used for training.
        val_ratio: Fraction of samples used for validation.
        seed: Random seed for reproducible train/val/test splitting.
    """

    def __init__(
        self,
        path: str,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        ds = xr.open_dataset(path)
        n_samples = ds.sizes["sample"]

        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)

        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        if split == "train":
            idx = indices[:n_train]
        elif split == "val":
            idx = indices[n_train : n_train + n_val]
        elif split == "test":
            idx = indices[n_train + n_val :]
        else:
            raise ValueError(f"Unknown split: {split!r}. Must be 'train', 'val', or 'test'.")

        self.sim = torch.from_numpy(ds["sim"].values[idx]).float()
        self.real = torch.from_numpy(ds["real"].values[idx]).float()
        ds.close()

    def __len__(self) -> int:
        return len(self.sim)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sim[idx], self.real[idx]


def create_dummy_dataset(
    path: str,
    n_samples: int = 100,
    channels: int = 3,
    height: int = 64,
    width: int = 64,
    seed: int = 0,
) -> None:
    """Create a dummy NetCDF dataset for testing.

    Args:
        path: Output file path.
        n_samples: Number of sample pairs to generate.
        channels: Number of image channels.
        height: Image height in pixels.
        width: Image width in pixels.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    sim = rng.standard_normal((n_samples, channels, height, width)).astype(np.float32)
    real = sim + 0.1 * rng.standard_normal((n_samples, channels, height, width)).astype(np.float32)

    ds = xr.Dataset(
        {
            "sim": (["sample", "channel", "height", "width"], sim),
            "real": (["sample", "channel", "height", "width"], real),
        }
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
