"""Hydra-based training entry point for MuNet."""

import logging

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from munet.data.dataset import SimRealDataset
from munet.models.unet import UNet
from munet.training.trainer import Trainer

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Train UNet model using Hydra configuration."""
    logger.info("Loading datasets from %s", cfg.data.path)
    train_ds = SimRealDataset(
        path=cfg.data.path,
        split="train",
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.seed,
    )
    val_ds = SimRealDataset(
        path=cfg.data.path,
        split="val",
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    logger.info("Initializing UNet model")
    model = UNet(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        base_features=cfg.model.base_features,
        bilinear=cfg.model.bilinear,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.training.lr,
        epochs=cfg.training.epochs,
        checkpoint_dir=cfg.training.checkpoint_dir,
        log_dir=cfg.training.log_dir,
        device=cfg.training.device,
    )

    logger.info("Starting training")
    trainer.train()


if __name__ == "__main__":
    main()
