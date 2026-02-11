#!/usr/bin/env python3
"""Main training script with Hydra configuration."""

import os
import random

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from src.data.dataset import ImageDataset, get_default_transforms
from src.models.unet import UNet
from src.training.trainer import Trainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_optimizer(model: nn.Module, cfg: DictConfig):
    """Create optimizer from config."""
    opt_cfg = cfg.training.optimizer
    if opt_cfg.name.lower() == "adam":
        return Adam(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
        )
    elif opt_cfg.name.lower() == "sgd":
        momentum = getattr(opt_cfg, "momentum", 0.9)
        return SGD(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            momentum=momentum,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg.name}")


def get_scheduler(optimizer, cfg: DictConfig):
    """Create learning rate scheduler from config."""
    sched_cfg = cfg.training.scheduler
    if sched_cfg.name.lower() == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=sched_cfg.T_max, eta_min=sched_cfg.eta_min
        )
    elif sched_cfg.name.lower() == "step":
        return StepLR(optimizer, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma)
    else:
        return None


def get_criterion(cfg: DictConfig):
    """Create loss function from config."""
    loss_name = cfg.training.loss.name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "dice":
        return DiceLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1 - dice


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    # Print configuration
    print(OmegaConf.to_yaml(cfg))

    # Set random seed
    set_seed(cfg.experiment.seed)

    # Setup device
    device = cfg.hardware.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Get transforms
    train_transform, val_transform = get_default_transforms(cfg.data.image_size)

    # Create datasets
    train_dataset = ImageDataset(
        input_dir=cfg.data.train_input_dir,
        target_dir=cfg.data.train_target_dir,
        transform=train_transform,
    )

    val_dataset = ImageDataset(
        input_dir=cfg.data.val_input_dir,
        target_dir=cfg.data.val_target_dir,
        transform=val_transform,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True if device == "cuda" else False,
    )

    # Create model
    model = UNet(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        base_features=cfg.model.base_features,
        bilinear=cfg.model.bilinear,
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {cfg.model.name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer, scheduler, and criterion
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    criterion = get_criterion(cfg)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        log_dir=cfg.output.log_dir,
        checkpoint_dir=cfg.output.checkpoint_dir,
        scheduler=scheduler,
    )

    # Start training
    print(f"\nStarting training for {cfg.training.epochs} epochs...")
    print(f"TensorBoard logs: {cfg.output.log_dir}")
    print(f"Checkpoints: {cfg.output.checkpoint_dir}")
    print("-" * 50)

    history = trainer.train(
        num_epochs=cfg.training.epochs,
        save_every=cfg.training.save_every,
        log_images_every=cfg.training.log_images_every,
    )

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
