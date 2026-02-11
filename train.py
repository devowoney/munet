"""Training script for sim-to-real mapping with Hydra configuration."""

import os
from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from munet.models import SimToRealMapper
from munet.data import SimToRealDataset
from munet.utils import calculate_psnr, save_comparison_grid

logger = logging.getLogger(__name__)


class PerceptualLoss(nn.Module):
    """Simple perceptual loss using VGG features"""
    
    def __init__(self):
        super().__init__()
        # For simplicity, we'll use L1 loss as a placeholder
        # In production, you'd use VGG features
        self.loss_fn = nn.L1Loss()
    
    def forward(self, pred, target):
        return self.loss_fn(pred, target)


def setup_model(cfg: DictConfig, device: torch.device) -> SimToRealMapper:
    """Initialize the model."""
    model = SimToRealMapper(
        n_channels=cfg.model.n_channels,
        n_classes=cfg.model.n_classes,
        bilinear=cfg.model.bilinear,
        base_filters=cfg.model.base_filters
    )
    model = model.to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def setup_dataloaders(cfg: DictConfig):
    """Setup train and validation dataloaders."""
    train_dataset = SimToRealDataset(
        root=cfg.data.train_dir,
        img_size=tuple(cfg.data.img_size)
    )
    
    val_dataset = SimToRealDataset(
        root=cfg.data.val_dir,
        img_size=tuple(cfg.data.img_size)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    return train_loader, val_loader


def setup_optimizer_scheduler(cfg: DictConfig, model: nn.Module):
    """Setup optimizer and learning rate scheduler."""
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=(cfg.training.beta1, cfg.training.beta2),
        weight_decay=cfg.training.weight_decay
    )
    
    if cfg.training.scheduler.type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs - cfg.training.scheduler.warmup_epochs,
            eta_min=cfg.training.scheduler.min_lr
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    l1_loss: nn.Module,
    perceptual_loss: nn.Module,
    cfg: DictConfig,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}")
    for batch_idx, (sim_imgs, real_imgs) in enumerate(pbar):
        sim_imgs = sim_imgs.to(device)
        real_imgs = real_imgs.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        generated = model(sim_imgs)
        
        # Calculate losses
        loss_l1 = l1_loss(generated, real_imgs)
        loss_perceptual = perceptual_loss(generated, real_imgs)
        
        # Weighted combination
        loss = (cfg.training.loss.l1_weight * loss_l1 + 
                cfg.training.loss.perceptual_weight * loss_perceptual)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
        if batch_idx % cfg.logging.log_freq == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/l1_loss', loss_l1.item(), step)
            writer.add_scalar('train/perceptual_loss', loss_perceptual.item(), step)
            
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    l1_loss: nn.Module,
    cfg: DictConfig,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_psnr = 0
    
    for batch_idx, (sim_imgs, real_imgs) in enumerate(tqdm(val_loader, desc="Validation")):
        sim_imgs = sim_imgs.to(device)
        real_imgs = real_imgs.to(device)
        
        # Forward pass
        generated = model(sim_imgs)
        
        # Calculate loss
        loss = l1_loss(generated, real_imgs)
        total_loss += loss.item()
        
        # Calculate PSNR
        psnr = calculate_psnr(generated, real_imgs)
        total_psnr += psnr
        
        # Save sample images
        if batch_idx == 0:
            output_path = Path(cfg.logging.log_dir) / f"comparison_epoch_{epoch}.png"
            save_comparison_grid(
                sim_imgs[:4],
                real_imgs[:4],
                generated[:4],
                str(output_path),
                nrow=2
            )
    
    avg_loss = total_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    
    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/psnr', avg_psnr, epoch)
    
    logger.info(f"Validation - Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f} dB")
    
    return avg_loss, avg_psnr


@hydra.main(version_base=None, config_path="configs/train", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    
    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create directories
    Path(cfg.checkpoint.save_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup model
    model = setup_model(cfg, device)
    
    # Setup dataloaders
    try:
        train_loader, val_loader = setup_dataloaders(cfg)
    except Exception as e:
        logger.warning(f"Could not load datasets: {e}")
        logger.warning("Creating dummy data directories for demonstration...")
        # Create dummy directories if data doesn't exist
        for split in ['train', 'val']:
            data_dir = cfg.data.get(f'{split}_dir')
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            Path(data_dir, 'sim').mkdir(exist_ok=True)
            Path(data_dir, 'real').mkdir(exist_ok=True)
        logger.info("Please add your data to the created directories and rerun training.")
        return
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_scheduler(cfg, model)
    
    # Setup loss functions
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss()
    
    # Setup tensorboard
    writer = SummaryWriter(cfg.logging.log_dir) if cfg.logging.tensorboard else None
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(1, cfg.training.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, l1_loss, perceptual_loss,
            cfg, device, epoch, writer
        )
        
        logger.info(f"Epoch {epoch}/{cfg.training.epochs} - Train Loss: {train_loss:.4f}")
        
        # Validate
        if epoch % cfg.logging.val_freq == 0:
            val_loss, val_psnr = validate(
                model, val_loader, l1_loss, cfg, device, epoch, writer
            )
            
            # Save best model
            if cfg.checkpoint.save_best and val_loss < best_loss:
                best_loss = val_loss
                checkpoint_path = Path(cfg.checkpoint.save_dir) / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'psnr': val_psnr,
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if epoch % cfg.checkpoint.save_freq == 0:
            checkpoint_path = Path(cfg.checkpoint.save_dir) / f"model_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Update learning rate
        if scheduler is not None and epoch > cfg.training.scheduler.warmup_epochs:
            scheduler.step()
            if writer:
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
    
    if writer:
        writer.close()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
