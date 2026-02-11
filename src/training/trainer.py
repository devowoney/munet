"""Training module with TensorBoard logging."""

import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    """
    Trainer class for UNet model with TensorBoard logging.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to train on (cuda/cpu)
        log_dir: Directory for TensorBoard logs
        checkpoint_dir: Directory to save model checkpoints
        scheduler: Optional learning rate scheduler
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        log_dir: str = "runs",
        checkpoint_dir: str = "checkpoints",
        scheduler: Optional[_LRScheduler] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

        # Setup directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False
        )

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Log to TensorBoard
            self.writer.add_scalar("train/batch_loss", loss.item(), self.global_step)
            self.global_step += 1

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / num_batches

        # Log epoch metrics
        self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        if self.scheduler:
            self.writer.add_scalar(
                "train/learning_rate", self.scheduler.get_last_lr()[0], epoch
            )

        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False)

        for inputs, targets in progress_bar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / num_batches

        # Log validation metrics
        self.writer.add_scalar("val/epoch_loss", avg_loss, epoch)

        return {"val_loss": avg_loss}

    def train(
        self,
        num_epochs: int,
        save_every: int = 5,
        log_images_every: int = 10,
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            log_images_every: Log sample images every N epochs
        """
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 30)

            # Train
            train_metrics = self.train_epoch(epoch)
            history["train_loss"].append(train_metrics["train_loss"])

            # Validate
            val_metrics = self.validate(epoch)
            history["val_loss"].append(val_metrics["val_loss"])

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            # Print metrics
            print(
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )

            # Save best model
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(epoch, is_best=True)

            # Save periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch)

            # Log sample images
            if epoch % log_images_every == 0:
                self.log_sample_images(epoch)

        self.writer.close()
        return history

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save checkpoint
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]

    @torch.no_grad()
    def log_sample_images(self, epoch: int, num_samples: int = 4):
        """Log sample input/output images to TensorBoard."""
        self.model.eval()

        # Get a batch from validation set
        inputs, targets = next(iter(self.val_loader))
        inputs = inputs[:num_samples].to(self.device)
        targets = targets[:num_samples].to(self.device)

        outputs = self.model(inputs)

        # Log images
        self.writer.add_images("samples/input", inputs, epoch)
        self.writer.add_images("samples/target", targets, epoch)
        self.writer.add_images("samples/output", outputs.sigmoid(), epoch)
